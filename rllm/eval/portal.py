"""Main evaluation portal for unified evaluation.

This module provides the EvalPortal class, which serves as the main entry point
for evaluating all environments in the rLLM framework.
"""

import asyncio
import json
import logging
import os
from functools import wraps
from pathlib import Path
from typing import Any

import torch
import yaml

from rllm.eval.metrics import (
    aggregate_token_metrics_by_text,
    compare_success_vs_failure_tokens,
    compute_all_logprob_metrics,
    compute_detailed_trajectory_metrics,
    compute_metrics,
    get_high_entropy_tokens,
    print_metrics,
)
from rllm.eval.registry import EnvConfigRegistry

logger = logging.getLogger(__name__)

# Closed-source (OpenAI-compatible) API configuration.
# Set CLOSED_API_BASE to your gateway URL (e.g. an OpenAI-compatible endpoint).
CLOSED_API_BASE = os.environ.get("CLOSED_API_BASE", "")
CLOSED_API_KEY_ENV = "CLOSED_API_KEY"


def _sanitize_path_for_workers(path_value: str) -> str:
    """Remove inaccessible/root-prefixed entries from PATH for worker subprocesses."""
    sanitized_parts: list[str] = []
    seen: set[str] = set()
    for raw_part in path_value.split(":"):
        part = raw_part.strip()
        if not part or part in seen:
            continue
        # /root/* is often inaccessible in non-root worker processes.
        if part.startswith("/root"):
            continue
        try:
            # Keep only resolvable directories to avoid PermissionError in downstream scanners.
            if Path(part).is_dir():
                sanitized_parts.append(part)
                seen.add(part)
        except (PermissionError, OSError):
            continue
    return ":".join(sanitized_parts)


def _setup_verl_rollout_manager(config: dict):
    """
    Set up verl's AgentLoopManager for evaluation.

    Args:
        config: Verl-compatible configuration dictionary

    Returns:
        Initialized AgentLoopManager
    """
    import os
    import ray
    from omegaconf import OmegaConf
    from verl.experimental.agent_loop.agent_loop import AgentLoopManager

    _current_path = os.environ.get("PATH", "")
    if _current_path:
        _filtered_path = _sanitize_path_for_workers(_current_path)
        if _filtered_path:
            os.environ["PATH"] = _filtered_path

    # Keep TVM-related env vars deterministic for Ray workers used by sglang/flashinfer.
    os.environ["TVM_HOME"] = os.environ.get("TVM_HOME", "")
    os.environ["TVM_LIBRARY_PATH"] = os.environ.get("TVM_LIBRARY_PATH", "")

    verl_config = config.get("verl", {})
    n_gpus = verl_config.get("n_gpus", 1)
    ray_address = verl_config.get("ray_address")
    ray_runtime_env = {
        "env_vars": {
            "PATH": os.environ.get("PATH", ""),
            "TVM_HOME": os.environ.get("TVM_HOME", ""),
            "TVM_LIBRARY_PATH": os.environ.get("TVM_LIBRARY_PATH", ""),
        }
    }

    if not ray.is_initialized():
        if ray_address:
            ray.init(address=ray_address, runtime_env=ray_runtime_env)
            logger.info(f"Connected to Ray cluster at {ray_address}")
        else:
            ray.init(
                num_gpus=n_gpus,
                ignore_reinit_error=True,
                runtime_env=ray_runtime_env,
            )
            logger.info(f"Initialized local Ray with {n_gpus} GPUs")

    config_obj = OmegaConf.create(config)

    rollout_config = config_obj.actor_rollout_ref.rollout
    model_path = config_obj.actor_rollout_ref.model.path

    logger.info(f"Setting up VerlEngine with model: {model_path}")
    logger.info(f"Rollout backend: {rollout_config.name}")
    logger.info(f"Tensor parallelism: {rollout_config.get('tensor_model_parallel_size', 1)}")

    rollout_manager = AgentLoopManager(
        config=config_obj,
    )

    logger.info("VerlEngine rollout manager initialized successfully")

    return rollout_manager, config_obj


def _is_closed_model(model: str) -> bool:
    """Check if model uses a closed-source API (prefixed with 'closed/')."""
    return model.startswith("closed/")


def _get_closed_config(
    model: str,
    configured_api_key: str | None = None,
    configured_base_url: str | None = None,
    configured_api_keys=None,
) -> dict:
    """Get closed-source API configuration for a model.

    Args:
        model: Model name with 'closed/' prefix.
        configured_api_key: Single api key from config (backward compatible).
        configured_base_url: Base url from config.
        configured_api_keys: Optional list (or comma-separated string) of api keys
            for per-request failover. Takes precedence over the single key when
            non-empty; the single key is still kept as api_keys[0] fallback.

    Returns:
        Dictionary with actual_model, base_url, api_key (first key), api_keys (list).
    """
    actual_model = model.replace("closed/", "", 1)
    base_url = configured_base_url or CLOSED_API_BASE

    # Normalize api_keys into a list. Accept list/tuple, comma-separated string,
    # or a bare scalar (int/float) — the last case guards against a CLI override
    # like `--set model.api_keys=<YOUR_API_KEY>` being coerced to int before
    # it reaches here; without this, an int would match neither branch and the
    # key would be silently dropped in favor of the YAML single key.
    api_keys: list[str] = []
    if configured_api_keys is not None and configured_api_keys != "":
        if isinstance(configured_api_keys, str):
            api_keys = [k.strip() for k in configured_api_keys.split(",") if k.strip()]
        elif isinstance(configured_api_keys, (list, tuple)):
            api_keys = [str(k).strip() for k in configured_api_keys if str(k).strip()]
        else:
            # Bare scalar (e.g. int/float from an over-eager CLI type coercion).
            scalar = str(configured_api_keys).strip()
            if scalar:
                api_keys = [scalar]
    # Fall back to single key (config or env) when no list provided.
    single_key = configured_api_key or os.environ.get(CLOSED_API_KEY_ENV, "")
    if not api_keys and single_key:
        api_keys = [single_key]

    if not api_keys:
        logger.warning(
            "Closed-source API key is empty. Set CLOSED_API_KEY or model.api_key/model.api_keys in config."
        )
    return {
        "actual_model": actual_model,
        "base_url": base_url,
        "api_key": api_keys[0] if api_keys else "",
        "api_keys": api_keys,
    }


class EvalPortal:
    """Unified evaluation portal for all rLLM environments.

    This class provides a unified interface for evaluating all environments
    supported by the rLLM framework: sokoban, minesweeper, rush_hour, alfworld,
    and webshop.

    Example:
        >>> from rllm.eval import EvalPortal
        >>> portal = EvalPortal({
        ...     "env": "webshop",
        ...     "data": {"split": "test", "limit": 100},
        ...     "model": {"name": "Qwen/Qwen3-4B"},
        ... })
        >>> results = asyncio.run(portal.run())
        >>> metrics = portal.compute_metrics()
        >>> portal.print_summary()
    """

    def __init__(self, config: dict | str):
        """Initialize portal with config dict or path to YAML/JSON file.

        Args:
            config: Configuration dictionary or path to config file.
        """
        self._config_file: str | None = None
        if isinstance(config, str):
            self._config_file = config
            config = self._load_config_file(config)
        self.config = self._validate_config(config)
        self.engine = None
        self.results = None
        self._tokenizer = None
        self._offline_logprobs_computed = False  # Track if offline logprobs have been computed

    def _load_config_file(self, path: str) -> dict:
        """Load configuration from YAML or JSON file.

        Args:
            path: Path to configuration file.

        Returns:
            Configuration dictionary.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, encoding="utf-8") as f:
            if path.suffix in [".yaml", ".yml"]:
                config = yaml.safe_load(f)
            else:
                config = json.load(f)

        # Handle nested 'eval' key if present
        if "eval" in config:
            config = config["eval"]

        return config

    def _validate_config(self, config: dict) -> dict:
        """Validate and fill in defaults for configuration.

        Args:
            config: Raw configuration dictionary.

        Returns:
            Validated configuration with defaults filled in.
        """
        # Required fields
        if "env" not in config:
            raise ValueError("Configuration must specify 'env' field")

        # Set defaults
        defaults = {
            "engine": "openai",  # Engine type: openai, sglang, verl, tinker
            "model": {
                "name": "Qwen/Qwen3-4B",
                "base_url": "http://localhost:30000/v1",
                "api_key": "None",
            },
            "sampling": {
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": -1,
                "do_sample": True,
            },
            "execution": {
                "n_parallel_agents": 64,
                "max_steps": 50,
                "max_response_length": 16384,
                "max_prompt_length": 4096,
                "max_workers": 64,
                "n_rollouts_per_prompt": 1,  # Number of rollouts per prompt for pass@k evaluation
                "log_trajectory_turn_progress": False,
                "trajectory_progress_log_interval": 1,
            },
            "data": {
                "split": "test",
                "limit": None,
            },
            "output": {
                "save_trajectories": True,
                "save_dir": "./results",
            },
            "metrics": {
                "include_logprob_metrics": False,  # Whether to compute logprob-based metrics
                "save_detailed_metrics": False,  # Whether to save detailed token/step level metrics
                "logprobs_source": "none",  # none | online | offline
                "offline_logprobs": {
                    "model_name": None,
                    "tokenizer_name": None,
                    "device": "cuda",
                    "dtype": "bfloat16",
                    "batch_size": 1,
                    "gpu_memory_utilization": 0.9,
                    "parallel_strategy": "naive",  # naive | accelerate | data_parallel | fsdp | vllm | deepspeed
                    # Advanced options
                    "use_flash_attention": True,
                    "gradient_checkpointing": False,
                    "sequence_bucketing": True,
                    "bucket_size": 128,
                    "max_sequence_length": 8192,
                    # FSDP options
                    "fsdp_cpu_offload": False,
                    "fsdp_sharding_strategy": "FULL_SHARD",
                    # vLLM options
                    "vllm_tensor_parallel_size": 1,
                    "vllm_gpu_memory_utilization": 0.9,
                    # DeepSpeed options
                    "deepspeed_zero_stage": 3,
                    "deepspeed_offload_device": "cpu",
                },
            },
        }

        # Merge defaults with provided config
        for key, default_value in defaults.items():
            if key not in config:
                config[key] = default_value
            elif isinstance(default_value, dict):
                for sub_key, sub_value in default_value.items():
                    if sub_key not in config[key]:
                        config[key][sub_key] = sub_value

        # Canonicalize engine selector to a single key (`engine`) for compatibility.
        # Accept `engine_name` as input alias, but do not keep both keys.
        if config.get("engine_name") is not None:
            engine_name = config.get("engine_name")
        else:
            engine_name = config.get("engine") or "openai"
        if isinstance(engine_name, str):
            engine_name = engine_name.lower()
        config["engine"] = engine_name
        config.pop("engine_name", None)

        return config

    def _get_engine_name(self) -> str:
        """Get normalized engine name from config."""
        engine_name = self.config.get("engine") or "openai"
        if not isinstance(engine_name, str):
            raise ValueError(f"Invalid engine name type: {type(engine_name)}")
        return engine_name.lower()

    def print_config(self, cli_args: list[str] | None = None) -> None:
        """Print the effective evaluation configuration in a readable format.

        Args:
            cli_args: sys.argv list, used to show which CLI overrides were applied.
        """
        sep = "=" * 60
        print(sep)
        print("  EVALUATION CONFIG")
        print(sep)
        if self._config_file:
            print(f"  config file : {self._config_file}")
        if cli_args:
            print(f"  cli command : {' '.join(cli_args)}")
        print()
        print(yaml.dump(self.config, default_flow_style=False, sort_keys=False, allow_unicode=True), end="")
        print(sep)
        print()

    def _get_tokenizer(self):
        """Get or create tokenizer.

        Returns:
            Tokenizer instance, or None for closed-source models without explicit tokenizer.
        """
        if self._tokenizer is None:
            model_name = self.config["model"]["name"]
            engine_name = self._get_engine_name()
            use_api_tokenizer = self.config["model"].get("use_api_tokenizer")
            if engine_name == "openai" and use_api_tokenizer is not False:
                logger.info(
                    "OpenAI-compatible engine configured for eval; "
                    "deferring tokenizer and token usage to the API."
                )
                return None

            # For closed-source models, tokenizer is optional
            # Only load if explicitly specified in config
            if _is_closed_model(model_name):
                tokenizer_name = self.config["model"].get("tokenizer_name")
                if tokenizer_name is None:
                    logger.info(
                        f"No tokenizer specified for closed-source model '{model_name}', "
                        "using chat completions endpoint"
                    )
                    return None
                logger.info(
                    f"Using tokenizer '{tokenizer_name}' for closed-source model '{model_name}'"
                )
                model_name = tokenizer_name

            from transformers import AutoTokenizer

            # Some model repos ship a malformed/unsupported fast tokenizer file (tokenizer.json).
            # If loading the fast tokenizer fails, fall back to the slow tokenizer.
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    model_name, trust_remote_code=True
                )
            except Exception as e:
                logger.warning(
                    "Failed to load tokenizer for '%s' (fast). Retrying with use_fast=False. Error: %s",
                    model_name,
                    e,
                )
                self._tokenizer = AutoTokenizer.from_pretrained(
                    model_name, trust_remote_code=True, use_fast=False
                )
        return self._tokenizer

    def setup_engine(self):
        """Configure and return AgentExecutionEngine.

        Returns:
            Configured AgentExecutionEngine instance.
        """
        from rllm.engine.agent_execution_engine import AgentExecutionEngine

        env_name = self.config["env"]

        # Import env_configs to ensure all environments are registered
        import rllm.eval.env_configs  # noqa: F401

        env_config = EnvConfigRegistry.get(env_name)
        
        tokenizer = self._get_tokenizer()

        model_name = self.config["model"]["name"]

        sampling_params = self.config.get("sampling", {}).copy()
        sampling_params["model"] = model_name

        metrics_cfg = self.config.get("metrics", {})
        logprobs_source = metrics_cfg.get("logprobs_source", "none")
        if logprobs_source != "online":
            # Default: do not request logprobs/entropy from the rollout engines
            sampling_params["return_logprobs"] = False
            sampling_params["entropy_mode"] = "none"
        else:
            sampling_params["return_logprobs"] = True

        # Merge default and user-provided args
        env_args = {**env_config.env_args, **self.config.get("env_args", {})}
        agent_args = {**env_config.agent_args, **self.config.get("agent_args", {})}

        agent_class = env_config.agent_class

        # Handle environment-specific configuration
        if env_name == "tau2":
            # Tau2 requires domain configuration
            domain = self.config.get("domain", env_args.get("domain", "airline"))
            env_args["domain"] = domain
            if "solo_mode" in self.config:
                env_args["solo_mode"] = self.config["solo_mode"]
            # Handle closed-source API for user simulator
            user_llm = self.config.get("user_llm") or env_args.get("user_llm")
            if user_llm:
                env_args["user_llm"] = user_llm
            user_llm_args = self.config.get("user_llm_args") or env_args.get(
                "user_llm_args"
            )
            if user_llm_args:
                env_args["user_llm_args"] = user_llm_args
        elif env_name == "appworld" or env_name == "alfworld":
            # AppWorld: handle world model file for agent
            world_model_file = self.config.get("world_model_file") or agent_args.get(
                "world_model_file"
            )
            if world_model_file:
                agent_args["world_model_file"] = world_model_file
                logger.info(f"{env_name} agent using world model file: {world_model_file}")

        # Determine engine type
        engine_name = self._get_engine_name()

        # Build rollout engine args based on engine type
        if engine_name == "verl":
            from omegaconf import OmegaConf

            verl_cfg = self.config.get("verl", {})
            rollout_name = verl_cfg.get("rollout_name", "vllm")
            tp_size = verl_cfg.get("tp_size", 1)
            gpu_memory_utilization = verl_cfg.get("gpu_memory_utilization", 0.9)
            dp_size = verl_cfg.get("dp_size", 1)
            pp_size = verl_cfg.get("pipeline_parallel_size", 1)
            rollout_mode = verl_cfg.get("rollout_mode", "async")
            free_cache_engine = verl_cfg.get("free_cache_engine", False)
            enforce_eager = verl_cfg.get("enforce_eager", False)

            # Keep rollout (train-like) and validation sampling parameters separable.
            # For eval we primarily use validation sampling (see _patch_engine_for_eval).
            val_kwargs_cfg = verl_cfg.get("val_kwargs", {})
            rollout_n = int(
                verl_cfg.get(
                    "rollout_n",
                    self.config["execution"].get("n_rollouts_per_prompt", 1),
                )
            )
            val_n = int(
                val_kwargs_cfg.get(
                    "n",
                    verl_cfg.get(
                        "val_n",
                        self.config["execution"].get("n_rollouts_per_prompt", 1),
                    ),
                )
            )
            rollout_temperature = float(
                verl_cfg.get(
                    "rollout_temperature",
                    sampling_params.get("temperature", 0.6),
                )
            )
            rollout_top_p = float(
                verl_cfg.get("rollout_top_p", sampling_params.get("top_p", 0.95))
            )
            rollout_top_k = int(
                verl_cfg.get("rollout_top_k", sampling_params.get("top_k", -1))
            )
            rollout_do_sample = bool(
                verl_cfg.get("rollout_do_sample", sampling_params.get("do_sample", True))
            )

            val_temperature = float(
                val_kwargs_cfg.get(
                    "temperature",
                    verl_cfg.get("val_temperature", sampling_params.get("temperature", 0.6)),
                )
            )
            val_top_p = float(
                val_kwargs_cfg.get(
                    "top_p",
                    verl_cfg.get("val_top_p", sampling_params.get("top_p", 0.95)),
                )
            )
            val_top_k = int(
                val_kwargs_cfg.get(
                    "top_k",
                    verl_cfg.get("val_top_k", sampling_params.get("top_k", -1)),
                )
            )
            val_do_sample = bool(
                val_kwargs_cfg.get(
                    "do_sample",
                    verl_cfg.get("val_do_sample", sampling_params.get("do_sample", True)),
                )
            )
            if rollout_name == "sglang" and dp_size > 1 and not verl_cfg.get(
                "allow_standalone_sglang_dp", False
            ):
                logger.warning(
                    "SGLang standalone eval with dp_size>1 is unstable and may fail with "
                    "DistNetworkError (EADDRINUSE). Falling back dp_size from %s to 1. "
                    "Set verl.allow_standalone_sglang_dp=true to disable this safeguard.",
                    dp_size,
                )
                dp_size = 1
            n_gpus = int(verl_cfg.get("n_gpus", tp_size * dp_size * pp_size))
            nnodes = int(verl_cfg.get("nnodes", 1))
            n_gpus_per_node = int(verl_cfg.get("n_gpus_per_node", n_gpus))
            # Keep eval lightweight: we only need server replicas for VerlEngine.
            # Agent loop workers are not used by rllm.eval + VerlEngine path.
            agent_loop_workers = int(verl_cfg.get("agent_loop_num_workers", 0))

            verl_config = {
                "actor_rollout_ref": {
                    "model": {
                        "path": model_name,
                    },
                    "rollout": {
                        "_target_": "verl.workers.config.RolloutConfig",
                        "name": rollout_name,
                        "gpu_memory_utilization": gpu_memory_utilization,
                        "data_parallel_size": dp_size,
                        "tensor_model_parallel_size": tp_size,
                        "pipeline_model_parallel_size": pp_size,
                        "mode": rollout_mode,
                        "free_cache_engine": free_cache_engine,
                        "enforce_eager": enforce_eager,
                        "disable_log_stats": True,
                        "prompt_length": self.config["execution"].get("max_prompt_length", 4096),
                        "response_length": self.config["execution"].get("max_response_length", 16384),
                        "prometheus": {
                            "_target_": "verl.workers.config.PrometheusConfig",
                            "enable": False,
                        },
                        "agent": {
                            "_target_": "verl.workers.config.AgentLoopConfig",
                            "num_workers": agent_loop_workers,
                            "agent_loop_config_path": None,
                            "custom_async_server": {
                                "_target_": "verl.workers.config.CustomAsyncServerConfig",
                                "path": None,
                                "name": None,
                            },
                        },
                        "temperature": rollout_temperature,
                        "top_p": rollout_top_p,
                        "top_k": rollout_top_k,
                        "do_sample": rollout_do_sample,
                        "n": rollout_n,
                        "val_kwargs": {
                            "_target_": "verl.workers.config.SamplingConfig",
                            "temperature": val_temperature,
                            "top_p": val_top_p,
                            "top_k": val_top_k,
                            "do_sample": val_do_sample,
                            "n": val_n,
                        },
                    },
                    "hybrid_engine": True,
                },
                "trainer": {
                    "nnodes": nnodes,
                    "n_gpus_per_node": n_gpus_per_node,
                    "project_name": "rllm_eval",
                    "experiment_name": f"{env_name}_eval",
                },
                "data": {
                    "max_prompt_length": self.config["execution"].get("max_prompt_length", 4096),
                    "max_response_length": self.config["execution"].get("max_response_length", 16384),
                },
                "reward_model": {
                    "enable": False,
                    "enable_resource_pool": False,
                },
                "rllm": {
                    "disable_thinking": self.config.get("disable_thinking", False),
                    "accumulate_reasoning": self.config.get("accumulate_reasoning", False),
                },
                "verl": verl_cfg,
            }

            rollout_manager, verl_config_obj = _setup_verl_rollout_manager(verl_config)
            rollout_engine_args = rollout_manager
            verl_full_config = verl_config_obj

        elif engine_name == "sglang":
            rollout_engine_args = {
                "model_path": model_name,
                "tp_size": self.config.get("sglang", {}).get("tp_size", 1),
                "dp_size": self.config.get("sglang", {}).get("dp_size", 1),
                "mem_fraction_static": self.config.get("sglang", {}).get(
                    "mem_fraction_static", 0.85
                ),
                "trust_remote_code": self.config.get("sglang", {}).get(
                    "trust_remote_code", True
                ),
            }
            for key in ["dtype", "quantization", "context_length", "device", "enable_deterministic_inference"]:
                if key in self.config.get("sglang", {}):
                    rollout_engine_args[key] = self.config["sglang"][key]
            verl_full_config = None

        else:
            max_key_cycles = int(self.config["model"].get("max_key_cycles", 2))
            if _is_closed_model(model_name):
                closed_config = _get_closed_config(
                    model_name,
                    configured_api_key=self.config["model"].get("api_key"),
                    configured_base_url=self.config["model"].get("base_url"),
                    configured_api_keys=self.config["model"].get("api_keys"),
                )
                rollout_engine_args = {
                    "base_url": closed_config["base_url"],
                    "api_key": closed_config["api_key"],
                    "api_keys": closed_config["api_keys"],
                    "max_key_cycles": max_key_cycles,
                    "model": closed_config["actual_model"],
                    "force_chat_completions": True,
                }
                sampling_params["model"] = closed_config["actual_model"]
                logger.info(
                    f"Using closed-source API for model '{model_name}' -> "
                    f"'{closed_config['actual_model']}' "
                    f"({len(closed_config['api_keys'])} api key(s), max_key_cycles={max_key_cycles})"
                )
            else:
                # Non-closed OpenAI-compatible endpoint. Support optional api_keys
                # list for failover; falls back to single api_key (unchanged default).
                _raw_keys = self.config["model"].get("api_keys")
                _api_keys = []
                if isinstance(_raw_keys, str):
                    _api_keys = [k.strip() for k in _raw_keys.split(",") if k.strip()]
                elif isinstance(_raw_keys, (list, tuple)):
                    _api_keys = [str(k).strip() for k in _raw_keys if str(k).strip()]
                elif _raw_keys is not None:
                    # Bare scalar (e.g. int from an over-eager CLI type coercion).
                    _scalar = str(_raw_keys).strip()
                    if _scalar:
                        _api_keys = [_scalar]
                rollout_engine_args = {
                    "base_url": self.config["model"]["base_url"],
                    "api_key": self.config["model"].get("api_key", "None"),
                    "model": model_name,
                }
                if _api_keys:
                    rollout_engine_args["api_keys"] = _api_keys
                    rollout_engine_args["max_key_cycles"] = max_key_cycles
            verl_full_config = None

        runtime_sampling_params = sampling_params.copy()
        if engine_name == "verl":
            # VerlEngine already gets training/validation sampling defaults from
            # actor_rollout_ref.rollout. Remove API-style/non-sglang keys here.
            for key in ["model", "do_sample", "return_logprobs", "entropy_mode"]:
                runtime_sampling_params.pop(key, None)

        # When ContextManager is active (history_length >= 0), enable per-step
        # prompt budget so that the engine does NOT apply cumulative response
        # truncation.  ContextManager's sliding window keeps the prompt bounded,
        # but without this flag the engine would still sum up all response tokens
        # across steps and terminate with TRUNCATION once that sum exceeds
        # max_response_length — defeating the purpose of the sliding window.
        history_length = agent_args.get("history_length")
        use_context_manager = history_length is not None and int(history_length) >= 0
        enforce_max_prompt = (
            self.config["execution"].get("enforce_max_prompt_length", use_context_manager)
        )
        if use_context_manager and enforce_max_prompt:
            logger.info(
                "ContextManager active (history_length=%s) — enabling "
                "enforce_max_prompt_length to disable cumulative response truncation.",
                history_length,
            )

        engine_kwargs = {
            "agent_class": agent_class,
            "env_class": env_config.env_class,
            "agent_args": agent_args,
            "env_args": env_args,
            "engine_name": engine_name,
            "tokenizer": tokenizer,
            "sampling_params": runtime_sampling_params,
            "max_response_length": self.config["execution"].get("max_response_length", 16384),
            "max_prompt_length": self.config["execution"].get("max_prompt_length", 4096),
            "n_parallel_agents": self.config["execution"]["n_parallel_agents"],
            "max_steps": self.config["execution"]["max_steps"],
            "max_workers": self.config["execution"].get("max_workers", 64),
            "enforce_max_prompt_length": enforce_max_prompt,
            "log_trajectory_turn_progress": self.config["execution"].get(
                "log_trajectory_turn_progress", False
            ),
            "trajectory_progress_log_interval": self.config["execution"].get(
                "trajectory_progress_log_interval", 1
            ),
            "progress_jsonl_path": self._progress_jsonl_path(),
        }

        if engine_name == "verl":
            engine_kwargs["config"] = verl_full_config
            engine_kwargs["rollout_engine"] = rollout_engine_args
        else:
            engine_kwargs["rollout_engine_args"] = rollout_engine_args

        self.engine = AgentExecutionEngine(**engine_kwargs)
        self._patch_engine_for_eval(self.engine, engine_name)

        return self.engine

    def _patch_engine_for_eval(self, engine, engine_name: str) -> None:
        """Apply eval-time compatibility patches without touching engine core code."""
        if engine_name != "verl":
            return

        # Eval should use validation sampling params for VerlEngine to stay aligned
        # with training-time validation behavior.
        rollout_engine = getattr(engine, "rollout_engine", None)
        if rollout_engine is not None and hasattr(rollout_engine, "validate"):
            rollout_engine.validate = True

        if not hasattr(engine, "get_model_response"):
            return

        original_get_model_response = engine.get_model_response
        if getattr(original_get_model_response, "_rllm_eval_appid_str_patch", False):
            return

        @wraps(original_get_model_response)
        async def _wrapped_get_model_response(prompt, application_id, **kwargs):
            return await original_get_model_response(
                prompt, application_id=str(application_id), **kwargs
            )

        _wrapped_get_model_response._rllm_eval_appid_str_patch = True  # type: ignore[attr-defined]
        engine.get_model_response = _wrapped_get_model_response

    def load_data(self) -> list[dict]:
        """Load tasks using environment-specific data loader.

        Returns:
            List of task dictionaries.
        """
        env_name = self.config["env"]

        # Import env_configs to ensure all environments are registered
        import rllm.eval.env_configs  # noqa: F401

        env_config = EnvConfigRegistry.get(env_name)

        data_config = self.config.get("data", {})
        split = data_config.get("split", "test")
        limit = data_config.get("limit")

        # Build kwargs for data loader
        loader_kwargs = {"split": split, "limit": limit}

        # Pass data_file if specified (allows selecting a specific parquet file)
        data_file = data_config.get("data_file")
        if data_file is not None:
            loader_kwargs["data_file"] = data_file

        # Add environment-specific kwargs
        if env_name == "tau2":
            loader_kwargs["domain"] = self.config.get(
                "domain", env_config.env_args.get("domain", "airline")
            )
        elif env_name == "miniwob":
            loader_kwargs["subtask"] = self.config.get(
                "subtask", env_config.env_args.get("subtask", "miniwob")
            )

        if env_config.data_loader is None:
            raise ValueError(f"No data loader configured for environment '{env_name}'")

        tasks = env_config.data_loader(**loader_kwargs)

        # Expand tasks for multiple rollouts per prompt (for pass@k evaluation)
        n_rollouts = self.config.get("execution", {}).get("n_rollouts_per_prompt", 1)
        if n_rollouts > 1:
            expanded_tasks = []
            for task in tasks:
                for rollout_id in range(n_rollouts):
                    task_copy = task.copy()
                    task_copy["_rollout_id"] = rollout_id
                    expanded_tasks.append(task_copy)
            logger.info(
                f"Expanded {len(tasks)} tasks to {len(expanded_tasks)} "
                f"with {n_rollouts} rollouts per prompt"
            )
            return expanded_tasks

        return tasks

    def _progress_jsonl_path(self) -> str:
        """Path of the per-trajectory progress jsonl, under the same save_dir.

        Returns ``{save_dir}/{env}_progress.jsonl``. Always defined (engine only
        writes to it; resume only reads it when ``execution.resume`` is true).
        """
        output_config = self.config.get("output", {})
        save_dir = output_config.get("save_dir", "./results")
        env_name = self.config["env"]
        return os.path.join(save_dir, f"{env_name}_progress.jsonl")

    def _load_progress_jsonl(self):
        """Read the progress jsonl, returning (done_keys set, recovered Trajectory list).

        Corrupt/half-written lines are skipped. Missing file -> empty. The newest
        record for a duplicate task_key wins (later line overrides earlier).
        """
        from rllm.agents.agent import Trajectory

        path = self._progress_jsonl_path()
        done = {}  # task_key -> Trajectory (dedupe; last wins)
        if not os.path.exists(path):
            return set(), []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    key = rec["task_key"]
                    traj = Trajectory.from_dict(rec["trajectory"])
                    done[key] = traj
                except Exception:
                    # Half-written / corrupt line (e.g. process killed mid-write): skip.
                    continue
        return set(done.keys()), list(done.values())

    async def run(self) -> list:
        """Execute evaluation and return results.

        Returns:
            List of Trajectory objects.
        """
        if self.engine is None:
            self.setup_engine()

        tasks = self.load_data()
        logger.info(f"Loaded {len(tasks)} tasks for evaluation")
        print(f"Loaded {len(tasks)} tasks for evaluation")

        # --- Resume support ---
        # When execution.resume is on, skip tasks already recorded in the progress
        # jsonl and merge their recovered trajectories back into the final results.
        # Default off => byte-for-byte identical to the previous behavior.
        resume = bool(self.config.get("execution", {}).get("resume", False))
        recovered: list = []
        progress_path = self._progress_jsonl_path()
        if resume:
            from rllm.eval.metrics import compute_task_key

            done_keys, recovered = self._load_progress_jsonl()
            if done_keys:
                before = len(tasks)
                tasks = [t for t in tasks if compute_task_key(t) not in done_keys]
                msg = (
                    f"[resume] Recovered {len(recovered)} completed trajectories; "
                    f"skipping {before - len(tasks)} done task(s), {len(tasks)} remaining."
                )
                logger.info(msg)
                print(msg)
        else:
            # Not resuming but a stale progress file exists -> back it up so it does
            # not silently contaminate a fresh run's progress log.
            if os.path.exists(progress_path):
                import time

                backup = f"{progress_path}.bak.{int(time.time())}"
                try:
                    os.replace(progress_path, backup)
                    logger.info(f"[resume] Backed up stale progress file -> {backup}")
                except OSError as e:
                    logger.warning(f"[resume] Could not back up stale progress file: {e}")

        if not tasks:
            logger.warning("No tasks to evaluate")
            # All tasks already done (resume) -> results are the recovered set.
            self.results = recovered
            return self.results

        new_results = await self.engine.execute_tasks(tasks)
        # Final results = recovered (resumed) + newly run, so save_results /
        # compute_metrics see the complete set.
        self.results = recovered + new_results
        return self.results

    def run_sync(self) -> list:
        """Synchronous wrapper for run().

        Returns:
            List of Trajectory objects.
        """
        # Setup engine and initialize in main thread before async operations
        # This is required for engines like sglang that need main thread
        # initialization for signal handlers
        if self.engine is None:
            self.setup_engine()
        self.engine.initialize()
        return asyncio.run(self.run())

    def _ensure_offline_logprobs_computed(self) -> None:
        """Ensure offline logprobs are computed if configured.

        This method handles:
        1. Checking if offline logprobs source is configured
        2. Shutting down the online engine to free GPU memory
        3. Loading Transformers model and computing logprobs/entropies
        4. Tracking state to avoid recomputation

        Must be called before computing metrics that need logprob data.
        """
        metrics_cfg = self.config.get("metrics", {})
        logprobs_source = metrics_cfg.get("logprobs_source", "none")

        # Skip if not using offline mode or already computed
        if logprobs_source != "offline" or self._offline_logprobs_computed:
            return

        if self.results is None:
            logger.warning("No results available for offline logprob computation")
            return

        try:
            from rllm.eval.offline_logprob import populate_offline_logprobs

            # CRITICAL: Shutdown rollout engine FIRST to free GPU memory
            # This must happen before loading the Transformers model
            if self.engine is not None:
                logger.info("Shutting down online engine before offline logprob computation...")
                try:
                    self.engine.shutdown()
                except Exception as e:
                    logger.warning(f"Error shutting down engine: {e}")
                self.engine = None

            # Get offline configuration
            offline_cfg = metrics_cfg.get("offline_logprobs", {})
            model_name = offline_cfg.get("model_name") or self.config["model"]["name"]
            tokenizer_name = offline_cfg.get("tokenizer_name")
            device = offline_cfg.get("device", "auto")  # Default to auto for multi-GPU
            dtype = offline_cfg.get("dtype", "bfloat16")
            batch_size = int(offline_cfg.get("batch_size", 1))
            gpu_memory_utilization = float(offline_cfg.get("gpu_memory_utilization", 0.9))
            parallel_strategy = offline_cfg.get("parallel_strategy", "naive")

            # Advanced options
            advanced_opts = {
                "use_flash_attention": offline_cfg.get("use_flash_attention", True),
                "gradient_checkpointing": offline_cfg.get("gradient_checkpointing", False),
                "sequence_bucketing": offline_cfg.get("sequence_bucketing", True),
                "bucket_size": int(offline_cfg.get("bucket_size", 128)),
                "max_sequence_length": int(offline_cfg.get("max_sequence_length", 8192)),
                # FSDP options
                "fsdp_cpu_offload": offline_cfg.get("fsdp_cpu_offload", False),
                "fsdp_sharding_strategy": offline_cfg.get("fsdp_sharding_strategy", "FULL_SHARD"),
                # vLLM options
                "vllm_tensor_parallel_size": int(offline_cfg.get("vllm_tensor_parallel_size", 1)),
                "vllm_gpu_memory_utilization": float(offline_cfg.get("vllm_gpu_memory_utilization", 0.9)),
                # DeepSpeed options
                "deepspeed_zero_stage": int(offline_cfg.get("deepspeed_zero_stage", 3)),
                "deepspeed_offload_device": offline_cfg.get("deepspeed_offload_device", "cpu"),
            }

            logger.info(f"Computing offline logprobs/entropies with Transformers model: {model_name}")
            logger.info(f"Using parallel strategy: {parallel_strategy}")
            print(f"Computing offline logprobs/entropies with model: {model_name}")
            print(f"Using parallel strategy: {parallel_strategy}")
            populate_offline_logprobs(
                self.results,
                model_name=model_name,
                tokenizer_name=tokenizer_name,
                device=device,
                dtype=dtype,
                batch_size=batch_size,
                gpu_memory_utilization=gpu_memory_utilization,
                parallel_strategy=parallel_strategy,
                **advanced_opts,
            )
            logger.info("Offline logprobs/entropies computed successfully.")
            print("Offline logprobs/entropies computed successfully.")

        except Exception as e:
            logger.warning(f"Offline logprob computation failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Mark as computed (even on failure) to avoid repeated attempts
            self._offline_logprobs_computed = True

    def compute_metrics(self, include_logprob_metrics: bool | None = None) -> dict[str, Any]:
        """Compute metrics on results.

        Args:
            include_logprob_metrics: Whether to include logprob-based metrics
                (SLP, TLP, SLE, TLE, etc.). If None, uses config setting.

        Returns:
            Dictionary of computed metrics.

        Raises:
            ValueError: If no results are available.
        """
        if self.results is None:
            raise ValueError("No results available. Run evaluation first.")

        # Ensure offline logprobs are computed before metrics calculation
        # This will shutdown the online engine and load Transformers model if needed
        self._ensure_offline_logprobs_computed()

        # Import env_configs to ensure all environments are registered
        import rllm.eval.env_configs  # noqa: F401

        env_config = EnvConfigRegistry.get(self.config["env"])

        # Determine whether to include logprob metrics
        if include_logprob_metrics is None:
            include_logprob_metrics = self.config.get("metrics", {}).get(
                "include_logprob_metrics", False
            )

        print(f"Success threshold is {env_config.success_threshold}")
        metrics = compute_metrics(
            self.results,
            env_config.metrics,
            include_logprob_metrics=include_logprob_metrics,
            success_threshold=env_config.success_threshold,
            env_name=self.config["env"],
        )

        # Per-split breakdown: when a single run mixes multiple dataset splits
        # (e.g. ALFWorld's combined test.parquet = eval_in_distribution +
        # eval_out_of_distribution), also compute metrics SEPARATELY for each
        # split so the user gets per-split numbers in addition to the pooled
        # ones. The breakdown only activates when >=2 distinct splits are
        # present, so single-split runs and other environments are unaffected.
        split_groups: dict[str, list] = {}
        for traj in self.results:
            task = getattr(traj, "task", None)
            split_val = task.get("split") if isinstance(task, dict) else None
            if split_val is None:
                continue
            split_groups.setdefault(str(split_val), []).append(traj)

        if len(split_groups) >= 2:
            print(
                f"Detected {len(split_groups)} dataset splits in results "
                f"({', '.join(sorted(split_groups))}); computing per-split metrics."
            )
            for split_val, split_results in sorted(split_groups.items()):
                split_metrics = compute_metrics(
                    split_results,
                    env_config.metrics,
                    include_logprob_metrics=include_logprob_metrics,
                    success_threshold=env_config.success_threshold,
                    env_name=self.config["env"],
                )
                # Namespace each split's metrics under the split name so they
                # don't collide with the pooled metrics (e.g.
                # "eval_in_distribution/pass_at_1").
                for k, v in split_metrics.items():
                    metrics[f"{split_val}/{k}"] = v

        return metrics

    def save_results(self) -> str | None:
        """Save trajectories and metrics.

        Returns:
            Path to saved file, or None if saving is disabled.
        """
        output_config = self.config.get("output", {})
        if not output_config.get("save_trajectories", True):
            return None

        if self.results is None:
            logger.warning("No results to save")
            return None

        save_dir = output_config.get("save_dir", "./results")
        os.makedirs(save_dir, exist_ok=True)

        # Generate filename
        env_name = self.config["env"]
        filename = output_config.get("filename", f"{env_name}_trajectories.pt")
        save_path = os.path.join(save_dir, filename)

        # Ensure offline logprobs are computed before saving (if configured)
        # This will shutdown the online engine and load Transformers model if needed
        self._ensure_offline_logprobs_computed()

        # Save trajectories (after offline logprobs are computed, so they're included)
        torch.save(self.results, save_path)
        logger.info(f"Trajectories saved to {save_path}")
        print(f"Trajectories saved to {save_path}")

        # Also save metrics as JSON
        try:
            metrics = self.compute_metrics()
            metrics_path = os.path.join(save_dir, f"{env_name}_metrics.json")
            self._save_metrics_to_json(metrics, metrics_path)
            logger.info(f"Metrics saved to {metrics_path}")
        except Exception as e:
            logger.warning(f"Could not save metrics: {e}")

        # Save logprob metrics if enabled
        include_logprob_metrics = self.config.get("metrics", {}).get(
            "include_logprob_metrics", False
        )
        if include_logprob_metrics:
            try:
                logprob_metrics = compute_all_logprob_metrics(
                    self.results,
                    success_threshold=EnvConfigRegistry.get(self.config["env"]).success_threshold,
                    env_name=self.config["env"],
                )
                logprob_metrics_path = os.path.join(
                    save_dir, f"{env_name}_logprob_metrics.json"
                )
                self._save_metrics_to_json(logprob_metrics, logprob_metrics_path)
                logger.info(f"Logprob metrics saved to {logprob_metrics_path}")
                print(f"Logprob metrics saved to {logprob_metrics_path}")
            except Exception as e:
                logger.warning(f"Could not save logprob metrics: {e}")

        # Save detailed token/step level metrics if enabled
        save_detailed_metrics = self.config.get("metrics", {}).get(
            "save_detailed_metrics", False
        )
        if save_detailed_metrics:
            self._save_detailed_metrics(save_dir, env_name)

        return save_path

    def _save_detailed_metrics(self, save_dir: str, env_name: str) -> None:
        """Save detailed token/step/trajectory level metrics.

        Args:
            save_dir: Directory to save metrics.
            env_name: Environment name for file naming.
        """
        try:
            tokenizer = self._get_tokenizer()

            # Compute detailed trajectory metrics
            print("Computing detailed trajectory metrics (this may take a moment)...")
            detailed_metrics = compute_detailed_trajectory_metrics(
                self.results,
                tokenizer=tokenizer,
                include_token_text=tokenizer is not None,
            )

            # Save detailed trajectory metrics
            detailed_path = os.path.join(save_dir, f"{env_name}_detailed_metrics.json")
            self._save_metrics_to_json(detailed_metrics, detailed_path)
            logger.info(f"Detailed trajectory metrics saved to {detailed_path}")
            print(f"Detailed trajectory metrics saved to {detailed_path}")

            # Compute and save token aggregations if tokenizer is available
            if tokenizer is not None:
                # Save token-level aggregated statistics
                token_stats = aggregate_token_metrics_by_text(detailed_metrics)
                token_stats_path = os.path.join(
                    save_dir, f"{env_name}_token_stats.json"
                )
                self._save_metrics_to_json(token_stats, token_stats_path)
                logger.info(f"Token statistics saved to {token_stats_path}")
                print(f"Token statistics saved to {token_stats_path}")

            # Save high entropy tokens
            high_entropy_tokens = get_high_entropy_tokens(detailed_metrics, top_k=500)
            high_entropy_path = os.path.join(
                save_dir, f"{env_name}_high_entropy_tokens.json"
            )
            self._save_metrics_to_json(high_entropy_tokens, high_entropy_path)
            logger.info(f"High entropy tokens saved to {high_entropy_path}")
            print(f"High entropy tokens saved to {high_entropy_path}")

            # Save success vs failure comparison
            comparison = compare_success_vs_failure_tokens(detailed_metrics)
            comparison_path = os.path.join(
                save_dir, f"{env_name}_success_failure_comparison.json"
            )
            self._save_metrics_to_json(comparison, comparison_path)
            logger.info(f"Success/failure comparison saved to {comparison_path}")
            print(f"Success/failure comparison saved to {comparison_path}")

        except Exception as e:
            logger.warning(f"Could not save detailed metrics: {e}")
            import traceback
            traceback.print_exc()

    def _save_metrics_to_json(self, metrics: dict[str, Any], path: str) -> None:
        """Save metrics dictionary to JSON file.

        Args:
            metrics: Dictionary of metrics to save.
            path: Path to save the JSON file.
        """
        import math

        def make_serializable(obj: Any) -> Any:
            """Convert object to JSON-serializable format."""
            if isinstance(obj, float):
                if math.isnan(obj) or math.isinf(obj):
                    return str(obj)
                return obj
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, (int, str, bool, type(None))):
                return obj
            else:
                return str(obj)

        serializable_metrics = make_serializable(metrics)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable_metrics, f, indent=2)

    def print_summary(self, include_logprob_metrics: bool | None = None) -> None:
        """Print evaluation summary.

        Args:
            include_logprob_metrics: Whether to include logprob-based metrics.
                If None, uses config setting.
        """
        if self.results is None:
            print("No results available. Run evaluation first.")
            return

        # Determine whether to include logprob metrics
        if include_logprob_metrics is None:
            include_logprob_metrics = self.config.get("metrics", {}).get(
                "include_logprob_metrics", False
            )

        metrics = self.compute_metrics(include_logprob_metrics=include_logprob_metrics)
        title = f"EVALUATION RESULTS: {self.config['env'].upper()}"
        print_metrics(metrics, title, compact=True)

    @classmethod
    def from_args(cls, args) -> "EvalPortal":
        """Create portal from CLI arguments.

        Args:
            args: Parsed command-line arguments.

        Returns:
            EvalPortal instance.
        """
        config: dict[str, Any] = {
            "env": args.env,
            "model": {
                "name": args.model,
                "base_url": args.base_url,
                "api_key": getattr(args, "api_key", "None"),
            },
            "data": {
                "split": args.split,
                "limit": args.limit,
            },
            "execution": {
                "n_parallel_agents": args.n_parallel,
                "max_steps": args.max_steps,
                "n_rollouts_per_prompt": getattr(args, "n_rollouts", 1),
            },
            "output": {
                "save_dir": args.output_dir,
                "save_trajectories": not getattr(args, "no_save", False),
            },
            "metrics": {
                "include_logprob_metrics": getattr(args, "logprob_metrics", False),
                "save_detailed_metrics": getattr(args, "detailed_metrics", False),
            },
        }

        # Add environment-specific args
        if hasattr(args, "domain") and args.domain:
            config["domain"] = args.domain
        if hasattr(args, "solo_mode") and args.solo_mode:
            config["solo_mode"] = args.solo_mode
        if hasattr(args, "subtask") and args.subtask:
            config["subtask"] = args.subtask
        # Closed-source API and user simulator support
        if hasattr(args, "user_llm") and args.user_llm:
            config["user_llm"] = args.user_llm
        if hasattr(args, "tokenizer_name") and args.tokenizer_name:
            config["model"]["tokenizer_name"] = args.tokenizer_name
        # AppWorld world model file support
        if hasattr(args, "world_model_file") and args.world_model_file:
            config["world_model_file"] = args.world_model_file

        return cls(config)

    def shutdown(self) -> None:
        """Clean up resources."""
        if self.engine is not None:
            self.engine.shutdown()
            self.engine = None

        if self._get_engine_name() == "verl":
            import ray
            if ray.is_initialized():
                ray.shutdown()
                logger.info("Ray cluster shut down")
