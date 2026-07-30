#!/usr/bin/env python3
"""CLI entry point for unified evaluation.

This module provides the command-line interface for the rLLM unified
evaluation portal.

Usage:
    # Evaluate WebShop
    python -m rllm.eval.cli --env webshop --limit 100

    # Evaluate Sokoban
    python -m rllm.eval.cli --env sokoban --limit 50

    # Use config file
    python -m rllm.eval.cli --config rllm/eval/config/webshop.yaml
"""

import argparse
import asyncio
import logging
import re
import sys
from typing import Any

from rllm.eval.portal import EvalPortal
from rllm.eval.registry import EnvConfigRegistry

# Supported environments
SUPPORTED_ENVS = [
    "sokoban",
    "minesweeper",
    "rush_hour",
    "alfworld",
    "webshop",
]

SET_OVERRIDE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")


def setup_logging(verbose: bool = False) -> None:
    """Configure logging.

    Args:
        verbose: If True, set logging level to DEBUG.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description="Unified rLLM Evaluation Portal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate WebShop with 100 examples
    python -m rllm.eval.cli --env webshop --limit 100

    # Evaluate Sokoban
    python -m rllm.eval.cli --env sokoban --limit 50

    # Use a configuration file
    python -m rllm.eval.cli --config rllm/eval/config/webshop.yaml

    # Evaluate with logprob-based metrics (SLP, TLP, SLE, TLE, etc.)
    python -m rllm.eval.cli --env webshop --limit 100 --logprob-metrics

    # Evaluate with detailed token/step level metrics for trajectory analysis
    python -m rllm.eval.cli --env webshop --limit 100 --detailed-metrics

    # List available environments
    python -m rllm.eval.cli --list-envs
        """,
    )

    # Environment selection
    parser.add_argument(
        "--env",
        type=str,
        choices=SUPPORTED_ENVS,
        help="Environment to evaluate",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config YAML/JSON file (overrides other arguments)",
    )
    parser.add_argument(
        "--list-envs",
        action="store_true",
        help="List available environments and exit",
    )

    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--engine",
        type=str,
        choices=["openai", "sglang", "verl", "tinker"],
        default=None,
        help="Engine type override. If omitted in --config mode, keep engine from YAML.",
    )
    model_group.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-4B",
        help="Model name or path (default: Qwen/Qwen3-4B)",
    )
    model_group.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:30000/v1",
        help="LLM server URL (default: http://localhost:30000/v1)",
    )
    model_group.add_argument(
        "--api-key",
        type=str,
        default="None",
        help="API key for LLM server (default: None)",
    )

    # Data configuration
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument(
        "--split",
        type=str,
        default="test",
        help="Data split to evaluate (default: test)",
    )
    data_group.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of examples to evaluate",
    )

    # Execution configuration
    exec_group = parser.add_argument_group("Execution Configuration")
    exec_group.add_argument(
        "--n-parallel",
        type=int,
        default=64,
        help="Number of parallel agents (default: 64)",
    )
    exec_group.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Maximum steps per episode (default: 50)",
    )
    exec_group.add_argument(
        "--n-rollouts",
        type=int,
        default=1,
        help="Number of rollouts per prompt for pass@k evaluation (default: 1)",
    )
    exec_group.add_argument(
        "--resume",
        action="store_true",
        help="Resume a prior interrupted run: skip tasks already recorded in "
             "{save_dir}/{env}_progress.jsonl and only run the remaining ones.",
    )
    exec_group.add_argument(
        "--max-key-cycles",
        type=int,
        default=None,
        help="Max number of full cycles over the API key pool before a request "
             "gives up (only relevant when model.api_keys has >1 key; default 2).",
    )

    # Output configuration
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Directory to save results (default: ./results)",
    )
    output_group.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save trajectories",
    )

    # Metrics configuration
    metrics_group = parser.add_argument_group("Metrics Configuration")
    metrics_group.add_argument(
        "--logprob-metrics",
        action="store_true",
        help="Compute and save logprob-based metrics (SLP, TLP, SLE, TLE, etc.)",
    )
    metrics_group.add_argument(
        "--detailed-metrics",
        action="store_true",
        help="Save detailed token/step level metrics paired with trajectory content. "
             "Enables analysis of specific tokens (e.g., 'Wait') and their metrics.",
    )

    # Environment-specific arguments
    env_group = parser.add_argument_group("Environment-Specific Arguments")
    env_group.add_argument(
        "--domain",
        type=str,
        choices=["airline", "retail", "telecom"],
        help="Domain for tau2 environment",
    )
    env_group.add_argument(
        "--solo-mode",
        action="store_true",
        help="Solo mode for tau2 (no user simulator)",
    )
    env_group.add_argument(
        "--subtask",
        type=str,
        help="Subtask for miniwob environment",
    )
    env_group.add_argument(
        "--user-llm",
        type=str,
        help="LLM model for user simulator (tau2). Use 'closed/' prefix for a closed-source API.",
    )
    env_group.add_argument(
        "--tokenizer-name",
        type=str,
        help="Tokenizer name for closed-source models (defaults to Qwen/Qwen3-4B)",
    )
    env_group.add_argument(
        "--world-model-file",
        type=str,
        help="Path to JSON file containing world model summary (for appworld environment)",
    )

    # Misc
    parser.add_argument(
        "--set",
        dest="sets",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Generic config override (repeatable), e.g. "
            "--set execution.max_steps=80 --set verl.tp_size=4"
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser


def list_environments() -> None:
    """Print list of available environments."""
    # Import env_configs to ensure all environments are registered
    try:
        import rllm.eval.env_configs  # noqa: F401
    except ImportError:
        pass

    print("\nAvailable Environments:")
    print("=" * 50)

    registered = EnvConfigRegistry.list_envs()
    for env_name in SUPPORTED_ENVS:
        status = "✓ registered" if env_name in registered else "○ not registered"
        print(f"  {env_name:15} {status}")

        if env_name in registered:
            config = EnvConfigRegistry.get(env_name)
            if config.description:
                print(f"                  {config.description}")

    print("=" * 50)
    print(f"\nTotal: {len(registered)} environments registered")


def _init_ray(n_parallel: int = 64) -> None:
    """Initialize Ray before creating any environments.

    Some environments (like ALFWorld) use Ray for process-level isolation.
    Initializing Ray once at the start prevents race conditions when multiple
    threads try to initialize Ray simultaneously.

    Args:
        n_parallel: Number of parallel agents (used to set num_cpus).
    """
    import os

    # Opt-in skip: environments that do NOT use Ray (e.g. sokoban / minesweeper /
    # rush_hour over a closed-source OpenAI-compatible gateway) gain nothing from a local Ray
    # cluster, and each cluster is ~5-8 processes + plasma store. When many such
    # evals run concurrently on one host, the stacked Ray clusters exhaust memory
    # and the platform SIGTERMs the whole process group. Set RLLM_EVAL_SKIP_RAY=1
    # to skip ray.init() for these. Default unset = initialize as before, so every
    # existing caller (ALFWorld/WebShop/etc.) is unaffected.
    if os.environ.get("RLLM_EVAL_SKIP_RAY", "0") == "1":
        print("RLLM_EVAL_SKIP_RAY=1; skipping Ray initialization (env does not need Ray).")
        return

    try:
        import ray

        if not ray.is_initialized():
            print("Initializing Ray...")
            # The Ray dashboard / metrics agent is unstable on some nodes
            # ("Failed to export metrics to the metrics agent"); when that agent
            # dies Ray marks the whole node as crashed, which surfaces as
            # "Owner's node has crashed" / ActorDiedError and hangs env.reset().
            # The dashboard is not needed for eval, so honor RAY_DISABLE_DASHBOARD
            # (set by the launch scripts) and default to disabling it.
            disable_dashboard = os.environ.get("RAY_DISABLE_DASHBOARD", "1") not in ("0", "false", "False")
            init_kwargs = dict(
                ignore_reinit_error=True,
                include_dashboard=not disable_dashboard,
                num_cpus=min(os.cpu_count() or 8, n_parallel),
            )
            # Allow pinning a stable temp dir (e.g. /tmp/ray) to avoid /dev/shm
            # exhaustion or stale-session conflicts across runs.
            ray_temp_dir = os.environ.get("RAY_EVAL_TEMP_DIR")
            if ray_temp_dir:
                init_kwargs["_temp_dir"] = ray_temp_dir
            ray.init(**init_kwargs)
            print(f"Ray initialized successfully (dashboard={'off' if disable_dashboard else 'on'})")
    except ImportError:
        # Ray not installed, skip initialization
        pass
    except Exception as e:
        logging.warning(f"Failed to initialize Ray: {e}")


def _shutdown_ray() -> None:
    """Shut down Ray if it was initialized."""
    try:
        import ray

        if ray.is_initialized():
            ray.shutdown()
            print("Ray shut down")
    except Exception:
        pass


def _parse_typed_override(value: str) -> Any:
    """Parse CLI override value into bool/int/float/null/str."""
    lower_value = value.lower()
    if lower_value == "true":
        return True
    if lower_value == "false":
        return False
    if lower_value in {"null", "none"}:
        return None

    if re.fullmatch(r"[+-]?\d+", value):
        try:
            return int(value)
        except ValueError:
            pass
    if re.fullmatch(r"[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?", value):
        try:
            return float(value)
        except ValueError:
            pass

    return value


def _set_nested_config_value(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set nested dict value by dotted path, creating intermediate dicts."""
    if not dotted_key or not SET_OVERRIDE_PATTERN.match(dotted_key):
        raise ValueError(
            f"Invalid --set key '{dotted_key}'. Use dotted keys like execution.max_steps."
        )

    keys = dotted_key.split(".")
    cur = config
    for key in keys[:-1]:
        existing = cur.get(key)
        if existing is None:
            cur[key] = {}
        elif not isinstance(existing, dict):
            raise ValueError(
                f"Cannot set '{dotted_key}': '{key}' is not a mapping in current config."
            )
        cur = cur[key]
    cur[keys[-1]] = value


def _apply_explicit_cli_overrides(portal: EvalPortal, args, argv: list[str]) -> None:
    """Apply explicit CLI overrides onto loaded config using a declarative map."""
    argv_set = set(argv)

    override_specs = [
        ("--engine", ("engine",), lambda value: value.lower()),
        ("--model", ("model", "name"), None),
        ("--api-key", ("model", "api_key"), None),
        ("--base-url", ("model", "base_url"), None),
        ("--n-rollouts", ("execution", "n_rollouts_per_prompt"), None),
        ("--n-parallel", ("execution", "n_parallel_agents"), None),
        ("--max-steps", ("execution", "max_steps"), None),
        ("--output-dir", ("output", "save_dir"), None),
        ("--split", ("data", "split"), None),
        ("--limit", ("data", "limit"), None),
        ("--world-model-file", ("agent_args", "world_model_file"), None),
        ("--resume", ("execution", "resume"), None),
        ("--max-key-cycles", ("model", "max_key_cycles"), None),
    ]

    arg_name_map = {
        "--engine": "engine",
        "--model": "model",
        "--api-key": "api_key",
        "--base-url": "base_url",
        "--n-rollouts": "n_rollouts",
        "--n-parallel": "n_parallel",
        "--max-steps": "max_steps",
        "--output-dir": "output_dir",
        "--split": "split",
        "--limit": "limit",
        "--world-model-file": "world_model_file",
        "--resume": "resume",
        "--max-key-cycles": "max_key_cycles",
    }

    for flag, path_keys, transform in override_specs:
        if flag not in argv_set:
            continue
        value = getattr(args, arg_name_map[flag])
        if transform is not None:
            value = transform(value)

        target = portal.config
        for key in path_keys[:-1]:
            if key not in target or not isinstance(target[key], dict):
                target[key] = {}
            target = target[key]
        target[path_keys[-1]] = value

    # Keep metrics model_name aligned with runtime model when model is overridden.
    if "--model" in argv_set:
        portal.config.setdefault("metrics", {})
        portal.config["metrics"].setdefault("offline_logprobs", {})
        portal.config["metrics"]["offline_logprobs"]["model_name"] = args.model

    # Boolean feature toggles intentionally only enable, never force-disable YAML.
    if "--logprob-metrics" in argv_set and args.logprob_metrics:
        portal.config.setdefault("metrics", {})
        portal.config["metrics"]["include_logprob_metrics"] = True
        print("Logprob metrics enabled")
    if "--detailed-metrics" in argv_set and args.detailed_metrics:
        portal.config.setdefault("metrics", {})
        portal.config["metrics"]["save_detailed_metrics"] = True
        print("Detailed token/step level metrics enabled")

    # Generic dotted overrides (`--set key=value`) are applied last.
    for raw_item in args.sets:
        if "=" not in raw_item:
            raise ValueError(
                f"Invalid --set '{raw_item}'. Expected KEY=VALUE (e.g. execution.max_steps=80)."
            )
        key, raw_value = raw_item.split("=", 1)
        dotted_key = key.strip()
        if not dotted_key:
            raise ValueError(f"Invalid --set '{raw_item}': empty KEY.")
        # API key/secret fields must stay strings. A long all-digit key (e.g.
        # a closed-source API key like <YOUR_API_KEY>) would otherwise be coerced to
        # int by _parse_typed_override, and downstream str-only handling in
        # portal._get_closed_config would silently drop it and fall back to the
        # YAML single key. Preserve the raw string for these fields.
        if dotted_key.split(".")[-1] in {"api_key", "api_keys"}:
            typed_value = raw_value.strip()
        else:
            typed_value = _parse_typed_override(raw_value.strip())
        _set_nested_config_value(portal.config, dotted_key, typed_value)


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Handle --list-envs
    if args.list_envs:
        list_environments()
        return 0

    # Validate arguments
    if args.config is None and args.env is None:
        parser.error("Either --env or --config must be specified")

    portal = None
    try:
        # Create portal
        if args.config:
            print(f"Loading configuration from {args.config}")
            portal = EvalPortal(args.config)
        else:
            print(f"Starting evaluation for {args.env}...")
            portal = EvalPortal.from_args(args)

        # Apply CLI overrides only when flags are explicitly provided.
        # Never auto-switch engine due to --base-url.
        try:
            _apply_explicit_cli_overrides(portal, args, sys.argv)
        except ValueError as e:
            parser.error(str(e))

        # Initialize Ray for environment orchestration unless using Verl engine.
        # Verl engine owns Ray initialization and may require a custom runtime_env.
        engine_name = (portal.config.get("engine") or "openai").lower()
        if engine_name != "verl":
            _init_ray(args.n_parallel)
        else:
            print("Skipping CLI Ray pre-initialization for Verl engine (portal will initialize Ray).")

        # Print effective config before starting
        portal.print_config(cli_args=sys.argv)

        # Setup engine and initialize in main thread before async operations
        # This is required for engines like sglang that need main thread
        # initialization for signal handlers
        portal.setup_engine()
        portal.engine.initialize()

        # Run evaluation
        results = asyncio.run(portal.run())

        if not results:
            print("No results generated")
            return 1

        # Compute and display metrics
        portal.print_summary()

        # Save results
        if not args.no_save:
            save_path = portal.save_results()
            if save_path:
                print(f"Results saved to {save_path}")

        return 0

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        return 130
    except Exception as e:
        logging.exception(f"Evaluation failed: {e}")
        return 1
    finally:
        # Ensure cleanup happens even on error
        if portal is not None:
            portal.shutdown()
        _shutdown_ray()


if __name__ == "__main__":
    sys.exit(main())
