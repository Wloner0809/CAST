"""Standalone sglang engine for direct model inference without a separate server.

This engine loads the model directly using sglang's Engine class, allowing
evaluation without needing to start a separate LLM server process.
"""

import logging
import math
import os
from typing import Optional

import numpy as np

from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine
from rllm.parser import ChatTemplateParser
from rllm.workflows import TerminationEvent, TerminationReason

logger = logging.getLogger(__name__)


def _sanitize_worker_runtime_env() -> None:
    """Sanitize PATH/TVM env vars for non-root worker processes."""
    current_path = os.environ.get("PATH", "")
    if current_path:
        path_parts = current_path.split(":")
        filtered_path = [p for p in path_parts if not p.startswith("/root")]
        os.environ["PATH"] = ":".join(filtered_path)

    os.environ.setdefault("TVM_HOME", "")
    os.environ.setdefault("TVM_LIBRARY_PATH", "")


def _compute_entropy_from_logprobs_numpy(logprobs_list: list) -> float:
    """Compute entropy from a list of (logprob, token_id, ...) tuples using numpy.
    
    This is much faster than Python loops for large vocabularies.
    Entropy = -sum(p * log(p)) = -sum(exp(logp) * logp)
    """
    if not logprobs_list:
        return 0.0
    
    # Extract logprobs, filtering None values
    lps = []
    for item in logprobs_list:
        if isinstance(item, (list, tuple)) and len(item) >= 1 and item[0] is not None:
            lps.append(float(item[0]))
        elif isinstance(item, (int, float)) and item is not None:
            lps.append(float(item))
    
    if not lps:
        return 0.0
    
    # Use numpy for vectorized computation
    lps_arr = np.array(lps, dtype=np.float64)
    # Clip to avoid numerical issues with very small probabilities
    lps_arr = np.clip(lps_arr, -100.0, 0.0)
    probs = np.exp(lps_arr)
    # Entropy = -sum(p * log(p))
    entropy = -np.sum(probs * lps_arr)
    return float(entropy)

_sanitize_worker_runtime_env()


class SglangEngine(RolloutEngine):
    """Standalone sglang engine that loads the model directly.

    This engine uses sglang's Engine class to load and run the model
    in-process, without requiring a separate server.

    Example:
        >>> engine = SglangEngine(
        ...     model_path="/path/to/model",
        ...     tokenizer=tokenizer,
        ...     tp_size=1,
        ... )
        >>> output = await engine.get_model_response(messages)
    """

    def __init__(
        self,
        model_path: str,
        tokenizer=None,
        max_prompt_length: int = 4096,
        max_response_length: int = 4096,
        tp_size: int = 1,
        dp_size: int = 1,
        mem_fraction_static: float = 0.85,
        sampling_params: dict | None = None,
        trust_remote_code: bool = True,
        **kwargs,
    ):
        """Initialize the sglang engine.

        Args:
            model_path: Path to the model weights.
            tokenizer: Tokenizer instance (if None, will be loaded from model_path).
            max_prompt_length: Maximum prompt length in tokens.
            max_response_length: Maximum response length in tokens.
            tp_size: Tensor parallelism size.
            dp_size: Data parallelism size.
            mem_fraction_static: GPU memory fraction for static allocation.
            sampling_params: Default sampling parameters.
            trust_remote_code: Whether to trust remote code when loading model.
            **kwargs: Additional arguments passed to sglang.Engine.
        """
        self.model_path = model_path
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.max_model_length = max_prompt_length + max_response_length
        self.sampling_params = sampling_params or {}
        self.tp_size = tp_size
        self.dp_size = dp_size
        self.mem_fraction_static = mem_fraction_static
        self.trust_remote_code = trust_remote_code
        self.extra_kwargs = kwargs

        # Initialize tokenizer
        if tokenizer is None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=trust_remote_code
            )
        else:
            self.tokenizer = tokenizer

        # Initialize chat parser
        self.chat_parser = ChatTemplateParser.get_parser(
            self.tokenizer,
            disable_thinking=kwargs.get("disable_thinking", False)
        )

        # Lazy initialization of sglang engine
        self._engine = None
        self._initialized = False

        logger.info(
            f"SglangEngine configured with model_path={model_path}, "
            f"tp_size={tp_size}, dp_size={dp_size}"
        )

    def initialize(self):
        """Initialize the sglang engine.

        This method MUST be called from the main thread before any async
        operations, as sglang's Engine sets up signal handlers which can
        only be done from the main thread.
        """
        self._lazy_init()

    def _lazy_init(self):
        """Lazily initialize the sglang engine."""
        if self._initialized:
            return

        try:
            import sglang as sgl

            logger.info(f"Loading model from {self.model_path}...")

            # Build engine kwargs
            engine_kwargs = {
                "model_path": self.model_path,
                "tp_size": self.tp_size,
                "dp_size": self.dp_size,
                "mem_fraction_static": self.mem_fraction_static,
                "trust_remote_code": self.trust_remote_code,
            }

            # Add any extra kwargs
            for key in ["dtype", "quantization", "context_length", "device", "enable_deterministic_inference"]:
                if key in self.extra_kwargs:
                    engine_kwargs[key] = self.extra_kwargs[key]

            self._engine = sgl.Engine(**engine_kwargs)
            self._initialized = True
            logger.info("sglang Engine initialized successfully")

        except ImportError as e:
            raise ImportError(
                "sglang is required for SglangEngine. "
                "Install it with: pip install sglang"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to initialize sglang engine: {e}") from e

    async def completion(self, prompt: str | list[int], **kwargs) -> ModelOutput:
        """Generate completion for a prompt.

        Args:
            prompt: Input prompt string or token IDs.
            **kwargs: Additional sampling parameters.

        Returns:
            ModelOutput with generated text and metadata.
        """
        kwargs.pop("application_id", None)
        kwargs.pop("validate", None)
        kwargs.pop("model", None)
        enforce_max_prompt_length = kwargs.pop("enforce_max_prompt_length", True)

        sampling_params = self.sampling_params.copy()
        sampling_params.update(kwargs)

        # Convert token IDs to string if needed
        if isinstance(prompt, list):
            prompt_ids = prompt
            prompt_str = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
        else:
            prompt_str = prompt
            prompt_ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)

        prompt_length = len(prompt_ids)
        if enforce_max_prompt_length and prompt_length > self.max_prompt_length:
            raise TerminationEvent(TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED)

        # Adjust max_tokens based on remaining context
        max_tokens = sampling_params.get("max_tokens", self.max_response_length)
        remaining = self.max_model_length - prompt_length
        if remaining < max_tokens:
            max_tokens = max(remaining, 1)
            sampling_params["max_tokens"] = max_tokens
            logger.warning(f"Reducing max_tokens to {max_tokens} to fit context")

        # Build sglang sampling params
        sgl_params = {
            "max_new_tokens": sampling_params.get("max_tokens", self.max_response_length),
            "temperature": sampling_params.get("temperature", 0.7),
            "top_p": sampling_params.get("top_p", 0.95),
            "top_k": sampling_params.get("top_k", -1),
        }

        # Remove None values
        sgl_params = {k: v for k, v in sgl_params.items() if v is not None}

        request_logprobs = sampling_params.pop("return_logprobs", False)

        # Entropy computation mode:
        # - "none": No entropy computation (fastest)
        # - "gpu": Use sglang's built-in GPU entropy computation (fast & exact, RECOMMENDED)
        # - "topk": Use top-K logprobs to approximate entropy on CPU (fallback)
        # - "full": Use full vocabulary logprobs on CPU (very slow, exact)
        entropy_mode = sampling_params.pop("entropy_mode", "none")
        top_k_for_entropy = sampling_params.pop("top_k_for_entropy", 100)
        if not request_logprobs:
            entropy_mode = "none"
        
        # Determine what to request from sglang
        token_ids_logprob = None
        top_logprobs_num = 0
        return_entropy = False
        
        if entropy_mode == "gpu" and request_logprobs:
            # Use sglang's built-in GPU entropy computation (fastest & exact)
            return_entropy = True
            logger.debug("Using sglang's GPU-based entropy computation")
        elif entropy_mode == "full":
            # Request logprobs for the full vocabulary to compute exact entropy on CPU
            # WARNING: this can be extremely expensive in memory/time
            vocab_size = getattr(self.tokenizer, "vocab_size", None)
            if vocab_size is None:
                vocab_size = len(self.tokenizer)
            token_ids_logprob = np.arange(int(vocab_size), dtype=np.int32).tolist()
            logger.debug(f"Requesting full-vocab logprobs for CPU entropy (vocab_size={vocab_size})")
        elif entropy_mode == "topk":
            # Use top-K logprobs to approximate entropy on CPU
            top_logprobs_num = top_k_for_entropy
            logger.debug(f"Using top-{top_k_for_entropy} logprobs for CPU entropy approximation")

        # Use GenerateReqInput directly to access return_entropy parameter
        # (async_generate doesn't expose it)
        from sglang.srt.managers.io_struct import GenerateReqInput
        
        req = GenerateReqInput(
            text=prompt_str,
            sampling_params=sgl_params,
            return_logprob=request_logprobs,
            return_entropy=return_entropy,
            token_ids_logprob=token_ids_logprob,
            top_logprobs_num=top_logprobs_num,
        )
        
        generator = self._engine.tokenizer_manager.generate_request(req, None)
        output = await generator.__anext__()

        # Extract text and logprobs from output
        logprobs = []
        token_entropies = []
        completion_ids = []
        if isinstance(output, dict):
            text = output.get("text", "")
            meta_info = output.get("meta_info", {})
            output_ids = output.get("output_ids") or []
            
            # Extract primary logprobs (logprob of selected token)
            # sglang returns output_token_logprobs as list of tuples: (logprob, token_id, token_text)
            if request_logprobs and meta_info and "output_token_logprobs" in meta_info:
                raw_logprobs = meta_info["output_token_logprobs"]
                try:
                    if isinstance(raw_logprobs, list) and len(raw_logprobs) > 0:
                        # Check if it's a list of tuples (logprob, token_id, token_text)
                        if isinstance(raw_logprobs[0], (list, tuple)) and len(raw_logprobs[0]) >= 2:
                            logprobs = [float(item[0]) if item[0] is not None else None for item in raw_logprobs]
                            completion_ids = [int(item[1]) for item in raw_logprobs]
                        else:
                            # Fallback: assume it's just a list of logprobs
                            logprobs = [float(lp) if lp is not None else None for lp in raw_logprobs]
                except Exception as e:
                    logger.warning(f"Failed to extract logprobs: {e}")
                    logprobs = []
            
            # Extract GPU-computed entropy (if requested)
            # sglang returns output_token_entropy directly computed on GPU
            if entropy_mode == "gpu" and meta_info and "output_token_entropy" in meta_info:
                raw_entropies = meta_info["output_token_entropy"]
                try:
                    if isinstance(raw_entropies, list):
                        token_entropies = [float(e) if e is not None else None for e in raw_entropies]
                except Exception as e:
                    logger.warning(f"Failed to extract GPU entropy: {e}")
                    token_entropies = []
            
            # Extract entropy from full-vocab logprobs (CPU computation)
            elif entropy_mode == "full" and meta_info and "output_token_ids_logprobs" in meta_info:
                raw_full_logprobs = meta_info["output_token_ids_logprobs"]
                try:
                    if isinstance(raw_full_logprobs, list):
                        for position_logprobs in raw_full_logprobs:
                            if not position_logprobs:
                                token_entropies.append(None)
                                continue
                            entropy = _compute_entropy_from_logprobs_numpy(position_logprobs)
                            token_entropies.append(entropy)
                except Exception as e:
                    logger.warning(f"Failed to extract full-vocab entropy: {e}")
                    token_entropies = []
            
            # Extract entropy from top-K logprobs (CPU approximation)
            elif entropy_mode == "topk" and meta_info and "output_top_logprobs" in meta_info:
                raw_top_logprobs = meta_info["output_top_logprobs"]
                try:
                    if isinstance(raw_top_logprobs, list):
                        for position_top_logprobs in raw_top_logprobs:
                            if not position_top_logprobs:
                                token_entropies.append(None)
                                continue
                            entropy = _compute_entropy_from_logprobs_numpy(position_top_logprobs)
                            token_entropies.append(entropy)
                except Exception as e:
                    logger.warning(f"Failed to extract top-k entropy: {e}")
                    token_entropies = []
            
            # If we didn't get completion_ids from logprobs, encode the text
            if not completion_ids:
                if output_ids:
                    completion_ids = output_ids
                else:
                    completion_ids = self.tokenizer.encode(text, add_special_tokens=False)
        else:
            # Handle different output formats
            text = str(output)
            completion_ids = self.tokenizer.encode(text, add_special_tokens=False)

        # Parse completion
        parsed_output = self.chat_parser.parse_completion(completion_ids)

        # Determine finish reason
        finish_reason = "stop"
        if len(completion_ids) >= max_tokens:
            finish_reason = "length"

        return ModelOutput(
            text=text,
            content=parsed_output["content"],
            reasoning=parsed_output["reasoning"],
            tool_calls=parsed_output["tool_calls"],
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            logprobs=logprobs,
            prompt_logprobs=[],
            token_entropies=token_entropies if token_entropies else None,
            prompt_length=prompt_length,
            completion_length=len(completion_ids),
            finish_reason=finish_reason,
        )

    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        """Generate response for chat messages.

        Args:
            messages: List of chat messages in OpenAI format.
            **kwargs: Additional sampling parameters.

        Returns:
            ModelOutput with generated response.
        """
        tools = kwargs.pop("tools", [])
        accumulate_reasoning = kwargs.pop("accumulate_reasoning", False)

        # Parse messages to prompt string
        prompt = self.chat_parser.parse(
            messages,
            add_generation_prompt=True,
            is_first_msg=True,
            tools=tools,
            accumulate_reasoning=accumulate_reasoning,
        )

        return await self.completion(prompt, **kwargs)

    def shutdown(self):
        """Shutdown the engine and release resources."""
        if self._engine is not None:
            try:
                self._engine.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down sglang engine: {e}")
            self._engine = None
        self._initialized = False

    def __del__(self):
        """Cleanup on deletion."""
        self.shutdown()
