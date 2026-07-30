import asyncio
import base64
import logging
import os
from io import BytesIO

import openai
import httpx
from urllib.parse import urlparse
from PIL import Image

from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine
from rllm.globals import THOUGHT_DELIMITER_END, THOUGHT_DELIMITER_START
from rllm.parser import ChatTemplateParser
from rllm.tools.tool_base import Tool
from rllm.workflows import TerminationEvent, TerminationReason
from rllm.utils import clean_special_tokens


class OpenAIEngine(RolloutEngine):
    def __init__(self, model: str = "", tokenizer=None, max_prompt_length: int = 4096, max_response_length: int = 4096, max_model_length: int | None = None, api_retries: int = 3, base_url: str = "https://api.openai.com/v1", api_key: str = os.getenv("OPENAI_API_KEY"), api_keys: list[str] | None = None, max_key_cycles: int | None = None, key_cycle_backoff_s: float | None = None, empty_completion_retries: int | None = None, empty_completion_backoff_s: float | None = None, sampling_params: dict | None = None, tools: list[Tool | dict] = None, accumulate_reasoning: bool = False, force_chat_completions: bool = False, **kwargs):
        self.model = model
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.max_model_length = max_model_length - 1 if max_model_length is not None else max_prompt_length + max_response_length - 1
        self.api_retries = api_retries
        self.sampling_params = sampling_params or {}
        self.tools = tools or []
        self.accumulate_reasoning = accumulate_reasoning
        self.reasoning_effort = self.sampling_params.pop("reasoning_effort", "medium")

        self.tokenizer = tokenizer
        # Always use chat completions endpoint - the server's reasoning parser handles
        # special token parsing much better than client-side parsing via completions endpoint
        self._use_chat_completions = True

        # --- API key pool for per-request failover ---
        # Backward compatible: if api_keys not given, fall back to the single api_key.
        # When the pool has >1 key, a RateLimit/auth failure rotates to the next key
        # instead of failing the whole request (see chat_completion/completion).
        normalized_keys = list(api_keys) if api_keys else []
        normalized_keys = [k for k in normalized_keys if k]
        if not normalized_keys and api_key:
            normalized_keys = [api_key]
        self._api_keys = normalized_keys
        self._key_idx = 0
        # max_key_cycles / cycle-backoff are env-overridable so 429 (per-minute)
        # rate limits can be ridden out without editing code. Explicit constructor
        # args (not None) win over env; env wins over the built-in defaults
        # (3 cycles, 20s backoff). 20s+ backoff is chosen to straddle the
        # per-minute quota window — a 5s backoff barely dents a minute-scoped 429.
        if max_key_cycles is None:
            max_key_cycles = int(os.getenv("RLLM_MAX_KEY_CYCLES", "3"))
        self._max_key_cycles = max(1, int(max_key_cycles))
        if key_cycle_backoff_s is None:
            key_cycle_backoff_s = float(os.getenv("RLLM_KEY_CYCLE_BACKOFF_S", "20"))
        self._key_cycle_backoff_s = max(0.0, float(key_cycle_backoff_s))
        self._base_url = base_url

        # Retry well-formed responses that contain no generated content.
        if empty_completion_retries is None:
            empty_completion_retries = int(os.getenv("RLLM_EMPTY_COMPLETION_RETRIES", "8"))
        self._empty_completion_retries = max(0, int(empty_completion_retries))
        if empty_completion_backoff_s is None:
            empty_completion_backoff_s = float(os.getenv("RLLM_EMPTY_COMPLETION_BACKOFF_S", "2"))
        self._empty_completion_backoff_s = max(0.0, float(empty_completion_backoff_s))

        # Whether to disable httpx env-proxy for local endpoints (preserves old behavior).
        parsed_base = urlparse(base_url)
        self._is_local_base = parsed_base.hostname in {"localhost", "127.0.0.1", "0.0.0.0"}

        # Build the initial client with the first key (equivalent to old single-key path).
        initial_key = self._api_keys[0] if self._api_keys else api_key
        self.client = self._build_client(initial_key)
        logging.getLogger("httpx").setLevel(logging.WARNING)

    def _build_client(self, key: str) -> "openai.AsyncOpenAI":
        """Construct an AsyncOpenAI client for a given api key (reused on key rotation)."""
        http_client = httpx.AsyncClient(trust_env=False) if self._is_local_base else None
        return openai.AsyncOpenAI(
            base_url=self._base_url,
            api_key=key,
            http_client=http_client,
        )

    def _rotate_key(self) -> bool:
        """Rotate to the next api key and rebuild the client.

        Returns True if rotation wrapped back to index 0 (i.e. a full cycle over
        all keys has completed). With 0 or 1 key, rotation is a no-op returning True
        so callers treat the single-key case as 'one cycle done'.
        """
        if len(self._api_keys) <= 1:
            return True
        self._key_idx = (self._key_idx + 1) % len(self._api_keys)
        self.client = self._build_client(self._api_keys[self._key_idx])
        return self._key_idx == 0

    @staticmethod
    def _pil_to_base64(image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def _convert_messages_to_openai_format(self, messages: list[dict]) -> list[dict]:
        """Convert messages from rllm format to OpenAI multimodal format."""
        converted_messages = []
        for message in messages:
            if "images" in message and message["images"]:
                content_text = clean_special_tokens(message["content"])
                content = [{"type": "text", "text": content_text}]
                for img in message["images"]:
                    base64_image = self._pil_to_base64(img)
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}})

                converted_messages.append({"role": message["role"], "content": content})
            else:
                cleaned = message.copy()
                if "content" in cleaned and isinstance(cleaned["content"], str):
                    cleaned["content"] = clean_special_tokens(cleaned["content"])
                converted_messages.append(cleaned)

        return converted_messages

    def _prepare_max_tokens_param(self, sampling_params: dict, prompt_length: int = None) -> dict:
        """Prepare max tokens parameter for API call (supports O3's max_completion_tokens)."""
        if "max_completion_tokens" in sampling_params:
            return {"max_completion_tokens": sampling_params.pop("max_completion_tokens")}

        max_tokens = sampling_params.pop("max_tokens", sampling_params.pop("max_new_tokens", self.max_response_length))

        # Adjust for prompt length if provided (completion method needs this)
        if prompt_length and self.max_model_length:
            remaining = self.max_model_length - prompt_length
            if remaining <= max_tokens:
                max_tokens = remaining
                print(f"Warning: Decreasing max_tokens to {max_tokens} to stay within max_model_length")

        return {"max_tokens": max_tokens}

    @staticmethod
    def _sanitize_openai_sampling_params(sampling_params: dict) -> dict:
        """Drop params that OpenAI chat.completions does not accept.

        Some eval configs are shared with local engines (sglang/verl) and may
        include generation knobs like top_k/do_sample. The OpenAI Python client
        validates kwargs strictly and raises TypeError for unknown fields.
        """
        unsupported_keys = [
            "top_k",
            "do_sample",
            "entropy_mode",
            "return_logprobs",
            "top_k_for_entropy",
        ]
        for key in unsupported_keys:
            sampling_params.pop(key, None)
        return sampling_params

    @staticmethod
    def _is_empty_completion(output: "ModelOutput") -> bool:
        """Return whether an API response contains no generated content.

        Require both zero completion length and blank content so a short,
        non-empty answer is not retried. A length-truncated response is retained.
        """
        if output.finish_reason == "length":
            return False
        try:
            clen = int(output.completion_length)
        except (TypeError, ValueError):
            clen = 0
        if clen > 0:
            return False
        content = output.content or ""
        return content.strip() == ""

    async def chat_completion(self, messages: list[dict], **kwargs) -> ModelOutput:
        """Chat completion with empty-completion retry on top of key failover.

        Delegates each attempt to _chat_completion_once (which owns the per-key
        failover / transient-error retry). If an attempt returns an empty API
        turn (see _is_empty_completion) and retries are enabled, re-issue the same
        request up to empty_completion_retries times with a backoff. After the
        budget is exhausted the last (still-empty) output is returned rather than
        raised, so a deterministically-empty response can never hang the run.
        Non-empty outputs and exceptions are propagated unchanged on the first
        attempt, preserving prior behavior when the retry budget is 0.
        """
        output = await self._chat_completion_once(messages, **kwargs)
        if self._empty_completion_retries <= 0 or not self._is_empty_completion(output):
            return output

        for attempt in range(1, self._empty_completion_retries + 1):
            logging.warning(
                "Empty completion (finish_reason=%s, completion_length=%s); "
                "retrying single turn %d/%d after %.1fs backoff.",
                output.finish_reason,
                output.completion_length,
                attempt,
                self._empty_completion_retries,
                self._empty_completion_backoff_s,
            )
            if self._empty_completion_backoff_s > 0:
                await asyncio.sleep(self._empty_completion_backoff_s)
            output = await self._chat_completion_once(messages, **kwargs)
            if not self._is_empty_completion(output):
                return output

        logging.warning(
            "Empty completion persisted after %d retries; returning empty output.",
            self._empty_completion_retries,
        )
        return output

    async def _chat_completion_once(self, messages: list[dict], **kwargs) -> ModelOutput:
        kwargs.pop("application_id", None)
        kwargs.pop("validate", None)
        kwargs.pop("model", None)
        kwargs.pop("enforce_max_prompt_length", None)

        sampling_params = self.sampling_params.copy()
        sampling_params.update(kwargs)
        request_logprobs = sampling_params.pop("return_logprobs", False)
        sampling_params = self._sanitize_openai_sampling_params(sampling_params)

        create_params = self._prepare_max_tokens_param(sampling_params)
        converted_messages = self._convert_messages_to_openai_format(messages)

        # Per-request key failover: outer loop over cycles, inner over keys. A
        # RateLimit/auth/permission error rotates to the next key; a transient
        # APIError/network error is retried in place up to api_retries before
        # also rotating. After all keys fail in a cycle, back off and retry the
        # whole pool up to max_key_cycles times. TerminationEvent (business
        # signal) is never swallowed.
        num_keys = max(1, len(self._api_keys))
        last_error: Exception | None = None
        for cycle in range(self._max_key_cycles):
            for _ in range(num_keys):
                local_retries = self.api_retries
                rotate = False
                while local_retries > 0:
                    try:
                        response = await self.client.chat.completions.create(model=self.model, messages=converted_messages, timeout=3600, **create_params, **sampling_params)

                        # Normalize a missing choice/message to an empty completion
                        # so chat_completion() can apply its bounded retry policy.
                        choices = getattr(response, "choices", None) or []
                        message = choices[0].message if choices else None
                        if message is None:
                            logging.warning(
                                "Malformed API response (no choices or null message); "
                                "treating as an empty completion. usage=%s",
                                getattr(response, "usage", None),
                            )
                            return ModelOutput(
                                text="",
                                content="",
                                reasoning="",
                                tool_calls=[],
                                prompt_ids=[],
                                completion_ids=[],
                                logprobs=[],
                                prompt_logprobs=[],
                                prompt_length=getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0,
                                completion_length=0,
                                finish_reason="stop",
                            )

                        content = message.content or ""
                        reasoning = message.reasoning_content if hasattr(message, "reasoning_content") and isinstance(message.reasoning_content, str) else ""
                        tool_calls = message.tool_calls if hasattr(message, "tool_calls") and isinstance(message.tool_calls, list) else []

                        # Clean special tokens from content (server-side parser may not fully clean them)
                        # This handles cases where the model generates multiple "turns" in a single response
                        content = clean_special_tokens(content)

                        # Build text with reasoning if available, otherwise use content
                        if reasoning:
                            text = f"{THOUGHT_DELIMITER_START}\n{reasoning}\n{THOUGHT_DELIMITER_END}\n\n{content}"
                        else:
                            text = content

                        prompt_length = response.usage.prompt_tokens
                        completion_length = response.usage.completion_tokens
                        finish_reason = response.choices[0].finish_reason

                        return ModelOutput(
                            text=text,
                            content=content,
                            reasoning=reasoning,
                            tool_calls=tool_calls,
                            prompt_ids=[],
                            completion_ids=[],
                            logprobs=[],
                            prompt_logprobs=[],
                            prompt_length=prompt_length,
                            completion_length=completion_length,
                            finish_reason=finish_reason,
                        )

                    except TerminationEvent:
                        # Business signal — never swallow, never rotate keys.
                        raise

                    except (openai.RateLimitError, openai.AuthenticationError, openai.PermissionDeniedError) as e:
                        # Key-level failure: stop in-place retry, rotate to next key.
                        last_error = e
                        rotate = True
                        break

                    except Exception as e:
                        # Transient (5xx/network): retry in place; rotate when exhausted.
                        last_error = e
                        local_retries -= 1
                        if local_retries == 0:
                            rotate = True
                            break
                        print(f"Error: {e}, retrying...")
                        await asyncio.sleep(1)

                if rotate:
                    self._rotate_key()

            # All keys failed this cycle; back off before the next cycle.
            if cycle < self._max_key_cycles - 1:
                print(f"All {num_keys} key(s) failed (cycle {cycle + 1}/{self._max_key_cycles}); sleeping {self._key_cycle_backoff_s}s before next cycle.")
                await asyncio.sleep(self._key_cycle_backoff_s)

        raise Exception(f"All API keys exhausted across {self._max_key_cycles} cycle(s). Last error: {last_error}") from last_error

    async def completion(self, prompt: str | list[int], **kwargs) -> ModelOutput:
        kwargs.pop("application_id", None)
        kwargs.pop("validate", None)
        kwargs.pop("model", None)
        enforce_max_prompt_length = kwargs.pop("enforce_max_prompt_length", True)

        sampling_params = self.sampling_params.copy()
        sampling_params.update(kwargs)
        request_logprobs = sampling_params.pop("return_logprobs", False)
        sampling_params = self._sanitize_openai_sampling_params(sampling_params)

        if isinstance(prompt, list):
            prompt_ids = prompt
        else:
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        prompt_length = len(prompt_ids)
        if enforce_max_prompt_length and (prompt_length > self.max_prompt_length or prompt_length > self.max_model_length):
            raise TerminationEvent(TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED)

        create_params = self._prepare_max_tokens_param(sampling_params, prompt_length)
        sampling_params.update(create_params)

        # Per-request key failover, mirroring chat_completion().
        num_keys = max(1, len(self._api_keys))
        last_error: Exception | None = None
        for cycle in range(self._max_key_cycles):
            for _ in range(num_keys):
                local_retries = self.api_retries
                rotate = False
                while local_retries > 0:
                    try:
                        create_kwargs = {
                            "model": self.model,
                            "prompt": prompt,
                            **sampling_params,
                        }
                        if request_logprobs:
                            create_kwargs["logprobs"] = 1
                        response = await self.client.completions.create(**create_kwargs)
                        text = response.choices[0].text
                        try:
                            completion_ids = response.choices[0].token_ids
                            assert completion_ids is not None
                        except Exception:
                            completion_ids = self.tokenizer.encode(text, add_special_tokens=False)

                        parsed_output = self.chat_parser.parse_completion(completion_ids)

                        prompt_length = response.usage.prompt_tokens
                        completion_length = response.usage.completion_tokens
                        finish_reason = response.choices[0].finish_reason

                        logprobs = []
                        prompt_logprobs = []
                        if request_logprobs:
                            try:
                                assert response.choices[0].logprobs is not None
                                logprobs = response.choices[0].logprobs.token_logprobs
                            except Exception:
                                logprobs = []

                            try:
                                assert response.choices[0].prompt_logprobs is not None
                                prompt_logprobs = [None]
                                for tid, lp in zip(prompt_ids[1:], response.choices[0].prompt_logprobs[1:], strict=False):
                                    prompt_logprobs.append(float(lp[str(tid)]["logprob"]))
                            except Exception:
                                prompt_logprobs = []

                        return ModelOutput(
                            text=text,
                            content=parsed_output["content"],
                            reasoning=parsed_output["reasoning"],
                            tool_calls=parsed_output["tool_calls"],
                            prompt_ids=prompt_ids,
                            completion_ids=completion_ids,
                            logprobs=logprobs,
                            prompt_logprobs=prompt_logprobs,
                            prompt_length=prompt_length,
                            completion_length=completion_length,
                            finish_reason=finish_reason,
                        )

                    except TerminationEvent:
                        raise

                    except (openai.RateLimitError, openai.AuthenticationError, openai.PermissionDeniedError) as e:
                        last_error = e
                        rotate = True
                        break

                    except Exception as e:
                        last_error = e
                        local_retries -= 1
                        if local_retries == 0:
                            rotate = True
                            break
                        print(f"Error: {e}, retrying...")
                        await asyncio.sleep(1)

                if rotate:
                    self._rotate_key()

            if cycle < self._max_key_cycles - 1:
                print(f"All {num_keys} key(s) failed (cycle {cycle + 1}/{self._max_key_cycles}); sleeping 5s before next cycle.")
                await asyncio.sleep(5)

        raise Exception(f"All API keys exhausted across {self._max_key_cycles} cycle(s). Last error: {last_error}") from last_error
    

    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        accumulate_reasoning = kwargs.pop("accumulate_reasoning", self.accumulate_reasoning)
        if accumulate_reasoning:
            raise ValueError("Accumulate reasoning is not supported for chat completions endpoint.")
        return await self.chat_completion(messages, **kwargs)
