import asyncio
from types import SimpleNamespace

from rllm.engine.rollout.openai_engine import OpenAIEngine


def test_completion_extracts_internal_logprob_flag_before_api_call():
    captured = {}

    async def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    text="ok",
                    token_ids=[2],
                    finish_reason="stop",
                    logprobs=SimpleNamespace(token_logprobs=[-0.25]),
                    prompt_logprobs=None,
                )
            ],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        )

    engine = object.__new__(OpenAIEngine)
    engine.model = "test-model"
    engine.sampling_params = {
        "return_logprobs": True,
        "top_k": 20,
        "max_tokens": 4,
    }
    engine.max_prompt_length = 8
    engine.max_response_length = 4
    engine.max_model_length = 16
    engine.api_retries = 1
    engine._max_key_cycles = 1
    engine._api_keys = []
    engine.client = SimpleNamespace(
        completions=SimpleNamespace(create=create),
    )
    engine.chat_parser = SimpleNamespace(
        parse_completion=lambda _ids: {
            "content": "ok",
            "reasoning": None,
            "tool_calls": None,
        }
    )

    output = asyncio.run(engine.completion([1]))

    assert captured["logprobs"] == 1
    assert "return_logprobs" not in captured
    assert "top_k" not in captured
    assert output.logprobs == [-0.25]

