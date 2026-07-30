"""Common utilities for interactive agent runners.

This module provides shared utility functions used across all interactive
runner scripts to reduce code duplication.
"""

import re
from typing import Any

import httpx
from openai import OpenAI


# Special tokens used by some models that need to be cleaned from responses
SPECIAL_TOKEN_PATTERNS = [
    r"<\|channel\|>",
    r"<\|message\|>",
    r"<\|end\|>",
    r"<\|im_start\|>",
    r"<\|im_end\|>",
    r"<\|start\|>",
    r"<\|endoftext\|>",
]

# Pattern to match harmony format multi-turn artifacts
# Matches: <|end|><|start|>assistant<|channel|>...<|message|>...(content until next <|end|> or end of string)
HARMONY_MULTI_TURN_PATTERN = re.compile(
    r"<\|end\|><\|start\|>assistant.*?(?=<\|end\|>|$)",
    re.DOTALL
)


def clean_special_tokens(text: str) -> str:
    """Remove special tokens from model response.

    Args:
        text: The model response text to clean.

    Returns:
        The cleaned text with special tokens removed.
    """
    if not text:
        return text

    cleaned = text

    # First, remove harmony format multi-turn artifacts (model generating extra turns)
    # This removes everything after the first <|end|> that starts a new turn
    cleaned = HARMONY_MULTI_TURN_PATTERN.sub("", cleaned)

    # Then remove individual special tokens
    for pattern in SPECIAL_TOKEN_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned)

    return cleaned.strip()


def create_llm_client(
    base_url: str = "http://localhost:8000/v1",
    api_key: str = "EMPTY",
) -> OpenAI:
    """Create OpenAI client for local LLM server.

    Args:
        base_url: The base URL of the LLM server.
        api_key: The API key for authentication.

    Returns:
        An OpenAI client configured for the local server.
    """
    http_client = httpx.Client(trust_env=False)
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        http_client=http_client,
    )
    return client


def get_model_response(
    client: OpenAI,
    model_id: str,
    messages: list[dict],
    stream: bool = True,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> str:
    """Get response from LLM model with optional streaming output.

    Args:
        client: OpenAI client for LLM inference.
        model_id: Model ID to use for inference.
        messages: List of message dictionaries for the conversation.
        stream: Whether to stream model output.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.

    Returns:
        The model's response text.
    """
    response_text = ""
    reasoning_text = ""

    if stream:
        stream_response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream_response:
            if chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                reasoning = getattr(delta, "reasoning_content", None)

                if content:
                    print(content, end="", flush=True)
                    response_text += content
                elif reasoning:
                    # Some models (e.g. Qwen3 with thinking enabled) stream
                    # text via reasoning_content and leave content empty.
                    print(reasoning, end="", flush=True)
                    reasoning_text += reasoning
        print()  # newline after streaming
    else:
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        message = response.choices[0].message
        response_text = message.content or ""
        reasoning_text = getattr(message, "reasoning_content", None) or ""
        if response_text:
            print(response_text)
        elif reasoning_text:
            print(reasoning_text)

    return response_text if response_text.strip() else reasoning_text


def get_model_response_with_tools(
    client: OpenAI,
    model_id: str,
    messages: list[dict],
    tools: list[dict],
    stream: bool = True,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> tuple[str, list[dict] | None]:
    """Get response from LLM model with tool calling support.

    Args:
        client: OpenAI client for LLM inference.
        model_id: Model ID to use for inference.
        messages: List of message dictionaries for the conversation.
        tools: List of tool schemas in OpenAI format.
        stream: Whether to stream model output.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.

    Returns:
        A tuple of (response_text, tool_calls) where tool_calls is None
        if no tools were called.
    """
    response_text = ""
    reasoning_text = ""
    tool_calls = None

    if stream:
        stream_response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            tools=tools if tools else None,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        collected_tool_calls: dict[int, dict] = {}
        for chunk in stream_response:
            if chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta
                if delta.content:
                    print(delta.content, end="", flush=True)
                    response_text += delta.content
                elif getattr(delta, "reasoning_content", None):
                    reasoning_piece = delta.reasoning_content
                    print(reasoning_piece, end="", flush=True)
                    reasoning_text += reasoning_piece
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in collected_tool_calls:
                            collected_tool_calls[idx] = {
                                "id": tc.id or "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        if tc.id:
                            collected_tool_calls[idx]["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                collected_tool_calls[idx]["function"]["name"] = tc.function.name
                            if tc.function.arguments:
                                collected_tool_calls[idx]["function"]["arguments"] += tc.function.arguments
        print()  # newline after streaming
        if collected_tool_calls:
            tool_calls = [collected_tool_calls[i] for i in sorted(collected_tool_calls.keys())]
    else:
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            tools=tools if tools else None,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = response.choices[0]
        if choice.message.content:
            response_text = choice.message.content
            print(response_text)
        elif getattr(choice.message, "reasoning_content", None):
            reasoning_text = choice.message.reasoning_content
            print(reasoning_text)
        if choice.message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in choice.message.tool_calls
            ]

    final_text = response_text if response_text.strip() else reasoning_text
    return final_text, tool_calls


def print_separator(char: str = "=", length: int = 70) -> None:
    """Print a separator line.

    Args:
        char: The character to use for the separator.
        length: The length of the separator line.
    """
    print(char * length)


def print_header(title: str, length: int = 70) -> None:
    """Print a section header.

    Args:
        title: The title to display in the header.
        length: The length of the separator lines.
    """
    print_separator("=", length)
    print(f"  {title}")
    print_separator("=", length)


def truncate_text(
    text: str,
    max_lines: int = 50,
    max_chars_per_line: int = 200,
) -> str:
    """Truncate text to a reasonable size for display.

    Args:
        text: The text to truncate.
        max_lines: Maximum number of lines to keep.
        max_chars_per_line: Maximum characters per line.

    Returns:
        The truncated text.
    """
    lines = text.split("\n")
    truncated_lines = []
    for line in lines[:max_lines]:
        if len(line) > max_chars_per_line:
            truncated_lines.append(line[:max_chars_per_line] + "...")
        else:
            truncated_lines.append(line)
    if len(lines) > max_lines:
        truncated_lines.append(f"... ({len(lines) - max_lines} more lines)")
    return "\n".join(truncated_lines)


def connect_to_llm_server(
    server_url: str,
    api_key: str = "EMPTY",
) -> tuple[OpenAI, str]:
    """Connect to LLM server and return client and model ID.

    Args:
        server_url: The URL of the LLM server.
        api_key: The API key for authentication.

    Returns:
        A tuple of (client, model_id).

    Raises:
        Exception: If connection fails.
    """
    print("Connecting to LLM server...")
    client = create_llm_client(base_url=server_url, api_key=api_key)
    models = client.models.list()
    model_id = models.data[0].id
    print(f"Connected! Using model: {model_id}")
    return client, model_id
