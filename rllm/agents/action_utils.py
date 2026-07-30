"""Shared action parsing utilities for rllm agents.

All game/environment agents use ``` ... ``` fenced blocks to delimit
actions in the model's response.  The content inside the block IS the
action from the environment's action space directly.

This module provides the canonical parser so that every agent shares
the same extraction logic.
"""

import re
from typing import Optional


# Primary: ``` ... ``` (generic fenced block — the content is the action)
_GENERIC_BLOCK_RE = re.compile(r"```\s*(.*?)\s*```", re.DOTALL)

# Backward-compat fallback: ```action ... ``` (legacy format)
_ACTION_BLOCK_RE = re.compile(r"```action\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def parse_action_block(response: str) -> tuple[str, Optional[str]]:
    """Extract thought and action text from a model response.

    Parsing order:
    1. ````` ```\\n<action>\\n``` ````` (preferred — action is the content).
    2. ````` ```action\\n<action>\\n``` ````` (backward-compatible fallback).
    3. If neither is found, return ``(response, None)``.

    When multiple blocks exist the **last** match is used (chain-of-thought
    friendly -- the model typically reasons first and outputs the action
    block at the end).

    Args:
        response: The full model response string.

    Returns:
        ``(thought, action_text)`` where *thought* is everything before the
        last matched block and *action_text* is the content inside the
        block.  *action_text* is ``None`` when no block is found.
    """
    if not response:
        return "", None

    # 1. Try generic ``` ... ``` blocks first.
    matches = _GENERIC_BLOCK_RE.findall(response)
    if matches:
        raw = matches[-1].strip()
        # Strip common language tags that an LLM might prepend.
        for tag in ("action", "python", "text", "bash"):
            if raw.lower().startswith(tag):
                candidate = raw[len(tag):].strip()
                if candidate:
                    raw = candidate
                    break
        # Locate last occurrence of ``` for thought split.
        last_open = _find_last_block_start(response)
        thought = response[:last_open].strip() if last_open > 0 else ""
        return thought, raw

    # 2. No block found.
    return response, None


def _find_last_block_start(response: str) -> int:
    """Return the index of the opening ``` of the last fenced block.

    Scans backwards through all ``` ... ``` pairs and returns the start
    index of the opening ````` ``` ````` of the last complete pair.
    """
    # Find all opening positions of ``` blocks
    positions = [m.start() for m in re.finditer(r"```", response)]
    # Blocks come in pairs; the last pair's opening ``` is at positions[-2]
    if len(positions) >= 2:
        return positions[-2]
    return 0
