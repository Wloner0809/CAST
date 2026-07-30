"""Rush Hour agent for rllm.

Extends ``GameAgent`` which provides the standard ``__init__``,
``update_from_env``, ``update_from_model``, ``reset``, ``chat_completions``,
and ``trajectory`` implementations.

The agent only needs to define the system prompt and action parsing. The
environment parses the action via :func:`rllm.environments.rush_hour.board.parse_action`,
which accepts forms like ``"A+2"``, ``"B-1"``, and ``"[A+]"``; the agent simply
extracts and forwards the move string from the model's ``` block.
"""

import logging
import re
from typing import Any

from rllm.environments.rush_hour.board import parse_action
from rllm.agents.game_agent import GameAgent

logger = logging.getLogger(__name__)


class RushHourAgent(GameAgent):
    """Agent for the Rush Hour environment.

    Action format (inside ``` blocks):
        A single move ``<car><sign><steps>``, e.g. ``A+2`` (slide car A right
        2 cells) or ``B-1`` (slide car B up/left 1 cell).
    """

    SYSTEM_PROMPT: str = """You are a player solving a Rush Hour sliding-block puzzle.

Rush Hour Quick Guide
Goal: Slide the red car (labeled A) horizontally until it reaches the exit on the right edge of its row.

Board:
- A 6x6 grid of cells, zero-indexed (row, col).
- Cars and trucks are labeled A, B, C, ... Each occupies 2 or 3 cells in a line.
- The red car A is horizontal and sits on the exit row; the exit is on the right edge of that row.
- Horizontal cars can only move left or right. Vertical cars can only move up or down.
- Cells marked 'x' are walls and never move. Empty cells are '.' (or 'o').
- Cars cannot overlap, pass through each other, or pass through walls.

Action format:
- Output a single move as <car><sign><steps>.
- For a horizontal car: '+' moves right, '-' moves left; For a vertical car: '+' moves down, '-' moves up.
- <steps> is the number of cells to slide (omit it to slide 1 cell).
- All-or-nothing: <steps> must be exactly how far the car can slide. Overshooting (more than the free cells) does nothing.
- One slide of any distance counts as ONE move; prefer a single multi-cell slide (e.g. A+3) over several 1-cell moves.
- Examples: A+2 (move car A right 2 cells), B-1 (move car B up 1 cell), C+3 (move car C down 3 cells).

You will be provided the current observation, please decide on the next action.
You should only output the NEXT ACTION at each iteration in the ``` block.
For example, if you want to move car A right 2 cells, you should output: ```A+2```
Please put the final action in a ``` block.
"""

    # Matches a single move like "A+2", "B-1", "[A+]" (optional brackets/spaces).
    _ACTION_TEXT_RE = re.compile(r"\[?\s*([A-Za-z])\s*([+-])\s*(\d*)\s*\]?")

    def __init__(
        self,
        max_steps: int | None = None,
        use_accumulate_thinking: bool = True,
        use_accumulate_history: bool = True,
        **kwargs: Any,
    ):
        super().__init__(
            max_steps=max_steps,
            use_accumulate_thinking=use_accumulate_thinking,
            use_accumulate_history=use_accumulate_history,
            **kwargs,
        )
        logger.info(
            "RushHourAgent initialized with max_steps: %s, "
            "use_accumulate_thinking: %s, use_accumulate_history: %s",
            max_steps,
            use_accumulate_thinking,
            use_accumulate_history,
        )

    # ------------------------------------------------------------------
    # GameAgent abstract / override methods
    # ------------------------------------------------------------------

    def _parse_action(self, raw_action_text: str) -> str:
        """Convert extracted action text into the env-specific action string.

        Returns a canonical move string like ``"A+2"`` that the environment's
        :func:`parse_action` accepts, or ``""`` on parse failure.

        Uses ``fullmatch`` (not ``search``) so the ``` block must contain a
        single move and nothing else. This prevents two silent hijacks that a
        positional ``search`` allows: (1) prose with an incidental move-like
        token (``"move car b-1 now"`` -> ``B-1``), and (2) a block with several
        tokens where ``search`` grabs the wrong first one (``"b-1 but A+2"`` ->
        ``B-1`` instead of ``A+2``). A non-conforming block returns ``""`` and is
        surfaced to the model as an invalid action, matching the prompt's "only
        output the NEXT ACTION in the block" contract.
        """
        if not raw_action_text:
            return ""
        match = self._ACTION_TEXT_RE.fullmatch(raw_action_text.strip())
        if match is None:
            return ""
        label = match.group(1).upper()
        sign = match.group(2)
        digits = match.group(3)
        magnitude = digits if digits else "1"
        if int(magnitude) == 0:
            return ""
        candidate = f"{label}{sign}{magnitude}"
        # Final safety check: ensure the env parser accepts it.
        if parse_action(candidate) is None:
            return ""
        return candidate

    def _invalid_action_feedback(self, last_info: dict) -> str:
        return (
            "Your last action was invalid. Recheck the action format. "
            "You should only output the NEXT ACTION in ``` ```, as a single "
            "move like A+2 or B-1 (a valid car label, '+' or '-', and a step count)."
        )

    def _ineffective_action_feedback(self, last_info: dict) -> str:
        return (
            "Your last action was valid but ineffective (the board did not "
            "change). Either the car is blocked in that direction by a wall, the "
            "edge, or another car, or your <steps> overshot how far it can "
            "actually slide (the move is all-or-nothing). Choose a different car "
            "or direction, or reduce <steps> to a distance the car can move."
        )

    # ------------------------------------------------------------------
    # Validation helper (kept for external callers)
    # ------------------------------------------------------------------

    def _process_action_for_validation(self, response: str) -> str:
        return self._parse_action(response)
