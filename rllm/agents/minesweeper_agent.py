"""Minesweeper agent for rllm.

Extends ``GameAgent`` which provides the standard ``__init__``,
``update_from_env``, ``update_from_model``, ``reset``,
``chat_completions``, and ``trajectory`` implementations.

The agent only needs to define the system prompt, action parsing, and
env-specific feedback / observation formatting.
"""

import re

from rllm.agents.game_agent import GameAgent


class MinesweeperAgent(GameAgent):
    """Agent for the Minesweeper environment.

    Action format (inside ``` blocks):
        ``reveal R C``  -- reveal the cell at row R, column C.
        ``flag R C``    -- toggle the flag on the cell at row R, column C.
    """

    SYSTEM_PROMPT: str = """You are a player playing a Minesweeper game.

Minesweeper Rules:
- The board is a grid of cells. Some cells contain hidden mines.
- Your goal is to reveal all cells that do NOT contain mines.
- When you reveal a cell:
  - If it contains a mine, the game is over (you lose).
  - If it does not contain a mine, a number is shown indicating how many of its 8 neighboring cells contain mines (0-8).
  - If the number is 0, all adjacent cells are automatically revealed.
- You may also place or remove a flag on a cell to mark or unmark it as a suspected mine.
- Removing flags is allowed and encouraged if your later reasoning shows the flag was incorrect.

Board Symbols:
  .  Unrevealed cell
  F  Flagged cell (suspected mine)
 0-8 Revealed cell showing the count of adjacent mines
  *  Mine (shown only when you hit one)

Actions (one per turn):
  reveal R C  -- Reveal the cell at row R, column C.
  flag R C    -- Toggle a flag on the cell at row R, column C (place a flag if none exists, or REMOVE it if already flagged).

You should show your thinking process, then output exactly ONE action inside a ``` block.
For example: ```reveal 3 2``` or ```flag 0 4```
Use logic and deduction to avoid mines. Think step-by-step before acting.
"""

    # Regex to parse the action text itself.
    _ACTION_TEXT_RE = re.compile(r"(reveal|flag)\s+(\d+)\s+(\d+)", re.IGNORECASE)

    # ------------------------------------------------------------------
    # GameAgent abstract / override methods
    # ------------------------------------------------------------------

    def _parse_action(self, raw_action_text: str) -> str:
        """Convert extracted action text into the env-specific action string.

        Returns:
            A string like ``"reveal 3 2"`` or ``"flag 0 4"``.
            Returns ``""`` if parsing fails.
        """
        match = self._ACTION_TEXT_RE.search(raw_action_text)
        if match:
            action_type = match.group(1).lower()
            row = match.group(2)
            col = match.group(3)
            return f"{action_type} {row} {col}"
        return ""

    def _invalid_action_feedback(self, last_info: dict) -> str:
        """Return env-specific feedback for an invalid action."""
        if last_info.get("format_error"):
            return (
                "Your last action was invalid. Your response could not be parsed. "
                "Please use the format: ```reveal R C``` or ```flag R C```."
            )
        if last_info.get("out_of_bounds"):
            return (
                "Your last action was invalid. The coordinates you chose were "
                "outside the board. Please pick valid row and column indices."
            )
        if last_info.get("already_revealed"):
            return (
                "Your last action was invalid. The cell you chose was already "
                "revealed or flagged. Please choose a different unrevealed cell."
            )
        return (
            "Your last action was invalid. "
            "Please use the format: ```reveal R C``` or ```flag R C```."
        )

    def _build_observation_section(self, obs_str: str, info: dict) -> str:
        """Build the observation section using the board suffix from info."""
        board_text = info.get("suffix", obs_str)
        return f"Current Observation (Step {self.step}):\n{board_text}\nYou have not achieved the goal yet."
