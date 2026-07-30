"""ALFWorld Agent for text-based household task environments.

This agent is designed to interact with the ALFWorld environment,
performing household tasks like picking up objects, placing them,
heating, cooling, cleaning, etc.

Extends ``GameAgent`` to share common agent logic (prompt assembly,
action parsing, trajectory management) while only implementing
ALFWorld-specific details.
"""

import logging
from typing import Any

from rllm.agents.agent import Step
from rllm.agents.game_agent import GameAgent

logger = logging.getLogger(__name__)


class ALFWorldAgent(GameAgent):
    """Agent for ALFWorld text-based household task environment.

    This agent processes text observations from ALFWorld and generates
    appropriate actions to complete household tasks.

    Attributes:
        max_steps: Maximum number of steps per episode.
        use_admissible_commands: Alias for ``use_available_actions`` (backward compat).
    """

    SYSTEM_PROMPT: str = """You are an expert agent completing household tasks in an embodied environment. Your task is to complete household tasks by navigating rooms and interacting with objects.

Action Space:
Your action must be one of the following, strictly following the command (argument) format.

Navigation & Observation
- look: Look around your current location to get more details.
- inventory: Check the object you are currently holding (you can only hold one).
- go to (receptacle): Move to a receptacle (e.g., table, fridge, sink).

Interacting with Receptacles
- open (receptacle): Open a receptacle.
- close (receptacle): Close a receptacle.

Interacting with Objects
- take (object) from (receptacle): Pick up an object from a receptacle.
- move (object) to (receptacle): Place the object you are holding into or onto a receptacle.
- examine (object): Examine an object closely to learn its properties.

Changing Object States
- heat (object) with (receptacle): Heat an object with a device (e.g., microwave).
- cool (object) with (receptacle): Cool an object with a device (e.g., fridge).
- clean (object) with (receptacle): Clean an object with a device (e.g., sink).
- slice (object) with (object): Slice an object using a sharp object (e.g., knife).

Important Rules:
1. You must first navigate to a location before interacting with objects there
2. You can only hold one object at a time
3. Some objects need to be heated, cooled, or cleaned before placing them
4. Always check available actions to see what actions are currently valid

Response Format:
Think through your approach step by step, then provide your action in a ``` block.
For example: ```go to countertop 1```
Always provide exactly ONE action at a time.
"""

    def __init__(
        self,
        max_steps: int | None = None,
        use_admissible_commands: bool = True,
        use_accumulate_history: bool = True,
        use_accumulate_thinking: bool = True,
        use_available_actions: bool | None = None,
        **kwargs: Any,
    ):
        """Initialize ALFWorld agent.

        Args:
            max_steps: Maximum steps per episode.
            use_admissible_commands: Whether to show valid commands in prompts.
                Backward-compatible alias for ``use_available_actions``.
            use_accumulate_history: Whether to keep full conversation history.
            use_accumulate_thinking: Whether to keep thinking tokens in history.
            use_available_actions: If provided, takes precedence over
                ``use_admissible_commands``.
            **kwargs: Additional keyword arguments forwarded to ``GameAgent``.
        """
        # Backward compat: use_admissible_commands maps to use_available_actions
        if use_available_actions is None:
            use_available_actions = use_admissible_commands

        super().__init__(
            max_steps=max_steps,
            use_accumulate_thinking=use_accumulate_thinking,
            use_accumulate_history=use_accumulate_history,
            use_available_actions=use_available_actions,
            **kwargs,
        )
        logger.info(
            "ALFWorldAgent initialized with max_steps=%s, "
            "use_available_actions=%s",
            max_steps,
            use_available_actions,
        )

    # ------------------------------------------------------------------
    # Backward-compatibility properties
    # ------------------------------------------------------------------

    @property
    def use_admissible_commands(self) -> bool:
        """Alias for ``use_available_actions`` (backward compat)."""
        return self.use_available_actions

    @use_admissible_commands.setter
    def use_admissible_commands(self, value: bool) -> None:
        self.use_available_actions = value

    # ------------------------------------------------------------------
    # Template-method overrides
    # ------------------------------------------------------------------

    def _build_available_actions_section(self, info: dict) -> str:
        """Build the admissible commands section.

        ALFWorld environments provide commands under both
        ``admissible_commands`` and ``available_actions`` info keys.
        We prefer ``admissible_commands`` for legacy consistency but
        fall back to ``available_actions``.
        """
        if not self.use_available_actions:
            return ""

        commands = info.get("admissible_commands") or info.get("available_actions", [])
        if not commands:
            return ""

        commands_str = "\n".join(f"  - {cmd}" for cmd in commands)
        return f"Admissible commands:\n{commands_str}"

    def _build_feedback_section(self, current_obs_str: str, info: dict) -> str:
        """Build feedback about the previous action.

        In addition to the standard ``action_is_valid`` / ``action_is_effective``
        checks from the base class, ALFWorld uses the "Nothing happens"
        observation string as a signal that an action had no effect.
        """
        # Delegate to base class first for standard validity/effectiveness checks
        base_feedback = super()._build_feedback_section(current_obs_str, info)
        if base_feedback:
            return base_feedback

        # ALFWorld-specific: detect "Nothing happens" in the observation
        if not self._trajectory.steps:
            return ""
        last_step = self._trajectory.steps[-1]
        if last_step.action and "Nothing happens" in current_obs_str:
            return (
                f"Note: Your last action '{last_step.action}' had no effect. "
                f"Please try a different action."
            )

        return ""

    def _invalid_action_feedback(self, last_info: dict) -> str:
        """Return guidance about using ``` blocks."""
        return (
            "Your last action was invalid. Recheck the action format. "
            "You should only output the NEXT ACTION in ``` ``` and choose from the admissible commands."
        )

    # ------------------------------------------------------------------
    # Action parsing (required by GameAgent)
    # ------------------------------------------------------------------

    def _parse_action(self, raw_action_text: str) -> str:
        """Convert extracted action text into an ALFWorld command string.

        Cleans up common LLM quirks (e.g. prefixing with "action" or ":").
        Returns ``""`` when no valid action text can be extracted.
        """
        if not raw_action_text:
            return ""

        action_str = raw_action_text.strip()

        # Strip leading "action" prefix (case-insensitive)
        if action_str.lower().startswith("action"):
            action_str = action_str[6:].strip()

        # Strip leading ":"
        if action_str.startswith(":"):
            action_str = action_str[1:].strip()

        return action_str if action_str else ""

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def get_current_state(self) -> Step | None:
        """Get the most recent step."""
        if self._trajectory.steps:
            return self._trajectory.steps[-1]
        return None
