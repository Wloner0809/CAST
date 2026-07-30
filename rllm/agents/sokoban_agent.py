import logging
from typing import Any

from rllm.agents.game_agent import GameAgent
from rllm.environments.sokoban.sokoban import SokobanEnv

logger = logging.getLogger(__name__)


class SokobanAgent(GameAgent):
    SYSTEM_PROMPT: str = """You are a player solving a Sokoban puzzle.

Sokoban Quick Guide
Goal: Push all boxes (X) onto targets (O).

Symbols:
# Wall | _ Empty | O Target | X Box | * Box on target | P Player | S Player on target

Rules:
1. You can push a box by moving into it if the space behind it is free.
2. You cannot pull boxes.
3. Walls are impassable.

Valid Action (separated by | ):
Up | Down | Left | Right

You will be provided the current observation, please decide on the next action.
You should show your thought process and then input the final action in a ``` block.
You should only output the NEXT ACTION at each iteration in the ``` block.
For example, if you want to move up, you should output: ```Up```
Please show your thinking process and put the final action in a ``` block.
In every turn, the final action MUST be one of Up, Down, Left, Right.
"""

    MULTI_STEP_SYSTEM_PROMPT: str = """You are a player solving a Sokoban puzzle.

Sokoban Quick Guide
Goal: Push all boxes (X) onto targets (O).

Symbols:
# Wall | _ Empty | O Target | X Box | * Box on target | P Player | S Player on target

Rules:
1. You can push a box by moving into it if the space behind it is free.
2. You cannot pull boxes.
3. Walls are impassable.

Valid Action (separated by | ):
Up | Down | Left | Right

You will be provided the current observation, please decide on the next action.
You should show your thought process and then input the final action in a ``` block.
For example, if you want to move up, you should output: ```Up```
You can output multiple actions at once by separating them with ||.
For example, if you want to move up, then down, then right, you should output: ```Up || Down || Right```
Each action will be executed in sequence.
Please show your thinking process and put the final action in a ``` block.
"""

    def __init__(
        self,
        max_steps: int | None = None,
        use_accumulate_thinking: bool = True,
        use_multistep_prompt: bool = False,
        use_accumulate_history: bool = True,
        **kwargs: Any,
    ):
        self.multistep_prompt: bool = use_multistep_prompt
        super().__init__(
            max_steps=max_steps,
            use_accumulate_thinking=use_accumulate_thinking,
            use_accumulate_history=use_accumulate_history,
            **kwargs,
        )
        logger.info(
            "SokobanAgent initialized with max_steps: %s, "
            "use_accumulate_thinking: %s, use_multistep_prompt: %s, "
            "use_accumulate_history: %s",
            max_steps,
            use_accumulate_thinking,
            use_multistep_prompt,
            use_accumulate_history,
        )

    # ------------------------------------------------------------------
    # Template-method overrides
    # ------------------------------------------------------------------

    def _get_system_prompt(self) -> str:
        if self.multistep_prompt:
            return self.MULTI_STEP_SYSTEM_PROMPT
        return self.SYSTEM_PROMPT

    def _invalid_action_feedback(self, last_info: dict) -> str:
        return (
            "Your last action was invalid. Recheck the action format. "
            "You should only output the NEXT ACTION in ``` ```, "
            "and it must be one of Up/Down/Left/Right."
        )

    def _ineffective_action_feedback(self, last_info: dict) -> str:
        return (
            "Your last action was valid but ineffective "
            "(the state did not change). Choose a different direction "
            "that can move the player or push a box."
        )

    # ------------------------------------------------------------------
    # Action display
    # ------------------------------------------------------------------

    _ACTION_ID_TO_NAME = {"1": "Up", "2": "Down", "3": "Left", "4": "Right"}

    def _format_action_display(self, action: str) -> str:
        """Convert numeric Sokoban action(s) to readable direction names.

        Handles single actions (``"2"`` → ``"Down"``) and multi-step
        actions (``"1||3||2"`` → ``"Up || Left || Down"``).
        """
        if "||" in action:
            parts = [self._ACTION_ID_TO_NAME.get(a.strip(), a.strip()) for a in action.split("||")]
            return " || ".join(parts)
        return self._ACTION_ID_TO_NAME.get(action.strip(), action)

    # ------------------------------------------------------------------
    # Action parsing (required by GameAgent)
    # ------------------------------------------------------------------

    def _parse_action(self, raw_action_text: str) -> str:
        """Convert extracted action text to Sokoban env action integer(s).

        Handles both single actions and ``||``-separated multi-step actions.
        Returns a string of integer action id(s) for the environment.
        """
        direction_map = {"up": 1, "down": 2, "left": 3, "right": 4}

        if not raw_action_text:
            return str(SokobanEnv.INVALID_ACTION)

        # Multi-step actions separated by ||
        if "||" in raw_action_text:
            action_parts = [part.strip().lower() for part in raw_action_text.split("||")]
            action_list = []
            for part in action_parts:
                if part in direction_map:
                    action_list.append(str(direction_map[part]))
                elif part.isdigit() and int(part) in direction_map.values():
                    action_list.append(str(int(part)))
                else:
                    action_list.append(str(SokobanEnv.INVALID_ACTION))
            return "||".join(action_list)

        # Single action
        extracted = raw_action_text.strip().lower()
        if extracted in direction_map:
            return str(direction_map[extracted])
        if extracted.isdigit() and int(extracted) in direction_map.values():
            return str(int(extracted))
        return str(SokobanEnv.INVALID_ACTION)

    # ------------------------------------------------------------------
    # Validation helper (kept for external callers)
    # ------------------------------------------------------------------

    def _process_action_for_validation(self, response: str) -> str:
        return self._parse_action(response)
