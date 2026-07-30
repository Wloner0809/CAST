"""WebShop Agent for web shopping simulation environment.

This agent navigates a simulated e-commerce website to find and purchase
products that match given instructions. Built on top of ``GameAgent``,
it uses ``` blocks for action formatting.
"""

import logging
import re

from rllm.agents.agent import Step
from rllm.agents.game_agent import GameAgent

logger = logging.getLogger(__name__)


class WebShopAgent(GameAgent):
    """Agent for WebShop web shopping simulation environment.

    This agent processes text observations from WebShop pages and generates
    search queries and click actions to find and purchase products.

    Action format uses ``` blocks, consistent with other rllm agents.

    Attributes:
        max_steps: Maximum number of steps per episode.
        use_available_actions: Whether to include available actions in prompts.
        accumulate_history: Whether to accumulate full conversation history.
    """

    SYSTEM_PROMPT: str = """You are an expert agent completing a web-based shopping task. You need to interact with the web environment to complete the task.

Available actions:
- search[query]: Search for products using keywords
- click[element]: Click on a page element (product link, option, Buy Now, etc.)

You will be provided the current state of the webpage, please decide on the next action.
You should show your thought process and then put the final action in a ``` block.
You should only output the NEXT ACTION at each iteration in the ``` block.
For example, if you want to search for shoes, you should output: ```search[shoes]```
If you want to click Buy Now, you should output: ```click[Buy Now]```
Please show your thinking process and put the final action in a ``` block.
Output exactly ONE action per turn.
"""

    def __init__(
        self,
        max_steps: int | None = None,
        use_available_actions: bool = True,
        use_accumulate_history: bool = True,
        use_accumulate_thinking: bool = True,
        **kwargs,
    ):
        """Initialize WebShop agent.

        Args:
            max_steps: Maximum steps per episode.
            use_available_actions: Whether to show valid actions in prompts.
            use_accumulate_history: Whether to keep full conversation history.
            use_accumulate_thinking: Whether to keep thinking tokens in history.
        """
        super().__init__(
            max_steps=max_steps,
            use_accumulate_thinking=use_accumulate_thinking,
            use_accumulate_history=use_accumulate_history,
            use_available_actions=use_available_actions,
            **kwargs,
        )
        self.current_instruction: str = ""
        self.current_available_actions: dict = {}

    def reset(self) -> None:
        """Reset agent state for a new episode."""
        super().reset()
        self.current_instruction = ""
        self.current_available_actions = {}

    # ------------------------------------------------------------------
    # Action parsing
    # ------------------------------------------------------------------

    def _parse_action(self, raw_action_text: str) -> str:
        """Convert extracted action text into a search[...] or click[...] string.

        Args:
            raw_action_text: Text extracted from the ``` block.

        Returns:
            A well-formed action string, or "" on failure.
        """
        if not raw_action_text:
            return ""

        text = raw_action_text.strip()

        # If already well-formed, return as-is
        if self._is_well_formed_action(text):
            return text

        # Try to extract search[...] or click[...] from the text
        extracted = self._extract_action_from_text(text)
        if extracted:
            return extracted

        logger.debug(
            "WebShopAgent: Could not parse valid action from response, "
            "returning empty action"
        )
        return ""

    def _is_well_formed_action(self, text: str) -> bool:
        """Check if text is a well-formed search[...] or click[...] action."""
        text_lower = text.lower().strip()
        return text_lower.startswith("search[") or text_lower.startswith("click[")

    def _extract_action_from_text(self, text: str) -> str | None:
        """Try to extract a search[...] or click[...] action from text."""
        # Try search first
        search_match = re.search(r"(search\[[^\]]+\])", text, re.IGNORECASE)
        if search_match:
            return search_match.group(1)

        click_match = re.search(r"(click\[[^\]]+\])", text, re.IGNORECASE)
        if click_match:
            return click_match.group(1)

        return None

    # ------------------------------------------------------------------
    # Template method overrides
    # ------------------------------------------------------------------

    def _build_observation_section(self, obs_str: str, info: dict) -> str:
        """Build the observation section.

        On the first step, includes the task instruction and render instructions.
        On subsequent steps, shows only the observation.
        """
        # Update tracked state from info
        self.current_instruction = info.get("instruction", self.current_instruction)
        self.current_available_actions = info.get("available_actions", {})

        if self.step == 0:
            from rllm.environments.webshop.webshop_env import RENDER_INSTRUCTIONS

            return (
                f"Current Observation (Step {self.step}):\n"
                f"Shopping Task: {self.current_instruction}\n"
                f"{RENDER_INSTRUCTIONS}\n"
                f"{obs_str}\n"
                "You have not achieved the goal yet."
            )
        return f"Current Observation (Step {self.step}):\n{obs_str}\nYou have not achieved the goal yet."

    def _build_available_actions_section(self, info: dict) -> str:
        """Build available actions using WebShop-specific formatting."""
        if not self.use_available_actions:
            return ""
        actions_str = self._format_available_actions()
        if actions_str:
            return f"You must choose from these actions: {actions_str}."
        return ""

    def _invalid_action_feedback(self, last_info: dict) -> str:
        """Return feedback for an invalid action."""
        return (
            "Your last action was invalid. Recheck the action format. "
            "You should only output the NEXT ACTION in ``` ```, "
            "and it must be search[...] or click[...]."
        )

    # ------------------------------------------------------------------
    # WebShop-specific helpers
    # ------------------------------------------------------------------

    def _format_available_actions(self) -> str:
        """Format available actions as a string list.

        Returns a string like: "search[<content>], click[item1], click[buy now]"
        Filters out 'search' from clickables and removes 'next >' at end of page.
        """
        formatted = []

        if self.current_available_actions.get("has_search_bar", False):
            formatted.append("search[<content>]")

        clickables = self.current_available_actions.get("clickables", [])
        for clickable in clickables:
            if clickable.lower() != "search":
                formatted.append(f"click[{clickable}]")

        # Remove 'click[next >]' when at end of page
        is_end_of_page = tuple(formatted) == (
            "click[back to search]",
            "click[< prev]",
            "click[next >]",
        )
        if is_end_of_page:
            formatted.remove("click[next >]")

        return ", ".join(formatted)

    def get_current_state(self) -> Step | None:
        """Get the most recent step."""
        if self._trajectory.steps:
            return self._trajectory.steps[-1]
        return None
