"""Shared base class for game / environment agents.

``GameAgent`` extracts the common logic that is duplicated across the
environment agents (ALFWorld, Minesweeper, Rush Hour, Sokoban, WebShop).
Each subclass only needs to provide:

* ``SYSTEM_PROMPT`` -- the system-level instruction string.
* ``_parse_action(raw_text)`` -- convert the text extracted from the
  ````` ``` ````` block into the env-specific action string.

Subclasses *may* override any of the ``_build_*`` template methods to
customise prompt sections.
"""

import copy
import logging
from abc import abstractmethod
from typing import Any

from rllm.agents.action_utils import parse_action_block
from rllm.agents.agent import Action, BaseAgent, Step, Trajectory
from rllm.agents.context_manager import ContextManager

logger = logging.getLogger(__name__)


class GameAgent(BaseAgent):
    """Shared base for game/environment agents.

    Subclasses **must** define:

    * ``SYSTEM_PROMPT: str`` -- the system-level instruction.
    * ``_parse_action(self, raw_action_text: str) -> str`` -- convert the
      extracted text from the ````` ``` ````` block into the
      environment-specific action string.  Return ``""`` or the
      environment's ``INVALID_ACTION`` sentinel for parse failures.

    Subclasses **may** override:

    * ``_build_observation_section``
    * ``_build_feedback_section``
    * ``_build_available_actions_section``
    * ``_build_process_feedback_section`` (for verbal env feedback)
    * ``_build_action_prompt_section``
    * ``_get_system_prompt`` (for dynamic system prompts)
    """

    SYSTEM_PROMPT: str = ""

    def __init__(
        self,
        max_steps: int | None = None,
        use_accumulate_thinking: bool = True,
        use_accumulate_history: bool = True,
        use_available_actions: bool = False,
        history_length: int = -1,
        **kwargs,
    ):
        self._trajectory = Trajectory()
        self.messages: list[dict[str, str]] = []
        self.step: int = 0
        self.accumulate_thinking: bool = use_accumulate_thinking
        self.accumulate_history: bool = use_accumulate_history
        self.use_available_actions: bool = use_available_actions
        self.max_steps: int | None = max_steps
        self.current_observation: Any = None

        # --- Context Manager integration ---
        # Use ContextManager when history_length is explicitly configured to a
        # non-default value, or when the legacy ``accumulate_history=False``
        # flag is used.  Otherwise fall back to the original message list.
        self._use_context_manager = (history_length != -1) or (not use_accumulate_history)
        if self._use_context_manager:
            # Map legacy flag to an effective history_length:
            #   accumulate_history=True  -> -1 (all history)
            #   accumulate_history=False -> 0  (no history)
            # An explicit history_length always takes priority.
            if history_length != -1:
                effective_history = history_length
            else:
                effective_history = -1 if use_accumulate_history else 0
            self._context_manager: ContextManager | None = ContextManager(
                history_length=effective_history,
                system_prompt=self._get_system_prompt(),
            )
        else:
            self._context_manager = None

        self.reset()

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def update_from_env(
        self,
        observation: Any,
        reward: float,
        done: bool,
        info: dict,
        **kwargs,
    ):
        """Build the next user prompt from the environment observation.

        The prompt is assembled from a sequence of template methods that
        subclasses can override individually.
        """
        # 1. Update previous step's reward/done/info
        if self._trajectory.steps:
            cur_step = self._trajectory.steps[-1]
            cur_step.reward = reward
            cur_step.done = done
            cur_step.info = info

        current_obs_str = self._format_observation(observation)
        self._current_anchor_obs = current_obs_str  # For GiGPO state grouping

        # 2. Build user prompt from template sections
        parts: list[str] = []

        parts.append(self._build_observation_section(current_obs_str, info))

        feedback = self._build_feedback_section(current_obs_str, info)
        if feedback:
            parts.append(feedback)

        available_actions = self._build_available_actions_section(info)
        if available_actions:
            parts.append(available_actions)

        action_prompt = self._build_action_prompt_section()
        if action_prompt:
            parts.append(action_prompt)
        
        process_feedback = self._build_process_feedback_section(info)
        if process_feedback:
            parts.append(process_feedback)

        user_prompt = "\n".join(parts)

        if self._context_manager is not None:
            anchor_obs = kwargs.get("anchor_obs", current_obs_str)
            self._context_manager.store_user_turn(user_prompt, current_obs_str, anchor_obs)
        else:
            self.messages.append({"role": "user", "content": user_prompt})

        self.current_observation = current_obs_str

    def update_from_model(self, response: str, **kwargs) -> Action:
        """Extract action from model response via ``` blocks."""
        content = response if response else ""

        # Strip thinking tags if not accumulating
        if not self.accumulate_thinking:
            _, sep, after = content.partition("</think>")
            if sep:
                content = after

        thought, raw_action = parse_action_block(content)

        # Convert raw extracted text into env-specific action string
        if raw_action is not None:
            action_str = self._parse_action(raw_action)
        else:
            action_str = self._parse_action("")

        if self._context_manager is not None:
            self._context_manager.store_assistant_turn(content)
        else:
            self.messages.append({"role": "assistant", "content": content})

        step_info: dict = {}
        step_info["anchor_obs"] = getattr(self, "_current_anchor_obs", None)
        if self._context_manager is not None:
            step_info["anchor_obs"] = self._context_manager.get_anchor_obs()

        new_step = Step(
            chat_completions=copy.deepcopy(self.chat_completions),
            thought=thought,
            action=action_str,
            model_response=content,
            observation=self.current_observation,
            info=step_info,
        )
        self._trajectory.steps.append(new_step)

        self.step += 1

        return Action(action=action_str)

    def reset(self) -> None:
        """Reset agent for a new episode."""
        self._trajectory = Trajectory()
        self.messages = [
            {"role": "system", "content": self._get_system_prompt()}
        ]
        self.step = 0
        self.current_observation = None
        if getattr(self, "_context_manager", None) is not None:
            self._context_manager.reset()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        if self._context_manager is not None:
            return self._context_manager.build_chat_completions()
        # Legacy path: original accumulate_history logic
        if self.accumulate_history:
            return self.messages
        if len(self.messages) <= 1:
            return self.messages
        # Return system message + last user-assistant pair.
        # Walk backward from the end to find the last user message,
        # then take everything from that user message onward.
        # This ensures that:
        #   - After update_from_env (user appended):  [system, user]
        #   - After update_from_model (assistant appended): [system, user, assistant]
        system_msg = self.messages[0]
        tail: list[dict[str, str]] = []
        for msg in reversed(self.messages[1:]):
            tail.append(msg)
            if msg.get("role") == "user":
                break
        tail.reverse()
        return [system_msg] + tail

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    # ------------------------------------------------------------------
    # Template methods -- subclasses may override
    # ------------------------------------------------------------------

    def _get_system_prompt(self) -> str:
        """Return the system prompt string.

        Override for agents that need a dynamic system prompt, e.g. one built
        from the env's action space on the first observation.
        """
        return self.SYSTEM_PROMPT

    def _format_observation(self, observation: Any) -> str:
        """Format the raw observation into a string.

        Override for environments whose observations are dicts or need
        special formatting.
        """
        return str(observation)

    def _build_observation_section(self, obs_str: str, info: dict) -> str:
        """Build the observation header + body."""
        return f"Current Observation (Step {self.step}):\n{obs_str}\nYou have not achieved the goal yet."

    def _build_feedback_section(self, current_obs_str: str, info: dict) -> str:
        """Build feedback about the previous action (invalid / ineffective).

        Returns an empty string when there is no feedback to give.
        """
        if not self._trajectory.steps:
            return ""
        last_step = self._trajectory.steps[-1]
        if last_step.action is None:
            return ""

        last_info = last_step.info or {}

        if last_info.get("action_is_valid") is False:
            return self._invalid_action_feedback(last_info)

        if (
            last_info.get("action_is_effective") is False
            and last_step.observation == current_obs_str
        ):
            return self._ineffective_action_feedback(last_info)

        return ""

    def _invalid_action_feedback(self, last_info: dict) -> str:
        """Return the feedback string for an invalid action.

        Subclasses should override to give env-specific guidance.
        """
        return (
            "Your last action was invalid. Recheck the action format. "
            "You should only output the NEXT ACTION in ``` ```."
        )

    def _ineffective_action_feedback(self, last_info: dict) -> str:
        """Return the feedback string when the action had no effect."""
        return (
            "Your last action was valid but ineffective "
            "(the state did not change). Try a different action."
        )

    def _build_available_actions_section(self, info: dict) -> str:
        """Build the available-actions section.

        Only included when ``use_available_actions`` is ``True`` and the
        env provides them.  Subclasses may override for custom formatting.
        """
        if not self.use_available_actions:
            return ""
        actions = info.get("available_actions", [])
        if not actions:
            return ""
        if isinstance(actions, dict):
            # Some envs (WebShop) use a dict with clickables / search bar
            return ""
        actions_str = ", ".join(str(a) for a in actions)
        return f"Available actions: {actions_str}"

    def _build_process_feedback_section(self, info: dict) -> str:
        """Build the verbal process-feedback section.

        Override in feedback-aware agents to inject environment-provided
        feedback (e.g. ``info['verbal_feedback']``) into the prompt.
        The section is placed *after* available-actions and *before* the
        action-prompt, so that the action request is always the last thing
        the model sees.

        Returns an empty string by default (no feedback).
        """
        return ""

    def _build_action_prompt_section(self) -> str:
        """Build the trailing section with remaining steps and action request.

        Combines the remaining-steps info and the action prompt into one block.
        """
        parts: list[str] = []
        if self.max_steps is not None and self.max_steps - self.step > 0:
            parts.append(f"You have {self.max_steps - self.step} steps remaining.")
        parts.append("Please give the next action.")
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Action display helper
    # ------------------------------------------------------------------

    def _format_action_display(self, action: str) -> str:
        """Convert an env-level action string to a human-readable form.

        By default returns the action unchanged.  Subclasses whose
        ``_parse_action`` returns numeric IDs (e.g. Sokoban ``"2"``,
        2048 ``"1"``) should override this to return readable names
        (e.g. ``"Down"``, ``"Right"``).
        """
        return action

    # ------------------------------------------------------------------
    # Abstract method -- subclasses must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def _parse_action(self, raw_action_text: str) -> str:
        """Convert extracted action text into the env-specific action string.

        Args:
            raw_action_text: The text extracted from the ````` ``` `````
                block (already stripped of the fences).  May be empty if no
                block was found.

        Returns:
            The action string to send to the environment.  Return ``""``
            or the environment's INVALID_ACTION sentinel on parse failure.
        """
        ...
