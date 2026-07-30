"""Configurable context window management for multi-turn agent interactions.

``ContextManager`` decouples prompt construction from raw message accumulation.
Instead of appending every user/assistant message to a single growing list, it
stores structured per-step records and rebuilds the chat-completion messages on
demand using a configurable history window.

Design inspired by verl-agent's ``SimpleMemory`` + ``build_text_obs()`` pattern:

* Each step's ``(user_prompt, assistant_content, anchor_obs)`` is recorded.
* ``build_chat_completions()`` returns ``[system] + recent_N_steps + current_user``.
* ``anchor_obs`` (the raw observation *without* history) is preserved for
  future step-level advantage algorithms such as GiGPO.

Backward compatibility
----------------------
* ``history_length=-1`` reproduces the existing ``accumulate_history=True``
  behaviour (all history included).
* ``history_length=0`` reproduces ``accumulate_history=False`` (only the
  current user turn, no prior history).
"""

from __future__ import annotations

from typing import Any


class ContextManager:
    """Manage an agent's context window with a configurable history length.

    Parameters
    ----------
    history_length : int
        How many *completed* (user+assistant) turns to include before the
        current user turn.

        * ``-1`` -- include **all** prior turns (default, matches legacy
          ``accumulate_history=True``).
        * ``0`` -- include **no** prior turns (matches legacy
          ``accumulate_history=False``).
        * ``N > 0`` -- include the most recent *N* turns.

    system_prompt : str
        The system-level instruction placed at the start of every message
        list returned by :meth:`build_chat_completions`.
    """

    def __init__(
        self,
        history_length: int = -1,
        system_prompt: str = "",
    ) -> None:
        self.history_length = history_length
        self.system_prompt = system_prompt

        # Each entry: {user_prompt, assistant_content, anchor_obs}
        self._history: list[dict[str, Any]] = []

        # Staging area for the *current* step (before the assistant responds).
        self._current_user_prompt: str | None = None
        self._current_anchor_obs: Any = None
        # Tracks whether the current user prompt has been paired with an
        # assistant response (i.e., committed to history).  When True,
        # ``build_chat_completions`` includes the latest history entry in the
        # "tail" position rather than as a standalone current-user message.
        self._current_committed: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all history.  Call at the start of every new episode."""
        self._history.clear()
        self._current_user_prompt = None
        self._current_anchor_obs = None
        self._current_committed = False

    def store_user_turn(
        self,
        user_prompt: str,
        observation: Any,
        anchor_obs: Any = None,
    ) -> None:
        """Record the user (environment) turn for the current step.

        Called from ``GameAgent.update_from_env`` after the user prompt has
        been assembled.

        Parameters
        ----------
        user_prompt : str
            The fully-assembled user prompt string.
        observation : Any
            The formatted observation string (used as fallback for *anchor_obs*).
        anchor_obs : Any, optional
            The *raw* observation without any history context prepended.  If
            ``None``, falls back to *observation*.
        """
        self._current_user_prompt = user_prompt
        self._current_anchor_obs = anchor_obs if anchor_obs is not None else observation
        self._current_committed = False

    def store_assistant_turn(self, assistant_content: str) -> None:
        """Record the assistant turn and commit the step to history.

        Called from ``GameAgent.update_from_model`` after the model response
        has been processed.
        """
        self._history.append(
            {
                "user_prompt": self._current_user_prompt,
                "assistant_content": assistant_content,
                "anchor_obs": self._current_anchor_obs,
                # Populated later by inject_stop_token_id() after the model
                # output is available.  Defaults to None (use parser default).
                "_stop_token_id": None,
            }
        )
        self._current_committed = True

    def inject_stop_token_id(self, stop_token_id: int | None) -> None:
        """Attach the actual stop token ID to the latest committed history entry.

        This must be called **after** ``store_assistant_turn`` for the same
        step.  The stop token ID is used by ``parse_assistant`` to select the
        correct eot string when re-rendering the message (e.g., Qwen2.5 may
        use ``<|endoftext|>`` instead of ``<|im_end|>``).
        """
        if self._history and stop_token_id is not None:
            self._history[-1]["_stop_token_id"] = stop_token_id

    def build_chat_completions(self) -> list[dict[str, str]]:
        """Build the message list for the current step's LLM call.

        Returns
        -------
        list[dict[str, str]]
            A list of ``{"role": ..., "content": ...}`` dicts ready for the
            chat-completion API.

        Lifecycle
        ---------
        The method is called at two points in the agent lifecycle:

        1. **After ``update_from_env``** — the user prompt is staged but not
           yet committed.  The returned list is the *prompt* for model
           generation:  ``[system] + [history window] + [current user]``.

        2. **After ``update_from_model``** — the user-assistant pair has been
           committed to history.  The returned list should reflect the full
           exchange including the assistant response.  To achieve this, the
           *latest committed turn* is always appended to the tail, regardless
           of the history window, to mirror the legacy
           ``accumulate_history=False`` behaviour of returning
           ``[system, user, assistant]`` after the model response.
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
        ]

        if not self._current_committed:
            # ----------------------------------------------------------
            # State: after store_user_turn, before store_assistant_turn.
            # This is the prompt that will be sent to the model.
            # ----------------------------------------------------------
            # History window (only *completed* turns before the current one)
            history_window = self._get_history_window(self._history)
            for entry in history_window:
                messages.append({"role": "user", "content": entry["user_prompt"]})
                messages.append(self._build_assistant_msg(entry))

            # Current (pending) user turn
            if self._current_user_prompt is not None:
                messages.append({"role": "user", "content": self._current_user_prompt})
        else:
            # ----------------------------------------------------------
            # State: after store_assistant_turn.
            # The latest turn has been committed to history.  Show the
            # history window AND ensure the latest committed turn is
            # visible (as user + assistant), matching the legacy
            # ``[system, ..., user, assistant]`` snapshot behaviour.
            # ----------------------------------------------------------
            # History window from all committed turns.
            # For history_length=0, this is empty — but we still want to
            # show the *latest* committed pair.
            history_window = self._get_history_window(self._history)

            # Check if the latest committed turn is already in the window.
            latest_in_window = (
                len(history_window) > 0
                and history_window[-1] is self._history[-1]
            )

            if not latest_in_window and self._history:
                # Show everything in the window first …
                for entry in history_window:
                    messages.append({"role": "user", "content": entry["user_prompt"]})
                    messages.append(self._build_assistant_msg(entry))
                # … then append the latest committed pair.
                latest = self._history[-1]
                messages.append({"role": "user", "content": latest["user_prompt"]})
                messages.append(self._build_assistant_msg(latest))
            else:
                for entry in history_window:
                    messages.append({"role": "user", "content": entry["user_prompt"]})
                    messages.append(self._build_assistant_msg(entry))

        return messages

    def get_anchor_obs(self) -> Any:
        """Return the anchor observation for the current step.

        The anchor observation is the *raw* environment observation without
        any history context.  It can be used for step-level state grouping
        (e.g., GiGPO).
        """
        return self._current_anchor_obs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_assistant_msg(entry: dict[str, Any]) -> dict[str, Any]:
        """Build an assistant message dict from a history entry.

        Includes ``_stop_token_id`` when present so that
        ``parse_assistant()`` can select the correct eot string.
        """
        msg: dict[str, Any] = {
            "role": "assistant",
            "content": entry["assistant_content"],
        }
        stop_id = entry.get("_stop_token_id")
        if stop_id is not None:
            msg["_stop_token_id"] = stop_id
        return msg

    def _get_history_window(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return the slice of *history* governed by ``self.history_length``."""
        if self.history_length == -1:
            return history
        elif self.history_length > 0:
            return history[-self.history_length :]
        else:
            return []

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def step_count(self) -> int:
        """Number of completed (user+assistant) turns stored so far."""
        return len(self._history)
