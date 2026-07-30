"""Comprehensive tests for GameAgent.chat_completions with accumulate_history=False.

Validates that the chat_completions property returns the correct messages
at every point in the agent lifecycle, and that Step.chat_completions
snapshots captured inside update_from_model contain both the user and
assistant messages (not just the assistant).

This test uses MinesweeperAgent and SokobanAgent as concrete GameAgent
subclasses that exercise the shared base-class logic.
"""

import copy

import pytest

from rllm.agents.minesweeper_agent import MinesweeperAgent
from rllm.environments.minesweeper.minesweeper_env import MinesweeperEnv

# SokobanAgent requires gym_sokoban; skip these tests if unavailable.
sokoban_available = True
try:
    from rllm.agents.sokoban_agent import SokobanAgent
except ImportError:
    sokoban_available = False


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _roles(messages: list[dict[str, str]]) -> list[str]:
    """Extract just the roles from a message list for concise assertions."""
    return [m["role"] for m in messages]


# ---------------------------------------------------------------------------
# Core lifecycle tests with MinesweeperAgent
# ---------------------------------------------------------------------------

class TestChatCompletionsNoHistory:
    """Tests for chat_completions when accumulate_history=False."""

    def _make_agent_and_env(self):
        agent = MinesweeperAgent(max_steps=25, use_accumulate_history=False)
        env = MinesweeperEnv(rows=5, cols=5, num_mines=3, seed=42)
        obs, info = env.reset()
        return agent, env, obs, info

    # -- After reset (only system message) --

    def test_after_reset_returns_system_only(self):
        agent, _, _, _ = self._make_agent_and_env()
        cc = agent.chat_completions
        assert len(cc) == 1
        assert cc[0]["role"] == "system"

    # -- After first update_from_env --

    def test_after_first_update_from_env(self):
        """After update_from_env, chat_completions = [system, user]."""
        agent, env, obs, info = self._make_agent_and_env()
        agent.update_from_env(obs, 0.0, False, info)

        cc = agent.chat_completions
        assert _roles(cc) == ["system", "user"]
        # User message should contain the board observation
        assert "step 0" in cc[1]["content"].lower()

    # -- After first update_from_model --

    def test_after_first_update_from_model(self):
        """After update_from_model, chat_completions = [system, user, assistant]."""
        agent, env, obs, info = self._make_agent_and_env()
        agent.update_from_env(obs, 0.0, False, info)
        agent.update_from_model("I'll reveal a cell. ```action\nreveal 2 2\n```")

        cc = agent.chat_completions
        assert _roles(cc) == ["system", "user", "assistant"]
        # The user message with board info must still be present
        assert "step 0" in cc[1]["content"].lower()
        # The assistant response must be present
        assert "reveal 2 2" in cc[2]["content"]

    # -- After second update_from_env (new turn) --

    def test_after_second_update_from_env(self):
        """After second update_from_env, previous turn is dropped: [system, user_new]."""
        agent, env, obs, info = self._make_agent_and_env()
        agent.update_from_env(obs, 0.0, False, info)
        agent.update_from_model("```action\nreveal 2 2\n```")

        obs2, reward2, done2, info2 = env.step("reveal 2 2")
        agent.update_from_env(obs2, reward2, done2, info2)

        cc = agent.chat_completions
        assert _roles(cc) == ["system", "user"]
        # New user message should reference step 1
        assert "step 1" in cc[1]["content"].lower()

    # -- After second update_from_model --

    def test_after_second_update_from_model(self):
        """After second update_from_model, [system, user, assistant] for the latest pair."""
        agent, env, obs, info = self._make_agent_and_env()

        # Turn 1
        agent.update_from_env(obs, 0.0, False, info)
        agent.update_from_model("```action\nreveal 2 2\n```")

        # Turn 2
        obs2, reward2, done2, info2 = env.step("reveal 2 2")
        agent.update_from_env(obs2, reward2, done2, info2)
        agent.update_from_model("```action\nreveal 0 0\n```")

        cc = agent.chat_completions
        assert _roles(cc) == ["system", "user", "assistant"]
        # Should be the second turn's content
        assert "step 1" in cc[1]["content"].lower()
        assert "reveal 0 0" in cc[2]["content"]

    # -- Multi-turn: messages list grows but chat_completions stays short --

    def test_messages_grow_but_chat_completions_stays_short(self):
        """Internal messages list grows over turns, but chat_completions stays [system, user(, assistant)]."""
        agent, env, obs, info = self._make_agent_and_env()

        for turn in range(5):
            agent.update_from_env(f"obs_turn_{turn}", 0.0, False, info)
            cc_before_model = agent.chat_completions
            assert _roles(cc_before_model) == ["system", "user"]

            agent.update_from_model(f"```action\nreveal {turn} {turn}\n```")
            cc_after_model = agent.chat_completions
            assert _roles(cc_after_model) == ["system", "user", "assistant"]

        # With ContextManager active (accumulate_history=False), messages are
        # managed by the ContextManager rather than the raw messages list.
        # The ContextManager stores 5 completed turns internally.
        if agent._context_manager is not None:
            assert agent._context_manager.step_count == 5
        else:
            # Legacy path: internal messages list has system + 5*(user + assistant) = 11
            assert len(agent.messages) == 11
        # But chat_completions only shows the latest pair
        assert len(agent.chat_completions) <= 3


# ---------------------------------------------------------------------------
# Step.chat_completions snapshot tests
# ---------------------------------------------------------------------------

class TestStepChatCompletionsSnapshot:
    """Verify that Step.chat_completions (taken inside update_from_model)
    contains both user and assistant messages."""

    def _make_agent_and_env(self):
        agent = MinesweeperAgent(max_steps=25, use_accumulate_history=False)
        env = MinesweeperEnv(rows=5, cols=5, num_mines=3, seed=42)
        obs, info = env.reset()
        return agent, env, obs, info

    def test_step_snapshot_includes_user_and_assistant(self):
        """Step.chat_completions snapshot must have [system, user, assistant]."""
        agent, env, obs, info = self._make_agent_and_env()
        agent.update_from_env(obs, 0.0, False, info)
        agent.update_from_model("```action\nreveal 2 2\n```")

        step = agent.trajectory.steps[0]
        snapshot = step.chat_completions
        assert _roles(snapshot) == ["system", "user", "assistant"]

    def test_step_snapshot_user_contains_board_info(self):
        """Step.chat_completions snapshot user message must contain board/observation info."""
        agent, env, obs, info = self._make_agent_and_env()
        agent.update_from_env(obs, 0.0, False, info)
        agent.update_from_model("```action\nreveal 2 2\n```")

        step = agent.trajectory.steps[0]
        user_msg = step.chat_completions[1]
        assert user_msg["role"] == "user"
        # The user message should contain the observation (board state)
        assert "step 0" in user_msg["content"].lower()

    def test_step_snapshot_is_deep_copy(self):
        """Step.chat_completions snapshot should be a deep copy, not a reference."""
        agent, env, obs, info = self._make_agent_and_env()
        agent.update_from_env(obs, 0.0, False, info)
        agent.update_from_model("```action\nreveal 2 2\n```")

        step0_snapshot = agent.trajectory.steps[0].chat_completions

        # Continue interaction
        obs2, reward2, done2, info2 = env.step("reveal 2 2")
        agent.update_from_env(obs2, reward2, done2, info2)
        agent.update_from_model("```action\nreveal 0 0\n```")

        step1_snapshot = agent.trajectory.steps[1].chat_completions

        # Step 0 snapshot should NOT have been mutated by step 1
        assert _roles(step0_snapshot) == ["system", "user", "assistant"]
        assert _roles(step1_snapshot) == ["system", "user", "assistant"]
        # Content should differ between steps
        assert step0_snapshot[1]["content"] != step1_snapshot[1]["content"]

    def test_multiple_steps_each_have_correct_snapshot(self):
        """Each step's snapshot should reflect its own turn's user-assistant pair."""
        agent, env, obs, info = self._make_agent_and_env()

        # Use SokobanAgent (simpler _build_observation_section) to pass
        # raw string observations through directly.
        if not sokoban_available:
            pytest.skip("gym_sokoban not available")
        agent = SokobanAgent(max_steps=10, use_accumulate_history=False)

        for turn in range(3):
            agent.update_from_env(f"board_turn_{turn}", 0.0, False, {})
            agent.update_from_model(f"```action\nUp\n```")

        assert len(agent.trajectory.steps) == 3

        for turn in range(3):
            snapshot = agent.trajectory.steps[turn].chat_completions
            assert _roles(snapshot) == ["system", "user", "assistant"]
            assert f"board_turn_{turn}" in snapshot[1]["content"]


# ---------------------------------------------------------------------------
# Comparison with accumulate_history=True
# ---------------------------------------------------------------------------

class TestChatCompletionsWithHistory:
    """Ensure accumulate_history=True is unaffected by the fix."""

    def test_with_history_returns_all_messages(self):
        agent = MinesweeperAgent(max_steps=25, use_accumulate_history=True)
        env = MinesweeperEnv(rows=5, cols=5, num_mines=3, seed=42)
        obs, info = env.reset()

        agent.update_from_env(obs, 0.0, False, info)
        agent.update_from_model("```action\nreveal 2 2\n```")

        obs2, reward2, done2, info2 = env.step("reveal 2 2")
        agent.update_from_env(obs2, reward2, done2, info2)
        agent.update_from_model("```action\nreveal 0 0\n```")

        cc = agent.chat_completions
        # system + user1 + assistant1 + user2 + assistant2 = 5
        assert len(cc) == 5
        assert _roles(cc) == ["system", "user", "assistant", "user", "assistant"]


# ---------------------------------------------------------------------------
# SokobanAgent cross-check (if available)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not sokoban_available, reason="gym_sokoban not installed")
class TestSokobanChatCompletionsNoHistory:
    """Verify fix applies to SokobanAgent (another GameAgent subclass)."""

    def test_sokoban_after_update_from_model(self):
        agent = SokobanAgent(max_steps=10, use_accumulate_history=False)
        agent.update_from_env("grid_obs", 0.0, False, {})
        agent.update_from_model("```action\nUp\n```")

        cc = agent.chat_completions
        assert _roles(cc) == ["system", "user", "assistant"]

    def test_sokoban_step_snapshot(self):
        agent = SokobanAgent(max_steps=10, use_accumulate_history=False)
        agent.update_from_env("grid_obs", 0.0, False, {})
        agent.update_from_model("```action\nUp\n```")

        step = agent.trajectory.steps[0]
        assert _roles(step.chat_completions) == ["system", "user", "assistant"]
        assert "grid_obs" in step.chat_completions[1]["content"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestChatCompletionsEdgeCases:
    """Edge cases for the chat_completions property."""

    def test_empty_messages_after_reset(self):
        """After reset, only system message is present."""
        agent = MinesweeperAgent(max_steps=25, use_accumulate_history=False)
        cc = agent.chat_completions
        assert len(cc) == 1
        assert cc[0]["role"] == "system"

    def test_single_user_message(self):
        """With only system + user, returns [system, user].

        When ContextManager is active, we use update_from_env (the proper API)
        rather than directly appending to agent.messages, since the
        ContextManager manages messages separately.
        """
        agent = MinesweeperAgent(max_steps=25, use_accumulate_history=False)
        if agent._context_manager is not None:
            # Use the proper API to register a user turn
            agent.update_from_env("test", 0.0, False, {})
        else:
            agent.messages.append({"role": "user", "content": "test"})
        cc = agent.chat_completions
        assert _roles(cc) == ["system", "user"]

    def test_system_only_with_accumulate_false(self):
        """With only system msg and accumulate=False, returns [system]."""
        agent = MinesweeperAgent(max_steps=25, use_accumulate_history=False)
        cc = agent.chat_completions
        assert _roles(cc) == ["system"]

    def test_chat_completions_does_not_mutate_messages(self):
        """Calling chat_completions should not modify the internal messages list."""
        agent = MinesweeperAgent(max_steps=25, use_accumulate_history=False)
        env = MinesweeperEnv(rows=5, cols=5, num_mines=3, seed=42)
        obs, info = env.reset()

        agent.update_from_env(obs, 0.0, False, info)
        agent.update_from_model("```action\nreveal 2 2\n```")

        messages_before = copy.deepcopy(agent.messages)
        _ = agent.chat_completions
        assert agent.messages == messages_before
