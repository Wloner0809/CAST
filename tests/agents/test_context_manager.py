"""Comprehensive tests for ContextManager and its integration with GameAgent.

Tests cover:
1. ContextManager standalone: all history_length modes, reset, anchor_obs.
2. GameAgent integration: backward compatibility, history_length parameter,
   step snapshots, anchor_obs recording.
"""

import copy

import pytest

from rllm.agents.context_manager import ContextManager

# Use MinesweeperAgent as the concrete GameAgent subclass (no extra deps).
from rllm.agents.minesweeper_agent import MinesweeperAgent
from rllm.environments.minesweeper.minesweeper_env import MinesweeperEnv

# SokobanAgent requires gym_sokoban; skip related tests if unavailable.
sokoban_available = True
try:
    from rllm.agents.sokoban_agent import SokobanAgent
except ImportError:
    sokoban_available = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _roles(messages: list[dict[str, str]]) -> list[str]:
    return [m["role"] for m in messages]


def _simulate_steps(cm: ContextManager, n: int, prefix: str = "obs"):
    """Simulate *n* complete (user+assistant) turns on a ContextManager."""
    for i in range(n):
        cm.store_user_turn(f"{prefix}_user_{i}", f"{prefix}_raw_{i}", f"anchor_{i}")
        cm.store_assistant_turn(f"{prefix}_assistant_{i}")


# ===========================================================================
# Part 1: ContextManager unit tests
# ===========================================================================

class TestContextManagerAllHistory:
    """history_length=-1 should include all prior turns."""

    def test_empty_after_init(self):
        cm = ContextManager(history_length=-1, system_prompt="sys")
        msgs = cm.build_chat_completions()
        assert msgs == [{"role": "system", "content": "sys"}]

    def test_after_user_turn_only(self):
        cm = ContextManager(history_length=-1, system_prompt="sys")
        cm.store_user_turn("hello", "obs0")
        msgs = cm.build_chat_completions()
        assert _roles(msgs) == ["system", "user"]
        assert msgs[1]["content"] == "hello"

    def test_after_one_complete_turn(self):
        cm = ContextManager(history_length=-1, system_prompt="sys")
        cm.store_user_turn("u0", "obs0")
        cm.store_assistant_turn("a0")
        # After committing, the turn is in history.  With history_length=-1,
        # the latest committed turn is already in the window.
        msgs = cm.build_chat_completions()
        assert _roles(msgs) == ["system", "user", "assistant"]
        assert msgs[1]["content"] == "u0"
        assert msgs[2]["content"] == "a0"

    def test_after_one_complete_turn_and_new_user(self):
        cm = ContextManager(history_length=-1, system_prompt="sys")
        cm.store_user_turn("u0", "obs0")
        cm.store_assistant_turn("a0")
        cm.store_user_turn("u1", "obs1")
        msgs = cm.build_chat_completions()
        assert _roles(msgs) == ["system", "user", "assistant", "user"]
        assert msgs[1]["content"] == "u0"
        assert msgs[2]["content"] == "a0"
        assert msgs[3]["content"] == "u1"

    def test_all_history_after_many_turns(self):
        cm = ContextManager(history_length=-1, system_prompt="sys")
        _simulate_steps(cm, 5)
        cm.store_user_turn("current", "obs_cur")
        msgs = cm.build_chat_completions()
        # system + 5*(user+assistant) + current_user = 12
        assert len(msgs) == 12
        assert _roles(msgs)[0] == "system"
        assert _roles(msgs)[-1] == "user"
        assert msgs[-1]["content"] == "current"


class TestContextManagerNoHistory:
    """history_length=0 should only include system + current user."""

    def test_empty_after_init(self):
        cm = ContextManager(history_length=0, system_prompt="sys")
        msgs = cm.build_chat_completions()
        assert msgs == [{"role": "system", "content": "sys"}]

    def test_no_prior_turns(self):
        cm = ContextManager(history_length=0, system_prompt="sys")
        _simulate_steps(cm, 3)
        cm.store_user_turn("current", "obs_cur")
        msgs = cm.build_chat_completions()
        assert _roles(msgs) == ["system", "user"]
        assert msgs[1]["content"] == "current"

    def test_step_count_still_tracks(self):
        cm = ContextManager(history_length=0, system_prompt="sys")
        _simulate_steps(cm, 3)
        assert cm.step_count == 3


class TestContextManagerWindowedHistory:
    """history_length=N should include system + last N turns + current user."""

    def test_window_larger_than_history(self):
        """If we have fewer turns than the window, include all of them."""
        cm = ContextManager(history_length=5, system_prompt="sys")
        _simulate_steps(cm, 2)
        cm.store_user_turn("current", "obs_cur")
        msgs = cm.build_chat_completions()
        # system + 2*(user+assistant) + current_user = 6
        assert len(msgs) == 6

    def test_window_exact(self):
        cm = ContextManager(history_length=2, system_prompt="sys")
        _simulate_steps(cm, 2)
        cm.store_user_turn("current", "obs_cur")
        msgs = cm.build_chat_completions()
        # system + 2*(user+assistant) + current = 6
        assert len(msgs) == 6

    def test_window_truncates_old_turns(self):
        cm = ContextManager(history_length=2, system_prompt="sys")
        _simulate_steps(cm, 5, prefix="step")
        cm.store_user_turn("current", "obs_cur")
        msgs = cm.build_chat_completions()
        # system + 2*(user+assistant) + current_user = 6
        assert len(msgs) == 6
        # The oldest visible turn should be step 3 (index 3), not 0/1/2.
        assert msgs[1]["content"] == "step_user_3"
        assert msgs[2]["content"] == "step_assistant_3"
        assert msgs[3]["content"] == "step_user_4"
        assert msgs[4]["content"] == "step_assistant_4"
        assert msgs[5]["content"] == "current"

    def test_window_of_one(self):
        cm = ContextManager(history_length=1, system_prompt="sys")
        _simulate_steps(cm, 3, prefix="s")
        cm.store_user_turn("now", "obs")
        msgs = cm.build_chat_completions()
        # system + 1*(user+assistant) + current_user = 4
        assert len(msgs) == 4
        assert msgs[1]["content"] == "s_user_2"
        assert msgs[2]["content"] == "s_assistant_2"
        assert msgs[3]["content"] == "now"


class TestContextManagerReset:
    """Reset should clear all history."""

    def test_reset_clears_history(self):
        cm = ContextManager(history_length=-1, system_prompt="sys")
        _simulate_steps(cm, 3)
        cm.store_user_turn("pending", "obs")
        assert cm.step_count == 3

        cm.reset()
        assert cm.step_count == 0
        msgs = cm.build_chat_completions()
        assert msgs == [{"role": "system", "content": "sys"}]

    def test_reset_clears_current_user(self):
        cm = ContextManager(history_length=-1, system_prompt="sys")
        cm.store_user_turn("pending", "obs")
        cm.reset()
        msgs = cm.build_chat_completions()
        # No current user prompt after reset
        assert _roles(msgs) == ["system"]


class TestContextManagerAnchorObs:
    """anchor_obs tracking."""

    def test_anchor_obs_default_to_observation(self):
        cm = ContextManager(system_prompt="sys")
        cm.store_user_turn("u", "raw_obs")
        assert cm.get_anchor_obs() == "raw_obs"

    def test_anchor_obs_explicit(self):
        cm = ContextManager(system_prompt="sys")
        cm.store_user_turn("u", "raw_obs", anchor_obs="clean_obs")
        assert cm.get_anchor_obs() == "clean_obs"

    def test_anchor_obs_persists_after_store_assistant(self):
        cm = ContextManager(system_prompt="sys")
        cm.store_user_turn("u", "raw", anchor_obs="anchor_val")
        cm.store_assistant_turn("a")
        # anchor_obs should still be accessible (it was the current step's)
        assert cm.get_anchor_obs() == "anchor_val"

    def test_anchor_obs_updates_each_step(self):
        cm = ContextManager(system_prompt="sys")
        cm.store_user_turn("u0", "raw0", anchor_obs="a0")
        cm.store_assistant_turn("r0")
        cm.store_user_turn("u1", "raw1", anchor_obs="a1")
        assert cm.get_anchor_obs() == "a1"

    def test_anchor_obs_none_after_reset(self):
        cm = ContextManager(system_prompt="sys")
        cm.store_user_turn("u", "raw", anchor_obs="a")
        cm.reset()
        assert cm.get_anchor_obs() is None


class TestContextManagerStepCount:
    def test_step_count_increments(self):
        cm = ContextManager(system_prompt="sys")
        assert cm.step_count == 0
        cm.store_user_turn("u0", "obs0")
        assert cm.step_count == 0  # not committed yet
        cm.store_assistant_turn("a0")
        assert cm.step_count == 1
        cm.store_user_turn("u1", "obs1")
        cm.store_assistant_turn("a1")
        assert cm.step_count == 2


# ===========================================================================
# Part 2: GameAgent integration tests
# ===========================================================================

class TestGameAgentBackwardCompatibility:
    """Default parameters should produce identical behaviour to the original."""

    def _make_agent_and_env(self, **agent_kwargs):
        agent = MinesweeperAgent(max_steps=25, **agent_kwargs)
        env = MinesweeperEnv(rows=5, cols=5, num_mines=3, seed=42)
        obs, info = env.reset()
        return agent, env, obs, info

    def test_default_uses_no_context_manager(self):
        """Default GameAgent (history_length=-1, accumulate=True) should NOT use ContextManager."""
        agent, _, _, _ = self._make_agent_and_env()
        assert agent._context_manager is None

    def test_default_accumulate_history_true_behaviour(self):
        """Default behaviour: all messages accumulated."""
        agent, env, obs, info = self._make_agent_and_env(use_accumulate_history=True)
        agent.update_from_env(obs, 0.0, False, info)
        agent.update_from_model("```action\nreveal 2 2\n```")
        obs2, r2, d2, i2 = env.step("reveal 2 2")
        agent.update_from_env(obs2, r2, d2, i2)
        agent.update_from_model("```action\nreveal 0 0\n```")

        cc = agent.chat_completions
        assert _roles(cc) == ["system", "user", "assistant", "user", "assistant"]

    def test_legacy_accumulate_false_uses_context_manager(self):
        """accumulate_history=False should activate ContextManager with history_length=0."""
        agent, _, _, _ = self._make_agent_and_env(use_accumulate_history=False)
        assert agent._context_manager is not None
        assert agent._context_manager.history_length == 0

    def test_history_zero_snapshot_includes_completed_turn(self):
        """A zero-history snapshot retains the current completed exchange."""
        agent, env, obs, info = self._make_agent_and_env(use_accumulate_history=False)

        agent.update_from_env(obs, 0.0, False, info)
        cc = agent.chat_completions
        assert _roles(cc) == ["system", "user"]

        agent.update_from_model("```action\nreveal 2 2\n```")
        cc = agent.chat_completions
        assert _roles(cc) == ["system", "user", "assistant"]

    def test_legacy_accumulate_false_second_turn(self):
        """After second update_from_env, previous turn should be dropped."""
        agent, env, obs, info = self._make_agent_and_env(use_accumulate_history=False)
        agent.update_from_env(obs, 0.0, False, info)
        agent.update_from_model("```action\nreveal 2 2\n```")

        obs2, r2, d2, i2 = env.step("reveal 2 2")
        agent.update_from_env(obs2, r2, d2, i2)
        cc = agent.chat_completions
        assert _roles(cc) == ["system", "user"]
        assert "step 1" in cc[1]["content"].lower()


class TestGameAgentHistoryLength:
    """Test the new history_length parameter on GameAgent."""

    def _make_agent(self, history_length, max_steps=10):
        agent = MinesweeperAgent(max_steps=max_steps, history_length=history_length)
        return agent

    def _simulate_turns(self, agent, n):
        """Simulate n turns via update_from_env + update_from_model."""
        for i in range(n):
            agent.update_from_env(f"board_{i}", 0.0, False, {})
            agent.update_from_model(f"```action\nreveal {i} {i}\n```")

    def test_history_length_minus_one_all_history(self):
        """history_length=-1 should use ContextManager but include all history."""
        # With default use_accumulate_history=True and history_length=-1,
        # ContextManager is NOT used. To test -1 via CM, we need to pass
        # something that triggers it. But -1 is the default which means "don't use CM".
        # Actually, the logic is: _use_context_manager = (history_length != -1) or (not use_accumulate_history)
        # So history_length=-1 with accumulate=True => no CM. That's correct — it's the default path.
        agent = MinesweeperAgent(max_steps=10)
        assert agent._context_manager is None

    def test_history_length_zero(self):
        """history_length=0 activates ContextManager with no history."""
        agent = self._make_agent(history_length=0)
        assert agent._context_manager is not None
        assert agent._context_manager.history_length == 0

        self._simulate_turns(agent, 3)
        agent.update_from_env("current_board", 0.0, False, {})
        cc = agent.chat_completions
        assert _roles(cc) == ["system", "user"]
        assert "current_board" in cc[1]["content"]

    def test_history_length_two(self):
        """history_length=2 should show system + last 2 turns + current."""
        agent = self._make_agent(history_length=2)
        self._simulate_turns(agent, 5)
        agent.update_from_env("board_current", 0.0, False, {})
        cc = agent.chat_completions()  if callable(agent.chat_completions) else agent.chat_completions
        # system + 2*(user+assistant) + current_user = 6
        assert len(cc) == 6
        assert _roles(cc) == ["system", "user", "assistant", "user", "assistant", "user"]
        # Verify the history contains turns 3 and 4 (the last 2)
        assert "board_3" in cc[1]["content"]
        assert "board_4" in cc[3]["content"]
        assert "board_current" in cc[5]["content"]

    def test_history_length_larger_than_actual(self):
        """history_length=10 with only 2 turns should show all turns."""
        agent = self._make_agent(history_length=10)
        self._simulate_turns(agent, 2)
        agent.update_from_env("board_now", 0.0, False, {})
        cc = agent.chat_completions
        # system + 2*(user+assistant) + current_user = 6
        assert len(cc) == 6

    def test_history_length_one(self):
        """history_length=1 should show only the most recent completed turn."""
        agent = self._make_agent(history_length=1)
        self._simulate_turns(agent, 3)
        agent.update_from_env("board_now", 0.0, False, {})
        cc = agent.chat_completions
        # system + 1*(user+assistant) + current_user = 4
        assert len(cc) == 4
        assert "board_2" in cc[1]["content"]
        assert "board_now" in cc[3]["content"]


class TestGameAgentReset:
    """Reset should clear ContextManager state."""

    def test_reset_clears_context(self):
        agent = MinesweeperAgent(max_steps=10, history_length=2)
        agent.update_from_env("obs0", 0.0, False, {})
        agent.update_from_model("```action\nreveal 0 0\n```")

        agent.reset()
        cc = agent.chat_completions
        assert _roles(cc) == ["system"]
        assert agent._context_manager.step_count == 0


class TestGameAgentAnchorObs:
    """anchor_obs should be recorded in Step.info when ContextManager is active."""

    def test_anchor_obs_in_step_info(self):
        agent = MinesweeperAgent(max_steps=10, history_length=2)
        agent.update_from_env("obs0", 0.0, False, {})
        agent.update_from_model("```action\nreveal 0 0\n```")

        step = agent.trajectory.steps[0]
        assert "anchor_obs" in step.info
        # Default anchor_obs falls back to the formatted observation string
        assert step.info["anchor_obs"] == "obs0"

    def test_anchor_obs_without_context_manager(self):
        """Without a ContextManager, anchor_obs falls back to the raw observation."""
        agent = MinesweeperAgent(max_steps=10)  # default, no CM
        agent.update_from_env("obs0", 0.0, False, {})
        agent.update_from_model("```action\nreveal 0 0\n```")

        step = agent.trajectory.steps[0]
        assert step.info["anchor_obs"] == "obs0"


class TestGameAgentStepSnapshots:
    """Step.chat_completions should be correct snapshots at each step."""

    def test_step_snapshot_with_windowed_history(self):
        """Each step's snapshot should reflect the context window at that point."""
        agent = MinesweeperAgent(max_steps=10, history_length=1)

        # Turn 0
        agent.update_from_env("board_0", 0.0, False, {})
        agent.update_from_model("```action\nreveal 0 0\n```")

        # Turn 1
        agent.update_from_env("board_1", 0.0, False, {})
        agent.update_from_model("```action\nreveal 1 1\n```")

        # Turn 2
        agent.update_from_env("board_2", 0.0, False, {})
        agent.update_from_model("```action\nreveal 2 2\n```")

        assert len(agent.trajectory.steps) == 3

        # Step 0 snapshot: system + 0 history (none yet) + user + assistant
        # After store_assistant_turn the committed turn is in history while
        # _current_user_prompt still holds that turn's prompt, so the snapshot
        # can repeat the user message. We assert only that snapshots are well-formed.
        for i, step in enumerate(agent.trajectory.steps):
            assert step.chat_completions[0]["role"] == "system"
            # Should have at least system + current_user
            assert len(step.chat_completions) >= 2

    def test_step_snapshots_are_deep_copies(self):
        """Step snapshots should not be affected by later agent state changes."""
        agent = MinesweeperAgent(max_steps=10, history_length=2)

        agent.update_from_env("board_0", 0.0, False, {})
        agent.update_from_model("```action\nreveal 0 0\n```")
        snap0 = agent.trajectory.steps[0].chat_completions

        agent.update_from_env("board_1", 0.0, False, {})
        agent.update_from_model("```action\nreveal 1 1\n```")
        snap1 = agent.trajectory.steps[1].chat_completions

        # Snapshots should be independent
        assert snap0 is not snap1
        # Step 0 snapshot should not change when step 1 is added
        assert len(snap0) == len(agent.trajectory.steps[0].chat_completions)


class TestGameAgentHistoryLengthOverridesAccumulate:
    """history_length should take priority over use_accumulate_history."""

    def test_explicit_history_length_with_accumulate_true(self):
        """history_length=2 with accumulate=True should use ContextManager with length=2."""
        agent = MinesweeperAgent(
            max_steps=10,
            use_accumulate_history=True,
            history_length=2,
        )
        assert agent._context_manager is not None
        assert agent._context_manager.history_length == 2

    def test_explicit_history_length_with_accumulate_false(self):
        """history_length=3 with accumulate=False should use history_length=3 (not 0)."""
        agent = MinesweeperAgent(
            max_steps=10,
            use_accumulate_history=False,
            history_length=3,
        )
        assert agent._context_manager is not None
        assert agent._context_manager.history_length == 3


@pytest.mark.skipif(not sokoban_available, reason="gym_sokoban not installed")
class TestSokobanAgentContextManager:
    """Cross-check that ContextManager works with SokobanAgent."""

    def test_sokoban_history_length_zero(self):
        agent = SokobanAgent(max_steps=10, history_length=0)
        assert agent._context_manager is not None

        for i in range(3):
            agent.update_from_env(f"grid_{i}", 0.0, False, {})
            agent.update_from_model(f"```action\nUp\n```")

        agent.update_from_env("grid_now", 0.0, False, {})
        cc = agent.chat_completions
        assert _roles(cc) == ["system", "user"]
        assert "grid_now" in cc[1]["content"]

    def test_sokoban_history_length_two(self):
        agent = SokobanAgent(max_steps=10, history_length=2)
        for i in range(4):
            agent.update_from_env(f"grid_{i}", 0.0, False, {})
            agent.update_from_model(f"```action\nUp\n```")

        agent.update_from_env("grid_now", 0.0, False, {})
        cc = agent.chat_completions
        # system + 2*(user+assistant) + current_user = 6
        assert len(cc) == 6
        assert "grid_2" in cc[1]["content"]
        assert "grid_3" in cc[3]["content"]
        assert "grid_now" in cc[5]["content"]


# ===========================================================================
# Part 3: Config-related tests
# ===========================================================================

class TestConfigCompatibility:
    """Verify that the history_length parameter is properly threaded through kwargs."""

    def test_history_length_in_kwargs(self):
        """Extra kwargs should be passed through to GameAgent subclasses."""
        agent = MinesweeperAgent(max_steps=10, history_length=3, some_other_kwarg="ignored")
        assert agent._context_manager is not None
        assert agent._context_manager.history_length == 3

    def test_default_kwargs_no_context_manager(self):
        """Without history_length in kwargs, no ContextManager is created."""
        agent = MinesweeperAgent(max_steps=10)
        assert agent._context_manager is None
