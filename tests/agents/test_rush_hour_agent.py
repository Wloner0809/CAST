"""Tests for RushHourAgent (GameAgent-based) action parsing.

Focuses on the ``.search`` -> ``.fullmatch`` regression: the action extracted
from the ``` block must be a single clean move, and prose / multi-token blocks
must NOT be silently turned into a (possibly wrong) board move.
"""

from rllm.agents.rush_hour_agent import RushHourAgent
from rllm.agents.agent import Trajectory
from rllm.agents.action_utils import parse_action_block


class TestRushHourAgent:
    def test_init_default(self):
        agent = RushHourAgent()
        assert isinstance(agent._trajectory, Trajectory)
        assert len(agent.messages) == 1
        assert agent.step == 0
        assert "Rush Hour Quick Guide" in agent.messages[0]["content"]

    def test_clean_action_is_parsed(self):
        agent = RushHourAgent()
        agent.update_from_env("obs", 0.0, False, {})
        action = agent.update_from_model("I'll slide A right. ```\nA+2\n```")
        assert action.action == "A+2"
        assert agent._trajectory.steps[-1].action == "A+2"

    def test_parse_action_variants(self):
        agent = RushHourAgent()
        # (block content) -> expected canonical action ("" means rejected)
        cases = [
            ("A+2", "A+2"),
            ("B-1", "B-1"),
            ("[A+]", "A+1"),          # bracket form, implicit magnitude 1
            ("  c+3  ", "C+3"),        # surrounding whitespace ok
            ("A+0", ""),              # zero magnitude rejected
            # Prose / multi-token blocks must be rejected, not silently parsed.
            ("move car b-1 now", ""),
            ("I'd move b-1 but actually A+2", ""),
            ("Let me slide A: A+2", ""),
            ("first do c then maybe d+1", ""),
        ]
        for block, expected in cases:
            assert agent._parse_action(block) == expected, block

    def test_prose_block_yields_empty_action(self):
        """A ``` block full of prose must not become a real move."""
        agent = RushHourAgent()
        agent.update_from_env("obs", 0.0, False, {})
        action = agent.update_from_model("hmm ```\nI think I'll move car b-1 now\n```")
        assert action.action == ""
