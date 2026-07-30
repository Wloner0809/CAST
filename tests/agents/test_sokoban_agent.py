import pytest

pytest.importorskip("gym_sokoban")
pytest.importorskip("gym")

from rllm.agents.sokoban_agent import SokobanAgent
from rllm.agents.agent import Trajectory
from rllm.agents.action_utils import parse_action_block


class TestSokobanAgent:
    """Test suite for SokobanAgent (GameAgent-based) core functionality."""

    def test_init_default(self):
        agent = SokobanAgent()
        assert isinstance(agent._trajectory, Trajectory)
        assert len(agent.messages) == 1
        assert agent.step == 0
        assert "Sokoban Quick Guide" in agent.messages[0]["content"]

    def test_update_from_env_basic(self):
        agent = SokobanAgent()
        observation = "#####\n#P_X#\n#_O_#\n#####"
        agent.update_from_env(observation, 0.0, False, {})
        assert agent.messages[-1]["role"] == "user"
        assert observation in agent.messages[-1]["content"]

    def test_update_from_model_basic(self):
        agent = SokobanAgent()
        agent.update_from_env("P", 0.0, False, {})
        response = "I should move right. ```action\nRight\n```"
        action = agent.update_from_model(response)
        assert action.action == "4"
        assert agent._trajectory.steps[-1].action == "4"

    def test_parse_action_block_and_parse_action(self):
        """Test parse_action_block + agent._parse_action for direction mapping."""
        agent = SokobanAgent()
        test_cases = [
            ("Move ```action\nUp\n```", "Move", "1"),
            ("Move ```action\nDown\n```", "Move", "2"),
            ("Move ```action\nLeft\n```", "Move", "3"),
            ("Move ```action\nRight\n```", "Move", "4"),
            ("No action block here", "No action block here", "0"),
        ]
        for response, expected_thought, expected_action in test_cases:
            thought, raw_action = parse_action_block(response)
            assert thought == expected_thought
            if raw_action is not None:
                action = agent._parse_action(raw_action)
            else:
                action = agent._parse_action("")
            assert action == expected_action

    def test_update_from_env_ineffective_action_message(self):
        agent = SokobanAgent()
        agent.update_from_env("obs0", 0.0, False, {})
        agent.update_from_model("Try ```action\nUp\n```")

        # Simulate env feedback for a valid but ineffective move.
        # Note: ineffective feedback requires observation to match the previous one.
        agent.update_from_env(
            "obs0",
            0.0,
            False,
            {"action_is_valid": True, "action_is_effective": False},
        )
        # SokobanAgent._ineffective_action_feedback returns:
        # "Your last action was valid but ineffective (the state did not change).
        #  Choose a different direction that can move the player or push a box."
        assert "different direction" in agent.messages[-1]["content"]
        assert "Recheck the action format" not in agent.messages[-1]["content"]

    def test_update_from_env_invalid_action_message(self):
        agent = SokobanAgent()
        agent.update_from_env("obs0", 0.0, False, {})
        agent.update_from_model("Try ```action\nUp\n```")

        # Simulate env feedback for an invalid action.
        agent.update_from_env(
            "obs0",
            0.0,
            False,
            {"action_is_valid": False, "action_is_effective": False},
        )
        # SokobanAgent._invalid_action_feedback returns:
        # "Your last response was invalid. Recheck the action format.
        #  You should only output the NEXT ACTION in ``` ```,
        #  and it must be one of Up/Down/Left/Right."
        assert "Recheck the action format" in agent.messages[-1]["content"]

    def test_parse_action_multistep(self):
        """Test multi-step action parsing with || separator."""
        agent = SokobanAgent(use_multistep_prompt=True)
        result = agent._parse_action("Up || Down || Right")
        assert result == "1||2||4"

    def test_parse_action_invalid_direction(self):
        """Test that invalid direction text returns INVALID_ACTION (0)."""
        agent = SokobanAgent()
        assert agent._parse_action("jump") == "0"
        assert agent._parse_action("") == "0"

    def test_parse_action_numeric_input(self):
        """Test that numeric action ids are accepted."""
        agent = SokobanAgent()
        assert agent._parse_action("1") == "1"
        assert agent._parse_action("4") == "4"
        assert agent._parse_action("9") == "0"  # invalid numeric

    def test_system_prompt_uses_action_blocks(self):
        """Verify the system prompt instructs the model to use ``` blocks."""
        agent = SokobanAgent()
        system_msg = agent.messages[0]["content"]
        assert "```" in system_msg

    def test_multistep_system_prompt(self):
        """Verify multi-step prompt mentions || separator."""
        agent = SokobanAgent(use_multistep_prompt=True)
        system_msg = agent.messages[0]["content"]
        assert "||" in system_msg

    def test_update_from_model_no_action_block(self):
        """When model response has no action block, action should be INVALID_ACTION."""
        agent = SokobanAgent()
        agent.update_from_env("P", 0.0, False, {})
        response = "I'm not sure what to do."
        action = agent.update_from_model(response)
        assert action.action == "0"

    def test_generic_block_fallback(self):
        """Test that generic ``` ... ``` blocks also work (backward compat)."""
        agent = SokobanAgent()
        agent.update_from_env("P", 0.0, False, {})
        response = "I should go up. ```Up```"
        action = agent.update_from_model(response)
        assert action.action == "1"

    def test_observation_section_includes_step_number(self):
        """Verify observation prompt includes step number."""
        agent = SokobanAgent()
        agent.update_from_env("obs_text", 0.0, False, {})
        user_msg = agent.messages[-1]["content"]
        assert "Current Observation (Step 0):" in user_msg

    def test_remaining_steps_shown(self):
        """When max_steps is set, remaining steps should appear in prompt."""
        agent = SokobanAgent(max_steps=10)
        agent.update_from_env("obs", 0.0, False, {})
        user_msg = agent.messages[-1]["content"]
        assert "You have 10 steps remaining." in user_msg

    def test_step_counter_increments(self):
        """Verify step counter increments after update_from_model."""
        agent = SokobanAgent()
        assert agent.step == 0
        agent.update_from_env("obs", 0.0, False, {})
        agent.update_from_model("```action\nUp\n```")
        assert agent.step == 1
        agent.update_from_env("obs2", 0.0, False, {})
        agent.update_from_model("```action\nDown\n```")
        assert agent.step == 2


if __name__ == "__main__":
    # implement here a test case to test the sokoban agent, run agent on a random sokoban map
    pass
