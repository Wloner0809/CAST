"""Tests for ALFWorldAgent: initialization, prompt construction, action-block parsing, and trajectory building."""

import pytest

from rllm.agents.alfworld_agent import ALFWorldAgent
from rllm.agents.agent import Trajectory, Action
from rllm.agents.action_utils import parse_action_block


class TestALFWorldAgent:
    """Test suite for ALFWorldAgent."""

    def test_init_default(self):
        """Test default initialization."""
        agent = ALFWorldAgent()
        assert isinstance(agent._trajectory, Trajectory)
        assert len(agent.messages) == 1
        assert agent.messages[0]["role"] == "system"
        assert agent.step == 0
        assert agent.use_admissible_commands is True
        assert agent.use_available_actions is True
        assert agent.accumulate_history is True

    def test_init_with_params(self):
        """Test initialization with custom parameters."""
        agent = ALFWorldAgent(
            max_steps=100,
            use_admissible_commands=False,
            use_accumulate_history=False,
        )
        assert agent.max_steps == 100
        assert agent.use_admissible_commands is False
        assert agent.use_available_actions is False
        assert agent.accumulate_history is False

    def test_init_use_available_actions_overrides(self):
        """Test that use_available_actions takes precedence over use_admissible_commands."""
        agent = ALFWorldAgent(
            use_admissible_commands=True,
            use_available_actions=False,
        )
        assert agent.use_available_actions is False
        assert agent.use_admissible_commands is False

    def test_system_prompt_content(self):
        """Test that system prompt contains key information."""
        agent = ALFWorldAgent()
        system_prompt = agent.SYSTEM_PROMPT
        assert "go to" in system_prompt
        assert "take" in system_prompt
        assert "move" in system_prompt or "put" in system_prompt
        assert "open" in system_prompt
        assert "action" in system_prompt.lower()

    def test_update_from_env_basic(self):
        """Test basic environment update."""
        agent = ALFWorldAgent()
        observation = "You are in the middle of a room."
        info = {
            "admissible_commands": ["go to kitchen", "look around"],
            "task_type": "pick_and_place_simple",
        }
        agent.update_from_env(observation, 0.0, False, info)

        assert agent.messages[-1]["role"] == "user"
        assert observation in agent.messages[-1]["content"]
        assert "Current Observation (Step 0):" in agent.messages[-1]["content"]
        assert agent.current_observation == observation

    def test_update_from_env_with_admissible_commands(self):
        """Test that admissible commands are included in prompt."""
        agent = ALFWorldAgent(use_admissible_commands=True)
        observation = "You are in the kitchen."
        info = {
            "admissible_commands": ["open fridge", "go to counter"],
            "task_type": "pick_heat_then_place_in_recep",
        }
        agent.update_from_env(observation, 0.0, False, info)

        user_prompt = agent.messages[-1]["content"]
        assert "open fridge" in user_prompt
        assert "Admissible" in user_prompt

    def test_update_from_env_without_admissible_commands(self):
        """Test that admissible commands are NOT included when disabled."""
        agent = ALFWorldAgent(use_admissible_commands=False)
        observation = "You are in the kitchen."
        info = {
            "admissible_commands": ["open fridge", "go to counter"],
        }
        agent.update_from_env(observation, 0.0, False, info)

        user_prompt = agent.messages[-1]["content"]
        assert "Admissible" not in user_prompt

    def test_update_from_model_basic(self):
        """Test basic model response processing."""
        agent = ALFWorldAgent()
        agent.update_from_env("Room observation", 0.0, False, {})

        response = "I should look around first.\n```action\nlook\n```"
        action = agent.update_from_model(response)

        assert isinstance(action, Action)
        assert action.action == "look"
        assert len(agent._trajectory.steps) == 1
        assert agent._trajectory.steps[-1].action == "look"

    def test_parse_action_block_with_action_tag(self):
        """Test parsing action from ```action ... ``` blocks using parse_action_block + _parse_action."""
        agent = ALFWorldAgent()
        test_cases = [
            ("Let me go.\n```action\ngo to kitchen\n```", "go to kitchen"),
            ("I'll open it.\n```action\nopen fridge 1\n```", "open fridge 1"),
            ("Thinking...\n```action\ntake apple from counter\n```", "take apple from counter"),
        ]
        for response, expected_action in test_cases:
            thought, raw_action = parse_action_block(response)
            assert raw_action is not None, f"Failed to parse action block from: {response}"
            action_str = agent._parse_action(raw_action)
            assert action_str == expected_action, f"Expected '{expected_action}', got '{action_str}'"

    def test_parse_action_block_generic_block(self):
        """Test parsing action from generic ``` ``` blocks."""
        agent = ALFWorldAgent()
        response = "Let me check.\n```\nlook around\n```"
        thought, raw_action = parse_action_block(response)
        assert raw_action is not None
        action_str = agent._parse_action(raw_action)
        assert action_str == "look around"

    def test_parse_action_block_no_block(self):
        """Test parsing when no action block is found returns empty string."""
        agent = ALFWorldAgent()
        response = "I need to think about this more."
        thought, raw_action = parse_action_block(response)
        # No block found -> raw_action is None
        assert raw_action is None
        # _parse_action("") returns "" (no fallback to "look")
        action_str = agent._parse_action("")
        assert action_str == ""

    def test_parse_action_strips_action_prefix(self):
        """Test that _parse_action strips 'action' prefix and ':' from text."""
        agent = ALFWorldAgent()
        assert agent._parse_action("action go to kitchen") == "go to kitchen"
        assert agent._parse_action("Action: go to fridge 1") == "go to fridge 1"
        assert agent._parse_action(": open drawer 1") == "open drawer 1"
        assert agent._parse_action("go to table 1") == "go to table 1"

    def test_update_from_model_no_action_block(self):
        """Test update_from_model with no action block returns empty action."""
        agent = ALFWorldAgent()
        agent.update_from_env("Room observation", 0.0, False, {})

        response = "I need to think more carefully."
        action = agent.update_from_model(response)

        assert isinstance(action, Action)
        assert action.action == ""

    def test_chat_completions_with_history(self):
        """Test chat completions with history accumulation."""
        agent = ALFWorldAgent(use_accumulate_history=True)
        agent.update_from_env("Obs 1", 0.0, False, {})
        agent.update_from_model("```action\nlook\n```")
        agent.update_from_env("Obs 2", 0.0, False, {})

        completions = agent.chat_completions
        # System + user + assistant + user = 4
        assert len(completions) >= 3

    def test_chat_completions_without_history(self):
        """Test chat completions without history accumulation."""
        agent = ALFWorldAgent(use_accumulate_history=False)
        agent.update_from_env("Obs 1", 0.0, False, {})
        agent.update_from_model("```action\nlook\n```")
        agent.update_from_env("Obs 2", 0.0, False, {})

        completions = agent.chat_completions
        # Should only have system + last user
        assert len(completions) == 2
        assert completions[0]["role"] == "system"
        assert completions[1]["role"] == "user"

    def test_reset(self):
        """Test agent reset."""
        agent = ALFWorldAgent()
        agent.update_from_env("Observation", 0.0, False, {})
        agent.update_from_model("```action\nlook\n```")

        agent.reset()

        assert len(agent.messages) == 1
        assert agent.step == 0
        assert len(agent._trajectory.steps) == 0
        assert agent.current_observation is None

    def test_trajectory_building(self):
        """Test that trajectory is properly built."""
        agent = ALFWorldAgent()

        # First turn
        agent.update_from_env("Room 1", 0.0, False, {"task_type": "test"})
        agent.update_from_model("```action\ngo to kitchen\n```")

        # Second turn with reward
        agent.update_from_env("Kitchen", 5.0, False, {})
        agent.update_from_model("```action\nopen fridge\n```")

        # Check trajectory
        assert len(agent._trajectory.steps) == 2
        assert agent._trajectory.steps[0].action == "go to kitchen"
        assert agent._trajectory.steps[0].reward == 5.0  # Updated by second env update
        assert agent._trajectory.steps[1].action == "open fridge"

    def test_get_current_state(self):
        """Test getting current state."""
        agent = ALFWorldAgent()
        assert agent.get_current_state() is None

        agent.update_from_env("Test", 0.0, False, {})
        agent.update_from_model("```action\nlook\n```")

        state = agent.get_current_state()
        assert state is not None
        assert state.action == "look"

    def test_nothing_happens_feedback(self):
        """Test that 'Nothing happens' observation triggers ALFWorld-specific feedback."""
        agent = ALFWorldAgent()

        # First step: normal action
        agent.update_from_env("Room 1", 0.0, False, {})
        agent.update_from_model("```action\nopen drawer 1\n```")

        # Second step: "Nothing happens" observation
        agent.update_from_env("Nothing happens.", 0.0, False, {})
        user_prompt = agent.messages[-1]["content"]
        assert "had no effect" in user_prompt or "Nothing happens" in user_prompt

    def test_remaining_steps_section(self):
        """Test that remaining steps are shown in the prompt."""
        agent = ALFWorldAgent(max_steps=10)
        agent.update_from_env("Room 1", 0.0, False, {})
        user_prompt = agent.messages[-1]["content"]
        assert "10 steps remaining" in user_prompt

    def test_step_counter_increments(self):
        """Test that step counter increments after update_from_model."""
        agent = ALFWorldAgent()
        assert agent.step == 0

        agent.update_from_env("Obs", 0.0, False, {})
        agent.update_from_model("```action\nlook\n```")
        assert agent.step == 1

        agent.update_from_env("Obs 2", 0.0, False, {})
        agent.update_from_model("```action\ngo to desk 1\n```")
        assert agent.step == 2

    def test_admissible_commands_property_setter(self):
        """Test that use_admissible_commands setter updates use_available_actions."""
        agent = ALFWorldAgent(use_admissible_commands=True)
        assert agent.use_admissible_commands is True
        assert agent.use_available_actions is True

        agent.use_admissible_commands = False
        assert agent.use_admissible_commands is False
        assert agent.use_available_actions is False


if __name__ == "__main__":
    # Interactive test with mock environment
    print("=" * 60)
    print("ALFWorld Agent Unit Test")
    print("=" * 60)

    agent = ALFWorldAgent(max_steps=10)

    # Simulate a simple interaction
    observations = [
        "You are in the middle of a room. Looking quickly around you, you see a bed 1, a desk 1, and a sidetable 1.",
        "You arrive at the desk 1. On the desk 1, you see a book 1 and a pencil 1.",
        "You pick up the book 1 from the desk 1.",
    ]

    infos = [
        {"admissible_commands": ["go to desk 1", "go to bed 1", "look"], "task_type": "pick_and_place"},
        {"admissible_commands": ["take book 1", "take pencil 1", "look"], "task_type": "pick_and_place"},
        {"admissible_commands": ["go to sidetable 1", "put book 1 in/on bed 1"], "task_type": "pick_and_place"},
    ]

    responses = [
        "I need to find a book. Let me check the desk first.\n```action\ngo to desk 1\n```",
        "Found the book! Let me pick it up.\n```action\ntake book 1 from desk 1\n```",
        "Now I should put the book somewhere.\n```action\nput book 1 in/on bed 1\n```",
    ]

    print("\nSimulating agent interaction:")
    print("-" * 60)

    for i, (obs, info, resp) in enumerate(zip(observations, infos, responses)):
        print(f"\n[Step {i+1}]")
        print(f"Observation: {obs[:80]}...")

        agent.update_from_env(obs, 0.0 if i < 2 else 10.0, i == 2, info)
        action = agent.update_from_model(resp)

        print(f"Agent action: {action.action}")
        print(f"Trajectory steps: {len(agent._trajectory.steps)}")

    print("\n" + "=" * 60)
    print("Final Trajectory:")
    for i, step in enumerate(agent._trajectory.steps):
        print(f"  Step {i+1}: action={step.action}, reward={step.reward}")
    print("=" * 60)
