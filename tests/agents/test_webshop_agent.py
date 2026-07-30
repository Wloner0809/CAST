"""Tests for WebShop agent."""

import pytest

from rllm.agents.webshop_agent import WebShopAgent
from rllm.agents.agent import Trajectory, Action


class TestWebShopAgent:
    """Test suite for WebShopAgent."""

    def test_init_default(self):
        """Test default initialization."""
        agent = WebShopAgent()
        assert isinstance(agent._trajectory, Trajectory)
        assert len(agent.messages) == 1
        assert agent.messages[0]["role"] == "system"
        assert agent.step == 0
        assert agent.use_available_actions is True
        assert agent.accumulate_history is True

    def test_init_with_params(self):
        """Test initialization with custom parameters."""
        agent = WebShopAgent(
            max_steps=100,
            use_available_actions=False,
            use_accumulate_history=False,
        )
        assert agent.max_steps == 100
        assert agent.use_available_actions is False
        assert agent.accumulate_history is False

    def test_system_prompt_content(self):
        """Test that system prompt contains key information."""
        agent = WebShopAgent()
        system_prompt = agent.SYSTEM_PROMPT
        assert "search" in system_prompt.lower()
        assert "click" in system_prompt.lower()
        assert "Buy Now" in system_prompt or "buy" in system_prompt.lower()
        assert "action" in system_prompt.lower()

    def test_update_from_env_first_turn(self):
        """Test first turn includes instruction."""
        agent = WebShopAgent()
        observation = "Welcome to WebShop. Search for products."
        info = {
            "instruction": "Find me a red shirt size medium",
            "available_actions": {"has_search_bar": True, "clickables": []},
        }
        agent.update_from_env(observation, 0.0, False, info)

        user_prompt = agent.messages[-1]["content"]
        assert "Shopping Task" in user_prompt or "red shirt" in user_prompt
        assert agent.current_instruction == "Find me a red shirt size medium"

    def test_update_from_env_with_available_actions(self):
        """Test that available actions are included in prompt."""
        agent = WebShopAgent(use_available_actions=True)
        observation = "Search results page"
        info = {
            "instruction": "Find a blue jacket",
            "available_actions": {
                "has_search_bar": True,
                "clickables": ["Product 1", "Product 2", "Next"],
            },
        }
        agent.update_from_env(observation, 0.0, False, info)

        user_prompt = agent.messages[-1]["content"]
        assert "search" in user_prompt.lower() or "Available" in user_prompt

    def test_update_from_model_search(self):
        """Test processing search action."""
        agent = WebShopAgent()
        agent.update_from_env("Page", 0.0, False, {"instruction": "test", "available_actions": {}})

        response = "I should search for the product.\n```action\nsearch[red shoes]\n```"
        action = agent.update_from_model(response)

        assert isinstance(action, Action)
        assert action.action == "search[red shoes]"

    def test_update_from_model_click(self):
        """Test processing click action."""
        agent = WebShopAgent()
        agent.update_from_env("Page", 0.0, False, {"instruction": "test", "available_actions": {}})

        response = "I'll click on this product.\n```action\nclick[Product 1]\n```"
        action = agent.update_from_model(response)

        assert action.action == "click[Product 1]"

    def test_parse_action_well_formed(self):
        """Test _parse_action with well-formed search[...] and click[...] strings."""
        agent = WebShopAgent()
        test_cases = [
            ("search[blue jacket]", "search[blue jacket]"),
            ("click[Buy Now]", "click[Buy Now]"),
            ("click[back to search]", "click[back to search]"),
        ]
        for raw_text, expected_action in test_cases:
            assert agent._parse_action(raw_text) == expected_action

    def test_parse_action_extracts_from_text(self):
        """Test _parse_action extracts action patterns from surrounding text."""
        agent = WebShopAgent()
        assert agent._parse_action("I want to search[laptop] now") == "search[laptop]"
        assert agent._parse_action("Let me click[Product 1] here") == "click[Product 1]"

    def test_parse_action_empty_on_failure(self):
        """Test _parse_action returns empty string on failure."""
        agent = WebShopAgent()
        assert agent._parse_action("") == ""
        assert agent._parse_action("no valid action here") == ""

    def test_action_block_parsing(self):
        """Test parsing action from ```action ... ``` blocks via update_from_model."""
        agent = WebShopAgent()
        agent.update_from_env("Page", 0.0, False, {"instruction": "test", "available_actions": {}})

        response = "Search first.\n```action\nsearch[blue jacket]\n```"
        action = agent.update_from_model(response)
        assert action.action == "search[blue jacket]"

    def test_generic_block_parsing(self):
        """Test parsing action from generic ``` ``` blocks via update_from_model."""
        agent = WebShopAgent()
        agent.update_from_env("Page", 0.0, False, {"instruction": "test", "available_actions": {}})

        response = "Let me search.\n```\nsearch[running shoes]\n```"
        action = agent.update_from_model(response)
        assert action.action == "search[running shoes]"

    def test_chat_completions_with_history(self):
        """Test chat completions with history accumulation."""
        agent = WebShopAgent(use_accumulate_history=True)
        agent.update_from_env("Page 1", 0.0, False, {"instruction": "test", "available_actions": {}})
        agent.update_from_model("```action\nsearch[test]\n```")
        agent.update_from_env("Page 2", 0.0, False, {"instruction": "test", "available_actions": {}})

        completions = agent.chat_completions
        assert len(completions) >= 3

    def test_chat_completions_without_history(self):
        """Test chat completions without history accumulation."""
        agent = WebShopAgent(use_accumulate_history=False)
        agent.update_from_env("Page 1", 0.0, False, {"instruction": "test", "available_actions": {}})
        agent.update_from_model("```action\nsearch[test]\n```")
        agent.update_from_env("Page 2", 0.0, False, {"instruction": "test", "available_actions": {}})

        completions = agent.chat_completions
        # Should only have system + last user
        assert len(completions) == 2
        assert completions[0]["role"] == "system"
        assert completions[1]["role"] == "user"

    def test_reset(self):
        """Test agent reset."""
        agent = WebShopAgent()
        agent.update_from_env("Page", 0.0, False, {"instruction": "test", "available_actions": {}})
        agent.update_from_model("```action\nsearch[test]\n```")

        agent.reset()

        assert len(agent.messages) == 1
        assert agent.step == 0
        assert len(agent._trajectory.steps) == 0
        assert agent.current_observation is None
        assert agent.current_instruction == ""

    def test_trajectory_building(self):
        """Test that trajectory is properly built."""
        agent = WebShopAgent()

        # First turn - search
        agent.update_from_env("Start page", 0.0, False, {"instruction": "find shoes", "available_actions": {}})
        agent.update_from_model("```action\nsearch[shoes]\n```")

        # Second turn - click
        agent.update_from_env("Results page", 0.0, False, {"instruction": "find shoes", "available_actions": {}})
        agent.update_from_model("```action\nclick[Product 1]\n```")

        # Third turn with reward
        agent.update_from_env("Product page", 10.0, True, {"instruction": "find shoes", "available_actions": {}})

        # Check trajectory
        assert len(agent._trajectory.steps) == 2
        assert agent._trajectory.steps[0].action == "search[shoes]"
        assert agent._trajectory.steps[1].action == "click[Product 1]"
        assert agent._trajectory.steps[1].reward == 10.0

    def test_get_current_state(self):
        """Test getting current state."""
        agent = WebShopAgent()
        assert agent.get_current_state() is None

        agent.update_from_env("Test", 0.0, False, {"instruction": "test", "available_actions": {}})
        agent.update_from_model("```action\nsearch[test]\n```")

        state = agent.get_current_state()
        assert state is not None
        assert state.action == "search[test]"


if __name__ == "__main__":
    # Interactive test with mock environment
    print("=" * 60)
    print("WebShop Agent Unit Test")
    print("=" * 60)

    agent = WebShopAgent(max_steps=10)

    # Simulate a shopping episode
    observations = [
        "WebShop [SEP] Instruction: Find me running shoes size 10 [SEP] [button] Search [button]",
        "Results [SEP] Product 1: Nike Running Shoes $50 [SEP] Product 2: Adidas Runners $45 [SEP] [button] Next [button]",
        "Product Details [SEP] Nike Running Shoes [SEP] Size: 9, 10, 11 [SEP] Color: Black, White [SEP] [button] Buy Now [button]",
        "Thank you for your purchase!",
    ]

    infos = [
        {"instruction": "Find me running shoes size 10", "available_actions": {"has_search_bar": True, "clickables": []}},
        {"instruction": "Find me running shoes size 10", "available_actions": {"has_search_bar": False, "clickables": ["Product 1", "Product 2", "Next"]}},
        {"instruction": "Find me running shoes size 10", "available_actions": {"has_search_bar": False, "clickables": ["10", "Black", "Buy Now"]}},
        {"instruction": "Find me running shoes size 10", "available_actions": {"has_search_bar": False, "clickables": []}, "won": True},
    ]

    responses = [
        "I need to search for running shoes.\n```action\nsearch[running shoes size 10]\n```",
        "Let me look at the Nike shoes.\n```action\nclick[Product 1]\n```",
        "I'll select size 10 and buy.\n```action\nclick[Buy Now]\n```",
    ]

    print("\nSimulating shopping episode:")
    print("-" * 60)

    for i in range(len(responses)):
        obs, info = observations[i], infos[i]
        print(f"\n[Step {i+1}]")
        print(f"Page: {obs[:60]}...")

        reward = 10.0 if i == len(responses) - 1 else 0.0
        done = i == len(responses) - 1

        agent.update_from_env(obs, reward, done, info)
        action = agent.update_from_model(responses[i])

        print(f"Agent action: {action.action}")
        print(f"Trajectory steps: {len(agent._trajectory.steps)}")

    print("\n" + "=" * 60)
    print("Final Trajectory:")
    for i, step in enumerate(agent._trajectory.steps):
        print(f"  Step {i+1}: action={step.action}, reward={step.reward}")
    print("=" * 60)
