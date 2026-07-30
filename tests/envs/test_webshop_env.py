"""Tests for WebShop environment.

Note: These tests require the WebShop package and data to be installed.
Some tests are marked as skip if dependencies are not available.
"""

import pytest

# Try to import dependencies
try:
    import gym
    from bs4 import BeautifulSoup
    WEBSHOP_AVAILABLE = True
except ImportError:
    WEBSHOP_AVAILABLE = False

from rllm.environments.webshop.webshop_env import (
    RENDER_INSTRUCTIONS,
    WebShopEnv,
)


class TestWebShopEnvBasic:
    """Basic tests for WebShopEnv that don't require full initialization."""

    def test_init_without_lazy_init(self):
        """Test that environment can be initialized without triggering lazy init."""
        env = WebShopEnv(
            observation_mode="text",
            max_steps=50,
            seed=42,
        )
        assert env.max_steps == 50
        assert env.observation_mode == "text"
        assert env.seed == 42
        assert env._initialized is False

    def test_from_dict(self):
        """Test from_dict factory method."""
        env_args = {
            "observation_mode": "html",
            "max_steps": 100,
            "seed": 123,
            "session_id": 5,
            "extra_ignored_key": "ignored",
        }
        env = WebShopEnv.from_dict(env_args)
        assert env.observation_mode == "html"
        assert env.max_steps == 100
        assert env.seed == 123
        assert env.session_id == 5

    def test_is_multithread_safe(self):
        """Test that WebShop reports as thread-safe (uses Ray process isolation)."""
        assert WebShopEnv.is_multithread_safe() is True

    def test_repr(self):
        """Test string representation."""
        env = WebShopEnv(observation_mode="text_rich", max_steps=30)
        repr_str = repr(env)
        assert "WebShopEnv" in repr_str
        assert "observation_mode=text_rich" in repr_str
        assert "max_steps=30" in repr_str

    def test_observation_modes(self):
        """Test different observation modes can be set."""
        for mode in ["text", "html", "text_rich"]:
            env = WebShopEnv(observation_mode=mode)
            assert env.observation_mode == mode

    def test_render_instructions_constant(self):
        """Test that RENDER_INSTRUCTIONS is a non-empty string with tips."""
        assert isinstance(RENDER_INSTRUCTIONS, str)
        assert len(RENDER_INSTRUCTIONS) > 100
        assert "buy" in RENDER_INSTRUCTIONS.lower()
        assert "search" in RENDER_INSTRUCTIONS.lower()
        assert "click" in RENDER_INSTRUCTIONS.lower()

    def test_render_empty_before_reset(self):
        """Test render returns empty string before environment is reset."""
        env = WebShopEnv(observation_mode="text", max_steps=50, seed=42)
        assert env.render() == ""

    def test_success_property_default(self):
        """Test success property is False before any episode."""
        env = WebShopEnv(observation_mode="text", max_steps=50, seed=42)
        assert env.success is False

    def test_get_available_actions_formatted_empty(self):
        """Test get_available_actions_formatted returns empty list before reset."""
        env = WebShopEnv(observation_mode="text", max_steps=50, seed=42)
        result = env.get_available_actions_formatted()
        assert isinstance(result, list)
        assert len(result) == 0


@pytest.mark.skipif(not WEBSHOP_AVAILABLE, reason="WebShop dependencies not available")
class TestWebShopEnvFull:
    """Full tests that require WebShop to be installed."""

    @pytest.fixture
    def env(self):
        """Create a WebShop environment."""
        # Check if data files are available
        import os
        webshop_data = os.environ.get("WEBSHOP_DATA_PATH")
        if webshop_data:
            env = WebShopEnv(
                observation_mode="text",
                max_steps=50,
                seed=42,
                file_path=os.path.join(webshop_data, "all_products.json"),
                attr_path=os.path.join(webshop_data, "attrs"),
            )
        else:
            # Try default initialization
            try:
                env = WebShopEnv(observation_mode="text", max_steps=50, seed=42)
            except Exception:
                pytest.skip("WebShop data not available")
        yield env
        env.close()

    def test_reset(self, env):
        """Test environment reset."""
        obs, info = env.reset()
        assert isinstance(obs, str)
        assert isinstance(info, dict)
        assert "available_actions" in info
        assert "instruction" in info
        assert isinstance(info["instruction"], str)

    def test_step_search(self, env):
        """Test search action."""
        obs, info = env.reset()
        if info.get("available_actions", {}).get("has_search_bar", False):
            obs, reward, done, info = env.step("search[shoes]")
            assert isinstance(obs, str)
            assert isinstance(reward, (int, float))
            assert isinstance(done, bool)

    def test_step_click(self, env):
        """Test click action."""
        obs, info = env.reset()
        # First search for something
        if info.get("available_actions", {}).get("has_search_bar", False):
            obs, reward, done, info = env.step("search[shoes]")

            # Try to click on a clickable element
            clickables = info.get("available_actions", {}).get("clickables", [])
            if clickables:
                action = f"click[{clickables[0]}]"
                obs, reward, done, info = env.step(action)
                assert isinstance(obs, str)

    def test_available_actions(self, env):
        """Test that available actions are returned correctly."""
        obs, info = env.reset()
        available = env.get_available_actions()
        assert isinstance(available, dict)
        assert "has_search_bar" in available or "clickables" in available

    def test_instruction(self, env):
        """Test that instruction is returned."""
        obs, info = env.reset()
        instruction = env.get_instruction()
        assert isinstance(instruction, str)
        assert len(instruction) > 0

    def test_valid_action_check(self, env):
        """Test action validation."""
        obs, info = env.reset()
        # Search should be valid on start page
        assert env._is_valid_action("search[test]") == info.get("available_actions", {}).get("has_search_bar", False)

    def test_reset_with_seed(self, env):
        """Test reset accepts per-episode seed."""
        obs, info = env.reset(seed=123)
        assert isinstance(obs, str)
        assert isinstance(info, dict)

    def test_render_includes_instructions(self, env):
        """Test that render() returns observation + RENDER_INSTRUCTIONS + available actions."""
        obs, info = env.reset()
        rendered = env.render()
        assert isinstance(rendered, str)
        assert len(rendered) > 0
        # Should contain parts of the render instructions
        assert "buy" in rendered.lower()
        assert "search" in rendered.lower()
        # Should contain available actions section
        assert "You must choose from these actions" in rendered

    def test_get_available_actions_formatted(self, env):
        """Test formatted available actions list."""
        obs, info = env.reset()
        formatted = env.get_available_actions_formatted()
        assert isinstance(formatted, list)
        assert len(formatted) > 0
        # On start page, should have search action
        if info.get("available_actions", {}).get("has_search_bar", False):
            assert "search[<content>]" in formatted
        # 'search' should not appear as a click option
        for action in formatted:
            if action.startswith("click["):
                assert action != "click[search]"

    def test_step_info_enriched_fields(self, env):
        """Test that step() info dict contains all expected fields."""
        obs, info = env.reset()
        if info.get("available_actions", {}).get("has_search_bar", False):
            obs, reward, done, info = env.step("search[shoes]")
            # Check all expected fields exist
            assert "action_is_valid" in info
            assert "action_is_effective" in info
            assert "success" in info
            assert "end_of_page" in info
            assert "success_purchase" in info
            assert "raw_reward" in info
            assert "task_score" in info
            # Check types
            assert isinstance(info["action_is_valid"], bool)
            assert isinstance(info["action_is_effective"], bool)
            assert isinstance(info["success"], bool)
            assert isinstance(info["end_of_page"], bool)
            assert isinstance(info["success_purchase"], bool)

    def test_reward_is_one_not_ten(self, env):
        """Test that reward for success is 1.0, not 10.0 (aligned with RAGEN)."""
        obs, info = env.reset()
        # We can't guarantee a successful purchase in test,
        # but we can verify the reward is never 10.0 in a simple flow
        if info.get("available_actions", {}).get("has_search_bar", False):
            obs, reward, done, info = env.step("search[shoes]")
            # Intermediate steps should have reward 0.0
            assert reward == 0.0

    def test_success_property_after_reset(self, env):
        """Test success property is False after reset."""
        env.reset()
        assert env.success is False

    def test_action_effectiveness_detection(self, env):
        """Test that action_is_effective correctly detects state changes."""
        obs, info = env.reset()
        if info.get("available_actions", {}).get("has_search_bar", False):
            obs, reward, done, info = env.step("search[shoes]")
            # Search should be effective (changes the page)
            assert info["action_is_effective"] is True


if __name__ == "__main__":
    # Interactive test that runs a shopping episode
    import random

    print("=" * 60)
    print("WebShop Environment Interactive Test")
    print("=" * 60)

    # Check if WebShop is available
    if not WEBSHOP_AVAILABLE:
        print("ERROR: WebShop dependencies not available.")
        print("Please install gym, beautifulsoup4, flask, and torch.")
        exit(1)

    try:
        print("Initializing WebShop environment...")
        env = WebShopEnv(
            observation_mode="text",
            max_steps=20,
            seed=42,
        )

        obs, info = env.reset()
        print(f"\nInstruction: {info.get('instruction', 'unknown')}")
        print(f"\nInitial page:\n{obs[:500]}...")

        # Test render()
        rendered = env.render()
        print(f"\nRendered output (first 500 chars):\n{rendered[:500]}...")

        # Test get_available_actions_formatted()
        formatted_actions = env.get_available_actions_formatted()
        print(f"\nFormatted actions ({len(formatted_actions)}): {formatted_actions[:5]}...")

        available = info.get("available_actions", {})
        print(f"\nSearch bar available: {available.get('has_search_bar', False)}")
        clickables = available.get("clickables", [])
        print(f"Clickables ({len(clickables)}): {clickables[:5]}...")

        # Do a simple shopping flow
        total_reward = 0
        max_steps = 15
        step = 0

        # Step 1: Search for something from the instruction
        instruction = info.get("instruction", "")
        search_query = instruction.split()[:3]  # Use first 3 words
        search_action = f"search[{' '.join(search_query)}]"

        print(f"\n--- Step 1: {search_action} ---")
        obs, reward, done, info = env.step(search_action)
        total_reward += reward
        step = 1
        print(f"Reward: {reward}, Done: {done}")
        print(f"action_is_valid: {info.get('action_is_valid')}")
        print(f"action_is_effective: {info.get('action_is_effective')}")
        print(f"end_of_page: {info.get('end_of_page')}")
        print(f"Page: {obs[:300]}...")

        # Step 2+: Random clicks
        for step in range(2, max_steps + 1):
            if done:
                break

            available = info.get("available_actions", {})
            clickables = available.get("clickables", [])

            if not clickables:
                print("No clickable elements!")
                break

            # Prefer "buy now" if available, otherwise random click
            action = None
            for c in clickables:
                if "buy" in c.lower():
                    action = f"click[{c}]"
                    break
            if action is None:
                element = random.choice(clickables)
                action = f"click[{element}]"

            print(f"\n--- Step {step}: {action} ---")
            obs, reward, done, info = env.step(action)
            total_reward += reward
            print(f"Reward: {reward}, Done: {done}")
            print(f"  action_is_valid: {info.get('action_is_valid')}")
            print(f"  action_is_effective: {info.get('action_is_effective')}")
            print(f"  success_purchase: {info.get('success_purchase')}")
            print(f"  Task Score: {info.get('task_score', 0)}")

            if done:
                print("\nShopping completed!")
                break

        print("\n" + "=" * 60)
        print(f"Final Results:")
        print(f"  Total Reward: {total_reward}")
        print(f"  Task Score: {info.get('task_score', 0)}")
        print(f"  Won: {info.get('won', False)}")
        print(f"  Success: {env.success}")
        print(f"  Steps: {step}")
        print("=" * 60)

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'env' in locals():
            env.close()
