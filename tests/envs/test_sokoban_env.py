import pytest

pytest.importorskip("gym_sokoban")
pytest.importorskip("gym")

from rllm.environments.sokoban.sokoban import SokobanEnv


class TestSokobanEnv:
    """Test suite for SokobanEnv class."""

    def test_init_default(self):
        env = SokobanEnv(seed=42)
        assert env.config.dim_room == (6, 6)
        assert env.config.num_boxes == 3
        assert env.seed == 42
        assert hasattr(env, "ACTION_SPACE")

    def test_action_constants(self):
        env = SokobanEnv()
        assert env.INVALID_ACTION == 0
        assert env.ACTION_LOOKUP[1] == "Up"
        assert env.ACTION_LOOKUP[2] == "Down"
        assert env.ACTION_LOOKUP[3] == "Left"
        assert env.ACTION_LOOKUP[4] == "Right"

    def test_reset(self):
        env = SokobanEnv(seed=123)
        obs, info = env.reset()
        assert isinstance(obs, str)
        assert isinstance(info, dict)

    def test_step_invalid_action(self):
        env = SokobanEnv(seed=123)
        env.reset()
        obs, reward, done, info = env.step(0)
        assert isinstance(obs, str)
        assert reward == 0
        assert done is False
        assert info["action_is_valid"] is False

    def test_step_valid_action(self):
        env = SokobanEnv(seed=123)
        env.reset()
        obs, reward, done, info = env.step(1)
        assert isinstance(obs, str)
        assert isinstance(reward, int | float)
        assert isinstance(done, bool)
        assert info["action_is_valid"] is True

    def test_step_text_action(self):
        env = SokobanEnv(seed=123)
        env.reset()
        obs, reward, done, info = env.step("Up")
        assert isinstance(obs, str)
        assert isinstance(reward, int | float)
        assert isinstance(done, bool)
        assert info["action_is_valid"] is True

    def test_step_invalid_text_action(self):
        env = SokobanEnv(seed=123)
        env.reset()
        obs, reward, done, info = env.step("Jump")
        assert isinstance(obs, str)
        assert reward == 0
        assert done is False
        assert info["action_is_valid"] is False

    def test_render_modes(self):
        env = SokobanEnv(seed=123)
        env.reset()
        assert isinstance(env.render("grid"), str)
        assert isinstance(env.render("coord"), str)
        assert isinstance(env.render("grid_coord"), str)

    def test_from_dict(self):
        env_info = {"seed": 1, "dim_x": 6, "dim_y": 6, "num_boxes": 1, "max_steps": 50}
        env = SokobanEnv.from_dict(env_info)
        assert env.config.dim_room == (6, 6)
        assert env.config.num_boxes == 1
        assert env.config.max_steps == 50
        assert env.seed == 1

    def test_step_multi_reports_executed_count(self):
        env = SokobanEnv(seed=42, dim_room=(6, 6), num_boxes=1, max_steps=1)
        env.reset()
        _, _, done, info = env.step(["1", "2", "3", "4"])
        assert done is True
        assert info["actions_executed"] == 1

    def test_step_multi_requires_a_list(self):
        """A "||"-joined string is a single invalid action, not a multi-action."""
        env = SokobanEnv(seed=42, dim_room=(6, 6), num_boxes=1, max_steps=1)
        env.reset()
        _, _, _, info = env.step("1||2||3||4")
        assert info["action_is_valid"] is False
        assert "actions_executed" not in info


if __name__ == "__main__":
    # random generate some maps and print them, then random select one and try do some random actions on it and print the env and final reward
    import random

    num_maps = 5
    num_actions = 10

    print("=" * 50)
    print(f"Generating {num_maps} random Sokoban maps...")
    print("=" * 50)

    envs = []
    for i in range(num_maps):
        seed = random.randint(0, 10000)
        env = SokobanEnv(seed=seed, dim_room=(6, 6), num_boxes=2)
        obs, _ = env.reset()
        envs.append(env)
        print(f"\n--- Map {i + 1} (seed={seed}) ---")
        print(obs)

    # Randomly select one map
    selected_idx = random.randint(0, len(envs) - 1)
    env = envs[selected_idx]
    print("\n" + "=" * 50)
    print(f"Selected Map {selected_idx + 1} for random actions")
    print("=" * 50)

    # Reset the selected environment to start fresh
    obs, _ = env.reset()
    print("\nInitial State:")
    print(obs)

    # Perform random actions
    total_reward = 0
    actions = env.get_all_actions()  # [1, 2, 3, 4] for Up, Down, Left, Right

    for step in range(num_actions):
        action = random.choice(actions)
        action_name = env.ACTION_LOOKUP[action]
        obs, reward, done, info = env.step(action)
        total_reward += reward

        print(f"\n--- Step {step + 1}: Action = {action_name} ({action}) ---")
        print(f"Reward: {reward}, Done: {done}, Effective: {info['action_is_effective']}")
        print(obs)

        if done:
            print("\nEnvironment finished!")
            break

    print("\n" + "=" * 50)
    print(f"Final Results:")
    print(f"  Total Reward: {total_reward}")
    print(f"  Success: {env.success()}")
    print(f"  Steps Taken: {step + 1}")
    print("=" * 50)
