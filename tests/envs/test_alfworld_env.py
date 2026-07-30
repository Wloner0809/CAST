"""Tests for ALFWorld environment.

Note: These tests require the ALFWorld package and data to be installed.
Some tests are marked as skip if dependencies are not available.
"""

import pytest
import os
from pathlib import Path

# Set config path before imports
RLLM_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = RLLM_ROOT / "rllm" / "environments" / "alfworld" / "alfworld_pkg" / "configs" / "config_tw.yaml"
DEFAULT_DATA_PATH = RLLM_ROOT / "data" / "datasets" / "alfworld" / "eval_id.parquet"

os.environ["ALFWORLD_CONFIG_PATH"] = str(DEFAULT_CONFIG_PATH)

# Try to import dependencies
try:
    import textworld
    import yaml
    import pandas as pd
    ALFWORLD_AVAILABLE = True
except ImportError:
    ALFWORLD_AVAILABLE = False

from rllm.environments.alfworld.alfworld_env import ALFWorldEnv, TASK_TYPES, ALF_ACTION_LIST


class TestALFWorldEnvBasic:
    """Basic tests for ALFWorldEnv that don't require full initialization."""

    def test_task_types_defined(self):
        """Test that task types are properly defined."""
        assert len(TASK_TYPES) == 6
        assert 1 in TASK_TYPES
        assert TASK_TYPES[1] == "pick_and_place_simple"
        assert TASK_TYPES[6] == "pick_two_obj_and_place"

    def test_action_list_defined(self):
        """Test that action list is properly defined."""
        assert "goto" in ALF_ACTION_LIST
        assert "pick" in ALF_ACTION_LIST
        assert "put" in ALF_ACTION_LIST
        assert "open" in ALF_ACTION_LIST
        assert "close" in ALF_ACTION_LIST
        assert "heat" in ALF_ACTION_LIST
        assert "cool" in ALF_ACTION_LIST
        assert "clean" in ALF_ACTION_LIST

    def test_init_without_lazy_init(self):
        """Test that environment can be initialized without triggering lazy init."""
        env = ALFWorldEnv(
            config_path="/nonexistent/path.yaml",
            max_steps=50,
            train_eval="train",
        )
        assert env.max_steps == 50
        assert env.train_eval == "train"
        assert env._initialized is False

    def test_from_dict(self):
        """Test from_dict factory method."""
        env_args = {
            "config_path": "/some/path.yaml",
            "max_steps": 100,
            "train_eval": "eval_in_distribution",
            "task_types": [1, 2, 3],
            "extra_ignored_key": "ignored",
        }
        env = ALFWorldEnv.from_dict(env_args)
        assert env.config_path == "/some/path.yaml"
        assert env.max_steps == 100
        assert env.train_eval == "eval_in_distribution"
        assert env.task_types == [1, 2, 3]

    def test_is_multithread_safe(self):
        """Test that ALFWorld reports as thread-safe (via Ray process isolation)."""
        assert ALFWorldEnv.is_multithread_safe() is True

    def test_repr(self):
        """Test string representation."""
        env = ALFWorldEnv(max_steps=30, train_eval="eval_out_of_distribution")
        repr_str = repr(env)
        assert "ALFWorldEnv" in repr_str
        assert "max_steps=30" in repr_str
        assert "eval_out_of_distribution" in repr_str

    def test_tmpdir_configuration(self, tmp_path):
        """Test that tmpdir helper configures tempfile + env vars."""
        import tempfile

        from rllm.utils.tmpdir import configure_tmpdir

        keys = ["RLLM_TMPDIR", "TMPDIR", "TEMP", "TMP"]
        original_env = {k: os.environ.get(k) for k in keys}
        original_tempdir = tempfile.tempdir
        try:
            for k in keys:
                os.environ.pop(k, None)
            tempfile.tempdir = None

            choice = configure_tmpdir(preferred=str(tmp_path))
            assert choice.path == str(tmp_path)
            assert os.environ.get("TMPDIR") == str(tmp_path)
            assert os.environ.get("TEMP") == str(tmp_path)
            assert os.environ.get("TMP") == str(tmp_path)

            with tempfile.NamedTemporaryFile() as f:
                assert Path(f.name).resolve().is_relative_to(tmp_path.resolve())
        finally:
            for k, v in original_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            tempfile.tempdir = original_tempdir

    def test_cleanup_old_tmpdirs_only_removes_tempfile_dirs(self, tmp_path):
        """Test cleanup_old_tmpdirs only deletes `tempfile`-style tmp* dirs."""
        from rllm.utils.tmpdir import cleanup_old_tmpdirs

        to_delete = tmp_path / "tmpabc123"
        to_keep = tmp_path / "keep_me"
        to_delete.mkdir()
        to_keep.mkdir()

        # Make them "old"
        os.utime(to_delete, (0, 0))
        os.utime(to_keep, (0, 0))

        dirs_removed, _, removed_paths = cleanup_old_tmpdirs(
            base_path=str(tmp_path),
            max_age_seconds=0,
            include_common_fallbacks=False,
        )

        assert dirs_removed == 1
        assert str(to_delete) in removed_paths
        assert not to_delete.exists()
        assert to_keep.exists()


def _check_data_available():
    """Check if ALFWorld data is available."""
    return DEFAULT_DATA_PATH.exists()


def _load_sample_game_file():
    """Load a sample game file from the dataset."""
    if not _check_data_available():
        return None
    try:
        df = pd.read_parquet(DEFAULT_DATA_PATH)
        if len(df) > 0:
            game_file = df.iloc[0]["game_file"]
            if os.path.exists(game_file):
                return game_file
    except Exception:
        pass
    return None


@pytest.mark.skipif(not ALFWORLD_AVAILABLE, reason="ALFWorld dependencies not available")
class TestALFWorldEnvFull:
    """Full tests that require ALFWorld to be installed."""

    @pytest.fixture
    def config_path(self):
        """Get the ALFWorld config path."""
        config = os.environ.get("ALFWORLD_CONFIG_PATH")
        if config is None:
            pytest.skip("ALFWORLD_CONFIG_PATH environment variable not set")
        if not os.path.exists(config):
            pytest.skip(f"Config file not found: {config}")
        return config

    @pytest.fixture
    def game_file(self):
        """Get a valid game file from the dataset."""
        game_file = _load_sample_game_file()
        if game_file is None:
            pytest.skip("No valid game file available. Please ensure ALFWorld data is installed.")
        return game_file

    def test_reset(self, config_path, game_file):
        """Test environment reset."""
        env = ALFWorldEnv(config_path=config_path, max_steps=50, game_file=game_file)
        try:
            obs, info = env.reset()
            assert isinstance(obs, str)
            assert isinstance(info, dict)
            assert "admissible_commands" in info
            assert isinstance(info["admissible_commands"], list)
            assert info.get("game_file") == game_file
            assert "tmp_dir" in info
            assert info["tmp_dir"] is None or os.path.isdir(info["tmp_dir"])
        finally:
            env.close()

    def test_step(self, config_path, game_file):
        """Test environment step."""
        env = ALFWorldEnv(config_path=config_path, max_steps=50, game_file=game_file)
        try:
            obs, info = env.reset()
            admissible = info.get("admissible_commands", [])

            if admissible:
                action = admissible[0]
                obs, reward, done, info = env.step(action)
                assert isinstance(obs, str)
                assert isinstance(reward, (int, float))
                assert isinstance(done, bool)
                assert isinstance(info, dict)
        finally:
            env.close()

    def test_admissible_commands(self, config_path, game_file):
        """Test that admissible commands are returned."""
        env = ALFWorldEnv(config_path=config_path, max_steps=50, game_file=game_file)
        try:
            obs, info = env.reset()
            commands = env.get_admissible_commands()
            assert isinstance(commands, list)
            assert len(commands) > 0
        finally:
            env.close()

    def test_reset_with_seed(self, config_path, game_file):
        """Test reset accepts per-episode seed."""
        env = ALFWorldEnv(config_path=config_path, max_steps=50, game_file=game_file)
        try:
            obs, info = env.reset(seed=123)
            assert isinstance(obs, str)
            assert isinstance(info, dict)
        finally:
            env.close()

    def test_game_file_in_info(self, config_path, game_file):
        """Test that game_file is included in info dict."""
        env = ALFWorldEnv(config_path=config_path, max_steps=50, game_file=game_file)
        try:
            obs, info = env.reset()
            assert isinstance(obs, str)
            assert info.get("game_file") == game_file
            assert "task_type" in info
        finally:
            env.close()

    def test_max_steps_termination(self, config_path, game_file):
        """Test that environment terminates at max_steps."""
        max_steps = 5
        env = ALFWorldEnv(config_path=config_path, max_steps=max_steps, game_file=game_file)
        try:
            obs, info = env.reset()
            done = False
            step_count = 0

            while not done and step_count < max_steps + 5:
                admissible = info.get("admissible_commands", ["look"])
                action = admissible[0] if admissible else "look"
                obs, reward, done, info = env.step(action)
                step_count += 1

            # Should have terminated at or before max_steps
            assert step_count <= max_steps
            assert done or step_count == max_steps
        finally:
            env.close()


if __name__ == "__main__":
    # Interactive test that runs a few random actions
    import random

    print("=" * 60)
    print("ALFWorld Environment Interactive Test")
    print("=" * 60)

    # Check if ALFWorld is available
    if not ALFWORLD_AVAILABLE:
        print("ERROR: ALFWorld dependencies not available.")
        print("Please install textworld and other required packages.")
        exit(1)

    # Get config path
    config_path = os.environ.get("ALFWORLD_CONFIG_PATH", str(DEFAULT_CONFIG_PATH))
    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found: {config_path}")
        exit(1)

    print(f"Config path: {config_path}")

    # Get game file from dataset
    game_file = _load_sample_game_file()
    if game_file is None:
        print("ERROR: No valid game file found in dataset.")
        print(f"Please ensure data exists at: {DEFAULT_DATA_PATH}")
        exit(1)

    print(f"Game file: {game_file}")

    try:
        env = ALFWorldEnv(
            config_path=config_path,
            max_steps=30,
            train_eval="eval_in_distribution",
            game_file=game_file,
        )

        obs, info = env.reset()
        print(f"\nTask type: {info.get('task_type', 'unknown')}")
        print(f"\nInitial observation:\n{obs}")
        print(f"\nAdmissible commands ({len(info.get('admissible_commands', []))}):")
        for cmd in info.get("admissible_commands", [])[:10]:
            print(f"  - {cmd}")

        # Do some random actions
        num_steps = 10
        total_reward = 0

        for step in range(num_steps):
            admissible = info.get("admissible_commands", ["look"])
            action = random.choice(admissible) if admissible else "look"

            print(f"\n--- Step {step + 1}: {action} ---")
            obs, reward, done, info = env.step(action)
            total_reward += reward

            print(f"Reward: {reward}, Done: {done}")
            print(f"Observation: {obs[:200]}...")

            if done:
                print("\nEpisode finished!")
                break

        print("\n" + "=" * 60)
        print(f"Final Results:")
        print(f"  Total Reward: {total_reward}")
        print(f"  Won: {info.get('won', False)}")
        print(f"  Steps: {step + 1}")
        print("=" * 60)

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'env' in locals():
            env.close()
        # Cleanup: Shutdown the worker pool
        print("\nShutting down ALFWorld worker pool...")
        ALFWorldEnv.shutdown_pool()
        print("Done!")
