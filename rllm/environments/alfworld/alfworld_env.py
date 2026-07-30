"""ALFWorld environment wrapper for rllm framework.

ALFWorld is a text-based game environment for household tasks.
This module provides a single-environment wrapper around the ALFWorld TextWorld environment.
Uses Ray remote actors for process-level isolation to ensure thread safety.
Workers are pooled and reused to avoid expensive re-initialization.
"""

import asyncio
import logging
import os
import re
import shutil
import sys
import threading
import uuid
from typing import ClassVar

import ray

from rllm.environments.base.base_env import BaseEnv
from rllm.utils.tmpdir import (
    cleanup_old_tmpdirs,
    cleanup_stale_tmp_files,
    PeriodicTmpCleaner,
    configure_tmpdir,
)

logger = logging.getLogger(__name__)

# Hardcoded target path for ALFWorld temp files (shared filesystem)
ALFWORLD_TMP_TARGET = os.path.abspath(
    os.environ.get("ALFWORLD_TMP_TARGET", "./outputs/alfworld_tmp")
)

# Task type mapping
TASK_TYPES = {
    1: "pick_and_place_simple",
    2: "look_at_obj_in_light",
    3: "pick_clean_then_place_in_recep",
    4: "pick_heat_then_place_in_recep",
    5: "pick_cool_then_place_in_recep",
    6: "pick_two_obj_and_place",
}

# Available actions in ALFWorld
ALF_ACTION_LIST = [
    "goto",
    "pick",
    "put",
    "open",
    "close",
    "toggle",
    "heat",
    "clean",
    "cool",
    "slice",
    "inventory",
    "examine",
    "look",
]


@ray.remote
class ALFWorldWorker:
    """Ray remote actor that holds one ALFWorld environment instance.

    Each worker runs in its own process, providing process-level isolation
    to ensure thread safety for the async execution engine.

    Workers are pooled by (config_path, train_eval, task_types) and can be
    reused across different game files by calling reset() with game_file.

    IMPORTANT: We do NOT pre-collect game files during init because:
    1. The game_file is passed explicitly from the dataset at reset() time
    2. collect_game_files() is expensive (walks entire directory, parses JSONs)
    3. We lazily initialize the TextWorld env on first reset with the specific game
    """

    def __init__(
        self,
        config_path: str | None = None,
        train_eval: str = "train",
        task_types: list[int] | None = None,
        max_episode_steps: int | None = None,
        seed: int | None = None,
        tmp_dir: str | None = None,
    ):
        """Initialize the worker with ALFWorld configuration (lazy env init).

        Args:
            config_path: Path to ALFWorld config YAML file.
            train_eval: Data split to use.
            task_types: List of task type IDs (1-6) to include.
            max_episode_steps: Override the underlying TextWorld max_episode_steps.
            seed: Random seed for reproducibility.
        """
        import yaml

        # Add parent directory of alfworld_pkg to path
        pkg_parent_path = os.path.dirname(__file__)
        if pkg_parent_path not in sys.path:
            sys.path.insert(0, pkg_parent_path)

        enforced_tmp_dir = tmp_dir or "/tmp/rllm_tmp"
        if not os.path.abspath(enforced_tmp_dir).startswith("/tmp"):
            raise RuntimeError(
                f"ALFWorld tmpdir must be under /tmp. Got: {enforced_tmp_dir}"
            )

        # Create symlink to shared filesystem (idempotent)
        # This ensures all Ray workers on all nodes can access the same temp directory
        target_path = ALFWORLD_TMP_TARGET
        try:
            os.makedirs(target_path, exist_ok=True)
        except OSError as e:
            logger.warning(f"Failed to create target directory {target_path}: {e}")
            raise RuntimeError(
                f"Cannot create ALFWorld tmp target directory: {target_path}. "
                f"Ensure the shared filesystem is accessible. Error: {e}"
            )

        # Create or update the symlink (atomic, handles race conditions)
        self._create_symlink_atomic(target_path, enforced_tmp_dir)

        # Verify the symlink is working by checking if we can access the target
        if not os.path.isdir(enforced_tmp_dir):
            raise RuntimeError(
                f"Symlink verification failed: {enforced_tmp_dir} is not accessible as a directory. "
                f"Target: {target_path}, exists: {os.path.exists(target_path)}"
            )

        # Use min_free_inodes=0 because the shared network filesystem doesn't
        # support traditional inode counting and reports 0 for f_favail
        tmp_choice = configure_tmpdir(preferred=enforced_tmp_dir, min_free_inodes=0)
        if os.path.abspath(tmp_choice.path) != os.path.abspath(enforced_tmp_dir):
            raise RuntimeError(
                f"ALFWorld tmpdir enforcement failed. "
                f"Expected {enforced_tmp_dir}, got {tmp_choice.path} ({tmp_choice.reason})"
            )
        self._tmp_dir = tmp_choice.path
        self._tmp_cleaner = PeriodicTmpCleaner(
            interval_seconds=30,
            max_age_seconds=300,
        )
        try:
            self._tmp_cleaner.maybe_cleanup(force=True)
        except Exception as e:
            logger.warning(f"Failed initial tmp cleanup: {e}")

        # Store config for lazy initialization
        self._config_path = config_path
        self._train_eval = train_eval
        self._task_types = task_types
        self._max_episode_steps = max_episode_steps
        self._seed = seed

        # Load config
        if config_path is None:
            default_config = os.path.join(
                os.path.dirname(__file__),
                "alfworld_pkg",
                "configs",
                "base_config.yaml",
            )
            if os.path.exists(default_config):
                config_path = default_config
            else:
                raise FileNotFoundError(
                    f"No config file provided and default config not found at {default_config}."
                )

        with open(config_path) as f:
            self._config = yaml.safe_load(f)

        # Override task types in config
        if task_types:
            self._config["env"]["task_types"] = task_types

        # Lazy initialization - env created on first reset()
        self._env = None
        self._current_game_file = None
        self._admissible_commands = []
        self._logged_tmp_dir = False

        logger.info(
            f"ALFWorldWorker initialized (lazy mode), split={train_eval}, tmp_dir={self._tmp_dir} ({tmp_choice.reason})"
        )

    def _create_symlink_atomic(self, target_path: str, link_path: str, max_retries: int = 3):
        """Create symlink atomically, handling race conditions from concurrent workers.

        Uses try-first approach instead of check-then-act to avoid TOCTOU races.
        Multiple Ray workers may try to create the same symlink simultaneously.
        """
        for attempt in range(max_retries):
            try:
                os.symlink(target_path, link_path)
                logger.info(f"Created symlink {link_path} -> {target_path}")
                return
            except FileExistsError:
                # Something exists at link_path - check what it is
                if os.path.islink(link_path):
                    current_target = os.path.normpath(os.readlink(link_path))
                    expected_target = os.path.normpath(target_path)
                    if current_target == expected_target:
                        # Another worker created the correct symlink - success!
                        logger.debug(f"Symlink {link_path} already exists with correct target")
                        return
                    else:
                        # Wrong target - try to remove and recreate
                        logger.warning(
                            f"Symlink {link_path} points to {current_target}, "
                            f"expected {expected_target}, updating..."
                        )
                        try:
                            os.unlink(link_path)
                        except FileNotFoundError:
                            pass  # Another worker removed it
                elif os.path.exists(link_path):
                    # Regular file/directory - remove it
                    logger.warning(f"Removing existing path {link_path} to create symlink")
                    try:
                        if os.path.isdir(link_path):
                            shutil.rmtree(link_path)
                        else:
                            os.unlink(link_path)
                    except FileNotFoundError:
                        pass  # Another worker removed it
                # Retry after cleanup
                continue
            except OSError as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Symlink creation attempt {attempt + 1} failed: {e}")
                    continue
                raise

        # Final verification - if symlink exists with correct target, we're good
        if os.path.islink(link_path):
            current_target = os.path.normpath(os.readlink(link_path))
            expected_target = os.path.normpath(target_path)
            if current_target == expected_target:
                return

        raise RuntimeError(
            f"Failed to create symlink {link_path} -> {target_path} after {max_retries} attempts"
        )

    def _reconfigure_tmpdir_after_no_space(self):
        """Attempt to recover from ENOSPC by aggressively cleaning stale temp files."""
        # First, try to clean up old temp directories to free space.
        try:
            dirs_removed, bytes_freed, _ = cleanup_old_tmpdirs(
                base_path=self._tmp_dir,
                max_age_seconds=30,
                include_common_fallbacks=True,
            )
            files_removed, file_bytes_freed = cleanup_stale_tmp_files(
                max_age_seconds=30,
            )
            if dirs_removed > 0 or files_removed > 0:
                logger.info(
                    f"ENOSPC recovery: cleaned {dirs_removed} dirs + {files_removed} files, "
                    f"freed {(bytes_freed + file_bytes_freed) / 1024 / 1024:.1f}MB"
                )
        except Exception as e:
            logger.warning(f"Failed to cleanup during ENOSPC recovery: {e}")
        # Re-apply tmpdir to ensure `tempfile` is configured consistently.
        try:
            tmp_choice = configure_tmpdir(preferred=self._tmp_dir)
            if os.path.abspath(tmp_choice.path) != os.path.abspath(self._tmp_dir):
                raise RuntimeError(
                    f"ALFWorld tmpdir enforcement failed after ENOSPC cleanup. "
                    f"Expected {self._tmp_dir}, got {tmp_choice.path} ({tmp_choice.reason})"
                )
        except Exception as e:
            logger.warning(f"Failed to re-apply tmpdir after ENOSPC cleanup: {e}")

    def _init_env_for_game(self, game_file: str):
        """Initialize TextWorld environment for a specific game file.

        This is called lazily on reset() to avoid expensive collect_game_files().
        """
        import textworld
        import textworld.gym

        from alfworld_pkg.agents.environment.alfred_tw_env import (
            AlfredDemangler,
            AlfredInfos,
        )

        if not os.path.exists(game_file):
            raise FileNotFoundError(f"Game file not found: {game_file}")

        domain_randomization = self._config["env"]["domain_randomization"]
        if self._train_eval != "train":
            domain_randomization = False

        alfred_demangler = AlfredDemangler(shuffle=domain_randomization)
        wrappers = [alfred_demangler, AlfredInfos]

        request_infos = textworld.EnvInfos(
            won=True,
            admissible_commands=True,
            extras=["gamefile"]
        )

        # Get max steps from config
        training_method = self._config["general"]["training_method"]
        if training_method == "dqn":
            max_nb_steps = self._config["rl"]["training"]["max_nb_steps_per_episode"]
        elif training_method == "dagger":
            max_nb_steps = self._config["dagger"]["training"]["max_nb_steps_per_episode"]
        else:
            max_nb_steps = 100  # Default fallback

        # Allow overriding the underlying TextWorld max_episode_steps.
        # This is important because rllm's eval config may set max_steps=100,
        # while alfworld_pkg configs default to 50.
        if self._max_episode_steps is not None:
            max_nb_steps = int(self._max_episode_steps)

        # Register ONLY the specific game file (not all games)
        env_id = textworld.gym.register_games(
            [game_file],  # Single game file!
            request_infos,
            batch_size=1,
            asynchronous=True,
            max_episode_steps=max_nb_steps,
            wrappers=wrappers,
        )

        # Create the gym environment
        self._env = textworld.gym.make(env_id)

        if self._seed is not None:
            try:
                self._env.seed(self._seed)
            except Exception as e:
                logger.warning(f"Failed to set seed: {e}")

        self._current_game_file = game_file
        logger.debug(f"Initialized TextWorld env for game: {game_file}")

    def _is_enospc_error(self, e: Exception) -> bool:
        """Check if an exception is an ENOSPC (no space left on device) error."""
        if isinstance(e, OSError) and getattr(e, "errno", None) == 28:
            return True
        # Check nested exceptions
        if hasattr(e, "__cause__") and e.__cause__ is not None:
            return self._is_enospc_error(e.__cause__)
        return False

    def _reset_with_enospc_recovery(
        self, game_file: str, seed: int | None, max_retries: int = 2
    ) -> tuple:
        """Reset environment with ENOSPC recovery logic.

        Args:
            game_file: Game file to load.
            seed: Random seed.
            max_retries: Maximum number of recovery attempts.

        Returns:
            Tuple of (obs, infos) from the environment reset.
        """
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                # Re-initialize env if game_file changed or first call
                if self._env is None or self._current_game_file != game_file:
                    if self._env is not None:
                        try:
                            self._env.close()
                        except Exception:
                            pass
                    self._init_env_for_game(game_file)

                if seed is not None:
                    try:
                        self._env.seed(seed)
                    except Exception as e:
                        logger.warning(f"Failed to set ALFWorld seed={seed}: {e}")

                return self._env.reset()

            except OSError as e:
                if not self._is_enospc_error(e):
                    raise

                last_error = e
                if attempt < max_retries:
                    logger.warning(
                        f"ENOSPC error on attempt {attempt + 1}/{max_retries + 1}, "
                        f"reconfiguring tmpdir and retrying..."
                    )
                    self._reconfigure_tmpdir_after_no_space()
                    # Clean up current env state
                    if self._env is not None:
                        try:
                            self._env.close()
                        except Exception:
                            pass
                    self._env = None
                    self._current_game_file = None
                else:
                    logger.error(
                        f"ENOSPC error persists after {max_retries + 1} attempts. "
                        f"Current tmpdir: {self._tmp_dir}"
                    )

        raise OSError(
            f"Failed to reset ALFWorld environment after {max_retries + 1} attempts "
            f"due to ENOSPC. Last tmpdir: {self._tmp_dir}"
        ) from last_error

    def reset(self, seed: int | None = None, game_file: str | None = None) -> tuple[str, dict]:
        """Reset the environment to start a new episode.

        Args:
            seed: Random seed for this episode.
            game_file: Specific game file to load. Required for training.

        Returns:
            Tuple of (observation, info).
        """
        if game_file is None:
            raise ValueError(
                "game_file is required for ALFWorldWorker.reset(). "
                "Pass the game_file from the dataset task."
            )

        try:
            self._tmp_cleaner.maybe_cleanup()
        except Exception as e:
            logger.warning(f"Failed periodic tmp cleanup: {e}")

        if not self._logged_tmp_dir:
            logger.info(f"ALFWorldWorker reset using tmp_dir={self._tmp_dir}")
            self._logged_tmp_dir = True

        obs, infos = self._reset_with_enospc_recovery(game_file, seed)

        # Extract observation (batch size is 1)
        observation = obs[0] if isinstance(obs, (list, tuple)) else obs

        # Process info dict
        for key in infos:
            if isinstance(infos[key], list) and len(infos[key]) == 1:
                infos[key] = infos[key][0]

        self._admissible_commands = infos.get("admissible_commands", [])

        info = {
            "admissible_commands": self._admissible_commands,
            "game_file": self._current_game_file,
            "tmp_dir": self._tmp_dir,
        }

        return observation, info
    
    def step(self, action: str) -> tuple[str, float, bool, dict]:
        """Execute an action in the environment."""
        obs, scores, dones, infos = self._env.step([action])
        
        # Extract from batch (size 1)
        observation = obs[0] if isinstance(obs, (list, tuple)) else obs
        done = dones[0] if isinstance(dones, (list, tuple)) else dones
        done = bool(done)
        
        # Process info dict
        for key in infos:
            if isinstance(infos[key], list) and len(infos[key]) == 1:
                infos[key] = infos[key][0]
        
        self._admissible_commands = infos.get("admissible_commands", [])
        won = infos.get("won", False)
        
        # Compute reward: 10.0 if won, 0.0 otherwise
        reward = 10.0 if won else 0.0
        
        # Detect ineffective actions: "Nothing happens" is ALFWorld's signal
        action_is_effective = "nothing happens" not in observation.lower() if isinstance(observation, str) else True

        info = {
            "admissible_commands": self._admissible_commands,
            "available_actions": self._admissible_commands,
            "game_file": self._current_game_file,
            "won": won,
            "action_is_valid": action.lower() in [cmd.lower() for cmd in self._admissible_commands],
            "action_is_effective": action_is_effective,
            "tmp_dir": self._tmp_dir,
        }
        
        return observation, reward, done, info
    
    def get_admissible_commands(self) -> list[str]:
        """Get the list of valid commands for the current state."""
        return self._admissible_commands
    
    def close(self):
        """Clean up environment resources (called when worker is destroyed)."""
        if self._env is not None:
            try:
                self._env.close()
            except Exception as e:
                logger.warning(f"Error closing ALFWorld environment in worker: {e}")
            self._env = None


@ray.remote
class ALFWorldWorkerPoolActor:
    """A Ray actor that manages a pool of ALFWorld workers.

    This actor provides centralized coordination of workers across all processes,
    ensuring that workers are properly reused and the total number is limited.

    Also performs periodic cleanup of temp directories to prevent disk exhaustion.
    """

    def __init__(self, max_workers: int = 64, cleanup_interval_seconds: float = 300):
        """Initialize the worker pool.

        Args:
            max_workers: Maximum number of workers to create per configuration.
            cleanup_interval_seconds: Interval between automatic temp cleanups.
        """
        self._max_workers = max_workers
        self._pools: dict[str, list] = {}  # key -> list of available workers
        self._all_workers: dict[str, list] = {}  # key -> list of all workers (keeps owner refs)
        self._worker_counts: dict[str, int] = {}  # track total workers created per config
        self._pending_requests: dict[str, list] = {}  # key -> list of waiting events
        self._pending_count: dict[str, int] = {}
        self._lock = threading.Lock()

        # Cleanup tracking
        self._cleanup_interval = cleanup_interval_seconds
        self._last_cleanup_time = 0.0
        self._release_count = 0
        self._cleanup_stats = {"dirs_removed": 0, "bytes_freed": 0}

        logger.info(f"ALFWorldWorkerPoolActor initialized with max_workers={max_workers}")
    
    def _get_pool_key(
        self,
        config_path: str | None,
        train_eval: str,
        task_types: list[int] | None,
        max_episode_steps: int | None,
    ) -> str:
        """Generate a unique key for the worker pool based on configuration.

        Note: game_file is NOT part of the key because workers can handle any game_file
        via reset(). This allows worker reuse across different tasks.
        """
        task_types_str = ",".join(map(str, sorted(task_types))) if task_types else "all"
        steps_str = str(max_episode_steps) if max_episode_steps is not None else "default"
        return f"{config_path}|{train_eval}|{task_types_str}|steps={steps_str}"
    
    async def acquire_worker(
        self,
        config_path: str | None,
        train_eval: str,
        task_types: list[int] | None,
        max_episode_steps: int | None,
        seed: int | None,
        timeout: float = 600.0,
    ):
        """Acquire a worker from the pool, queuing if necessary.

        Args:
            config_path: Path to ALFWorld config YAML file.
            train_eval: Data split to use.
            task_types: List of task type IDs to include.
            seed: Random seed (not used for pooling).
            timeout: Maximum time to wait for a worker.

        Returns:
            A Ray actor reference to an ALFWorldWorker, or None if at max capacity.
        """
        pool_key = self._get_pool_key(config_path, train_eval, task_types, max_episode_steps)

        with self._lock:
            # Initialize pool structures if needed
            if pool_key not in self._pools:
                self._pools[pool_key] = []
                self._all_workers[pool_key] = []
                self._worker_counts[pool_key] = 0
                self._pending_requests[pool_key] = []
                self._pending_count[pool_key] = 0

            # Try to get a worker from the pool
            if self._pools[pool_key]:
                worker = self._pools[pool_key].pop()
                return worker

            # Check if we can create a new worker
            if self._worker_counts[pool_key] < self._max_workers:
                worker = ALFWorldWorker.remote(
                    config_path=config_path,
                    train_eval=train_eval,
                    task_types=task_types,
                    max_episode_steps=max_episode_steps,
                    seed=seed,
                    tmp_dir=os.environ.get("RLLM_TMPDIR"),
                )

                self._worker_counts[pool_key] += 1
                self._all_workers[pool_key].append(worker)
                count = self._worker_counts[pool_key]
                logger.info(f"Created ALFWorldWorker {count}/{self._max_workers} for config: {pool_key[:50]}...")
                return worker

            # At capacity - queue the request and wait
            request_id = str(uuid.uuid4())
            event = asyncio.Event()
            result_holder = {"worker": None}
            request_info = {
                "request_id": request_id,
                "event": event,
                "result": result_holder,
            }
            self._pending_requests[pool_key].append(request_info)
            self._pending_count[pool_key] += 1
            queue_position = self._pending_count[pool_key]
            logger.debug(
                f"Request {request_id[:8]} queued at position {queue_position} "
                f"for config {pool_key[:30]}... (workers: {self._worker_counts[pool_key]}/{self._max_workers})"
            )

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
            if result_holder["worker"] is None:
                raise RuntimeError(f"Request {request_id[:8]} was signaled but no worker assigned")
            return result_holder["worker"]
        except asyncio.TimeoutError:
            with self._lock:
                self._pending_requests[pool_key] = [
                    r for r in self._pending_requests[pool_key]
                    if r["request_id"] != request_id
                ]
            raise TimeoutError(
                f"Timeout waiting for ALFWorld worker after {timeout}s. "
                f"Queue depth was {queue_position}, max_workers={self._max_workers}"
            )
    
    def release_worker(
        self,
        worker,
        config_path: str | None,
        train_eval: str,
        task_types: list[int] | None,
        max_episode_steps: int | None,
    ):
        """Return a worker to the pool for reuse."""
        pool_key = self._get_pool_key(config_path, train_eval, task_types, max_episode_steps)

        with self._lock:
            if pool_key not in self._pools:
                self._pools[pool_key] = []
                self._all_workers[pool_key] = []
                self._pending_requests[pool_key] = []
                self._pending_count[pool_key] = 0

            # Check if there are pending requests waiting for a worker
            if self._pending_requests[pool_key]:
                request_info = self._pending_requests[pool_key].pop(0)
                request_info["result"]["worker"] = worker
                request_info["event"].set()
                logger.debug(
                    f"Fulfilled pending request {request_info['request_id'][:8]} "
                    f"({len(self._pending_requests[pool_key])} still waiting)"
                )
            else:
                self._pools[pool_key].append(worker)

            # Track releases and trigger periodic cleanup
            self._release_count += 1

        # Trigger periodic cleanup (outside lock to avoid blocking)
        self._maybe_cleanup()

    def _maybe_cleanup(self):
        """Run cleanup if enough time has passed since the last run."""
        import time

        current_time = time.time()
        elapsed = current_time - self._last_cleanup_time

        if elapsed < self._cleanup_interval:
            return

        self._last_cleanup_time = current_time

        try:
            # Clean up old temp directories (older than 30 minutes)
            dirs_removed, bytes_freed, _ = cleanup_old_tmpdirs(
                max_age_seconds=1800,
            )
            # Also clean stale files
            files_removed, file_bytes_freed = cleanup_stale_tmp_files(
                max_age_seconds=1800,
            )

            total_freed = bytes_freed + file_bytes_freed
            if dirs_removed > 0 or files_removed > 0:
                self._cleanup_stats["dirs_removed"] += dirs_removed
                self._cleanup_stats["bytes_freed"] += total_freed
                logger.info(
                    f"Periodic cleanup: removed {dirs_removed} dirs + {files_removed} files, "
                    f"freed {total_freed / 1024 / 1024:.1f}MB "
                    f"(total: {self._cleanup_stats['bytes_freed'] / 1024 / 1024:.1f}MB)"
                )
        except Exception as e:
            logger.warning(f"Periodic cleanup failed: {e}")

    def trigger_cleanup(self, max_age_seconds: float = 600) -> dict:
        """Manually trigger a cleanup operation.

        Args:
            max_age_seconds: Maximum age of directories to keep.

        Returns:
            Dictionary with cleanup statistics.
        """
        try:
            dirs_removed, bytes_freed, removed_paths = cleanup_old_tmpdirs(
                max_age_seconds=max_age_seconds,
            )
            files_removed, file_bytes_freed = cleanup_stale_tmp_files(
                max_age_seconds=max_age_seconds,
            )

            total_freed = bytes_freed + file_bytes_freed
            self._cleanup_stats["dirs_removed"] += dirs_removed
            self._cleanup_stats["bytes_freed"] += total_freed

            return {
                "dirs_removed": dirs_removed,
                "files_removed": files_removed,
                "bytes_freed": total_freed,
                "mb_freed": total_freed / 1024 / 1024,
                "removed_paths": removed_paths[:10],  # Limit to first 10
            }
        except Exception as e:
            logger.warning(f"Manual cleanup failed: {e}")
            return {"error": str(e)}
    
    def get_stats(self) -> dict:
        """Get statistics about the worker pool."""
        return {
            "max_workers": self._max_workers,
            "worker_counts": dict(self._worker_counts),
            "available_workers": {k: len(v) for k, v in self._pools.items()},
            "total_workers": {k: len(v) for k, v in self._all_workers.items()},
            "pending_requests": {k: len(v) for k, v in self._pending_requests.items()},
            "release_count": self._release_count,
            "cleanup_stats": dict(self._cleanup_stats),
        }
    
    def shutdown(self):
        """Shutdown all workers in all pools."""
        # Cancel pending requests
        for pool_key, requests in self._pending_requests.items():
            for request_info in requests:
                request_info["event"].set()
            self._pending_requests[pool_key] = []

        for pool_key, workers in self._all_workers.items():
            for worker in workers:
                try:
                    ray.get(worker.close.remote())
                    ray.kill(worker)
                except Exception as e:
                    logger.warning(f"Error shutting down worker: {e}")
            self._pools[pool_key] = []
            self._all_workers[pool_key] = []
        self._worker_counts.clear()
        self._pending_count.clear()


class ALFWorldWorkerPool:
    """Client-side wrapper for accessing the Ray-based worker pool.
    
    This class provides a simple interface to acquire and release workers
    from the centralized Ray actor pool.
    """
    
    _pool_actor = None
    _lock: ClassVar[threading.Lock] = threading.Lock()
    _pool_actor_name: ClassVar[str | None] = None
    _auto_run_id: ClassVar[str | None] = None
    
    # Default max workers - can be overridden via set_max_workers()
    _max_workers: ClassVar[int] = 64
    
    @classmethod
    def set_max_workers(cls, max_workers: int):
        """Set the maximum number of workers before the pool is created.
        
        Args:
            max_workers: Maximum number of workers to create.
        """
        cls._max_workers = max_workers
        logger.info(f"ALFWorldWorkerPool max_workers set to {max_workers}")
    
    @classmethod
    def _get_pool_actor(cls):
        """Get or create the Ray actor that manages the worker pool."""
        if cls._pool_actor is None:
            with cls._lock:
                if cls._pool_actor is None:
                    pool_name = cls._get_pool_actor_name()
                    # Try to get existing named actor, or create a new one
                    try:
                        cls._pool_actor = ray.get_actor(pool_name)
                        cls._pool_actor_name = pool_name
                        logger.info(f"Connected to existing ALFWorldWorkerPoolActor: {pool_name}")
                    except ValueError:
                        # Actor doesn't exist, create it
                        # Note: We don't use lifetime="detached" because:
                        # 1. The pool only needs to live for the training run
                        # 2. Detached actors in anonymous namespaces cause warnings
                        # 3. The actor will be cleaned up when the driver exits
                        cls._pool_actor = ALFWorldWorkerPoolActor.options(
                            name=pool_name,
                            max_concurrency=1000,  # Allow many concurrent calls
                        ).remote(max_workers=cls._max_workers)
                        cls._pool_actor_name = pool_name
                        logger.info(f"Created new ALFWorldWorkerPoolActor {pool_name} with max_workers={cls._max_workers}")
        return cls._pool_actor

    @classmethod
    def _get_pool_actor_name(cls) -> str:
        """Get a per-run unique actor name (prefers RLLM_RUN_ID)."""
        run_id = os.environ.get("RLLM_RUN_ID")
        if not run_id:
            if cls._auto_run_id is None:
                cls._auto_run_id = str(uuid.uuid4())
            run_id = cls._auto_run_id
        safe_run_id = re.sub(r"[^A-Za-z0-9_.-]", "_", run_id)
        return f"alfworld_worker_pool_{safe_run_id}"

    @classmethod
    def get_pool_name(cls) -> str:
        """Return the resolved pool actor name for this process."""
        return cls._pool_actor_name or cls._get_pool_actor_name()
    
    @classmethod
    def acquire_worker(
        cls,
        config_path: str | None,
        train_eval: str,
        task_types: list[int] | None,
        max_episode_steps: int | None,
        seed: int | None,
        timeout: float = 600.0,
    ):
        """Acquire a worker from the pool, queuing if necessary.

        Args:
            config_path: Path to ALFWorld config YAML file.
            train_eval: Data split to use.
            task_types: List of task type IDs to include.
            seed: Random seed.
            timeout: Maximum time to wait for a worker.

        Returns:
            A Ray actor reference to an ALFWorldWorker.
        """
        pool_actor = cls._get_pool_actor()
        try:
            worker = ray.get(pool_actor.acquire_worker.remote(
                config_path, train_eval, task_types, max_episode_steps, seed, timeout
            ))
            return worker
        except ray.exceptions.RayTaskError as e:
            if "TimeoutError" in str(e):
                raise TimeoutError(
                    f"Timeout waiting for ALFWorld worker. "
                    f"max_workers={cls._max_workers}. "
                    f"Consider increasing max_workers or reducing parallel environments."
                ) from e
            raise RuntimeError(f"Failed to acquire ALFWorld worker: {e}") from e
    
    @classmethod
    def release_worker(
        cls,
        worker,
        config_path: str | None,
        train_eval: str,
        task_types: list[int] | None,
        max_episode_steps: int | None,
    ):
        """Return a worker to the pool for reuse."""
        pool_actor = cls._get_pool_actor()
        ray.get(pool_actor.release_worker.remote(
            worker, config_path, train_eval, task_types, max_episode_steps
        ))
    
    @classmethod
    def get_stats(cls) -> dict:
        """Get statistics about the worker pool."""
        pool_actor = cls._get_pool_actor()
        return ray.get(pool_actor.get_stats.remote())

    @classmethod
    def trigger_cleanup(cls, max_age_seconds: float = 600) -> dict:
        """Trigger a cleanup of old temp directories.

        Args:
            max_age_seconds: Maximum age of directories to keep.

        Returns:
            Dictionary with cleanup statistics.
        """
        pool_actor = cls._get_pool_actor()
        return ray.get(pool_actor.trigger_cleanup.remote(max_age_seconds))
    
    @classmethod
    def shutdown(cls):
        """Shutdown all workers and the pool actor."""
        if not ray.is_initialized():
            cls._pool_actor = None
            return
        pool_name = cls._pool_actor_name or cls._get_pool_actor_name()
        actor = cls._pool_actor
        if actor is None:
            try:
                actor = ray.get_actor(pool_name)
            except ValueError:
                actor = None
        if actor is not None:
            try:
                ray.get(actor.shutdown.remote())
                ray.kill(actor)
            except Exception as e:
                logger.warning(f"Error shutting down pool actor: {e}")
        cls._pool_actor = None
        cls._pool_actor_name = None
        cls._auto_run_id = None


class ALFWorldEnv(BaseEnv):
    """Text-based household task environment with Ray-based process isolation.

    ALFWorld provides text-based interactive fiction games for household tasks
    like picking up objects, placing them in receptacles, heating/cooling items, etc.

    The environment uses a Ray remote actor from a shared pool to run the actual
    TextWorld environment in an isolated process, ensuring thread safety for use
    with async execution engines. Workers are reused to avoid expensive re-initialization.

    Attributes:
        max_steps: Maximum number of steps per episode.
        train_eval: Whether to use train or eval data split.
        task_types: List of task type IDs to include.
    """

    def __init__(
        self,
        config_path: str | None = None,
        max_steps: int = 50,
        train_eval: str = "train",
        task_types: list[int] | None = None,
        game_file: str | None = None,
        seed: int | None = None,
        **kwargs,
    ):
        """Initialize ALFWorld environment.

        Args:
            config_path: Path to ALFWorld config YAML file. If None, uses default.
            max_steps: Maximum steps per episode.
            train_eval: Data split to use ("train", "eval_in_distribution", "eval_out_of_distribution").
            task_types: List of task type IDs (1-6) to include. If None, includes all.
            game_file: Specific game file to load. If provided, overrides random selection.
            seed: Random seed for reproducibility.
            **kwargs: Additional arguments passed to the underlying environment.
        """
        self.config_path = config_path
        self.max_steps = max_steps
        self.train_eval = train_eval
        self.task_types = task_types or list(TASK_TYPES.keys())
        self.game_file = game_file
        self.seed = seed
        self.worker_acquire_timeout = float(kwargs.pop("worker_acquire_timeout", 600.0))
        self.kwargs = kwargs

        self._worker = None
        self._current_game_file = None
        self._step_count = 0
        self._admissible_commands = []
        self._task_type = None
        self._initialized = False
        self._logged_tmp_dir = False

    def _lazy_init(self):
        """Lazily acquire a Ray worker from the pool."""
        if self._initialized:
            return

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        try:
            # Acquire a worker from the centralized Ray pool (reuses existing workers)
            # Note: game_file is NOT passed here - it's passed at reset() time
            self._worker = ALFWorldWorkerPool.acquire_worker(
                config_path=self.config_path,
                train_eval=self.train_eval,
                task_types=self.task_types,
                max_episode_steps=self.max_steps,
                seed=self.seed,
                timeout=self.worker_acquire_timeout,
            )

            self._initialized = True

        except TimeoutError as e:
            raise RuntimeError(
                f"Timeout acquiring ALFWorld worker: {e}. "
                "Consider increasing max_workers or reducing parallel environments."
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to acquire ALFWorld Ray worker: {e}. "
                "Please ensure textworld and other required packages are installed."
            ) from e

    def reset(self, seed: int | None = None) -> tuple[str, dict]:
        """Reset the environment to start a new episode.

        Returns:
            A tuple of (observation, info) where:
                - observation: Text description of the initial state
                - info: Dictionary containing admissible_commands, task_type, etc.
        """
        self._lazy_init()

        self._step_count = 0

        if seed is not None:
            self.seed = seed

        # Reset the environment via Ray worker, passing game_file for this specific task
        observation, worker_info = ray.get(
            self._worker.reset.remote(self.seed, game_file=self.game_file)
        )

        self._admissible_commands = worker_info.get("admissible_commands", [])
        self._current_game_file = worker_info.get("game_file", None)
        tmp_dir = worker_info.get("tmp_dir")
        if tmp_dir and not self._logged_tmp_dir:
            logger.info(f"ALFWorldEnv reset using tmp_dir={tmp_dir}")
            self._logged_tmp_dir = True

        # Determine task type from game file path
        self._task_type = self._get_task_type_from_path(self._current_game_file)

        info = {
            "admissible_commands": self._admissible_commands,
            "task_type": self._task_type,
            "game_file": self._current_game_file,
            "tmp_dir": tmp_dir,
            "step": self._step_count,
            "max_steps": self.max_steps,
        }

        return observation, info

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        """Execute an action in the environment.

        Args:
            action: Text action to execute (e.g., "go to kitchen", "pick up apple").

        Returns:
            A tuple of (observation, reward, done, info) where:
                - observation: Text description of the new state
                - reward: 10.0 if task completed successfully, 0.0 otherwise
                - done: Whether the episode has ended
                - info: Dictionary with admissible_commands, won status, etc.
        """
        self._lazy_init()

        self._step_count += 1

        # Execute action via Ray worker
        observation, reward, done, worker_info = ray.get(self._worker.step.remote(action))

        self._admissible_commands = worker_info.get("admissible_commands", [])
        won = worker_info.get("won", False)
        tmp_dir = worker_info.get("tmp_dir")

        # Check if max steps reached
        if self._step_count >= self.max_steps:
            done = True

        info = {
            "admissible_commands": self._admissible_commands,
            "available_actions": self._admissible_commands,
            "task_type": self._task_type,
            "game_file": self._current_game_file,
            "tmp_dir": tmp_dir,
            "step": self._step_count,
            "max_steps": self.max_steps,
            "success": won,
            "action_is_valid": worker_info.get("action_is_valid", False),
            "action_is_effective": worker_info.get("action_is_effective", True),
        }

        return observation, reward, done, info

    def _get_task_type_from_path(self, game_file: str | None) -> str | None:
        """Extract task type from game file path."""
        if game_file is None:
            return None

        for task_type in TASK_TYPES.values():
            if task_type in game_file:
                return task_type
        return None

    def get_admissible_commands(self) -> list[str]:
        """Get the list of valid commands for the current state."""
        return self._admissible_commands

    def close(self):
        """Return the worker to the pool for reuse (don't destroy it)."""
        if self._worker is not None:
            try:
                # Return worker to the centralized Ray pool for reuse
                # Note: game_file is NOT part of pool key, so don't pass it
                ALFWorldWorkerPool.release_worker(
                    self._worker,
                    config_path=self.config_path,
                    train_eval=self.train_eval,
                    task_types=self.task_types,
                    max_episode_steps=self.max_steps,
                )
            except Exception as e:
                logger.warning(f"Error returning ALFWorld worker to pool: {e}")
                # If we can't return to pool, kill it
                try:
                    ray.kill(self._worker)
                except Exception:
                    pass
            self._worker = None
        self._initialized = False

    @staticmethod
    def set_max_workers(max_workers: int):
        """Set the maximum number of Ray workers in the pool.
        
        This should be called BEFORE any environments are created.
        A good value is typically the number of concurrent trajectories you expect.
        
        Args:
            max_workers: Maximum number of workers to create.
        """
        ALFWorldWorkerPool.set_max_workers(max_workers)

    @staticmethod
    def shutdown_pool():
        """Shutdown the shared ALFWorld worker pool for this run."""
        ALFWorldWorkerPool.shutdown()
    
    @staticmethod
    def from_dict(env_args: dict) -> "ALFWorldEnv":
        """Create an ALFWorldEnv instance from a dictionary.

        Args:
            env_args: Dictionary containing environment configuration.
                Supported keys: config_path, max_steps, train_eval, task_types,
                game_file, seed, worker_acquire_timeout, max_workers (for pool configuration).

        Returns:
            A new ALFWorldEnv instance.
        """
        # Handle max_workers separately - it configures the pool, not the env
        if "max_workers" in env_args:
            ALFWorldWorkerPool.set_max_workers(env_args["max_workers"])
        
        allowed_keys = {
            "config_path",
            "max_steps",
            "train_eval",
            "task_types",
            "game_file",
            "seed",
            "worker_acquire_timeout",
        }
        filtered = {k: v for k, v in env_args.items() if k in allowed_keys}
        return ALFWorldEnv(**filtered)

    @staticmethod
    def is_multithread_safe() -> bool:
        """ALFWorld is multithread-safe due to Ray process-level isolation.
        
        Each ALFWorldEnv instance runs its environment in a separate Ray actor
        process, ensuring thread safety for concurrent access.
        """
        return True

    @classmethod
    def get_pool_stats(cls) -> dict | None:
        """Get statistics about the worker pool.
        
        Returns:
            Dictionary with pool stats, or None if pool not initialized.
        """
        try:
            return ALFWorldWorkerPool.get_stats()
        except Exception:
            return None
    
    @classmethod
    def print_pool_stats(cls):
        """Print a summary of the worker pool statistics."""
        stats = cls.get_pool_stats()
        if stats:
            print(f"[ALFWorld Worker Pool] max_workers={stats['max_workers']}")
            for config, count in stats.get('worker_counts', {}).items():
                available = stats.get('available_workers', {}).get(config, 0)
                print(f"[ALFWorld Worker Pool] Config '{config[:60]}...': {count} workers created, {available} available")
        else:
            print("[ALFWorld Worker Pool] Pool not initialized yet")

    def __repr__(self) -> str:
        return (
            f"ALFWorldEnv(config_path={self.config_path}, max_steps={self.max_steps}, "
            f"train_eval={self.train_eval}, task_types={self.task_types})"
        )
