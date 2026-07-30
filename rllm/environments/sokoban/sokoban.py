import dataclasses
import logging
import time
from typing import Any

import gym
import numpy as np
from gym_sokoban.envs.sokoban_env import SokobanEnv as GymSokobanEnv

from rllm.agents.agent import Action
from rllm.environments.base.base_env import BaseEnv
from rllm.environments.sokoban.config import SokobanEnvConfig
from rllm.environments.sokoban.solver import (
    get_min_steps as _solver_get_min_steps,
    is_deadlocked as _pattern_is_deadlocked,
)
from rllm.environments.sokoban.utils import (
    collect_entity_coordinates,
    derive_retry_seed,
    format_coordinate_render,
    generate_room,
    seed_context,
)
from rllm.environments.sokoban.solver import get_min_steps

logger = logging.getLogger(__name__)


class SokobanEnv(BaseEnv, GymSokobanEnv):
    """
    Sokoban environment wrapping gym-sokoban with text observations.

    Action Space (custom, 1-indexed):
    - 0: None (invalid)
    - 1: Up
    - 2: Down
    - 3: Left
    - 4: Right
    """

    ACTION_LOOKUP = {
        0: "None",
        1: "Up",
        2: "Down",
        3: "Left",
        4: "Right",
    }

    INVALID_ACTION = 0
    _ACTION_NAME_TO_INT = {
        "up": 1,
        "down": 2,
        "left": 3,
        "right": 4,
    }

    def __init__(self, config: SokobanEnvConfig | None = None, **kwargs):
        seed = kwargs.pop("seed", None)
        config = self._apply_overrides(config or SokobanEnvConfig(), kwargs)

        self.config = config
        self.seed = seed
        self.search_depth = config.search_depth
        self.GRID_LOOKUP = config.grid_lookup
        self.ACTION_LOOKUP = {0: "None", **config.action_lookup}
        self.render_mode = config.render_mode
        self.observation_format = config.observation_format
        self.max_reset_retries = config.max_reset_retries
        self.reward_mode = config.reward_mode
        self.max_repeated_obs = config.max_repeated_obs
        self._last_obs: str | None = None
        self._repeated_obs_count: int = 0
        self.deadlock_detection = config.deadlock_detection
        self.deadlock_solver_max_nodes = config.deadlock_solver_max_nodes
        self.game_over_on_deadlock = config.game_over_on_deadlock
        self.compute_oracle_advantage = config.compute_oracle_advantage
        self.oracle_advantage_cache = config.oracle_advantage_cache
        # Cached N(s_{t+1}) from the previous step, reused as N(s_t) next step when
        # oracle_advantage_cache is on. None means "no valid cache" (episode start
        # or feature off), forcing a fresh solver call. weight=1.0 only — never mix
        # with the deadlock solver's weight=2.5.
        self._cached_oracle_min: float | None = None
        # Per-step solver wall time accumulators (ablation profiling only). Reset at
        # the start of every step() and surfaced via info["solver_time"] /
        # info["deadlock_solver_time"]. Deadlock timing accrues inside _step_single
        # (called 1+ times per step, including multi-action), so it must be an
        # instance field rather than a step() local.
        self._deadlock_solver_time_step: float = 0.0
        self._consecutive_deadlock_count: int = 0

        GymSokobanEnv.__init__(
            self,
            dim_room=config.dim_room,
            max_steps=config.max_steps,
            num_boxes=config.num_boxes,
        )
        self.ACTION_SPACE = gym.spaces.Discrete(4, start=1)

        self.env_kwargs = {
            "seed": seed,
            "dim_room": config.dim_room,
            "max_steps": config.max_steps,
            "num_boxes": config.num_boxes,
            "search_depth": config.search_depth,
            "render_mode": config.render_mode,
            "observation_format": config.observation_format,
            "max_reset_retries": config.max_reset_retries,
            "grid_lookup": config.grid_lookup,
            "grid_vocab": config.grid_vocab,
            "reward_mode": config.reward_mode,
            "max_repeated_obs": config.max_repeated_obs,
            "deadlock_detection": config.deadlock_detection,
            "deadlock_solver_max_nodes": config.deadlock_solver_max_nodes,
            "game_over_on_deadlock": config.game_over_on_deadlock,
        }

    def _apply_overrides(self, config: SokobanEnvConfig, overrides: dict[str, Any]) -> SokobanEnvConfig:
        if not overrides:
            return config

        field_names = {field.name for field in dataclasses.fields(SokobanEnvConfig)}
        filtered = {key: value for key, value in overrides.items() if key in field_names}
        if "grid_lookup" in filtered and isinstance(filtered["grid_lookup"], dict):
            filtered["grid_lookup"] = {int(k): v for k, v in filtered["grid_lookup"].items()}
        return dataclasses.replace(config, **filtered)

    def reset(self, working_seed: int | None = None) -> tuple[str, dict]:
        # Use working_seed for retries, but preserve self.seed as the original seed.
        original_seed = working_seed if working_seed is not None else self.seed
        current_seed = original_seed
        last_exc: Exception | None = None

        for attempt in range(1, self.max_reset_retries + 1):
            try:
                with seed_context(current_seed) as (py_rng, np_rng):
                    self.room_fixed, self.room_state, self.box_mapping, _ = generate_room(
                        dim=self.config.dim_room,
                        num_boxes=self.config.num_boxes,
                        search_depth=self.search_depth,
                        py_rng=py_rng,
                        np_rng=np_rng,
                    )
                self.num_env_steps = 0
                self.reward_last = 0
                self.boxes_on_target = 0
                self.player_position = np.argwhere(self.room_state == 5)[0]

                obs = self.render()
                self._last_obs = obs
                self._repeated_obs_count = 0
                self._consecutive_deadlock_count = 0
                self._cached_oracle_min = None  # new episode: invalidate oracle cache

                if attempt > 1:
                    logger.info(
                        "Sokoban reset recovered after %s attempts (initial seed=%s, final seed=%s).",
                        attempt,
                        original_seed,
                        current_seed,
                    )
                return obs, {}
            except (RuntimeError, RuntimeWarning) as exc:
                last_exc = exc
                next_seed = derive_retry_seed(current_seed)
                remaining = self.max_reset_retries - attempt
                if remaining > 0:
                    logger.debug(
                        "Sokoban reset retry (attempt=%s/%s, seed=%s -> %s): %s",
                        attempt,
                        self.max_reset_retries,
                        current_seed,
                        next_seed,
                        exc,
                    )
                else:
                    logger.error(
                        "Sokoban reset exhausted retries (attempt=%s/%s, initial seed=%s, last seed=%s): %s",
                        attempt,
                        self.max_reset_retries,
                        original_seed,
                        current_seed,
                        exc,
                    )
                if remaining <= 0:
                    break
                current_seed = next_seed

        raise RuntimeError(
            "Sokoban reset failed after "
            f"{self.max_reset_retries} attempts (initial seed={original_seed}, last seed={current_seed})."
        ) from last_exc

    def step(self, action):
        action_val = getattr(action, "action", action)

        # Per-step solver profiling (ablation). Reset accumulators for this step;
        # _check_deadlock adds to _deadlock_solver_time_step during _step_single.
        self._deadlock_solver_time_step = 0.0
        oracle_solver_time = 0.0

        if self.compute_oracle_advantage:
            # 1. Compute the optimal remaining steps BEFORE the action: N(s_t)
            #    With oracle_advantage_cache on, N(s_t) equals the previous step's
            #    N(s_{t+1}) because nothing mutates room_state between steps (render
            #    is read-only). weight=1.0 matches the fresh call exactly, so the
            #    reused value is numerically identical to recomputing.
            if self.oracle_advantage_cache and self._cached_oracle_min is not None:
                prev_min = self._cached_oracle_min
            else:
                _t0 = time.time()
                prev_min = get_min_steps(self.room_state, self.room_fixed, weight=1.0)
                oracle_solver_time += time.time() - _t0

        if isinstance(action_val, list):
            obs, reward, done, info = self._step_multi(action_val)
        else:
            obs, reward, done, info = self._step_single(action_val)

        if self.compute_oracle_advantage:
            # 2. Compute the optimal remaining steps AFTER the action: N(s_{t+1})
            _t0 = time.time()
            curr_min = get_min_steps(self.room_state, self.room_fixed, weight=1.0)
            oracle_solver_time += time.time() - _t0

            # Cache N(s_{t+1}) so the next step can reuse it as its N(s_t).
            self._cached_oracle_min = curr_min

            # 3. Oracle Advantage = N(s_t) - N(s_{t+1})
            #
            #    Interpretation (potential-based shaping, Φ(s) = -N(s)):
            #      A(s,a) = Φ(s') - Φ(s) = N(s_t) - N(s_{t+1})
            #
            #    Signal properties:
            #      Optimal action  (N decreases by 1):  adv = +1  (positive reinforcement)
            #      Wasteful action (N unchanged):        adv =  0  (no signal)
            #      Harmful action  (N increases):        adv < 0   (penalty)
            #      Fatal action    (causes deadlock):    adv = -N(s_t)  (proportional penalty)
            #      Already in deadlock:                  adv = 0   (no gradient)
            if prev_min == float('inf'):
                adv = 0.0
            elif curr_min == float('inf'):
                adv = float(-prev_min)
            else:
                adv = float(prev_min - curr_min)

            info['oracle_advantage'] = adv

        # Surface solver wall time for ablation profiling. Written unconditionally
        # so the engine always sees these keys (values are 0.0 when no solver ran).
        info['solver_time'] = oracle_solver_time
        info['deadlock_solver_time'] = self._deadlock_solver_time_step

        return obs, reward, done, info

    def _step_single(self, action: int | str) -> tuple[str, float, bool, dict]:
        """Execute a single action."""
        action_int = self._normalize_action(action)

        if action_int is None or action_int == self.INVALID_ACTION or action_int not in self.ACTION_LOOKUP:
            return self.render(), 0, False, {"action_is_effective": False, "action_is_valid": False, "success": self.success()}

        previous_pos = np.array(self.player_position)
        _, reward, done, _ = GymSokobanEnv.step(self, action_int)

        # Binary reward mode: all steps return 0, except terminal success returns 1
        if self.reward_mode == "binary":
            reward = 1.0 if (done and self.success()) else 0.0

        next_obs = self.render()
        action_effective = not np.array_equal(previous_pos, self.player_position)

        # Detect consecutive identical observations (stale state)
        if self.max_repeated_obs > 0:
            if next_obs == self._last_obs:
                self._repeated_obs_count += 1
            else:
                self._repeated_obs_count = 0
            self._last_obs = next_obs

            if self._repeated_obs_count >= self.max_repeated_obs:
                done = True

        # Deadlock detection — terminate early on unsolvable states
        # game_over_on_deadlock controls tolerance:
        #   0 = detection runs (for info/feedback) but never terminates
        #   1 = immediate game over on first deadlock
        #   N>1 = require N consecutive deadlock detections before game over
        terminated_by_deadlock = False
        if not done and self.deadlock_detection != "off":
            is_deadlocked = self._check_deadlock()
            if is_deadlocked:
                self._consecutive_deadlock_count += 1
                if (
                    self.game_over_on_deadlock >= 1
                    and self._consecutive_deadlock_count >= self.game_over_on_deadlock
                ):
                    terminated_by_deadlock = True
                    done = True
            else:
                self._consecutive_deadlock_count = 0

        info = {
            "action_is_effective": action_effective,
            "action_is_valid": True,
            "success": self.success(),
            "terminated_by_repeated_obs": (
                self.max_repeated_obs > 0
                and self._repeated_obs_count >= self.max_repeated_obs
            ),
            "terminated_by_deadlock": terminated_by_deadlock,
            "consecutive_deadlock_count": self._consecutive_deadlock_count,
        }
        return next_obs, reward, done, info

    def _step_multi(self, actions: list[str]) -> tuple[str, float, bool, dict]:
        """Execute multiple actions in sequence."""
        total_reward = 0.0
        done = False
        effective_count = 0
        valid_count = 0
        executed_count = 0
        terminated_by_repeated_obs = False
        terminated_by_deadlock = False

        for action in actions:
            if done:
                break
            executed_count += 1
            obs, reward, done, info = self._step_single(action)
            total_reward += reward
            if info.get("action_is_effective", False):
                effective_count += 1
            if info.get("action_is_valid", False):
                valid_count += 1
            if info.get("terminated_by_repeated_obs", False):
                terminated_by_repeated_obs = True
            if info.get("terminated_by_deadlock", False):
                terminated_by_deadlock = True

        next_obs = self.render()
        info = {
            "action_is_effective": effective_count > 0,
            "action_is_valid": executed_count > 0 and valid_count == executed_count,
            "success": self.success(),
            "actions_executed": executed_count,
            "effective_actions": effective_count,
            "valid_actions": valid_count,
            "terminated_by_repeated_obs": terminated_by_repeated_obs,
            "terminated_by_deadlock": terminated_by_deadlock,
        }
        return next_obs, total_reward, done, info

    def _normalize_action(self, action: int | str | None) -> int | None:
        """Normalize mixed action inputs into custom int action ids."""
        if action is None:
            return self.INVALID_ACTION
        if isinstance(action, int):
            return action
        action_str = str(action).strip()
        if not action_str:
            return self.INVALID_ACTION
        if action_str.isdigit():
            return int(action_str)
        return self._ACTION_NAME_TO_INT.get(action_str.lower())

    def _check_deadlock(self) -> bool:
        """Check if current state is an unsolvable deadlock.

        Uses the mode configured by ``self.deadlock_detection``:
        - ``"pattern"``: fast corner / frozen-line pattern check only.
        - ``"solver"``: pattern check first, then full A* verification.
        - ``"off"``: always returns False (caller should not invoke).
        """
        if self.deadlock_detection == "off":
            return False
        # Cheap pattern check (corner + frozen-line)
        if _pattern_is_deadlocked(self.room_state, self.room_fixed):
            return True
        # Full solver verification
        if self.deadlock_detection == "solver":
            _t0 = time.time()
            min_steps = _solver_get_min_steps(
                self.room_state,
                self.room_fixed,
                weight=2.5,
                max_nodes=self.deadlock_solver_max_nodes,
            )
            self._deadlock_solver_time_step += time.time() - _t0
            if min_steps == float("inf"):
                return True
        return False

    def render(self, mode: str | None = None):
        if mode in {"grid", "coord", "grid_coord"}:
            return self._render_text(mode)

        if mode == "tiny_rgb_array":
            mode = "text"

        render_mode = mode if mode is not None else self.render_mode
        if render_mode == "text":
            return self._render_text(self.observation_format)
        if render_mode == "rgb_array":
            return self.get_image(mode="rgb_array", scale=1)
        raise ValueError(f"Invalid mode: {render_mode}")

    def _render_text(self, observation_format: str) -> str:
        if observation_format == "grid":
            room = np.where((self.room_state == 5) & (self.room_fixed == 2), 6, self.room_state)
            return "\n".join("".join(self.GRID_LOOKUP.get(int(cell), "?") for cell in row) for row in room.tolist())
        if observation_format == "coord":
            entity_coords = collect_entity_coordinates(self.room_state, self.room_fixed)
            return format_coordinate_render(entity_coords, self.config.dim_room)
        if observation_format == "grid_coord":
            entity_coords = collect_entity_coordinates(self.room_state, self.room_fixed)
            return "Coordinates:\n" + format_coordinate_render(entity_coords, self.config.dim_room) + "\nGrid Map:\n" + self._render_text("grid")
        raise ValueError(f"Invalid observation_format: {observation_format}")

    def success(self) -> bool:
        return self.boxes_on_target == self.num_boxes

    def finished(self) -> bool:
        return self.success()

    def get_all_actions(self) -> list[int]:
        return [action for action in self.ACTION_LOOKUP.keys() if action != self.INVALID_ACTION]

    @staticmethod
    def from_dict(env_info: dict) -> "SokobanEnv":
        allowed_keys = {field.name for field in dataclasses.fields(SokobanEnvConfig)} | {"seed"}
        filtered = {key: value for key, value in env_info.items() if key in allowed_keys}
        return SokobanEnv(**filtered)
