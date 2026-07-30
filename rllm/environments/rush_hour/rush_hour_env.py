"""Rush Hour sliding-block puzzle environment for rllm.

Wraps the pure-Python :class:`RushHourBoard` engine, the IDA* solver oracle,
and the per-seed generator into the rllm :class:`BaseEnv` interface (4-tuple
``step``, ``from_dict`` factory, ``success`` flag).

Mirrors the structure and conventions of ``rllm/environments/sokoban/sokoban.py``:
binary reward, solver-based potential oracle advantage (``N(s_t) - N(s_{t+1})``),
static deadlock detection, and the standard info keys
(``action_is_effective`` / ``action_is_valid`` / ``success``).

Thread-safety: each instance owns its own board; generation seeds an isolated
``random.Random`` inside the generator, so concurrent instances do not share
global random state.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from typing import Any

from rllm.environments.base.base_env import BaseEnv
from rllm.environments.rush_hour.board import RushHourBoard, parse_action
from rllm.environments.rush_hour.config import RushHourEnvConfig, filter_config_kwargs
from rllm.environments.rush_hour.generator import generate_board
from rllm.environments.rush_hour.oracle import NTableCache, SidecarStore
from rllm.environments.rush_hour.solver import (
    INF,
    get_min_moves,
    is_deadlocked,
)

logger = logging.getLogger(__name__)


# Difficulty presets --------------------------------------------------------
# Applied as defaults by ``from_dict`` when a dataset row names a difficulty.
DIFFICULTY_CONFIGS = {
    "id": {
        "width": 6, "height": 6, "num_vehicles": 7,
        "min_moves": 6, "max_moves": 8, "max_steps": 30,
        "observation_format": "grid_coord",
    },
    "unseen": {
        "width": 6, "height": 6, "num_vehicles": 9,
        "min_moves": 7, "max_moves": 9, "max_steps": 30,
        "observation_format": "grid_coord",
    },
}


class RushHourEnv(BaseEnv):
    """Text-based Rush Hour environment.

    Observation:
        A string rendering of the board (grid / coordinate / both, per
        ``observation_format``).

    Action space:
        A single move string parsed by :func:`parse_action`, e.g. ``"A+2"``
        (slide car A right/down 2 cells), ``"B-1"``, or ``"[A+]"``.

    Reward:
        Binary: 1.0 when the red car (A) reaches the exit, else 0.0.

    Invalid actions:
        Unparseable strings or unknown car labels do NOT terminate the episode
        (unless ``max_steps`` is reached); ``info["action_is_valid"] = False``
        so the agent can receive corrective feedback and retry.

    Termination:
        * Red car reaches exit (success, reward 1.0).
        * ``max_steps`` exceeded.
        * Optionally on deadlock (see ``game_over_on_deadlock``).

    Success:
        ``info["success"]`` is ``True`` when the board is solved.
    """

    def __init__(
        self,
        config: RushHourEnvConfig | None = None,
        seed: int | None = None,
        board_config: str | None = None,
        **kwargs,
    ):
        super().__init__()
        config = config or RushHourEnvConfig()
        # Apply only valid RushHourEnvConfig field overrides from kwargs.
        overrides = filter_config_kwargs(kwargs)
        if overrides:
            config = dataclasses.replace(config, **overrides)

        self.config = config
        self.seed = seed
        # Optional flat board string lets a dataset row pin an exact puzzle.
        self.board_config = board_config

        # Mirror config onto attributes for convenient access.
        self.width = config.width
        self.height = config.height
        self.observation_format = config.observation_format
        self.max_steps = config.max_steps
        self.max_reset_retries = config.max_reset_retries
        self.compute_oracle_advantage = config.compute_oracle_advantage
        self.deadlock_detection = config.deadlock_detection
        self.game_over_on_deadlock = config.game_over_on_deadlock
        self.solver_max_depth = config.solver_max_depth
        self.solver_max_nodes = config.solver_max_nodes
        self.oracle_backend = config.oracle_backend
        self.oracle_max_states = config.oracle_max_states
        self.oracle_cache_capacity = config.oracle_cache_capacity
        self.oracle_sidecar_path = config.oracle_sidecar_path
        self.oracle_potential = config.oracle_potential
        # RL-network teacher backend (oracle_backend="rlnet"); see rlnet_oracle.py.
        self.rlnet_ckpt_path = config.rlnet_ckpt_path
        self.rlnet_potential_scale = config.rlnet_potential_scale
        self.rlnet_solved_value = config.rlnet_solved_value
        self.rlnet_device = config.rlnet_device
        self.rlnet_max_pieces = config.rlnet_max_pieces

        # rlnet teacher handle, lazily resolved in _build_n_table when the rlnet
        # backend is selected. None for all solver backends.
        self._rlnet = None

        # Validate the rlnet backend up front (fail fast, not mid-rollout).
        # NOTE: do NOT require compute_oracle_advantage here — validation mode
        # legitimately sets it to False (the trainer skips oracle work during
        # eval, see agent_ppo_trainer.py), exactly like the solver backends which
        # silently no-op in _build_n_table when oracle is off. The only hard
        # requirement is the checkpoint; without it rlnet cannot load a teacher.
        if self.oracle_backend == "rlnet" and self.compute_oracle_advantage:
            if not self.rlnet_ckpt_path:
                raise ValueError(
                    "oracle_backend='rlnet' requires rlnet_ckpt_path (the trained "
                    "SB3 checkpoint to use as the soft teacher)."
                )

        # Episode state (initialised in reset).
        self.board: RushHourBoard | None = None
        self._min_moves: int = 0
        self._step_count: int = 0
        self._done: bool = False
        self._consecutive_deadlock_count: int = 0
        # Solver wall time accumulators (ablation profiling). _reset_solver_time
        # captures the N(s)-table build / difficulty-label solve during reset();
        # _solver_time_step captures per-step _solve_n calls (reset each step()).
        self._reset_solver_time: float = 0.0
        self._solver_time_step: float = 0.0
        # Exact N(s) table for the current board (built lazily in reset when the
        # oracle is enabled and the "table" backend is selected). None means the
        # table backend is unused or the board was too large -> IDA* fallback.
        self._n_table: dict | None = None

    # ------------------------------------------------------------------
    # BaseEnv interface
    # ------------------------------------------------------------------

    def _build_n_table(self) -> None:
        """Resolve the exact N(s) table for the current board per oracle backend.

        Sets ``self._n_table`` to:
        * a dict table — exact O(1) per-step lookups, or
        * ``None`` — no table (use IDA* in ``_solve_n``): backend is "ida", oracle
          is disabled, the board's component exceeded ``oracle_max_states``, or a
          precomputed entry recorded the board as too-large.

        Backends:
        * "table"       — build/fetch the table via the process-wide cache.
        * "precomputed" — look the board up in the offline sidecar by board_config
          md5; on a hit use it, on a miss (stale/incomplete sidecar, or no
          board_config to key on) fall back to the online "table" build so the
          result is always correct.
        * "ida"         — leave the table None.
        """
        if not self.compute_oracle_advantage or self.oracle_backend == "ida":
            self._n_table = None
            return

        # rlnet backend: hold a (process-cached) RL teacher network instead of an
        # N(s) table. _solve_n routes to it; no table is built.
        if self.oracle_backend == "rlnet":
            from rllm.environments.rush_hour.rlnet_oracle import RLNetPotential

            self._rlnet = RLNetPotential.get(
                ckpt_path=self.rlnet_ckpt_path,
                device=self.rlnet_device,
                scale=self.rlnet_potential_scale,
                max_pieces=self.rlnet_max_pieces,
                width=self.width,
                height=self.height,
                solved_value=self.rlnet_solved_value,
            )
            self._n_table = None
            return

        if self.oracle_backend == "precomputed":
            # The sidecar is keyed by board_config md5, so it only applies to
            # pinned boards. With a sidecar path and a board_config, try a hit.
            if self.oracle_sidecar_path and self.board_config is not None:
                result = SidecarStore.lookup(
                    self.oracle_sidecar_path,
                    self.board_config,
                    expected_potential=self.oracle_potential,
                )
                if result is not SidecarStore.MISSING:
                    # Hit: dict table, or None meaning "too large -> IDA* fallback".
                    self._n_table = result
                    return
                # Miss: stale/incomplete sidecar — fall through to online build.
            # No sidecar / no board_config / miss: build online so we stay correct.

        # "table" backend, or "precomputed" fallback: build/fetch via the cache.
        self._n_table = NTableCache.get_or_build(
            self.board,
            cache_key=self.board.to_string(),
            max_states=self.oracle_max_states,
            capacity=self.oracle_cache_capacity,
            potential=self.oracle_potential,
        )

    def _solve_n(self, board: RushHourBoard) -> float:
        """Optimal remaining move count N(s) for ``board``'s current state.

        Uses the exact N(s) table when available (O(1) lookup; absent key = true
        INF) and otherwise IDA* ``get_min_moves``. The table is exact, so unlike
        budget-limited IDA* it never reports a false INF for a solvable state.
        """
        if self.oracle_backend == "rlnet":
            # Soft teacher: returns -scale*V(s) so the unchanged adv=prev-curr in
            # step() becomes scale*(V(s')-V(s)). Never INF (see rlnet_oracle.py).
            # Guard against a None handle: _build_n_table only loads the teacher
            # when compute_oracle_advantage is True, so a config that flips the
            # flag on after a reset that ran with it off would otherwise NoneType-
            # deref here. Lazily resolve instead of crashing mid-rollout.
            if self._rlnet is None:
                self._build_n_table()
            # Pass THIS env's scale/solved_value explicitly: the teacher is a
            # process-wide singleton keyed only on (ckpt, device, geometry), so two
            # env configs sharing a checkpoint but differing in scale/solved_value
            # share one instance. Passing them per call ensures this env uses its
            # OWN configured scalars rather than whichever config loaded first.
            return self._rlnet.potential(
                board,
                scale=self.rlnet_potential_scale,
                solved_value=self.rlnet_solved_value,
            )
        if self._n_table is not None:
            return self._n_table.get(board.memo_key(), INF)
        if self.oracle_potential == "cells":
            # No exact IDA* fallback exists for the cells potential, and the
            # moves-based IDA* would silently mix cost units into the advantage.
            # Fail early instead of degrading to moves.
            raise RuntimeError(
                "oracle_potential='cells' requires the table/precomputed backend "
                "(component exceeded oracle_max_states or backend=ida)"
            )
        return get_min_moves(
            board,
            max_depth=self.solver_max_depth,
            max_nodes=self.solver_max_nodes,
        )

    def reset(self, working_seed: int | None = None) -> tuple[str, dict]:
        """Reset the environment to an initial puzzle.

        If ``board_config`` is set, rebuilds the exact puzzle from the flat
        board string; otherwise generates a new puzzle from the seed within the
        configured difficulty window.
        """
        current_seed = working_seed if working_seed is not None else self.seed

        # Solver profiling (ablation): time the whole board-build region, which is
        # where all reset-time solving happens (N(s)-table build, difficulty-label
        # get_min_moves, and the generator's internal solve). Rendering/bookkeeping
        # after this is negligible and solver-free.
        self._reset_solver_time = 0.0
        _reset_t0 = time.time()

        if self.board_config is not None:
            self.board = RushHourBoard.from_string(
                self.board_config, width=self.width, height=self.height
            )
            # Build the exact N(s) table (table backend) before labelling so the
            # difficulty label can reuse the same exact N(s) source as the oracle.
            self._build_n_table()
            # Difficulty label is always in MOVE units (aligned with the
            # DIFFICULTY_CONFIGS windows), independent of oracle_potential.
            # When the table holds MOVE-unit values (oracle_potential="moves"),
            # read the exact N(s_0) straight from it: O(1) and never a false INF.
            # Only when the table is absent (oracle off / component too large /
            # ida backend) or holds CELL-unit values do we run get_min_moves.
            if self.oracle_backend == "rlnet":
                # rlnet is a solver-free soft teacher: never call get_min_moves
                # (honors the firewall claim AND avoids an IDA* solve per reset
                # under 64-thread concurrency). The exact difficulty label is
                # simply unavailable from the teacher; -1 = unknown (same value
                # used elsewhere for an unsolved/over-budget label).
                n = INF
            elif self._n_table is not None and self.oracle_potential == "moves":
                n = self._n_table.get(self.board.memo_key(), INF)
            else:
                n = get_min_moves(
                    self.board,
                    max_depth=self.solver_max_depth,
                    max_nodes=self.solver_max_nodes,
                )
            self._min_moves = int(n) if n != INF else -1
        else:
            self.board, self._min_moves = generate_board(
                seed=current_seed,
                width=self.config.width,
                height=self.config.height,
                num_vehicles=self.config.num_vehicles,
                num_walls=self.config.num_walls,
                min_moves=self.config.min_moves,
                max_moves=self.config.max_moves,
                max_retries=self.max_reset_retries,
                solver_max_depth=self.solver_max_depth,
            )
            # Generated path already has the exact label from the generator;
            # still build the table so per-step oracle queries are O(1).
            self._build_n_table()

        self._reset_solver_time = time.time() - _reset_t0

        self._step_count = 0
        self._done = False
        self._consecutive_deadlock_count = 0

        obs = self.board.render(self.observation_format)
        info = {
            "success": self.board.solved,  # False at start
            "min_moves": self._min_moves,
            "board": self.board.render("grid"),
            # Reset-time solver wall time for ablation profiling (0.0 when oracle off
            # and no label solve ran). Consumed by the engine per-trajectory.
            "reset_solver_time": self._reset_solver_time,
        }
        return obs, info

    def step(self, action) -> tuple[str, float, bool, dict]:
        action_val = getattr(action, "action", action)
        action_str = action_val if isinstance(action_val, str) else str(action_val)

        # Per-step solver profiling (ablation). Reset before any _solve_n call.
        self._solver_time_step = 0.0

        parsed = parse_action(action_str)

        # --- Invalid action: unparseable string or unknown car label ---
        if parsed is None or self.board.piece_index_for_label(parsed[0]) is None:
            return self._invalid_action_step()

        label, steps = parsed

        # --- Valid action: apply the move ---
        if self.compute_oracle_advantage:
            # 1. Optimal remaining moves BEFORE the action: N(s_t)
            _t0 = time.time()
            prev_min = self._solve_n(self.board)
            self._solver_time_step += time.time() - _t0

        moved = self.board.apply_action(label, steps)
        action_is_effective = moved != 0

        self._step_count += 1
        solved = self.board.solved
        done = solved or self._step_count >= self.max_steps

        info: dict[str, Any] = {
            "action_is_effective": action_is_effective,
            "action_is_valid": True,
            "success": solved,
            "min_moves": self._min_moves,
        }

        if self.compute_oracle_advantage:
            # 2. Optimal remaining moves AFTER the action: N(s_{t+1})
            _t0 = time.time()
            curr_min = self._solve_n(self.board)
            self._solver_time_step += time.time() - _t0

            # 3. Oracle Advantage = N(s_t) - N(s_{t+1})
            #
            #    Interpretation (potential-based shaping, Phi(s) = -N(s)):
            #      A(s,a) = Phi(s') - Phi(s) = N(s_t) - N(s_{t+1})
            #
            #    NOTE: the per-move magnitudes below assume oracle_potential="moves"
            #    (N counts whole slides, so one optimal slide of any distance moves
            #    N by exactly 1). With oracle_potential="cells" the SIGNS are the
            #    same but the magnitudes scale with slid distance: an optimal slide
            #    of k cells gives adv = +k, not +1 (N counts cells, not moves).
            #
            #    Signal properties (oracle_potential="moves"):
            #      Optimal move  (N decreases by 1):  adv = +1  (positive reinforcement)
            #      Wasteful move (N unchanged):        adv =  0  (no signal)
            #      Harmful move  (N increases):        adv < 0   (penalty)
            #      Fatal move    (causes deadlock):    adv = -N(s_t)  (proportional penalty)
            #      Already in deadlock:                adv = 0   (no gradient)
            if prev_min == INF:
                adv = 0.0
            elif curr_min == INF:
                adv = float(-prev_min)
            else:
                adv = float(prev_min - curr_min)
            info["oracle_advantage"] = adv

        # --- Deadlock detection (mirrors sokoban) ---
        # game_over_on_deadlock controls tolerance:
        #   0 = detection runs (for info) but never terminates
        #   1 = immediate game over on first deadlock
        #   N>1 = require N consecutive deadlock detections before game over
        #
        # The counter tracks consecutive deadlocked states the agent actually
        # *moved into*. An ineffective move (blocked / overshoot, board
        # unchanged) is treated like an invalid action: it does not advance the
        # deadlock streak and resets the counter — consistent with
        # _invalid_action_step, which also resets on a board-unchanged action.
        if not done and self.deadlock_detection != "off":
            if action_is_effective and is_deadlocked(self.board):
                self._consecutive_deadlock_count += 1
                if (
                    self.game_over_on_deadlock >= 1
                    and self._consecutive_deadlock_count >= self.game_over_on_deadlock
                ):
                    done = True
                    info["terminated_by_deadlock"] = True
            else:
                self._consecutive_deadlock_count = 0

        if done:
            self._done = True

        # Per-step solver wall time (ablation). 0.0 when oracle off.
        info["solver_time"] = self._solver_time_step
        obs = self.board.render(self.observation_format)
        return obs, self._reward(), done, info

    def _invalid_action_step(self) -> tuple[str, float, bool, dict]:
        """Handle an unparseable or unknown-label action without solving.

        The action does not change the board, so it is never effective/valid and
        emits no oracle signal; it still consumes a step and can hit max_steps.
        """
        self._step_count += 1
        done = self._step_count >= self.max_steps
        if done:
            self._done = True
        # An action that does not change the board breaks any run of
        # consecutive deadlocked states, so reset the counter.
        self._consecutive_deadlock_count = 0
        obs = self.board.render(self.observation_format)
        info = {
            "action_is_effective": False,
            "action_is_valid": False,
            "success": self.board.solved,
            "min_moves": self._min_moves,
        }
        if self.compute_oracle_advantage:
            # No state change -> no learning signal.
            info["oracle_advantage"] = 0.0
        # No _solve_n on the invalid path, so this is 0.0 (reset at step() entry).
        info["solver_time"] = self._solver_time_step
        return obs, self._reward(), done, info

    def _reward(self) -> float:
        """Binary reward: 1.0 on solve, else 0.0."""
        return 1.0 if self.board.solved else 0.0

    def success(self) -> bool:
        return self.board is not None and self.board.solved

    def render(self, mode: str | None = None) -> str:
        render_mode = mode if mode is not None else self.observation_format
        return self.board.render(render_mode)

    def close(self):
        """No external resources to release."""
        return

    @staticmethod
    def from_dict(env_info: dict) -> "RushHourEnv":
        """Create a RushHourEnv from a serialised dictionary.

        Recognised keys: ``seed``, ``board_config``, ``difficulty`` (optional),
        plus any :class:`RushHourEnvConfig` field. When ``difficulty`` names a
        preset, its values are applied as defaults (explicit keys override).
        """
        info = dict(env_info)

        difficulty = info.pop("difficulty", None)
        if difficulty and difficulty in DIFFICULTY_CONFIGS:
            preset = DIFFICULTY_CONFIGS[difficulty]
            for key, value in preset.items():
                info.setdefault(key, value)

        # Config-field overrides (width/height/num_vehicles/... etc.).
        config_kwargs = filter_config_kwargs(info)

        # Normalise numeric config fields that may arrive as numpy/str (e.g.
        # from parquet) to plain ints so downstream arithmetic is exact.
        int_fields = (
            "width", "height", "num_vehicles", "num_walls",
            "min_moves", "max_moves", "max_steps",
            "max_reset_retries", "game_over_on_deadlock", "solver_max_depth",
            "solver_max_nodes", "oracle_max_states", "oracle_cache_capacity",
            "rlnet_max_pieces",
        )
        for field in int_fields:
            if field in config_kwargs and config_kwargs[field] is not None:
                config_kwargs[field] = int(config_kwargs[field])

        # Normalise float config fields that may arrive as numpy/str from parquet.
        for float_field in ("rlnet_potential_scale", "rlnet_solved_value"):
            if config_kwargs.get(float_field) is not None:
                config_kwargs[float_field] = float(config_kwargs[float_field])

        # Normalise string config fields that may arrive as numpy.str_ from
        # parquet, mirroring the board_config coercion below.
        for str_field in (
            "oracle_backend", "oracle_sidecar_path", "oracle_potential",
            "rlnet_ckpt_path", "rlnet_device",
        ):
            if config_kwargs.get(str_field) is not None:
                config_kwargs[str_field] = str(config_kwargs[str_field])

        seed = info.get("seed", None)
        if seed is not None:
            seed = int(seed)

        board_config = info.get("board_config", None)
        if board_config is not None:
            board_config = str(board_config)

        return RushHourEnv(seed=seed, board_config=board_config, **config_kwargs)

    @staticmethod
    def is_multithread_safe() -> bool:
        return True
