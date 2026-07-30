"""Configuration dataclass for the Rush Hour environment."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field


@dataclass
class RushHourEnvConfig:
    """Configuration for :class:`RushHourEnv`.

    Mirrors the structure of ``SokobanEnvConfig``: board geometry, difficulty,
    rendering, reward mode, and solver-oracle / deadlock options.
    """

    # Board geometry
    width: int = 6
    height: int = 6
    num_vehicles: int = 8  # total pieces including the red car
    num_walls: int = 0

    # Difficulty: accepted optimal-move range for the generated puzzle.
    min_moves: int = 1
    max_moves: int = 20

    # Episode
    max_steps: int = 30

    # Observation format fed to the agent: grid | coord | grid_coord.
    observation_format: str = "grid_coord"

    # Reward is binary (1.0 on solve, else 0.0); see RushHourEnv._reward.

    # Generation retries when a seed fails to yield a puzzle in the move range.
    max_reset_retries: int = 200

    # Solver oracle (potential-based advantage) + deadlock detection.
    compute_oracle_advantage: bool = False
    deadlock_detection: str = "static"  # "static" | "off"
    game_over_on_deadlock: int = 0  # 0 = never terminate, N>=1 = terminate after N
    solver_max_depth: int = 40  # cap IDA* depth (bounds worst-case solve time)
    # IDA* node budget per solve. Must be >= the generator's acceptance budget,
    # otherwise re-solving a pinned puzzle at reset can exceed the budget and
    # report inf (difficulty label -> -1) for a puzzle the generator accepted.
    solver_max_nodes: int = 300_000

    # Oracle backend for N(s) when compute_oracle_advantage is True:
    #   "table"       — build an exact N(s) table once per board (multi-source
    #                   reverse BFS) and look up per-step in O(1). Exact (no false
    #                   INF) and fast for 6x6. Falls back to IDA* per-query if the
    #                   component exceeds ``oracle_max_states``.
    #   "precomputed" — load an offline sidecar ({board md5 -> N table}) once per
    #                   process and look up by board_config. Zero online solve
    #                   cost. On a sidecar miss falls back to online "table"
    #                   build (then IDA*), so a stale sidecar is safe.
    #   "ida"         — call IDA* get_min_moves per step (legacy; can report false
    #                   INF on hard-but-solvable states when the budget is hit).
    #   "rlnet"       — soft teacher: a trained RL net's value V(s)=max_a Q(s,a)
    #                   instead of the exact solver N(s). See the rlnet_* fields
    #                   below. Firewall-safe (no solver call); mutually exclusive.
    oracle_backend: str = "table"
    # Max component size before the table backend gives up and falls back to IDA*.
    oracle_max_states: int = 1_000_000
    # LRU capacity (distinct board tables kept in the process-wide cache).
    oracle_cache_capacity: int = 256
    # Path to the precomputed sidecar (only used when oracle_backend="precomputed").
    # None/"" means no sidecar -> every lookup misses and falls back to online build.
    oracle_sidecar_path: str = ""
    # Potential function N(s) for the oracle advantage:
    #   "moves" — one slide of any distance counts 1 (default; legacy behavior).
    #   "cells" — a slide of k cells counts k (accumulated by slid cell count).
    oracle_potential: str = "moves"

    # --- RL-network teacher backend (oracle_backend="rlnet") ---------------
    # A *soft* process-supervision teacher: instead of the solver's exact N(s),
    # use a trained MaskableDQN's state value V(s)=max_a Q(s,a) as the potential
    # Φ=V (the env uses -scale*V so the unchanged adv=prev-curr yields
    # scale*(V'-V)).
    # NEVER calls solver (firewall-safe); mutually exclusive with solver backends.
    rlnet_ckpt_path: str = ""        # trained SB3 MaskableDQN checkpoint (required for rlnet)
    # Linear scale on V=max_a Q(s,a). Keep it small enough that the per-step
    # advantage stays inside the credit-assignment clip range: too large a scale
    # saturates the clip and flattens the graded ranking.
    rlnet_potential_scale: float = 1.0
    # Fixed potential V(s) for a SOLVED board, where the teacher is untrained and
    # its Q values are out-of-distribution. Must exceed the teacher's achievable
    # maxQ on non-solved boards so the winning move's advantage
    # scale*(solved_value - parent_maxQ) stays positive; raise it if your
    # checkpoint's maxQ approaches this value.
    rlnet_solved_value: float = 1.5
    rlnet_device: str = "cpu"        # forward device; CPU recommended (keeps the teacher off the training GPU)
    rlnet_max_pieces: int = 8        # action-space sizing, MUST match the teacher's training


def filter_config_kwargs(env_info: dict) -> dict:
    """Return only the keys of ``env_info`` that name RushHourEnvConfig fields."""
    field_names = {f.name for f in dataclasses.fields(RushHourEnvConfig)}
    return {k: v for k, v in env_info.items() if k in field_names}
