from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class SokobanEnvConfig:
    dim_room: Tuple[int, int] = (6, 6)
    max_steps: int = 100
    num_boxes: int = 3
    search_depth: int = 300
    grid_lookup: Dict[int, str] = field(default_factory=lambda: {0: "#", 1: "_", 2: "O", 3: "*", 4: "X", 5: "P", 6: "S"})
    grid_vocab: Dict[str, str] = field(default_factory=lambda: {"#": "wall", "_": "empty", "O": "target", "*": "box on target", "X": "box", "P": "player", "S": "player on target"})
    action_lookup: Dict[int, str] = field(default_factory=lambda: {1: "Up", 2: "Down", 3: "Left", 4: "Right"})
    dim_x: Optional[int] = None
    dim_y: Optional[int] = None
    render_mode: str = "text"
    observation_format: str = "grid"
    max_reset_retries: int = 50
    reward_mode: str = "default"  # "default" or "binary"
    max_repeated_obs: int = 0  # terminate after N consecutive identical observations; 0 = disabled
    deadlock_detection: str = "solver"  # "off", "pattern", or "solver"
    deadlock_solver_max_nodes: int = 200000  # max A* nodes for solver-based deadlock detection
    game_over_on_deadlock: int = 3  # consecutive deadlock detections before game over; 0=disabled, 1=immediate
    compute_oracle_advantage: bool = True  # whether to compute oracle advantage via solver each step (disable for validation to save time)
    # Reuse the previous N(s_{t+1}) as the next N(s_t), reducing oracle calls
    # after the first step. Equivalence is covered by test_sokoban_oracle_cache.py.
    oracle_advantage_cache: bool = False

    def __post_init__(self) -> None:
        if self.dim_x is not None and self.dim_y is not None:
            self.dim_room = (self.dim_x, self.dim_y)
        if self.observation_format not in {"grid", "coord", "grid_coord"}:
            raise ValueError(f"Unsupported observation_format: {self.observation_format}")
        if self.max_reset_retries < 1:
            raise ValueError(
                f"max_reset_retries must be >= 1, got {self.max_reset_retries}"
            )
        if self.reward_mode not in {"default", "binary"}:
            raise ValueError(f"Unsupported reward_mode: {self.reward_mode}")
        if self.max_repeated_obs < 0:
            raise ValueError(
                f"max_repeated_obs must be >= 0, got {self.max_repeated_obs}"
            )
        if self.deadlock_detection not in {"off", "pattern", "solver"}:
            raise ValueError(
                f"Unsupported deadlock_detection: {self.deadlock_detection}"
            )
        if self.deadlock_solver_max_nodes < 1:
            raise ValueError(
                f"deadlock_solver_max_nodes must be >= 1, got {self.deadlock_solver_max_nodes}"
            )
        if self.game_over_on_deadlock < 0:
            raise ValueError(
                f"game_over_on_deadlock must be >= 0, got {self.game_over_on_deadlock}"
            )
