from __future__ import annotations

import os
import sys
import threading

import numpy as np

_SB3_DIR = os.environ.get("RUSH_HOUR_SB3_DIR") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    )))),
    "stable-baselines3",
)


class RLNetPotential:
    _instances: dict[tuple, "RLNetPotential"] = {}
    _lock = threading.Lock()

    def __init__(self, ckpt_path: str, device: str, scale: float, max_pieces: int,
                 width: int, height: int, solved_value: float = 1.5):
        self.ckpt_path = ckpt_path
        self.device = device
        self.scale = float(scale)
        self.solved_value = float(solved_value)
        self.max_pieces = int(max_pieces)
        self.width = int(width)
        self.height = int(height)

        # --- lazy imports (only when rlnet is actually used) ---
        if not os.path.isdir(_SB3_DIR):
            raise FileNotFoundError(
                f"SB3 teacher package not found at {_SB3_DIR!r}. The learned DQN "
                "value-network oracle (ORACLE_BACKEND=rlnet) needs the external "
                "stable-baselines3 / rush_hour_rl package, which is not bundled "
                "with this repository. Set RUSH_HOUR_SB3_DIR to your copy, or use "
                "a bundled backend (e.g. ORACLE_BACKEND=solver)."
            )
        if _SB3_DIR not in sys.path:
            sys.path.insert(0, _SB3_DIR)

        from rush_hour_rl.algos.masked_dqn import MaskableDQN
        from rush_hour_rl.core.action_codec import ActionCodec
        from rush_hour_rl.core.obs_encoder import encode_board

        self._encode_board = encode_board
        self._codec = ActionCodec(self.width, self.height, self.max_pieces)

        import torch as th

        self._th = th
        self._orig_num_threads = th.get_num_threads()
        self._model = MaskableDQN.load(ckpt_path, device=device)
        self._model.policy.set_training_mode(False)

        self._value_cache: dict[str, float] = {}

    _CACHE_MAX = 200_000  # ~tens of MB of str->float; bounds the value cache.

    # ------------------------------------------------------------------
    @classmethod
    def get(cls, ckpt_path: str, device: str, scale: float, max_pieces: int,
            width: int, height: int, solved_value: float = 1.5) -> "RLNetPotential":
        key = (ckpt_path, device, int(max_pieces), int(width), int(height))
        inst = cls._instances.get(key)
        if inst is not None:
            return inst
        with cls._lock:
            inst = cls._instances.get(key)
            if inst is None:
                inst = cls(ckpt_path, device, scale, max_pieces, width, height,
                           solved_value=solved_value)
                cls._instances[key] = inst
            return inst

    # ------------------------------------------------------------------
    def _value(self, board) -> float:
        if board.solved:
            return self.solved_value

        key = board.to_string()
        cached = self._value_cache.get(key)
        if cached is not None:
            return cached

        th = self._th
        grid = self._encode_board(board)                       # (C,H,W) float32
        mask = np.asarray(self._codec.legal_mask(board), dtype=np.float32)  # (n_actions,)
        legal = np.flatnonzero(mask > 0.5)
        if not len(legal):
            self._cache_put(key, 0.0)
            return 0.0
        flat = np.concatenate([grid.reshape(-1), mask])        # (grid_dim + n_actions,)
        obs_t = th.as_tensor(flat[None], dtype=th.float32, device=self._model.device)
        th.set_num_threads(1)
        try:
            with th.no_grad():
                q = self._model.q_net(obs_t).cpu().numpy().reshape(-1)  # (n_actions,)
        finally:
            th.set_num_threads(self._orig_num_threads)
        if q.shape[0] != mask.shape[0]:
            raise ValueError(
                f"rlnet obs/action mismatch: q_net emitted {q.shape[0]} Q-values "
                f"but action mask has {mask.shape[0]} slots. Check rlnet_max_pieces/"
                f"width/height match the checkpoint's training config."
            )
        val = float(q[legal].max())
        self._cache_put(key, val)
        return val

    def _cache_put(self, key: str, val: float) -> None:
        if len(self._value_cache) >= self._CACHE_MAX:
            self._value_cache.clear()
        self._value_cache[key] = val

    def potential(self, board, *, scale: float | None = None,
                  solved_value: float | None = None) -> float:
        s = self.scale if scale is None else float(scale)
        sv = self.solved_value if solved_value is None else float(solved_value)
        if board.solved:
            return -s * sv
        return -s * self._value(board)

    @classmethod
    def clear(cls) -> None:
        with cls._lock:
            cls._instances.clear()
