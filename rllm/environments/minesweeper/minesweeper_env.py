"""Minesweeper game environment for rllm.

Adapted from gem/gem/envs/game_env/minesweeper.py (AxonRL / gem framework).
The environment wraps the core Minesweeper game logic into the rllm BaseEnv
interface (4-tuple step, from_dict factory, success flag).

Thread-safety: Each instance uses a private ``random.Random`` object so that
concurrent instances in training do not share global random state.
"""

import logging
import random
import re
import time
from collections import deque
from typing import Any

from rllm.environments.base.base_env import BaseEnv
from rllm.environments.minesweeper.oracle import (
    INF as _ORACLE_INF,
    oracle_advantage,
    solver_value_K,
)

logger = logging.getLogger(__name__)

# Difficulty presets ---------------------------------------------------------
DIFFICULTY_CONFIGS = {
    "id": {"rows": 6, "cols": 6, "num_mines": 7, "max_steps": 36},
    "unseen": {"rows": 7, "cols": 7, "num_mines": 10, "max_steps": 49},
}

# Reward constants -----------------------------------------------------------
# Binary reward: 1.0 on success, 0.0 otherwise.  No intermediate shaping.
_SUCCESS_REWARD = 1.0
_DEFAULT_REWARD = 0.0


class MinesweeperEnv(BaseEnv):
    """Text-based Minesweeper environment.

    Observation:
        A string containing step feedback text and the current ASCII board.

    Action space:
        A string in the format ``reveal R C`` or ``flag R C`` (parsed from the
        agent's ````` action ````` block by the agent, but here we accept the
        raw text produced by the agent).

    Reward:
        Binary reward: 1.0 on success (all non-mine cells revealed), 0.0
        otherwise.  No intermediate shaping rewards.

    Invalid actions:
        Format errors, out-of-bounds, already-revealed/flagged cells do NOT
        terminate the episode.  The environment returns ``info["action_is_valid"]
        = False`` so the agent can receive corrective feedback and retry.

    Termination:
        * Mine hit: terminates (reward 0.0).
        * Max turns exceeded: terminates (reward 0.0).
        * All non-mine cells revealed: terminates (reward 1.0).

    Success:
        ``info["success"]`` is ``True`` when all non-mine cells are revealed.
    """

    # The action pattern the *environment* recognises.  The *agent* wraps the
    # action in  ``` ``` blocks; parsing those is the agent's job.
    _ACTION_PATTERN = re.compile(r"(reveal|flag)\s+(\d+)\s+(\d+)", re.IGNORECASE)

    def __init__(
        self,
        rows: int = 8,
        cols: int = 8,
        num_mines: int = 10,
        max_steps: int = 64,
        seed: int | None = None,
        mine_positions: list[list[int]] | None = None,
        first_click: list[int] | None = None,
        compute_oracle_advantage: bool = False,
        oracle_solver_mode: str = "strong",
        **kwargs,
    ):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.num_mines = num_mines
        self.max_steps = max_steps
        self.seed = seed

        # Pre-generated layout (optional).  When both are set, ``reset()``
        # places mines and auto-reveals the first_click cell so that every
        # rollout of the same data sample starts from an identical board.
        self._preset_mine_positions: list[list[int]] | None = mine_positions
        self._preset_first_click: list[int] | None = first_click

        # --- Solver-based oracle advantage (credit assignment) ---
        # When enabled, step() emits info['oracle_advantage'] = K(s_t) - K(s_{t+1}),
        # the linear-potential signal.
        # K(s) is computed by a peek-free solver rollout that DISCARDS agent flags.
        # NOTE: meaningful only with a deterministic board (preset mine_positions +
        # first_click); under lazy mine placement the pre-first-reveal advantage
        # is emitted as 0 (the board is undetermined). The training pipeline
        # consumes info['oracle_advantage'] generically (see Sokoban).
        self.compute_oracle_advantage = compute_oracle_advantage
        self.oracle_solver_mode = oracle_solver_mode
        self._last_K: float | None = None  # cached K(s_{t+1}) -> next step's K(s_t)
        # Per-step solver wall time (ablation profiling). Accrued inside _oracle_K
        # (the single chokepoint for solver_value_K), reset at each step() entry,
        # surfaced via info["solver_time"].
        self._solver_time_step: float = 0.0

        # Per-instance RNG for thread safety.
        self._rng = random.Random(seed)

        # Board state (initialised in reset).
        self.grid: list[list[int]] = []
        self.revealed: list[list[bool]] = []
        self.flags: list[list[bool]] = []
        self.first_reveal: bool = True
        self.turn_count: int = 0
        self._done: bool = False
        self._last_feedback: str = ""

    # ------------------------------------------------------------------
    # BaseEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> tuple[str, dict]:
        """Reset the board.

        If ``mine_positions`` and ``first_click`` were supplied at construction
        time, mines are placed deterministically and the first click is
        auto-revealed so every rollout of the same data sample starts from an
        identical partially-revealed board.  Otherwise the board starts fully
        unrevealed and mines are placed lazily on the agent's first reveal
        (original behaviour).
        """
        self._rng = random.Random(self.seed)

        self.grid = [[0] * self.cols for _ in range(self.rows)]
        self.revealed = [[False] * self.cols for _ in range(self.rows)]
        self.flags = [[False] * self.cols for _ in range(self.rows)]
        self.first_reveal = True
        self.turn_count = 0
        self._done = False
        self._last_feedback = ""
        self._last_K = None  # reset oracle-advantage K cache

        # --- Pre-generated layout: place mines & auto-reveal first click ---
        if self._preset_mine_positions is not None and self._preset_first_click is not None:
            for r, c in self._preset_mine_positions:
                self.grid[r][c] = -1
            self._calculate_adjacent_numbers()
            self.first_reveal = False  # mines already placed; skip lazy placement

            fc_r, fc_c = self._preset_first_click
            self._flood_reveal(fc_r, fc_c)

        # The auto-revealed first click can flood-fill the entire board, leaving
        # only mines hidden. That state is already a win, so mark the episode
        # terminal here — otherwise the agent is forced to act on a solved board
        # where every remaining cell is a mine and any reveal is an instant loss.
        won = self.success()
        if won:
            self._done = True

        board_str = self._render_board()
        obs = f"Here is the current board layout:\n{board_str}\nEnter your move."
        info = {
            "success": won,
            "board": board_str,
            "suffix": f"Here is the current board layout:\n{board_str}\nEnter your move.",
        }
        return obs, info

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        """Execute one action, optionally attaching the solver oracle advantage.

        Thin wrapper over :meth:`_step_core`. When ``compute_oracle_advantage`` is
        enabled, sets ``info['oracle_advantage'] = K(s_t) - K(s_{t+1})`` (the
        linear-potential signal). Exactly one solver rollout per
        valid non-terminal action: ``K(s_t)`` is reused from the previous step's
        ``K(s_{t+1})`` cache, and terminal/invalid actions need no rollout.
        """
        if not self.compute_oracle_advantage:
            return self._step_core(action)

        # Per-step solver profiling (ablation). Reset before any _oracle_K call so
        # info["solver_time"] reflects only this step's rollouts.
        self._solver_time_step = 0.0

        # Already-terminal board: no signal.
        if self._done:
            obs, reward, done, info = self._step_core(action)
            info["oracle_advantage"] = 0.0
            info["solver_time"] = self._solver_time_step
            return obs, reward, done, info

        # K(s_t): reuse cached K(s_{t+1}) from the previous step when available.
        k_before = self._last_K if self._last_K is not None else self._oracle_K()

        obs, reward, done, info = self._step_core(action)

        if k_before is None:
            # Board was undetermined before the action (lazy mine placement, no
            # presets): K(s_t) is undefined, so emit a neutral 0 and (re)seed the
            # cache now that mines may have been placed by this action.
            info["oracle_advantage"] = 0.0
            self._last_K = self._oracle_K()
        else:
            adv, k_after = self._compute_oracle_adv(k_before, info)
            info["oracle_advantage"] = adv
            self._last_K = k_after
        info["solver_time"] = self._solver_time_step
        return obs, reward, done, info

    def _oracle_K(self) -> float | None:
        """K(s) for the current board, or None if mines are not yet placed."""
        if self.first_reveal:
            return None  # lazy placement: board undetermined, K undefined
        _t0 = time.time()
        result = solver_value_K(
            self.rows, self.cols, self.grid, self.revealed, self.flags,
            total_mines=self.num_mines, mode=self.oracle_solver_mode,
        )
        self._solver_time_step += time.time() - _t0
        return result

    def _compute_oracle_adv(self, k_before: float, info: dict) -> tuple[float, float]:
        """Return (advantage, K(s_{t+1})) for the linear-potential oracle.

        Branches:
          * invalid action  -> 0 (state unchanged), cache unchanged
          * success (win)   -> K(s_t) (K_after = 0); rescue-win capped at #safe
          * mine_hit        -> -K(s_t) (catastrophe); 0 if already doomed
          * normal          -> oracle_advantage(K(s_t), K(s_{t+1}))
        """
        if not info.get("action_is_valid", True):
            return 0.0, k_before
        if info.get("success"):
            n_safe = self.rows * self.cols - self.num_mines
            adv = float(n_safe) if k_before == _ORACLE_INF else float(k_before)
            return adv, 0.0
        if info.get("mine_hit"):
            adv = 0.0 if k_before == _ORACLE_INF else -float(k_before)
            return adv, _ORACLE_INF
        k_after = self._oracle_K()
        if k_after is None:  # should not happen post-first-reveal, but be safe
            return 0.0, k_before
        return oracle_advantage(k_before, k_after), k_after

    def _step_core(self, action: str) -> tuple[str, float, bool, dict]:
        """Execute one action on the board.

        Returns:
            ``(observation, reward, done, info)`` where *observation* is the
            board-state string (same as ``info["suffix"]``).  The per-turn
            feedback text is stored in ``info["feedback"]`` for consumers that
            need it, but is **not** the primary observation.
        """
        if self._done:
            board_str = self._render_board()
            observation = f"Here is the current board layout:\n{board_str}"
            return (
                observation,
                _DEFAULT_REWARD,
                True,
                {"success": self.success(), "board": board_str,
                 "action_is_valid": True, "action_is_effective": False,
                 "feedback": "The game is already over.",
                 "suffix": observation},
            )

        self.turn_count += 1
        action_str = str(action).strip()

        # Try to parse the action.
        match = self._ACTION_PATTERN.search(action_str)
        if match is None:
            # Format error -> invalid action, do NOT terminate.
            feedback = (
                f"At turn {self.turn_count}, you did not provide a valid action. "
                "Expected format: 'reveal R C' or 'flag R C'."
            )
            done = self.turn_count >= self.max_steps
            if done:
                self._done = True
            self._last_feedback = feedback
            board_str = self._render_board()
            observation = f"Here is the current board layout:\n{board_str}"
            return (
                observation,
                _DEFAULT_REWARD,
                done,
                {"success": False, "board": board_str,
                 "action_is_valid": False, "action_is_effective": False,
                 "format_error": True,
                 "feedback": feedback,
                 "suffix": observation},
            )

        action_type = match.group(1).lower()
        row = int(match.group(2))
        col = int(match.group(3))

        # Out-of-bounds check.
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            feedback = (
                f"At turn {self.turn_count}, you chose cell ({row}, {col}), "
                "which is outside the bounds of the grid."
            )
            done = self.turn_count >= self.max_steps
            if done:
                self._done = True
            self._last_feedback = feedback
            board_str = self._render_board()
            observation = f"Here is the current board layout:\n{board_str}"
            return (
                observation,
                _DEFAULT_REWARD,
                done,
                {"success": False, "board": board_str,
                 "action_is_valid": False, "action_is_effective": False,
                 "out_of_bounds": True,
                 "feedback": feedback,
                 "suffix": observation},
            )

        # ---- REVEAL ----
        if action_type == "reveal":
            return self._handle_reveal(row, col)

        # ---- FLAG ----
        if action_type == "flag":
            return self._handle_flag(row, col)

        # Unknown action type (should not happen with the regex, but just in case).
        feedback = (
            f"At turn {self.turn_count}, you chose an invalid action '{action_type}'. "
            "Valid actions are 'reveal' or 'flag'."
        )
        done = self.turn_count >= self.max_steps
        if done:
            self._done = True
        self._last_feedback = feedback
        board_str = self._render_board()
        observation = f"Here is the current board layout:\n{board_str}"
        return (
            observation,
            _DEFAULT_REWARD,
            done,
            {"success": False, "board": board_str,
             "action_is_valid": False, "action_is_effective": False,
             "feedback": feedback,
             "suffix": observation},
        )

    def close(self):
        """No external resources to release."""
        pass

    def success(self) -> bool:
        """Return True when all non-mine cells have been revealed."""
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] != -1 and not self.revealed[r][c]:
                    return False
        return True

    def render(self) -> str:
        """Return the current ASCII board."""
        return self._render_board()

    @staticmethod
    def from_dict(env_info: dict) -> "MinesweeperEnv":
        """Create a MinesweeperEnv from a serialised dictionary.

        Recognised keys: rows, cols, num_mines, max_steps, seed, difficulty.
        If ``difficulty`` is present it overrides rows/cols/num_mines/max_steps.

        For preset (deterministic) boards — both ``mine_positions`` and
        ``first_click`` supplied — the layout is the source of truth, so this
        normalises ``num_mines`` to ``len(mine_positions)`` and validates the
        board invariants that the solver-based oracle and win-check rely on:
          * every mine lies inside the grid and there are no duplicates,
          * ``first_click`` lies inside the grid and is NOT a mine.
        A mismatch here would silently corrupt the oracle's ``total_mines``
        (wrong CSP probabilities) or the ``hidden == total_mines`` win-check,
        so we fail loudly instead.
        """
        info = dict(env_info)

        difficulty = info.pop("difficulty", None)
        if difficulty and difficulty in DIFFICULTY_CONFIGS:
            preset = DIFFICULTY_CONFIGS[difficulty]
            info.setdefault("rows", preset["rows"])
            info.setdefault("cols", preset["cols"])
            info.setdefault("num_mines", preset["num_mines"])
            info.setdefault("max_steps", preset["max_steps"])

        allowed = {"rows", "cols", "num_mines", "max_steps", "seed",
                   "mine_positions", "first_click",
                   "compute_oracle_advantage", "oracle_solver_mode"}
        filtered = {k: v for k, v in info.items() if k in allowed}

        # --- Preset board validation & num_mines normalization ---
        # mine_positions / first_click may arrive as numpy arrays (from parquet);
        # normalise to plain python ints so downstream list-indexing is exact.
        mine_positions = filtered.get("mine_positions")
        first_click = filtered.get("first_click")
        if mine_positions is not None and first_click is not None:
            norm_mines = [(int(r), int(c)) for r, c in mine_positions]
            fc_r, fc_c = int(first_click[0]), int(first_click[1])

            rows = int(filtered.get("rows", 8))
            cols = int(filtered.get("cols", 8))

            for r, c in norm_mines:
                if not (0 <= r < rows and 0 <= c < cols):
                    raise ValueError(
                        f"MinesweeperEnv.from_dict: mine ({r}, {c}) is outside "
                        f"the {rows}x{cols} grid."
                    )
            if len(set(norm_mines)) != len(norm_mines):
                raise ValueError(
                    "MinesweeperEnv.from_dict: mine_positions contains duplicates."
                )
            if not (0 <= fc_r < rows and 0 <= fc_c < cols):
                raise ValueError(
                    f"MinesweeperEnv.from_dict: first_click ({fc_r}, {fc_c}) is "
                    f"outside the {rows}x{cols} grid."
                )
            if (fc_r, fc_c) in set(norm_mines):
                raise ValueError(
                    f"MinesweeperEnv.from_dict: first_click ({fc_r}, {fc_c}) "
                    "lands on a mine; a preset board must open on a safe cell."
                )

            filtered["mine_positions"] = [list(m) for m in norm_mines]
            filtered["first_click"] = [fc_r, fc_c]

            # Layout is authoritative: align num_mines so the oracle's
            # total_mines and the win-check use the true mine count.
            true_num = len(norm_mines)
            if int(filtered.get("num_mines", true_num)) != true_num:
                logger.warning(
                    "MinesweeperEnv.from_dict: num_mines=%s != len(mine_positions)=%s; "
                    "using %s (preset layout is authoritative).",
                    filtered.get("num_mines"), true_num, true_num,
                )
            filtered["num_mines"] = true_num

        return MinesweeperEnv(**filtered)

    @staticmethod
    def is_multithread_safe() -> bool:
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _handle_reveal(self, row: int, col: int) -> tuple[str, float, bool, dict]:
        # Already revealed or flagged -> invalid action, do NOT terminate.
        # This MUST be checked before mine placement and the mine-hit test so
        # that a flagged cell is protected symmetrically: clicking a flagged
        # mine is invalid (the flag did its job), not an instant detonation.
        if self.revealed[row][col] or self.flags[row][col]:
            state = "revealed" if self.revealed[row][col] else "flagged"
            feedback = (
                f"At turn {self.turn_count}, you chose to reveal cell ({row}, {col}), "
                f"which has already been {state}."
            )
            done = self.turn_count >= self.max_steps
            if done:
                self._done = True
            self._last_feedback = feedback
            board_str = self._render_board()
            observation = f"Here is the current board layout:\n{board_str}"
            return (
                observation,
                _DEFAULT_REWARD,
                done,
                {"success": self.success(), "board": board_str,
                 "action_is_valid": False, "action_is_effective": False,
                 "already_revealed": True,
                 "feedback": feedback,
                 "suffix": observation},
            )

        # First reveal -> place mines (ensures safe zone around first click).
        if self.first_reveal:
            self._setup_mines(row, col)
            self.first_reveal = False

        # Hit a mine.
        if self.grid[row][col] == -1:
            self._done = True
            self.revealed[row][col] = True  # Show the mine
            board_str = self._render_board()
            feedback = f"Game over! You hit a mine at ({row}, {col})."
            self._last_feedback = feedback
            observation = f"Here is the current board layout:\n{board_str}"
            return (
                observation,
                _DEFAULT_REWARD,
                True,
                {"success": False, "board": board_str, "mine_hit": True,
                 "action_is_valid": True, "action_is_effective": True,
                 "feedback": feedback,
                 "suffix": observation},
            )

        # Valid reveal.
        self._flood_reveal(row, col)

        if self.success():
            self._done = True
            board_str = self._render_board()
            feedback = "Congratulations! You have successfully cleared the Minesweeper board."
            self._last_feedback = feedback
            observation = f"Here is the current board layout:\n{board_str}"
            return (
                observation,
                _SUCCESS_REWARD,
                True,
                {"success": True, "board": board_str,
                 "action_is_valid": True, "action_is_effective": True,
                 "feedback": feedback,
                 "suffix": observation},
            )

        done = self.turn_count >= self.max_steps
        if done:
            self._done = True
            feedback = "You have reached the maximum number of turns."
        else:
            feedback = (
                f"At turn {self.turn_count}, you successfully revealed cell ({row}, {col})."
            )
        self._last_feedback = feedback
        board_str = self._render_board()
        observation = f"Here is the current board layout:\n{board_str}"
        return (
            observation,
            _DEFAULT_REWARD,
            done,
            {"success": self.success(), "board": board_str,
             "action_is_valid": True, "action_is_effective": True,
             "feedback": feedback,
             "suffix": observation},
        )

    def _handle_flag(self, row: int, col: int) -> tuple[str, float, bool, dict]:
        action_is_valid = True
        action_is_effective = True
        if self.revealed[row][col]:
            feedback = (
                f"At turn {self.turn_count}, you chose to flag cell ({row}, {col}), "
                "which has already been revealed."
            )
            action_is_valid = False
            action_is_effective = False
        else:
            self.flags[row][col] = not self.flags[row][col]
            verb = "added" if self.flags[row][col] else "removed"
            feedback = (
                f"At turn {self.turn_count}, you {verb} a flag on cell ({row}, {col})."
            )

        done = self.turn_count >= self.max_steps
        if done:
            self._done = True
        self._last_feedback = feedback
        board_str = self._render_board()
        observation = f"Here is the current board layout:\n{board_str}"
        info = {
            "success": self.success(),
            "board": board_str,
            "action_is_valid": action_is_valid,
            "action_is_effective": action_is_effective,
            "feedback": feedback,
            "suffix": observation,
        }
        return (
            observation,
            _DEFAULT_REWARD,
            done,
            info,
        )

    # ---- Mine placement & adjacency ----

    def _setup_mines(self, safe_row: int, safe_col: int):
        """Place mines avoiding a safe zone around the first click.

        The preferred safe zone is 3x3 around ``(safe_row, safe_col)``.
        If the board is too small for the 3x3 zone to leave enough room for
        all mines, the zone is narrowed to just the clicked cell.  If even
        that is impossible (e.g. more mines than non-clicked cells), the
        number of mines is clamped to available positions.
        """
        total_cells = self.rows * self.cols
        safe_3x3 = {
            (safe_row + dr, safe_col + dc)
            for dr in range(-1, 2)
            for dc in range(-1, 2)
            if 0 <= safe_row + dr < self.rows and 0 <= safe_col + dc < self.cols
        }
        available_with_3x3 = total_cells - len(safe_3x3)

        if available_with_3x3 >= self.num_mines:
            # Normal case: use full 3×3 safe zone.
            safe_zone = safe_3x3
        elif total_cells - 1 >= self.num_mines:
            # Board too small for 3×3 but can still protect the click cell.
            safe_zone = {(safe_row, safe_col)}
        else:
            # Extreme edge-case: clamp mines to what fits.
            safe_zone = {(safe_row, safe_col)}
            self.num_mines = total_cells - 1

        mines: set[tuple[int, int]] = set()
        while len(mines) < self.num_mines:
            r = self._rng.randint(0, self.rows - 1)
            c = self._rng.randint(0, self.cols - 1)
            if (r, c) in mines:
                continue
            if (r, c) in safe_zone:
                continue
            mines.add((r, c))
            self.grid[r][c] = -1
        self._calculate_adjacent_numbers()

    def _calculate_adjacent_numbers(self):
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == -1:
                    continue
                count = 0
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr][nc] == -1:
                        count += 1
                self.grid[r][c] = count

    # ---- Flood-fill reveal ----

    def _flood_reveal(self, row: int, col: int) -> int:
        """BFS flood-fill starting from ``(row, col)``.

        Returns the number of newly revealed cells.
        """
        queue = deque([(row, col)])
        self.revealed[row][col] = True
        count = 1

        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]

        while queue:
            cr, cc = queue.popleft()
            if self.grid[cr][cc] == 0:
                for dr, dc in directions:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        if not self.revealed[nr][nc] and not self.flags[nr][nc]:
                            self.revealed[nr][nc] = True
                            count += 1
                            queue.append((nr, nc))
        return count

    # ---- Board rendering ----

    def _render_board(self) -> str:
        """Render the board as an ASCII grid.

        Column headers are right-justified.  Each cell is 3 characters wide.
        Symbols: ``.`` unrevealed, ``F`` flagged, ``*`` mine, ``0``-``8`` count.
        """
        header = "   " + " ".join(str(c).rjust(2) for c in range(self.cols))
        lines = [header]
        for r in range(self.rows):
            row_str = f"{r:2} "
            for c in range(self.cols):
                if self.revealed[r][c]:
                    if self.grid[r][c] == -1:
                        row_str += " * "
                    else:
                        row_str += f" {self.grid[r][c]} "
                elif self.flags[r][c]:
                    row_str += " F "
                else:
                    row_str += " . "
            lines.append(row_str)
        return "\n".join(lines)
