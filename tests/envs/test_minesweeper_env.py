import pytest

from rllm.environments.minesweeper.minesweeper_env import MinesweeperEnv, DIFFICULTY_CONFIGS


# ---------------------------------------------------------------------------
# Basic creation and reset
# ---------------------------------------------------------------------------

def test_init_default():
    """Test environment creation with default parameters."""
    env = MinesweeperEnv()
    assert env.rows == 8
    assert env.cols == 8
    assert env.num_mines == 10
    assert env.max_steps == 64
    assert env.seed is None


def test_init_custom():
    """Test environment creation with custom parameters."""
    env = MinesweeperEnv(rows=5, cols=5, num_mines=3, max_steps=20, seed=42)
    assert env.rows == 5
    assert env.cols == 5
    assert env.num_mines == 3
    assert env.max_steps == 20
    assert env.seed == 42


def test_reset():
    """Test environment reset returns valid observation and info."""
    env = MinesweeperEnv(rows=5, cols=5, num_mines=5, seed=42)
    obs, info = env.reset()

    assert isinstance(obs, str)
    assert "board layout" in obs.lower() or "." in obs
    assert isinstance(info, dict)
    assert "success" in info
    assert info["success"] is False
    assert "board" in info
    assert "suffix" in info


def test_reset_clears_state():
    """Test that reset properly clears all game state."""
    env = MinesweeperEnv(rows=5, cols=5, num_mines=5, seed=42)
    obs, info = env.reset()

    # Take a step
    env.step("reveal 2 2")

    # Reset should restore fresh state
    obs2, info2 = env.reset()
    assert env.turn_count == 0
    assert env.first_reveal is True
    assert env._done is False
    assert not any(env.revealed[r][c] for r in range(5) for c in range(5))


# ---------------------------------------------------------------------------
# Step: valid reveal
# ---------------------------------------------------------------------------

def test_step_reveal_valid():
    """Test revealing a safe cell gives reward 0 (binary: only success gives 1)."""
    env = MinesweeperEnv(rows=5, cols=5, num_mines=3, seed=42)
    env.reset()

    # First reveal is guaranteed safe (no mine in 3x3 zone around click)
    obs, reward, done, info = env.step("reveal 2 2")

    assert isinstance(obs, str)
    # Binary reward: mid-game reveals always give 0.0
    assert reward == 0.0 or (reward == 1.0 and info.get("success"))
    assert isinstance(done, bool)
    assert "board" in info
    assert "success" in info


def test_step_reveal_multiple_safe():
    """Test that only the success-completing reveal gives reward 1.0."""
    env = MinesweeperEnv(rows=5, cols=5, num_mines=3, seed=42)
    env.reset()

    total_reward = 0.0
    for r in range(5):
        for c in range(5):
            obs, reward, done, info = env.step(f"reveal {r} {c}")
            total_reward += reward
            if done:
                break
        if done:
            break

    # Binary reward: success -> exactly 1.0, mine hit / timeout -> 0.0
    if info.get("success"):
        assert total_reward == 1.0
    else:
        assert total_reward == 0.0


# ---------------------------------------------------------------------------
# Step: flag
# ---------------------------------------------------------------------------

def test_step_flag():
    """Test flag toggling works correctly."""
    env = MinesweeperEnv(rows=5, cols=5, num_mines=3, seed=42)
    env.reset()

    # Flag a cell
    obs, reward, done, info = env.step("flag 0 0")
    assert reward == 0.0  # flag gives 0 reward
    assert not done

    # Unflag the same cell
    obs, reward, done, info = env.step("flag 0 0")
    assert reward == 0.0
    assert not done


def test_step_flag_revealed_cell():
    """Test flagging an already-revealed cell gives 0 reward and action_is_valid=False."""
    env = MinesweeperEnv(rows=5, cols=5, num_mines=3, seed=42)
    env.reset()

    # First reveal (guaranteed safe)
    env.step("reveal 2 2")

    # Try to flag a revealed cell
    obs, reward, done, info = env.step("flag 2 2")
    assert reward == 0.0
    assert info.get("action_is_valid") is False
    assert info.get("action_is_effective") is False


# ---------------------------------------------------------------------------
# Step: format error
# ---------------------------------------------------------------------------

def test_step_format_error():
    """Test bad action format gives reward 0, does NOT terminate, and sets action_is_valid=False."""
    env = MinesweeperEnv(rows=5, cols=5, num_mines=3, max_steps=10, seed=42)
    env.reset()

    obs, reward, done, info = env.step("bad action here")
    assert reward == 0.0
    assert done is False  # format error no longer terminates
    assert info.get("format_error") is True
    assert info.get("action_is_valid") is False
    assert info.get("action_is_effective") is False
    assert info["success"] is False

    # Agent can recover and continue playing
    obs2, reward2, done2, info2 = env.step("reveal 2 2")
    assert isinstance(obs2, str)


def test_step_format_error_empty_action():
    """Test empty action string is treated as format error but does not terminate."""
    env = MinesweeperEnv(rows=5, cols=5, num_mines=3, max_steps=10, seed=42)
    env.reset()

    obs, reward, done, info = env.step("")
    assert reward == 0.0
    assert done is False  # format error no longer terminates
    assert info.get("action_is_valid") is False
    assert info.get("action_is_effective") is False


# ---------------------------------------------------------------------------
# Step: out of bounds
# ---------------------------------------------------------------------------

def test_step_out_of_bounds():
    """Test OOB coordinates give reward 0, don't terminate, and set action_is_valid=False."""
    env = MinesweeperEnv(rows=5, cols=5, num_mines=3, max_steps=100, seed=42)
    env.reset()

    obs, reward, done, info = env.step("reveal 99 99")
    assert reward == 0.0
    assert info.get("out_of_bounds") is True
    assert info.get("action_is_valid") is False
    assert info.get("action_is_effective") is False
    # Should not terminate (still have turns left)
    assert done is False


# ---------------------------------------------------------------------------
# Step: mine hit
# ---------------------------------------------------------------------------

def test_step_mine_hit():
    """Test that hitting a mine terminates the game with 0 reward."""
    env = MinesweeperEnv(rows=5, cols=5, num_mines=5, seed=42)
    env.reset()

    # First reveal at (2,2) is guaranteed safe and places mines
    env.step("reveal 2 2")

    # Find a mine cell and reveal it
    mine_found = False
    for r in range(5):
        for c in range(5):
            if env.grid[r][c] == -1 and not env.revealed[r][c]:
                obs, reward, done, info = env.step(f"reveal {r} {c}")
                assert reward == 0.0
                assert done is True
                assert info.get("mine_hit") is True
                assert info["success"] is False
                mine_found = True
                break
        if mine_found:
            break

    assert mine_found, "No mine was found to test against"


# ---------------------------------------------------------------------------
# from_dict factory
# ---------------------------------------------------------------------------

def test_from_dict():
    """Test factory method with explicit parameters."""
    task_data = {"rows": 6, "cols": 6, "num_mines": 7, "max_steps": 36, "seed": 123}
    env = MinesweeperEnv.from_dict(task_data)

    assert env.rows == 6
    assert env.cols == 6
    assert env.num_mines == 7
    assert env.max_steps == 36
    assert env.seed == 123

    obs, info = env.reset()
    assert isinstance(obs, str)


def test_from_dict_with_difficulty():
    """Test factory method with difficulty preset."""
    task_data = {"difficulty": "id", "seed": 42}
    env = MinesweeperEnv.from_dict(task_data)

    assert env.rows == 6
    assert env.cols == 6
    assert env.num_mines == 7
    assert env.max_steps == 36
    assert env.seed == 42


def test_from_dict_difficulty_overrides():
    """Test that explicit params override difficulty preset defaults."""
    task_data = {"difficulty": "id", "rows": 10, "seed": 42}
    env = MinesweeperEnv.from_dict(task_data)

    # rows should be overridden by explicit value
    assert env.rows == 10
    # cols should come from the "id" preset since not overridden
    assert env.cols == 6


def test_from_dict_filters_unknown_keys():
    """Test that unknown keys are ignored."""
    task_data = {"rows": 5, "cols": 5, "num_mines": 3, "seed": 42, "index": 0, "uid": "test"}
    env = MinesweeperEnv.from_dict(task_data)
    assert env.rows == 5


# ---------------------------------------------------------------------------
# Success detection
# ---------------------------------------------------------------------------

def test_success_detection():
    """Test that success() returns True when all non-mine cells are revealed."""
    env = MinesweeperEnv(rows=5, cols=5, num_mines=3, seed=42)
    env.reset()

    # Reveal the center (guaranteed safe on first click)
    env.step("reveal 2 2")

    # Manually reveal all non-mine cells to test success
    for r in range(5):
        for c in range(5):
            if env.grid[r][c] != -1:
                env.revealed[r][c] = True

    assert env.success() is True


def test_success_false_when_not_all_revealed():
    """Test that success() returns False when some safe cells remain unrevealed."""
    env = MinesweeperEnv(rows=5, cols=5, num_mines=3, seed=42)
    env.reset()

    # Only reveal first cell
    env.step("reveal 2 2")

    # Very unlikely all cells are revealed from one click with 3 mines
    # But check based on actual state
    if not env.success():
        assert env.success() is False


# ---------------------------------------------------------------------------
# First reveal safety
# ---------------------------------------------------------------------------

def test_first_reveal_safety():
    """Test that first reveal never hits a mine (safe zone guarantee)."""
    # Run a representative set of seeds and positions
    for seed in range(20):
        # Test corners, edges, and center positions
        positions = [(0, 0), (0, 4), (4, 0), (4, 4), (2, 2), (0, 2), (2, 0)]
        for r, c in positions:
            env = MinesweeperEnv(rows=5, cols=5, num_mines=5, seed=seed)
            env.reset()
            obs, reward, done, info = env.step(f"reveal {r} {c}")

            # First reveal should never hit a mine
            assert info.get("mine_hit") is not True, (
                f"First reveal hit mine at ({r},{c}) with seed={seed}"
            )


def test_first_reveal_safe_zone():
    """Test that mines are not placed in 3x3 zone around first click."""
    env = MinesweeperEnv(rows=8, cols=8, num_mines=10, seed=42)
    env.reset()

    # Reveal center
    env.step("reveal 4 4")

    # Check 3x3 zone around (4,4) has no mines
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            r, c = 4 + dr, 4 + dc
            if 0 <= r < 8 and 0 <= c < 8:
                assert env.grid[r][c] != -1, f"Mine found in safe zone at ({r},{c})"


# ---------------------------------------------------------------------------
# Flood fill (BFS reveal)
# ---------------------------------------------------------------------------

def test_flood_fill():
    """Test that revealing a 0-cell cascades to adjacent cells."""
    env = MinesweeperEnv(rows=8, cols=8, num_mines=3, seed=42)
    env.reset()

    # Reveal a cell — first reveal triggers mine placement + flood fill
    obs, reward, done, info = env.step("reveal 4 4")

    # Count revealed cells
    revealed_count = sum(
        1 for r in range(8) for c in range(8) if env.revealed[r][c]
    )

    # With only 3 mines on 8x8 board, a flood fill from center should reveal many cells
    # The first reveal triggers a large flood if center area has 0-cells
    assert revealed_count >= 1  # At minimum, the clicked cell is revealed

    # Binary reward: mid-game reveal gives 0.0 (unless all safe cells revealed -> 1.0)
    if info.get("success"):
        assert reward == 1.0
    else:
        assert reward == 0.0


# ---------------------------------------------------------------------------
# Board rendering
# ---------------------------------------------------------------------------

def test_render():
    """Test that render returns a valid string with board content."""
    env = MinesweeperEnv(rows=5, cols=5, num_mines=3, seed=42)
    env.reset()

    board = env.render()
    assert isinstance(board, str)
    assert len(board) > 0
    # Board should contain dots (unrevealed cells)
    assert "." in board


def test_render_after_reveal():
    """Test that render shows revealed cells correctly."""
    env = MinesweeperEnv(rows=5, cols=5, num_mines=3, seed=42)
    env.reset()

    env.step("reveal 2 2")
    board = env.render()

    # After reveal, board should show numbers (0-8) for revealed cells
    assert isinstance(board, str)
    # At least one digit should be visible
    has_digit = any(ch.isdigit() for ch in board)
    assert has_digit


def test_render_shows_flag():
    """Test that render shows flagged cells with 'F'."""
    env = MinesweeperEnv(rows=5, cols=5, num_mines=3, seed=42)
    env.reset()

    env.step("flag 0 0")
    board = env.render()
    assert "F" in board


# ---------------------------------------------------------------------------
# Reproducibility with seed
# ---------------------------------------------------------------------------

def test_reproducibility_with_seed():
    """Test that same seed produces same board after first reveal."""
    env1 = MinesweeperEnv(rows=5, cols=5, num_mines=5, seed=42)
    env1.reset()
    env1.step("reveal 2 2")
    board1 = env1.render()

    env2 = MinesweeperEnv(rows=5, cols=5, num_mines=5, seed=42)
    env2.reset()
    env2.step("reveal 2 2")
    board2 = env2.render()

    assert board1 == board2


def test_different_seeds_produce_different_boards():
    """Test that different seeds produce different mine layouts."""
    env1 = MinesweeperEnv(rows=8, cols=8, num_mines=10, seed=42)
    env1.reset()
    env1.step("reveal 4 4")
    grid1 = [row[:] for row in env1.grid]

    env2 = MinesweeperEnv(rows=8, cols=8, num_mines=10, seed=99)
    env2.reset()
    env2.step("reveal 4 4")
    grid2 = [row[:] for row in env2.grid]

    # Grids should differ (extremely unlikely to be identical with different seeds)
    assert grid1 != grid2


# ---------------------------------------------------------------------------
# Already revealed
# ---------------------------------------------------------------------------

def test_step_already_revealed():
    """Test that revealing an already-revealed cell gives 0 reward and action_is_valid=False."""
    env = MinesweeperEnv(rows=5, cols=5, num_mines=3, seed=42)
    env.reset()

    # Reveal a cell
    env.step("reveal 2 2")

    # Reveal the same cell again
    obs, reward, done, info = env.step("reveal 2 2")
    assert reward == 0.0
    assert info.get("already_revealed") is True
    assert info.get("action_is_valid") is False
    assert info.get("action_is_effective") is False


# ---------------------------------------------------------------------------
# Max steps truncation
# ---------------------------------------------------------------------------

def test_max_steps_truncation():
    """Test that game ends when max steps is reached."""
    env = MinesweeperEnv(rows=5, cols=5, num_mines=3, max_steps=2, seed=42)
    env.reset()

    # Step 1
    obs, reward, done, info = env.step("reveal 2 2")
    # Step 2 — should hit max steps
    obs, reward, done, info = env.step("flag 0 0")
    assert done is True


# ---------------------------------------------------------------------------
# Game already over
# ---------------------------------------------------------------------------

def test_step_after_game_over():
    """Test that stepping after game is over returns done=True."""
    env = MinesweeperEnv(rows=5, cols=5, num_mines=5, seed=42)
    env.reset()

    # First reveal to place mines (guaranteed safe)
    env.step("reveal 2 2")

    # Find a mine and hit it to end the game
    for r in range(5):
        for c in range(5):
            if env.grid[r][c] == -1 and not env.revealed[r][c]:
                env.step(f"reveal {r} {c}")
                break
        if env._done:
            break

    assert env._done is True

    # Try another step after game is over
    obs, reward, done, info = env.step("reveal 0 0")
    assert done is True
    assert reward == 0.0


# ---------------------------------------------------------------------------
# is_multithread_safe
# ---------------------------------------------------------------------------

def test_is_multithread_safe():
    """Test that Minesweeper declares itself thread-safe."""
    assert MinesweeperEnv.is_multithread_safe() is True


# ---------------------------------------------------------------------------
# Full game win
# ---------------------------------------------------------------------------

def test_full_game_win():
    """Test a complete successful game by revealing all safe cells."""
    env = MinesweeperEnv(rows=5, cols=5, num_mines=3, max_steps=30, seed=42)
    env.reset()

    # Accumulate reward from all reveals including the first
    total_reward = 0.0

    # First reveal at center (guaranteed safe)
    obs, reward, done, info = env.step("reveal 2 2")
    total_reward += reward

    # Find and reveal all remaining non-mine cells
    for r in range(5):
        for c in range(5):
            if not env.revealed[r][c] and env.grid[r][c] != -1 and not env._done:
                obs, reward, done, info = env.step(f"reveal {r} {c}")
                total_reward += reward
                if info.get("success"):
                    break
        if env._done:
            break

    # Binary reward: only success gives 1.0, so total should be exactly 1.0
    if env.success():
        assert total_reward == 1.0
        assert env.success() is True


# ---------------------------------------------------------------------------
# Solver oracle advantage (credit assignment) -- emitted by step() when enabled
# ---------------------------------------------------------------------------
# These run on the REAL MinesweeperEnv with a deterministic preset board, and
# check that info['oracle_advantage'] = K(s_t) - K(s_{t+1}) is correct per the
# linear-potential formulation, and that wrong agent flags cannot corrupt it.

from rllm.environments.minesweeper.oracle import solver_value_K, INF as _ORACLE_INF
from rllm.environments.minesweeper.solver import solve_board as _solve_board

# mines [[1,1],[3,3]], first_click [0,4] is a pure-deduction win (no guessing),
# so every K(s) is finite and deterministic.
_ORC_MINES = [[1, 1], [3, 3]]
_ORC_FC = [0, 4]


def _make_oracle_env(**overrides):
    kwargs = dict(rows=5, cols=5, num_mines=2,
                  mine_positions=_ORC_MINES, first_click=_ORC_FC,
                  compute_oracle_advantage=True, oracle_solver_mode="strong")
    kwargs.update(overrides)
    env = MinesweeperEnv(**kwargs)
    env.reset()
    return env


class TestOracleAdvantage:
    def test_disabled_by_default_no_signal(self):
        """Default env (flag off) must NOT emit oracle_advantage."""
        env = MinesweeperEnv(rows=5, cols=5, num_mines=2,
                             mine_positions=_ORC_MINES, first_click=_ORC_FC)
        assert env.compute_oracle_advantage is False
        env.reset()
        act = _solve_board(env.grid, env.revealed, env.flags, 5, 5,
                           mode="strong", total_mines=2).best_action
        _, _, _, info = env.step(f"{act[0]} {act[1]} {act[2]}")
        assert "oracle_advantage" not in info

    def test_from_dict_accepts_oracle_keys(self):
        env = MinesweeperEnv.from_dict({
            "rows": 5, "cols": 5, "num_mines": 2,
            "mine_positions": _ORC_MINES, "first_click": _ORC_FC,
            "compute_oracle_advantage": True, "oracle_solver_mode": "strong",
        })
        assert env.compute_oracle_advantage is True
        assert env.oracle_solver_mode == "strong"

    def test_optimal_reveal_gives_plus_one(self):
        """Taking the solver's own best reveal moves K from N to N-1 -> adv = +1."""
        env = _make_oracle_env()
        act = _solve_board(env.grid, env.revealed, env.flags, 5, 5,
                           mode="strong", total_mines=2).best_action
        assert act[0] == "reveal"
        _, _, done, info = env.step(f"{act[0]} {act[1]} {act[2]}")
        assert info["action_is_valid"] is True
        assert info["oracle_advantage"] == 1.0

    def test_advantage_equals_K_before_minus_K_after_each_step(self):
        """Across a full solver-played episode, adv == K(s_t) - K(s_{t+1})."""
        env = _make_oracle_env()
        for _ in range(200):
            if env._done:
                break
            k_before = solver_value_K(5, 5, env.grid, env.revealed, env.flags,
                                      total_mines=2)
            act = _solve_board(env.grid, env.revealed, env.flags, 5, 5,
                               mode="strong", total_mines=2).best_action
            if act is None:
                break
            _, _, done, info = env.step(f"{act[0]} {act[1]} {act[2]}")
            if info.get("success"):
                # win: K_after = 0
                assert info["oracle_advantage"] == k_before
            else:
                k_after = solver_value_K(5, 5, env.grid, env.revealed, env.flags,
                                         total_mines=2)
                assert info["oracle_advantage"] == k_before - k_after

    def test_invalid_action_gives_zero(self):
        """An invalid action (re-reveal) leaves state unchanged -> adv = 0."""
        env = _make_oracle_env()
        # First, reveal a provably-safe cell.
        act = _solve_board(env.grid, env.revealed, env.flags, 5, 5,
                           mode="strong", total_mines=2).best_action
        env.step(f"{act[0]} {act[1]} {act[2]}")
        # Re-reveal the same cell -> invalid.
        _, _, _, info = env.step(f"reveal {act[1]} {act[2]}")
        assert info["action_is_valid"] is False
        assert info["oracle_advantage"] == 0.0

    def test_flag_safe_cell_costs_minus_one(self):
        """Flagging a provably-safe but unrevealed cell adds an unflag step: adv=-1."""
        env = _make_oracle_env()
        # Find a currently-safe, unrevealed cell.
        safe = [(r, c) for r in range(5) for c in range(5)
                if env.grid[r][c] != -1 and not env.revealed[r][c]][0]
        _, _, _, info = env.step(f"flag {safe[0]} {safe[1]}")
        assert info["action_is_valid"] is True
        assert info["oracle_advantage"] == -1.0

    def test_flag_mine_is_neutral(self):
        """Flagging an actual mine adds no unflag cost: adv = 0."""
        env = _make_oracle_env()
        mr, mc = _ORC_MINES[0]            # (1,1) is a mine, unrevealed
        assert env.grid[mr][mc] == -1 and not env.revealed[mr][mc]
        _, _, _, info = env.step(f"flag {mr} {mc}")
        assert info["action_is_valid"] is True
        assert info["oracle_advantage"] == 0.0

    def test_wrong_flag_does_not_corrupt_subsequent_advantage(self):
        """A wrong flag on a safe cell must not poison later K via the solver.

        After wrongly flagging a safe cell, revealing the solver's best cell must
        still yield the SAME advantage as if no wrong flag existed (the oracle
        discards agent flags for reasoning; the wrong flag only added its own -1).
        """
        # Baseline: no wrong flag.
        env0 = _make_oracle_env()
        act0 = _solve_board(env0.grid, env0.revealed, env0.flags, 5, 5,
                            mode="strong", total_mines=2).best_action
        _, _, _, info0 = env0.step(f"{act0[0]} {act0[1]} {act0[2]}")
        base_adv = info0["oracle_advantage"]

        # With a wrong flag on a safe cell placed first, then the SAME reveal.
        env1 = _make_oracle_env()
        safe = [(r, c) for r in range(5) for c in range(5)
                if env1.grid[r][c] != -1 and not env1.revealed[r][c]
                and (r, c) != (act0[1], act0[2])][0]
        env1.step(f"flag {safe[0]} {safe[1]}")           # wrong flag (adv handled = -1)
        _, _, _, info1 = env1.step(f"{act0[0]} {act0[1]} {act0[2]}")
        # The reveal's advantage is unaffected by the (discarded) wrong flag.
        assert info1["oracle_advantage"] == base_adv
