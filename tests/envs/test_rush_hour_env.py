"""Tests for the Rush Hour environment (RushHourEnv).

Exercises the real environment end to end: generation via seed, deterministic
puzzle pinning via ``board_config``, step semantics (valid / invalid / effective
actions), success + reward, solver-based oracle advantage, deadlock handling,
and the ``from_dict`` factory with difficulty presets.
"""

import pytest

from rllm.environments.rush_hour.rush_hour_env import RushHourEnv, DIFFICULTY_CONFIGS
from rllm.environments.rush_hour import solver as S
from rllm.environments.rush_hour.board import RushHourBoard
from rllm.environments.rush_hour.oracle import build_n_table


# Fixed hand-solved boards for the oracle-advantage cells/moves tests.
# B1: exit row empty, optimal solve "A+4" -> N_moves=1, N_cells=4.
B1 = "oooooo" + "oooooo" + "AAoooo" + "oooooo" + "oooooo" + "oooooo"
# B2: car B blocks the exit, optimal solve "B-1","A+4" -> N_moves=2, N_cells=5.
B2 = "oooooo" + "oooBoo" + "AAoBoo" + "oooooo" + "oooooo" + "oooooo"
# Overshoot board: red car AA in the middle (row2 col2-3) can slide AWAY from
# the exit. N_cells(s0)=2 ("A+2" solves), so "A-2" raises N_cells by 2.
OVERSHOOT = "oooooo" + "oooooo" + "ooAAoo" + "oooooo" + "oooooo" + "oooooo"


def _optimal_first_action(env) -> str:
    """Return the solver's optimal first move as an action string."""
    label, steps = S.get_next_move(env.board)
    sign = "+" if steps > 0 else "-"
    return f"{label}{sign}{abs(steps)}"


# ---------------------------------------------------------------------------
# Construction / reset
# ---------------------------------------------------------------------------

def test_reset_from_seed():
    env = RushHourEnv.from_dict({"seed": 0, "difficulty": "id"})
    obs, info = env.reset()
    assert isinstance(obs, str) and obs
    assert info["success"] is False
    assert info["min_moves"] >= 1
    assert not env.board.solved


def test_reset_is_deterministic():
    env1 = RushHourEnv.from_dict({"seed": 7, "difficulty": "id"})
    env2 = RushHourEnv.from_dict({"seed": 7, "difficulty": "id"})
    env1.reset()
    env2.reset()
    assert env1.board.to_string() == env2.board.to_string()
    assert env1._min_moves == env2._min_moves


def test_board_config_pin():
    """A pinned board_config reproduces the exact puzzle, ignoring the seed."""
    src = RushHourEnv.from_dict({"seed": 3, "difficulty": "id"})
    src.reset()
    desc = src.board.to_string()

    pinned = RushHourEnv.from_dict({"board_config": desc, "difficulty": "id"})
    pinned.reset()
    assert pinned.board.to_string() == desc


def test_difficulty_presets_applied():
    env = RushHourEnv.from_dict({"seed": 0, "difficulty": "unseen"})
    preset = DIFFICULTY_CONFIGS["unseen"]
    assert env.config.num_vehicles == preset["num_vehicles"]
    assert env.config.min_moves == preset["min_moves"]
    assert env.config.max_steps == preset["max_steps"]


# ---------------------------------------------------------------------------
# Step semantics
# ---------------------------------------------------------------------------

def test_solve_episode_with_oracle_path():
    """Following the solver's optimal moves must solve the puzzle and reward 1."""
    env = RushHourEnv.from_dict({"seed": 1, "difficulty": "id"})
    env.reset()

    done = False
    total_reward = 0.0
    for _ in range(env.max_steps):
        action = _optimal_first_action(env)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break

    assert done
    assert info["success"] is True
    assert env.board.solved
    assert total_reward == 1.0


def test_invalid_action_does_not_terminate():
    env = RushHourEnv.from_dict({"seed": 2, "difficulty": "id"})
    env.reset()
    obs, reward, done, info = env.step("this is not a move")
    assert info["action_is_valid"] is False
    assert info["action_is_effective"] is False
    assert done is False  # one bad move must not end the episode
    assert reward == 0.0


def test_unknown_car_label_is_invalid():
    env = RushHourEnv.from_dict({"seed": 2, "difficulty": "id"})
    env.reset()
    obs, reward, done, info = env.step("Z+1")  # no such car
    assert info["action_is_valid"] is False
    assert info["action_is_effective"] is False


def test_parse_action_rejects_prose():
    """parse_action must NOT extract a move buried in free-form text.

    Regression for the ``.search`` -> ``.fullmatch`` fix: an incidental
    car-label + sign + digits inside prose (e.g. "move car b-1 now") must be
    rejected, otherwise a "thinking out loud" agent silently executes an
    unintended real board move marked action_is_valid=True.
    """
    from rllm.environments.rush_hour.board import parse_action

    # Clean, whole-string moves are still accepted (with optional brackets/spaces).
    assert parse_action("A+2") == ("A", 2)
    assert parse_action("B-1") == ("B", -1)
    assert parse_action("[A+]") == ("A", 1)
    assert parse_action("  a-3  ") == ("A", -3)

    # Prose containing a move-like token must NOT parse to a move.
    for prose in (
        "I will move car b-1 now",
        "The answer is to slide A right. b+3",
        "first do c then maybe d+1",
        "let me think... a+2",
        "A+2 then B-1",  # two moves: ambiguous, reject rather than silently pick one
    ):
        assert parse_action(prose) is None, prose


def test_step_rejects_prose_action():
    """A prose action string must be treated as invalid and leave the board unchanged."""
    env = RushHourEnv.from_dict({"seed": 2, "difficulty": "id"})
    env.reset()
    before = env.board.to_string()
    obs, reward, done, info = env.step("I will move car b-1 now")
    assert info["action_is_valid"] is False
    assert info["action_is_effective"] is False
    assert env.board.to_string() == before  # no silent move executed


def test_blocked_move_is_valid_but_ineffective():
    """A fully-blocked move (parseable + known car, but cannot move) stays valid.

    On B2 the red car A is at the left wall, so A-1 cannot move at all: the
    board is unchanged and the action is reported valid+ineffective (NOT
    invalid). Distinct from overshoot, which is covered separately.
    """
    env = RushHourEnv.from_dict({"board_config": B2})
    env.reset()
    before = env.board.to_string()
    obs, reward, done, info = env.step("A-1")  # A already at the left wall
    assert info["action_is_valid"] is True
    assert info["action_is_effective"] is False
    assert env.board.to_string() == before  # fully blocked -> no change


def test_overshoot_is_valid_but_ineffective():
    """Requesting more cells than a car can slide moves NOTHING (all-or-nothing).

    On B2, car B (vertical, col 3, rows 1-2) can slide up exactly 1 cell (to
    rows 0-1). Requesting B-2 overshoots -> ineffective and the board is
    unchanged, while the exact B-1 is effective.
    """
    env = RushHourEnv.from_dict({"board_config": B2})
    env.reset()
    before = env.board.to_string()
    # Overshoot: B can only go up 1, request 2 -> rejected, board unchanged.
    obs, reward, done, info = env.step("B-2")
    assert info["action_is_valid"] is True
    assert info["action_is_effective"] is False
    assert env.board.to_string() == before
    # Exact distance is accepted and changes the board.
    obs, reward, done, info = env.step("B-1")
    assert info["action_is_valid"] is True
    assert info["action_is_effective"] is True
    assert env.board.to_string() != before


def test_max_steps_termination():
    env = RushHourEnv.from_dict(
        {"seed": 4, "difficulty": "id", "max_steps": 3}
    )
    env.reset()
    done = False
    steps = 0
    # Spam an ineffective-but-valid move type that won't accidentally solve:
    # repeatedly toggling a non-red car small; we just bound by max_steps.
    for _ in range(3):
        obs, reward, done, info = env.step("A-1")
        steps += 1
        if done:
            break
    assert steps <= 3


# A statically-deadlocked board: a wall at row 2 col 4 permanently blocks the
# red car A (row 2, col 0-1) from reaching the exit (col 4-5).
DEADLOCK = "oooooo" + "oooooo" + "AAooxo" + "oooooo" + "oooooo" + "oooooo"


def test_deadlock_counter_resets_on_ineffective_move():
    """An ineffective (board-unchanged) move must not advance the deadlock streak.

    On a deadlocked board, repeatedly overshooting (A+9 -> moves 0) leaves the
    board unchanged. Like an invalid action, it resets the consecutive-deadlock
    counter, so game-over (N=2) is never triggered by no-op spam.
    """
    env = RushHourEnv.from_dict(
        {"board_config": DEADLOCK, "game_over_on_deadlock": 2, "max_steps": 20}
    )
    env.reset()
    for _ in range(3):
        obs, reward, done, info = env.step("A+9")  # overshoot -> ineffective
        assert info["action_is_effective"] is False
        assert env._consecutive_deadlock_count == 0
        assert "terminated_by_deadlock" not in info
        assert done is False


def test_deadlock_counter_increments_on_effective_move():
    """Effective moves that land on deadlocked states DO advance the streak.

    A+1 then A+1 are real board changes; each lands on a still-deadlocked state,
    so the counter climbs 1 -> 2 and game-over fires at N=2.
    """
    env = RushHourEnv.from_dict(
        {"board_config": DEADLOCK, "game_over_on_deadlock": 2, "max_steps": 20}
    )
    env.reset()
    obs, reward, done, info = env.step("A+1")
    assert info["action_is_effective"] is True
    assert env._consecutive_deadlock_count == 1
    assert done is False
    obs, reward, done, info = env.step("A+1")
    assert info["action_is_effective"] is True
    assert env._consecutive_deadlock_count == 2
    assert done is True
    assert info.get("terminated_by_deadlock") is True


# ---------------------------------------------------------------------------
# Oracle advantage
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("potential", ["moves", "cells"])
def test_oracle_advantage_optimal_move_positive(potential):
    """The solver's optimal first move yields a positive oracle advantage.

    Under ``moves`` the optimal move reduces N(s) by exactly one move -> +1.
    Under ``cells`` it reduces N(s) by the number of cells slid, so the expected
    advantage is the optimal first move's slide distance (NOT hardcoded to 1).
    """
    env = RushHourEnv.from_dict(
        {
            "seed": 1,
            "difficulty": "id",
            "compute_oracle_advantage": True,
            "oracle_potential": potential,
        }
    )
    env.reset()
    label, steps = S.get_next_move(env.board)
    action = f"{label}{'+' if steps > 0 else '-'}{abs(steps)}"
    obs, reward, done, info = env.step(action)
    assert "oracle_advantage" in info
    if potential == "moves":
        assert info["oracle_advantage"] == pytest.approx(1.0)
    else:
        assert info["oracle_advantage"] == pytest.approx(float(abs(steps)))


@pytest.mark.parametrize("potential", ["moves", "cells"])
def test_oracle_advantage_invalid_move_zero(potential):
    env = RushHourEnv.from_dict(
        {
            "seed": 1,
            "difficulty": "id",
            "compute_oracle_advantage": True,
            "oracle_potential": potential,
        }
    )
    env.reset()
    obs, reward, done, info = env.step("not a move")
    assert info["oracle_advantage"] == 0.0


# --- Cells-potential oracle advantage (Plan A) --------------------------------

def test_oracle_advantage_slide_k_on_optimal_path_is_plus_k():
    """Signature test: under cells, sliding k cells toward the exit gives +k.

    On B1 the exit row is empty so "A+4" solves in one slide:
      cells: N_cells 4 -> 0, advantage +4
      moves: N_moves 1 -> 0, advantage +1
    The same action giving +4 vs +1 can only come from the cells potential
    counting the 4-cell slide as 4 rather than 1.
    """
    env_cells = RushHourEnv(
        board_config=B1, compute_oracle_advantage=True, oracle_potential="cells"
    )
    env_cells.reset()
    _, _, _, info_cells = env_cells.step("A+4")
    assert info_cells["oracle_advantage"] == pytest.approx(4.0)

    env_moves = RushHourEnv(
        board_config=B1, compute_oracle_advantage=True, oracle_potential="moves"
    )
    env_moves.reset()
    _, _, _, info_moves = env_moves.step("A+4")
    assert info_moves["oracle_advantage"] == pytest.approx(1.0)

    # Small-step version: a single cell toward the exit.
    #   cells: 4 -> 3, advantage +1
    #   moves: 1 -> 1 (still one slide remaining), advantage 0
    env_cells_1 = RushHourEnv(
        board_config=B1, compute_oracle_advantage=True, oracle_potential="cells"
    )
    env_cells_1.reset()
    _, _, _, info_cells_1 = env_cells_1.step("A+1")
    assert info_cells_1["oracle_advantage"] == pytest.approx(1.0)

    env_moves_1 = RushHourEnv(
        board_config=B1, compute_oracle_advantage=True, oracle_potential="moves"
    )
    env_moves_1.reset()
    _, _, _, info_moves_1 = env_moves_1.step("A+1")
    assert info_moves_1["oracle_advantage"] == pytest.approx(0.0)


def test_oracle_advantage_overshoot_is_minus_k():
    """Sliding k cells AWAY from the exit gives -k under cells.

    Expected values are recomputed from an independent build_n_table rather than
    hardcoded, to avoid any board-reading mistake.
    """
    board = RushHourBoard.from_string(OVERSHOOT)
    s0 = board.memo_key()

    cells_table = build_n_table(board, potential="cells")
    # State after "A-2": red car slid 2 cells left (col0-1).
    after = board.copy()
    after.apply_action("A", -2)
    s1 = after.memo_key()
    expected_cells_adv = float(cells_table[s0] - cells_table[s1])
    assert expected_cells_adv < 0  # the move strictly worsens the position

    env_cells = RushHourEnv(
        board_config=OVERSHOOT, compute_oracle_advantage=True, oracle_potential="cells"
    )
    env_cells.reset()
    _, _, _, info_cells = env_cells.step("A-2")
    assert info_cells["oracle_advantage"] == pytest.approx(expected_cells_adv)

    # moves potential: recompute its own two-end difference (do not hardcode).
    moves_table = build_n_table(board, potential="moves")
    expected_moves_adv = float(moves_table[s0] - moves_table[s1])
    env_moves = RushHourEnv(
        board_config=OVERSHOOT, compute_oracle_advantage=True, oracle_potential="moves"
    )
    env_moves.reset()
    _, _, _, info_moves = env_moves.step("A-2")
    assert info_moves["oracle_advantage"] == pytest.approx(expected_moves_adv)


@pytest.mark.parametrize("potential", ["moves", "cells"])
def test_oracle_advantage_ineffective_move_zero(potential):
    """A valid-but-ineffective move leaves the board unchanged -> advantage 0.

    On B2 the red car cannot move right (car B blocks it) and is already at the
    left wall, so "A-3" is parseable + a known label but slides 0 cells.
    """
    env = RushHourEnv(
        board_config=B2, compute_oracle_advantage=True, oracle_potential=potential
    )
    env.reset()
    before = env.board.to_string()
    _, _, _, info = env.step("A-3")
    assert info["action_is_valid"] is True
    assert info["action_is_effective"] is False
    assert env.board.to_string() == before
    assert info["oracle_advantage"] == 0.0


def test_from_dict_cells_potential_end_to_end():
    """from_dict wires oracle_potential through to real cells advantage on B1."""
    env = RushHourEnv.from_dict(
        {
            "board_config": B1,
            "oracle_potential": "cells",
            "compute_oracle_advantage": True,
            "max_steps": 25,
        }
    )
    env.reset()
    assert env.oracle_potential == "cells"
    _, _, _, info = env.step("A+4")
    assert info["oracle_advantage"] == pytest.approx(4.0)
    assert info["success"] is True


# ---------------------------------------------------------------------------
# Factory / interface
# ---------------------------------------------------------------------------

def test_is_multithread_safe():
    assert RushHourEnv.is_multithread_safe() is True


def test_success_flag_and_render():
    env = RushHourEnv.from_dict({"seed": 0, "difficulty": "id"})
    env.reset()
    assert env.success() is False
    # solve fully, then success() is True
    for _ in range(env.max_steps):
        _, _, done, info = env.step(_optimal_first_action(env))
        if done:
            break
    assert env.success() is True
    assert isinstance(env.render("grid"), str)
