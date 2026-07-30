"""Tests for the Rush Hour board engine, IDA* solver oracle, and generator.

These exercise REAL behaviour against ground truth, not parameter passing:
  * The solver's optimal move count is validated against the labelled
    ``rush_18k.txt`` database (known optimal move counts).
  * Every solver solution is replayed on the board and must actually solve it.
  * The generator is checked for determinism, solvability, and difficulty range.
  * Static deadlock detection is validated on a hand-built unsolvable board.
"""

import os
from pathlib import Path

import pytest

from rllm.environments.rush_hour.board import (
    HORIZONTAL,
    VERTICAL,
    RushHourBoard,
    parse_action,
)
from rllm.environments.rush_hour.generator import generate_board
from rllm.environments.rush_hour import solver as S


# Optional external dataset: the labelled puzzle database `rush_18k.txt` from the
# upstream reasoning-gym project. Tests using it are skipped when it is absent.
_RUSH_DB = (
    Path(__file__).resolve().parents[2]
    / "Rush-Hour-Repo"
    / "reasoning-gym"
    / "reasoning_gym"
    / "data"
    / "rush_18k.txt"
)


# ---------------------------------------------------------------------------
# Board engine
# ---------------------------------------------------------------------------

def _simple_board() -> RushHourBoard:
    # Red car AA at row 2, vertical car B at column 2 (rows 0-1). Solvable in 1.
    desc = (
        "ooBooo"
        "ooBooo"
        "AAoooo"
        "oooooo"
        "oooooo"
        "oooooo"
    )
    return RushHourBoard.from_string(desc)


def test_from_string_roundtrip():
    board = _simple_board()
    assert board.to_string() == "ooBoooooBoooAAoooooooooooooooooooooo"
    # Primary piece is index 0 and horizontal on the exit row.
    assert board.pieces[0].orientation == HORIZONTAL
    assert board.pieces[0].row(board.width) == 2
    assert not board.solved


def test_target_and_solved():
    board = _simple_board()
    # target = exit cell index for the red car's left cell.
    assert board.target == 2 * 6 + (6 - 2)  # row 2, col 4
    board.apply_action("A", 4)  # slide red car fully right
    assert board.solved


def test_apply_action_blocked_returns_zero():
    board = _simple_board()
    # B is vertical at col 2; cannot move up (at top), reports 0 cells moved.
    moved = board.apply_action("B", -1)
    assert moved == 0
    # A can slide exactly 4 cells right to the exit; that exact request works.
    moved = board.apply_action("A", 4)
    assert moved == 4
    assert board.solved


def test_apply_action_overshoot_returns_zero():
    """All-or-nothing: an over-request that cannot fully complete moves nothing."""
    board = _simple_board()
    before = board.to_string()
    # A can only travel 4 cells to the exit; requesting 10 overshoots -> rejected.
    moved = board.apply_action("A", 10)
    assert moved == 0
    assert board.to_string() == before  # board unchanged, no partial slide
    # The exact distance still works afterwards.
    assert board.apply_action("A", 4) == 4
    assert board.solved


def test_apply_action_unknown_label():
    board = _simple_board()
    assert board.apply_action("Z", 1) == 0


def test_legal_moves_and_undo():
    board = _simple_board()
    before = board.memo_key()
    moves = board.legal_moves()
    assert moves, "expected at least one legal move"
    pi, steps = moves[0]
    board.do_move(pi, steps)
    assert board.memo_key() != before
    board.undo_move(pi, steps)
    assert board.memo_key() == before


def test_parse_action_formats():
    assert parse_action("A+2") == ("A", 2)
    assert parse_action("B-1") == ("B", -1)
    assert parse_action("[A+]") == ("A", 1)
    assert parse_action("[C-3]") == ("C", -3)
    assert parse_action("garbage") is None
    assert parse_action("A+0") is None


def test_render_modes():
    board = _simple_board()
    grid = board.render("grid")
    coord = board.render("coord")
    both = board.render("grid_coord")
    assert "A A" in grid
    assert "red car" in coord
    assert "Grid:" in both and "Coordinates:" in both


# ---------------------------------------------------------------------------
# Solver: optimality vs. labelled database
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _RUSH_DB.exists(), reason="rush_18k.txt database not present")
@pytest.mark.parametrize("line_index", [0, 600, 3000, 9000])
def test_solver_optimal_matches_database(line_index):
    """get_min_moves must equal the database's labelled optimal move count."""
    lines = []
    with _RUSH_DB.open() as fh:
        for ln in fh:
            parts = ln.split()
            if len(parts) >= 2:
                lines.append((int(parts[0]), parts[1]))
    expected, desc = lines[line_index]
    board = RushHourBoard.from_string(desc)
    got = S.get_min_moves(board, max_depth=expected + 5)
    assert got == expected


@pytest.mark.skipif(not _RUSH_DB.exists(), reason="rush_18k.txt database not present")
def test_solver_solution_actually_solves():
    """Replaying the solver's move list must reach a solved board."""
    with _RUSH_DB.open() as fh:
        for ln in fh:
            parts = ln.split()
            if len(parts) >= 2:
                desc = parts[1]
                break
    board = RushHourBoard.from_string(desc)
    solution = S.solve(board, max_depth=20)
    assert solution is not None
    for pi, steps in solution:
        board.do_move(pi, steps)
    assert board.solved


def test_solver_already_solved():
    board = _simple_board()
    board.apply_action("A", 4)
    assert board.solved
    assert S.get_min_moves(board) == 0
    assert S.solve(board) == []


# ---------------------------------------------------------------------------
# Static deadlock detection
# ---------------------------------------------------------------------------

def test_deadlock_detection_unsolvable():
    """A vertical truck spanning the whole exit-lane column blocks the red car.

    Column 4 is fully occupied by a size-3 truck plus another, so the cell the
    red car must pass through is forced-occupied in every configuration.
    """
    # Red car AA at row2 col0-1. Vertical truck C size 3 at col4 rows0-2, and
    # vertical car D size 3 at col4 rows3-5 => column 4 fully blocked forever.
    desc = (
        "ooooCo"
        "ooooCo"
        "AAooCo"
        "ooooDo"
        "ooooDo"
        "ooooDo"
    )
    board = RushHourBoard.from_string(desc)
    assert S.is_deadlocked(board) is True
    assert S.get_min_moves(board) == S.INF


def test_not_deadlocked_solvable():
    board = _simple_board()
    assert S.is_deadlocked(board) is False


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def test_generator_determinism():
    b1, n1 = generate_board(123, num_vehicles=7, min_moves=3, max_moves=12)
    b2, n2 = generate_board(123, num_vehicles=7, min_moves=3, max_moves=12)
    assert b1.to_string() == b2.to_string()
    assert n1 == n2


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_generator_solvable_in_range(seed):
    board, n = generate_board(seed, num_vehicles=8, min_moves=4, max_moves=15)
    assert 4 <= n <= 15
    assert not board.solved
    assert not S.is_deadlocked(board)
    # Solver agrees with the reported difficulty and the solution solves it.
    solution = S.solve(board, max_depth=20)
    assert solution is not None and len(solution) == n
    for pi, steps in solution:
        board.do_move(pi, steps)
    assert board.solved


def test_generator_difficulty_separation():
    """Harder configs should yield higher optimal move counts than easy ones.

    The hard window (9-14) requires hill-climbing and exact verification, which
    is slow, so we sample a couple of seeds rather than many.
    """
    easy = [generate_board(s, num_vehicles=6, min_moves=1, max_moves=6)[1] for s in range(5)]
    hard = [generate_board(s, num_vehicles=8, min_moves=9, max_moves=14)[1] for s in range(2)]
    assert max(easy) <= 6
    assert min(hard) >= 9
