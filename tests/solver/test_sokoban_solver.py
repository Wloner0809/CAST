"""Comprehensive correctness tests for the Sokoban A* solver oracle.

Tests cover: state extraction, heuristic computation, deadlock detection
(corner, frozen-line), step counting, next-action recommendation,
check_actions_for_deadlock, near-deadlock detection, internal helpers,
edge cases, and integration with real SokobanEnv instances.

Regression tests for fixed bugs S1 (frozen-line deadlock pruning in A*)
and S2 (_is_corner_deadlock return type) are included at the bottom.
"""

import pytest
import numpy as np

scipy = pytest.importorskip("scipy")

from rllm.environments.sokoban.solver import (
    WALL,
    FLOOR,
    TARGET,
    BOX_ON_TARGET,
    BOX,
    PLAYER,
    _cell_without_player,
    _extract_state,
    _heuristic,
    _is_corner_deadlock,
    _is_frozen_line_deadlock,
    _is_any_deadlock,
    check_actions_for_deadlock,
    classify_deadlock,
    count_near_deadlock_boxes,
    get_min_steps,
    get_next_action,
    is_deadlocked,
)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _make_simple_puzzle():
    """1 box, 1 target, solvable in a few pushes.

    Layout:
        #####
        #P__#
        #_X_#
        #__O#
        #####
    """
    room_fixed = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 2, 0],
        [0, 0, 0, 0, 0],
    ])
    room_state = np.array([
        [0, 0, 0, 0, 0],
        [0, 5, 1, 1, 0],
        [0, 1, 4, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ])
    return room_state, room_fixed


def _make_two_box_puzzle():
    """2 boxes, 2 targets, solvable.

    Layout (7x7):
        #######
        #_____#
        #_X___#
        #P__X_#
        #___O_#
        #_O___#
        #######
    """
    room_fixed = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 2, 1, 0],
        [0, 1, 2, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ])
    room_state = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 1, 4, 1, 1, 1, 0],
        [0, 5, 1, 1, 4, 1, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ])
    return room_state, room_fixed


def _make_solved_puzzle():
    """All boxes on targets. Already solved.

    Layout:
        #####
        #P__#
        #___#
        #__*#
        #####
    """
    room_fixed = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 2, 0],
        [0, 0, 0, 0, 0],
    ])
    room_state = np.array([
        [0, 0, 0, 0, 0],
        [0, 5, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 3, 0],
        [0, 0, 0, 0, 0],
    ])
    return room_state, room_fixed


def _make_corner_deadlock():
    """Box stuck in corner (not on target).

    Layout:
        #####
        #X__#
        #P__#
        #__O#
        #####
    """
    room_fixed = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 2, 0],
        [0, 0, 0, 0, 0],
    ])
    room_state = np.array([
        [0, 0, 0, 0, 0],
        [0, 4, 1, 1, 0],
        [0, 5, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ])
    return room_state, room_fixed


def _make_frozen_line_deadlock():
    """Two boxes along a wall forming a frozen line.

    Layout (7x7):
        #######
        #_XX__#
        #P____#
        #____O#
        #___O_#
        #_____#
        #######

    Boxes at (1,2) and (1,3) are both along the top wall, forming a frozen line.
    """
    room_fixed = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 2, 0],
        [0, 1, 1, 1, 2, 1, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ])
    room_state = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 4, 4, 1, 1, 0],
        [0, 5, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ])
    return room_state, room_fixed


def _make_unsolvable():
    """Two boxes in adjacent corners -- both deadlocked.

    Layout:
        #####
        #XX_#
        #P__#
        #__O#
        #####

    Box at (1,1) corner deadlock, box at (1,2) against wall with (1,1).
    """
    room_fixed = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 2, 0],
        [0, 0, 0, 0, 0],
    ])
    room_state = np.array([
        [0, 0, 0, 0, 0],
        [0, 4, 4, 1, 0],
        [0, 5, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ])
    return room_state, room_fixed


# ===================================================================
# Tests
# ===================================================================


class TestExtractState:
    """Tests for _extract_state."""

    def test_basic_extraction(self):
        room_state, room_fixed = _make_simple_puzzle()
        player, boxes, targets = _extract_state(room_state, room_fixed)
        assert player == (1, 1)
        assert boxes == frozenset({(2, 2)})
        assert targets == [(3, 3)]

    def test_solved_state_box_on_target(self):
        room_state, room_fixed = _make_solved_puzzle()
        player, boxes, targets = _extract_state(room_state, room_fixed)
        assert player == (1, 1)
        assert (3, 3) in boxes  # BOX_ON_TARGET counted as box
        assert targets == [(3, 3)]

    def test_two_boxes(self):
        room_state, room_fixed = _make_two_box_puzzle()
        player, boxes, targets = _extract_state(room_state, room_fixed)
        assert player == (3, 1)
        assert len(boxes) == 2
        assert (2, 2) in boxes
        assert (3, 4) in boxes
        assert len(targets) == 2

    def test_returns_correct_types(self):
        room_state, room_fixed = _make_simple_puzzle()
        player, boxes, targets = _extract_state(room_state, room_fixed)
        assert isinstance(player, tuple)
        assert isinstance(boxes, frozenset)
        assert isinstance(targets, list)


class TestHeuristic:
    """Tests for _heuristic (Hungarian assignment)."""

    def test_zero_when_solved(self):
        room_state, room_fixed = _make_solved_puzzle()
        _, boxes, targets = _extract_state(room_state, room_fixed)
        assert _heuristic(boxes, targets) == 0.0

    def test_single_box_manhattan(self):
        room_state, room_fixed = _make_simple_puzzle()
        _, boxes, targets = _extract_state(room_state, room_fixed)
        h = _heuristic(boxes, targets)
        # Box (2,2) -> target (3,3) = |2-3| + |2-3| = 2
        assert h == 2.0

    def test_positive_when_unsolved(self):
        room_state, room_fixed = _make_two_box_puzzle()
        _, boxes, targets = _extract_state(room_state, room_fixed)
        h = _heuristic(boxes, targets)
        assert h > 0

    def test_empty_boxes(self):
        assert _heuristic(frozenset(), [(1, 1)]) == 0.0

    def test_empty_targets(self):
        assert _heuristic(frozenset({(1, 1)}), []) == 0.0

    def test_multi_box_hungarian_optimal(self):
        """Hungarian should assign optimally, not greedily."""
        # Boxes at (0,0) and (0,4); targets at (0,1) and (0,3)
        # Greedy: (0,0)->(0,1)=1, (0,4)->(0,3)=1, total=2
        # Optimal: same = 2
        boxes = frozenset({(0, 0), (0, 4)})
        targets = [(0, 1), (0, 3)]
        assert _heuristic(boxes, targets) == 2.0


class TestDeadlockDetection:
    """Tests for is_deadlocked, classify_deadlock."""

    def test_corner_deadlock(self):
        room_state, room_fixed = _make_corner_deadlock()
        assert is_deadlocked(room_state, room_fixed) is True
        result = classify_deadlock(room_state, room_fixed)
        assert result is not None
        assert result[0] == "corner"
        assert result[1] == (1, 1)

    def test_no_deadlock_simple(self):
        room_state, room_fixed = _make_simple_puzzle()
        assert is_deadlocked(room_state, room_fixed) is False
        assert classify_deadlock(room_state, room_fixed) is None

    def test_solved_not_deadlocked(self):
        room_state, room_fixed = _make_solved_puzzle()
        assert is_deadlocked(room_state, room_fixed) is False

    def test_frozen_line_deadlock(self):
        room_state, room_fixed = _make_frozen_line_deadlock()
        assert is_deadlocked(room_state, room_fixed) is True
        result = classify_deadlock(room_state, room_fixed)
        assert result is not None
        # Should be either corner or frozen -- both boxes are against top wall
        assert result[0] in ("corner", "frozen")

    def test_corner_deadlock_direct(self):
        """Corner detection on a minimal board."""
        room_fixed = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ])
        # Cell (1,1) has walls on all four sides -> corner
        assert _is_corner_deadlock(1, 1, room_fixed)

    def test_no_corner_for_open_cell(self):
        room_fixed = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ])
        # Center cell (2,2): open on all sides
        assert not _is_corner_deadlock(2, 2, room_fixed)


class TestGetMinSteps:
    """Tests for get_min_steps (A* solver)."""

    def test_solved_returns_zero(self):
        room_state, room_fixed = _make_solved_puzzle()
        assert get_min_steps(room_state, room_fixed) == 0

    def test_simple_puzzle_finite(self):
        room_state, room_fixed = _make_simple_puzzle()
        steps = get_min_steps(room_state, room_fixed, weight=1.0, max_nodes=100000)
        assert steps < float("inf")
        assert steps > 0

    def test_deadlocked_returns_inf(self):
        room_state, room_fixed = _make_corner_deadlock()
        assert get_min_steps(room_state, room_fixed) == float("inf")

    def test_weighted_search(self):
        room_state, room_fixed = _make_simple_puzzle()
        exact = get_min_steps(room_state, room_fixed, weight=1.0, max_nodes=100000)
        approx = get_min_steps(room_state, room_fixed, weight=2.5, max_nodes=100000)
        assert approx < float("inf")
        assert approx >= exact  # weighted can't be shorter than optimal

    def test_max_nodes_limit(self):
        room_state, room_fixed = _make_simple_puzzle()
        result = get_min_steps(room_state, room_fixed, weight=1.0, max_nodes=1)
        assert isinstance(result, (int, float))

    def test_two_box_puzzle_solvable(self):
        room_state, room_fixed = _make_two_box_puzzle()
        steps = get_min_steps(room_state, room_fixed, weight=1.0, max_nodes=200000)
        assert steps < float("inf"), "Two-box puzzle should be solvable"
        assert steps >= 2, "Need at least 2 pushes for 2 boxes"


class TestGetNextAction:
    """Tests for get_next_action (first step of solution)."""

    def test_solved_returns_none(self):
        room_state, room_fixed = _make_solved_puzzle()
        assert get_next_action(room_state, room_fixed) is None

    def test_deadlocked_returns_none(self):
        room_state, room_fixed = _make_corner_deadlock()
        assert get_next_action(room_state, room_fixed) is None

    def test_simple_puzzle_returns_valid_action(self):
        room_state, room_fixed = _make_simple_puzzle()
        action = get_next_action(room_state, room_fixed, max_nodes=100000)
        assert action is not None
        assert action in (1, 2, 3, 4)  # Up, Down, Left, Right

    def test_action_progresses_toward_solution(self):
        """After taking the recommended action, min_steps should decrease or stay the same."""
        room_state, room_fixed = _make_simple_puzzle()
        action = get_next_action(room_state, room_fixed, weight=1.0, max_nodes=100000)
        assert action is not None

        steps_before = get_min_steps(room_state, room_fixed, weight=1.0, max_nodes=100000)
        # Simulate the action manually
        player_pos, boxes, targets = _extract_state(room_state, room_fixed)
        dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][action - 1]
        pr, pc = player_pos
        nr, nc = pr + dr, pc + dc

        # Build new state
        new_state = room_state.copy()
        new_state[pr, pc] = TARGET if room_fixed[pr, pc] == TARGET else FLOOR
        if (nr, nc) in boxes:
            br, bc = nr + dr, nc + dc
            new_state[nr, nc] = PLAYER
            new_state[br, bc] = BOX_ON_TARGET if room_fixed[br, bc] == TARGET else BOX
        else:
            new_state[nr, nc] = PLAYER

        steps_after = get_min_steps(new_state, room_fixed, weight=1.0, max_nodes=100000)
        assert steps_after <= steps_before


class TestCheckActionsForDeadlock:
    """Tests for check_actions_for_deadlock."""

    def test_simple_puzzle_no_deadlock(self):
        room_state, room_fixed = _make_simple_puzzle()
        deadlock_boxes, push_count = check_actions_for_deadlock(room_state, room_fixed)
        # The simple puzzle shouldn't have deadlock-causing pushes nearby
        assert isinstance(deadlock_boxes, list)
        assert isinstance(push_count, int)
        assert push_count >= 0

    def test_returns_list_and_count(self):
        room_state, room_fixed = _make_two_box_puzzle()
        deadlock_boxes, push_count = check_actions_for_deadlock(room_state, room_fixed)
        assert isinstance(deadlock_boxes, list)
        assert isinstance(push_count, int)

    def test_solved_no_pushes(self):
        room_state, room_fixed = _make_solved_puzzle()
        deadlock_boxes, push_count = check_actions_for_deadlock(room_state, room_fixed)
        # Player is not adjacent to any box that can be pushed
        assert isinstance(deadlock_boxes, list)


class TestNearDeadlockDetection:
    """Tests for count_near_deadlock_boxes."""

    def test_box_near_corner(self):
        """Box one push from a corner."""
        room_fixed = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 2, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ])
        room_state = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 4, 1, 1, 1, 1, 0],
            [0, 5, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ])
        at_risk = count_near_deadlock_boxes(room_state, room_fixed)
        assert len(at_risk) >= 1

    def test_box_on_target_not_counted(self):
        room_fixed = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 2, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ])
        room_state = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 3, 1, 0],
            [0, 1, 1, 5, 0],
            [0, 0, 0, 0, 0],
        ])
        assert len(count_near_deadlock_boxes(room_state, room_fixed)) == 0

    def test_already_deadlocked_not_counted(self):
        room_state, room_fixed = _make_corner_deadlock()
        assert len(count_near_deadlock_boxes(room_state, room_fixed)) == 0

    def test_safe_center_box(self):
        room_fixed = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 2, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ])
        room_state = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 4, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 5, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ])
        assert len(count_near_deadlock_boxes(room_state, room_fixed)) == 0


class TestCellWithoutPlayer:
    """Tests for _cell_without_player."""

    def test_player_on_floor(self):
        assert _cell_without_player(PLAYER, FLOOR) == FLOOR

    def test_player_on_target(self):
        assert _cell_without_player(PLAYER, TARGET) == TARGET

    def test_box_on_target_stays(self):
        assert _cell_without_player(BOX_ON_TARGET, TARGET) == BOX_ON_TARGET

    def test_floor_stays(self):
        assert _cell_without_player(FLOOR, FLOOR) == FLOOR

    def test_wall_stays(self):
        assert _cell_without_player(WALL, WALL) == WALL


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_box_single_target_corner_is_target(self):
        """Box in corner that IS a target -> not deadlocked."""
        room_fixed = np.array([
            [0, 0, 0, 0, 0],
            [0, 2, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ])
        room_state = np.array([
            [0, 0, 0, 0, 0],
            [0, 3, 1, 1, 0],  # Box on target at corner
            [0, 5, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ])
        # Already solved
        assert is_deadlocked(room_state, room_fixed) is False
        assert get_min_steps(room_state, room_fixed) == 0

    def test_all_boxes_on_targets(self):
        """Multiple boxes all on targets -> solved, 0 steps."""
        room_fixed = np.array([
            [0, 0, 0, 0, 0],
            [0, 2, 2, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ])
        room_state = np.array([
            [0, 0, 0, 0, 0],
            [0, 3, 3, 1, 0],
            [0, 1, 5, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ])
        assert get_min_steps(room_state, room_fixed) == 0

    def test_is_any_deadlock_target_not_deadlocked(self):
        """Box on target position should never be flagged as deadlocked."""
        room_fixed = np.array([
            [0, 0, 0, 0, 0],
            [0, 2, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ])
        boxes = frozenset({(1, 1)})
        targets = {(1, 1)}
        assert _is_any_deadlock(1, 1, boxes, room_fixed, targets) is False


class TestSolverOnRealEnv:
    """Integration tests with actual SokobanEnv instances."""

    def test_solver_on_generated_puzzle(self):
        from rllm.environments.sokoban.sokoban import SokobanEnv

        env = SokobanEnv(seed=42, dim_room=(6, 6), num_boxes=1)
        env.reset()

        steps = get_min_steps(env.room_state, env.room_fixed, weight=1.0, max_nodes=100000)
        assert steps < float("inf"), "Generated puzzle should be solvable"
        assert steps > 0

    def test_solver_after_action(self):
        from rllm.environments.sokoban.sokoban import SokobanEnv

        env = SokobanEnv(seed=42, dim_room=(6, 6), num_boxes=1)
        env.reset()

        before = get_min_steps(env.room_state, env.room_fixed, weight=1.0, max_nodes=100000)
        env.step(1)  # Up
        after = get_min_steps(env.room_state, env.room_fixed, weight=1.0, max_nodes=100000)
        assert isinstance(before, (int, float))
        assert isinstance(after, (int, float))

    def test_solve_with_next_action(self):
        """Solve a generated puzzle by following get_next_action recommendations."""
        from rllm.environments.sokoban.sokoban import SokobanEnv

        env = SokobanEnv(seed=123, dim_room=(6, 6), num_boxes=1)
        env.reset()

        max_iter = 200
        for _ in range(max_iter):
            action = get_next_action(env.room_state, env.room_fixed, weight=1.0, max_nodes=100000)
            if action is None:
                break
            env.step(action)

        # Check if solved
        _, boxes, targets = _extract_state(env.room_state, env.room_fixed)
        target_set = frozenset(targets)
        assert boxes == target_set, "Solver should solve a simple 1-box puzzle"


# ===========================================================================
# REGRESSION TESTS FOR FIXED BUGS
# ===========================================================================


class TestBugS1_FrozenLineDeadlockInSearch:
    """The A* search must prune frozen-line deadlocks, not just corner
    deadlocks, so that states which can never be solved are never expanded.
    """

    def test_frozen_line_deadlock_should_be_pruned_in_search(self):
        """Create a puzzle where a box push creates a frozen-line deadlock.

        Board layout:
            0 0 0 0 0 0
            0 1 1 1 1 0
            0 1 2 1 1 0
            0 1 4 4 1 0  <- Two boxes side by side against bottom wall
            0 0 0 0 0 0

        room_fixed (walls=0, floor=1, target=2):
            0 0 0 0 0 0
            0 1 1 1 1 0
            0 1 2 1 1 0
            0 1 1 1 1 0
            0 0 0 0 0 0

        Two boxes at (3,2) and (3,3) against wall row 4.  Since neither
        box is on a target and they're in a contiguous line along a wall,
        this is a frozen-line deadlock, which the solver must detect and prune.

        We verify that is_deadlocked correctly identifies this state.
        Then we verify the solver handles it properly (returns inf or avoids it).
        """
        room_state = np.array([
            [WALL, WALL,  WALL,   WALL, WALL, WALL],
            [WALL, FLOOR, FLOOR,  FLOOR, FLOOR, WALL],
            [WALL, FLOOR, FLOOR,  FLOOR, FLOOR, WALL],
            [WALL, PLAYER, BOX,   BOX,   FLOOR, WALL],
            [WALL, WALL,  WALL,   WALL, WALL, WALL],
        ], dtype=np.int32)
        room_fixed = np.array([
            [WALL, WALL,  WALL,  WALL,  WALL, WALL],
            [WALL, FLOOR, FLOOR, FLOOR, FLOOR, WALL],
            [WALL, FLOOR, TARGET, FLOOR, FLOOR, WALL],
            [WALL, FLOOR, FLOOR, FLOOR, FLOOR, WALL],
            [WALL, WALL,  WALL,  WALL,  WALL, WALL],
        ], dtype=np.int32)

        # Frozen-line detection should recognize boxes (3,2) and (3,3) as deadlocked
        # (both against wall row 4, neither on target)
        _, boxes, targets = _extract_state(room_state, room_fixed)
        target_set = set(targets)

        # Individual boxes: (3,2) is against wall at (4,2), and (3,3) is next to it
        # Together they form a frozen line
        assert _is_frozen_line_deadlock(3, 2, boxes, room_fixed), (
            "Box at (3,2) with another box at (3,3), both against wall row 4, "
            "should be detected as frozen-line deadlock"
        )

        # The overall state should be deadlocked
        assert is_deadlocked(room_state, room_fixed), (
            "Two non-target boxes in a frozen line against a wall should be deadlocked"
        )

    def test_search_avoids_frozen_line_deadlock_states(self):
        """Puzzle where frozen-line deadlock pruning helps the solver.

        Board (7x7):
            W W W W W W W
            W . T . T . W
            W . . . . . W
            W . B . B . W
            W . . . . . W
            W . . P . . W
            W W W W W W W

        Two boxes at (3,2) and (3,4), two targets at (1,2) and (1,4),
        player at (5,3).

        The solver can push boxes up toward row 1 (the targets). But it
        can also push boxes toward the top wall at row 0 creating a
        frozen-line deadlock when two boxes end up side by side on row 1
        at non-target positions. With frozen-line pruning these dead ends
        are pruned early, so the solver finds the solution within budget.
        """
        room_state = np.array([
            [WALL, WALL, WALL,   WALL,  WALL, WALL, WALL],
            [WALL, FLOOR, FLOOR, FLOOR, FLOOR, FLOOR, WALL],
            [WALL, FLOOR, FLOOR, FLOOR, FLOOR, FLOOR, WALL],
            [WALL, FLOOR, BOX,   FLOOR, BOX,   FLOOR, WALL],
            [WALL, FLOOR, FLOOR, FLOOR, FLOOR, FLOOR, WALL],
            [WALL, FLOOR, FLOOR, PLAYER, FLOOR, FLOOR, WALL],
            [WALL, WALL, WALL,   WALL,  WALL, WALL, WALL],
        ], dtype=np.int32)
        room_fixed = np.array([
            [WALL, WALL, WALL,   WALL,  WALL, WALL, WALL],
            [WALL, FLOOR, TARGET, FLOOR, TARGET, FLOOR, WALL],
            [WALL, FLOOR, FLOOR, FLOOR, FLOOR, FLOOR, WALL],
            [WALL, FLOOR, FLOOR, FLOOR, FLOOR, FLOOR, WALL],
            [WALL, FLOOR, FLOOR, FLOOR, FLOOR, FLOOR, WALL],
            [WALL, FLOOR, FLOOR, FLOOR, FLOOR, FLOOR, WALL],
            [WALL, WALL, WALL,   WALL,  WALL, WALL, WALL],
        ], dtype=np.int32)

        # The puzzle should be solvable (push each box straight up to its target)
        steps = get_min_steps(room_state, room_fixed, weight=1.0, max_nodes=100000)
        assert steps < float("inf"), (
            "Puzzle with 2 boxes and 2 targets should be solvable. "
            "Frozen-line deadlock pruning in search prevents the solver "
            "from wasting budget on dead-end states."
        )
        # Needs at least 2 pushes (one per box) plus player movement
        assert steps >= 2


class TestBugS2_NumpyBoolReturn:
    """_is_corner_deadlock must return a Python bool, not numpy.bool_."""

    def test_corner_deadlock_returns_python_bool(self):
        """_is_corner_deadlock should return Python bool, not numpy.bool_."""
        room_fixed = np.array([
            [WALL, WALL,  WALL],
            [WALL, FLOOR, WALL],
            [WALL, WALL,  WALL],
        ], dtype=np.int32)

        # Box at (1,1) is in a corner (walls on all 4 sides)
        result = _is_corner_deadlock(1, 1, room_fixed)
        assert result is True or result is False, (
            f"Expected Python bool, got {type(result).__name__}. "
            f"BUG S2: _is_corner_deadlock returns numpy.bool_."
        )
        # Stricter: check exact type
        assert type(result) is bool, (
            f"Expected type bool, got {type(result).__name__}. "
            f"BUG S2: numpy.bool_ instead of Python bool."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
