"""Tests for the Sokoban A* solver oracle."""
import pytest
import numpy as np

pytest.importorskip("gym_sokoban")
pytest.importorskip("gym")
pytest.importorskip("scipy")

from rllm.environments.sokoban.solver import (
    _extract_state,
    _heuristic,
    _is_corner_deadlock,
    classify_deadlock,
    count_near_deadlock_boxes,
    get_min_steps,
    is_deadlocked,
)


def _make_simple_puzzle():
    """Create a simple 5x5 puzzle: 1 box, 1 target, solvable in a few steps.

    Layout (room_state):
        #####
        #P__#
        #_X_#
        #__O#
        #####

    room_fixed has TARGET=2 where 'O' is, FLOOR=1 elsewhere inside walls.
    room_state has PLAYER=5, BOX=4.
    Solution: push box down then right (2 pushes).
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


def _make_corner_deadlock():
    """Box stuck in top-left corner (not on target).

    Layout:
        #####
        #X__#
        #___#
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


def _make_solved_puzzle():
    """Box already on target.

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


class TestExtractState:
    def test_basic_extraction(self):
        room_state, room_fixed = _make_simple_puzzle()
        player, boxes, targets = _extract_state(room_state, room_fixed)
        assert player == (1, 1)
        assert boxes == frozenset({(2, 2)})
        assert targets == [(3, 3)]

    def test_solved_state(self):
        room_state, room_fixed = _make_solved_puzzle()
        player, boxes, targets = _extract_state(room_state, room_fixed)
        assert player == (1, 1)
        assert (3, 3) in boxes  # BOX_ON_TARGET=3 is also a box
        assert targets == [(3, 3)]


class TestHeuristic:
    def test_zero_when_solved(self):
        room_state, room_fixed = _make_solved_puzzle()
        _, boxes, targets = _extract_state(room_state, room_fixed)
        assert _heuristic(boxes, targets) == 0.0

    def test_positive_when_unsolved(self):
        room_state, room_fixed = _make_simple_puzzle()
        _, boxes, targets = _extract_state(room_state, room_fixed)
        h = _heuristic(boxes, targets)
        assert h > 0
        # Manhattan distance from (2,2) to (3,3) = 2
        assert h == 2.0

    def test_empty_boxes(self):
        assert _heuristic(frozenset(), [(1, 1)]) == 0.0


class TestDeadlockDetection:
    def test_corner_deadlock(self):
        room_state, room_fixed = _make_corner_deadlock()
        assert is_deadlocked(room_state, room_fixed) is True
        result = classify_deadlock(room_state, room_fixed)
        assert result is not None
        assert result[0] == "corner"

    def test_no_deadlock_simple(self):
        room_state, room_fixed = _make_simple_puzzle()
        assert is_deadlocked(room_state, room_fixed) is False
        assert classify_deadlock(room_state, room_fixed) is None

    def test_solved_not_deadlocked(self):
        room_state, room_fixed = _make_solved_puzzle()
        assert is_deadlocked(room_state, room_fixed) is False

    def test_corner_deadlock_detection_function(self):
        room_fixed = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ])
        # (1,1) is surrounded by walls on all sides → corner
        assert _is_corner_deadlock(1, 1, room_fixed) == True


class TestGetMinSteps:
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
        # Weighted A* should find a solution (possibly suboptimal)
        assert approx < float("inf")
        assert approx >= exact  # weighted can't be shorter than optimal

    def test_max_nodes_limit(self):
        room_state, room_fixed = _make_simple_puzzle()
        # With very few nodes, might not find solution
        result = get_min_steps(room_state, room_fixed, weight=1.0, max_nodes=1)
        # Either finds it or returns inf
        assert isinstance(result, (int, float))


class TestNearDeadlockDetection:
    def test_box_near_corner(self):
        """Box at (2,1) in 7x7 room can be pushed to corner (1,1) → at risk."""
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
        assert len(at_risk) >= 1, "Box at (2,1) is one push from corner (1,1)"

    def test_box_on_target_not_counted(self):
        """Box on target should never be counted as near-deadlock."""
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
            [0, 1, 3, 1, 0],  # BOX_ON_TARGET
            [0, 1, 1, 5, 0],
            [0, 0, 0, 0, 0],
        ])
        assert len(count_near_deadlock_boxes(room_state, room_fixed)) == 0

    def test_box_already_deadlocked_not_counted(self):
        """Already-deadlocked boxes should not count as near-deadlock."""
        room_state, room_fixed = _make_corner_deadlock()
        # Box at (1,1) is already in a corner deadlock
        assert len(count_near_deadlock_boxes(room_state, room_fixed)) == 0

    def test_safe_center_box(self):
        """Box in center of large room with no nearby corners → 0 risk."""
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

    def test_near_frozen_line_deadlock(self):
        """Pushing a box next to another box along a wall → frozen line risk."""
        # Two boxes: one already against top wall at (1,2), another at (2,3).
        # Pushing (2,3) up to (1,3) would create a frozen line along top wall.
        # But (1,2) is already a wall-segment deadlock, so it's already dead.
        # Use a layout where the wall has a target to avoid wall-segment deadlock.
        room_fixed = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 2, 1, 1, 1, 0],  # target at (1,2)
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 2, 0],  # target at (4,5)
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ])
        room_state = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 4, 1, 1, 1, 0],  # box at (1,2) on wall, on target segment
            [0, 1, 1, 4, 1, 1, 0],  # box at (2,3), pushable up to (1,3)
            [0, 1, 1, 5, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ])
        at_risk = count_near_deadlock_boxes(room_state, room_fixed)
        # Box at (2,3) can be pushed to (1,3) creating a frozen line with (1,2)
        assert len(at_risk) >= 1, "Box at (2,3) is one push from frozen-line deadlock"


class TestSolverOnRealEnv:
    """Test solver on actual SokobanEnv-generated puzzles."""

    def test_solver_on_generated_puzzle(self):
        from rllm.environments.sokoban.sokoban import SokobanEnv

        env = SokobanEnv(seed=42, dim_room=(6, 6), num_boxes=1)
        env.reset()

        steps = get_min_steps(env.room_state, env.room_fixed, weight=1.0, max_nodes=100000)
        # Generated puzzles should be solvable
        assert steps < float("inf"), "Generated puzzle should be solvable"
        assert steps > 0, "Generated puzzle should require at least one step"

    def test_solver_after_action(self):
        from rllm.environments.sokoban.sokoban import SokobanEnv

        env = SokobanEnv(seed=42, dim_room=(6, 6), num_boxes=1)
        env.reset()

        before = get_min_steps(env.room_state, env.room_fixed, weight=1.0, max_nodes=100000)
        env.step(1)  # Up
        after = get_min_steps(env.room_state, env.room_fixed, weight=1.0, max_nodes=100000)

        # Both should be finite or inf, but the relationship depends on the action
        assert isinstance(before, (int, float))
        assert isinstance(after, (int, float))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
