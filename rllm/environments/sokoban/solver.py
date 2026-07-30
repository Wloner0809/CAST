"""Weighted A* solver oracle for Sokoban process guidance.

Provides optimal/near-optimal step counts and deadlock detection to generate
verbal feedback during agent rollouts. Solver weight controls feedback strength:
  - w=1.0  → exact A* (strong guidance)
  - w≈2.5  → approximate (moderate guidance)
  - w→∞    → greedy best-first (weak guidance)
"""

import heapq
from typing import Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

# Sokoban cell values (matching gym-sokoban / SokobanEnv conventions)
WALL = 0
FLOOR = 1
TARGET = 2
BOX_ON_TARGET = 3
BOX = 4
PLAYER = 5

# Movement deltas: up, down, left, right
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _extract_state(
    room_state: np.ndarray, room_fixed: np.ndarray
) -> Tuple[Tuple[int, int], frozenset, list]:
    """Extract (player_pos, box_positions, target_positions) from arrays."""
    player_positions = list(zip(*np.where(room_state == PLAYER)))
    player_pos = player_positions[0] if player_positions else (0, 0)

    boxes = set()
    for r, c in zip(*np.where(room_state == BOX)):
        boxes.add((int(r), int(c)))
    for r, c in zip(*np.where(room_state == BOX_ON_TARGET)):
        boxes.add((int(r), int(c)))

    targets = []
    for r, c in zip(*np.where(room_fixed == TARGET)):
        targets.append((int(r), int(c)))

    return player_pos, frozenset(boxes), targets


def _heuristic(boxes: frozenset, targets: list) -> float:
    """Greedy-assigned Manhattan distance heuristic (boxes → targets)."""
    if not boxes or not targets:
        return 0.0
    box_list = list(boxes)
    n = len(box_list)
    m = len(targets)
    cost = np.zeros((n, m), dtype=np.float64)
    for i, (br, bc) in enumerate(box_list):
        for j, (tr, tc) in enumerate(targets):
            cost[i, j] = abs(br - tr) + abs(bc - tc)
    row_ind, col_ind = linear_sum_assignment(cost)
    return float(cost[row_ind, col_ind].sum())


def _is_wall(room_fixed: np.ndarray, r: int, c: int) -> bool:
    """Check if position is a wall (out of bounds counts as wall)."""
    rows, cols = room_fixed.shape
    if r < 0 or r >= rows or c < 0 or c >= cols:
        return True
    return room_fixed[r, c] == WALL


def _is_corner_deadlock(
    box_r: int, box_c: int, room_fixed: np.ndarray
) -> bool:
    """Check if a box is stuck in a corner (two adjacent walls)."""
    wall_up = _is_wall(room_fixed, box_r - 1, box_c)
    wall_down = _is_wall(room_fixed, box_r + 1, box_c)
    wall_left = _is_wall(room_fixed, box_r, box_c - 1)
    wall_right = _is_wall(room_fixed, box_r, box_c + 1)
    return bool((wall_up or wall_down) and (wall_left or wall_right))


def _is_frozen_line_deadlock(
    box_r: int, box_c: int, boxes: frozenset, room_fixed: np.ndarray
) -> bool:
    """Check if a box is part of a frozen line of 2+ boxes along a wall."""
    rows, cols = room_fixed.shape
    # Check horizontal freeze: wall above or below entire contiguous box line
    for dr in [-1, 1]:
        if _is_wall(room_fixed, box_r + dr, box_c):
            # Count adjacent boxes in this line
            line_count = 1
            frozen = True
            # Walk left
            c = box_c - 1
            while c >= 0 and (box_r, c) in boxes:
                if not _is_wall(room_fixed, box_r + dr, c):
                    frozen = False
                    break
                line_count += 1
                c -= 1
            if frozen:
                # Walk right
                c = box_c + 1
                while c < cols and (box_r, c) in boxes:
                    if not _is_wall(room_fixed, box_r + dr, c):
                        frozen = False
                        break
                    line_count += 1
                    c += 1
            if frozen and line_count >= 2:
                return True

    # Check vertical freeze: wall left or right entire contiguous box line
    for dc in [-1, 1]:
        if _is_wall(room_fixed, box_r, box_c + dc):
            line_count = 1
            frozen = True
            r = box_r - 1
            while r >= 0 and (r, box_c) in boxes:
                if not _is_wall(room_fixed, r, box_c + dc):
                    frozen = False
                    break
                line_count += 1
                r -= 1
            if frozen:
                r = box_r + 1
                while r < rows and (r, box_c) in boxes:
                    if not _is_wall(room_fixed, r, box_c + dc):
                        frozen = False
                        break
                    line_count += 1
                    r += 1
            if frozen and line_count >= 2:
                return True

    return False


def _is_any_deadlock(
    box_r: int, box_c: int, boxes: frozenset, room_fixed: np.ndarray,
    target_set: set
) -> bool:
    """Check if a box at (box_r, box_c) is in any known deadlock state."""
    if (box_r, box_c) in target_set:
        return False
    if _is_corner_deadlock(box_r, box_c, room_fixed):
        return True
    if _is_frozen_line_deadlock(box_r, box_c, boxes, room_fixed):
        return True
    return False


def count_near_deadlock_boxes(
    room_state: np.ndarray, room_fixed: np.ndarray
) -> list[tuple[int, int]]:
    """Find non-target boxes that are one push away from any deadlock.

    For each non-target box that isn't already deadlocked, simulates all 4
    push directions. If pushing the box in any direction would create a
    deadlock (corner or frozen-line), the box is included.

    Returns:
        List of (row, col) positions of at-risk boxes.
    """
    _, boxes, targets = _extract_state(room_state, room_fixed)
    target_set = set(targets)
    rows, cols = room_fixed.shape
    at_risk = []

    for br, bc in boxes:
        if (br, bc) in target_set:
            continue
        # Already deadlocked boxes don't count as "near" deadlock
        if _is_any_deadlock(br, bc, boxes, room_fixed, target_set):
            continue

        for dr, dc in MOVES:
            nr, nc = br + dr, bc + dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            if room_fixed[nr, nc] == WALL:
                continue
            if (nr, nc) in boxes:
                continue
            # Pushing to a target is safe
            if (nr, nc) in target_set:
                continue
            # Simulate the push and check all deadlock types
            new_boxes = boxes - {(br, bc)} | {(nr, nc)}
            if _is_any_deadlock(nr, nc, new_boxes, room_fixed, target_set):
                at_risk.append((br, bc))
                break

    return at_risk


def classify_deadlock(
    room_state: np.ndarray, room_fixed: np.ndarray
) -> tuple[str, tuple[int, int]] | None:
    """Classify the type of deadlock in the current state.

    Returns:
        Tuple of (deadlock_type, (row, col)) for the first deadlocked box found,
        or None if no deadlock detected.
        deadlock_type is "corner" or "frozen".
    """
    _, boxes, targets = _extract_state(room_state, room_fixed)
    target_set = set(targets)

    for br, bc in boxes:
        if (br, bc) in target_set:
            continue
        if _is_corner_deadlock(br, bc, room_fixed):
            return "corner", (br, bc)

    for br, bc in boxes:
        if (br, bc) in target_set:
            continue
        if _is_frozen_line_deadlock(br, bc, boxes, room_fixed):
            return "frozen", (br, bc)

    return None


def is_deadlocked(room_state: np.ndarray, room_fixed: np.ndarray) -> bool:
    """Check if the current state has any deadlocked boxes."""
    return classify_deadlock(room_state, room_fixed) is not None


def get_min_steps(
    room_state: np.ndarray,
    room_fixed: np.ndarray,
    weight: float = 1.0,
    max_nodes: int = 50000,
) -> float:
    """Weighted A* search returning minimum steps to solve from current state.

    Args:
        room_state: Current room state array.
        room_fixed: Fixed room structure array.
        weight: Heuristic weight. 1.0 = exact A*, higher = faster but less optimal.
        max_nodes: Maximum nodes to expand before giving up.

    Returns:
        Minimum steps (int) if solution found, float('inf') otherwise.
    """
    player_pos, boxes, targets = _extract_state(room_state, room_fixed)
    target_set = frozenset(targets)

    if boxes == target_set:
        return 0

    if is_deadlocked(room_state, room_fixed):
        return float("inf")

    # State: (player_row, player_col, frozenset_of_boxes)
    start = (player_pos[0], player_pos[1], boxes)
    g_score = {(start[0], start[1], start[2]): 0}
    h = weight * _heuristic(boxes, targets)
    # Priority queue: (f_score, tie_breaker, state)
    counter = 0
    open_set = [(h, counter, start)]
    closed = set()

    rows, cols = room_fixed.shape

    while open_set and len(closed) < max_nodes:
        f, _, state = heapq.heappop(open_set)
        pr, pc, cur_boxes = state
        state_key = (pr, pc, cur_boxes)

        if state_key in closed:
            continue
        closed.add(state_key)

        cur_g = g_score.get(state_key, float("inf"))

        for dr, dc in MOVES:
            nr, nc = pr + dr, pc + dc

            # Out of bounds or wall
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            if room_fixed[nr, nc] == WALL:
                continue

            new_boxes = cur_boxes
            if (nr, nc) in cur_boxes:
                # Pushing a box
                br, bc = nr + dr, nc + dc
                if br < 0 or br >= rows or bc < 0 or bc >= cols:
                    continue
                if room_fixed[br, bc] == WALL:
                    continue
                if (br, bc) in cur_boxes:
                    continue
                new_boxes = cur_boxes - {(nr, nc)} | {(br, bc)}

                # Quick deadlock check on pushed box (corner + frozen-line)
                if _is_any_deadlock(br, bc, new_boxes, room_fixed, target_set):
                    continue

            new_state_key = (nr, nc, new_boxes)
            new_g = cur_g + 1

            if new_g < g_score.get(new_state_key, float("inf")):
                g_score[new_state_key] = new_g

                if new_boxes == target_set:
                    return new_g

                new_h = weight * _heuristic(new_boxes, targets)
                new_f = new_g + new_h
                counter += 1
                heapq.heappush(open_set, (new_f, counter, (nr, nc, new_boxes)))

    return float("inf")


def check_actions_for_deadlock(
    room_state: np.ndarray,
    room_fixed: np.ndarray,
    max_nodes: int = 20000,
) -> tuple[list[tuple[int, int]], int]:
    """Check which available push actions would lead to an unsolvable state.

    Simulates each of the 4 movement directions. For moves that push a box,
    builds the resulting room state and runs the A* solver to check whether
    the puzzle is still solvable. This catches ALL deadlock types — corner,
    frozen-line, 2x2 squares, mutual blocking, and any other unsolvable
    configuration — not just simple pattern-based checks.

    Args:
        room_state: Current room state array.
        room_fixed: Fixed room structure array.
        max_nodes: Max A* nodes per lookahead check. Lower = faster but may
            miss some solvable states (false positives). Default 20000.

    Returns:
        Tuple of (deadlock_boxes, total_push_action_count).
        deadlock_boxes: list of (row, col) of boxes whose push leads to
            an unsolvable state.
        total_push_action_count: number of valid actions that push a box.
    """
    player_pos, boxes, targets = _extract_state(room_state, room_fixed)
    target_set = frozenset(targets)
    rows, cols = room_fixed.shape
    pr, pc = player_pos

    deadlock_boxes: list[tuple[int, int]] = []
    push_count = 0

    for dr, dc in MOVES:
        nr, nc = pr + dr, pc + dc
        # Out of bounds or wall — not a valid move
        if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
            continue
        if room_fixed[nr, nc] == WALL:
            continue

        # No box at destination — just a player move, no deadlock risk
        if (nr, nc) not in boxes:
            continue

        # This move pushes a box
        br, bc = nr + dr, nc + dc
        # Push blocked by boundary, wall, or another box
        if br < 0 or br >= rows or bc < 0 or bc >= cols:
            continue
        if room_fixed[br, bc] == WALL:
            continue
        if (br, bc) in boxes:
            continue

        push_count += 1

        # Build the resulting state after the push
        new_boxes = boxes - {(nr, nc)} | {(br, bc)}

        # Already solved — definitely not a deadlock
        if new_boxes == target_set:
            continue

        # Quick pattern check first (cheap)
        if _is_any_deadlock(br, bc, new_boxes, room_fixed, set(targets)):
            deadlock_boxes.append((nr, nc))
            continue

        # Full solver check: build a temporary room_state and verify solvability
        new_room = room_state.copy()
        # Move player: clear old position, set new position
        new_room[pr, pc] = _cell_without_player(room_state[pr, pc], room_fixed[pr, pc])
        new_room[nr, nc] = PLAYER  # player moves to where the box was
        # Move box: clear old box position (now player), set new box position
        # The box's old position is now the player position (already set above)
        new_room[br, bc] = BOX_ON_TARGET if room_fixed[br, bc] == TARGET else BOX

        min_steps = get_min_steps(new_room, room_fixed, weight=2.5, max_nodes=max_nodes)
        if min_steps == float("inf"):
            deadlock_boxes.append((nr, nc))

    return deadlock_boxes, push_count


def _cell_without_player(cell_value: int, fixed_value: int) -> int:
    """Return the cell value after removing the player from it."""
    if cell_value == PLAYER:
        return TARGET if fixed_value == TARGET else FLOOR
    return cell_value


def get_next_action(
    room_state: np.ndarray,
    room_fixed: np.ndarray,
    weight: float = 1.0,
    max_nodes: int = 50000,
) -> int | None:
    """Run weighted A* and return the first action in the optimal path.

    Args:
        room_state: Current room state array.
        room_fixed: Fixed room structure array.
        weight: Heuristic weight. 1.0 = exact A*, higher = faster but less optimal.
        max_nodes: Maximum nodes to expand before giving up.

    Returns:
        Action integer (1=Up, 2=Down, 3=Left, 4=Right), or None if no solution found.
    """
    player_pos, boxes, targets = _extract_state(room_state, room_fixed)
    target_set = frozenset(targets)

    if boxes == target_set:
        return None  # Already solved

    if is_deadlocked(room_state, room_fixed):
        return None

    start = (player_pos[0], player_pos[1], boxes)
    g_score = {(start[0], start[1], start[2]): 0}
    h = weight * _heuristic(boxes, targets)
    counter = 0
    # Priority queue entries: (f_score, counter, state, first_action)
    open_set = [(h, counter, start, None)]
    closed = set()

    rows, cols = room_fixed.shape

    while open_set and len(closed) < max_nodes:
        f, _, state, first_action = heapq.heappop(open_set)
        pr, pc, cur_boxes = state
        state_key = (pr, pc, cur_boxes)

        if state_key in closed:
            continue
        closed.add(state_key)

        cur_g = g_score.get(state_key, float("inf"))

        for move_idx, (dr, dc) in enumerate(MOVES):
            action = move_idx + 1  # 1-indexed: Up=1, Down=2, Left=3, Right=4
            nr, nc = pr + dr, pc + dc

            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            if room_fixed[nr, nc] == WALL:
                continue

            new_boxes = cur_boxes
            if (nr, nc) in cur_boxes:
                br, bc = nr + dr, nc + dc
                if br < 0 or br >= rows or bc < 0 or bc >= cols:
                    continue
                if room_fixed[br, bc] == WALL:
                    continue
                if (br, bc) in cur_boxes:
                    continue
                new_boxes = cur_boxes - {(nr, nc)} | {(br, bc)}
                if _is_any_deadlock(br, bc, new_boxes, room_fixed, target_set):
                    continue

            new_state_key = (nr, nc, new_boxes)
            new_g = cur_g + 1

            if new_g < g_score.get(new_state_key, float("inf")):
                g_score[new_state_key] = new_g
                next_first_action = first_action if first_action is not None else action

                if new_boxes == target_set:
                    return next_first_action

                new_h = weight * _heuristic(new_boxes, targets)
                new_f = new_g + new_h
                counter += 1
                heapq.heappush(open_set, (new_f, counter, (nr, nc, new_boxes), next_first_action))

    return None
