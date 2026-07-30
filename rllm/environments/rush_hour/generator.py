"""Per-seed Rush Hour puzzle generator.

Deterministic, dependency-free puzzle generation in the style of sokoban's
``generate_room(seed)``: a given ``(seed, config)`` always yields the same
puzzle, and the result is validated with the IDA* solver so its optimal-move
count falls inside the requested difficulty window.

Strategy (random layout + solver validation):

1. Seed an isolated ``random.Random``.
2. Place the red car (size 2, horizontal) on the exit row at a random non-exit
   column.
3. Randomly place ``num_vehicles - 1`` other cars/trucks (size 2-3, random
   orientation) and ``num_walls`` walls, rejecting overlaps.
4. Reject if the board is already solved, deadlocked, or its optimal move count
   is outside ``[min_moves, max_moves]``.
5. On rejection, derive a new sub-seed and retry up to ``max_retries``.

Walls and horizontal cars are forbidden on the exit row (except the red car) so
the red car always has a well-defined exit lane, matching the authoritative
generator's invariants.
"""

from __future__ import annotations

import random

from rllm.environments.rush_hour.board import (
    HORIZONTAL,
    VERTICAL,
    PRIMARY_ROW,
    PRIMARY_SIZE,
    RushHourBoard,
)
from rllm.environments.rush_hour import solver as _solver


def _derive_seed(seed: int) -> int:
    """Deterministic 32-bit LCG step for retry sub-seeds (matches sokoban)."""
    return (1664525 * (int(seed) & 0xFFFFFFFF) + 1013904223) & 0xFFFFFFFF


def _try_build(
    rng: random.Random,
    width: int,
    height: int,
    num_vehicles: int,
    num_walls: int,
) -> RushHourBoard | None:
    """Attempt one random board layout. Returns a board or None on failure."""
    occupied = [False] * (width * height)
    primary_row = PRIMARY_ROW if PRIMARY_ROW < height else height // 2

    # Red car: random left column in [0, width - PRIMARY_SIZE - 1] so it is NOT
    # already at the exit.
    max_start = width - PRIMARY_SIZE - 1
    if max_start < 0:
        return None
    pc = rng.randint(0, max_start)
    for j in range(PRIMARY_SIZE):
        occupied[primary_row * width + pc + j] = True

    vehicles: list[tuple[int, int, int]] = []
    attempts = 0
    max_attempts = num_vehicles * 30
    while len(vehicles) < num_vehicles - 1 and attempts < max_attempts:
        attempts += 1
        size = rng.choice([2, 2, 3])  # bias toward cars
        orientation = rng.choice([HORIZONTAL, VERTICAL])
        if orientation == HORIZONTAL:
            row = rng.randint(0, height - 1)
            if row == primary_row:
                continue  # no other horizontal cars on the exit row
            col = rng.randint(0, width - size)
            cells = [row * width + col + k for k in range(size)]
        else:
            col = rng.randint(0, width - 1)
            row = rng.randint(0, height - size)
            cells = [(row + k) * width + col for k in range(size)]
        if any(occupied[c] for c in cells):
            continue
        for c in cells:
            occupied[c] = True
        vehicles.append((cells[0], size, orientation))

    if len(vehicles) < num_vehicles - 1:
        return None

    walls: list[int] = []
    wall_attempts = 0
    while len(walls) < num_walls and wall_attempts < num_walls * 30:
        wall_attempts += 1
        cell = rng.randint(0, width * height - 1)
        # Never wall the exit row (would block the lane) or an occupied cell.
        if cell // width == primary_row:
            continue
        if occupied[cell]:
            continue
        occupied[cell] = True
        walls.append(cell)

    return RushHourBoard.from_pieces(
        width, height, (primary_row, pc), vehicles, walls
    )


def _decompose(board: RushHourBoard) -> tuple[int, int, list[tuple[int, int, int]], list[int]]:
    """Split a board into (primary_row, primary_col, vehicles, walls).

    ``vehicles`` is a list of ``(position, size, orientation)`` for every piece
    except the red car; ``walls`` is the wall cell list.  Used by the hill
    climber to mutate the layout and rebuild a fresh board.
    """
    primary = board.pieces[0]
    pr = primary.row(board.width)
    pc = primary.col(board.width)
    vehicles = [
        (p.position, p.size, p.orientation) for p in board.pieces[1:]
    ]
    return pr, pc, vehicles, list(board.walls)


def _relocate_one(
    rng: random.Random,
    width: int,
    height: int,
    primary_row: int,
    primary_col: int,
    vehicles: list[tuple[int, int, int]],
    walls: list[int],
) -> RushHourBoard | None:
    """Move one random non-primary vehicle to a new legal cell, rebuild board.

    Returns a new board or None if a non-overlapping relocation was not found.
    """
    if not vehicles:
        return None
    occupied = [False] * (width * height)
    for j in range(primary_col, primary_col + PRIMARY_SIZE):
        occupied[primary_row * width + j] = True
    for w in walls:
        occupied[w] = True

    target = rng.randrange(len(vehicles))
    kept = []
    for i, (pos, size, orient) in enumerate(vehicles):
        if i == target:
            continue
        idx = pos
        stride = 1 if orient == HORIZONTAL else width
        for _ in range(size):
            occupied[idx] = True
            idx += stride
        kept.append((pos, size, orient))

    _, size, _ = vehicles[target]
    for _ in range(40):
        orient = rng.choice([HORIZONTAL, VERTICAL])
        if orient == HORIZONTAL:
            row = rng.randint(0, height - 1)
            if row == primary_row:
                continue
            col = rng.randint(0, width - size)
            cells = [row * width + col + k for k in range(size)]
        else:
            col = rng.randint(0, width - 1)
            row = rng.randint(0, height - size)
            cells = [(row + k) * width + col for k in range(size)]
        if any(occupied[c] for c in cells):
            continue
        new_vehicles = kept + [(cells[0], size, orient)]
        return RushHourBoard.from_pieces(
            width, height, (primary_row, primary_col), new_vehicles, walls
        )
    return None


def _hill_climb(
    board: RushHourBoard,
    start_n: int,
    rng: random.Random,
    width: int,
    height: int,
    min_moves: int,
    max_moves: int,
    cap: int,
    budget: int,
) -> tuple[RushHourBoard, int] | None:
    """Climb toward harder layouts by relocating pieces (energy = move count).

    Pure random placement is heavily skewed toward easy puzzles, so to reach
    higher difficulty windows we hill-climb: repeatedly relocate one vehicle and
    keep the change whenever it does not lower the optimal move count (and stays
    solvable and within ``max_moves``).  Returns ``(board, n)`` as soon as the
    move count enters ``[min_moves, max_moves]``, else ``None`` after ``budget``
    iterations.
    """
    best = board
    best_n = start_n
    if min_moves <= best_n <= max_moves:
        return best, best_n

    # Use a tighter node budget per solve during climbing: borderline-too-hard
    # candidates reject quickly, keeping each iteration cheap. A returned board's
    # move count was confirmed with this budget, so it is solvable and its optimum
    # lies in [min_moves, max_moves]; generate_board returns it without re-solving.
    climb_nodes = 150_000
    pr, pc, vehicles, walls = _decompose(best)
    for _ in range(budget):
        candidate = _relocate_one(rng, width, height, pr, pc, vehicles, walls)
        if candidate is None or candidate.solved:
            continue
        if _solver.is_deadlocked(candidate):
            continue
        n = _solver.get_min_moves(candidate, max_depth=cap + 1, max_nodes=climb_nodes)
        if n == _solver.INF or n > max_moves:
            continue
        if min_moves <= n <= max_moves:
            return candidate, int(n)
        # Accept non-worsening moves to keep climbing toward the window.
        if n >= best_n:
            best, best_n = candidate, int(n)
            pr, pc, vehicles, walls = _decompose(best)
    return None


def generate_board(
    seed: int,
    width: int = 6,
    height: int = 6,
    num_vehicles: int = 8,
    num_walls: int = 0,
    min_moves: int = 1,
    max_moves: int = 20,
    max_retries: int = 200,
    solver_max_depth: int = 40,
) -> tuple[RushHourBoard, int]:
    """Generate a solvable puzzle whose optimal move count is in range.

    Args:
        seed: Base seed; deterministic for a given ``(seed, config)``.
        width, height: Board dimensions.
        num_vehicles: Total pieces including the red car.
        num_walls: Number of immovable wall cells.
        min_moves, max_moves: Inclusive optimal-move difficulty window.
        max_retries: Max sub-seed attempts before giving up.
        solver_max_depth: IDA* depth cap (also an upper bound on accepted moves).

    Returns:
        ``(board, min_moves_optimal)`` — the puzzle and its true optimal move
        count (the difficulty label).

    Raises:
        RuntimeError: if no valid puzzle is found within ``max_retries``.
    """
    cur = int(seed)
    cap = min(max_moves, solver_max_depth)
    # Keep the acceptance budget synchronized with
    # RushHourEnvConfig.solver_max_nodes so every accepted puzzle can be
    # re-solved under the environment's default budget.
    accept_nodes = 300_000
    # Hill-climbing is only needed for harder windows that random sampling rarely
    # hits directly; for easy windows the direct path almost always succeeds.
    climb_budget = 0 if max_moves <= 8 else 40
    for _ in range(max_retries):
        rng = random.Random(cur)
        board = _try_build(rng, width, height, num_vehicles, num_walls)
        if board is not None and not board.solved:
            if not _solver.is_deadlocked(board):
                n = _solver.get_min_moves(board, max_depth=cap + 1, max_nodes=accept_nodes)
                if n != _solver.INF:
                    if min_moves <= n <= max_moves:
                        return board, int(n)
                    if climb_budget and n < min_moves:
                        # Too easy: climb toward the target difficulty window.
                        climbed = _hill_climb(
                            board, int(n), rng, width, height,
                            min_moves, max_moves, cap, climb_budget,
                        )
                        if climbed is not None:
                            return climbed
        cur = _derive_seed(cur)

    raise RuntimeError(
        f"Failed to generate a Rush Hour puzzle in {max_retries} attempts "
        f"(seed={seed}, width={width}, height={height}, "
        f"num_vehicles={num_vehicles}, moves=[{min_moves},{max_moves}])."
    )
