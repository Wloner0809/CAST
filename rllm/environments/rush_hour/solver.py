"""IDA* solver oracle + static deadlock analysis for Rush Hour.

Ported from michaelfogleman's authoritative Go solver
(https://www.michaelfogleman.com/rush/, ``solver.go`` + ``static.go``).  The
public API deliberately mirrors ``rllm/environments/sokoban/solver.py`` so the
environment can compute the same potential-based oracle advantage and deadlock
signals:

* :func:`get_min_moves(board)` -> ``int | inf`` — optimal remaining number of
  *moves* to free the red car (a move = sliding one piece any distance), the
  analogue of sokoban's ``get_min_steps``.  ``inf`` means unsolvable.
* :func:`is_deadlocked(board)` -> ``bool`` — fast static check (blocked-square
  analysis) for "the red car can never reach the exit", the analogue of
  sokoban's corner/frozen-line ``is_deadlocked``.
* :func:`get_next_move(board)` -> ``(label, steps) | None`` — first move on an
  optimal path, for hint/feedback providers.

The IDA* search is exact when it completes within the configured limits
(admissible lower bound = number of occupied cells between the red car and the
exit), so a finite ``get_min_moves`` result is the true optimal move count.
However, ``inf`` also covers search-budget exhaustion, not only proven
unsolvability.
"""

from __future__ import annotations

from rllm.environments.rush_hour.board import (
    HORIZONTAL,
    VERTICAL,
    RushHourBoard,
)

INF = float("inf")


# ============================================================================
# Static deadlock analysis (static.go port)
# ============================================================================

def _blocked_squares_1d(n: int, positions: list[int], sizes: list[int], blocked: list[int]) -> list[int]:
    """Squares occupied in EVERY legal placement of a row/column's pieces.

    Given a line of length ``n`` containing pieces at ``positions`` with ``sizes``
    (sorted ascending by position) and a set of ``blocked`` squares (from the
    perpendicular direction), enumerate all non-overlapping placements that avoid
    the blocked squares and return the squares occupied in all of them.
    """
    m = len(positions)
    if m == 0:
        return []

    order = sorted(range(m), key=lambda k: positions[k])
    positions = [positions[k] for k in order]
    sizes = [sizes[k] for k in order]

    placements: list[list[int]] = []
    for i in range(m):
        s = sizes[i]
        x0 = 0
        x1 = n - s
        p = positions[i]
        for b in blocked:
            if b < p:
                x0 = max(x0, b + 1)
            elif b > p:
                x1 = min(x1, b - s)
        placements.append(list(range(x0, x1 + 1)))
        if not placements[-1]:
            # No legal placement for this piece given the blocks: the line is
            # internally inconsistent; treat as no forced squares.
            return []

    counts = [0] * n
    total = 0
    idx = [0] * m

    while True:
        ok = True
        for i in range(1, m):
            if placements[i][idx[i]] - placements[i - 1][idx[i - 1]] < sizes[i - 1]:
                ok = False
                break
        if ok:
            total += 1
            for i in range(m):
                p = placements[i][idx[i]]
                for j in range(sizes[i]):
                    counts[p + j] += 1
        # advance odometer
        i = m - 1
        while i >= 0 and idx[i] == len(placements[i]) - 1:
            idx[i] = 0
            i -= 1
        if i < 0:
            break
        idx[i] += 1

    if total == 0:
        return []
    return [i for i in range(n) if counts[i] == total]


def _compute_blocked(board: RushHourBoard) -> tuple[list[bool], list[bool]]:
    """Iterate row/column blocked-square analysis to a fixed point.

    Returns ``(horz_blocked, vert_blocked)`` cell masks.  ``vert`` squares are
    blocked for vertical movement (forced by row analysis); ``horz`` squares are
    blocked for horizontal movement (forced by column analysis).  Walls block
    both.
    """
    w, h = board.width, board.height
    n = w * h
    horz = [False] * n
    vert = [False] * n
    for wall in board.walls:
        horz[wall] = True
        vert[wall] = True

    changed = True
    while changed:
        changed = False
        # Rows -> vertical blocks
        for y in range(h):
            positions, sizes = [], []
            for piece in board.pieces:
                if piece.orientation == HORIZONTAL and piece.row(w) == y:
                    positions.append(piece.col(w))
                    sizes.append(piece.size)
            if not positions:
                continue
            row_blocked = [i for i in range(w) if horz[y * w + i]]
            for col in _blocked_squares_1d(w, positions, sizes, row_blocked):
                cell = y * w + col
                if not vert[cell]:
                    vert[cell] = True
                    changed = True
        # Cols -> horizontal blocks
        for x in range(w):
            positions, sizes = [], []
            for piece in board.pieces:
                if piece.orientation == VERTICAL and piece.col(w) == x:
                    positions.append(piece.row(w))
                    sizes.append(piece.size)
            if not positions:
                continue
            col_blocked = [i for i in range(h) if vert[i * w + x]]
            for row in _blocked_squares_1d(h, positions, sizes, col_blocked):
                cell = row * w + x
                if not horz[cell]:
                    horz[cell] = True
                    changed = True
    return horz, vert


def is_deadlocked(board: RushHourBoard) -> bool:
    """True if the red car provably can never reach the exit (static analysis).

    Mirrors sokoban's ``is_deadlocked``: a fast, sound (no false positives)
    pattern check.  If any square between the red car and the exit is forced to
    be occupied in every legal configuration, the puzzle is unsolvable.
    """
    if board.solved:
        return False
    horz, vert = _compute_blocked(board)
    primary = board.pieces[0]
    w = board.width
    i0 = primary.position + primary.size
    i1 = (primary.row(w) + 1) * w
    for i in range(i0, i1):
        if horz[i] or vert[i]:
            return True
    return False


# ============================================================================
# IDA* exact solver (solver.go port)
# ============================================================================

def _search(board: RushHourBoard, depth: int, max_depth: int, prev_piece: int,
            memo: dict, path: list, budget: list) -> bool:
    height = max_depth - depth
    if height == 0:
        return board.solved

    # Node budget: abort the whole search when exhausted (signalled via budget[0] < 0).
    budget[0] -= 1
    if budget[0] < 0:
        return False

    key = board.memo_key()
    prev = memo.get(key, -1)
    if prev >= height:
        return False
    memo[key] = height

    primary = board.pieces[0]
    w = board.width
    i0 = primary.position + primary.size
    i1 = board.target + primary.size
    min_moves = 0
    for i in range(i0, i1):
        if i < len(board.occupied) and board.occupied[i]:
            min_moves += 1
    if min_moves >= height:
        return False

    for piece_index, steps in board.legal_moves():
        if piece_index == prev_piece:
            continue
        board.do_move(piece_index, steps)
        solved = _search(board, depth + 1, max_depth, piece_index, memo, path, budget)
        board.undo_move(piece_index, steps)
        if budget[0] < 0:
            return False
        if solved:
            path[depth] = (piece_index, steps)
            return True
    return False


def solve(board: RushHourBoard, max_depth: int = 60, max_nodes: int = 200_000) -> list[tuple[int, int]] | None:
    """Return an optimal move list ``[(piece_index, steps), ...]`` or None.

    Iterative deepening A*: tries successively larger depth bounds until a
    solution of that exact length is found, guaranteeing optimality in moves.

    ``max_nodes`` bounds total search effort across all depth iterations; if the
    budget is exhausted before a solution is found, returns ``None`` (the caller
    treats this as "too hard / unsolvable within budget").  This prevents the
    exponential blow-up of deep searches from stalling generation or oracle
    queries, mirroring sokoban's ``max_nodes`` guard.
    """
    work = board.copy()
    if work.solved:
        return []
    if is_deadlocked(work):
        return None

    budget = [max_nodes]
    for depth in range(1, max_depth + 1):
        path: list = [None] * depth
        memo: dict = {}
        if _search(work, 0, depth, -1, memo, path, budget):
            return list(path)
        if budget[0] < 0:
            return None
    return None


def get_min_moves(board: RushHourBoard, max_depth: int = 60, max_nodes: int = 200_000) -> float:
    """Optimal remaining move count to free the red car, or ``inf``.

    Analogue of sokoban ``get_min_steps``.  A "move" is sliding one piece any
    number of cells (matching the agent's single-action semantics), so this is
    the natural potential ``N(s)`` for oracle-advantage shaping.  Returns ``inf``
    when the puzzle is unsolvable OR when the search exceeds ``max_nodes`` (too
    hard to resolve within budget).
    """
    if board.solved:
        return 0
    if is_deadlocked(board):
        return INF
    solution = solve(board, max_depth=max_depth, max_nodes=max_nodes)
    if solution is None:
        return INF
    return len(solution)


def get_next_move(board: RushHourBoard, max_depth: int = 60, max_nodes: int = 200_000) -> tuple[str, int] | None:
    """First move on an optimal path as ``(label, steps)``, or None."""
    solution = solve(board, max_depth=max_depth, max_nodes=max_nodes)
    if not solution:
        return None
    piece_index, steps = solution[0]
    return chr(ord("A") + piece_index), steps
