"""Tests for the Rush Hour exact ``N(s)`` table oracle (``oracle.py``).

These exercise REAL behaviour against ground truth, not parameter passing:
  * The two potentials (``"moves"`` / ``"cells"``) are checked numerically on
    hand-solved boards (B1, B2) where the optimal move count and the optimal
    total slid-cell count are both known by hand.
  * The weighted Dijkstra used by the cells potential is validated against an
    INDEPENDENT second shortest-path implementation written in the test
    (multi-source weighted Dijkstra over ``_neighbors_weighted``), since the
    IDA* solver optimizes move count, not slid cells, and so cannot serve as
    the cells ground truth.
  * The process-wide cache must not serve a moves table where a cells table was
    requested (the potential is folded into the cache key).
  * Solved / deadlock invariants (N==0 / absent==INF) hold under both potentials.
  * The offline sidecar carries its potential and refuses a mismatched lookup.
"""

import heapq

import pytest

from rllm.environments.rush_hour.board import RushHourBoard
from rllm.environments.rush_hour import oracle as O
from rllm.environments.rush_hour.oracle import (
    NTableCache,
    SidecarStore,
    build_n_table,
    build_sidecar,
    load_sidecar,
    save_sidecar,
)
from rllm.environments.rush_hour.generator import generate_board


INF = float("inf")

# Fixed hand-solved boards (6x6 flat strings) -------------------------------
# B1: red car AA at row2 col0-1, exit row fully empty. Optimal solve = "A+4"
#     (one slide of 4 cells): N_moves(s0)=1, N_cells(s0)=4.
B1 = "oooooo" + "oooooo" + "AAoooo" + "oooooo" + "oooooo" + "oooooo"
# B2: red car AA at row2 col0-1, vertical car B blocking (1,3)(2,3). Optimal
#     solve = "B-1" then "A+4": N_moves(s0)=2, N_cells(s0)=1+4=5.
B2 = "oooooo" + "oooBoo" + "AAoBoo" + "oooooo" + "oooooo" + "oooooo"
# Deadlock board (column 4 permanently blocked) from test_rush_hour_solver.py.
DEADLOCK = "ooooCo" + "ooooCo" + "AAooCo" + "ooooDo" + "ooooDo" + "ooooDo"


def _simple_board() -> RushHourBoard:
    """Mirror of the solver test's helper: red car AA solvable in 1 (4 cells)."""
    desc = (
        "ooBooo"
        "ooBooo"
        "AAoooo"
        "oooooo"
        "oooooo"
        "oooooo"
    )
    return RushHourBoard.from_string(desc)


# ---------------------------------------------------------------------------
# #1 — cells potential numeric correctness
# ---------------------------------------------------------------------------

def test_n_cells_potential_value():
    """cells = minimum total slid cells; moves = minimum slide count."""
    b1 = RushHourBoard.from_string(B1)
    b2 = RushHourBoard.from_string(B2)

    t1_cells = build_n_table(b1, potential="cells")
    t1_moves = build_n_table(b1, potential="moves")
    assert t1_cells[b1.memo_key()] == 4
    assert t1_moves[b1.memo_key()] == 1

    t2_cells = build_n_table(b2, potential="cells")
    t2_moves = build_n_table(b2, potential="moves")
    assert t2_cells[b2.memo_key()] == 5
    assert t2_moves[b2.memo_key()] == 2

    # Invariant: every move slides at least one cell, so cells >= moves
    # for every solvable state in the component (not just s0).
    for board, cells_t, moves_t in (
        (b1, t1_cells, t1_moves),
        (b2, t2_cells, t2_moves),
    ):
        assert set(cells_t) == set(moves_t)
        for state in cells_t:
            assert cells_t[state] >= moves_t[state]


# ---------------------------------------------------------------------------
# #5 — weighted-edge inequality (Dijkstra distances are a valid metric)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("desc", [B1, B2])
def test_weighted_dijkstra_edge_weight_inequality(desc):
    """For every weighted edge: |dist[s] - dist[s']| <= weight, with tight edges."""
    board = RushHourBoard.from_string(desc)
    dist = build_n_table(board, potential="cells")
    work = board.copy()

    saw_tight_edge = False
    for state, ds in dist.items():
        for nk, w in O._neighbors_weighted(work, state):
            # Neighbor must be solvable too (undirected graph, same component).
            assert nk in dist, "weighted neighbor missing from solvable set"
            dn = dist[nk]
            assert abs(ds - dn) <= w, (
                f"edge violates triangle inequality: |{ds}-{dn}| > {w}"
            )
            if ds - dn == w:
                saw_tight_edge = True
    assert saw_tight_edge, "expected at least one tight edge on an optimal path"


# ---------------------------------------------------------------------------
# #6 — independent forward Dijkstra cross-check (the highest-risk check) ★
# ---------------------------------------------------------------------------

def _independent_multisource_dijkstra(board: RushHourBoard) -> dict:
    """A second, independently written weighted multi-source Dijkstra.

    Seeded from ALL solved states (red car at the exit), relaxing edges with
    weight = cells slid. Intentionally a different loop than ``build_n_table``
    (own component enumeration + own heap) so agreement is real cross-validation
    rather than re-running the same code. Returns ``{state: min_total_cells}``.
    """
    work = board.copy()
    target = board.target
    start = board.memo_key()

    # 1. Enumerate the full connected component (own BFS), collecting goals.
    component = {start}
    goals = [start] if start[0] == target else []
    stack = [start]
    while stack:
        key = stack.pop()
        for nk, _w in O._neighbors_weighted(work, key):
            if nk not in component:
                component.add(nk)
                if nk[0] == target:
                    goals.append(nk)
                stack.append(nk)

    # 2. Multi-source Dijkstra from every goal (own heap, own relaxation).
    best: dict = {g: 0 for g in goals}
    heap = [(0, g) for g in goals]
    heapq.heapify(heap)
    while heap:
        d, key = heapq.heappop(heap)
        if d > best.get(key, INF):
            continue
        for nk, w in O._neighbors_weighted(work, key):
            cand = d + w
            if cand < best.get(nk, INF):
                best[nk] = cand
                heapq.heappush(heap, (cand, nk))
    return best


@pytest.mark.parametrize(
    "board_factory",
    [
        lambda: RushHourBoard.from_string(B1),
        lambda: RushHourBoard.from_string(B2),
        _simple_board,
        lambda: generate_board(0, num_vehicles=6, min_moves=1, max_moves=6)[0],
        lambda: generate_board(3, num_vehicles=6, min_moves=1, max_moves=6)[0],
    ],
)
def test_independent_forward_dijkstra_cross_check(board_factory):
    """build_n_table(cells) must match an independent Dijkstra state-by-state."""
    board = board_factory()
    table = build_n_table(board, potential="cells")
    reference = _independent_multisource_dijkstra(board)

    # Same solvable set (states absent from either dict are INF).
    assert set(table) == set(reference)
    # Pointwise agreement on every state in the component.
    for state in table:
        assert table[state] == reference[state], (
            f"cells mismatch at {state}: table={table[state]} ref={reference[state]}"
        )


def test_multisource_required_for_dense_board():
    """Seeding from a single goal would overstate N — multi-source is required.

    A board can have many solved states (only the red car's column is fixed at
    the exit), so a single-source Dijkstra from just one goal yields distances
    that are >= the true multi-source distances, and strictly greater for at
    least one state when more than one goal exists.
    """
    board = _simple_board()
    table = build_n_table(board, potential="cells")
    target = board.target

    # Find all goals in the component via the independent enumeration.
    reference = _independent_multisource_dijkstra(board)
    goals = [s for s in reference if s[0] == target]
    assert len(goals) >= 2, "test board must have multiple solved states"

    # Single-source Dijkstra from one goal only.
    work = board.copy()
    single = {goals[0]: 0}
    heap = [(0, goals[0])]
    while heap:
        d, key = heapq.heappop(heap)
        if d > single.get(key, INF):
            continue
        for nk, w in O._neighbors_weighted(work, key):
            if nk in reference and d + w < single.get(nk, INF):
                single[nk] = d + w
                heapq.heappush(heap, (d + w, nk))

    # Single-source never undershoots, and is strictly larger somewhere.
    assert all(single[s] >= table[s] for s in table)
    assert any(single[s] > table[s] for s in table)


# ---------------------------------------------------------------------------
# #7 — cache key distinguishes potential (no cross-contamination) ★
# ---------------------------------------------------------------------------

def test_ntable_cache_key_distinguishes_potential():
    """Same board string, two potentials -> two distinct cached tables."""
    b = RushHourBoard.from_string(B1)
    s0 = b.memo_key()

    # moves first, then cells.
    NTableCache.clear()
    t_moves = NTableCache.get_or_build(b, cache_key=b.to_string(), potential="moves")
    t_cells = NTableCache.get_or_build(b, cache_key=b.to_string(), potential="cells")
    assert t_moves[s0] == 1
    assert t_cells[s0] == 4
    assert t_moves is not t_cells

    # Reverse order: cells first, then moves — must give the same result,
    # proving the first build never poisons the second.
    NTableCache.clear()
    t_cells2 = NTableCache.get_or_build(b, cache_key=b.to_string(), potential="cells")
    t_moves2 = NTableCache.get_or_build(b, cache_key=b.to_string(), potential="moves")
    assert t_cells2[s0] == 4
    assert t_moves2[s0] == 1

    # White-box: both potentials coexist under distinct keys in the LRU.
    assert f"moves:{b.to_string()}" in NTableCache._tables
    assert f"cells:{b.to_string()}" in NTableCache._tables
    NTableCache.clear()


# ---------------------------------------------------------------------------
# #8 — solved / deadlock invariants under the cells potential
# ---------------------------------------------------------------------------

def test_solved_state_cells_is_zero():
    board = _simple_board()
    board.apply_action("A", 4)  # slide red car to the exit -> solved
    assert board.solved
    table = build_n_table(board, potential="cells")
    assert table[board.memo_key()] == 0


def test_deadlock_start_absent_from_cells_table():
    """A genuinely unsolvable start state is absent (lookup -> INF), same as moves."""
    board = RushHourBoard.from_string(DEADLOCK)
    cells = build_n_table(board, potential="cells")
    moves = build_n_table(board, potential="moves")
    s0 = board.memo_key()
    assert s0 not in cells
    assert s0 not in moves
    assert cells.get(s0, INF) == INF


# ---------------------------------------------------------------------------
# #9 — sidecar regression across the potential dimension
# ---------------------------------------------------------------------------

def test_sidecar_potential_regression(tmp_path):
    """A cells sidecar round-trips; a mismatched expected_potential is refused."""
    payload = build_sidecar([B1], 6, 6, potential="cells")
    assert payload["version"] == 2
    assert payload["potential"] == "cells"

    path = str(tmp_path / "sidecar.pkl.gz")
    save_sidecar(payload, path)

    # load_sidecar returns a (tables, potential) tuple.
    tables, loaded_potential = load_sidecar(path)
    assert loaded_potential == "cells"
    b1 = RushHourBoard.from_string(B1)
    assert tables[O.puzzle_md5(B1)][b1.memo_key()] == 4

    # Through the store: a matching potential returns the cells table.
    SidecarStore.clear()
    result = SidecarStore.lookup(path, B1, expected_potential="cells")
    assert result is not SidecarStore.MISSING
    assert result[b1.memo_key()] == 4

    # A mismatched expected potential must be refused (no moves<-cells confusion).
    assert (
        SidecarStore.lookup(path, B1, expected_potential="moves")
        is SidecarStore.MISSING
    )
    SidecarStore.clear()
