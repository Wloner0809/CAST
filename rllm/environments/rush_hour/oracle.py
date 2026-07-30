"""Exact ``N(s)`` table oracle for Rush Hour credit assignment.

This is the BFS-full-table backend for the oracle advantage.
It replaces per-step IDA* solving for the oracle advantage with a single
multi-source reverse BFS that labels every state in a board's connected
component with its EXACT optimal remaining move count ``N(s)``.

Why a table instead of IDA*:

* **Correctness.** Budget-limited IDA* (``solver.get_min_moves`` with a finite
  ``max_nodes``) can report ``INF`` for states that are actually solvable but
  whose proof is search-heavy. Feeding that false ``INF`` into the oracle gives a
  correct move a catastrophic ``-N`` penalty and corrupts the gradient. The BFS
  table is exact: a state is ``INF`` if and only if it is genuinely unreachable
  from any solved state.
* **Speed.** Rush Hour slides are reversible, so one BFS over the component
  (~10^3-10^4 states for 6x6) yields ``N(s)`` for every state at once; per-step
  lookups are then ``O(1)``. The same table also serves every rollout of the
  same board (cached by layout).

Key correctness properties validated by ``tests/envs/test_rush_hour_oracle.py``:

1. The move graph is undirected (every legal ``s -> s'`` has the reverse
   ``s' -> s``), so reverse BFS from the goals reaches exactly the solvable set.
2. A board can have MANY solved states (only the red car's position is
   constrained), so the reverse BFS must be seeded from ALL of them.
3. The "move" unit (one ``legal_moves`` slide of any distance) equals the unit
   IDA* counts, so table values match ``get_min_moves`` (under a large budget).
"""

from __future__ import annotations

import gzip
import hashlib
import heapq
import os
import pickle
from collections import OrderedDict, deque
from concurrent.futures import ProcessPoolExecutor
from threading import Lock

from rllm.environments.rush_hour.board import RushHourBoard

INF = float("inf")


def puzzle_md5(board_config: str) -> str:
    """MD5 of the flat board string — the stable key for sidecar lookups.

    Matches ``examples/rush_hour/prepare_rush_hour_data.py::_puzzle_hash`` so the
    precomputed sidecar and the dataset records agree on puzzle identity.
    """
    return hashlib.md5(board_config.encode()).hexdigest()


def _set_state(board: RushHourBoard, key: tuple[int, ...]) -> None:
    """Mutate ``board`` in place so its pieces sit at the positions in ``key``.

    ``key`` is a :meth:`RushHourBoard.memo_key` (one position per piece, in piece
    order). Rebuilds each piece's bitmask and the board occupancy grid so the
    board is a fully consistent state ready for :meth:`RushHourBoard.legal_moves`.
    Walls are static and already present in ``board.walls`` / occupancy.
    """
    width = board.width
    occupied = [False] * (width * board.height)
    for piece, pos in zip(board.pieces, key):
        piece.position = pos
        mask = 0
        cell = pos
        for _ in range(piece.size):
            mask |= 1 << cell
            cell += piece.stride
        piece.mask = mask
    for piece in board.pieces:
        cell = piece.position
        for _ in range(piece.size):
            occupied[cell] = True
            cell += piece.stride
    for wall in board.walls:
        occupied[wall] = True
    board.occupied = occupied


def _neighbors(work: RushHourBoard, key: tuple[int, ...]) -> list[tuple[int, ...]]:
    """Return the memo_keys reachable from ``key`` by one legal slide.

    Sets ``work`` to ``key`` then enumerates every ``(piece, steps)`` slide,
    recording the resulting state and undoing it so ``work`` is restored to
    ``key`` between probes.
    """
    _set_state(work, key)
    out: list[tuple[int, ...]] = []
    for piece_index, steps in work.legal_moves():
        work.do_move(piece_index, steps)
        out.append(work.memo_key())
        work.undo_move(piece_index, steps)
    return out


def _neighbors_weighted(
    work: RushHourBoard, key: tuple[int, ...]
) -> list[tuple[tuple[int, ...], int]]:
    """Return ``(neighbor_key, weight)`` for each legal slide from ``key``.

    Weight is the number of cells slid (``abs(steps)``), used by the cells
    potential. ``legal_moves`` guarantees ``steps != 0`` so every weight >= 1.
    """
    _set_state(work, key)
    out: list[tuple[tuple[int, ...], int]] = []
    for piece_index, steps in work.legal_moves():
        work.do_move(piece_index, steps)
        out.append((work.memo_key(), abs(steps)))
        work.undo_move(piece_index, steps)
    return out


def build_n_table(
    board: RushHourBoard, max_states: int = 1_000_000, potential: str = "moves"
) -> dict[tuple[int, ...], int] | None:
    """Build the exact ``{state: N(s)}`` table for ``board``'s component.

    Args:
        board: Any state of the layout to analyze (the component, and therefore
            the table, is identical for every state in it).
        max_states: Safety cap on component size. If the component exceeds this
            (e.g. a large board), returns ``None`` so the caller can fall back to
            IDA* single-query — the table would be too big to hold/build cheaply.
        potential: ``"moves"`` (default) labels each state with the optimal
            remaining move count (one slide of any distance counts 1, via
            unweighted reverse BFS). ``"cells"`` labels each state with the
            minimum total slid cells (a slide of k cells counts k, via a
            multi-source reverse Dijkstra over edge weights ``abs(steps)``).

    Returns:
        A dict mapping each SOLVABLE state to its optimal remaining cost.
        States absent from the dict are genuinely unsolvable (true ``INF``).
        Returns ``None`` if the component exceeds ``max_states``.
    """
    work = board.copy()
    # The red car only slides horizontally, so its row — and thus the exit cell
    # ``target`` — is invariant across the whole component. Compute it once.
    target = board.target
    start = board.memo_key()

    # --- Pass 1: enumerate the full connected component from the start, noting
    # every solved state (red car at the exit) as a BFS source for pass 2. ---
    component = {start}
    goals: list[tuple[int, ...]] = []
    if start[0] == target:
        goals.append(start)
    frontier = [start]
    while frontier:
        nxt: list[tuple[int, ...]] = []
        for key in frontier:
            for nk in _neighbors(work, key):
                if nk not in component:
                    if len(component) >= max_states:
                        return None  # too large; caller falls back to IDA*
                    component.add(nk)
                    if nk[0] == target:
                        goals.append(nk)
                    nxt.append(nk)
        frontier = nxt

    # --- Pass 2: multi-source reverse search from ALL solved states. Because the
    # graph is undirected (and edge weights are symmetric), distance-to-nearest-
    # goal == optimal cost to solve. States never reached here are unsolvable and
    # are intentionally omitted (lookup -> INF). ---
    dist: dict[tuple[int, ...], int] = {}
    if potential == "cells":
        # Weighted edges (abs(steps)): multi-source Dijkstra.
        heap: list[tuple[int, tuple[int, ...]]] = []
        for goal in goals:
            dist[goal] = 0
            heapq.heappush(heap, (0, goal))
        while heap:
            d, key = heapq.heappop(heap)
            if d > dist[key]:
                continue  # stale heap entry
            for nk, w in _neighbors_weighted(work, key):
                nd = d + w
                if nk not in dist or nd < dist[nk]:
                    dist[nk] = nd
                    heapq.heappush(heap, (nd, nk))
    else:
        # "moves": unweighted reverse BFS (one slide of any distance counts 1).
        queue: deque[tuple[int, ...]] = deque()
        for goal in goals:
            dist[goal] = 0
            queue.append(goal)
        while queue:
            key = queue.popleft()
            d = dist[key]
            for nk in _neighbors(work, key):
                if nk not in dist:
                    dist[nk] = d + 1
                    queue.append(nk)
    return dist


class NTableCache:
    """Process-wide LRU cache of exact ``N(s)`` tables, keyed by board layout.

    In training the same board (``board_config``) is rolled out ``rollout.n``
    times and stepped many times, so building the table once and sharing it
    across all those envs/steps is the dominant win. The cache is keyed by the
    board's flat string (``board.to_string()``), which uniquely identifies the
    layout, and bounded by an LRU capacity to cap memory.

    Thread-safety: ``reset`` may run from multiple threads. A small registry lock
    guards the LRU dict; the expensive build for a given key happens under a
    per-key lock so distinct boards build in parallel while the same board is
    built at most once.
    """

    _tables: "OrderedDict[str, dict | None]" = OrderedDict()
    _key_locks: dict[str, Lock] = {}
    _registry_lock = Lock()
    _capacity = 256

    @classmethod
    def configure(cls, capacity: int) -> None:
        """Set the LRU capacity (number of distinct board tables kept)."""
        with cls._registry_lock:
            cls._capacity = max(1, int(capacity))
            cls._evict_locked()

    @classmethod
    def _evict_locked(cls) -> None:
        """Drop least-recently-used tables until within capacity. Caller holds lock."""
        while len(cls._tables) > cls._capacity:
            cls._tables.popitem(last=False)

    @classmethod
    def get_or_build(
        cls,
        board: RushHourBoard,
        cache_key: str,
        max_states: int = 1_000_000,
        capacity: int | None = None,
        potential: str = "moves",
    ) -> dict[tuple[int, ...], int] | None:
        """Return the cached table for ``cache_key``, building it once if absent.

        Returns ``None`` (cached) when the board's component exceeds
        ``max_states``; the caller should then fall back to IDA* single-query.

        ``potential`` selects the cost unit (``"moves"`` vs ``"cells"``). It is
        folded into the actual cache key so the process-wide cache never serves a
        moves table where a cells table was requested (or vice versa).
        """
        # Fold potential into the key: the process-wide cache is shared across
        # boards/configs, so a moves table and a cells table for the same layout
        # MUST NOT collide on the same key.
        real_key = f"{potential}:{cache_key}"

        # Adopt a larger capacity if a caller asks for one, but never shrink here:
        # get_or_build runs on every reset and configs in one process are expected
        # to agree, so shrinking per-call could thrash-evict freshly built tables
        # if two configs ever differ. Deliberate shrink is still possible via the
        # explicit configure() classmethod.
        if capacity is not None and capacity > cls._capacity:
            cls.configure(capacity)

        # Fast path: already cached (record LRU use). ``None`` is a valid cached
        # value meaning "too large — use the fallback", so distinguish presence
        # from a real miss with a sentinel.
        with cls._registry_lock:
            if real_key in cls._tables:
                cls._tables.move_to_end(real_key)
                return cls._tables[real_key]
            key_lock = cls._key_locks.get(real_key)
            if key_lock is None:
                key_lock = Lock()
                cls._key_locks[real_key] = key_lock

        # Build under the per-key lock so the same board is built at most once
        # while different boards build concurrently.
        with key_lock:
            with cls._registry_lock:
                if real_key in cls._tables:
                    cls._tables.move_to_end(real_key)
                    return cls._tables[real_key]

            table = build_n_table(board, max_states=max_states, potential=potential)

            with cls._registry_lock:
                cls._tables[real_key] = table
                cls._tables.move_to_end(real_key)
                cls._evict_locked()
                # The per-key lock is no longer needed once the result is cached.
                cls._key_locks.pop(real_key, None)
            return table

    @classmethod
    def clear(cls) -> None:
        """Drop all cached tables (test/debug helper)."""
        with cls._registry_lock:
            cls._tables.clear()
            cls._key_locks.clear()


# ======================================================================
# Offline precomputed sidecar (the "precomputed" backend)
# ======================================================================
#
# A sidecar maps ``puzzle_md5(board_config) -> N(s) table`` for every board in a
# dataset, computed once offline so the online oracle is a pure lookup with zero
# solve cost. The format is a gzipped pickle of:
#     {"version": 2, "potential": "moves"|"cells", "tables": {md5: {state: int}}}
# Build it with ``build_sidecar`` (or the prepare script), load it once per
# process via ``SidecarStore``.

SIDECAR_VERSION = 2


def _tabulate_one(task):
    """Process-pool worker: build one board's N(s) table from its flat string.

    ``task`` is ``(board_config, width, height, max_states, potential)``. Returns
    ``(puzzle_md5(board_config), table_or_None)``. Module-level + pure (depends
    only on the task args, no shared state) so it is picklable for
    ``ProcessPoolExecutor`` and yields output identical to a serial build.
    """
    board_config, width, height, max_states, potential = task
    board = RushHourBoard.from_string(board_config, width=width, height=height)
    return puzzle_md5(board_config), build_n_table(
        board, max_states=max_states, potential=potential
    )


def build_sidecar(
    board_configs: list[str],
    width: int,
    height: int,
    max_states: int = 1_000_000,
    progress=None,
    num_workers: int = 1,
    potential: str = "moves",
) -> dict:
    """Build a sidecar payload mapping each board's md5 to its exact N(s) table.

    Args:
        board_configs: Flat board strings (the dataset's ``board_config`` column).
        width, height: Board geometry (all boards in a dataset share this).
        max_states: Component-size cap; a board exceeding it is recorded as
            ``None`` so the online side knows to fall back to IDA* rather than
            trusting a missing entry as INF.
        progress: Optional callable applied to the per-board result iterable to
            wrap it (e.g. ``tqdm``) for a progress bar. Identity if None.
        num_workers: Process count for the (CPU-heavy, embarrassingly parallel)
            per-board BFS. 1 = serial. ``_tabulate_one`` is pure, so the payload
            is identical regardless of worker count.
        potential: Cost unit (``"moves"`` default / ``"cells"``). Recorded in the
            payload so the online side can verify the table matches its config.

    Returns:
        A picklable dict ``{"version": SIDECAR_VERSION, "potential": potential,
        "tables": {md5: table|None}}``. Duplicate board_configs collapse to one
        entry (same md5, same table).
    """
    # Dedup up front (preserving first-seen order) so the progress bar total and
    # the work both reflect unique puzzles only.
    seen: set[str] = set()
    unique: list[str] = []
    for board_config in board_configs:
        key = puzzle_md5(board_config)
        if key not in seen:
            seen.add(key)
            unique.append(board_config)

    tasks = [(bc, width, height, max_states, potential) for bc in unique]
    tables: dict[str, dict | None] = {}
    if num_workers <= 1:
        results = (_tabulate_one(t) for t in tasks)
        results = progress(results) if progress is not None else results
        for md5, table in results:
            tables[md5] = table
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # chunksize=1: per-board build time varies widely, so fine-grained
            # dispatch keeps all workers busy.
            results = executor.map(_tabulate_one, tasks, chunksize=1)
            results = progress(results) if progress is not None else results
            for md5, table in results:
                tables[md5] = table
    return {"version": SIDECAR_VERSION, "potential": potential, "tables": tables}


def save_sidecar(payload: dict, path: str) -> None:
    """Write a sidecar payload to ``path`` as a gzipped pickle (atomic rename)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp"
    with gzip.open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def load_sidecar(path: str) -> tuple[dict[str, dict | None], str]:
    """Load a sidecar file and return its ``({md5: table|None}, potential)``.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if the payload version is unrecognised.
    """
    with gzip.open(path, "rb") as f:
        payload = pickle.load(f)
    version = payload.get("version")
    if version != SIDECAR_VERSION:
        raise ValueError(
            f"Unsupported rush_hour sidecar version {version!r} at {path} "
            f"(expected {SIDECAR_VERSION})"
        )
    # Default to "moves" for forward-compat, though v2 payloads always carry it.
    return payload["tables"], payload.get("potential", "moves")


class SidecarStore:
    """Process-wide registry of loaded sidecars, keyed by ``(path, potential)``.

    Each sidecar file is loaded at most once per process per expected potential
    (the load is the only cost; lookups are plain dict hits). Thread-safe load
    under a registry lock. A loaded sidecar maps ``puzzle_md5 -> table|None``; a
    key that is genuinely absent (puzzle not in the sidecar) is reported
    distinctly from a present-but-``None`` entry (board too large) so the env can
    choose the right fallback. A sidecar whose recorded ``potential`` does not
    match the env's ``expected_potential`` is treated as a load failure (all
    MISSING) so the env rebuilds online instead of trusting a mismatched table.
    """

    _stores: dict[tuple[str, str], dict[str, dict | None]] = {}
    _lock = Lock()
    # Sentinel returned when the md5 is not in the sidecar at all.
    MISSING = object()
    # (path, potential) pairs that failed to load (missing file / bad version /
    # potential mismatch). Treated as an all-MISSING sidecar so the env falls
    # back to online build instead of crashing; warned once per pair.
    _failed: set[tuple[str, str]] = set()

    @classmethod
    def _get_store(
        cls, path: str, expected_potential: str = "moves"
    ) -> dict[str, dict | None] | None:
        """Load (once) and return the {md5: table|None} mapping for ``path``.

        Returns None if the sidecar could not be loaded (missing file, bad
        version, or a ``potential`` that does not match ``expected_potential``).
        The failure is recorded so the load is attempted only once and the
        warning printed only once; callers treat None as "all MISSING" and fall
        back to online build, honoring the "stale sidecar is safe" contract.
        """
        store_key = (path, expected_potential)
        store = cls._stores.get(store_key)
        if store is not None:
            return store
        with cls._lock:
            store = cls._stores.get(store_key)
            if store is not None:
                return store
            if store_key in cls._failed:
                return None
            try:
                store, loaded_potential = load_sidecar(path)
            except (FileNotFoundError, ValueError, OSError, EOFError) as exc:
                # Unloadable sidecar: degrade to online build rather than crash.
                cls._failed.add(store_key)
                print(
                    f"[rush_hour.oracle] Warning: could not load sidecar {path!r} "
                    f"({type(exc).__name__}: {exc}); falling back to online table build."
                )
                return None
            if loaded_potential != expected_potential:
                # Potential mismatch: the table encodes a different cost unit than
                # the env requested. Refuse it so we don't serve wrong values.
                cls._failed.add(store_key)
                print(
                    f"[rush_hour.oracle] Warning: sidecar {path!r} has potential "
                    f"{loaded_potential!r} but env expects {expected_potential!r}; "
                    f"falling back to online table build."
                )
                return None
            cls._stores[store_key] = store
        return store

    @classmethod
    def lookup(cls, path: str, board_config: str, expected_potential: str = "moves"):
        """Return the table for ``board_config`` from sidecar ``path``.

        Returns one of:
        * a ``dict`` table (the puzzle's exact N(s)),
        * ``None`` — the puzzle is in the sidecar but was too large to tabulate,
        * :attr:`SidecarStore.MISSING` — the puzzle is not in the sidecar, the
          sidecar file could not be loaded (stale/incomplete/missing), or its
          recorded potential did not match ``expected_potential``; caller should
          rebuild online.
        """
        store = cls._get_store(path, expected_potential)
        if store is None:
            return cls.MISSING
        return store.get(puzzle_md5(board_config), cls.MISSING)

    @classmethod
    def clear(cls) -> None:
        """Drop all loaded sidecars (test/debug helper)."""
        with cls._lock:
            cls._stores.clear()
            cls._failed.clear()
