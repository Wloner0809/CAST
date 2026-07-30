"""Solver-based oracle value/advantage for Minesweeper credit assignment.

Implements the solver-guided oracle advantage:

    A(s_t, a_t) = K(s_t) - K(s_{t+1})

where ``K(s)`` is the number of REVEAL actions a peek-free deterministic solver
needs to clear every safe cell from state ``s`` (``INF`` if it dies / cannot
win). Under the sparse 0/1 reward this is the potential-based shaping signal
``Phi(s) = -K(s)``, exactly analogous to Sokoban's ``N(s_t) - N(s_{t+1})`` --
no discount factor is needed.

Flag handling:
    The agent's flags are DISCARDED for the solver's reasoning -- a wrong agent
    flag must never corrupt the CSP into declaring a real mine "safe". The
    solver re-derives mines via CSP and only uses its OWN proven flags. The
    agent's flags contribute ONLY a deterministic unflag cost: each agent-flagged
    SAFE cell must be unflagged once before it can be revealed (+1 to K).

This module is a PURE simulation: it imports only ``solve_board`` (the solver's
decision function, which is peek-free for action selection) and replicates the
environment's flood-fill / win-check / mine-hit semantics on local copies of
``(grid, revealed, flags)``. The true grid is used ONLY to resolve transitions,
never to inform solver decisions. Keeping it env-free avoids a circular import
with ``minesweeper_env`` and avoids per-call environment object churn.
"""

from __future__ import annotations

from collections import deque

from rllm.environments.minesweeper.solver import solve_board

INF = float("inf")

# 8-directional adjacency, identical to MinesweeperEnv.
_DIRECTIONS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def _all_safe_revealed(
    grid: list[list[int]], revealed: list[list[bool]], rows: int, cols: int
) -> bool:
    """True when every non-mine cell is revealed (mirrors MinesweeperEnv.success)."""
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != -1 and not revealed[r][c]:
                return False
    return True


def _flood_reveal(
    grid: list[list[int]],
    revealed: list[list[bool]],
    flags: list[list[bool]],
    row: int,
    col: int,
    rows: int,
    cols: int,
) -> None:
    """BFS flood-fill from (row, col), mirroring MinesweeperEnv._flood_reveal.

    Reveals the start cell; for each revealed 0-cell, reveals its unrevealed,
    unflagged neighbors and continues. Mutates ``revealed`` in place.
    """
    queue = deque([(row, col)])
    revealed[row][col] = True
    while queue:
        cr, cc = queue.popleft()
        if grid[cr][cc] == 0:
            for dr, dc in _DIRECTIONS:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if not revealed[nr][nc] and not flags[nr][nc]:
                        revealed[nr][nc] = True
                        queue.append((nr, nc))


def solver_value_K(
    rows: int,
    cols: int,
    true_grid: list[list[int]],
    revealed: list[list[bool]],
    agent_flags: list[list[bool]] | None = None,
    total_mines: int | None = None,
    mode: str = "strong",
    max_steps: int | None = None,
) -> float:
    """Oracle value ``K(s)``: solver reveal-steps to win from ``s``, else ``INF``.

    Args:
        rows, cols: Board dimensions.
        true_grid: Full layout, ``grid[r][c] == -1`` for mines (simulator truth).
        revealed: Revealed mask of state ``s``.
        agent_flags: The agent's (possibly WRONG) flags. DISCARDED for the
            solver's reasoning; counted only for the deterministic unflag cost.
            ``None`` means no agent flags.
        total_mines: Total mine count (derived from ``true_grid`` if omitted).
        mode: Solver mode ("strong" / "approximate" / "weak").
        max_steps: Safety cap on solver actions (default ``rows*cols*4``).

    Returns:
        ``K(s)`` (non-negative int as float) on a winnable rollout, else ``INF``.
    """
    if total_mines is None:
        total_mines = sum(
            1 for r in range(rows) for c in range(cols) if true_grid[r][c] == -1
        )
    if max_steps is None:
        max_steps = rows * cols * 4

    # Local copies so the caller's state is untouched. Agent flags are NOT loaded:
    # sim_flags starts empty and only accumulates the solver's own proven flags.
    sim_revealed = [row[:] for row in revealed]
    sim_flags = [[False] * cols for _ in range(rows)]

    k_reveal = 0
    for _ in range(max_steps):
        if _all_safe_revealed(true_grid, sim_revealed, rows, cols):
            break
        state = solve_board(
            true_grid, sim_revealed, sim_flags, rows, cols,
            mode=mode, total_mines=total_mines,
        )
        action = state.best_action
        if action is None:
            break
        kind, r, c = action
        if kind == "reveal":
            if true_grid[r][c] == -1:
                return INF                          # forced-guess death
            _flood_reveal(true_grid, sim_revealed, sim_flags, r, c, rows, cols)
            k_reveal += 1
        else:  # solver's own CSP-proven flag (correct); not counted in K
            sim_flags[r][c] = True

    if not _all_safe_revealed(true_grid, sim_revealed, rows, cols):
        return INF                                  # could not clear the board

    unflag_cost = 0
    if agent_flags is not None:
        unflag_cost = sum(
            1
            for r in range(rows)
            for c in range(cols)
            if agent_flags[r][c] and true_grid[r][c] != -1 and not revealed[r][c]
        )
    return float(k_reveal + unflag_cost)


def oracle_advantage(k_before: float, k_after: float) -> float:
    """Linear-potential oracle advantage ``A = K(s_t) - K(s_{t+1})`` (form L).

    Handles the four ``INF`` boundary cases symmetrically (docs section 6.3):
      * enter deadlock  (K_after = INF): ``-K_before``  (catastrophe penalty)
      * already deadlock (both INF):      ``0``          (no gradient)
      * rescue deadlock  (K_before = INF): ``+K_after``  (LLM beats solver)
      * normal:                            ``K_before - K_after``
    """
    if k_before == INF and k_after == INF:
        return 0.0
    if k_before != INF and k_after == INF:
        return -float(k_before)
    if k_before == INF and k_after != INF:
        return float(k_after)
    return float(k_before - k_after)
