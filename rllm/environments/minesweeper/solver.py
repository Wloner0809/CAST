"""CSP + probability-based solver oracle for Minesweeper process guidance.

Provides constraint-satisfaction analysis and mine probability estimation
to generate verbal feedback and hints during agent rollouts.

Solver modes control accuracy/speed trade-off:
  - "strong":      Full CSP + subset reasoning + area decomposition + exact probability
  - "approximate": CSP constraint propagation only (deterministic deductions)
  - "weak":        Single-cell constraint checking (fastest, least precise)

Optimisations over the naive approach:
  1. **Single-solve cache** (`MinesweeperSolverState` / `solve_board`):
     All downstream queries (safe cells, mine cells, probabilities, best action)
     are answered from one solve pass.  Feedback and hint providers call
     `solve_board()` once per step instead of 3-5 separate solver invocations.
  2. **Subset / superset constraint reasoning** (Phase 1b):
     Pairwise checking of overlapping constraints to derive new deterministic
     deductions that simple zero-propagation and full-flag detection miss.
  3. **Area decomposition** for CP enumeration:
     Cells sharing the same constraint set are grouped into "areas", reducing
     the search-tree width from #cells to #areas.
  4. **True global exact combinatorial probability**:
     Each boundary connected component's integer mine-count distribution is
     computed via area decomposition (solutions weighted by
     C(area_size, mines_in_area)), then ALL components and the unconstrained
     interior are coupled under the global constraint that exactly the
     remaining number of mines lie among the unknown cells.  This makes every
     boundary probability depend on the total mine count.  Falls back to a
     per-component approximation (plus interior uniform) when a component is too
     large or its enumeration is truncated.
  5. **Indexed constraint lookup** during backtracking:
     Only constraints involving the current cell/area are checked at each
     depth, giving O(degree) instead of O(K) per node.
  6. **Component size limit 40** (was 20) with 50 000-solution safeguard.
     ``_enumerate_with_areas`` reports whether the search was truncated at the
     solution cap; truncated components yield an incomplete solution set and are
     therefore NOT used for safe/mine deductions or exact probabilities.
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# 8-directional adjacency
DIRECTIONS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

# Guidance mode constants
GUIDANCE_WEIGHTS = {
    "strong": "strong",
    "approximate": "approximate",
    "weak": "weak",
}

# Solver limits
_MAX_COMPONENT_CELLS = 40
_MAX_SOLUTIONS = 50_000


# ============================================================================
# Cached solver result
# ============================================================================

@dataclass
class MinesweeperSolverState:
    """Cached result of a single full solver pass.

    All public helper functions can be answered from this object without
    re-running the solver.

    ``difficulty_level`` records the minimum solver phase needed to make
    deterministic progress from the current board state:

    - ``"basic_propagation"`` – single-cell constraint propagation suffices.
    - ``"subset_reasoning"``  – overlapping constraint (subset/superset)
      analysis is required.
    - ``"area_decomposition"`` – coupled constraint enumeration via area
      decomposition is required.
    - ``"stochastic"``         – no deterministic deductions exist; the agent
      must guess.
    """
    csp_result: dict[tuple[int, int], str] = field(default_factory=dict)
    probabilities: dict[tuple[int, int], float] = field(default_factory=dict)
    safe_cells: list[tuple[int, int]] = field(default_factory=list)
    mine_cells: list[tuple[int, int]] = field(default_factory=list)
    best_action: Optional[tuple[str, int, int]] = None
    difficulty_level: str = "stochastic"


# ============================================================================
# Low-level helpers
# ============================================================================

def _get_neighbors(r: int, c: int, rows: int, cols: int) -> list[tuple[int, int]]:
    """Return all valid neighbor coordinates for cell (r, c)."""
    result = []
    for dr, dc in DIRECTIONS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            result.append((nr, nc))
    return result


def _build_constraints(
    grid: list[list[int]],
    revealed: list[list[bool]],
    flags: list[list[bool]],
    rows: int,
    cols: int,
) -> list[tuple[list[tuple[int, int]], int]]:
    """Build constraint equations from revealed numbered cells.

    Each constraint is (unknown_cells, remaining_mine_count):
      - unknown_cells: list of unrevealed, unflagged neighbor positions
      - remaining_mine_count: number - flagged_neighbors
    """
    constraints = []
    for r in range(rows):
        for c in range(cols):
            if not revealed[r][c]:
                continue
            num = grid[r][c]
            if num < 0:
                continue

            neighbors = _get_neighbors(r, c, rows, cols)
            flagged = 0
            unknowns = []
            for nr, nc in neighbors:
                if flags[nr][nc]:
                    flagged += 1
                elif not revealed[nr][nc]:
                    unknowns.append((nr, nc))

            if unknowns:
                remaining = num - flagged
                constraints.append((unknowns, remaining))

    return constraints


# ============================================================================
# Phase 1: Iterative constraint propagation + subset reasoning
# ============================================================================

def _constraint_propagation(
    constraints: list[tuple[list[tuple[int, int]], int]],
    result: dict[tuple[int, int], str],
) -> bool:
    """Run one pass of basic constraint propagation.

    Modifies *result* in-place.  Returns True if any cell was deduced.
    """
    changed = False
    for unknowns, remaining in constraints:
        live = [cell for cell in unknowns if result.get(cell) == "unknown"]
        mines_found = sum(1 for cell in unknowns if result.get(cell) == "mine")
        adj_remaining = remaining - mines_found

        if adj_remaining < 0 or adj_remaining > len(live):
            continue
        if len(live) == 0:
            continue

        if adj_remaining == 0:
            for cell in live:
                if result[cell] != "safe":
                    result[cell] = "safe"
                    changed = True
        elif adj_remaining == len(live):
            for cell in live:
                if result[cell] != "mine":
                    result[cell] = "mine"
                    changed = True
    return changed


def _subset_reasoning(
    constraints: list[tuple[list[tuple[int, int]], int]],
    result: dict[tuple[int, int], str],
) -> bool:
    """Phase 1b: pairwise subset / superset constraint reasoning.

    For each pair (A, remA), (B, remB) where A is a subset of B:
      - If remA == remB  -> B \\ A is all safe.
      - If |B \\ A| == remB - remA  -> B \\ A is all mines.

    Operates only on overlapping (subset/superset) constraints and applies the
    two deterministic deductions above directly to *result*.
    Returns True if any cell status changed.
    """
    # Build adjusted constraints: only live unknowns
    adjusted: list[tuple[frozenset[tuple[int, int]], int]] = []
    for unknowns, remaining in constraints:
        live = frozenset(cell for cell in unknowns if result.get(cell) == "unknown")
        mines_found = sum(1 for cell in unknowns if result.get(cell) == "mine")
        adj = remaining - mines_found
        if live and 0 <= adj <= len(live):
            adjusted.append((live, adj))

    # De-duplicate
    seen: set[tuple[frozenset, int]] = set()
    unique: list[tuple[frozenset[tuple[int, int]], int]] = []
    for cells, rem in adjusted:
        key = (cells, rem)
        if key not in seen:
            seen.add(key)
            unique.append(key)
    adjusted = unique

    changed = False

    n = len(adjusted)
    for i in range(n):
        a_cells, a_rem = adjusted[i]
        for j in range(n):
            if i == j:
                continue
            b_cells, b_rem = adjusted[j]

            # Check if A ⊂ B (A is a proper subset of B)
            if len(a_cells) >= len(b_cells):
                continue
            if not a_cells.issubset(b_cells):
                continue

            diff = b_cells - a_cells
            diff_rem = b_rem - a_rem

            if diff_rem < 0 or diff_rem > len(diff):
                continue

            if diff_rem == 0:
                # All cells in diff are safe
                for cell in diff:
                    if result.get(cell) == "unknown":
                        result[cell] = "safe"
                        changed = True
            elif diff_rem == len(diff):
                # All cells in diff are mines
                for cell in diff:
                    if result.get(cell) == "unknown":
                        result[cell] = "mine"
                        changed = True

    return changed


# ============================================================================
# Phase 2: Area decomposition + enumeration
# ============================================================================

def _build_areas(
    comp_cells: list[tuple[int, int]],
    comp_constraints: list[tuple[list[tuple[int, int]], int]],
) -> tuple[list[list[tuple[int, int]]], list[tuple[list[int], int]]]:
    """Group cells that appear in the same set of constraints into areas.

    Returns:
        areas: list of cell groups (each group = one "area variable")
        area_constraints: constraints rewritten in terms of area indices
    """
    # Build signature: for each cell, the set of constraint indices it participates in
    cell_signature: dict[tuple[int, int], frozenset[int]] = {}
    for cell in comp_cells:
        sig_parts = []
        for ci, (cells, _) in enumerate(comp_constraints):
            if cell in cells:
                sig_parts.append(ci)
        cell_signature[cell] = frozenset(sig_parts)

    # Group cells by signature
    sig_to_cells: dict[frozenset[int], list[tuple[int, int]]] = defaultdict(list)
    for cell in comp_cells:
        sig_to_cells[cell_signature[cell]].append(cell)

    areas: list[list[tuple[int, int]]] = []
    cell_to_area: dict[tuple[int, int], int] = {}
    for cells_group in sig_to_cells.values():
        area_idx = len(areas)
        areas.append(sorted(cells_group))
        for cell in cells_group:
            cell_to_area[cell] = area_idx

    # Rewrite constraints in terms of areas
    area_constraints: list[tuple[list[int], int]] = []
    for cells, remaining in comp_constraints:
        area_indices_seen: set[int] = set()
        area_list: list[int] = []
        for cell in cells:
            ai = cell_to_area[cell]
            if ai not in area_indices_seen:
                area_indices_seen.add(ai)
                area_list.append(ai)
        area_constraints.append((area_list, remaining))

    return areas, area_constraints


def _enumerate_with_areas(
    areas: list[list[tuple[int, int]]],
    area_constraints: list[tuple[list[int], int]],
    max_solutions: int = _MAX_SOLUTIONS,
) -> tuple[list[list[int]], int, bool]:
    """Enumerate valid mine assignments using area decomposition.

    Each area variable ranges from 0 to area_size.  Constraints are checked
    incrementally using an indexed lookup.

    Returns:
        (solution_list, total_solutions, truncated):
          solution_list[s][a] = number of mines assigned to area a in solution s
          total_solutions = len(solution_list)
          truncated = True iff the search hit ``max_solutions`` and therefore may
            have omitted additional valid solutions (so the returned set is an
            incomplete subset and must NOT be used for soundness deductions).
    """
    n_areas = len(areas)
    area_sizes = [len(a) for a in areas]

    # Build area -> constraint index mapping for fast lookup
    area_to_constraints: list[list[int]] = [[] for _ in range(n_areas)]
    for ci, (area_indices, _) in enumerate(area_constraints):
        for ai in area_indices:
            if ci not in area_to_constraints[ai]:
                area_to_constraints[ai].append(ci)

    assignment = [0] * n_areas
    solutions: list[list[int]] = []

    def _check_constraints_for_area(depth: int) -> bool:
        """Check only constraints involving the area at `depth`."""
        for ci in area_to_constraints[depth]:
            area_indices, remaining = area_constraints[ci]
            assigned_mines = 0
            max_possible = 0
            for ai in area_indices:
                if ai <= depth:
                    assigned_mines += assignment[ai]
                    max_possible += assignment[ai]
                else:
                    max_possible += area_sizes[ai]
            if assigned_mines > remaining:
                return False
            if max_possible < remaining:
                return False
        return True

    def _backtrack(depth: int):
        if len(solutions) >= max_solutions:
            return

        if depth == n_areas:
            # Verify all constraints exactly satisfied
            for area_indices, remaining in area_constraints:
                total = sum(assignment[ai] for ai in area_indices)
                if total != remaining:
                    return
            solutions.append(assignment[:])
            return

        # Area variable ranges from 0 to area_size
        for val in range(area_sizes[depth] + 1):
            assignment[depth] = val
            if _check_constraints_for_area(depth):
                _backtrack(depth + 1)

    _backtrack(0)
    truncated = len(solutions) >= max_solutions
    return solutions, len(solutions), truncated


# ============================================================================
# Main solver entry points
# ============================================================================

def solve_csp(
    grid: list[list[int]],
    revealed: list[list[bool]],
    flags: list[list[bool]],
    rows: int,
    cols: int,
    mode: str = "strong",
) -> dict[tuple[int, int], str]:
    """CSP solver returning status for all unrevealed cells.

    Returns:
        Dict mapping (r, c) -> "safe" / "mine" / "unknown" for unrevealed cells.
    """
    result, _ = _solve_csp_with_level(grid, revealed, flags, rows, cols, mode)
    return result


def _solve_csp_with_level(
    grid: list[list[int]],
    revealed: list[list[bool]],
    flags: list[list[bool]],
    rows: int,
    cols: int,
    mode: str = "strong",
) -> tuple[dict[tuple[int, int], str], str]:
    """CSP solver that also reports the difficulty level of the board.

    Returns:
        (csp_result, difficulty_level) where difficulty_level is one of
        "basic_propagation", "subset_reasoning", "area_decomposition",
        or "stochastic".
    """
    # Initialize all unrevealed, unflagged cells as unknown
    result: dict[tuple[int, int], str] = {}
    for r in range(rows):
        for c in range(cols):
            if not revealed[r][c] and not flags[r][c]:
                result[(r, c)] = "unknown"

    if not result:
        return result, "stochastic"

    # Phase 1: Iterative constraint propagation (all modes)
    constraints = _build_constraints(grid, revealed, flags, rows, cols)

    changed = True
    while changed:
        changed = _constraint_propagation(constraints, result)

    # Snapshot after basic propagation: are there deductions?
    has_deductions_after_basic = any(
        v in ("safe", "mine") for v in result.values()
    )
    # Track remaining unknowns after basic propagation
    unknowns_after_basic = sum(1 for v in result.values() if v == "unknown")

    if mode == "weak":
        level = "basic_propagation" if has_deductions_after_basic else "stochastic"
        return result, level

    # Phase 1b: Subset / superset reasoning (approximate and strong modes)
    subset_changed = True
    while subset_changed:
        subset_changed = _subset_reasoning(constraints, result)
        if subset_changed:
            prop_changed = True
            while prop_changed:
                prop_changed = _constraint_propagation(constraints, result)

    unknowns_after_subset = sum(1 for v in result.values() if v == "unknown")
    gained_from_subset = unknowns_after_basic - unknowns_after_subset

    if mode == "approximate":
        if gained_from_subset > 0:
            level = "subset_reasoning"
        elif has_deductions_after_basic:
            level = "basic_propagation"
        else:
            level = "stochastic"
        return result, level

    # Phase 2: Coupled constraint analysis via area decomposition (strong mode)
    result = _coupled_constraint_analysis(constraints, result)

    unknowns_after_area = sum(1 for v in result.values() if v == "unknown")
    gained_from_area = unknowns_after_subset - unknowns_after_area

    # Determine difficulty: which phase is STILL needed to make progress
    # from the current state (i.e., if unknowns remain, what's the
    # minimum phase that can resolve at least one of them).
    if unknowns_after_area == 0:
        # Fully solved — report the hardest phase that was actually needed
        if gained_from_area > 0:
            level = "area_decomposition"
        elif gained_from_subset > 0:
            level = "subset_reasoning"
        else:
            level = "basic_propagation"
    elif gained_from_area > 0:
        # Area decomposition made progress but unknowns remain → stochastic
        # for the remaining cells (area decomp was the frontier).
        level = "area_decomposition"
    elif gained_from_subset > 0:
        level = "subset_reasoning"
    elif has_deductions_after_basic:
        level = "basic_propagation"
    else:
        level = "stochastic"

    return result, level


def _coupled_constraint_analysis(
    constraints: list[tuple[list[tuple[int, int]], int]],
    result: dict[tuple[int, int], str],
) -> dict[tuple[int, int], str]:
    """Analyze coupled constraints via backtracking on small groups.

    Uses area decomposition to reduce the search space: cells sharing
    identical constraint signatures are grouped into "areas" and treated
    as a single variable.
    """
    # Adjust constraints for already-deduced cells
    adjusted = []
    for unknowns, remaining in constraints:
        live = [cell for cell in unknowns if result.get(cell) == "unknown"]
        mines_found = sum(1 for cell in unknowns if result.get(cell) == "mine")
        adj_remaining = remaining - mines_found
        if live and 0 <= adj_remaining <= len(live):
            adjusted.append((live, adj_remaining))

    if not adjusted:
        return result

    # Build adjacency graph: which constraints share cells
    cell_to_constraints: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, (cells, _) in enumerate(adjusted):
        for cell in cells:
            cell_to_constraints[cell].append(i)

    # Find connected components of constraints
    visited_constraints: set[int] = set()
    components: list[list[int]] = []

    for i in range(len(adjusted)):
        if i in visited_constraints:
            continue
        component = []
        stack = [i]
        while stack:
            ci = stack.pop()
            if ci in visited_constraints:
                continue
            visited_constraints.add(ci)
            component.append(ci)
            for cell in adjusted[ci][0]:
                for neighbor_ci in cell_to_constraints[cell]:
                    if neighbor_ci not in visited_constraints:
                        stack.append(neighbor_ci)
        components.append(component)

    # For each small component, enumerate solutions with area decomposition
    for component in components:
        comp_cells_set: set[tuple[int, int]] = set()
        comp_constraints = []
        for ci in component:
            cells, remaining = adjusted[ci]
            comp_constraints.append((cells, remaining))
            comp_cells_set.update(cells)

        comp_cells = sorted(comp_cells_set)

        # Skip if component is too large
        if len(comp_cells) > _MAX_COMPONENT_CELLS:
            continue

        # Build areas (cells with identical constraint signatures)
        areas, area_constraints = _build_areas(comp_cells, comp_constraints)

        # Enumerate solutions
        solutions, total_solutions, truncated = _enumerate_with_areas(
            areas, area_constraints, max_solutions=_MAX_SOLUTIONS
        )

        if total_solutions == 0:
            continue

        # If enumeration was truncated, the solution set is an incomplete subset,
        # so always_zero/always_full deductions would be unsound. Skip them.
        if truncated:
            continue

        # Check for cells that are always safe or always mine
        n_areas = len(areas)
        for ai in range(n_areas):
            area_cells = areas[ai]
            # Check if this area is always 0 mines (all safe)
            always_zero = all(sol[ai] == 0 for sol in solutions)
            # Check if this area is always full mines
            always_full = all(sol[ai] == len(area_cells) for sol in solutions)

            if always_zero:
                for cell in area_cells:
                    result[cell] = "safe"
            elif always_full:
                for cell in area_cells:
                    result[cell] = "mine"

    return result


def compute_cell_probabilities(
    grid: list[list[int]],
    revealed: list[list[bool]],
    flags: list[list[bool]],
    rows: int,
    cols: int,
    mode: str = "strong",
    total_mines: int = 0,
) -> dict[tuple[int, int], float]:
    """Compute mine probability for each unrevealed, unflagged cell.

    Args:
        total_mines: Total number of mines on the board (from game metadata).

    Returns:
        Dict mapping (r, c) -> probability of being a mine (0.0 to 1.0).
    """
    state = solve_board(grid, revealed, flags, rows, cols, mode=mode, total_mines=total_mines)
    return state.probabilities


def _compute_exact_probabilities(
    grid: list[list[int]],
    revealed: list[list[bool]],
    flags: list[list[bool]],
    rows: int,
    cols: int,
    csp_result: dict[tuple[int, int], str],
    constraints: list[tuple[list[tuple[int, int]], int]],
    total_mines: int = 0,
) -> dict[tuple[int, int], float]:
    """Compute exact mine probabilities for all unrevealed cells.

    Performs a TRUE GLOBAL EXACT combinatorial computation: each boundary
    connected component's integer mine-count distribution is computed via area
    decomposition, then all components and the unconstrained interior are
    coupled under the global constraint that exactly ``remaining_mines`` mines
    lie among the unknown cells.  Falls back to a per-component approximate
    estimate (plus interior uniform) when any component is too large, its
    enumeration is truncated, it is inconsistent, ``remaining_mines < 0``, or
    the exact normalizer is degenerate.

    Args:
        total_mines: Total number of mines on the board (from game metadata).
    """
    probabilities: dict[tuple[int, int], float] = {}

    flagged_count = sum(
        1 for r in range(rows) for c in range(cols) if flags[r][c]
    )
    deduced_mines = sum(1 for v in csp_result.values() if v == "mine")
    unknown_cells = [cell for cell, status in csp_result.items() if status == "unknown"]
    remaining_mines = total_mines - flagged_count - deduced_mines

    # Set probabilities for already-deduced cells
    for cell, status in csp_result.items():
        if status == "safe":
            probabilities[cell] = 0.0
        elif status == "mine":
            probabilities[cell] = 1.0

    if not unknown_cells:
        return probabilities

    # Build adjusted constraints for unknown cells
    adjusted = []
    for unknowns, remaining in constraints:
        live = [cell for cell in unknowns if csp_result.get(cell) == "unknown"]
        mines_found = sum(1 for cell in unknowns if csp_result.get(cell) == "mine")
        adj_remaining = remaining - mines_found
        if live and 0 <= adj_remaining <= len(live):
            adjusted.append((live, adj_remaining))

    # Find boundary cells (cells that appear in at least one constraint)
    boundary_cells: set[tuple[int, int]] = set()
    for cells, _ in adjusted:
        boundary_cells.update(cells)

    # Interior cells = unknown but not in any constraint
    interior_cells = [c for c in unknown_cells if c not in boundary_cells]

    if not adjusted:
        # No constraints: all unknowns get uniform probability
        if unknown_cells:
            uniform_prob = max(0.0, min(1.0, remaining_mines / len(unknown_cells)))
            for cell in unknown_cells:
                probabilities[cell] = uniform_prob
        return probabilities

    # Build adjacency and find connected components
    cell_to_constraints: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, (cells, _) in enumerate(adjusted):
        for cell in cells:
            cell_to_constraints[cell].append(i)

    visited_constraints: set[int] = set()
    components: list[list[int]] = []
    for i in range(len(adjusted)):
        if i in visited_constraints:
            continue
        component = []
        stack = [i]
        while stack:
            ci = stack.pop()
            if ci in visited_constraints:
                continue
            visited_constraints.add(ci)
            component.append(ci)
            for cell in adjusted[ci][0]:
                for neighbor_ci in cell_to_constraints[cell]:
                    if neighbor_ci not in visited_constraints:
                        stack.append(neighbor_ci)
        components.append(component)

    # Gather per-component data (cells + constraints) for all components.
    component_cells: list[list[tuple[int, int]]] = []
    component_constraints_list: list[list[tuple[list[tuple[int, int]], int]]] = []
    for component in components:
        comp_cells_set: set[tuple[int, int]] = set()
        comp_constraints = []
        for ci in component:
            cells, remaining = adjusted[ci]
            comp_constraints.append((cells, remaining))
            comp_cells_set.update(cells)
        component_cells.append(sorted(comp_cells_set))
        component_constraints_list.append(comp_constraints)

    n_int = len(interior_cells)

    # Attempt the TRUE GLOBAL EXACT computation. Each component is EXACT only if
    # it is small enough, its enumeration is not truncated, and it is consistent.
    # If ANY component is INEXACT (or remaining_mines < 0) we fall back to the
    # approximate per-component path for the whole board.
    exact_ok = remaining_mines >= 0
    comp_exact_data: list[dict] = []  # one entry per component: {N, Mcount_cells}

    if exact_ok:
        for comp_cells, comp_constraints in zip(
            component_cells, component_constraints_list
        ):
            if len(comp_cells) > _MAX_COMPONENT_CELLS:
                exact_ok = False
                break

            areas, area_constraints = _build_areas(comp_cells, comp_constraints)
            solutions, total_solutions, truncated = _enumerate_with_areas(
                areas, area_constraints, max_solutions=_MAX_SOLUTIONS
            )

            if truncated or total_solutions == 0:
                exact_ok = False
                break

            comp_exact_data.append(
                _component_exact_distributions(areas, solutions)
            )

    if exact_ok and comp_exact_data:
        result_probs = _global_exact_combination(
            comp_exact_data,
            interior_cells,
            n_int,
            remaining_mines,
        )
        if result_probs is not None:
            for cell, prob in result_probs.items():
                probabilities[cell] = max(0.0, min(1.0, prob))
            return probabilities
        # Z == 0 (or otherwise degenerate): fall through to approximate path.

    # ---- Approximate fallback path (whole board) ----
    # Per-component local estimate via constraints; interior uniform.
    for comp_cells in component_cells:
        for cell in comp_cells:
            prob = _estimate_local_probability_from_constraints(
                cell, adjusted, csp_result
            )
            if prob is not None:
                probabilities[cell] = max(0.0, min(1.0, prob))
            else:
                # Uniform fallback
                if len(unknown_cells) > 0:
                    probabilities[cell] = max(
                        0.0, min(1.0, remaining_mines / len(unknown_cells))
                    )
                else:
                    probabilities[cell] = 0.0

    # Interior cells: uniform probability based on remaining mines
    boundary_mines_expected = sum(
        probabilities.get(cell, 0.0) for cell in boundary_cells
        if cell in probabilities
    )
    interior_remaining = remaining_mines - boundary_mines_expected
    if interior_cells:
        interior_prob = max(0.0, min(1.0, interior_remaining / len(interior_cells)))
        for cell in interior_cells:
            probabilities[cell] = interior_prob

    return probabilities


def _component_exact_distributions(
    areas: list[list[tuple[int, int]]],
    solutions: list[list[int]],
) -> dict:
    """Build integer mine-count distributions for one EXACT component.

    Each solution assigns a mine-count per area; the number of concrete
    cell-level configurations it represents is
    ``product over areas of comb(area_size, mines_in_area)``.

    Returns a dict with:
      - ``"N"``: dict mapping k (total mines in component) -> integer count of
        concrete configurations with exactly k mines in this component.
      - ``"Mcount_cells"``: dict mapping cell -> (dict k -> integer count of
        concrete configurations with k component-mines in which that cell is a
        mine).  Cells in the same area share the same distribution.
    """
    area_sizes = [len(a) for a in areas]
    n_areas = len(areas)

    n_dist: dict[int, int] = defaultdict(int)
    # Per-area, per-total-k mine-config count weighted appropriately.
    area_mcount: list[dict[int, int]] = [defaultdict(int) for _ in range(n_areas)]

    for sol in solutions:
        config_count = 1
        for ai in range(n_areas):
            config_count *= math.comb(area_sizes[ai], sol[ai])
        k = sum(sol)
        n_dist[k] += config_count

        for ai in range(n_areas):
            m_a = sol[ai]
            s_a = area_sizes[ai]
            if m_a >= 1:
                # config_count where a fixed cell in this area is a mine:
                # comb(s_a - 1, m_a - 1) * product over OTHER areas.
                other = 1
                for aj in range(n_areas):
                    if aj == ai:
                        continue
                    other *= math.comb(area_sizes[aj], sol[aj])
                area_mcount[ai][k] += math.comb(s_a - 1, m_a - 1) * other

    # Copy per-area distributions to each cell in the area.
    mcount_cells: dict[tuple[int, int], dict[int, int]] = {}
    for ai in range(n_areas):
        for cell in areas[ai]:
            mcount_cells[cell] = dict(area_mcount[ai])

    return {"N": dict(n_dist), "Mcount_cells": mcount_cells}


def _convolve_count_dists(dists: list[dict[int, int]]) -> dict[int, int]:
    """Convolve a list of integer count distributions (keyed by mine total).

    Returns a dict mapping total mine count -> number of concrete configurations
    achieving that total across all input distributions.  An empty list yields
    ``{0: 1}`` (the multiplicative identity).
    """
    acc: dict[int, int] = {0: 1}
    for dist in dists:
        new_acc: dict[int, int] = defaultdict(int)
        for t, ct in acc.items():
            for k, ck in dist.items():
                new_acc[t + k] += ct * ck
        acc = dict(new_acc)
    return acc


def _global_exact_combination(
    comp_exact_data: list[dict],
    interior_cells: list[tuple[int, int]],
    n_int: int,
    remaining_mines: int,
) -> Optional[dict[tuple[int, int], float]]:
    """Combine EXACT per-component distributions under the total-mine budget.

    Couples all boundary components and the unconstrained interior via the
    global constraint that exactly ``remaining_mines`` mines lie among the
    unknown (boundary + interior) cells.

    Returns a dict mapping every unknown cell -> mine probability, or ``None``
    if the configuration is degenerate (normalizer ``Z == 0``), signalling the
    caller to fall back to the approximate path.
    """
    n_components = len(comp_exact_data)
    comp_N = [data["N"] for data in comp_exact_data]

    def _interior_choose(k_int: int) -> int:
        if k_int < 0 or k_int > n_int:
            return 0
        return math.comb(n_int, k_int)

    # Boundary mine-count distribution over all components.
    bdist = _convolve_count_dists(comp_N)

    # Normalizer: sum over boundary totals T of bdist[T] * comb(n_int, rem - T).
    Z = 0
    for t, ct in bdist.items():
        Z += ct * _interior_choose(remaining_mines - t)
    if Z == 0:
        return None

    probs: dict[tuple[int, int], float] = {}

    # Per-component cell probabilities.
    for j in range(n_components):
        # Distribution of all OTHER components convolved together.
        rest = _convolve_count_dists(
            [comp_N[i] for i in range(n_components) if i != j]
        )
        # Precompute, for each possible k_j, the weighted interior+rest factor.
        # factor[k_j] = sum over T' of rest[T'] * comb(n_int, rem - k_j - T').
        factor: dict[int, int] = {}
        for cell, mcount in comp_exact_data[j]["Mcount_cells"].items():
            numerator = 0
            for k_j, mc in mcount.items():
                if mc == 0:
                    continue
                if k_j not in factor:
                    f = 0
                    for t_prime, ct in rest.items():
                        f += ct * _interior_choose(remaining_mines - k_j - t_prime)
                    factor[k_j] = f
                numerator += mc * factor[k_j]
            probs[cell] = numerator / Z

    # Interior cells: expected interior mines / n_int.
    if n_int > 0:
        interior_numerator = 0
        for t, ct in bdist.items():
            k_int = remaining_mines - t
            if 0 <= k_int <= n_int:
                interior_numerator += ct * k_int * math.comb(n_int, k_int)
        interior_expected_mines = interior_numerator / Z
        interior_prob = interior_expected_mines / n_int
        for cell in interior_cells:
            probs[cell] = interior_prob

    return probs


def _estimate_local_probability_from_constraints(
    target_cell: tuple[int, int],
    adjusted_constraints: list[tuple[list[tuple[int, int]], int]],
    csp_result: dict[tuple[int, int], str],
) -> Optional[float]:
    """Estimate mine probability using local constraints (fallback for large components)."""
    min_prob = 1.0
    max_prob = 0.0
    has_constraint = False

    for unknowns, remaining in adjusted_constraints:
        if target_cell not in unknowns:
            continue

        has_constraint = True
        live = [cell for cell in unknowns if csp_result.get(cell) == "unknown"]
        mines_found = sum(1 for cell in unknowns if csp_result.get(cell) == "mine")
        adj_remaining = remaining - mines_found

        if target_cell not in live:
            continue

        if len(live) > 0 and adj_remaining >= 0:
            prob = adj_remaining / len(live)
            min_prob = min(min_prob, prob)
            max_prob = max(max_prob, prob)

    if not has_constraint:
        return None

    return (min_prob + max_prob) / 2.0


# ============================================================================
# Top-level solve_board: single-solve cache entry point
# ============================================================================

def solve_board(
    grid: list[list[int]],
    revealed: list[list[bool]],
    flags: list[list[bool]],
    rows: int,
    cols: int,
    mode: str = "strong",
    total_mines: int = 0,
) -> MinesweeperSolverState:
    """Run the complete solver once and return a cached state object.

    This is the primary entry point for feedback and hint providers.
    All downstream queries are answered from the returned state without
    re-running the solver.

    When using this solver to compute an oracle value K(s) for credit
    assignment, callers MUST pass empty ``flags`` (all False) so the solver
    re-derives mines from revealed numbers via CSP rather than trusting
    external/agent flags, which may be wrong.

    Args:
        total_mines: Total number of mines on the board (from game metadata).
    """
    state = MinesweeperSolverState()

    # Step 1: CSP solve (with difficulty level tracking)
    state.csp_result, state.difficulty_level = _solve_csp_with_level(
        grid, revealed, flags, rows, cols, mode=mode,
    )

    # Step 2: Extract safe / mine cells
    state.safe_cells = sorted(
        [cell for cell, status in state.csp_result.items() if status == "safe"]
    )
    state.mine_cells = sorted(
        [cell for cell, status in state.csp_result.items() if status == "mine"]
    )

    # Step 3: Compute probabilities
    if mode == "strong":
        constraints = _build_constraints(grid, revealed, flags, rows, cols)
        state.probabilities = _compute_exact_probabilities(
            grid, revealed, flags, rows, cols, state.csp_result, constraints,
            total_mines=total_mines,
        )
    else:
        # For approximate/weak: set probabilities from CSP result
        flagged_count = sum(
            1 for r in range(rows) for c in range(cols) if flags[r][c]
        )
        deduced_mines = sum(1 for v in state.csp_result.values() if v == "mine")
        unknown_cells = [
            cell for cell, status in state.csp_result.items() if status == "unknown"
        ]
        remaining_mines = total_mines - flagged_count - deduced_mines

        for cell, status in state.csp_result.items():
            if status == "safe":
                state.probabilities[cell] = 0.0
            elif status == "mine":
                state.probabilities[cell] = 1.0
            elif status == "unknown":
                if unknown_cells:
                    state.probabilities[cell] = max(
                        0.0, min(1.0, remaining_mines / len(unknown_cells))
                    )
                else:
                    state.probabilities[cell] = 0.0

    # Step 4: Compute best action
    state.best_action = _compute_best_action(
        grid, revealed, flags, rows, cols, state, total_mines
    )

    return state


# ============================================================================
# Public convenience functions (backward-compatible API)
# ============================================================================

def get_safest_cells(
    grid: list[list[int]],
    revealed: list[list[bool]],
    flags: list[list[bool]],
    rows: int,
    cols: int,
    mode: str = "strong",
) -> list[tuple[int, int]]:
    """Return list of cells that are provably safe (probability = 0.0)."""
    csp_result = solve_csp(grid, revealed, flags, rows, cols, mode=mode)
    return sorted([cell for cell, status in csp_result.items() if status == "safe"])


def get_provably_mine_cells(
    grid: list[list[int]],
    revealed: list[list[bool]],
    flags: list[list[bool]],
    rows: int,
    cols: int,
    mode: str = "strong",
) -> list[tuple[int, int]]:
    """Return list of cells that are provably mines by CSP."""
    csp_result = solve_csp(grid, revealed, flags, rows, cols, mode=mode)
    return sorted([cell for cell, status in csp_result.items() if status == "mine"])


def count_provably_safe_cells(
    grid: list[list[int]],
    revealed: list[list[bool]],
    flags: list[list[bool]],
    rows: int,
    cols: int,
    mode: str = "strong",
) -> int:
    """Count unrevealed cells that are provably safe by CSP."""
    return len(get_safest_cells(grid, revealed, flags, rows, cols, mode=mode))


def get_best_action(
    grid: list[list[int]],
    revealed: list[list[bool]],
    flags: list[list[bool]],
    rows: int,
    cols: int,
    mode: str = "strong",
    total_mines: int = 0,
) -> Optional[tuple[str, int, int]]:
    """Return the optimal next action.

    Strategy:
    1. If any provably safe cell exists -> reveal the most informative one.
    2. If provably mine cells exist and are unflagged -> flag one.
    3. If no safe cells -> reveal the cell with lowest mine probability.
    4. Returns None if game is won or all non-mine cells are revealed.

    Args:
        total_mines: Total number of mines on the board (from game metadata).
    """
    state = solve_board(grid, revealed, flags, rows, cols, mode=mode, total_mines=total_mines)
    return state.best_action


def _compute_best_action(
    grid: list[list[int]],
    revealed: list[list[bool]],
    flags: list[list[bool]],
    rows: int,
    cols: int,
    state: MinesweeperSolverState,
    total_mines: int,
) -> Optional[tuple[str, int, int]]:
    """Compute the best action from a pre-computed solver state."""
    # Check if game is already won / nothing left to do, peek-free:
    # in a winnable game all mines remain hidden and everything else is
    # revealed, so the number of not-revealed cells equals total_mines.
    hidden = sum(
        1 for r in range(rows) for c in range(cols) if not revealed[r][c]
    )
    if hidden == total_mines:
        return None

    if state.safe_cells:
        best_cell = _pick_most_informative(state.safe_cells, grid, revealed, rows, cols)
        return ("reveal", best_cell[0], best_cell[1])

    if state.mine_cells:
        return ("flag", state.mine_cells[0][0], state.mine_cells[0][1])

    # No provably safe cells - find the least risky
    if not state.probabilities:
        return None

    candidates = {
        cell: prob for cell, prob in state.probabilities.items()
        if prob < 1.0 and not revealed[cell[0]][cell[1]] and not flags[cell[0]][cell[1]]
    }

    if not candidates:
        return None

    # Pick the cell with lowest mine probability, with tie-breaking
    # that prefers cells near revealed boundaries (more informative reveals)
    best_cell = min(
        candidates,
        key=lambda c: (
            candidates[c],
            -_informative_score(c, grid, revealed, rows, cols),
        ),
    )
    return ("reveal", best_cell[0], best_cell[1])


def _informative_score(
    cell: tuple[int, int],
    grid: list[list[int]],
    revealed: list[list[bool]],
    rows: int,
    cols: int,
) -> int:
    """Score how informative revealing a cell would be.

    Higher = more informative (prefers cells near numbered borders and
    cells likely to trigger flood-fill).
    """
    r, c = cell
    score = 0
    neighbors = _get_neighbors(r, c, rows, cols)
    unrevealed_neighbors = 0
    revealed_zero_neighbors = 0

    for nr, nc in neighbors:
        if not revealed[nr][nc]:
            unrevealed_neighbors += 1
        elif revealed[nr][nc] and grid[nr][nc] > 0:
            score += 2  # Adjacent to numbered cell: more constraint info
        elif revealed[nr][nc] and grid[nr][nc] == 0:
            revealed_zero_neighbors += 1

    score += unrevealed_neighbors
    # Bonus for likely flood-fill trigger (near many revealed 0s)
    score += revealed_zero_neighbors * 3

    return score


def _pick_most_informative(
    safe_cells: list[tuple[int, int]],
    grid: list[list[int]],
    revealed: list[list[bool]],
    rows: int,
    cols: int,
) -> tuple[int, int]:
    """Among safe cells, pick the one most likely to provide useful info."""
    best_cell = safe_cells[0]
    best_score = -1

    for r, c in safe_cells:
        score = _informative_score((r, c), grid, revealed, rows, cols)
        if score > best_score:
            best_score = score
            best_cell = (r, c)

    return best_cell


# ============================================================================
# Utility functions
# ============================================================================

def is_game_winnable(
    grid: list[list[int]],
    revealed: list[list[bool]],
    rows: int,
    cols: int,
) -> bool:
    """Check if the game can still be won (no mine has been revealed)."""
    for r in range(rows):
        for c in range(cols):
            if revealed[r][c] and grid[r][c] == -1:
                return False
    return True


def get_progress_info(
    grid: list[list[int]],
    revealed: list[list[bool]],
    flags: list[list[bool]],
    rows: int,
    cols: int,
) -> dict:
    """Get comprehensive progress information about the game state."""
    total_safe = 0
    revealed_safe = 0
    total_mines = 0
    flagged = 0
    correct_flags = 0
    incorrect_flags = 0
    unrevealed = 0

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == -1:
                total_mines += 1
            else:
                total_safe += 1

            if revealed[r][c] and grid[r][c] != -1:
                revealed_safe += 1

            if flags[r][c]:
                flagged += 1
                if grid[r][c] == -1:
                    correct_flags += 1
                else:
                    incorrect_flags += 1

            if not revealed[r][c] and not flags[r][c]:
                unrevealed += 1

    return {
        "total_safe": total_safe,
        "revealed_safe": revealed_safe,
        "total_mines": total_mines,
        "flagged": flagged,
        "correct_flags": correct_flags,
        "incorrect_flags": incorrect_flags,
        "unrevealed": unrevealed,
    }
