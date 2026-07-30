from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Optional

# Allow running as a standalone script (``python3 tests/solver/solver_selfplay.py``):
# add the repo root to sys.path so ``rllm`` is importable regardless of cwd.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rllm.environments.minesweeper.minesweeper_env import MinesweeperEnv
from rllm.environments.minesweeper.solver import solve_board
# Production oracle primitives (pure simulation); re-exported for the tests.
from rllm.environments.minesweeper.oracle import INF, solver_value_K  # noqa: F401


@dataclass
class SelfPlayStep:
    """One recorded solver move and the observation it acted on."""

    step: int
    board_before: str          # the board observation the solver saw
    action: Optional[str]      # "reveal R C" / "flag R C" / None
    kind: Optional[str]        # "reveal" / "flag" / None
    row: Optional[int]
    col: Optional[int]
    cell_status: str           # solver CSP verdict for the cell: safe/mine/unknown
    mine_probability: Optional[float]
    classification: str        # safe-reveal / mine-flag / guess-reveal / none
    n_provably_safe: int       # provably-safe cells available BEFORE the move
    n_provably_mine: int       # provably-mine cells available BEFORE the move
    difficulty: str            # solver difficulty level of the pre-move board
    action_is_valid: Optional[bool]
    mine_hit: bool


@dataclass
class SelfPlayTrajectory:
    """A full solver self-play game."""

    rows: int
    cols: int
    mine_positions: list[list[int]]
    first_click: list[int]
    total_mines: int
    mode: str
    outcome: str               # win / mine_hit / no_action / max_steps
    success: bool
    k_reveal_steps: int        # number of REVEAL actions taken (= K for K(s))
    n_flag_steps: int
    board_final: str
    steps: list[SelfPlayStep] = field(default_factory=list)


def play_solver_game(
    rows: int,
    cols: int,
    mine_positions: list[tuple[int, int]] | list[list[int]],
    first_click: tuple[int, int] | list[int],
    mode: str = "strong",
    max_steps: Optional[int] = None,
) -> SelfPlayTrajectory:
    """Run the solver to completion on a fixed board and record the trajectory.

    Args:
        rows, cols: Board dimensions.
        mine_positions: Mine coordinates (the simulator's hidden truth).
        first_click: Auto-revealed starting cell (must be a non-mine cell).
        mode: Solver mode ("strong" / "approximate" / "weak").
        max_steps: Safety cap on actions (default ``rows*cols*4``).

    Returns:
        A :class:`SelfPlayTrajectory`.
    """
    mine_positions = [list(p) for p in mine_positions]
    first_click = list(first_click)
    if max_steps is None:
        max_steps = rows * cols * 4

    env = MinesweeperEnv(
        rows=rows,
        cols=cols,
        num_mines=len(mine_positions),
        max_steps=max_steps,
        mine_positions=mine_positions,
        first_click=first_click,
    )
    env.reset()
    total_mines = len(mine_positions)

    steps: list[SelfPlayStep] = []
    outcome = "max_steps"
    k_reveal = 0
    n_flag = 0

    for t in range(1, max_steps + 1):
        if env.success():
            outcome = "win"
            break

        # The solver only ever sees the flags it placed itself (never external
        # flags); proven mines are excluded from the CSP next solve so it never
        # re-flags them (forward progress on the env's toggle-style flag).
        flags_for_solver = [row[:] for row in env.flags]
        state = solve_board(
            env.grid, env.revealed, flags_for_solver, rows, cols,
            mode=mode, total_mines=total_mines,
        )
        action = state.best_action
        board_before = env.render()
        n_safe = len(state.safe_cells)
        n_mine = len(state.mine_cells)

        if action is None:
            steps.append(SelfPlayStep(
                step=t, board_before=board_before, action=None, kind=None,
                row=None, col=None, cell_status="none", mine_probability=None,
                classification="none", n_provably_safe=n_safe,
                n_provably_mine=n_mine, difficulty=state.difficulty_level,
                action_is_valid=None, mine_hit=False,
            ))
            outcome = "no_action"
            break

        kind, r, c = action
        cell_status = state.csp_result.get((r, c), "unknown")
        prob = state.probabilities.get((r, c))
        if kind == "flag":
            classification = "mine-flag"
        elif cell_status == "safe":
            classification = "safe-reveal"
        else:
            classification = "guess-reveal"

        _, _reward, done, info = env.step(f"{kind} {r} {c}")
        mine_hit = bool(info.get("mine_hit", False))

        steps.append(SelfPlayStep(
            step=t, board_before=board_before, action=f"{kind} {r} {c}",
            kind=kind, row=r, col=c, cell_status=cell_status,
            mine_probability=prob, classification=classification,
            n_provably_safe=n_safe, n_provably_mine=n_mine,
            difficulty=state.difficulty_level,
            action_is_valid=info.get("action_is_valid"), mine_hit=mine_hit,
        ))

        if kind == "reveal" and info.get("action_is_valid"):
            k_reveal += 1
        elif kind == "flag" and info.get("action_is_valid"):
            n_flag += 1

        if mine_hit:
            outcome = "mine_hit"
            break
        if done:
            outcome = "win" if env.success() else "max_steps"
            break

    return SelfPlayTrajectory(
        rows=rows, cols=cols, mine_positions=mine_positions,
        first_click=first_click, total_mines=total_mines, mode=mode,
        outcome=outcome,
        success=env.success(), k_reveal_steps=k_reveal, n_flag_steps=n_flag,
        board_final=env.render(), steps=steps,
    )


# ``solver_value_K`` and ``INF`` are the production oracle primitives, defined in
# ``rllm.environments.minesweeper.oracle`` (a pure simulation -- no env churn, no
# circular import). They are imported at the top of this module and re-exported,
# so the existing self-play tests exercise the production code and cross-check it
# against the real-env ``play_solver_game`` rollout above
# (see TestSolverValueK.test_K_matches_selfplay_outcome_across_demos).


def render_trajectory_text(traj: SelfPlayTrajectory) -> str:
    """Render a trajectory as a human-readable report."""
    lines: list[str] = []
    lines.append("=" * 64)
    lines.append("MINESWEEPER SOLVER SELF-PLAY TRAJECTORY")
    lines.append("=" * 64)
    lines.append(
        f"board {traj.rows}x{traj.cols}  mines={traj.total_mines}  "
        f"first_click={traj.first_click}  mode={traj.mode}"
    )
    lines.append(f"mine_positions={traj.mine_positions}")
    lines.append(
        f"OUTCOME={traj.outcome}  success={traj.success}  "
        f"K(reveal_steps)={traj.k_reveal_steps}  flags={traj.n_flag_steps}"
    )
    lines.append("")
    for s in traj.steps:
        lines.append("-" * 64)
        prob = f"{s.mine_probability:.4f}" if s.mine_probability is not None else "n/a"
        lines.append(
            f"step {s.step}: action={s.action}  [{s.classification}]  "
            f"cell_status={s.cell_status}  p(mine)={prob}"
        )
        lines.append(
            f"  before: provably_safe={s.n_provably_safe}  "
            f"provably_mine={s.n_provably_mine}  difficulty={s.difficulty}  "
            f"valid={s.action_is_valid}  mine_hit={s.mine_hit}"
        )
        lines.append("  observation:")
        for bl in s.board_before.splitlines():
            lines.append(f"    {bl}")
    lines.append("-" * 64)
    lines.append("FINAL BOARD:")
    for bl in traj.board_final.splitlines():
        lines.append(f"  {bl}")
    lines.append("=" * 64)
    return "\n".join(lines)


def save_trajectory(traj: SelfPlayTrajectory, out_dir: str, name: str) -> tuple[str, str]:
    """Save a trajectory as both ``<name>.txt`` (readable) and ``<name>.json``.

    Returns the (txt_path, json_path).
    """
    os.makedirs(out_dir, exist_ok=True)
    txt_path = os.path.join(out_dir, f"{name}.txt")
    json_path = os.path.join(out_dir, f"{name}.json")
    with open(txt_path, "w") as f:
        f.write(render_trajectory_text(traj))
    with open(json_path, "w") as f:
        json.dump(asdict(traj), f, indent=2)
    return txt_path, json_path


# A few deterministic demo boards chosen to exercise distinct solver behaviours.
DEMO_BOARDS = [
    # Pure-deduction win: every move is a provably-safe reveal, no guessing.
    {"name": "5x5_pure_deduction_win", "rows": 5, "cols": 5,
     "mine_positions": [[1, 1], [3, 3]], "first_click": [0, 4]},
    # Mixed win: safe-reveals + provably-mine flags + a couple of forced guesses
    # (shows all three action classifications and a survived guess).
    {"name": "6x6_logic_flag_and_guess_win", "rows": 6, "cols": 6,
     "mine_positions": [[1, 1], [2, 2], [3, 3], [4, 4]], "first_click": [0, 5]},
    # Forced-guess death: the solver exhausts deductions, must guess, and hits a
    # mine (demonstrates the inherent "guess luck" of Minesweeper).
    {"name": "6x6_forced_guess_death", "rows": 6, "cols": 6,
     "mine_positions": [[0, 0], [1, 3], [2, 5], [4, 1], [5, 4]], "first_click": [5, 0]},
]


def main(out_dir: Optional[str] = None) -> None:
    """Play the demo boards and save their trajectories for inspection."""
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "selfplay_out")
    for board in DEMO_BOARDS:
        traj = play_solver_game(
            rows=board["rows"], cols=board["cols"],
            mine_positions=board["mine_positions"],
            first_click=board["first_click"], mode="strong",
        )
        txt_path, json_path = save_trajectory(traj, out_dir, board["name"])
        print(
            f"[{board['name']}] outcome={traj.outcome} success={traj.success} "
            f"K={traj.k_reveal_steps} -> {txt_path}"
        )


if __name__ == "__main__":
    main()
