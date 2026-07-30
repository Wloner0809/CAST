#!/usr/bin/env python3
"""Generate a text file with Sokoban board visualizations.

Modes:
- initial: save initial boards right after env reset
- stalemate: run short random rollouts and save deadlock states
"""

import argparse
import random
import sys
from pathlib import Path

# Make project root importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rllm.data.dataset import DatasetRegistry
from rllm.environments.sokoban.sokoban import SokobanEnv
from rllm.environments.sokoban.solver import classify_deadlock


def generate_initial_visualizations(
    split: str,
    out_path: Path,
    max_samples: int,
) -> int:
    if not DatasetRegistry.dataset_exists("sokoban", split):
        raise RuntimeError(
            f"sokoban/{split} dataset not found. "
            "Run examples/sokoban/prepare_sokoban_data.py first."
        )

    tasks = DatasetRegistry.load_dataset("sokoban", split).get_data()
    initial_states: list[dict] = []

    for task in tasks[: min(len(tasks), max_samples)]:
        env = SokobanEnv.from_dict(task)
        env.reset()
        initial_states.append(
            {
                "seed": task.get("seed"),
                "difficulty": task.get("difficulty"),
                "grid": env.render("grid"),
            }
        )
        env.close()

    if not initial_states:
        raise RuntimeError("No initial states found from dataset.")

    lines = [
        "Sokoban Initial Board Visualizations",
        "Generated from examples/sokoban/prepare_sokoban_data.py outputs",
        f"Dataset split: {split}",
        f"Total initial samples: {len(initial_states)}",
        "Legend: # wall, _ empty, O target, X box, * box on target, P player, S player on target",
        "",
    ]

    for i, item in enumerate(initial_states, 1):
        lines.append(
            f"[{i}] seed={item['seed']}, difficulty={item['difficulty']}, step=0 (initial)"
        )
        lines.append(item["grid"])
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return len(initial_states)


def generate_stalemate_visualizations(
    split: str,
    out_path: Path,
    max_samples: int,
    max_task_scan: int,
    max_steps: int,
    rng_seed: int,
) -> int:
    if not DatasetRegistry.dataset_exists("sokoban", split):
        raise RuntimeError(
            f"sokoban/{split} dataset not found. "
            "Run examples/sokoban/prepare_sokoban_data.py first."
        )

    rng = random.Random(rng_seed)
    tasks = DatasetRegistry.load_dataset("sokoban", split).get_data()
    stalemates: list[dict] = []

    for task in tasks[: min(len(tasks), max_task_scan)]:
        if len(stalemates) >= max_samples:
            break

        env = SokobanEnv.from_dict(task)
        env.reset()

        deadlock = classify_deadlock(env.room_state, env.room_fixed)
        if deadlock is not None:
            stalemates.append(
                {
                    "seed": task.get("seed"),
                    "difficulty": task.get("difficulty"),
                    "step": 0,
                    "deadlock_type": deadlock,
                    "grid": env.render("grid"),
                }
            )
            env.close()
            continue

        for step in range(1, max_steps + 1):
            action = rng.choice([1, 2, 3, 4])
            _, _, done, _ = env.step(action)
            deadlock = classify_deadlock(env.room_state, env.room_fixed)
            if deadlock is not None:
                stalemates.append(
                    {
                        "seed": task.get("seed"),
                        "difficulty": task.get("difficulty"),
                        "step": step,
                        "deadlock_type": deadlock,
                        "grid": env.render("grid"),
                    }
                )
                break
            if done:
                break

        env.close()

    if not stalemates:
        raise RuntimeError(
            "No stalemate states found. Increase --max-task-scan or --max-steps."
        )

    lines = [
        "Sokoban Stalemate (Deadlock) Visualizations",
        "Generated from examples/sokoban/prepare_sokoban_data.py outputs",
        f"Dataset split: {split}",
        f"Total stalemate samples: {len(stalemates)}",
        "Legend: # wall, _ empty, O target, X box, * box on target, P player, S player on target",
        "",
    ]

    for i, item in enumerate(stalemates, 1):
        lines.append(
            f"[{i}] seed={item['seed']}, difficulty={item['difficulty']}, "
            f"step={item['step']}, deadlock={item['deadlock_type']}"
        )
        lines.append(item["grid"])
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return len(stalemates)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Sokoban stalemate visualization .txt file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="initial",
        choices=["initial", "stalemate"],
        help="Visualization mode",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Dataset split to scan",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="examples/sokoban/sokoban_initial_chart_visualization.txt",
        help="Output .txt path",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=20,
        help="Number of stalemate visualizations to save",
    )
    parser.add_argument(
        "--max-task-scan",
        type=int,
        default=120,
        help="Maximum number of dataset tasks to scan",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Max random rollout steps per task",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=20260228,
        help="RNG seed for reproducible random actions",
    )
    args = parser.parse_args()

    if args.mode == "initial":
        count = generate_initial_visualizations(
            split=args.split,
            out_path=Path(args.output),
            max_samples=args.max_samples,
        )
        print(f"Wrote {count} initial visualizations to: {args.output}")
    else:
        count = generate_stalemate_visualizations(
            split=args.split,
            out_path=Path(args.output),
            max_samples=args.max_samples,
            max_task_scan=args.max_task_scan,
            max_steps=args.max_steps,
            rng_seed=args.rng_seed,
        )
        print(f"Wrote {count} stalemate visualizations to: {args.output}")


if __name__ == "__main__":
    main()
