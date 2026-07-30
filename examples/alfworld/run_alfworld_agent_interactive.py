#!/usr/bin/env python3
"""
Interactive ALFWorld Agent Runner

This script runs an ALFWorldAgent on a household task environment using
a locally served LLM model (via sglang server at localhost:8000).

It shows detailed interaction including:
- Agent thinking process
- Actions taken
- Environment state changes
- Final reward

Usage:
    1. Start the sglang server manually:
       python3 -m sglang.launch_server \
           --model-path <model-name-or-path> --host 0.0.0.0 --port 8000
    2. Run this script:
       python examples/alfworld/run_alfworld_agent_interactive.py --task-index 0

    To list available tasks:
       python examples/alfworld/run_alfworld_agent_interactive.py --list-tasks
"""

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from rllm.agents.alfworld_agent import ALFWorldAgent
from rllm.environments.alfworld.alfworld_env import ALFWorldEnv
from rllm.utils.interactive_runner import InteractiveRunnerBase
from rllm.utils.interactive_utils import print_header


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "datasets" / "alfworld"


def get_split_filename(train_eval: str) -> str:
    """Map train_eval argument to parquet filename."""
    mapping = {
        "train": "train.parquet",
        "eval_in_distribution": "eval_id.parquet",
        "eval_out_of_distribution": "eval_ood.parquet",
    }
    return mapping.get(train_eval, "eval_id.parquet")


def load_alfworld_tasks(train_eval: str, task_types: list[int] | None = None) -> pd.DataFrame:
    """Load ALFWorld tasks from the dataset.

    Args:
        train_eval: Data split to use.
        task_types: Optional list of task type IDs to filter by.

    Returns:
        DataFrame with task information.
    """
    filename = get_split_filename(train_eval)
    data_path = DEFAULT_DATA_DIR / filename

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_parquet(data_path)

    if task_types and "task_type" in df.columns:
        from rllm.environments.alfworld.alfworld_env import TASK_TYPES
        type_names = [TASK_TYPES.get(t) for t in task_types if t in TASK_TYPES]
        df = df[df["task_type"].isin(type_names)]

    return df


class ALFWorldInteractiveRunner(InteractiveRunnerBase):
    """Interactive runner for ALFWorld environment."""

    name = "ALFWorld Agent Interactive Runner"
    default_max_steps = 50
    supports_tools = False

    def add_env_specific_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--config-path",
            type=str,
            default="rllm/environments/alfworld/alfworld_pkg/configs/config_tw.yaml",
            help="Path to ALFWorld config YAML file",
        )
        parser.add_argument(
            "--train-eval",
            type=str,
            default="eval_in_distribution",
            choices=["train", "eval_in_distribution", "eval_out_of_distribution"],
            help="Data split to use",
        )
        parser.add_argument(
            "--task-types",
            type=int,
            nargs="+",
            default=None,
            help="Task type IDs to include (1-6)",
        )
        parser.add_argument(
            "--task-index",
            type=int,
            default=0,
            help="Index of the task to run (default: 0)",
        )
        parser.add_argument(
            "--list-tasks",
            action="store_true",
            help="List available tasks and exit",
        )

    def create_environment(self, args: argparse.Namespace) -> tuple[Any, dict]:
        # Load tasks from dataset
        tasks_df = load_alfworld_tasks(args.train_eval, args.task_types)

        if len(tasks_df) == 0:
            raise ValueError(
                f"No tasks found for split '{args.train_eval}' "
                f"with task_types={args.task_types}"
            )

        if args.task_index < 0 or args.task_index >= len(tasks_df):
            raise ValueError(
                f"task-index {args.task_index} out of range (0-{len(tasks_df) - 1})"
            )

        task = tasks_df.iloc[args.task_index]
        game_file = task["game_file"]

        print(f"Task index: {args.task_index}")
        print(f"Task ID: {task.get('task_id', 'N/A')}")
        print(f"Task type: {task.get('task_type', 'N/A')}")
        print(f"Game file: {game_file}")
        print()

        env = ALFWorldEnv(
            config_path=args.config_path,
            max_steps=args.max_steps,
            train_eval=args.train_eval,
            task_types=args.task_types,
            game_file=game_file,
        )
        return env, {"task": task}

    def create_agent(self, args: argparse.Namespace) -> ALFWorldAgent:
        return ALFWorldAgent(
            max_steps=args.max_steps,
            use_admissible_commands=True,
            use_accumulate_history=not args.no_history,
        )

    def get_system_prompt(self, agent: ALFWorldAgent, env: Any, info: dict) -> str:
        return agent.SYSTEM_PROMPT

    def format_observation_display(self, obs: Any, info: dict) -> str:
        obs_str = str(obs)
        return obs_str[:500] + "..." if len(obs_str) > 500 else obs_str

    def print_episode_start(
        self,
        env: Any,
        agent: Any,
        info: dict,
        args: argparse.Namespace,
        model_id: str,
    ) -> None:
        print_header("EPISODE START")
        task = info.get("task")
        if task is not None:
            print(f"Task index: {args.task_index}")
            print(f"Task ID: {task.get('task_id', 'N/A')}")
            print(f"Task type: {task.get('task_type', 'N/A')}")
        print(f"Data split: {args.train_eval}")
        print(f"Max steps: {args.max_steps}")
        print(f"Model: {model_id}")
        print()

    def print_step_result(
        self,
        step: int,
        obs: Any,
        reward: float,
        done: bool,
        info: dict,
        total_reward: float,
    ) -> None:
        print(f"[REWARD] {reward}")
        print(f"[CUMULATIVE REWARD] {total_reward}")
        print(f"[DONE] {done}")
        print(f"[WON] {info.get('won', False)}")

    def cleanup(self, env: Any) -> None:
        env.close()
        ALFWorldEnv.shutdown_pool()

    def run(self) -> None:
        """Main entry point with --list-tasks support."""
        parser = self.create_argument_parser()
        args = parser.parse_args()

        # Handle --list-tasks
        if args.list_tasks:
            print_header(f"AVAILABLE TASKS ({args.train_eval.upper()})")
            try:
                tasks_df = load_alfworld_tasks(args.train_eval, args.task_types)
                print(f"Total tasks: {len(tasks_df)}")
                print()
                for i, (_, task) in enumerate(tasks_df.head(20).iterrows()):
                    task_type = task.get("task_type", "N/A")
                    task_id = task.get("task_id", "N/A")
                    print(f"  {i}. [{task_type}] {task_id}")
                if len(tasks_df) > 20:
                    print(f"  ... and {len(tasks_df) - 20} more tasks")
            except FileNotFoundError as e:
                print(f"ERROR: {e}")
            return

        # Call parent run method
        super().run()


if __name__ == "__main__":
    ALFWorldInteractiveRunner().run()
