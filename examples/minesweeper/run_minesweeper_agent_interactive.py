#!/usr/bin/env python3
"""
Interactive Minesweeper Agent Runner

This script runs a MinesweeperAgent on a randomly generated board using
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
       python3 examples/minesweeper/run_minesweeper_agent_interactive.py
"""

import argparse
import random
from typing import Any

from rllm.agents.minesweeper_agent import MinesweeperAgent
from rllm.environments.minesweeper.minesweeper_env import MinesweeperEnv, DIFFICULTY_CONFIGS
from rllm.utils.interactive_runner import InteractiveRunnerBase
from rllm.utils.interactive_utils import print_header, print_separator


class MinesweeperInteractiveRunner(InteractiveRunnerBase):
    """Interactive runner for Minesweeper environment."""

    name = "Minesweeper Agent Interactive Runner"
    default_max_steps = 25
    supports_tools = False

    def add_env_specific_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--seed",
            type=int,
            default=None,
            help="Random seed for environment (default: random)",
        )
        parser.add_argument(
            "--rows",
            type=int,
            default=None,
            help="Number of rows (overrides difficulty preset)",
        )
        parser.add_argument(
            "--cols",
            type=int,
            default=None,
            help="Number of columns (overrides difficulty preset)",
        )
        parser.add_argument(
            "--num-mines",
            type=int,
            default=None,
            help="Number of mines (overrides difficulty preset)",
        )
        parser.add_argument(
            "--difficulty",
            type=str,
            default="id",
            choices=list(DIFFICULTY_CONFIGS.keys()),
            help="Difficulty preset (default: id)",
        )

    def create_environment(self, args: argparse.Namespace) -> tuple[Any, dict]:
        # Set seed if not provided
        if args.seed is None:
            args.seed = random.randint(0, 100000)

        # Use difficulty preset, allow overrides
        preset = DIFFICULTY_CONFIGS[args.difficulty]
        rows = args.rows if args.rows is not None else preset["rows"]
        cols = args.cols if args.cols is not None else preset["cols"]
        num_mines = args.num_mines if args.num_mines is not None else preset["num_mines"]
        max_steps = args.max_steps if args.max_steps != self.default_max_steps else preset["max_steps"]

        env = MinesweeperEnv(
            rows=rows,
            cols=cols,
            num_mines=num_mines,
            max_steps=max_steps,
            seed=args.seed,
        )
        return env, {
            "rows": rows,
            "cols": cols,
            "num_mines": num_mines,
            "difficulty": args.difficulty,
        }

    def create_agent(self, args: argparse.Namespace) -> MinesweeperAgent:
        preset = DIFFICULTY_CONFIGS[args.difficulty]
        max_steps = args.max_steps if args.max_steps != self.default_max_steps else preset["max_steps"]
        return MinesweeperAgent(
            max_steps=max_steps,
            use_accumulate_thinking=True,
            use_accumulate_history=not args.no_history,
        )

    def get_system_prompt(self, agent: MinesweeperAgent, env: Any, info: dict) -> str:
        return agent.SYSTEM_PROMPT

    def format_observation_display(self, obs: Any, info: dict) -> str:
        # The env returns feedback text as obs and the board in info["suffix"].
        # On reset(), obs already contains the board (same as suffix), so we
        # must avoid showing it twice.  On step(), obs is feedback text and
        # suffix is the board — both should be displayed.
        suffix = info.get("suffix", "") if isinstance(info, dict) else ""
        obs_str = str(obs)
        if suffix and suffix.strip() in obs_str:
            # Board already present in obs (e.g. after reset) — show obs only.
            return obs_str
        if suffix:
            return f"{obs_str}\n\n{suffix}"
        return obs_str

    def print_episode_start(
        self,
        env: Any,
        agent: Any,
        info: dict,
        args: argparse.Namespace,
        model_id: str,
    ) -> None:
        print_header("EPISODE START")
        print(f"Difficulty: {args.difficulty}")
        print(f"Board size: {env.rows}x{env.cols}")
        print(f"Mines: {env.num_mines}")
        print(f"Max steps: {env.max_steps}")
        print(f"Seed: {args.seed}")
        print(f"Model: {model_id}")
        print()

        # Show initial board
        print_header("INITIAL BOARD")
        print(env.render())
        print()

    def process_model_response(
        self,
        agent: Any,
        response: str,
    ) -> tuple[Any, str]:
        action = agent.update_from_model(response)
        action_str = action.action
        return action_str, action_str

    def print_step_result(
        self,
        step: int,
        obs: Any,
        reward: float,
        done: bool,
        info: dict,
        total_reward: float,
    ) -> None:
        print(f"[REWARD] {reward:.4f}")
        print(f"[CUMULATIVE REWARD] {total_reward:.4f}")
        print(f"[DONE] {done}")
        print(f"[SUCCESS] {info.get('success', False)}")
        if info.get("mine_hit"):
            print("[MINE HIT!]")
        if info.get("format_error"):
            print("[FORMAT ERROR]")


if __name__ == "__main__":
    MinesweeperInteractiveRunner().run()
