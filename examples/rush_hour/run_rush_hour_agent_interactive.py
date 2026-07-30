#!/usr/bin/env python3
"""
Interactive Rush Hour Agent Runner

This script runs a RushHourAgent on a randomly generated puzzle using
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
       python3 examples/rush_hour/run_rush_hour_agent_interactive.py
"""

import argparse
import random
from typing import Any

from rllm.agents.rush_hour_agent import RushHourAgent
from rllm.environments.rush_hour.rush_hour_env import RushHourEnv, DIFFICULTY_CONFIGS
from rllm.utils.interactive_runner import InteractiveRunnerBase
from rllm.utils.interactive_utils import print_header, print_separator


class RushHourInteractiveRunner(InteractiveRunnerBase):
    """Interactive runner for Rush Hour environment."""

    name = "Rush Hour Agent Interactive Runner"
    default_max_steps = 30
    supports_tools = False

    def add_env_specific_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--seed",
            type=int,
            default=None,
            help="Random seed for environment (default: random)",
        )
        parser.add_argument(
            "--difficulty",
            type=str,
            default="id",
            choices=list(DIFFICULTY_CONFIGS.keys()),
            help="Difficulty preset (default: id)",
        )
        parser.add_argument(
            "--num-vehicles",
            type=int,
            default=None,
            help="Number of vehicles (overrides difficulty preset)",
        )
        parser.add_argument(
            "--min-moves",
            type=int,
            default=None,
            help="Minimum solution moves (overrides difficulty preset)",
        )
        parser.add_argument(
            "--max-moves",
            type=int,
            default=None,
            help="Maximum solution moves (overrides difficulty preset)",
        )

    def create_environment(self, args: argparse.Namespace) -> tuple[Any, dict]:
        # Set seed if not provided
        if args.seed is None:
            args.seed = random.randint(0, 100000)

        env_info = {"seed": args.seed, "difficulty": args.difficulty}
        if args.num_vehicles is not None:
            env_info["num_vehicles"] = args.num_vehicles
        if args.min_moves is not None:
            env_info["min_moves"] = args.min_moves
        if args.max_moves is not None:
            env_info["max_moves"] = args.max_moves

        env = RushHourEnv.from_dict(env_info)
        return env, {"difficulty": args.difficulty, "seed": args.seed}

    def create_agent(self, args: argparse.Namespace) -> RushHourAgent:
        preset = DIFFICULTY_CONFIGS[args.difficulty]
        max_steps = args.max_steps if args.max_steps != self.default_max_steps else preset["max_steps"]
        return RushHourAgent(
            max_steps=max_steps,
            use_accumulate_thinking=True,
            use_accumulate_history=not args.no_history,
        )

    def get_system_prompt(self, agent: RushHourAgent, env: Any, info: dict) -> str:
        return agent.SYSTEM_PROMPT

    def format_observation_display(self, obs: Any, info: dict) -> str:
        return str(obs)

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
        print(f"Board size: {env.width}x{env.height}")
        print(f"Num vehicles: {env.config.num_vehicles}")
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
        print(f"[ACTION EFFECTIVE] {info.get('action_is_effective', 'N/A')}")
        print(f"[MIN_MOVES] {info.get('min_moves', 'N/A')}")


if __name__ == "__main__":
    RushHourInteractiveRunner().run()
