#!/usr/bin/env python3
"""
Interactive Sokoban Agent Runner

This script runs a SokobanAgent on a randomly generated environment using
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
       python examples/sokoban/run_sokoban_agent_interactive.py
"""

import argparse
import random
from typing import Any

from rllm.agents.sokoban_agent import SokobanAgent
from rllm.environments.sokoban.sokoban import SokobanEnv
from rllm.utils.interactive_runner import InteractiveRunnerBase
from rllm.utils.interactive_utils import print_header, print_separator


class SokobanInteractiveRunner(InteractiveRunnerBase):
    """Interactive runner for Sokoban environment."""

    name = "Sokoban Agent Interactive Runner"
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
            "--dim-room",
            type=int,
            nargs=2,
            default=[6, 6],
            help="Room dimensions (rows cols)",
        )
        parser.add_argument(
            "--num-boxes",
            type=int,
            default=2,
            help="Number of boxes",
        )
        parser.add_argument(
            "--render-mode",
            type=str,
            default="grid_coord",
            choices=["grid", "coord", "grid_coord"],
            help="Render mode",
        )
        parser.add_argument(
            "--multistep",
            action="store_true",
            help="Enable multi-step prompt (allows agent to output multiple actions)",
        )

    def create_environment(self, args: argparse.Namespace) -> tuple[Any, dict]:
        # Set seed if not provided
        if args.seed is None:
            args.seed = random.randint(0, 100000)

        env = SokobanEnv(
            seed=args.seed,
            dim_room=tuple(args.dim_room),
            num_boxes=args.num_boxes,
            max_steps=args.max_steps,
        )
        return env, {"render_mode": args.render_mode, "multistep": args.multistep}

    def create_agent(self, args: argparse.Namespace) -> SokobanAgent:
        return SokobanAgent(
            max_steps=args.max_steps,
            use_accumulate_thinking=True,
            use_accumulate_history=False,  # We manage history ourselves
            use_multistep_prompt=args.multistep,
        )

    def get_system_prompt(self, agent: SokobanAgent, env: Any, info: dict) -> str:
        multistep = info.get("multistep", False)
        return agent.MULTI_STEP_SYSTEM_PROMPT if multistep else agent.SYSTEM_PROMPT

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
        print(f"Environment seed: {args.seed}")
        print(f"Room dimensions: {args.dim_room}")
        print(f"Number of boxes: {args.num_boxes}")
        print(f"Max steps: {args.max_steps}")
        print(f"Model: {model_id}")
        print(f"Multi-step mode: {args.multistep}")
        print()

        # Show initial state
        render_mode = info.get("render_mode", "grid_coord")
        print_header("INITIAL STATE")
        print(env.render(render_mode))
        print()

    def process_model_response(
        self,
        agent: Any,
        response: str,
    ) -> tuple[Any, str]:
        action = agent.update_from_model(response)
        action_str = action.action
        # Handle multi-step actions display
        if "||" in action_str:
            return action_str, f"Multi-step: {action_str}"
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
        print(f"[REWARD] {reward}")
        print(f"[CUMULATIVE REWARD] {total_reward}")
        print(f"[DONE] {done}")
        print(f"[ACTION EFFECTIVE] {info.get('action_is_effective', 'N/A')}")
        if "actions_executed" in info:
            print(f"[ACTIONS EXECUTED] {info.get('actions_executed', 'N/A')}")
            print(f"[EFFECTIVE ACTIONS] {info.get('effective_actions', 'N/A')}")


if __name__ == "__main__":
    SokobanInteractiveRunner().run()
