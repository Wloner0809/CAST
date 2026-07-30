#!/usr/bin/env python3
"""
Interactive WebShop Agent Runner

This script runs a WebShopAgent on the web shopping simulation environment using
a locally served LLM model (via sglang server at localhost:8000).

It shows detailed interaction including:
- Agent thinking process
- Actions taken (search/click)
- Page content changes
- Final reward

Usage:
    1. Start the sglang server manually:
       python3 -m sglang.launch_server \
           --model-path <model-name-or-path> --host 0.0.0.0 --port 8000
    2. Run this script:
       python examples/webshop/run_webshop_agent_interactive.py
"""

import argparse
from typing import Any

from rllm.agents.webshop_agent import WebShopAgent
from rllm.environments.webshop.webshop_env import WebShopEnv
from rllm.utils.interactive_runner import InteractiveRunnerBase
from rllm.utils.interactive_utils import print_header


class WebShopInteractiveRunner(InteractiveRunnerBase):
    """Interactive runner for WebShop environment."""

    name = "WebShop Agent Interactive Runner"
    default_max_steps = 30
    supports_tools = False

    def add_env_specific_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--session-id",
            type=int,
            default=None,
            help="Specific session/goal ID to use (default: random)",
        )
        parser.add_argument(
            "--observation-mode",
            type=str,
            default="text",
            choices=["text", "html", "text_rich"],
            help="Observation format",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Random seed",
        )

    def create_environment(self, args: argparse.Namespace) -> tuple[Any, dict]:
        print("Initializing WebShop environment...")
        env = WebShopEnv(
            observation_mode=args.observation_mode,
            max_steps=args.max_steps,
            session_id=args.session_id,
            seed=args.seed,
        )
        print("Environment initialized successfully!")
        return env, {}

    def create_agent(self, args: argparse.Namespace) -> WebShopAgent:
        return WebShopAgent(
            max_steps=args.max_steps,
            use_available_actions=True,
            use_accumulate_history=not args.no_history,
        )

    def get_system_prompt(self, agent: WebShopAgent, env: Any, info: dict) -> str:
        return agent.SYSTEM_PROMPT

    def format_observation_display(self, obs: Any, info: dict) -> str:
        obs_str = str(obs)
        return obs_str[:800] + "..." if len(obs_str) > 800 else obs_str

    def print_episode_start(
        self,
        env: Any,
        agent: Any,
        info: dict,
        args: argparse.Namespace,
        model_id: str,
    ) -> None:
        print_header("EPISODE START")
        print(f"Session ID: {args.session_id or 'random'}")
        print(f"Observation mode: {args.observation_mode}")
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
        print(f"[TASK SCORE] {info.get('task_score', 0)}")
        print(f"[CUMULATIVE REWARD] {total_reward}")
        print(f"[DONE] {done}")
        print(f"[WON] {info.get('won', False)}")

    def cleanup(self, env: Any) -> None:
        env.close()


if __name__ == "__main__":
    WebShopInteractiveRunner().run()
