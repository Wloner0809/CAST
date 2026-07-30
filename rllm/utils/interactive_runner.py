"""Base class for interactive agent runners.

This module provides a unified base class that handles the common orchestration
logic for running agents interactively with LLM servers.
"""

import argparse
from abc import ABC, abstractmethod
from typing import Any

from openai import OpenAI

from rllm.utils.interactive_utils import (
    clean_special_tokens,
    connect_to_llm_server,
    get_model_response,
    get_model_response_with_tools,
    print_header,
    print_separator,
)


class InteractiveRunnerBase(ABC):
    """Base class for interactive agent runners with unified orchestration.

    This class provides a template for running agents interactively with LLM servers.
    Subclasses must implement the abstract methods to customize behavior for specific
    environments and agents.

    Attributes:
        name: Display name for the runner.
        default_server_url: Default URL for the LLM server.
        default_max_steps: Default maximum steps per episode.
        supports_tools: Whether this environment uses tool calling.
    """

    # Configuration - override in subclasses
    name: str = "Interactive Runner"
    default_server_url: str = "http://localhost:8000/v1"
    default_max_steps: int = 30
    supports_tools: bool = False

    # =========== Abstract Methods (MUST implement) ===========

    @abstractmethod
    def create_environment(self, args: argparse.Namespace) -> tuple[Any, dict]:
        """Create and return the environment.

        Args:
            args: Parsed command line arguments.

        Returns:
            A tuple of (env, extra_info) where extra_info contains any
            additional information needed for the episode.
        """
        pass

    @abstractmethod
    def create_agent(self, args: argparse.Namespace) -> Any:
        """Create and return the agent.

        Args:
            args: Parsed command line arguments.

        Returns:
            The agent instance.
        """
        pass

    @abstractmethod
    def add_env_specific_args(self, parser: argparse.ArgumentParser) -> None:
        """Add environment-specific CLI arguments.

        Args:
            parser: The argument parser to add arguments to.
        """
        pass

    @abstractmethod
    def get_system_prompt(self, agent: Any, env: Any, info: dict) -> str:
        """Return the system prompt for the agent.

        Args:
            agent: The agent instance.
            env: The environment instance.
            info: Extra info from create_environment.

        Returns:
            The system prompt string.
        """
        pass

    # =========== Optional Hooks (can override) ===========

    def get_tools(self, env: Any, info: dict) -> list[dict] | None:
        """Return OpenAI-compatible tools (for tool-calling environments).

        Args:
            env: The environment instance.
            info: Extra info from create_environment.

        Returns:
            List of tool schemas or None if not using tools.
        """
        return None

    def format_observation_display(self, obs: Any, info: dict) -> str:
        """Format observation for display in console.

        Args:
            obs: The observation from the environment.
            info: Step info from the environment.

        Returns:
            Formatted string for display.
        """
        obs_str = str(obs)
        return obs_str[:1000] + "..." if len(obs_str) > 1000 else obs_str

    def process_model_response(
        self,
        agent: Any,
        response: str,
    ) -> tuple[Any, str]:
        """Process model response and return action for environment.

        Args:
            agent: The agent instance.
            response: The model's response text.

        Returns:
            A tuple of (action_for_env, display_string).
        """
        action = agent.update_from_model(response)
        action_str = action.action
        return action_str, str(action_str)

    def print_episode_start(
        self,
        env: Any,
        agent: Any,
        info: dict,
        args: argparse.Namespace,
        model_id: str,
    ) -> None:
        """Print episode start information.

        Args:
            env: The environment instance.
            agent: The agent instance.
            info: Extra info from create_environment.
            args: Parsed command line arguments.
            model_id: The model ID being used.
        """
        print_header("EPISODE START")
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
        """Print step result information.

        Args:
            step: Current step number.
            obs: The observation from the environment.
            reward: The reward from this step.
            done: Whether the episode is done.
            info: Step info from the environment.
            total_reward: Cumulative reward so far.
        """
        print(f"[REWARD] {reward}")
        print(f"[CUMULATIVE REWARD] {total_reward}")
        print(f"[DONE] {done}")

    def print_episode_summary(self, results: dict) -> None:
        """Print final episode summary.

        Args:
            results: Dictionary of episode results.
        """
        print_header("EPISODE SUMMARY")
        for k, v in results.items():
            print(f"  {k}: {v}")
        print_separator()

    def cleanup(self, env: Any) -> None:
        """Cleanup after episode (close env, shutdown pools, etc.).

        Args:
            env: The environment instance.
        """
        if hasattr(env, "close"):
            env.close()

    # =========== Core Implementation ===========

    def create_argument_parser(self) -> argparse.ArgumentParser:
        """Create argument parser with common + env-specific args.

        Returns:
            Configured argument parser.
        """
        parser = argparse.ArgumentParser(description=f"Run {self.name}")
        # Common arguments
        parser.add_argument(
            "--server-url",
            type=str,
            default=self.default_server_url,
            help="LLM server URL",
        )
        parser.add_argument(
            "--max-steps",
            type=int,
            default=self.default_max_steps,
            help="Maximum steps per episode",
        )
        parser.add_argument(
            "--no-stream",
            action="store_true",
            help="Disable streaming output",
        )
        parser.add_argument(
            "--no-history",
            action="store_true",
            help="Disable conversation history accumulation",
        )
        parser.add_argument(
            "--quiet",
            action="store_true",
            help="Quiet mode: reduce verbose output",
        )
        # Add environment-specific arguments
        self.add_env_specific_args(parser)
        return parser

    def run_episode(
        self,
        client: OpenAI,
        model_id: str,
        env: Any,
        agent: Any,
        args: argparse.Namespace,
        system_prompt: str,
        tools: list[dict] | None = None,
    ) -> dict:
        """Run a single episode. Core orchestration logic.

        Args:
            client: OpenAI client for LLM inference.
            model_id: Model ID to use for inference.
            env: The environment instance.
            agent: The agent instance.
            args: Parsed command line arguments.
            system_prompt: The system prompt for the agent.
            tools: Optional list of tool schemas for tool-calling.

        Returns:
            Dictionary of episode results.
        """
        # Reset environment and agent
        obs, info = env.reset()
        agent.reset()

        total_reward = 0.0
        step = 0
        done = False
        reward = 0.0

        # Initialize message history
        messages = [{"role": "system", "content": system_prompt}]

        # Print start info
        if not args.quiet:
            self.print_episode_start(env, agent, info, args, model_id)

        # Show initial observation
        if not args.quiet and obs:
            print_header("INITIAL OBSERVATION")
            print(self.format_observation_display(obs, info))
            print()

        # Main loop
        while not done and step < args.max_steps:
            step += 1

            if not args.quiet:
                print_separator("-")
                print(f"  STEP {step}/{args.max_steps}")
                print_separator("-")

            # Update agent with current observation
            agent.update_from_env(obs, reward, done, info)

            # Build user prompt for API
            if hasattr(agent, "messages") and agent.messages:
                user_prompt = agent.messages[-1]["content"]
            else:
                user_prompt = str(obs)
            messages.append({"role": "user", "content": user_prompt})

            if not args.quiet:
                print("\n[OBSERVATION]")
                print(self.format_observation_display(obs, info))

            # Prepare messages for API call
            use_history = not args.no_history
            if use_history:
                messages_for_api = messages.copy()
            else:
                messages_for_api = [messages[0], messages[-1]]

            if not args.quiet:
                print("\n[MODEL RESPONSE]")

            # Get model response
            use_stream = not args.no_stream and not args.quiet

            if self.supports_tools and tools:
                response_text, tool_calls = get_model_response_with_tools(
                    client, model_id, messages_for_api, tools, stream=use_stream
                )
                cleaned = clean_special_tokens(response_text)

                if tool_calls:
                    messages.append({
                        "role": "assistant",
                        "content": cleaned or None,
                        "tool_calls": tool_calls,
                    })
                    obs, reward, done, info = env.step(tool_calls)
                    for tc in tool_calls:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.get("id", ""),
                            "content": str(obs) if obs else "Done",
                        })
                    if not args.quiet:
                        print(f"\n[TOOL CALLS] {len(tool_calls)} tool(s) called")
                else:
                    messages.append({"role": "assistant", "content": cleaned})
                    obs, reward, done, info = env.step(cleaned)
                    if obs:
                        messages.append({"role": "user", "content": str(obs)})
            else:
                response_text = get_model_response(
                    client, model_id, messages_for_api, stream=use_stream
                )
                cleaned = clean_special_tokens(response_text)
                messages.append({"role": "assistant", "content": cleaned})

                action_for_env, action_display = self.process_model_response(
                    agent, response_text
                )

                if not args.quiet:
                    print(f"\n[PARSED ACTION] {action_display}")

                obs, reward, done, info = env.step(action_for_env)

            total_reward += reward

            if not args.quiet:
                self.print_step_result(step, obs, reward, done, info, total_reward)

            if done:
                if not args.quiet:
                    print("\n*** EPISODE COMPLETED ***")
                break

        results = {
            "total_reward": total_reward,
            "steps_taken": step,
            "done": done,
        }

        if not args.quiet:
            self.print_episode_summary(results)

        return results

    def run(self) -> None:
        """Main entry point for the interactive runner."""
        parser = self.create_argument_parser()
        args = parser.parse_args()

        print_header(self.name.upper())
        print(f"Server URL: {args.server_url}")
        print()

        # Connect to LLM server
        try:
            client, model_id = connect_to_llm_server(args.server_url)
        except Exception as e:
            print(f"ERROR: Failed to connect to LLM server: {e}")
            print("\nPlease make sure an OpenAI-compatible server is running.")
            print("For SGLang, start one with:")
            print(
                "  python3 -m sglang.launch_server "
                "--model-path <model-name-or-path> --host 0.0.0.0 --port 8000"
            )
            return

        print()

        # Create environment and agent
        env, extra_info = self.create_environment(args)
        agent = self.create_agent(args)
        system_prompt = self.get_system_prompt(agent, env, extra_info)
        tools = self.get_tools(env, extra_info) if self.supports_tools else None

        try:
            results = self.run_episode(
                client, model_id, env, agent, args, system_prompt, tools
            )

            print("\n" + "=" * 70)
            print("  FINAL RESULTS")
            print("=" * 70)
            for k, v in results.items():
                print(f"  {k}: {v}")
            print("=" * 70)
        finally:
            self.cleanup(env)
