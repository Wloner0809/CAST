import asyncio
import logging
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

import torch

from rllm.agents.agent import Action, BaseAgent, Trajectory
from rllm.agents.utils import (
    convert_messages_to_tokens_and_masks,
    get_recent_assistant_user_messages,
)
from rllm.environments.base.base_env import BaseEnv
from rllm.environments.env_utils import (
    compute_mc_return,
    compute_trajectory_reward,
)
from rllm.parser import ChatTemplateParser
from rllm.utils import colorful_print

logger = logging.getLogger(__name__)


class AgentExecutionEngine:
    def __init__(
        self,
        engine_name="openai",
        tokenizer=None,
        rollout_engine=None,
        chat_parser=None,
        n_parallel_agents=128,  # The number of active agents
        trajectory_timeout=None,
        gamma=0.2,
        api_retries=3,
        retry_limit=3,
        max_steps=5,
        max_response_length=8192,
        max_prompt_length=1024,
        config=None,
        agent_class=None,
        env_class=None,
        agent_args=None,
        rollout_engine_args=None,
        env_args=None,
        max_workers=64,  # The number of concurrent env operations
        enforce_max_prompt_length=False,  # If enabled, applies max_prompt check per step
        overlong_filter=False,  # Filter for overlong trajectories (i.e. TRUNCATION, MAX_STEPS, TIMEOUT)
        explore_mode=False,  # If enabled, ignore env done signal and continue until agent decides or max_steps
        progress_jsonl_path=None,  # If set, append each completed trajectory as one jsonl line (for resume)
        **kwargs,
    ):
        if agent_args is None:
            agent_args = {}
        if rollout_engine_args is None:
            rollout_engine_args = {}
        if env_args is None:
            env_args = {}

        self.config = config
        self.tokenizer = tokenizer
        self.engine_name = engine_name
        self.n_parallel_agents = n_parallel_agents
        self.max_env_workers = max_workers
        self.overlong_filter = overlong_filter
        self.explore_mode = explore_mode
        # Path to append per-trajectory progress jsonl (one line per completed
        # trajectory) so a killed run can resume. None disables it (default).
        self.progress_jsonl_path = progress_jsonl_path
        self._progress_write_lock = None  # lazily created asyncio.Lock in execute_tasks
        self.use_api_token_usage = self.tokenizer is None and self.engine_name == "openai"

        # For interaction
        self.gamma = gamma
        self.retry_limit = retry_limit
        self.max_steps = max_steps
        self.max_response_length = max_response_length
        self.max_prompt_length = max_prompt_length
        self.enforce_max_prompt_length = enforce_max_prompt_length
        self.log_trajectory_turn_progress = bool(
            kwargs.get("log_trajectory_turn_progress", False)
        )
        self.trajectory_progress_log_interval = int(
            kwargs.get("trajectory_progress_log_interval", 1)
        )
        if self.trajectory_progress_log_interval < 1:
            self.trajectory_progress_log_interval = 1
        self.disable_thinking = (
            self.config.get("rllm", {}).get("disable_thinking", False)
            if self.config is not None
            else False
        )

        self.agent_class = agent_class
        self.agent_args = agent_args
        self.env_class = env_class
        self.env_args = env_args

        self.agents = [None for _ in range(n_parallel_agents)]
        self.envs = [None for _ in range(n_parallel_agents)]

        self.trajectory_timeout = trajectory_timeout
        if not trajectory_timeout:
            self.trajectory_timeout = int(1e9)

        if env_class is not None:
            assert env_class.is_multithread_safe(), (
                "Environment must be multithread safe for async engine"
            )

        if chat_parser is None:
            if self.tokenizer is None:
                self.chat_parser = None
            else:
                self.chat_parser = ChatTemplateParser.get_parser(
                    self.tokenizer, disable_thinking=self.disable_thinking
                )
        else:
            self.chat_parser = chat_parser

        self.rollout_engine_args = rollout_engine_args
        self.sampling_params = kwargs.get(
            "sampling_params", {}
        )  # for openai api requests

        assert self.engine_name in ["openai", "verl", "tinker", "sglang"], (
            "Currently only openai, verl, tinker and sglang are supported as rollout engine"
        )
        if self.engine_name == "openai":
            from rllm.engine.rollout.openai_engine import OpenAIEngine

            self.rollout_engine = OpenAIEngine(
                **rollout_engine_args,
                api_retries=api_retries,
                tokenizer=self.tokenizer,
                max_prompt_length=self.max_prompt_length,
                max_response_length=self.max_response_length,
                disable_thinking=self.disable_thinking,
            )
        elif self.engine_name == "verl":
            from rllm.engine.rollout.verl_engine import VerlEngine

            self.rollout_engine = VerlEngine(
                config=self.config,
                rollout_manager=rollout_engine,
                tokenizer=self.tokenizer,
                disable_thinking=self.disable_thinking,
            )
        elif self.engine_name == "tinker":
            from rllm.engine.rollout.tinker_engine import TinkerEngine

            self.rollout_engine = TinkerEngine(
                **rollout_engine_args,
            )
        elif self.engine_name == "sglang":
            if rollout_engine is not None:
                # Reuse a pre-built SglangEngine (e.g., shared across runs)
                self.rollout_engine = rollout_engine
            else:
                from rllm.engine.rollout.sglang_engine import SglangEngine

                self.rollout_engine = SglangEngine(
                    **rollout_engine_args,
                    tokenizer=self.tokenizer,
                    max_prompt_length=self.max_prompt_length,
                    max_response_length=self.max_response_length,
                    disable_thinking=self.disable_thinking,
                )

        # Create a thread pool executor for environment interactions (i.e. step, reset, close)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def initialize(self):
        """Initialize the rollout engine.

        This method should be called from the main thread before any async
        operations, especially for engines like sglang that require main
        thread initialization for signal handlers.
        """
        if hasattr(self.rollout_engine, "initialize"):
            self.rollout_engine.initialize()

    async def get_model_response(self, prompt, application_id, **kwargs) -> str:
        """
        Compute model response asynchronously based on the engine type.

        This function is multithread safe and routes the request to the appropriate
        engine-specific handler.

        Args:
            prompt: The input prompt to send to the model
            application_id: Unique identifier for the application
            **kwargs: Additional arguments to pass to the model

        Returns:
            The model's response text

        Raises:
            NotImplementedError: If the engine type is not supported
        """

        sampling_params = self.sampling_params.copy()
        sampling_params.update(kwargs)

        if self.engine_name != "sglang":
            sampling_params.pop("return_full_logprobs", None)

        if self.engine_name == "openai":
            output = await self.rollout_engine.get_model_response(
                prompt,
                application_id=application_id,
                enforce_max_prompt_length=False,
                **sampling_params,
            )
            return output
        elif self.engine_name == "verl":
            meta_data = sampling_params.pop("meta_info", {})
            validate = meta_data.get("validate", False)
            output = await self.rollout_engine.get_model_response(
                prompt,
                application_id=application_id,
                validate=validate,
                enforce_max_prompt_length=False,
                **sampling_params,
            )
            return output
        elif self.engine_name == "tinker":
            output = await self.rollout_engine.get_model_response(
                prompt,
                application_id=application_id,
                enforce_max_prompt_length=False,
                **sampling_params,
            )
            return output
        elif self.engine_name == "sglang":
            output = await self.rollout_engine.get_model_response(
                prompt, enforce_max_prompt_length=False, **sampling_params
            )
            return output
        else:
            raise NotImplementedError(f"Engine type '{self.engine_name}' not supported")

    def update_envs_and_agents(self, envs, agents):
        """
        Update the environments and agents.

        Args:
            envs: List of environments to use
            agents: List of agents to use
        """
        assert len(agents) == len(envs), (
            f"Number of agents must equal to number of environments but received, {len(agents)} and {len(envs)}"
        )
        self.envs = envs
        # For keeping track of the environment index in the batch.
        for idx, env in enumerate(envs):
            env.idx = idx
        self.agents = agents

    async def run_agent_trajectory_async(
        self, idx, application_id, seed=0, mode="Text", **kwargs
    ):
        """Run a single agent's trajectory asynchronously"""
        agent = self.agents[idx]
        env = self.envs[idx]
        # env_id = env.env_id

        termination_reason = None
        prompt_token_len = 0
        prompt_tokens = []
        response_token_len = 0
        response_tokens = []
        response_masks = []
        total_time = 0.0
        reward_time = None
        llm_time = 0.0
        env_time = 0.0
        # Solver profiling (ablation only). Always initialized so every trajectory's
        # metrics dict carries these keys (see _transform_agent_* which templates the
        # key set from traj_metrics[0] and would KeyError on a missing key). Values
        # stay 0.0 when the env emits no solver timing (e.g. baseline runs).
        solver_time = 0.0  # in-step solver wall time (env writes info["solver_time"])
        deadlock_solver_time = 0.0  # Sokoban deadlock A* (info["deadlock_solver_time"])
        reset_solver_time = 0.0  # Rush Hour N(s)-table build at reset (info["reset_solver_time"])
        reward = 0.0

        # for step return
        episode_steps = []

        # Reset environment with the task using the executor
        loop = asyncio.get_event_loop()
        observation, info = await loop.run_in_executor(self.executor, env.reset)
        info["max_steps"] = self.max_steps
        # Solver time spent during reset (e.g. Rush Hour N(s)-table build). Read
        # defensively: envs that don't emit it leave reset_solver_time at 0.0.
        if isinstance(info, dict):
            reset_solver_time += info.get("reset_solver_time", 0.0)

        # Reset agent
        agent.reset()
        # Update agent internal state from environment.
        agent.update_from_env(
            observation=observation,  # Raw observation from environment
            reward=0.0,
            done=False,
            info=info,
        )
        messages = agent.chat_completions
        if self.tokenizer is not None:
            prompt_tokens, _ = convert_messages_to_tokens_and_masks(
                messages,
                tokenizer=self.tokenizer,
                parser=self.chat_parser,
                contains_first_msg=True,
                contains_generation_msg=True,
            )
            prompt_token_len = len(prompt_tokens)
            # Note, this should never happen!
            if prompt_token_len > self.max_prompt_length:
                agent.reset()
                from pprint import pprint

                pprint(messages)
                raise Exception(
                    f"Trajectory {idx}: initial prompt length {prompt_token_len} already exceeded max_prompt_length {self.max_prompt_length}, retrying"
                )

        for step_idx in range(self.max_steps):
            # Get action from agent
            prompt_messages = agent.chat_completions.copy()
            # Max remaining tokens left for the response
            # For enforced max prompt at each step, no need to deduct here
            if not self.enforce_max_prompt_length:
                max_tokens = self.max_response_length - response_token_len
            else:
                max_tokens = self.max_response_length

                # since max prompt is enforced, we filter out too long prompts.
                if self.tokenizer is not None:
                    prompt_str = self.chat_parser.parse(
                        prompt_messages, add_generation_prompt=True, is_first_msg=True
                    )
                    prompt_len = len(
                        self.tokenizer.encode(prompt_str, add_special_tokens=False)
                    )
                    if prompt_len > self.max_prompt_length:
                        termination_reason = "PROMPT_TRUNCATION"
                        break

            kwargs["max_tokens"] = max_tokens

            start_time = time.time()
            model_output = await self.get_model_response(
                prompt_messages, application_id, **kwargs
            )
            if self.use_api_token_usage:
                prompt_token_len = model_output.prompt_length or 0
                response_token_len += model_output.completion_length or 0
            if not model_output.text:
                logger.warning(
                    "Model returned empty text. Raw model output: %s",
                    model_output.to_dict(),
                )
            response = model_output.text or ""
            delta_time = time.time() - start_time
            llm_time += delta_time
            total_time += delta_time
            # Update steps
            if self.chat_parser is not None:
                prompt_text = self.chat_parser.parse(
                    prompt_messages, add_generation_prompt=True, is_first_msg=True
                )
            else:
                prompt_text = None
            prompt_response_pair = {
                "prompt": prompt_text,
                "response": response,
                "prompt_ids": model_output.prompt_ids,
                "completion_ids": model_output.completion_ids,
                "logprobs": model_output.logprobs,
            }
            # Record anchor observation for step-level grouping (e.g. GiGPO).
            # Always capture from agent.current_observation; CM overrides when present.
            prompt_response_pair["anchor_obs"] = getattr(agent, "current_observation", None)
            cm = getattr(agent, "_context_manager", None)
            if cm is not None:
                prompt_response_pair["anchor_obs"] = cm.get_anchor_obs()
            episode_steps.append(prompt_response_pair)

            # Update agent with model response
            action: Action = agent.update_from_model(response)

            # Inject actual stop token ID into the last assistant message so
            # parse_assistant can use the correct eot string for re-rendering.
            # This avoids assemble_steps mismatch when Qwen2.5 generates
            # <|endoftext|> instead of <|im_end|>.
            if model_output.stop_token_id is not None:
                cm = getattr(agent, "_context_manager", None)
                if cm is not None:
                    # ContextManager stores history separately from
                    # agent.messages; inject via the dedicated method.
                    cm.inject_stop_token_id(model_output.stop_token_id)
                elif hasattr(agent, 'messages'):
                    for msg in reversed(agent.messages):
                        if msg.get("role") == "assistant":
                            msg["_stop_token_id"] = model_output.stop_token_id
                            break

            action = action.action

            # Take step in environment using the executor
            start_time = time.time()

            try:
                next_observation, reward, done, info = await asyncio.wait_for(
                    loop.run_in_executor(self.executor, env.step, action),
                    timeout=(self.trajectory_timeout - total_time),
                )
            except (asyncio.TimeoutError, asyncio.CancelledError):
                termination_reason = "ENV_TIMEOUT"
                if step_idx == 0:
                    colorful_print(
                        f"Warning: Trajectory {idx} completed due to: {termination_reason} before able to perform 1 complete action. This might cause unexpected behavior. Consider increasing trajectory timeout limit.\n",
                        "red",
                    )
                reward = 0

                cur_step = agent.get_current_state()
                done = True
                cur_step.done = done
                break

            delta_time = time.time() - start_time
            env_time += delta_time
            total_time += delta_time
            info["max_steps"] = self.max_steps
            info["cur_tokens"] = response_token_len
            # Record oracle advantage from solver-based environments (e.g., Sokoban)
            if len(episode_steps) > 0 and "oracle_advantage" in info:
                episode_steps[-1]["oracle_advantage"] = info["oracle_advantage"]
            # Accumulate solver wall time (ablation profiling). Envs that don't emit
            # these keys contribute 0.0, so baseline runs report solver_time == 0.
            solver_time += info.get("solver_time", 0.0)
            deadlock_solver_time += info.get("deadlock_solver_time", 0.0)

            # Update agent internal state.
            agent.update_from_env(
                observation=next_observation,
                reward=reward,
                done=done,
                info=info,
            )

            cur_step = agent.get_current_state()
            cur_step.reward = reward
            cur_step.done = done
            cur_step.info.update(info)
            # Populate logprobs and token IDs from model output for metrics computation
            if model_output.logprobs:
                cur_step.logprobs = model_output.logprobs
            if model_output.token_entropies:
                cur_step.token_entropies = model_output.token_entropies
            if model_output.completion_ids:
                cur_step.response_ids = model_output.completion_ids
            if model_output.prompt_ids:
                cur_step.prompt_ids = model_output.prompt_ids
            # Store the full model_output for detailed analysis
            cur_step.model_output = model_output

            if self.engine_name == "openai" and self.log_trajectory_turn_progress:
                turns = step_idx + 1
                if (
                    turns == 1
                    or done
                    or turns % self.trajectory_progress_log_interval == 0
                ):
                    colorful_print(
                        (
                            f"Trajectory {idx} progress: "
                            f"{turns}/{self.max_steps} turns generated "
                            f"(done={done}, last_reward={reward:.4f})"
                        ),
                        "cyan",
                    )

            chat_completions_messages = agent.chat_completions
            assistant_message, env_messages = get_recent_assistant_user_messages(
                chat_completions_messages
            )

            # Check and convert to tokens if necessary
            assert assistant_message is not None or mode != "Token", (
                "Assistant messages is none when accumulating token trajectories which should be conversations. This should not happen."
            )
            assert env_messages is not None or mode != "Token", (
                "Environment messages is none when accumulating token trajectories which should be conversations. This should not happen."
            )
            assistant_msg_tokens, assistant_msg_masks = [], []
            env_msg_tokens, env_msg_masks = [], []
            if self.tokenizer is not None:
                if assistant_message:
                    assistant_msg_tokens, assistant_msg_masks = (
                        convert_messages_to_tokens_and_masks(
                            [assistant_message],
                            tokenizer=self.tokenizer,
                            parser=self.chat_parser,
                            contains_first_msg=False,
                            contains_generation_msg=False,
                        )
                    )
                if env_messages:
                    env_msg_tokens, env_msg_masks = convert_messages_to_tokens_and_masks(
                        env_messages,
                        tokenizer=self.tokenizer,
                        parser=self.chat_parser,
                        contains_first_msg=False,
                        contains_generation_msg=True,
                    )

                # Update repsonse token length
                response_token_len += len(assistant_msg_tokens) + len(env_msg_tokens)
                # Reached maximum number of tokens for the trajectory
                if (
                    not self.enforce_max_prompt_length
                    and response_token_len >= self.max_response_length
                ):
                    # Truncation length
                    truncation_length = self.max_response_length - response_token_len
                    # Truncate the response and masks
                    if truncation_length < 0:
                        truncated_response_tokens = (
                            assistant_msg_tokens + env_msg_tokens
                        )[:truncation_length]
                        truncated_response_masks = (
                            assistant_msg_masks + env_msg_masks
                        )[:truncation_length]
                    else:
                        # Edge case where the response is exactly the max response length.
                        truncated_response_tokens = (
                            assistant_msg_tokens + env_msg_tokens
                        )
                        truncated_response_masks = assistant_msg_masks + env_msg_masks
                    # Update token collections
                    response_tokens.extend(truncated_response_tokens)
                    response_masks.extend(truncated_response_masks)

                    cur_step = agent.get_current_state()
                    if (
                        response_token_len - len(env_msg_tokens)
                        > self.max_response_length
                    ):
                        cur_step.reward = 0.0
                    cur_step.done = True
                    termination_reason = "TRUNCATION"
                    # handle returning
                    break

                # Update the token version of trajectory
                response_tokens.extend(assistant_msg_tokens)
                response_masks.extend(assistant_msg_masks)
            else:
                if (
                    not self.enforce_max_prompt_length
                    and response_token_len >= self.max_response_length
                ):
                    cur_step = agent.get_current_state()
                    cur_step.done = True
                    termination_reason = "TRUNCATION"
                    break
            observation = next_observation

            if total_time >= self.trajectory_timeout:
                termination_reason = "TIMEOUT"
                cur_step = agent.get_current_state()
                done = True
                cur_step.done = done
                break

            # Check if episode is done
            # In explore mode, ignore env done signal and only stop when agent decides
            if self.explore_mode:
                # Check if agent has exploration_complete attribute (ExploreAgent)
                agent_done = getattr(agent, "exploration_complete", False)
                if agent_done:
                    termination_reason = "AGENT_EXPLORATION_COMPLETE"
                    break
                # In explore mode, we continue even if env signals done
            elif done:
                termination_reason = "ENV_DONE"
                break

            response_tokens.extend(env_msg_tokens)
            response_masks.extend(env_msg_masks)

            if step_idx == self.max_steps - 1:
                termination_reason = "MAX_STEPS"

        masked_out = False
        if self.overlong_filter:
            if (
                termination_reason == "TRUNCATION"
                or termination_reason == "MAX_STEPS"
                or termination_reason == "TIMEOUT"
            ):
                # Mask out the entire response for overlong trajectories if the reward is 0.
                response_masks = [0] * len(response_masks)
                masked_out = True

        if hasattr(env, "compute_final_reward") and not masked_out:
            cur_step = agent.get_current_state()
            start_time = time.time()
            reward = await loop.run_in_executor(self.executor, env.compute_final_reward)
            reward_time = time.time() - start_time
            cur_step.reward = reward
        # Closing environment using the executor.
        await loop.run_in_executor(self.executor, env.close)
        if termination_reason:
            if reward > 0:
                color = "green"
            else:
                color = "yellow"
            colorful_print(
                f"Trajectory {idx} completed due to: {termination_reason}. Reward is {reward}. \n",
                color,
            )
            if masked_out:
                colorful_print(
                    f"Trajectory {idx} is masked out due to overlong filter.", "red"
                )

        trajectory: Trajectory = agent.trajectory
        # Aggregate final trajectory statistics
        compute_trajectory_reward(trajectory)
        compute_mc_return(trajectory, gamma=self.gamma)

        # In explore mode, add world_modeling to trajectory info if available
        if self.explore_mode:
            world_modeling = getattr(agent, "world_modeling_content", None)
            if world_modeling:
                trajectory.info["world_modeling"] = world_modeling
            exploration_trajectory = getattr(agent, "exploration_trajectory", None)
            if exploration_trajectory:
                trajectory.info["exploration_trajectory"] = exploration_trajectory

        # Extract common terminal metrics from the last step's info if available.
        if trajectory.steps:
            last_step_info = trajectory.steps[-1].info
            success = last_step_info.get("success", None)
            if success is not None:
                trajectory.info["success"] = success
            score = last_step_info.get("score", None)
            if score is not None:
                trajectory.info["score"] = score

        if mode == "Text":
            return trajectory
        elif mode == "Token":
            prompt_tokens, response_tokens, response_masks, oracle_advs, step_token_ids, is_valid_trajectory = (
                self.assemble_steps(episode_steps)
            )
            # Fallback: if assemble_steps failed (e.g., due to reasoning token drop
            # when accumulate_reasoning=False), use cumulative per-message tokenization
            # which avoids the retokenization problem entirely.
            if not is_valid_trajectory and hasattr(self.chat_parser, "tokenize_and_mask_cumulative"):
                try:
                    chat_completions = agent.chat_completions
                    if chat_completions and len(chat_completions) > 0:
                        fallback_prompt, fallback_response, fallback_mask = (
                            self.chat_parser.tokenize_and_mask_cumulative(chat_completions)
                        )
                        if len(fallback_response) > 0:
                            prompt_tokens = fallback_prompt
                            response_tokens = fallback_response
                            response_masks = fallback_mask
                            oracle_advs = torch.zeros(len(fallback_response), dtype=torch.float32)
                            # Place cumulative oracle advantage on last token
                            total_oracle_adv = sum(s.get("oracle_advantage", 0.0) for s in episode_steps)
                            if len(oracle_advs) > 0:
                                oracle_advs[-1] = total_oracle_adv
                            # The cumulative fallback rebuilds tokens from chat_completions and
                            # has NO per-step token structure, so GiGPO step-level advantage
                            # cannot be attributed token-by-token. Mark every token as -1
                            # (no step owner) → the GiGPO scatter assigns only episode-level
                            # advantage to this trajectory and skips the step-level term.
                            step_token_ids = torch.full((len(fallback_response),), -1, dtype=torch.long)
                            is_valid_trajectory = True
                            logger.info(
                                f"Trajectory {idx}: Fallback to cumulative tokenization succeeded "
                                f"(prompt={len(prompt_tokens)}, response={len(response_tokens)} tokens)"
                            )
                except Exception as e:
                    logger.warning(
                        f"Trajectory {idx}: Cumulative tokenization fallback failed: {e}"
                    )

            # Extract success from the last step's info if available
            success = None
            score = None
            if trajectory.steps:
                last_step_info = trajectory.steps[-1].info
                success = last_step_info.get("success", None)
                score = last_step_info.get("score", None)
            if success is None:
                success = trajectory.info.get("success", None)
            if score is None:
                score = trajectory.info.get("score", None)

            token_result = {
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
                "response_masks": response_masks,
                "trajectory_reward": trajectory.reward,
                "success": success,  # Add success boolean from environment
                "score": score,  # Add final env score from environment (e.g., 2048).
                "idx": env.idx,
                "oracle_advantages": oracle_advs,
                "turn_advantages": [s.get("oracle_advantage", 0.0) for s in episode_steps],
                "chat_completions": agent.chat_completions,
                # Per-step metadata for trajectory-level GiGPO advantage computation
                "step_rewards": [step.reward for step in trajectory.steps][:len(episode_steps)],
                "step_anchor_obs": [s.get("anchor_obs", "") for s in episode_steps],
                "step_completion_lengths": [len(s.get("completion_ids", [])) for s in episode_steps],
                # Per-token step ownership built in lockstep with response_tokens (see
                # assemble_steps). Each entry is the owning episode_step index, or -1 for
                # gap / non-completion tokens. Used by GiGPO trajectory-level advantage to
                # scatter per-step scalars without re-deriving alignment from token counts.
                "step_token_ids": step_token_ids,
                "metrics": {
                    # Total number of steps taken in the trajectory
                    "steps": len(trajectory.steps),
                    # Time to calculate reward
                    "reward_time": reward_time,
                    # Total time spent in environment execution (env.step)
                    "env_time": env_time,
                    # Time to calculate response tokens
                    "llm_time": llm_time,
                    # Total time spent in the trajectory
                    "total_time": total_time,
                    "token_mismatch": 0.0 if is_valid_trajectory else 1.0,
                    # Solver profiling (ablation). Always present so the metrics key
                    # set is identical across every trajectory in a batch.
                    "solver_time": solver_time,
                    "deadlock_solver_time": deadlock_solver_time,
                    "reset_solver_time": reset_solver_time,
                },
            }
            return token_result
        elif mode == "Conversation":
            return agent.chat_completions
        elif mode == "Step":
            # Extract success from the last step's info if available
            success = None
            score = None
            if trajectory.steps:
                last_step_info = trajectory.steps[-1].info
                success = last_step_info.get("success", None)
                score = last_step_info.get("score", None)
            if success is None:
                success = trajectory.info.get("success", None)
            if score is None:
                score = trajectory.info.get("score", None)

            oracle_advantages = [step.get("oracle_advantage", 0.0) for step in episode_steps]
            steps_result = {
                "steps": episode_steps,
                "trajectory_reward": trajectory.reward,
                "success": success,  # Add success boolean from environment
                "score": score,  # Add final env score from environment (e.g., 2048).
                "idx": env.idx,
                "mc_returns": [step.mc_return for step in trajectory.steps][
                    : len(episode_steps)
                ],
                "step_rewards": [step.reward for step in trajectory.steps][
                    : len(episode_steps)
                ],
                "oracle_advantages": oracle_advantages,
                "termination_reason": termination_reason,
                "chat_completions": agent.chat_completions,
                "metrics": {
                    "steps": len(trajectory.steps),
                    "reward_time": reward_time,
                    "env_time": env_time,
                    "llm_time": llm_time,
                    "total_time": total_time,
                    "token_mismatch": 0.0,
                    # Solver profiling (ablation). Always present so the metrics key
                    # set is identical across every trajectory in a batch.
                    "solver_time": solver_time,
                    "deadlock_solver_time": deadlock_solver_time,
                    "reset_solver_time": reset_solver_time,
                },
            }
            return steps_result
        else:
            raise ValueError(f"Mode {mode} not supported")

    def _log_assemble_mismatch(self, step_idx, accumulated_sequence, current_prompt_ids):
        """Log detailed diagnostics for an assemble_steps token mismatch.

        This method provides actionable debugging info to identify root causes:
        - Which token position first diverges
        - What the expected vs actual tokens/text are
        - Whether the issue is <think> tag stripping (common Qwen3 issue)
        - Whether the issue is a length mismatch

        Args:
            step_idx: The step index where the mismatch occurred
            accumulated_sequence: Expected token sequence from previous steps
            current_prompt_ids: Current step's re-tokenized prompt IDs
        """
        acc_len = len(accumulated_sequence)
        curr_len = len(current_prompt_ids)

        # Find first differing position
        compare_len = min(acc_len, curr_len)
        diff_pos = None
        for j in range(compare_len):
            if accumulated_sequence[j] != current_prompt_ids[j]:
                diff_pos = j
                break

        if diff_pos is None and acc_len != curr_len:
            # Tokens match up to the shorter length, but lengths differ
            diff_pos = compare_len

        # --- Build diagnostic message ---
        parts = [f"[assemble_steps] MISMATCH at step {step_idx}"]
        parts.append(f"  acc_len={acc_len}, prompt_len={curr_len}")

        if diff_pos is not None:
            parts.append(f"  first_diff_pos={diff_pos}")

            # Show token IDs around the diff
            window = 5
            exp_slice = accumulated_sequence[max(0, diff_pos - window):min(acc_len, diff_pos + window)]
            got_slice = current_prompt_ids[max(0, diff_pos - window):min(curr_len, diff_pos + window)]
            parts.append(f"  expected_tokens[{max(0,diff_pos-window)}:{min(acc_len,diff_pos+window)}]: {exp_slice}")
            parts.append(f"  actual_tokens  [{max(0,diff_pos-window)}:{min(curr_len,diff_pos+window)}]: {got_slice}")

            # Decode tokens for human-readable diff (if tokenizer available)
            if self.tokenizer is not None:
                try:
                    exp_text = self.tokenizer.decode(exp_slice, skip_special_tokens=False)
                    got_text = self.tokenizer.decode(got_slice, skip_special_tokens=False)
                    # Truncate long texts for readability
                    if len(exp_text) > 80:
                        exp_text = exp_text[:80] + "..."
                    if len(got_text) > 80:
                        got_text = got_text[:80] + "..."
                    parts.append(f"  expected_text: {repr(exp_text)}")
                    parts.append(f"  actual_text:   {repr(got_text)}")
                except Exception:
                    pass

                # Check for <think> tag issues (common Qwen3 problem)
                try:
                    # Look at a wider window around the diff to detect <think> pattern
                    ctx_start = max(0, diff_pos - 10)
                    ctx_end = min(acc_len, diff_pos + 10)
                    ctx_text = self.tokenizer.decode(
                        accumulated_sequence[ctx_start:ctx_end],
                        skip_special_tokens=False
                    )
                    if "<think>" in ctx_text or "</think>" in ctx_text:
                        parts.append(
                            "  HINT: <think> tag detected near diff. Note: <think>/<think> are "
                            "NOT special tokens in Qwen3 (special=False). Check accumulate_reasoning "
                            "config or parse_assistant rendering logic."
                        )
                except Exception:
                    pass
        else:
            parts.append("  No token-level diff found (unexpected)")

        # Log as a single message (avoid flooding)
        logger.warning("\n".join(parts))

    def assemble_steps(self, steps: list[dict]):
        """
        Transform step-by-step results into trajectory format for training.
        The assemble is aggresive, if steps is not cumulative, the response_masks is set to all 0s.

        Known mismatch sources and fixes:
        --------------------------------------------------------------------------
        1. DeepSeek <think>\n prefix missing (MAJOR):
           - generation_prompt includes <think>\n, but completion_ids do NOT contain it
           - Agent stores decoded text (skip_special_tokens=True) without <think>\n
           - With accumulate_reasoning=False, parse_assistant drops <think>\n from re-render
           - Fix: Set accumulate_reasoning=True in config (rllm.accumulate_reasoning=True)
        2. .strip() on content in parse_assistant (MINOR but real):
           - parse_assistant used to .strip() content, removing trailing whitespace/newlines
           - If model generates trailing \n before EOS, .strip() drops it from re-rendered text
           - accumulated_sequence keeps original tokens (with trailing \n), causing mismatch
           - Fix: Removed .strip() from content in all parse_assistant implementations
        3. DeepSeekV32Exp structural bug (MAJOR):
           - Old code: `if content: result += "</think>" + content` regardless of reasoning
           - With no reasoning field, produced </think>content instead of <think>content
           - Fix: Restructured to add <think> prefix when accumulate_reasoning=True
        4. Qwen2.5 <|endoftext|> stop token mismatch (MAJOR):
           - Qwen2.5 may generate <|endoftext|> (ID 151643) as stop token instead of <|im_end|> (ID 151645)
           - parse_assistant() hardcoded <|im_end|>\n as eot, causing re-rendered prompt to differ
           - accumulated_sequence ends with <|endoftext|> but re-tokenized prompt has <|im_end|>
           - Fix: Propagate actual stop_token_id through ModelOutput → agent message dict → parse_assistant
             which uses _stop_token_eot_map to select the correct eot string

        5. BPE retokenization divergence (FUNDAMENTAL, autoregressive vs canonical):
           - During autoregressive generation, the model can produce non-canonical BPE
             token sequences. E.g., for Sokoban board text "_#", the model generates two
             tokens [tok("_"), tok("#")] one-by-one, but canonical BPE merges them into
             a single token [tok("_#")]. When the full conversation is re-tokenized at
             the next step, the canonical tokenization is used, causing a prefix mismatch.
           - Fix: Before comparison, canonicalize accumulated_sequence via decode→re-encode.
             This resolves the alignment while preserving original completion_ids (mask=1)
             for correct policy gradient computation.
        """
        initial_prompt_ids = steps[0]["prompt_ids"]
        accumulated_sequence = initial_prompt_ids.copy()
        response_tokens = []
        response_masks = []
        oracle_advs = []
        # Per-token step ownership, built in lockstep with response_tokens exactly
        # like oracle_advs above. Each completion token carries the index ``i`` of the
        # episode_step that produced it; gap tokens (env-injected, mask=0) carry -1.
        # This lets a downstream consumer (GiGPO trajectory-level advantage) scatter a
        # per-step scalar onto its tokens WITHOUT re-deriving alignment from step token
        # counts — so it survives max_response_length truncation and BPE recanonicalization
        # for the same reason oracle_advs does (the id rides the token through pad/truncate).
        step_token_ids = []
        is_valid_trajectory = True


        for i, step in enumerate(steps):
            current_prompt_ids = step["prompt_ids"]
            current_completion_ids = step["completion_ids"]
            step_adv = step.get("oracle_advantage", 0.0)
            comp_advs = [step_adv] * len(current_completion_ids)
            comp_step_ids = [i] * len(current_completion_ids)

            if i == 0:
                # First step: just add completion
                response_tokens.extend(current_completion_ids)
                response_masks.extend(
                    [1] * len(current_completion_ids)
                )  # completion contributes to loss
                oracle_advs.extend(comp_advs)
                step_token_ids.extend(comp_step_ids)
                accumulated_sequence.extend(current_completion_ids)
            else:
                acc_len = len(accumulated_sequence)
                curr_len = len(current_prompt_ids)
                # Try strict exact prefix match first (fast path).
                if (
                    curr_len >= acc_len
                    and current_prompt_ids[:acc_len] == accumulated_sequence
                ):
                    # Perfect match - normal case
                    response_tokens.extend(
                        current_prompt_ids[acc_len:] + current_completion_ids
                    )
                    response_masks.extend(
                        [0] * (curr_len - acc_len) + [1] * len(current_completion_ids)
                    )
                    oracle_advs.extend([0.0] * (curr_len - acc_len) + comp_advs)
                    step_token_ids.extend([-1] * (curr_len - acc_len) + comp_step_ids)
                    accumulated_sequence = list(current_prompt_ids) + list(current_completion_ids)
                elif self.tokenizer is not None:
                    # Strict match failed. Try canonical re-tokenization.
                    # Autoregressive generation can produce non-canonical BPE sequences
                    # (e.g., 2 tokens where canonical BPE produces 1 merged token).
                    # Re-encode the accumulated text to get canonical tokenization.
                    canonical_acc = self.tokenizer.encode(
                        self.tokenizer.decode(accumulated_sequence, skip_special_tokens=False),
                        add_special_tokens=False,
                    )
                    canonical_len = len(canonical_acc)

                    if (
                        curr_len >= canonical_len
                        and current_prompt_ids[:canonical_len] == canonical_acc
                    ):
                        # Canonical match — retokenization divergence resolved.
                        # Gap tokens (env response) use canonical form (mask=0, no loss).
                        # Original completion_ids are preserved (mask=1, for policy gradient).
                        gap_ids = current_prompt_ids[canonical_len:]
                        response_tokens.extend(gap_ids)
                        response_masks.extend([0] * len(gap_ids))
                        oracle_advs.extend([0.0] * len(gap_ids))
                        step_token_ids.extend([-1] * len(gap_ids))
                        response_tokens.extend(current_completion_ids)
                        response_masks.extend([1] * len(current_completion_ids))
                        oracle_advs.extend(comp_advs)
                        step_token_ids.extend(comp_step_ids)
                        accumulated_sequence = list(current_prompt_ids) + list(current_completion_ids)

                        logger.debug(
                            f"[assemble_steps] BPE retokenization resolved at step {i}: "
                            f"original acc_len={acc_len}, canonical_len={canonical_len} "
                            f"(delta={acc_len - canonical_len} tokens)"
                        )
                    else:
                        # Even canonical comparison fails — genuine mismatch.
                        self._log_assemble_mismatch(
                            step_idx=i,
                            accumulated_sequence=canonical_acc,
                            current_prompt_ids=current_prompt_ids,
                        )
                        is_valid_trajectory = False
                        break
                else:
                    # No tokenizer available for canonical comparison.
                    self._log_assemble_mismatch(
                        step_idx=i,
                        accumulated_sequence=accumulated_sequence,
                        current_prompt_ids=current_prompt_ids,
                    )
                    is_valid_trajectory = False
                    break

        assert len(response_masks) == len(response_tokens)
        assert len(step_token_ids) == len(response_tokens), (
            f"step_token_ids length ({len(step_token_ids)}) != response_tokens "
            f"length ({len(response_tokens)}); lockstep construction broke"
        )

        prompt_tokens = torch.tensor(initial_prompt_ids, dtype=torch.long)
        response_tokens = torch.tensor(response_tokens, dtype=torch.long)
        response_masks = torch.tensor(response_masks, dtype=torch.long)
        oracle_advs = torch.tensor(oracle_advs, dtype=torch.float32)
        # Keep as long; -1 marks gap/non-completion tokens. NOT zeroed on
        # invalid-trajectory filtering below: multiplying ids by 0 would silently
        # reattribute every token to step 0. The response_mask zeroing already
        # neutralizes these tokens for the downstream scatter, so leaving the raw
        # ids intact is both correct and clearer.
        step_token_ids = torch.tensor(step_token_ids, dtype=torch.long)

        if self.config.rllm.filter_token_mismatch:
            response_masks = response_masks * int(is_valid_trajectory)
            oracle_advs = oracle_advs * int(is_valid_trajectory)

        return prompt_tokens, response_tokens, response_masks, oracle_advs, step_token_ids, is_valid_trajectory

    async def run_agent_trajectory_with_retry(self, idx, seed=0, mode="Text", **kwargs):
        outer_timeout = min(self.trajectory_timeout + 60, 7200)
        for _ in range(self.retry_limit):
            try:
                application_id = str(uuid.uuid4())
                return await asyncio.wait_for(
                    self.run_agent_trajectory_async(
                        idx,
                        application_id=application_id,
                        seed=seed,
                        mode=mode,
                        **kwargs,
                    ),
                    timeout=outer_timeout,
                )
            except Exception:
                traceback.print_exc()
                continue
        traceback.print_exc()
        raise Exception(
            f"Trajectory {idx} cannot complete. Please check the log message"
        )

    async def trajectory_generator(
        self, reset_seed=0, timing_raw=None, mode="Text", **kwargs
    ):
        if timing_raw is None:
            timing_raw = {}
        assert all(env is not None and isinstance(env, BaseEnv) for env in self.envs), (
            "All environments must be inheriting from BaseEnv"
        )
        assert all(env.is_multithread_safe() for env in self.envs), (
            "All environments must be multithread safe for async engine"
        )  # type: ignore
        max_concurrency = self.n_parallel_agents

        self.executor = ThreadPoolExecutor(max_workers=max_concurrency)

        if self.engine_name == "verl":
            await self.rollout_engine.wake_up()  # type: ignore

        semaphore = asyncio.Semaphore(self.n_parallel_agents)

        # Extract global progress info from meta_info if available (for validation)
        meta_info = kwargs.get("meta_info", {})
        val_progress = meta_info.get("val_progress", None) if meta_info else None
        global_offset = 0
        global_total = len(self.envs)
        batch_info = ""

        if val_progress:
            global_offset = val_progress.get("completed_trajectories", 0)
            global_total = val_progress.get("total_val_trajectories", len(self.envs))
            current_batch = val_progress.get("current_batch", 1)
            total_batches = val_progress.get("total_batches", 1)
            batch_info = f"[Batch {current_batch}/{total_batches}] "

        async def launch_one_trajectory_task(env_idx: int):
            async with semaphore:
                try:
                    result = await self.run_agent_trajectory_with_retry(
                        idx=env_idx,
                        seed=reset_seed,
                        mode=mode,
                        **kwargs,
                    )
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    raise e
                return result

        # Create all N conceptual tasks. Their execution will be throttled by the semaphore
        # and the availability of agent/env indices.
        tasks_to_run = [launch_one_trajectory_task(i) for i in range(len(self.envs))]

        tasks_completed = 0
        batch_size = len(self.envs)
        for coro in asyncio.as_completed(tasks_to_run):
            try:
                result = await coro
                tasks_completed += 1
                global_completed = global_offset + tasks_completed
                # Show both batch progress and global progress
                colorful_print(
                    f"{batch_info}Trajectories {tasks_completed}/{batch_size} (Global: {global_completed}/{global_total}) completed",
                    "cyan",
                )
                yield result
            except Exception as e:
                raise e

        if self.engine_name == "verl":
            await self.rollout_engine.sleep()  # type: ignore

        self.executor.shutdown(wait=False, cancel_futures=True)

    async def _append_progress_jsonl(self, task: dict, trajectory) -> None:
        """Append one completed trajectory to the progress jsonl (resume support).

        Writes a single line ``{"task_key": ..., "trajectory": <to_dict>}`` using
        the numpy-aware encoder (sokoban observations contain numpy arrays). The
        write is serialized by an asyncio.Lock so concurrent coroutines never
        interleave a half-line. Any IO/serialization error is logged and swallowed
        so a progress-write failure never aborts the eval itself.
        """
        if not self.progress_jsonl_path:
            return
        try:
            import json
            import os
            from rllm.eval.metrics import _NumpyEncoder, compute_task_key

            line = json.dumps(
                {"task_key": compute_task_key(task), "trajectory": trajectory.to_dict()},
                cls=_NumpyEncoder,
            )
            async with self._progress_write_lock:
                # The save_dir tree is only created by save_results() at the END
                # of eval, but progress is written DURING eval — so ensure the
                # parent dir exists first, otherwise every append fails with
                # ENOENT and resume silently never works.
                os.makedirs(os.path.dirname(self.progress_jsonl_path), exist_ok=True)
                with open(self.progress_jsonl_path, "a") as f:
                    f.write(line + "\n")
                    f.flush()
        except Exception as e:
            logging.warning(f"[rllm] Failed to append progress jsonl (non-fatal): {e}")

    async def execute_tasks(self, tasks: list[dict]):
        """
        Run asynchronous interactions between the agent and environment where each agent
        has its own environment instance and can proceed independently.

        Args:
            tasks: List of tasks to process
            max_concurrent: Maximum number of concurrent tasks to process (defaults to self.n_parallel_agents)

        Returns:
            A list of trajectories, one for each task.
        """
        if not hasattr(self, "executor") or self.executor._shutdown:
            self.executor = ThreadPoolExecutor(max_workers=self.max_env_workers)

        max_concurrent = self.n_parallel_agents

        # Lazily create the progress-write lock on the running event loop so
        # concurrent sem_wrapper coroutines never interleave jsonl appends.
        if self.progress_jsonl_path and self._progress_write_lock is None:
            self._progress_write_lock = asyncio.Lock()

        # Initialize results list to store trajectories for all tasks
        all_trajectories = {}

        # Create a queue of tasks to process
        task_queue = list(enumerate(tasks))
        semaphore = asyncio.Semaphore(max_concurrent)
        index_queue: asyncio.Queue[int] = asyncio.Queue(maxsize=max_concurrent)
        for i in range(max_concurrent):
            index_queue.put_nowait(i)

        # Track completed trajectories
        completed = 0
        total = len(tasks)

        async def sem_wrapper(task_id, task):
            nonlocal completed
            async with semaphore:
                # Get an available index
                index = await index_queue.get()
                try:
                    self.envs[index] = self.env_class.from_dict(
                        {**task, **self.env_args}
                    )
                    self.agents[index] = self.agent_class(**self.agent_args)
                    assert self.agents[index] is not None and isinstance(
                        self.agents[index], BaseAgent
                    ), "Agent is not initalized or not inheriting from BaseAgent"
                    self.agents[index].trajectory.task = task  # type: ignore
                    res = await self.run_agent_trajectory_async(
                        index, application_id=task_id
                    )
                    res.task = task
                    completed += 1
                    # Persist this completed trajectory immediately (resume support).
                    # Only successful trajectories reach here; a failed run raises
                    # before this point, so it is intentionally NOT written and will
                    # be re-run on the next resume.
                    await self._append_progress_jsonl(task, res)
                    colorful_print(
                        f"Progress: {completed}/{total} trajectories completed", "cyan"
                    )
                    return task_id, res
                finally:
                    # Put the index back in the queue when done
                    await index_queue.put(index)

        # Run all tasks concurrently
        results = await asyncio.gather(
            *[sem_wrapper(task_id, task) for task_id, task in task_queue]
        )

        all_trajectories = {task_id: trajectory for task_id, trajectory in results}
        ordered_trajectories = [
            all_trajectories[i] for i in range(len(all_trajectories))
        ]

        self.executor.shutdown(wait=False, cancel_futures=True)

        return ordered_trajectories

    def shutdown(self, keep_env_pool=False):
        if hasattr(self, "executor") and self.executor is not None:
            self.executor.shutdown()
            self.executor = None
        # The rollout engine is owned by the caller (trainer) and is intentionally
        # not shut down here.
        if not keep_env_pool and self.env_class is not None and hasattr(self.env_class, "shutdown_pool"):
            try:
                self.env_class.shutdown_pool()
            except Exception as e:
                logger.warning(f"Error shutting down env pool: {e}")


class AsyncAgentExecutionEngine(AgentExecutionEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
