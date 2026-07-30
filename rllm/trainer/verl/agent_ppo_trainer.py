import asyncio
import json
import math
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import reduce
from pprint import pprint
from queue import Queue
from threading import Thread

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    ResourcePoolManager,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.utils import Role, WorkerType
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics

from rllm.engine.agent_execution_engine import AsyncAgentExecutionEngine


def _compute_pass_at_k_unbiased(n: int, c: int, k: int) -> float:
    """Compute pass@k with the same unbiased estimator used by eval metrics."""
    if n < k or c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))


def _to_json_compatible(value):
    """Recursively convert NumPy containers/scalars into JSON-native values."""
    if isinstance(value, np.ndarray):
        return _to_json_compatible(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _to_json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_compatible(item) for item in value]
    return value


class AgentPPOTrainer(RayPPOTrainer):
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn=None,
        val_reward_fn=None,
        env_class=None,
        agent_class=None,
        env_args=None,
        agent_args=None,
    ):
        # verl expects data.train_files/val_files (plural); rllm uses train_file/val_file (singular).
        # Bridge the gap so verl's _create_dataloader can find the paths.
        train_file = config.data.get("train_file", None)
        val_file = config.data.get("val_file", None)
        if train_file and not config.data.get("train_files", None):
            with open_dict(config):
                config.data.train_files = train_file
        if val_file and not config.data.get("val_files", None):
            with open_dict(config):
                config.data.val_files = val_file

        super().__init__(config=config, tokenizer=tokenizer, role_worker_mapping=role_worker_mapping, resource_pool_manager=resource_pool_manager, ray_worker_group_cls=ray_worker_group_cls, reward_fn=reward_fn, val_reward_fn=val_reward_fn)
        self.env_class = env_class
        self.agent_class = agent_class
        self.env_args = env_args or {}
        self.agent_args = agent_args or {}

        assert self.config.actor_rollout_ref.hybrid_engine, "Only hybrid engine is supported"
        assert self.config.actor_rollout_ref.rollout.mode == "async", "Only async rollout mode is supported"

        if self.config.rllm.stepwise_advantage.enable:
            mode = self.config.rllm.stepwise_advantage.mode
            nbs = self.config.rllm.stepwise_advantage.normalize_by_steps
            nbl = self.config.rllm.stepwise_advantage.get("normalize_by_length", False)
            print(f"Using step-level advantage, max_prompt_length and max_response_length will be applied step-wise")
            print(f"  Stepwise advantage: mode={mode}, normalize_by_steps={nbs}, normalize_by_length={nbl}")
            if nbl:
                print(f"  normalize_by_length: reward divided by step count BEFORE GRPO (changes relative ordering)")
            if nbs:
                print(f"  normalize_by_steps: advantage divided by step count AFTER GRPO (preserves relative ordering)")
        else:
            print("Using trajectory-level advantage, max_prompt_length and max_response_length will be applied episode-wise")

    def init_workers(self):
        super().init_workers()

        engine_args = OmegaConf.to_container(self.config.rllm.agent.get("engine_args", {})) or {}
        n_parallel_agents = engine_args.pop("n_parallel_agents", None) or self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
        print(f"n_parallel_agents: {n_parallel_agents}")

        self.agent_execution_engine = AsyncAgentExecutionEngine(
            rollout_engine=self.async_rollout_manager,
            config=self.config,
            engine_name="verl",
            tokenizer=self.tokenizer,
            model_path=self.config.actor_rollout_ref.model.path,
            max_steps=self.config.rllm.agent.max_steps,
            max_response_length=self.config.data.max_response_length,
            max_prompt_length=self.config.data.max_prompt_length,
            agent_class=self.agent_class,
            agent_args=self.agent_args,
            env_class=self.env_class,
            env_args=self.env_args,
            enforce_max_prompt_length=(
                self.config.rllm.stepwise_advantage.enable
                and self.config.rllm.stepwise_advantage.get("mode", "per_step") != "stitched"
            ),
            trajectory_timeout=self.config.rllm.agent.trajectory_timeout,
            overlong_filter=self.config.rllm.agent.get("overlong_filter", False),
            disable_thinking=self.config.rllm.disable_thinking,
            n_parallel_agents=n_parallel_agents,
            **engine_args,
        )

    def init_envs_and_agents(self, batch):
        """
        Initialize environment depending on env_class with the necessary extra_info, also set uid of the batch.
        """
        assert self.agent_class is not None and self.env_class is not None, "Agent and environment classes must be provided"
        # Close any previously created envs to avoid leaking browser/process resources
        # across training steps (important for BrowserGym/Playwright envs).
        for env in getattr(self.agent_execution_engine, "envs", []):
            if env is None:
                continue
            try:
                env.close()
            except Exception as e:
                print(f"[rllm] Warning: failed to close env cleanly: {e}")

        env_args = batch.non_tensor_batch["extra_info"].tolist()

        full_agent_args = dict(self.config.rllm.agent.get("agent_args", {})) | self.agent_args
        base_env_args = dict(self.config.rllm.env.get("env_args", {})) | self.env_args

        # Pass validation mode flag so envs can adjust behavior (e.g., disable hints, skip solver)
        is_val = getattr(batch, "meta_info", None) or {}
        mode = is_val.get("mode") if isinstance(is_val, dict) else None
        if isinstance(mode, str) and mode.startswith("val"):
            print(f"[rllm] Validation mode detected, setting _is_validation to True")
            base_env_args["_is_validation"] = True
            # Skip expensive oracle advantage computation during validation —
            # it's only needed for training gradients, not for eval metrics.
            base_env_args["compute_oracle_advantage"] = False

        def _create_env(i):
            if isinstance(env_args[i], str):
                env_args[i] = json.loads(env_args[i])
            return i, self.env_class.from_dict({**env_args[i], **base_env_args})

        def _create_agent(i):
            return i, self.agent_class(**full_agent_args)

        # Create environments in parallel while preserving order
        envs = [None] * len(env_args)
        with ThreadPoolExecutor(max_workers=64) as executor:
            env_futures = [executor.submit(_create_env, i) for i in range(len(env_args))]
            for future in as_completed(env_futures):
                idx, env = future.result()
                envs[idx] = env

        # Create agents in parallel while preserving order
        agents = [None] * len(envs)
        with ThreadPoolExecutor(max_workers=64) as executor:
            agent_futures = [executor.submit(_create_agent, i) for i in range(len(envs))]
            for future in as_completed(agent_futures):
                idx, agent = future.result()
                agents[idx] = agent
        
        # Log environment and agent initialization summary
        print(f"[rllm] Initialized {len(envs)} environment instances and {len(agents)} agent instances")
        print(f"[rllm] Environment class: {self.env_class.__name__}, Agent class: {self.agent_class.__name__}")
        
        # Print pool stats if the environment class supports it
        if hasattr(self.env_class, 'print_pool_stats'):
            self.env_class.print_pool_stats()
        
        self.agent_execution_engine.update_envs_and_agents(envs, agents)
        return envs

    def _calculate_training_info(self):
        """
        Pre-calculate training information including total steps, trajectories, etc.
        Returns a dict with training statistics.
        """
        # Training parameters
        train_batch_size = self.config.data.train_batch_size
        rollout_n = self.config.actor_rollout_ref.rollout.n
        total_epochs = self.config.trainer.total_epochs
        rejection_enabled = self.config.rllm.rejection_sample.get("enable", False)
        rejection_multiplier = self.config.rllm.rejection_sample.get("multiplier", 1) if rejection_enabled else 1

        # gen_batch_size is the actual batch size used for data loading
        # It's typically set in config as: gen_batch_size = train_batch_size * rejection_multiplier
        # But can be overridden explicitly in the command line
        gen_batch_size = self.config.data.gen_batch_size

        # n_parallel_agents = train_batch_size * rollout_n
        n_parallel_agents = train_batch_size * rollout_n

        # Calculate steps per epoch based on train dataset size
        train_dataset_size = len(self.train_dataloader.dataset) if hasattr(self.train_dataloader, 'dataset') else 0
        steps_per_epoch = math.ceil(train_dataset_size / gen_batch_size) if gen_batch_size > 0 else 0

        # Total training steps
        total_training_steps = steps_per_epoch * total_epochs

        # Trajectories per step (after repeat with rollout.n)
        trajectories_per_step = gen_batch_size * rollout_n

        # Validation parameters
        val_batch_size = self.config.data.val_batch_size
        val_n = self.config.actor_rollout_ref.rollout.val_kwargs.n
        val_dataset_size = len(self.val_dataloader.dataset) if hasattr(self.val_dataloader, 'dataset') else 0
        val_batches = math.ceil(val_dataset_size / val_batch_size) if val_batch_size > 0 else 0
        total_val_trajectories = val_dataset_size * val_n

        return {
            "train_batch_size": train_batch_size,
            "gen_batch_size": gen_batch_size,
            "rollout_n": rollout_n,
            "rejection_enabled": rejection_enabled,
            "rejection_multiplier": rejection_multiplier,
            "n_parallel_agents": n_parallel_agents,
            "train_dataset_size": train_dataset_size,
            "steps_per_epoch": steps_per_epoch,
            "total_epochs": total_epochs,
            "total_training_steps": total_training_steps,
            "trajectories_per_step": trajectories_per_step,
            "val_batch_size": val_batch_size,
            "val_n": val_n,
            "val_dataset_size": val_dataset_size,
            "val_batches": val_batches,
            "total_val_trajectories": total_val_trajectories,
        }

    def _print_training_plan(self, info: dict):
        """Print a summary of the training plan."""
        print("\n" + "=" * 70)
        print("  TRAINING PLAN SUMMARY")
        print("=" * 70)
        print(f"  Training Dataset Size:     {info['train_dataset_size']} examples")
        print(f"  Validation Dataset Size:   {info['val_dataset_size']} examples")
        print(f"  ")
        print(f"  --- Training Configuration ---")
        print(f"  train_batch_size:          {info['train_batch_size']}")
        print(f"  gen_batch_size:            {info['gen_batch_size']} (batch size for data loading)")
        print(f"  rejection_sample.enable:   {info['rejection_enabled']} (multiplier={info['rejection_multiplier']})")
        print(f"  rollout.n:                 {info['rollout_n']} (trajectories per prompt)")
        print(f"  n_parallel_agents:         {info['n_parallel_agents']} (train_batch_size × rollout.n)")
        print(f"  trajectories_per_step:     {info['trajectories_per_step']} (gen_batch_size × rollout.n)")
        print(f"  ")
        print(f"  --- Training Progress ---")
        print(f"  Steps per Epoch:           {info['steps_per_epoch']}")
        print(f"  Total Epochs:              {info['total_epochs']}")
        print(f"  Total Training Steps:      {info['total_training_steps']}")
        print(f"  ")
        print(f"  --- Validation Configuration ---")
        print(f"  val_batch_size:            {info['val_batch_size']}")
        print(f"  val_kwargs.n:              {info['val_n']} (trajectories per prompt)")
        print(f"  val_batches:               {info['val_batches']}")
        print(f"  total_val_trajectories:    {info['total_val_trajectories']} (val_dataset × val_kwargs.n)")
        print("=" * 70 + "\n")

    def fit_agent(self):
        """
        The training loop of PPO. Adapted to train the underlying model of agent.
        """
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # Pre-calculate and display training info
        self.training_info = self._calculate_training_info()
        self._print_training_plan(self.training_info)

        # perform validation before training
        import time

        start_time = time.time()
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate_agent()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return
        print(f"Time taken to validate agent: {time.time() - start_time}")
        # we start from step 1
        self.global_steps += 1

        for epoch in range(self.config.trainer.total_epochs):
            pprint(f"epoch {epoch}, step {self.global_steps} started")
            step_in_epoch = 0
            for batch_dict in self.train_dataloader:
                step_in_epoch += 1
                total_steps = self.training_info["total_training_steps"]
                steps_per_epoch = self.training_info["steps_per_epoch"]
                trajectories_per_step = self.training_info["trajectories_per_step"]

                print("\n" + "=" * 70)
                print(f"  TRAINING STEP {self.global_steps}/{total_steps} (Epoch {epoch+1}/{self.config.trainer.total_epochs}, Step {step_in_epoch}/{steps_per_epoch})")
                print(f"  Generating {trajectories_per_step} trajectories this step")
                print("=" * 70)

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                batch = batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n,
                    interleave=True,
                )

                metrics = {}
                timing_raw = {}

                batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])

                with marked_timer("step", timing_raw):
                    self.init_envs_and_agents(batch)

                    stepwise_mode = self.config.rllm.stepwise_advantage.get("mode", None)
                    if self.config.rllm.stepwise_advantage.enable and stepwise_mode not in ("stitched",):
                        final_gen_batch_output, generate_metrics = self.generate_agent_steps(timing_raw=timing_raw, meta_info=batch.meta_info, uids=batch.non_tensor_batch["uid"])
                        metrics.update(generate_metrics)
                        repeat_counts = final_gen_batch_output.meta_info["repeat_counts"]
                        # need to repeat to make shape match
                        batch = batch.sample_level_repeat(repeat_counts)
                        final_gen_batch_output.meta_info.pop("repeat_counts", None)  # no longer needed after this
                        # batch needs to be padded to divisor of world size, we will pad with everything masked out
                        batch = batch.union(final_gen_batch_output)
                        batch = self._pad_dataproto_to_world_size(batch=batch)
                    else:
                        final_gen_batch_output, generate_metrics = self.generate_agent_trajectory(timing_raw=timing_raw, meta_info=batch.meta_info)
                        batch = batch.union(final_gen_batch_output)
                        metrics.update(generate_metrics)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw):
                        # compute scores using reward model and/or reward function
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # reward tensor for env-based trajectory data can be obtained by processing the trajectories
                        if "token_level_scores" not in batch.batch:
                            reward_tensor = self.reward_fn(batch)
                            batch.batch["token_level_scores"] = reward_tensor
                        else:
                            reward_tensor = batch.batch["token_level_scores"]  # filled in by environment collected trajectory transformation

                        # Rejection sampling based on rewards
                        # Group rewards by uid
                        uids = batch.non_tensor_batch["uid"]
                        unique_uids = np.unique(uids)
                        valid_mask = torch.ones(len(uids), dtype=torch.bool)
                        solve_none = 0
                        solve_all = 0
                        for uid in unique_uids:
                            uid_mask = uids == uid
                            uid_rewards = reward_tensor[uid_mask].sum(-1)  # Sum rewards for each sequence

                            # Check if all rewards are <= 0 or all are 1 >= for this uid
                            if (uid_rewards <= 0).all():
                                valid_mask[uid_mask] = False
                                solve_none += 1
                            elif (uid_rewards >= 1).all():
                                valid_mask[uid_mask] = False
                                solve_all += 1

                        # Log to metrics
                        metrics["batch/solve_none"] = solve_none
                        metrics["batch/solve_all"] = solve_all
                        metrics["batch/solve_partial"] = len(unique_uids) - solve_none - solve_all

                        if self.config.rllm.rejection_sample.enable:
                            # log the actual complete training rewards before rejection sampling
                            token_level_rewards = None  # for metrics calculation
                            if self.config.rllm.stepwise_advantage.enable:
                                is_pad_step = batch.non_tensor_batch["is_pad_step"]
                                non_pad_step_indices = np.where(is_pad_step == False)[0]
                                non_pad_steps = batch.select_idxs(non_pad_step_indices)
                                is_last_step = non_pad_steps.non_tensor_batch["is_last_step"]
                                valid_last_step_indices = np.where(is_last_step == True)[0]
                                last_step_batch = batch.select_idxs(valid_last_step_indices)
                                token_level_rewards = last_step_batch.batch["token_level_scores"]
                            else:
                                token_level_rewards = batch.batch["token_level_scores"]
                            full_sequence_score = token_level_rewards.sum(-1)
                            metrics["critic/full-score/mean"] = torch.mean(full_sequence_score).detach().item()
                            metrics["critic/full-score/max"] = torch.max(full_sequence_score).detach().item()
                            metrics["critic/full-score/min"] = torch.min(full_sequence_score).detach().item()

                            # If no valid samples remain, skip this batch and get a new one
                            if not valid_mask.any():
                                continue

                            # Filter batch to keep only valid samples
                            batch = batch[valid_mask]

                            if self.config.rllm.stepwise_advantage.enable and self.config.rllm.stepwise_advantage.mode == "broadcast":
                                # batch now only contains steps with valid uids
                                # filter out padding steps
                                is_pad_step = batch.non_tensor_batch["is_pad_step"]
                                non_pad_step_indices = np.where(is_pad_step == False)[0]
                                batch = batch.select_idxs(non_pad_step_indices)  # This batch only has non_pad steps

                                # need to make sure both number of last steps (number of uids) and number of total steps in the batch (batch size after processing) are all multiples of world size
                                # separate out last step and intermediate steps
                                is_last_step = batch.non_tensor_batch["is_last_step"]
                                valid_last_step_indices = np.where(is_last_step == True)[0]
                                not_last_step_indices = np.where(is_last_step == False)[0]
                                last_step_batch = batch.select_idxs(valid_last_step_indices)  # This batch only has valid last steps
                                non_last_step_batch = batch.select_idxs(not_last_step_indices)

                                # filter last_step_batch to make sure its multiple of world size
                                num_trainer_replicas = self.actor_rollout_wg.world_size
                                max_batch_size = (
                                    last_step_batch.batch["input_ids"].shape[0]  # 1 per trajectory
                                    // num_trainer_replicas
                                ) * num_trainer_replicas
                                if not max_batch_size:
                                    # give up, you got everything either all wrong or right.
                                    continue

                                size_mask = torch.zeros(last_step_batch.batch["input_ids"].shape[0], dtype=torch.bool)
                                size_mask[:max_batch_size] = True
                                last_step_batch = last_step_batch[size_mask]  # filtered last steps

                                # now we go through all the non_last_step_batch and keep everything that has same idxs that exists in the filtered last steps
                                valid_last_step_idxs = last_step_batch.non_tensor_batch["idxs"]
                                non_last_step_idxs = non_last_step_batch.non_tensor_batch["idxs"]
                                non_last_step_mask = np.isin(non_last_step_idxs, valid_last_step_idxs)
                                non_last_step_batch = non_last_step_batch[non_last_step_mask]

                                # concatenate then pad
                                batch = DataProto.concat([last_step_batch, non_last_step_batch])
                                batch = self._pad_dataproto_to_world_size(batch)
                            else:
                                # Round down to the nearest multiple of world size
                                num_trainer_replicas = self.actor_rollout_wg.world_size
                                max_batch_size = (batch.batch["input_ids"].shape[0] // num_trainer_replicas) * num_trainer_replicas
                                if not max_batch_size:
                                    # give up, you got everything either all wrong or right.
                                    continue

                                size_mask = torch.zeros(batch.batch["input_ids"].shape[0], dtype=torch.bool)
                                size_mask[:max_batch_size] = True
                                batch = batch[size_mask]

                        # recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)

                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                rollout_old_log_probs = batch.batch["rollout_log_probs"]
                                actor_old_log_probs = batch.batch["old_log_probs"]
                                attention_mask = batch.batch["attention_mask"]
                                responses = batch.batch["responses"]
                                response_length = responses.size(1)
                                response_mask = attention_mask[:, -response_length:]

                                rollout_probs = torch.exp(rollout_old_log_probs)
                                actor_probs = torch.exp(actor_old_log_probs)
                                rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                                rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                                rollout_probs_diff_max = torch.max(rollout_probs_diff)
                                rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                                rollout_probs_diff_std = torch.std(rollout_probs_diff)
                                metrics.update(
                                    {
                                        "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                        "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                        "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                    }
                                )

                        if self.use_reference_policy:
                            # compute reference log_prob
                            with marked_timer("ref", timing_raw):
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)

                        # compute rewards with KL penalty if needed

                        # Note: This kl penalty applied directly over the rewards is disabled for GRPO. The kl penalty is applied at dp_actor.py
                        # where it is subtracted directly from the policy loss

                        # if not self.config.actor_rollout_ref.actor.use_kl_loss:
                        #     batch, kl_metrics = apply_kl_penalty(batch,
                        #                                        kl_ctrl=self.kl_ctrl,
                        #                                        kl_penalty=self.config.algorithm.kl_penalty)
                        #     metrics.update(kl_metrics)
                        # else:
                        #     batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        if self.config.rllm.stepwise_advantage.enable:
                            if self.config.rllm.stepwise_advantage.mode == "per_step":
                                batch.batch["token_level_rewards"] = batch.batch["mc_returns"]
                                batch.non_tensor_batch["uid"] = batch.non_tensor_batch["step_ids"]

                                # normalize_by_length: divide reward by step count at reward level (before GRPO)
                                if self.config.rllm.stepwise_advantage.get("normalize_by_length", False):
                                    self._normalize_rewards_by_length(batch)

                                is_pad_step = batch.non_tensor_batch["is_pad_step"]
                                non_pad_step_indices = np.where(is_pad_step == False)[0]
                                batch = batch.select_idxs(non_pad_step_indices)  # This batch only has non_pad steps
                            elif self.config.rllm.stepwise_advantage.mode == "broadcast":
                                # In case of step-wise advantage broadcast, we would split out the final steps, then merge again
                                is_last_step = batch.non_tensor_batch["is_last_step"]
                                last_step_indices = np.where(is_last_step == True)[0]
                                other_step_indices = np.where(is_last_step == False)[0]
                                other_step_batch = batch.select_idxs(other_step_indices)
                                batch = batch.select_idxs(last_step_indices)  # This batch only has last steps

                                # normalize_by_length: divide reward by step count at reward level (before GRPO)
                                if self.config.rllm.stepwise_advantage.get("normalize_by_length", False):
                                    self._normalize_rewards_by_length(batch)
                            elif self.config.rllm.stepwise_advantage.mode == "stitched":
                                pass  # stitched mode: no splitting, oracle advantages already per-token
                            else:
                                raise ValueError(f"Stepwise advantage mode {self.config.rllm.stepwise_advantage.mode} not supported")

                        # --- Oracle Advantage path ---
                        # When the batch carries oracle_advantages (e.g., from solver-based
                        # environments like Sokoban), use them directly instead of GRPO/GAE.
                        # Set oracle_adv_mapping_mode="off" (the default) to skip oracle and
                        # use standard GRPO. Registered modes: "cast" / "cast_ablate".
                        mode = self.config.algorithm.get("oracle_adv_mapping_mode", "off")
                        if "oracle_advantages" in batch.batch and mode != "off":
                            from rllm.trainer.verl.oracle_advantage_mapping import apply_mapping

                            batch.batch["advantages"] = batch.batch["oracle_advantages"]
                            batch.batch["returns"] = torch.zeros_like(batch.batch["oracle_advantages"])
                            mapped_advs = apply_mapping(
                                mode=mode,
                                oracle_advs=batch.batch["advantages"],
                                response_mask=batch.batch["response_mask"],
                                uids=batch.non_tensor_batch.get("uid", np.array([])),
                                token_level_scores=batch.batch["token_level_scores"],
                                turn_advantages=batch.non_tensor_batch.get("turn_advantages", np.array([])),
                                rollout_n=self.config.actor_rollout_ref.rollout.n,
                                config=self.config.algorithm,
                            )
                            batch.batch["advantages"] = mapped_advs

                            # Metrics
                            mask = batch.batch["response_mask"].bool()
                            valid_advs = mapped_advs[mask]
                            if valid_advs.numel() > 0:
                                metrics["adv_oracle/mapped_mean"] = valid_advs.mean().item()
                                metrics["adv_oracle/mapped_max"] = valid_advs.max().item()
                                metrics["adv_oracle/mapped_min"] = valid_advs.min().item()
                                metrics["adv_oracle/pos_frac"] = (valid_advs > 0).float().mean().item()
                                metrics["adv_oracle/neg_frac"] = (valid_advs < 0).float().mean().item()

                            # CAST alpha-ablation diagnostics.
                            # Read-only; computed before the stepwise-broadcast concat below so the
                            # B rows of mapped_advs / uids / turn_advantages stay index-aligned.
                            try:
                                from rllm.trainer.verl.oracle_diagnostics import compute_cast_diagnostics

                                metrics.update(
                                    compute_cast_diagnostics(
                                        mode=mode,
                                        mapped_advs=mapped_advs,
                                        response_mask=batch.batch["response_mask"],
                                        token_level_scores=batch.batch["token_level_scores"],
                                        uids=batch.non_tensor_batch.get("uid", np.array([])),
                                        turn_advantages=batch.non_tensor_batch.get("turn_advantages", np.array([])),
                                        config=self.config.algorithm,
                                    )
                                )
                            except Exception as e:
                                print(f"[rllm] Warning: compute_cast_diagnostics failed: {e}")

                            traj_rewards = batch.batch["token_level_scores"].sum(dim=-1)
                            n_success = (traj_rewards > 0).sum().item()
                            print(f"\n[Oracle Mapping] mode={mode} | success: {n_success}/{traj_rewards.shape[0]}")

                            # Handle step-wise broadcast merging for oracle path
                            if self.config.rllm.stepwise_advantage.enable and self.config.rllm.stepwise_advantage.mode == "broadcast":
                                self._stepwise_advantage_broadcast(batch, other_step_batch=other_step_batch)
                                batch = DataProto.concat([batch, other_step_batch])

                        else:
                            # --- Standard GRPO/GAE advantage path ---
                            # compute advantages, executed on the driver process
                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=self.config.algorithm.norm_adv_by_std_in_grpo,
                                config=self.config.algorithm,
                            )

                            if self.config.rllm.stepwise_advantage.enable and self.config.rllm.stepwise_advantage.mode == "broadcast":
                                # remove the padded last steps
                                # Merging the separated out steps using the advantage from last steps
                                self._stepwise_advantage_broadcast(batch, other_step_batch=other_step_batch)
                                batch = DataProto.concat([batch, other_step_batch])

                    # Log trajectories with FINAL advantages (post-mapping) to jsonl
                    if "advantages" in batch.batch and "chat_completions" in batch.non_tensor_batch:
                        self._log_trajectories_with_advantages(batch, mode="train")

                    if self.config.rllm.mask_truncated_samples:
                        mask = batch.batch["attention_mask"][:, -1] == 1
                        batch = batch[~mask]

                    batch = self._pad_dataproto_to_world_size(batch=batch)
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and self.global_steps % self.config.trainer.test_freq == 0:
                        with marked_timer("testing", timing_raw):
                            val_metrics: dict = self._validate_agent()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (
                        self.global_steps % self.config.trainer.save_freq == 0
                        or self.global_steps >= self.total_training_steps
                    ):
                        try:
                            with marked_timer("save_checkpoint", timing_raw):
                                self._save_checkpoint()
                        except OSError as e:
                            print(f"[rllm] Warning: failed to save checkpoint at step {self.global_steps}: {e}")

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:
                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate_agent()
                        pprint(f"Final validation metrics: {val_metrics}")
                        logger.log(data=val_metrics, step=self.global_steps)
                    return

    def _save_checkpoint(self):
        """Save a full checkpoint, then relocate the HF model to a flat sibling dir.

        verl natively writes a full HF model into ``global_step_N/actor/huggingface``
        when ``hf_model`` is in ``actor.checkpoint.save_contents``. That subdir lives
        *inside* the rotated full-checkpoint dir, so with ``max_actor_ckpt_to_keep``
        rotation it would be deleted along with the optimizer shards.

        To keep HF models accumulating while full checkpoints roll, we move
        ``global_step_N/actor/huggingface`` out to ``global_step_N_hf_model/``, a flat
        sibling directory containing a standard HF model. Resume never reads
        ``huggingface/`` — it loads the per-rank ``*.pt`` shards — so relocating it
        does not affect resume.

        After relocation we also prune hollow ``global_step_M`` shells: verl's
        ``max_actor_ckpt_to_keep`` rotation deletes only the ``actor/`` subdir, leaving
        the parent ``global_step_M`` dir behind with a stale ``data.pt`` dataloader
        snapshot. Those shells accumulate every save step. A dir is pruned ONLY if it
        still has NO ``actor/`` subdir — i.e. rotation already emptied it — so the live,
        resumable checkpoint (the one that still owns ``actor/``) is never touched.

        This runs only on the driver (single process), so there is no multi-rank
        race. Any failure is logged and swallowed: the full checkpoint is already on
        disk, so training/resume is never compromised by an HF relocation error.
        """
        super()._save_checkpoint()

        save_contents = (
            self.config.actor_rollout_ref.actor.get("checkpoint", {}).get("save_contents", [])
        )
        if "hf_model" not in save_contents:
            return

        import re
        import shutil

        root = self.config.trainer.default_local_dir
        step_dir = os.path.join(root, f"global_step_{self.global_steps}")
        src = os.path.join(step_dir, "actor", "huggingface")
        dst = os.path.join(root, f"global_step_{self.global_steps}_hf_model")
        try:
            if not os.path.isdir(src):
                print(f"[rllm] Warning: expected HF model dir not found at {src}; skipping relocation")
                return
            # Remove a stale destination (e.g. from a crashed prior attempt) so the
            # move lands cleanly. This only ever touches an *_hf_model OUTPUT dir.
            if os.path.exists(dst):
                shutil.rmtree(dst, ignore_errors=True)
            # os.replace is atomic on the same filesystem; shutil.move falls back to
            # a copy+delete across devices.
            try:
                os.replace(src, dst)
            except OSError:
                shutil.move(src, dst)
            print(f"[rllm] Relocated HF model: {src} -> {dst}")
        except Exception as e:
            print(f"[rllm] Warning: failed to relocate HF model to {dst}: {e}")

        # Prune hollow global_step shells left by actor-only rotation. Match EXACTLY
        # global_step_<int> (never *_hf_model). Skip the current step and any dir that
        # still owns an actor/ subdir (= a live resumable checkpoint).
        shell_re = re.compile(r"^global_step_(\d+)$")
        try:
            for name in os.listdir(root):
                m = shell_re.match(name)
                if not m or int(m.group(1)) == self.global_steps:
                    continue
                cand = os.path.join(root, name)
                if not os.path.isdir(cand):
                    continue
                if os.path.isdir(os.path.join(cand, "actor")):
                    continue  # live full checkpoint — never delete
                shutil.rmtree(cand, ignore_errors=True)
                print(f"[rllm] Pruned hollow checkpoint shell: {cand}")
        except Exception as e:
            print(f"[rllm] Warning: failed while pruning hollow checkpoint shells: {e}")

    def _validate_agent(self):
        rewards_lst = []
        success_lst = []  # Track success boolean from environment
        score_lst = []  # Track score from environment (e.g., 2048 game score)
        data_source_lst = []
        uid_lst = []

        # Calculate validation progress info
        val_batch_size = self.config.data.val_batch_size
        val_n = self.config.actor_rollout_ref.rollout.val_kwargs.n
        val_dataset_size = len(self.val_dataloader.dataset) if hasattr(self.val_dataloader, 'dataset') else 0
        total_val_batches = math.ceil(val_dataset_size / val_batch_size) if val_batch_size > 0 else 0
        total_val_trajectories = val_dataset_size * val_n

        print("\n" + "-" * 70)
        print(f"  VALIDATION: {val_dataset_size} examples × {val_n} samples = {total_val_trajectories} total trajectories")
        print(f"  Processing in {total_val_batches} batches (val_batch_size={val_batch_size})")
        print("-" * 70)

        # Clear the val chat_completions file at the start of validation so
        # that appending across batches doesn't mix with a previous run.
        save_dir = os.path.join(self.config.trainer.default_local_dir, "chat_completions")
        os.makedirs(save_dir, exist_ok=True)
        val_filename = os.path.join(save_dir, f"val_{self.global_steps}.jsonl")
        open(val_filename, "w").close()

        completed_trajectories = 0
        current_batch = 0

        for test_data in self.val_dataloader:
            current_batch += 1
            test_batch = DataProto.from_single_dict(test_data)
            batch_examples = len(test_batch.batch)
            batch_trajectories = batch_examples * val_n

            print(f"\n[VAL Batch {current_batch}/{total_val_batches}] Processing {batch_examples} examples × {val_n} = {batch_trajectories} trajectories")
            print(f"[VAL Progress] Trajectories: {completed_trajectories}/{total_val_trajectories} completed so far")

            test_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object)
            n_val_samples = self.config.actor_rollout_ref.rollout.val_kwargs.n
            test_batch = test_batch.repeat(repeat_times=n_val_samples, interleave=True)
            test_batch.pop(["input_ids", "attention_mask", "position_ids"])  # these are not needed for environment based interaction
            test_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": False,
                "validate": True,
                "mode": "val",  # Mark as validation for trajectory saving
                "val_filename": val_filename,
                # Pass global progress info for trajectory_generator
                "val_progress": {
                    "completed_trajectories": completed_trajectories,
                    "total_val_trajectories": total_val_trajectories,
                    "current_batch": current_batch,
                    "total_batches": total_val_batches,
                },
            }
            self.init_envs_and_agents(test_batch)

            if self.config.rllm.stepwise_advantage.enable:
                test_output_gen_batch, _ = self.generate_agent_steps(meta_info=test_batch.meta_info, uids=test_batch.non_tensor_batch["uid"])
                # for validation, we only need the last step
                is_last_step = test_output_gen_batch.non_tensor_batch["is_last_step"]
                last_step_indices = np.where(is_last_step == True)[0]
                test_output_gen_batch = test_output_gen_batch.select_idxs(last_step_indices)  # This batch only has last steps
            else:
                test_output_gen_batch, _ = self.generate_agent_trajectory(meta_info=test_batch.meta_info)

            test_batch = test_batch.union(test_output_gen_batch)

            reward_tensor = test_batch.batch["token_level_scores"]
            success_array = test_batch.non_tensor_batch.get("success", np.array([None] * reward_tensor.shape[0]))
            score_array = test_batch.non_tensor_batch.get("score", np.array([None] * reward_tensor.shape[0]))

            # Update completed count
            completed_trajectories += batch_trajectories
            print(f"[VAL Batch {current_batch}/{total_val_batches}] Completed. Total progress: {completed_trajectories}/{total_val_trajectories} trajectories")

            rewards_lst.append(reward_tensor.sum(-1).cpu())
            success_lst.append(success_array)  # Collect success information
            score_lst.append(score_array)  # Collect score information
            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))
            uid_lst.append(test_batch.non_tensor_batch["uid"])

        print(f"\n[VAL COMPLETE] All {total_val_trajectories} validation trajectories finished")
        print("-" * 70 + "\n")

        reward_tensor = torch.cat(rewards_lst, dim=0)  # (batch_size,)
        success_array = np.concatenate(success_lst, axis=0)  # (batch_size,)
        score_array = np.concatenate(score_lst, axis=0)  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        uid_tensor = np.concatenate(uid_lst, axis=0)

        # Collect rewards, success, and scores by data source
        data_source_rewards = {}
        data_source_success = {}  # Track success boolean
        data_source_scores = {}  # Track scores (e.g., 2048 game score)
        data_source_uid_rewards = {}  # all rewards per uid for pass@k
        data_source_uid_success = {}  # all success flags per uid for pass@k
        data_source_uid_scores = {}  # for max score per uid

        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            uid = uid_tensor[i]
            reward = reward_tensor[i].item()
            success = success_array[i]
            score = score_array[i]

            # Collect all rewards, success, and scores for pass@1 and mean metrics
            if data_source not in data_source_rewards:
                data_source_rewards[data_source] = []
                data_source_success[data_source] = []
                data_source_scores[data_source] = []
            data_source_rewards[data_source].append(reward)
            data_source_success[data_source].append(success)
            if score is not None:
                data_source_scores[data_source].append(score)

            # Collect per-uid rewards/success for pass@k computation.
            if data_source not in data_source_uid_rewards:
                data_source_uid_rewards[data_source] = {}
                data_source_uid_success[data_source] = {}
                data_source_uid_scores[data_source] = {}
            if uid not in data_source_uid_rewards[data_source]:
                data_source_uid_rewards[data_source][uid] = [reward]
                data_source_uid_success[data_source][uid] = [success]
                if score is not None:
                    data_source_uid_scores[data_source][uid] = score
            else:
                data_source_uid_rewards[data_source][uid].append(reward)
                data_source_uid_success[data_source][uid].append(success)
                if score is not None and uid in data_source_uid_scores[data_source]:
                    data_source_uid_scores[data_source][uid] = max(
                        data_source_uid_scores[data_source][uid], score
                    )
                elif score is not None:
                    data_source_uid_scores[data_source][uid] = score

        metric_dict = {}

        # Determine if we should use success-based metrics
        env_name = self.config.rllm.env.name if hasattr(self.config.rllm, 'env') else 'unknown'
        env_name_lower = env_name.lower()
        supports_success = ('alfworld' in env_name_lower or 'sokoban' in env_name_lower or 'twenty_forty_eight' in env_name_lower or '2048' in env_name_lower or 'minesweeper' in env_name_lower or 'sudoku' in env_name_lower or 'webshop' in env_name_lower or 'rush_hour' in env_name_lower)
        use_success_metrics = any(s is not None for s in success_array) and supports_success

        if use_success_metrics:
            print(f"[VAL] Using success-based metrics (info['success']) for environment: {env_name}")
        else:
            print(f"[VAL] Using reward threshold metrics for environment: {env_name}")
            # Fallback to reward threshold if success info not available
            if 'sokoban' in env_name.lower():
                success_threshold = 5.0  # Sokoban finish bonus is 10.0
            else:
                success_threshold = 1.0  # Default for binary success environments

        for data_source, rewards in data_source_rewards.items():
            rewards_array = np.array(rewards)
            success_data = data_source_success[data_source]
            scores_data = data_source_scores.get(data_source, [])

            # Mean reward (unclipped, to see actual performance)
            metric_dict[f"val/reward_mean/{data_source}"] = np.mean(rewards_array)

            # Score metrics for environments like 2048
            if scores_data:
                scores_array = np.array(scores_data)
                metric_dict[f"val/score_mean/{data_source}"] = np.mean(scores_array)
                metric_dict[f"val/score_max/{data_source}"] = np.max(scores_array)
                metric_dict[f"val/score_min/{data_source}"] = np.min(scores_array)

                # 2048-specific score thresholds
                if 'twenty_forty_eight' in env_name_lower or '2048' in env_name_lower:
                    metric_dict[f"val/score_ge_512/{data_source}"] = np.mean(scores_array >= 512)
                    metric_dict[f"val/score_ge_1024/{data_source}"] = np.mean(scores_array >= 1024)
                    metric_dict[f"val/score_ge_2048/{data_source}"] = np.mean(scores_array >= 2048)

            # pass@1: proportion of all trajectories that succeeded
            if use_success_metrics:
                # Use success boolean from environment
                print("Using success metrics to calculate pass@1")
                success_count = sum(1 for s in success_data if s is True)
                metric_dict[f"val/test_score/pass@1/{data_source}"] = success_count / len(success_data) if success_data else 0.0
            else:
                # Fallback to reward threshold
                print("Using reward threshold to calculate pass@1")
                metric_dict[f"val/test_score/pass@1/{data_source}"] = np.mean(rewards_array >= success_threshold)

        # pass@k:
        # - val/test_score/pass@k/* keeps legacy "any success in k attempts" behavior
        # - val/test_score/pass@{2,4,...}/* adds unbiased pass@k to align with eval
        for data_source in data_source_rewards.keys():
            uid_rewards = data_source_uid_rewards[data_source]
            uid_success = data_source_uid_success[data_source]
            problem_total = {uid: len(rewards) for uid, rewards in uid_rewards.items()}

            if use_success_metrics:
                print("Using success metrics to calculate pass@k")
                # c should be the number of correct samples for each problem(uid),
                # not just a binary solved flag.
                problem_correct_counts = {
                    uid: sum(1 for s in success_list if s is True)
                    for uid, success_list in uid_success.items()
                }
            else:
                print("Using reward threshold to calculate pass@k")
                # c should be the number of correct samples for each problem(uid),
                # not just a binary solved flag.
                problem_correct_counts = {
                    uid: sum(1 for r in reward_list if r >= success_threshold)
                    for uid, reward_list in uid_rewards.items()
                }

            # Legacy pass@k metric: fraction of problems solved at least once.
            pass_at_k_any = (
                float(np.mean([c > 0 for c in problem_correct_counts.values()]))
                if problem_correct_counts
                else 0.0
            )
            metric_dict[f"val/test_score/pass@k/{data_source}"] = pass_at_k_any

            max_samples_per_problem = max(problem_total.values()) if problem_total else 0
            k = 2
            while k <= max_samples_per_problem:
                values = []
                for uid, n in problem_total.items():
                    if n < k:
                        continue
                    c = problem_correct_counts[uid]
                    values.append(_compute_pass_at_k_unbiased(n=n, c=c, k=k))
                metric_dict[f"val/test_score/pass@{k}/{data_source}"] = (
                    float(np.mean(values)) if values else 0.0
                )
                k *= 2

        return metric_dict

    def generate_agent_trajectory(self, timing_raw=None, meta_info=None):
        """
        Generates agent trajectories by interacting with the environment. Does not close or reset the environment afterwards

        Args:
            envs: The environments in which the agent interacts.
            agents: The agents to use for interation.
            timing_raw: Dictionary to store timing information for profiling.
            meta_info (optional): Metadata for veRL generation.

        Returns:
            DataProto: Representation of the agent's trajectories.
            Dict[str:float]: Metrics for the generation process.
        """
        if timing_raw is None:
            timing_raw = {}
        with marked_timer("collect_trajectory", timing_raw):
            trajectories = []
            if self.async_rollout_mode:
                gen_seq_generator = self.generate_agent_trajectories_async(timing_raw=timing_raw, meta_info=meta_info, mode="Token")
                for _, trajectory in enumerate(gen_seq_generator):
                    trajectories.append(trajectory)
            else:
                raise ValueError("Only async rollout mode is supported")
        # Sort trajectories by their idx, to ensure they are in order.
        trajectories.sort(key=lambda x: x["idx"])

        with marked_timer("transform_trajectory", timing_raw):
            # Transform the raw trajectories into DataProto format.
            # Get mode from meta_info, default to "train"
            mode = meta_info.get("mode", "train") if meta_info else "train"
            final_gen_batch_output, metrics = self._transform_agent_trajectories(
                trajectories,
                mode=mode,
                meta_info=meta_info,
            )
        return final_gen_batch_output, metrics

    def generate_agent_steps(self, timing_raw=None, meta_info=None, uids=None):
        """
        Generates agent trajectories by interacting with the environment. Does not close or reset the environment afterwards.

        Returns:
            Tuple of (DataProto, dict): Representation of the agent's trajectories and generation metrics.
        """
        if timing_raw is None:
            timing_raw = {}
        if uids is None:
            uids = []
        with marked_timer("collect_trajectory", timing_raw):
            steps = []
            gen_seq_generator = self.generate_agent_trajectories_async(timing_raw=timing_raw, meta_info=meta_info, mode="Step")
            for _, trajectory in enumerate(gen_seq_generator):
                steps.append(trajectory)
        # Sort trajectories by their idx, to ensure they are in order.
        steps.sort(key=lambda x: x["idx"])

        with marked_timer("transform_trajectory", timing_raw):
            # Transform the raw trajectories into DataProto format.
            mode = meta_info.get("mode", "train") if meta_info else "train"
            final_gen_batch_output, generate_metrics = self._transform_agent_steps(steps, uids=uids, mode=mode, meta_info=meta_info)
        return final_gen_batch_output, generate_metrics

    def _log_trajectories_with_advantages(self, batch: DataProto, mode: str = "train"):
        """Log trajectories to jsonl with the FINAL advantages used for policy updates.

        Called after oracle mapping so the logged values match what the actor sees.
        """
        try:
            save_dir = os.path.join(self.config.trainer.default_local_dir, "chat_completions")
            os.makedirs(save_dir, exist_ok=True)
            is_val_mode = isinstance(mode, str) and mode.startswith("val")
            filename = f"val_{self.global_steps}.jsonl" if is_val_mode else f"{self.global_steps}.jsonl"
            file_mode = "a" if is_val_mode else "w"

            chat_completions = batch.non_tensor_batch.get("chat_completions", [])
            turn_advs_list = batch.non_tensor_batch.get("turn_advantages", [])
            traj_rewards = batch.batch["token_level_scores"].sum(dim=-1)
            final_advs = batch.batch["advantages"]
            response_mask = batch.batch["response_mask"]

            with open(os.path.join(save_dir, filename), file_mode) as f:
                for i in range(len(chat_completions)):
                    # Extract per-token final advantages (only valid/masked positions)
                    mask_i = response_mask[i].bool()
                    token_advs = final_advs[i][mask_i].tolist()
                    raw_turn = turn_advs_list[i] if i < len(turn_advs_list) and turn_advs_list[i] is not None else []
                    turn_advs = _to_json_compatible(raw_turn)
                    reward = float(traj_rewards[i].item())
                    traj_data = {
                        "chat_completions": chat_completions[i],
                        "reward": reward,
                        "solved": reward >= 1.0,
                        "mode": mode,
                        "step": self.global_steps,
                        "turn_advantages_raw": turn_advs,
                        "token_advantages_final": token_advs,
                    }
                    f.write(json.dumps(_to_json_compatible(traj_data)) + "\n")
        except OSError as e:
            print(f"[rllm] Warning: failed to save chat completions to disk: {e}")

    def _transform_agent_trajectories(
        self,
        trajectories: list[dict],
        mode: str = "train",
        meta_info: dict | None = None,
    ):
        """
        Helper function to transform a list of trajectories into tokenized DataProto format.

        Args:
            trajectories (list of dict): List of trajectories to process.
            mode (str): Mode of the trajectories, either "train" or "val". Defaults to "train".

        Returns:
            DataProto: A structured dataset containing input tokens, masks, and rewards.
        """
        from verl.utils.torch_functional import pad_sequence_to_length

        all_initial_tokens_list = []
        all_response_tokens_list = []
        all_masks_list = []
        all_oracle_advs_list = []
        all_turn_advs_list = []
        traj_scores = []
        traj_success = []  # Track success boolean from environment
        traj_env_scores = []  # Track env score from environment info (e.g., 2048 score).
        chat_completions = []
        traj_metrics = []
        metrics = {}

        for traj in trajectories:
            prompt_tokens = traj["prompt_tokens"]
            response_tokens = traj["response_tokens"]
            # test if trajectory is empty
            assert prompt_tokens.numel() != 0 and response_tokens.numel() != 0, f"Both prompt {prompt_tokens.numel()} and response {response_tokens.numel()} of trajectory shouldn't be empty. Please check make sure environment is working and the config"
            all_initial_tokens_list.append(prompt_tokens)
            all_response_tokens_list.append(response_tokens)
            all_masks_list.append(traj["response_masks"])
            # Collect oracle advantages from solver-based environments
            traj_oracle_advs = traj.get("oracle_advantages")
            if traj_oracle_advs is not None:
                if not isinstance(traj_oracle_advs, torch.Tensor):
                    traj_oracle_advs = torch.tensor(traj_oracle_advs, dtype=torch.float32)
                assert not torch.isnan(traj_oracle_advs).any(), (
                    f"oracle_advantages contains NaN! Check solver return values."
                )
                assert traj_oracle_advs.shape[0] == response_tokens.shape[0], (
                    f"oracle_advantages length ({traj_oracle_advs.shape[0]}) != "
                    f"response_tokens length ({response_tokens.shape[0]}). "
                    f"In stitched mode, the Engine layer must broadcast each turn's scalar "
                    f"advantage to all tokens of that turn before passing to Trainer."
                )
                all_oracle_advs_list.append(traj_oracle_advs)
            all_turn_advs_list.append(traj.get("turn_advantages"))
            traj_scores.append(traj["trajectory_reward"])
            traj_success.append(traj.get("success", None))  # Extract success from trajectory
            traj_env_scores.append(traj.get("score", None))  # Extract score from trajectory
            chat_completions.append(traj["chat_completions"])
            traj_metrics.append(traj["metrics"])

        # Flatten traj_metrics into a dict of lists
        traj_metrics = {k: [d[k] for d in traj_metrics] for k in traj_metrics[0]}
        # Aggregate metrics (mean, min, max)
        for k, v_list in traj_metrics.items():
            v_list = [v for v in v_list if v is not None and v >= 0]
            if not v_list:
                continue
            v_list = np.array(v_list)
            metrics.update(
                {
                    f"traj/{k}_mean": v_list.mean(),
                    f"traj/{k}_min": v_list.min(),
                    f"traj/{k}_max": v_list.max(),
                }
            )

        # Save chat completions to a file with metadata
        try:
            save_dir = os.path.join(self.config.trainer.default_local_dir, "chat_completions")
            os.makedirs(save_dir, exist_ok=True)
            is_val_mode = isinstance(mode, str) and mode.startswith("val")
            # Validation can pass explicit output filename (e.g. dual val with suffix).
            explicit_val_filename = (
                meta_info.get("val_filename")
                if isinstance(meta_info, dict) and meta_info.get("val_filename")
                else None
            )
            if is_val_mode:
                filename = (
                    os.path.basename(explicit_val_filename)
                    if explicit_val_filename
                    else f"val_{self.global_steps}.jsonl"
                )
            else:
                filename = f"{self.global_steps}.jsonl"
            # Validation is processed in multiple batches, so append ("a") to
            # accumulate all batches.  The file is truncated at the start of
            # _validate_agent so stale data from a previous run is cleared.
            # Training writes each step exactly once, so "w" is fine.
            file_mode = "a" if is_val_mode else "w"
            with open(os.path.join(save_dir, filename), file_mode) as f:
                for i, chat_completion in enumerate(chat_completions):
                    traj_data = {
                        "chat_completions": chat_completion,
                        "reward": float(traj_scores[i]),
                        "solved": float(traj_scores[i]) >= 1.0,
                        "mode": mode,
                        "step": self.global_steps,
                    }
                    f.write(json.dumps(traj_data) + "\n")
        except OSError as e:
            print(f"[rllm] Warning: failed to save chat completions to disk: {e}")

        # left pad prompts
        max_prompt_length = self.config.data.max_prompt_length
        prompts_batch = torch.nn.utils.rnn.pad_sequence(
            [torch.flip(i, dims=[0]) for i in all_initial_tokens_list],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        ).flip(dims=[1])
        prompts_batch = pad_sequence_to_length(prompts_batch, max_prompt_length, self.tokenizer.pad_token_id, left_pad=True)
        prompts_batch = prompts_batch[:, -max_prompt_length:]

        # right pad responses
        max_response_length = self.config.data.max_response_length
        response_batch = torch.nn.utils.rnn.pad_sequence(
            all_response_tokens_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        response_batch = pad_sequence_to_length(response_batch, max_response_length, self.tokenizer.pad_token_id, left_pad=False)
        response_batch = response_batch[:, :max_response_length]

        # input_ids
        trajectory_batch = torch.concat([prompts_batch, response_batch], dim=1)

        # attention mask
        prompt_lengths = torch.as_tensor([len(t) for t in all_initial_tokens_list]).clamp_(min=0, max=max_prompt_length)
        prompt_pos = torch.arange(max_prompt_length).unsqueeze(0)
        prompt_mask = prompt_pos >= (max_prompt_length - prompt_lengths.unsqueeze(1))

        response_lengths = torch.as_tensor([len(t) for t in all_response_tokens_list]).clamp_(min=0, max=max_response_length)
        resp_pos = torch.arange(max_response_length).unsqueeze(0)
        response_mask = resp_pos < response_lengths.unsqueeze(1)

        attention_mask = torch.cat([prompt_mask, response_mask], dim=1).long()

        # loss mask
        traj_mask = torch.nn.utils.rnn.pad_sequence(all_masks_list, batch_first=True, padding_value=0)
        traj_mask = pad_sequence_to_length(traj_mask, max_response_length, 0, left_pad=False)
        traj_mask = traj_mask[:, :max_response_length]

        # Oracle advantage batch (if available from solver-based envs)
        if all_oracle_advs_list:
            oracle_adv_batch = torch.nn.utils.rnn.pad_sequence(all_oracle_advs_list, batch_first=True, padding_value=0.0)
            oracle_adv_batch = pad_sequence_to_length(oracle_adv_batch, max_response_length, 0.0, left_pad=False)
            oracle_adv_batch = oracle_adv_batch[:, :max_response_length]
        else:
            oracle_adv_batch = None

        # position_ids
        position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

        # Place all rewards to last response token (e.g., eos token)
        score_batch = torch.zeros_like(response_batch, dtype=torch.float32)

        for i, score in enumerate(traj_scores):
            resp_len = response_lengths[i]
            if resp_len > 0 and resp_len <= score_batch.shape[1]:
                score_batch[i, resp_len - 1] = score

        tensor_batch = {
            "input_ids": trajectory_batch,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": response_batch,
            "prompts": prompts_batch,
            "token_level_scores": score_batch,
            "response_mask": traj_mask,
        }
        if oracle_adv_batch is not None:
            tensor_batch["oracle_advantages"] = oracle_adv_batch

        # Add success information to non_tensor_batch
        non_tensor_batch = {
            "success": np.array(traj_success, dtype=object),  # Store success boolean
            "score": np.array(traj_env_scores, dtype=object),  # Store final env score
            "turn_advantages": np.array(all_turn_advs_list, dtype=object),
            "chat_completions": np.array(chat_completions, dtype=object),
        }

        # Log train-time score metrics so environments like 2048 are observable
        # during optimization instead of only in validation/evaluation.
        numeric_scores = [
            float(s)
            for s in traj_env_scores
            if isinstance(s, (int, float, np.number)) and not isinstance(s, bool)
        ]
        if numeric_scores:
            scores_array = np.array(numeric_scores, dtype=np.float32)
            if mode == "train":
                metrics["train/score_mean"] = float(np.mean(scores_array))
                metrics["train/score_max"] = float(np.max(scores_array))
                metrics["train/score_min"] = float(np.min(scores_array))

                env_name = self.config.rllm.env.name if hasattr(self.config.rllm, "env") else ""
                env_name_lower = env_name.lower()
                if "twenty_forty_eight" in env_name_lower or "2048" in env_name_lower:
                    metrics["train/score_ge_512"] = float(np.mean(scores_array >= 512))
                    metrics["train/score_ge_1024"] = float(np.mean(scores_array >= 1024))
                    metrics["train/score_ge_2048"] = float(np.mean(scores_array >= 2048))

        self.visualize_trajectory(DataProto.from_dict(tensors=tensor_batch, non_tensors=non_tensor_batch))

        return DataProto.from_dict(tensors=tensor_batch, non_tensors=non_tensor_batch), metrics

    def visualize_trajectory(self, tensor_batch, sample_idx=0, max_samples=1, mask_key="response_mask"):
        """
        Visualize the trajectory from tensor_batch using the shared visualization utility.
        """
        from rllm.utils.visualization import visualize_trajectories

        if len(tensor_batch) == 0:
            return

        end_idx = min(sample_idx + max_samples, len(tensor_batch))
        indices = list(range(sample_idx, end_idx))

        visualize_trajectories(
            batch=tensor_batch,
            tokenizer=self.tokenizer,
            sample_indices=indices,
            mask_key=mask_key,
            reward_key="token_level_scores",
            show_workflow_metadata=False,
        )

    def generate_agent_trajectories_async(self, timing_raw=None, meta_info=None, mode="Token"):
        """
        Generates agent trajectories asynchronously using the agent execution engine.

        This method runs the asynchronous `trajectory_generator` in a
        separate thread and yields the results synchronously through a queue.
        This allows the main training loop (which might be synchronous) to consume
        asynchronously generated trajectories.

        Args:
            timing_raw (dict, optional): Dictionary to store timing information. Defaults to {}.
            meta_info (dict, optional): Additional metadata for the generation process. Defaults to None.

        Yields:
            Any: Items generated by the `trajectory_generator`, typically
                 representing parts or results of agent trajectories in token format.
        """
        if timing_raw is None:
            timing_raw = {}
        queue = Queue()

        def runner():
            async def consume():
                async for item in self.agent_execution_engine.trajectory_generator(timing_raw=timing_raw, mode=mode, meta_info=meta_info):
                    queue.put(item)
                queue.put(None)  # sentinel to signal done

            asyncio.run(consume())

        Thread(target=runner, daemon=True).start()
        while True:
            item = queue.get()
            if item is None:
                break
            yield item

    def _transform_agent_steps(self, steps: list[dict], uids: np.ndarray, mode: str = "train", meta_info: dict | None = None):
        from verl.utils.torch_functional import pad_sequence_to_length

        overlong_filter = self.config.rllm.agent.get("overlong_filter", False)
        overlong_reasons = {"TRUNCATION", "MAX_STEPS", "TIMEOUT"}

        all_prompts_list = []
        all_responses_list = []

        step_numbers = []  # number of steps of each episode, 0 indexed
        all_steps_idx_list = []
        all_steps_is_last_step_list = []
        all_steps_step_num = []  # total number of steps the trajectory this step belongs to have
        all_steps_step_ids = []
        all_steps_masked_out = []  # whether this step should be masked out due to overlong filter
        all_steps_scores = []  # final env score repeated on each step (used by last-step val metrics)
        all_steps_success = []  # success boolean repeated on each step
        all_steps_is_pad = []  # whether this step is a dummy pad (from empty-response episodes)
        training_rewards = []
        all_mc_returns = []  # Monte Carlo returns for each episode
        all_oracle_advs = []  # per-step oracle advantages from solver
        # the last step will have reward assigned and be used for advantage calculation
        chat_completions = []
        traj_metrics = []
        metrics = {}

        for episode in steps:
            episode_steps = episode["steps"]
            idx = episode["idx"]
            training_reward = episode["trajectory_reward"]
            mc_returns = episode["mc_returns"]
            termination_reason = episode.get("termination_reason")
            episode_score = episode.get("score", None)
            episode_success = episode.get("success", None)

            # Mask out overlong trajectories
            masked_out = overlong_filter and termination_reason in overlong_reasons

            kept_step_indices = []  # track which step indices (within episode) were kept
            for step_i, s in enumerate(episode_steps):
                # Use original token IDs from model generation when available.
                # This avoids BPE re-tokenization divergence (autoregressive generation
                # can produce non-canonical BPE sequences that differ from canonical
                # re-tokenization) and preserves stop tokens stripped by skip_special_tokens.
                if s.get("prompt_ids") is not None:
                    prompt_ids = torch.tensor(s["prompt_ids"], dtype=torch.long)
                elif s.get("prompt") is not None:
                    prompt_ids = torch.tensor(
                        self.tokenizer.encode(s["prompt"], add_special_tokens=False),
                        dtype=torch.long,
                    )
                else:
                    raise ValueError(
                        "Step has neither prompt_ids nor prompt text; cannot build training input"
                    )

                if s.get("completion_ids") is not None:
                    resp_ids = torch.tensor(s["completion_ids"], dtype=torch.long)                    
                elif s.get("response") is not None:
                    resp_ids = torch.tensor(
                        self.tokenizer.encode(s["response"], add_special_tokens=False),
                        dtype=torch.long,
                    )
                else:
                    resp_ids = torch.tensor([], dtype=torch.long)

                # Guard against empty responses — they contribute nothing to training
                if resp_ids.numel() == 0:
                    print(
                        f"[rllm] Warning: Step for episode idx={idx} has empty response, skipping"
                    )
                    continue

                all_prompts_list.append(prompt_ids)
                all_responses_list.append(resp_ids)
                kept_step_indices.append(step_i)

            n_kept = len(kept_step_indices)
            if n_kept == 0:
                # All steps had empty responses. We still need to emit exactly one
                # padded step so that repeat_counts stays in 1:1 correspondence
                # with the caller's batch rows (sample_level_repeat requirement).
                print(f"[rllm] Warning: All steps for episode idx={idx} had empty responses, inserting pad step")
                # Create a single pad_token prompt + single pad_token response
                all_prompts_list.append(torch.tensor([self.tokenizer.pad_token_id], dtype=torch.long))
                all_responses_list.append(torch.tensor([self.tokenizer.pad_token_id], dtype=torch.long))
                step_numbers.append(0)  # 0-indexed: 1 step total
                training_rewards.append(training_reward)
                all_mc_returns.append(0.0)
                chat_completions.append(episode.get("chat_completions", []))
                if episode.get("metrics"):
                    traj_metrics.append(episode["metrics"])
                all_steps_idx_list.append(idx)
                all_steps_is_last_step_list.append(True)
                all_steps_step_num.append(1)
                all_steps_step_ids.append(f"{uids[idx]}_step_pad")
                all_steps_masked_out.append(True)  # fully masked out — contributes nothing to loss
                all_steps_is_pad.append(True)
                all_steps_scores.append(episode_score)
                all_steps_success.append(episode_success)
                continue

            step_numbers.append(n_kept - 1)
            training_rewards.append(training_reward)
            # Only keep mc_returns for steps that were actually kept
            all_mc_returns.extend([mc_returns[si] for si in kept_step_indices])
            # Collect oracle advantages per step (from solver-based environments)
            episode_oracle_advs = episode.get("oracle_advantages", [])
            if episode_oracle_advs:
                all_oracle_advs.extend([episode_oracle_advs[si] for si in kept_step_indices])
            else:
                all_oracle_advs.extend([0.0] * n_kept)
            chat_completions.append(episode.get("chat_completions", []))
            if episode.get("metrics"):
                traj_metrics.append(episode["metrics"])

            all_steps_idx_list.extend([idx for _ in range(n_kept)])
            all_steps_is_last_step_list.extend([False for _ in range(n_kept)])
            all_steps_is_last_step_list[-1] = True

            all_steps_step_num.extend([n_kept for _ in range(n_kept)])
            all_steps_step_ids.extend([f"{uids[idx]}_step{si}" for si in kept_step_indices])
            all_steps_masked_out.extend([masked_out for _ in range(n_kept)])
            all_steps_is_pad.extend([False for _ in range(n_kept)])
            all_steps_scores.extend([episode_score for _ in range(n_kept)])
            all_steps_success.extend([episode_success for _ in range(n_kept)])

        # Track prompt truncation for monitoring.
        max_prompt_length = self.config.data.max_prompt_length
        max_response_length = self.config.data.max_response_length
        prompt_truncation_count = 0
        for t in all_prompts_list:
            if len(t) > max_prompt_length:
                prompt_truncation_count += 1
        if prompt_truncation_count > 0:
            print(
                f"[rllm] Warning: {prompt_truncation_count}/{len(all_prompts_list)} step prompts "
                f"exceeded max_prompt_length={max_prompt_length} and were left-truncated"
            )
        metrics["traj/prompt_truncation_count"] = prompt_truncation_count
        metrics["traj/prompt_truncation_rate"] = prompt_truncation_count / max(len(all_prompts_list), 1)

        # left pad prompts
        prompts_batch = torch.nn.utils.rnn.pad_sequence(
            [torch.flip(i, dims=[0]) for i in all_prompts_list],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        ).flip(dims=[1])
        prompts_batch = pad_sequence_to_length(prompts_batch, max_prompt_length, self.tokenizer.pad_token_id, left_pad=True)
        prompts_batch = prompts_batch[:, -max_prompt_length:]

        # right pad responses
        response_batch = torch.nn.utils.rnn.pad_sequence(
            all_responses_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        response_batch = pad_sequence_to_length(response_batch, max_response_length, self.tokenizer.pad_token_id, left_pad=False)
        response_batch = response_batch[:, :max_response_length]

        # input_ids
        complete_step_batch = torch.concat([prompts_batch, response_batch], dim=1)

        # attention mask
        prompt_lengths = torch.as_tensor([len(t) for t in all_prompts_list]).clamp_(min=0, max=max_prompt_length)
        prompt_pos = torch.arange(max_prompt_length).unsqueeze(0)
        prompt_mask = prompt_pos >= (max_prompt_length - prompt_lengths.unsqueeze(1))

        response_lengths = torch.as_tensor([len(t) for t in all_responses_list]).clamp_(min=0, max=max_response_length)
        resp_pos = torch.arange(max_response_length).unsqueeze(0)
        response_mask = resp_pos < response_lengths.unsqueeze(1)

        attention_mask = torch.cat([prompt_mask, response_mask], dim=1).long()

        # loss mask — build explicit response mask from actual completion_ids lengths.
        # This mirrors the non-stepwise path's use of explicit response_masks from
        # assemble_steps(). Each step's response is purely model-generated, so mask=1
        # for all non-padding response tokens up to the actual completion length.
        # This is more robust than inferring from attention_mask[:, max_prompt_length:]
        # because it's computed directly from the source-of-truth response lengths.
        traj_mask = torch.zeros_like(response_batch, dtype=torch.long)
        
        for i in range(len(all_responses_list)):
            resp_len = min(len(all_responses_list[i]), max_response_length)
            traj_mask[i, :resp_len] = 1
        # Zero out masked steps: either overlong-filtered trajectories (when enabled)
        # or pad steps inserted for episodes with all-empty responses (always).
        masked_out_tensor = torch.tensor(all_steps_masked_out, dtype=torch.bool).unsqueeze(1)
        if overlong_filter or masked_out_tensor.any():
            traj_mask = traj_mask * (~masked_out_tensor).long()

        # position_ids
        position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

        # Place all rewards to last response token of each step
        score_batch = torch.zeros_like(response_batch, dtype=torch.float32)
        oracle_adv_batch = torch.zeros_like(response_batch, dtype=torch.float32)
        mc_return_batch = torch.zeros_like(response_batch, dtype=torch.float32)

        step_index = 0
        for i, traj_score in enumerate(training_rewards):
            step_num = step_numbers[i] + 1  # since step_numbers is 0 indexed
            for _ in range(step_num):
                resp_len = response_lengths[step_index]
                if resp_len > 0 and resp_len <= score_batch.shape[1]:
                    score_batch[step_index, resp_len - 1] = traj_score
                    mc_return_batch[step_index, resp_len - 1] = all_mc_returns[step_index]
                    # Oracle advantage: broadcast to all response tokens of this step
                    if all_oracle_advs and step_index < len(all_oracle_advs):
                        oracle_adv_batch[step_index, :resp_len] = all_oracle_advs[step_index]
                step_index += 1
        assert step_index == score_batch.shape[0], f"Number of total steps used should equal to batch size, but got {step_index} and {score_batch.shape[0]}"

        tensor_batch = {
            "input_ids": complete_step_batch,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": response_batch,
            "prompts": prompts_batch,
            "token_level_scores": score_batch,
            "mc_returns": mc_return_batch,
            "response_mask": traj_mask,
        }
        if all_oracle_advs:
            tensor_batch["oracle_advantages"] = oracle_adv_batch

        batch_id = str(uuid.uuid4())
        non_tensor_batch = {
            "idxs": np.array(all_steps_idx_list),
            "step_nums": np.array(all_steps_step_num),
            "is_last_step": np.array(all_steps_is_last_step_list),
            "is_pad_step": np.array(all_steps_is_pad),
            "batch_id": np.array([batch_id for _ in range(len(all_steps_idx_list))]),  # in case need to differentiate which iteration the step is coming from
            "step_ids": np.array(all_steps_step_ids),
            "score": np.array(all_steps_scores, dtype=object),
            "success": np.array(all_steps_success, dtype=object),  # Store success boolean from environment
        }

        meta_info_out = {"repeat_counts": [x + 1 for x in step_numbers]}

        result = DataProto.from_dict(tensors=tensor_batch, non_tensors=non_tensor_batch, meta_info=meta_info_out)

        # Find indices of last steps for visualization
        last_step_indices = [i for i, is_last in enumerate(non_tensor_batch["is_last_step"]) if is_last]
        if last_step_indices:
            sample_indices = np.random.choice(last_step_indices, size=min(2, len(last_step_indices)), replace=False)
            for idx in sample_indices:
                self.visualize_trajectory(result, sample_idx=idx, max_samples=1)

        # Aggregate traj_metrics (same as _transform_agent_trajectories)
        if traj_metrics:
            traj_metrics_dict = {k: [d[k] for d in traj_metrics] for k in traj_metrics[0]}
            for k, v_list in traj_metrics_dict.items():
                v_list = [v for v in v_list if v is not None and v >= 0]
                if not v_list:
                    continue
                v_list = np.array(v_list)
                metrics.update(
                    {
                        f"traj/{k}_mean": v_list.mean(),
                        f"traj/{k}_min": v_list.min(),
                        f"traj/{k}_max": v_list.max(),
                    }
                )

        # Save chat completions to a file with metadata (same logic as _transform_agent_trajectories)
        try:
            save_dir = os.path.join(self.config.trainer.default_local_dir, "chat_completions")
            os.makedirs(save_dir, exist_ok=True)
            is_val_mode = isinstance(mode, str) and mode.startswith("val")
            explicit_val_filename = (
                meta_info.get("val_filename")
                if isinstance(meta_info, dict) and meta_info.get("val_filename")
                else None
            )
            if is_val_mode:
                filename = (
                    os.path.basename(explicit_val_filename)
                    if explicit_val_filename
                    else f"val_{self.global_steps}.jsonl"
                )
            else:
                filename = f"{self.global_steps}.jsonl"
            file_mode = "a" if is_val_mode else "w"
            with open(os.path.join(save_dir, filename), file_mode) as f:
                for i, chat_completion in enumerate(chat_completions):
                    traj_data = {
                        "chat_completions": chat_completion,
                        "reward": float(training_rewards[i]),
                        "solved": float(training_rewards[i]) >= 1.0,
                        "mode": mode,
                        "step": self.global_steps,
                    }
                    f.write(json.dumps(traj_data) + "\n")
        except OSError as e:
            print(f"[rllm] Warning: failed to save chat completions to disk: {e}")

        # Log train-time score metrics (same logic as _transform_agent_trajectories)
        # Use per-episode scores (one per trajectory) instead of per-step scores
        episode_scores = [steps[i].get("score", None) for i in range(len(steps))]
        numeric_scores = [
            float(s)
            for s in episode_scores
            if isinstance(s, (int, float, np.number)) and not isinstance(s, bool)
        ]
        if numeric_scores:
            scores_array = np.array(numeric_scores, dtype=np.float32)
            if mode == "train":
                metrics["train/score_mean"] = float(np.mean(scores_array))
                metrics["train/score_max"] = float(np.max(scores_array))
                metrics["train/score_min"] = float(np.min(scores_array))

                env_name = self.config.rllm.env.name if hasattr(self.config.rllm, "env") else ""
                env_name_lower = env_name.lower()
                if "twenty_forty_eight" in env_name_lower or "2048" in env_name_lower:
                    metrics["train/score_ge_512"] = float(np.mean(scores_array >= 512))
                    metrics["train/score_ge_1024"] = float(np.mean(scores_array >= 1024))
                    metrics["train/score_ge_2048"] = float(np.mean(scores_array >= 2048))

        return result, metrics

    def _stepwise_advantage_broadcast(self, last_step_batch, other_step_batch):
        """
        Broadcast the advantage from last_step_batch to all other steps.
        """

        # NOTE: Currently takes the average of advantages. For GRPO, advantage and returns is uniform for each token so this makes no difference.
        # NOTE: For simplicity, assumes advantage and return is the same, which also holds for GRPO variants
        if "response_mask" not in other_step_batch.batch.keys():
            other_step_batch.batch["response_mask"] = compute_response_mask(other_step_batch)
        if "response_mask" not in last_step_batch.batch.keys():
            last_step_batch.batch["response_mask"] = compute_response_mask(last_step_batch)
        src_indices = last_step_batch.non_tensor_batch["idxs"]
        src_total_steps = last_step_batch.non_tensor_batch["step_nums"]
        tgt_indices = other_step_batch.non_tensor_batch["idxs"]
        src_advantages = last_step_batch.batch["advantages"]
        src_mask = last_step_batch.batch["response_mask"]
        tgt_mask = other_step_batch.batch["response_mask"]

        # Build idx -> scalar advantage
        idx_to_scalar_adv = {}
        for i, idx in enumerate(src_indices):
            mask = src_mask[i].bool()
            scalar = src_advantages[i][mask].mean()

            if self.config.rllm.stepwise_advantage.normalize_by_steps:
                # normalize the advantage against number of steps
                scalar = scalar / src_total_steps[i]
                # reassign the normalized advantage to last_step_batch as well
                last_step_batch.batch["advantages"][i][mask] = scalar

            idx_to_scalar_adv[int(idx)] = scalar

        # Create new tensor for other_step_batch with per-token assignment
        scalar_rows = torch.stack([torch.full_like(tgt_mask[i], fill_value=idx_to_scalar_adv[int(idx)], dtype=torch.float32) for i, idx in enumerate(tgt_indices)])  # shape: (N2, T)

        # Apply the response mask of the target batch
        final_advantage = scalar_rows * tgt_mask

        # Assignment
        other_step_batch.batch["advantages"] = final_advantage
        other_step_batch.batch["returns"] = final_advantage

    def _normalize_rewards_by_length(self, batch):
        """Normalize token_level_rewards by dividing by the trajectory step count.

        This implements verl-agent's normalize_by_length: score = episode_reward / N_steps.
        It modifies token_level_rewards in-place BEFORE compute_advantage() so that GRPO
        sees per-step-averaged rewards, which changes the relative ordering within groups
        (shorter trajectories with the same total reward get higher advantage).

        Unlike normalize_by_steps (which divides the advantage AFTER GRPO and preserves
        relative ordering), normalize_by_length operates at the reward level and actively
        favors trajectories that solve the task in fewer steps.
        """
        step_nums = batch.non_tensor_batch["step_nums"].astype(np.float64)
        step_nums_tensor = torch.tensor(
            step_nums,
            dtype=batch.batch["token_level_rewards"].dtype,
            device=batch.batch["token_level_rewards"].device,
        )
        # Avoid division by zero
        step_nums_tensor = step_nums_tensor.clamp(min=1.0)
        # token_level_rewards shape: (batch_size, response_length)
        # Divide all values (only last token is non-zero) by step count
        batch.batch["token_level_rewards"] = batch.batch["token_level_rewards"] / step_nums_tensor.unsqueeze(-1)

    def _pad_dataproto_to_world_size(self, batch):
        world_sizes = []
        if self.use_critic and self.critic_wg.world_size != 0:
            world_sizes.append(self.critic_wg.world_size)
        if self.use_reference_policy and self.ref_policy_wg.world_size != 0:
            world_sizes.append(self.ref_policy_wg.world_size)
        if self.use_rm and self.rm_wg.world_size != 0:
            world_sizes.append(self.rm_wg.world_size)
        if self.hybrid_engine:
            if self.actor_rollout_wg.world_size != 0:
                world_sizes.append(self.actor_rollout_wg.world_size)
        else:
            if self.actor_wg.world_size != 0:
                world_sizes.append(self.actor_wg.world_size)
            if self.rollout_wg.world_size != 0:
                world_sizes.append(self.rollout_wg.world_size)
        if not world_sizes:
            return batch

        world_size = reduce(math.lcm, world_sizes)

        original_batch_size = batch.batch["prompts"].shape[0]
        batch, pad_size = pad_dataproto_to_divisor(batch, world_size)

        # for the padded dataproto, make the traj mask to 0. is_last_step also False
        for i in range(pad_size):
            idx = original_batch_size + i
            if "is_last_step" in batch.non_tensor_batch:
                batch.non_tensor_batch["is_last_step"][idx] = False
            if "is_pad_step" in batch.non_tensor_batch:
                batch.non_tensor_batch["is_pad_step"][idx] = True

        return batch

    def shutdown(self):
        if hasattr(self, "agent_execution_engine") and self.agent_execution_engine is not None:
            self.agent_execution_engine.shutdown()
            self.agent_execution_engine = None
