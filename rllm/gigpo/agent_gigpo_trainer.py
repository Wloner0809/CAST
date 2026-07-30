"""
GiGPO Trainer — AgentPPOTrainer subclass with GiGPO advantage estimation.

Implements the GiGPO algorithm (https://arxiv.org/abs/2505.10978) by overriding
only the advantage computation and step transformation methods. The rest of the
training loop (rollout, actor update, validation, etc.) is inherited unchanged
from AgentPPOTrainer.

Design principle: NO modifications to agent_ppo_trainer.py. All GiGPO logic
lives in this file and rllm/gigpo/core_gigpo.py.
"""

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

from rllm.gigpo import core_gigpo
from rllm.trainer.verl.agent_ppo_trainer import AgentPPOTrainer


class AgentGiGPOTrainer(AgentPPOTrainer):
    """PPO trainer with GiGPO advantage estimation.

    Overrides:
    - ``_transform_agent_steps``: adds GiGPO-required fields to non_tensor_batch
      (anchor_obs, rewards, traj_uid, active_masks).
    - ``fit_agent``: replaces the GRPO/PPO advantage computation with GiGPO
      joint advantage (episode-level + step-level).
    """

    # ------------------------------------------------------------------
    # Override _transform_agent_steps to inject GiGPO metadata
    # ------------------------------------------------------------------

    def _transform_agent_steps(self, steps, uids=None, mode="train", meta_info=None):
        """Call parent, then append GiGPO-specific non_tensor_batch fields."""
        result, metrics = super()._transform_agent_steps(
            steps, uids=uids, mode=mode, meta_info=meta_info,
        )

        # Build per-step arrays aligned with the DataProto rows.
        # Each row in the DataProto corresponds to a single agent step.
        all_anchor_obs = []
        all_step_rewards = []
        all_traj_uids = []
        all_active_masks = []

        for episode in steps:
            episode_steps = episode["steps"]
            idx = episode["idx"]
            step_rewards_list = episode.get("step_rewards", [])
            # Unique trajectory ID. ``idx`` is the per-row env index (unique per
            # trajectory), while ``uids[idx]`` is the PROMPT-level uid SHARED across
            # the N rollouts of one prompt. Appending ``idx`` keeps the episode group
            # readable while guaranteeing each of the N rollouts gets a distinct
            # traj_uid — otherwise compute_step_discounted_returns would pool all N
            # rollouts into one trajectory and corrupt the discounted returns (Eq. 5).
            if uids is not None and idx < len(uids):
                traj_uid = f"{uids[idx]}_traj_{idx}"
            else:
                traj_uid = str(uuid.uuid4())

            n_kept = 0
            for step_i, s in enumerate(episode_steps):
                # Check if this step was actually kept (has non-empty response)
                if s.get("completion_ids") is not None:
                    resp_ids = s["completion_ids"]
                elif s.get("response") is not None:
                    resp_ids = s["response"]
                else:
                    resp_ids = []
                # Match the parent's empty-response skip logic
                if isinstance(resp_ids, list) and len(resp_ids) == 0:
                    continue
                if hasattr(resp_ids, '__len__') and len(resp_ids) == 0:
                    continue

                anchor_obs = s.get("anchor_obs", "")
                step_reward = step_rewards_list[step_i] if step_i < len(step_rewards_list) else 0.0
                all_anchor_obs.append(anchor_obs if anchor_obs is not None else "")
                all_step_rewards.append(float(step_reward))
                all_traj_uids.append(traj_uid)
                all_active_masks.append(1.0)
                n_kept += 1

            # Handle the pad step case (all steps had empty responses)
            if n_kept == 0:
                all_anchor_obs.append("")
                all_step_rewards.append(0.0)
                all_traj_uids.append(traj_uid)
                all_active_masks.append(0.0)  # inactive pad

        # Sanity check: arrays must match DataProto batch size
        batch_size = result.batch["input_ids"].shape[0]
        assert len(all_anchor_obs) == batch_size, (
            f"GiGPO anchor_obs count ({len(all_anchor_obs)}) != batch_size ({batch_size})"
        )

        result.non_tensor_batch["anchor_obs"] = np.array(all_anchor_obs, dtype=object)
        result.non_tensor_batch["rewards"] = np.array(all_step_rewards, dtype=np.float64)
        result.non_tensor_batch["traj_uid"] = np.array(all_traj_uids, dtype=object)
        result.non_tensor_batch["active_masks"] = np.array(all_active_masks, dtype=np.float64)

        return result, metrics

    # ------------------------------------------------------------------
    # Override _transform_agent_trajectories — inject GiGPO metadata
    # ------------------------------------------------------------------

    def _transform_agent_trajectories(self, trajectories, mode="train", meta_info=None):
        """Call parent, then append GiGPO per-step metadata for trajectory-level GiGPO."""
        result, metrics = super()._transform_agent_trajectories(
            trajectories, mode=mode, meta_info=meta_info,
        )

        # Store per-step metadata as JSON strings (variable-length per trajectory)
        all_step_rewards_list = []
        all_step_anchor_obs_list = []
        all_traj_uids = []
        # Per-token step ownership tensors (one 1-D long tensor per trajectory),
        # built in lockstep with response_tokens inside assemble_steps (engine).
        all_step_token_ids_list = []

        for traj in trajectories:
            step_rewards = traj.get("step_rewards", [])
            step_anchor_obs = traj.get("step_anchor_obs", [])
            idx = traj["idx"]

            all_step_rewards_list.append(json.dumps(step_rewards))
            all_step_anchor_obs_list.append(json.dumps(step_anchor_obs))
            # Unique traj_uid per trajectory row
            all_traj_uids.append(f"traj_{idx}_{uuid.uuid4()}")

            # Per-token step ids. Engine guarantees this is the same length as the
            # trajectory's response_tokens. Default to an empty long tensor if absent
            # (older engine / non-Token paths) so padding still produces an all -1 row.
            sti = traj.get("step_token_ids")
            if sti is None:
                sti = torch.empty(0, dtype=torch.long)
            elif not isinstance(sti, torch.Tensor):
                sti = torch.tensor(sti, dtype=torch.long)
            all_step_token_ids_list.append(sti.to(torch.long))

        batch_size = result.batch["input_ids"].shape[0]
        assert len(all_traj_uids) == batch_size, (
            f"GiGPO traj_uid count ({len(all_traj_uids)}) != batch_size ({batch_size})"
        )

        result.non_tensor_batch["step_rewards_list"] = np.array(all_step_rewards_list, dtype=object)
        result.non_tensor_batch["step_anchor_obs_list"] = np.array(all_step_anchor_obs_list, dtype=object)
        result.non_tensor_batch["traj_uid"] = np.array(all_traj_uids, dtype=object)

        # Build a per-token step-id tensor aligned exactly with ``responses`` /
        # ``response_mask`` by mirroring the parent's response padding (right pad to
        # max_response_length, then right-truncate). Pad / truncated positions get -1
        # (no step owner). Because each id rides its token through the SAME pad +
        # truncate ops the parent applies to response_tokens, alignment is structural —
        # it cannot drift the way a count-based remap (step_completion_lengths) does.
        from verl.utils.torch_functional import pad_sequence_to_length

        max_response_length = self.config.data.max_response_length
        step_id_batch = torch.nn.utils.rnn.pad_sequence(
            all_step_token_ids_list, batch_first=True, padding_value=-1,
        )
        step_id_batch = pad_sequence_to_length(step_id_batch, max_response_length, -1, left_pad=False)
        step_id_batch = step_id_batch[:, :max_response_length].to(torch.long)
        result.batch["step_token_ids"] = step_id_batch

        return result, metrics

    # ------------------------------------------------------------------
    # Trajectory-level GiGPO advantage computation
    # ------------------------------------------------------------------

    def _compute_gigpo_advantage_trajectory_level(self, batch, gigpo_gamma, gigpo_step_advantage_w,
                                                   gigpo_mode, gigpo_enable_similarity,
                                                   gigpo_similarity_thresh,
                                                   gigpo_episode_cross_steps=True):
        """Compute GiGPO advantages for trajectory-level batch (STEPWISE_ADV=False).

        Strategy:
        1. Extract per-step metadata from non_tensor_batch (JSON strings)
        2. Build flat per-step arrays, call core_gigpo to get per-step scalar advantages
        3. Scatter per-step scalar advantages onto per-token positions using the
           per-token ``step_token_ids`` tensor (built in lockstep with response tokens,
           so it survives truncation / BPE recanonicalization without desync).
        """
        batch_size = batch.batch["input_ids"].shape[0]
        response_length = batch.batch["responses"].shape[1]
        device = batch.batch["input_ids"].device

        # 1. Extract per-step metadata
        step_rewards_lists = [json.loads(s) for s in batch.non_tensor_batch["step_rewards_list"]]
        step_anchor_obs_lists = [json.loads(s) for s in batch.non_tensor_batch["step_anchor_obs_list"]]
        traj_uids = batch.non_tensor_batch["traj_uid"]
        uids = batch.non_tensor_batch["uid"]

        # 2. Build flat per-step arrays
        flat_rewards = []
        flat_anchor_obs = []
        flat_traj_uid = []
        flat_uid = []
        flat_active_masks = []
        # Track which trajectory each flat step belongs to, and its step index
        flat_traj_indices = []  # (traj_row_idx, step_idx_within_traj)

        for i in range(batch_size):
            step_rewards = step_rewards_lists[i]
            step_obs = step_anchor_obs_lists[i]
            n_steps = len(step_rewards)
            for si in range(n_steps):
                flat_rewards.append(float(step_rewards[si]) if si < len(step_rewards) else 0.0)
                flat_anchor_obs.append(step_obs[si] if si < len(step_obs) else "")
                flat_traj_uid.append(traj_uids[i])
                flat_uid.append(uids[i])
                flat_active_masks.append(1.0)
                flat_traj_indices.append((i, si))

        n_flat = len(flat_rewards)
        if n_flat == 0:
            # Edge case: no steps at all
            advantages = torch.zeros(batch_size, response_length, device=device)
            return advantages, advantages

        # Build temporary flat DataProto for core_gigpo
        flat_rewards_np = np.array(flat_rewards, dtype=np.float64)
        flat_anchor_obs_np = np.array(flat_anchor_obs, dtype=object)
        flat_traj_uid_np = np.array(flat_traj_uid, dtype=object)
        flat_uid_np = np.array(flat_uid, dtype=object)
        flat_active_masks_np = np.array(flat_active_masks, dtype=np.float64)

        # Create a minimal DataProto for compute_step_discounted_returns
        # It needs: batch['input_ids'] (for device), non_tensor_batch['rewards'], ['traj_uid'], ['active_masks']
        dummy_input_ids = torch.zeros(n_flat, 1, dtype=torch.long, device=device)
        flat_batch = DataProto.from_dict(
            tensors={"input_ids": dummy_input_ids},
            non_tensors={
                "rewards": flat_rewards_np,
                "traj_uid": flat_traj_uid_np,
                "active_masks": flat_active_masks_np,
            },
        )

        # Compute gamma-discounted step returns (Eq. 5)
        step_returns = core_gigpo.compute_step_discounted_returns(flat_batch, gamma=gigpo_gamma)

        # For episode_norm_reward: need token_level_rewards and response_mask at flat step level.
        # episode_norm_reward sums token_level_rewards per row to get episode score.
        # We assign the full trajectory reward to EVERY flat step of that trajectory.
        #
        # Episode baseline semantics — controlled by ``gigpo_episode_cross_steps``:
        #   True (default) = FAITHFUL to verl-agent / GiGPO paper. verl-agent flattens one
        #     row per env-step, assigns the episode reward to every step row
        #     (EpisodeRewardManager: reward_tensor[i, last] = episode_rewards), and computes
        #     the episode baseline with cross_steps=True. A K-step trajectory contributes
        #     its episode reward K times to the group mean/std → STEP-COUNT-WEIGHTED. Our
        #     flat layout mirrors that one-row-per-step batch (every step of trajectory i
        #     carries traj_rewards[i]), so cross_steps=True reproduces verl-agent exactly.
        #   False = unweighted per-trajectory mean = plain GRPO baseline parity (each
        #     trajectory counted once regardless of length). Use this only for a GRPO-parity
        #     ablation, not for a faithful GiGPO reproduction.
        # Exposed via rllm.gigpo.episode_cross_steps so it can be flipped from launch scripts.
        flat_token_level_rewards = torch.zeros(n_flat, 1, device=device)
        flat_response_mask = torch.ones(n_flat, 1, device=device)

        # Get trajectory rewards from the original batch
        traj_rewards = batch.batch["token_level_rewards"].sum(dim=-1)  # (batch_size,)

        # Assign full trajectory reward to ALL steps of each trajectory
        for flat_idx, (traj_row, step_idx) in enumerate(flat_traj_indices):
            flat_token_level_rewards[flat_idx, 0] = traj_rewards[traj_row]

        # ---- GiGPO diagnostic: decompose advantage into episode + step components ----
        # Call sub-functions separately so we can log each component.
        if gigpo_mode == "mean_std_norm":
            _remove_std = False
        else:
            _remove_std = True

        episode_advantages_flat = core_gigpo.episode_norm_reward(
            flat_token_level_rewards, flat_response_mask, flat_uid_np, flat_traj_uid_np,
            epsilon=1e-6, remove_std=_remove_std,
            compute_mean_std_cross_steps=gigpo_episode_cross_steps,
        )
        step_group_uids, step_group_sizes = core_gigpo.build_step_group(
            flat_anchor_obs_np, flat_uid_np, gigpo_enable_similarity, gigpo_similarity_thresh,
            return_group_sizes=True,
        )
        step_advantages_flat = core_gigpo.step_norm_reward(
            step_returns, flat_response_mask, step_group_uids,
            epsilon=1e-6, remove_std=_remove_std,
        )

        # Joint advantage (Eq. 8)
        flat_advantages = episode_advantages_flat + gigpo_step_advantage_w * step_advantages_flat

        # Log decomposition
        ep_adv_vals = episode_advantages_flat[:, 0]
        st_adv_vals = step_advantages_flat[:, 0]
        joint_vals = flat_advantages[:, 0]
        n_uid_groups = len(np.unique(flat_uid_np))
        n_traj_groups = len(np.unique(flat_traj_uid_np))
        n_step_groups = len(np.unique(step_group_uids))

        print(f"[GiGPO-Trajectory] ──── Advantage Decomposition ────")
        print(f"  Flat steps: {n_flat} | uid groups: {n_uid_groups} | "
              f"traj groups: {n_traj_groups} | anchor_obs step groups: {n_step_groups}")
        print(f"  Trajectory rewards: mean={traj_rewards.mean().item():.4f}, "
              f"std={traj_rewards.std().item():.4f}, "
              f"min={traj_rewards.min().item():.4f}, max={traj_rewards.max().item():.4f}")
        print(f"  Step returns (Eq.5): mean={step_returns.mean().item():.4f}, "
              f"std={step_returns.std().item():.4f}")
        print(f"  Episode advantage (Eq.3): mean={ep_adv_vals.mean().item():.4f}, "
              f"std={ep_adv_vals.std().item():.4f}, "
              f"|nonzero|={(ep_adv_vals != 0).sum().item()}/{n_flat}")
        print(f"  Step advantage    (Eq.7): mean={st_adv_vals.mean().item():.4f}, "
              f"std={st_adv_vals.std().item():.4f}, "
              f"|nonzero|={(st_adv_vals != 0).sum().item()}/{n_flat}")
        print(f"  Joint advantage   (Eq.8): mean={joint_vals.mean().item():.4f}, "
              f"std={joint_vals.std().item():.4f}, "
              f"|nonzero|={(joint_vals != 0).sum().item()}/{n_flat}")
        print(f"  step_advantage_w={gigpo_step_advantage_w}, mode={gigpo_mode}, gamma={gigpo_gamma}")
        print(f"[GiGPO-Trajectory] ──── End Decomposition ────")

        # flat_advantages shape: (n_flat, 1) — extract scalar per step
        flat_adv_scalars = flat_advantages[:, 0]  # (n_flat,)

        # 3. Scatter per-step advantages onto per-token positions via step_token_ids.
        #
        # ``step_token_ids`` (built in lockstep with response_tokens inside
        # assemble_steps and padded position-for-position with ``responses`` /
        # ``response_mask``) tells us, for every response token, which episode_step
        # produced it (or -1 for gap / pad / fallback tokens). The flat advantage
        # array lays out each trajectory's steps contiguously in step order, so
        # trajectory i's step s maps to flat index ``flat_offset[i] + s``. We can
        # therefore assign each token its step's scalar with a pure gather — NO
        # walking of token counts, hence NO desync under truncation or BPE
        # recanonicalization (the failure mode of the old step_completion_lengths
        # remap). This is the same structural guarantee oracle_advs relies on.
        #
        # NOTE (multistep prompt): this assumes one episode_step == one LLM call ==
        # one contiguous completion span, which holds for the default agents
        # (USE_MULTISTEP_PROMPT=False). If a single step ever emits multiple
        # completion spans, the per-token id still points at the right step, but the
        # flat-offset bookkeeping below (one flat entry per step) must be revisited.
        advantages = torch.zeros(batch_size, response_length, device=device)
        response_mask = batch.batch["response_mask"]
        # (batch_size, response_length), long, -1 = no owner. Move to the advantage
        # device so boolean-mask indexing below stays on one device.
        step_token_ids = batch.batch["step_token_ids"].to(device)
        flat_adv_scalars = flat_adv_scalars.to(device)

        # Per-trajectory start offset into flat_adv_scalars (steps are contiguous,
        # in step order, exactly as built in the flattening loop above).
        flat_offset = 0
        n_unmapped_tokens = 0  # masked tokens with no valid step id (should be ~0)
        for i in range(batch_size):
            n_steps_flat = len(step_rewards_lists[i])  # flat entries for this trajectory
            ids_i = step_token_ids[i]  # (response_length,)
            mask_i = response_mask[i].bool()

            # A token contributes only if it is in the loss mask AND carries a valid,
            # in-range step id. Truncated/fallback/gap tokens (id < 0 or id >=
            # n_steps_flat) are left at 0 advantage.
            valid = mask_i & (ids_i >= 0) & (ids_i < n_steps_flat)
            if valid.any():
                flat_idx = flat_offset + ids_i[valid]  # (n_valid,) absolute flat indices
                advantages[i, valid] = flat_adv_scalars[flat_idx]

            # Diagnostic: masked tokens that got no advantage (expect 0 except for
            # fully-truncated tails or the cumulative fallback path).
            n_unmapped_tokens += int((mask_i & ~valid).sum().item())

            flat_offset += n_steps_flat

        assert flat_offset == n_flat, (
            f"GiGPO scatter offset bookkeeping mismatch: consumed {flat_offset} flat "
            f"steps but n_flat={n_flat}"
        )

        if n_unmapped_tokens > 0:
            print(f"[GiGPO-Trajectory] NOTE: {n_unmapped_tokens} masked tokens received no "
                  f"step-level advantage (truncated tails / cumulative fallback / gap); "
                  f"they still carry episode-level advantage via the trajectory reward path.")

        # Apply response_mask so gap tokens are zeroed out (defensive; scatter already
        # only writes masked positions).
        advantages = advantages * response_mask

        # Final per-token advantage summary
        nonzero = advantages[advantages != 0]
        if nonzero.numel() > 0:
            print(f"[GiGPO-Trajectory] Per-token advantages: shape={advantages.shape}, "
                  f"n_flat_steps={n_flat}, "
                  f"mean={nonzero.mean().item():.4f}, std={nonzero.std().item():.4f}, "
                  f"min={nonzero.min().item():.4f}, max={nonzero.max().item():.4f}")
        else:
            print(f"[GiGPO-Trajectory] Per-token advantages: all zero (shape={advantages.shape})")

        # ---- GiGPO diagnostic metrics for wandb ----
        # These mirror the printed decomposition above but are returned so fit_agent
        # can log them to the tracking backend (console prints are not plottable).
        gigpo_metrics = self._build_gigpo_metrics(
            n_flat=n_flat,
            n_step_groups=n_step_groups,
            step_group_sizes=step_group_sizes,
            ep_adv_vals=ep_adv_vals,
            st_adv_vals=st_adv_vals,
            joint_vals=joint_vals,
            step_returns=step_returns,
            traj_rewards=traj_rewards,
            n_traj_groups=n_traj_groups,
            n_unmapped_tokens=n_unmapped_tokens,
            response_mask=response_mask,
            step_advantage_w=gigpo_step_advantage_w,
            episode_cross_steps=gigpo_episode_cross_steps,
        )

        return advantages, advantages, gigpo_metrics

    # ------------------------------------------------------------------
    # Shared GiGPO metric builder (used by both trajectory- and step-level paths)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_gigpo_metrics(
        n_flat,
        n_step_groups,
        step_group_sizes,
        ep_adv_vals,
        st_adv_vals,
        joint_vals,
        step_returns,
        traj_rewards,
        n_traj_groups,
        n_unmapped_tokens,
        response_mask,
        step_advantage_w,
        episode_cross_steps,
    ):
        """Assemble GiGPO diagnostic metrics for wandb/console logging.

        All values are plain Python floats/ints so they serialize cleanly. Inputs
        are the per-flat-step advantage components (1-D tensors), the step-group
        size list (from build_step_group), and batch-level counts. Grouped under the
        ``gigpo/`` prefix so they sit alongside the existing ``gigpo/advantage_*``.

        The three tiers most worth watching:
          - step_group_avg_size / step_group_singleton_frac: is anchor grouping alive
            (>1) or degenerate (~1 = GiGPO collapses to GRPO)?
          - step_adv_nonzero_frac / step_adv_std: is the step-level signal contributing?
          - step_to_episode_ratio: magnitude of the GiGPO increment over plain GRPO.
        """
        m = {}

        # --- Tier 1: is GiGPO's step layer alive or degenerate? ---
        if step_group_sizes:
            sizes = np.asarray(step_group_sizes, dtype=np.float64)
            m["gigpo/step_group_avg_size"] = float(sizes.mean())
            m["gigpo/step_group_singleton_frac"] = float((sizes == 1).mean())
            m["gigpo/step_group_max_size"] = int(sizes.max())
        m["gigpo/n_step_groups"] = int(n_step_groups)

        def _nonzero_frac(t):
            return float((t != 0).float().mean().item()) if t.numel() > 0 else 0.0

        def _std(t):
            return float(t.std().item()) if t.numel() > 1 else 0.0

        m["gigpo/step_adv_nonzero_frac"] = _nonzero_frac(st_adv_vals)
        m["gigpo/episode_adv_nonzero_frac"] = _nonzero_frac(ep_adv_vals)

        # --- Tier 2: relative strength of the two advantage layers ---
        ep_std = _std(ep_adv_vals)
        st_std = _std(st_adv_vals)
        m["gigpo/episode_adv_std"] = ep_std
        m["gigpo/step_adv_std"] = st_std
        m["gigpo/joint_adv_std"] = _std(joint_vals)
        # Weighted step contribution relative to the episode signal. 0 => pure GRPO.
        m["gigpo/step_to_episode_ratio"] = (
            (step_advantage_w * st_std) / ep_std if ep_std > 1e-8 else 0.0
        )

        # --- Tier 3: Eq.5 discounted-return health (step layer input quality) ---
        if step_returns.numel() > 0:
            m["gigpo/step_return_mean"] = float(step_returns.mean().item())
            m["gigpo/step_return_std"] = _std(step_returns)
            m["gigpo/step_return_max"] = float(step_returns.max().item())

        # Trajectory-reward stats (episode-layer input).
        if traj_rewards is not None and traj_rewards.numel() > 0:
            m["gigpo/traj_reward_mean"] = float(traj_rewards.mean().item())
            m["gigpo/traj_reward_std"] = _std(traj_rewards)

        # --- Tier 4/5: pipeline health + trajectory shape ---
        m["gigpo/flat_steps"] = int(n_flat)
        m["gigpo/traj_groups"] = int(n_traj_groups)
        n_masked = int(response_mask.bool().sum().item())
        m["gigpo/unmapped_token_frac"] = (
            float(n_unmapped_tokens) / n_masked if n_masked > 0 else 0.0
        )
        m["gigpo/avg_steps_per_traj"] = (
            float(n_flat) / n_traj_groups if n_traj_groups > 0 else 0.0
        )
        m["gigpo/episode_cross_steps"] = 1.0 if episode_cross_steps else 0.0

        return m


    def fit_agent(self):
        """
        The training loop of PPO with GiGPO advantage estimation.

        This is a copy of AgentPPOTrainer.fit_agent() with the advantage
        computation section (lines 650-694 in the original) replaced by
        GiGPO logic. Everything else is unchanged.
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

        # --- GiGPO config ---
        gigpo_cfg = OmegaConf.to_container(
            self.config.rllm.get("gigpo", {}), resolve=True
        ) or {}
        gigpo_gamma = gigpo_cfg.get("gamma", 1.0)
        gigpo_step_advantage_w = gigpo_cfg.get("step_advantage_w", 1.0)
        gigpo_mode = gigpo_cfg.get("mode", "mean_norm")
        gigpo_enable_similarity = gigpo_cfg.get("enable_similarity", False)
        gigpo_similarity_thresh = gigpo_cfg.get("similarity_thresh", 0.95)
        # Episode-baseline weighting for the trajectory-level path (Eq.3):
        #   True  (default) = step-count-weighted mean, FAITHFUL to verl-agent / GiGPO paper
        #                     (a K-step trajectory counts K times in the group mean/std).
        #   False = unweighted per-trajectory mean = plain GRPO baseline parity.
        # The step-level path always uses the core default (True); this knob only affects
        # the trajectory-level (STEPWISE_ADV=False) reconstruction.
        gigpo_episode_cross_steps = gigpo_cfg.get("episode_cross_steps", True)
        print(f"[GiGPO] Config: gamma={gigpo_gamma}, step_advantage_w={gigpo_step_advantage_w}, "
              f"mode={gigpo_mode}, enable_similarity={gigpo_enable_similarity}, "
              f"similarity_thresh={gigpo_similarity_thresh}, "
              f"episode_cross_steps={gigpo_episode_cross_steps}")

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

                    # GiGPO requires stepwise_advantage.enable=True
                    if self.config.rllm.stepwise_advantage.enable:
                        final_gen_batch_output, generate_metrics = self.generate_agent_steps(timing_raw=timing_raw, meta_info=batch.meta_info, uids=batch.non_tensor_batch["uid"])
                        metrics.update(generate_metrics)
                        repeat_counts = final_gen_batch_output.meta_info["repeat_counts"]
                        batch = batch.sample_level_repeat(repeat_counts)
                        final_gen_batch_output.meta_info.pop("repeat_counts", None)
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

                        if "token_level_scores" not in batch.batch:
                            reward_tensor = self.reward_fn(batch)
                            batch.batch["token_level_scores"] = reward_tensor
                        else:
                            reward_tensor = batch.batch["token_level_scores"]

                        # Rejection sampling based on rewards
                        uids = batch.non_tensor_batch["uid"]
                        unique_uids = np.unique(uids)
                        valid_mask = torch.ones(len(uids), dtype=torch.bool)
                        solve_none = 0
                        solve_all = 0
                        for uid in unique_uids:
                            uid_mask = uids == uid
                            uid_rewards = reward_tensor[uid_mask].sum(-1)

                            if (uid_rewards <= 0).all():
                                valid_mask[uid_mask] = False
                                solve_none += 1
                            elif (uid_rewards >= 1).all():
                                valid_mask[uid_mask] = False
                                solve_all += 1

                        metrics["batch/solve_none"] = solve_none
                        metrics["batch/solve_all"] = solve_all
                        metrics["batch/solve_partial"] = len(unique_uids) - solve_none - solve_all

                        if self.config.rllm.rejection_sample.enable:
                            token_level_rewards = None
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

                            if not valid_mask.any():
                                continue

                            batch = batch[valid_mask]

                            if self.config.rllm.stepwise_advantage.enable and self.config.rllm.stepwise_advantage.mode == "broadcast":
                                is_pad_step = batch.non_tensor_batch["is_pad_step"]
                                non_pad_step_indices = np.where(is_pad_step == False)[0]
                                batch = batch.select_idxs(non_pad_step_indices)

                                is_last_step = batch.non_tensor_batch["is_last_step"]
                                valid_last_step_indices = np.where(is_last_step == True)[0]
                                not_last_step_indices = np.where(is_last_step == False)[0]
                                last_step_batch = batch.select_idxs(valid_last_step_indices)
                                non_last_step_batch = batch.select_idxs(not_last_step_indices)

                                num_trainer_replicas = self.actor_rollout_wg.world_size
                                max_batch_size = (
                                    last_step_batch.batch["input_ids"].shape[0]
                                    // num_trainer_replicas
                                ) * num_trainer_replicas
                                if not max_batch_size:
                                    continue

                                size_mask = torch.zeros(last_step_batch.batch["input_ids"].shape[0], dtype=torch.bool)
                                size_mask[:max_batch_size] = True
                                last_step_batch = last_step_batch[size_mask]

                                valid_last_step_idxs = last_step_batch.non_tensor_batch["idxs"]
                                non_last_step_idxs = non_last_step_batch.non_tensor_batch["idxs"]
                                non_last_step_mask = np.isin(non_last_step_idxs, valid_last_step_idxs)
                                non_last_step_batch = non_last_step_batch[non_last_step_mask]

                                batch = DataProto.concat([last_step_batch, non_last_step_batch])
                                batch = self._pad_dataproto_to_world_size(batch)
                            else:
                                num_trainer_replicas = self.actor_rollout_wg.world_size
                                max_batch_size = (batch.batch["input_ids"].shape[0] // num_trainer_replicas) * num_trainer_replicas
                                if not max_batch_size:
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
                            with marked_timer("ref", timing_raw):
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)

                        # ============================================================
                        # GiGPO ADVANTAGE COMPUTATION — replaces GRPO/PPO advantage
                        # ============================================================
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        if self.config.rllm.stepwise_advantage.enable:
                            # --- Step-level GiGPO (one row per step) ---
                            # Filter out pad steps
                            is_pad_step = batch.non_tensor_batch.get("is_pad_step", np.array([False] * len(batch.batch["input_ids"])))
                            non_pad_mask = is_pad_step == False
                            non_pad_indices = np.where(non_pad_mask)[0]
                            if len(non_pad_indices) < len(is_pad_step):
                                batch = batch.select_idxs(non_pad_indices)

                            # Compute gamma-discounted step returns (Eq. 5)
                            step_returns = core_gigpo.compute_step_discounted_returns(
                                batch, gamma=gigpo_gamma,
                            )

                            # Compute GiGPO joint advantage (Eq. 3 + 6 + 7 + 8)
                            # Decompose into episode + step for diagnostic logging.
                            _tlr = batch.batch["token_level_rewards"]
                            _rmask = batch.batch["response_mask"]
                            _anchor = batch.non_tensor_batch["anchor_obs"]
                            _uid = batch.non_tensor_batch["uid"]
                            _traj_uid = batch.non_tensor_batch["traj_uid"]
                            _remove_std = gigpo_mode != "mean_std_norm"

                            episode_advantages = core_gigpo.episode_norm_reward(
                                _tlr, _rmask, _uid, _traj_uid,
                                epsilon=1e-6, remove_std=_remove_std,
                            )
                            step_group_uids, step_group_sizes = core_gigpo.build_step_group(
                                _anchor, _uid, gigpo_enable_similarity, gigpo_similarity_thresh,
                                return_group_sizes=True,
                            )
                            step_advantages = core_gigpo.step_norm_reward(
                                step_returns, _rmask, step_group_uids,
                                epsilon=1e-6, remove_std=_remove_std,
                            )
                            advantages = episode_advantages + gigpo_step_advantage_w * step_advantages
                            returns = advantages

                            batch.batch["advantages"] = advantages
                            batch.batch["returns"] = returns

                            # Diagnostic logging
                            n_steps = len(_uid)
                            n_uid_groups = len(np.unique(_uid))
                            n_traj_groups = len(np.unique(_traj_uid))
                            n_step_groups = len(np.unique(step_group_uids))
                            ep_vals = episode_advantages[_rmask.bool()]
                            st_vals = step_advantages[_rmask.bool()]
                            joint_vals = advantages[_rmask.bool()]
                            traj_rewards_per_step = _tlr.sum(dim=-1)

                            print(f"[GiGPO-Step] ──── Advantage Decomposition ────")
                            print(f"  Steps: {n_steps} | uid groups: {n_uid_groups} | "
                                  f"traj groups: {n_traj_groups} | anchor_obs step groups: {n_step_groups}")
                            print(f"  Trajectory rewards/step: mean={traj_rewards_per_step.mean().item():.4f}, "
                                  f"std={traj_rewards_per_step.std().item():.4f}")
                            print(f"  Step returns (Eq.5): mean={step_returns.mean().item():.4f}, "
                                  f"std={step_returns.std().item():.4f}")
                            print(f"  Episode advantage (Eq.3): mean={ep_vals.mean().item():.4f}, "
                                  f"std={ep_vals.std().item():.4f}, "
                                  f"|nonzero|={(ep_vals != 0).sum().item()}/{ep_vals.numel()}")
                            print(f"  Step advantage    (Eq.7): mean={st_vals.mean().item():.4f}, "
                                  f"std={st_vals.std().item():.4f}, "
                                  f"|nonzero|={(st_vals != 0).sum().item()}/{st_vals.numel()}")
                            print(f"  Joint advantage   (Eq.8): mean={joint_vals.mean().item():.4f}, "
                                  f"std={joint_vals.std().item():.4f}, "
                                  f"|nonzero|={(joint_vals != 0).sum().item()}/{joint_vals.numel()}")
                            print(f"  step_advantage_w={gigpo_step_advantage_w}, mode={gigpo_mode}, gamma={gigpo_gamma}")
                            print(f"[GiGPO-Step] ──── End Decomposition ────")

                            # ---- GiGPO diagnostic metrics for wandb ----
                            # In the step path each row is already one step, so ep/st/joint
                            # vals are the masked per-token values, step_returns is per-row,
                            # and there is no scatter (n_unmapped=0). The step path always uses
                            # the core default episode baseline (cross_steps=True).
                            metrics.update(self._build_gigpo_metrics(
                                n_flat=n_steps,
                                n_step_groups=n_step_groups,
                                step_group_sizes=step_group_sizes,
                                ep_adv_vals=ep_vals,
                                st_adv_vals=st_vals,
                                joint_vals=joint_vals,
                                step_returns=step_returns,
                                traj_rewards=traj_rewards_per_step,
                                n_traj_groups=n_traj_groups,
                                n_unmapped_tokens=0,
                                response_mask=_rmask,
                                step_advantage_w=gigpo_step_advantage_w,
                                episode_cross_steps=True,
                            ))
                        else:
                            # --- Trajectory-level GiGPO (one row per trajectory) ---
                            advantages, returns, gigpo_traj_metrics = self._compute_gigpo_advantage_trajectory_level(
                                batch,
                                gigpo_gamma=gigpo_gamma,
                                gigpo_step_advantage_w=gigpo_step_advantage_w,
                                gigpo_mode=gigpo_mode,
                                gigpo_enable_similarity=gigpo_enable_similarity,
                                gigpo_similarity_thresh=gigpo_similarity_thresh,
                                gigpo_episode_cross_steps=gigpo_episode_cross_steps,
                            )
                            batch.batch["advantages"] = advantages
                            batch.batch["returns"] = returns
                            metrics.update(gigpo_traj_metrics)

                        # ---- GiGPO metrics for wandb/console logger ----
                        _adv = batch.batch["advantages"]
                        _rmask_all = batch.batch["response_mask"]
                        _adv_masked = _adv[_rmask_all.bool()]
                        if _adv_masked.numel() > 0:
                            metrics["gigpo/advantage_mean"] = _adv_masked.mean().item()
                            metrics["gigpo/advantage_std"] = _adv_masked.std().item()
                            metrics["gigpo/advantage_abs_mean"] = _adv_masked.abs().mean().item()
                            metrics["gigpo/advantage_max"] = _adv_masked.max().item()
                            metrics["gigpo/advantage_min"] = _adv_masked.min().item()
                            metrics["gigpo/advantage_nonzero_frac"] = (
                                (_adv_masked != 0).sum().item() / _adv_masked.numel()
                            )
                        # ============================================================
                        # END GiGPO ADVANTAGE
                        # ============================================================

                    if self.config.rllm.mask_truncated_samples:
                        mask = batch.batch["attention_mask"][:, -1] == 1
                        batch = batch[~mask]

                    batch = self._pad_dataproto_to_world_size(batch=batch)
                    self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
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

                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate_agent()
                        pprint(f"Final validation metrics: {val_metrics}")
                        logger.log(data=val_metrics, step=self.global_steps)
                    return
