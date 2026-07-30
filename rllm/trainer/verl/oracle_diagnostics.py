"""
CAST alpha-ablation diagnostics.

Online training-time metrics for the CAST additive asinh oracle shaping scheme,
designed to support the alpha ablation reported in the paper.

All metrics are returned under the ``ablation/`` namespace and are merged into
the trainer's ``metrics`` dict (which is logged to wandb/tensorboard). Nothing
here mutates training tensors — it is read-only diagnostics.

Two families of metrics:

1. Token-level, mapping-agnostic (computed for ANY oracle mapping mode):
   within-uid reward-ordering preservation, confident sign-flip rate,
   GRPO-baseline degeneration, oracle-vs-reward alignment.

2. CAST-specific (when ``mode in CAST_MODES``): additive shaping-vs-base
   balance metrics and turn-level credit-assignment/effective-alpha metrics.
   The turn-level metrics reconstruct ``alpha * g(a) / (rms + eps)`` directly from
   trajectory-level ``turn_advantages`` using the SAME batch-RMS convention as
   ``mapping_cast`` (RMS over all turns in the batch, turn-weighted — NOT
   token-weighted). The transform ``g`` mirrors the applied mapping: asinh for
   "cast", or ``config.cast_transform`` for "cast_ablate". Stepwise ``per_step``
   batches that do not carry trajectory-level ``turn_advantages`` cannot report
   these turn-level metrics.

Design contract: a failure in any single metric must never crash training, so
each block is individually guarded. The caller additionally wraps the whole call
in try/except as a backstop.
"""

import numpy as np
import torch
from torch import Tensor

from rllm.trainer.verl.oracle_advantage_mapping import (
    CAST_MODES,
    MODE_CAST,
    _compute_grpo_trajectory_advantages,
    _oracle_transform,
)


# Trajectories with |b_j| >= this are treated as "confident" GRPO signals.
_CONFIDENT_BASE = 0.5
# Trajectories with |b_j| < this are treated as degenerate (no GRPO contrast).
_BASE_ZERO_EPS = 1e-8
# Oracle advantage <= this counts as a deadlock-tail turn (vs ordinary harmful -1).
_DEADLOCK_THRESH = -2.0
_EPS = 1e-8


def compute_cast_diagnostics(
    *,
    mode: str,
    mapped_advs: Tensor,
    response_mask: Tensor,
    token_level_scores: Tensor,
    uids: np.ndarray,
    turn_advantages: np.ndarray,
    config,
) -> dict[str, float]:
    """Compute CAST ablation diagnostics.

    Args:
        mode: oracle mapping mode (e.g. "cast"). Additive shaping balance and
            turn-level transform metrics are only computed when
            mode in CAST_MODES; ordering/flip/base-zero/
            oracle-correlation metrics are mode-agnostic.
        mapped_advs: (B, T) final per-token advantages AFTER apply_mapping.
        response_mask: (B, T) 1 on response (completion) tokens.
        token_level_scores: (B, T) per-token rewards; row-sum is the trajectory reward R_j.
        uids: (B,) prompt-group identifiers (same uid == same prompt / rollout group).
        turn_advantages: (B,) object array; entry i is a per-turn oracle-advantage
            array for trajectory i (may be None / empty).
        config: algorithm config (read cast_alpha, cast_eps).

    Returns:
        dict of {"ablation/<name>": float}. Empty on total failure.
    """
    metrics: dict[str, float] = {}

    mask = response_mask.bool()
    if mask.sum() == 0:
        return metrics

    # --- GRPO trajectory baseline b_j (same computation CAST uses internally) ---
    traj_rewards = token_level_scores.sum(dim=-1)
    base = _compute_grpo_trajectory_advantages(traj_rewards, uids)  # (B,)
    base_tok = base.unsqueeze(-1).expand_as(mapped_advs)            # (B, T)
    A_m = mapped_advs[mask]
    # Confident-base token mask, reused by the balance and conflict-safety metrics.
    conf_tok = (base.abs() >= _CONFIDENT_BASE).unsqueeze(-1) & mask

    # ===================================================================
    # Token-level mapping-agnostic diagnostics. These only compare final
    # advantage against the GRPO base, so they are meaningful for any oracle
    # mapping mode. CAST-specific additive shaping metrics are below the
    # ``if mode in CAST_MODES`` guard.
    # ===================================================================

    # --- C. Reward Ordering Preservation (within-uid Kendall tau) ---
    # Must be computed within each uid group: b_j is a within-group z-score, so
    # cross-prompt ranking would mix in problem difficulty. Degenerate groups
    # (b all-equal, e.g. all 0) are skipped and tracked via valid_frac.
    try:
        from scipy.stats import kendalltau

        # Per-trajectory masked mean of final advantage.
        A_mean = (mapped_advs * mask).sum(-1) / mask.sum(-1).clamp(min=1)  # (B,)
        base_np = base.detach().cpu().numpy()
        A_mean_np = A_mean.detach().cpu().numpy()
        unique_uids = np.unique(uids) if uids.size > 0 else np.array([])
        taus = []
        for u in unique_uids:
            idx = np.where(uids == u)[0]
            if idx.size < 2:
                continue
            b_u = base_np[idx]
            if np.std(b_u) < _BASE_ZERO_EPS:
                continue  # degenerate group: no GRPO ordering to preserve
            tau = kendalltau(b_u, A_mean_np[idx]).correlation
            if tau is not None and not np.isnan(tau):
                taus.append(tau)
        n_groups = len(unique_uids)
        metrics["ablation/kendall_tau_uid_mean"] = float(np.mean(taus)) if taus else 0.0
        metrics["ablation/kendall_tau_uid_valid_frac"] = (
            len(taus) / n_groups if n_groups > 0 else 0.0
        )
    except Exception:
        pass

    # --- D. Reward Conflict Safety (confident sign-flip rate) ---
    # Fraction of CONFIDENT-base tokens whose final advantage flipped sign vs base.
    # b ~= 0 tokens are excluded: their "flip" is benign oracle-only activation,
    # not a conflict with a real reward signal.
    try:
        if conf_tok.any():
            flip = (torch.sign(mapped_advs) != torch.sign(base_tok)) & conf_tok
            metrics["ablation/flip_rate_base_confident"] = float(
                (flip.float().sum() / conf_tok.float().sum().clamp(min=1)).item()
            )
        else:
            metrics["ablation/flip_rate_base_confident"] = 0.0
    except Exception:
        pass

    # --- E (part). GRPO baseline degeneration ---
    # Fraction of trajectories with b_j == 0: CAST degenerates to A_final = alpha*h
    # (pure oracle) on these — the irreversible collapse seed.
    try:
        metrics["ablation/base_zero_traj_frac"] = float(
            (base.abs() < _BASE_ZERO_EPS).float().mean().item()
        )
    except Exception:
        pass

    # --- Context. Oracle validity (oracle-vs-reward alignment) ---
    # corr_j( mean_s a_{j,s}, R_j ). Alpha-independent; sets the alpha safety ceiling.
    try:
        mean_a = np.array(
            [
                float(np.mean(t)) if t is not None and len(t) > 0 else 0.0
                for t in turn_advantages
            ],
            dtype=np.float64,
        )
        R = traj_rewards.detach().cpu().numpy()
        if mean_a.shape[0] == R.shape[0] and np.std(mean_a) > _BASE_ZERO_EPS and np.std(R) > _BASE_ZERO_EPS:
            metrics["ablation/oracle_reward_corr"] = float(np.corrcoef(mean_a, R)[0, 1])
        else:
            metrics["ablation/oracle_reward_corr"] = 0.0
    except Exception:
        pass

    # ===================================================================
    # CAST-specific additive shaping metrics and turn-level asinh diagnostics.
    # ===================================================================
    if mode in CAST_MODES:
        try:
            shape = mapped_advs - base_tok                          # (B, T) = alpha * h
            shape_m = shape[mask]

            # --- A. Shaping vs Base Balance ---
            # shaping_var_ratio = Var(alpha*h) / Var(A_final). NOT bounded to [0,1]:
            # Var(A)=Var(b)+Var(shape)+2Cov(b,shape); negative Cov can push it above 1,
            # which signals shaping/base conflict rather than a bug.
            denom = A_m.var(unbiased=False) + _EPS
            metrics["ablation/shaping_var_ratio"] = float(
                (shape_m.var(unbiased=False) / denom).item()
            )

            # shape_base_ratio_p95_confident = P95( |alpha*h| / |b_j| ) over
            # confident tokens. Precursor to confident sign-flip: a flip on
            # confident base needs |alpha*h| > |b_j|.
            if conf_tok.any():
                ratio = (shape.abs() / (base_tok.abs() + _EPS))[conf_tok]
                metrics["ablation/shape_base_ratio_p95_confident"] = float(
                    torch.quantile(ratio, 0.95).item()
                )
            else:
                metrics["ablation/shape_base_ratio_p95_confident"] = 0.0

            alpha = float(config.get("cast_alpha", 0.3))
            eps = float(config.get("cast_eps", 1e-8))
            # Transform must mirror the mapping actually applied. Production "cast"
            # is hardcoded to asinh, so force it here regardless of any stray
            # cast_transform value; only "cast_ablate" honors the config flag.
            transform = "asinh" if mode == MODE_CAST else str(config.get("cast_transform", "asinh"))
            # Same for normalization: "cast" always applies RMS norm; only
            # "cast_ablate" honors cast_norm. denom mirrors mapping_cast_ablate.
            norm = "rms" if mode == MODE_CAST else str(config.get("cast_norm", "rms"))

            # Per-trajectory turn arrays, mirroring mapping_cast's all_turn_advs.
            turn_arrays = [
                np.asarray(t, dtype=np.float64).ravel()
                for t in turn_advantages
                if t is not None and len(t) > 0
            ]
            if turn_arrays:
                all_a = np.concatenate(turn_arrays)
                g_all = _oracle_transform(all_a, transform)
                rms = float(np.sqrt(np.mean(g_all ** 2)))
                # Effective normalization denominator, mirroring mapping_cast_ablate:
                # "rms" divides by batch RMS (default); "none" divides by 1.
                shape_denom = (rms + eps) if norm == "rms" else 1.0

                # B. Credit Assignment Gain
                # intra_traj_shape_std_turn: per-trajectory std of the turn-level
                # shaping term, averaged over trajectories. 0 at alpha=0; >0 means
                # CAST differentiates turns within a trajectory. Turn-weighted.
                shape_turn_arrays = [
                    alpha * _oracle_transform(a, transform) / shape_denom for a in turn_arrays
                ]
                traj_stds = [float(np.std(x)) for x in shape_turn_arrays if len(x) > 1]
                metrics["ablation/intra_traj_shape_std_turn"] = (
                    float(np.mean(traj_stds)) if traj_stds else 0.0
                )

                # turn_shape_gap_pos_neg: mean shaping on oracle-positive turns
                # minus mean on oracle-negative turns (are good/bad turns separated?).
                shape_all = np.concatenate(shape_turn_arrays)
                pos, neg = all_a > 0, all_a < 0
                metrics["ablation/turn_shape_gap_pos_neg"] = (
                    float(shape_all[pos].mean() - shape_all[neg].mean())
                    if pos.any() and neg.any()
                    else 0.0
                )

                # E (part). Effective-alpha / RMS feedback + oracle distribution.
                eff = np.abs(alpha * g_all / shape_denom)
                metrics["ablation/turn_rms_asinh"] = rms
                metrics["ablation/effective_alpha_p95"] = float(np.percentile(eff, 95))
                metrics["ablation/turn_oracle_zero_frac"] = float(np.mean(all_a == 0))
                metrics["ablation/turn_oracle_deadlock_frac"] = float(
                    np.mean(all_a <= _DEADLOCK_THRESH)
                )
        except Exception:
            pass

    return metrics
