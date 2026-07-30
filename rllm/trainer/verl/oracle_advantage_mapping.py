"""
Oracle Advantage Mapping Registry.

Each mapping scheme is registered as a standalone function. The trainer calls
apply_mapping(mode, ...) and the registry dispatches to the right function.

To add a new scheme:
    1. Write a function with the standard signature
    2. Decorate with @register("your_mode_name")
    3. Done — no other file needs changes.

Registered modes:
    "cast"        — the CAST paper method (asinh + batch-RMS additive shaping)
    "cast_ablate" — same, but with the transform/normalization exposed as knobs
                    for the paper's asinh/RMS ablation

The trainer also accepts "off", which skips the oracle path entirely and falls
back to standard GRPO/GAE. "off" is deliberately NOT registered here: it is a
control-flow branch in the trainer, not a mapping.
"""

import numpy as np
import torch
from torch import Tensor


REGISTRY: dict[str, callable] = {}


def register(name: str):
    """Decorator to register a mapping function."""
    def decorator(fn):
        REGISTRY[name] = fn
        return fn
    return decorator


# Mode-string constants. Register with these rather than string literals so the
# registry keys and the names other modules branch on cannot drift apart —
# oracle_diagnostics imports CAST_MODES/MODE_CAST for exactly that reason.
MODE_CAST = "cast"
MODE_CAST_ABLATE = "cast_ablate"
CAST_MODES = (MODE_CAST, MODE_CAST_ABLATE)


def apply_mapping(
    mode: str,
    oracle_advs: Tensor,
    response_mask: Tensor,
    **kwargs,
) -> Tensor:
    """Unified entry point. Trainer calls only this."""
    if mode not in REGISTRY:
        raise ValueError(
            f"Unknown oracle_adv_mapping_mode: '{mode}'. "
            f"Available: {sorted(REGISTRY.keys())}"
        )
    return REGISTRY[mode](oracle_advs, response_mask, **kwargs)


# =============================================================================
# Helper: Step-Token Broadcast
# =============================================================================

def _broadcast_steps_to_tokens(row_mask: Tensor, step_values: np.ndarray) -> Tensor:
    """
    Broadcast per-step scalar values back to token level.

    Step boundaries are taken AUTHORITATIVELY from the contiguous runs of 1s in
    `row_mask` (= response_mask[i]), NOT from the oracle advantage values.

    Why this is correct
    -------------------
    assemble_steps() lays out each trajectory step as:
        [gap tokens mask=0] [completion tokens mask=1]
    The first step has no leading gap; every later step's completion is a fresh
    contiguous mask=1 run, separated from the previous one by >=1 gap token
    (env observation + chat-template tokens, all mask=0). Therefore the number
    of contiguous 1-runs equals the number of steps, and each run's length is
    that step's completion-token count. This is independent of the oracle
    VALUES, which frequently repeat across steps in Sokoban (e.g. consecutive
    optimal moves all map to +1, consecutive wasteful moves to 0).

    Boundaries must NOT be inferred from the oracle values themselves: equal
    values across consecutive steps hide a boundary, and distinct values that
    clip to the same final value create phantom ones.

    Ordering contract
    -----------------
    The returned tensor has length row_mask.sum() and is ordered identically to
    oracle_advs[i][row_mask], so callers assign it back via:
        mapped[i][row_mask] = result

    Args:
        row_mask:    (full_response_len,) the per-row response_mask (bool/0-1).
                     Contiguous 1-runs delimit steps.
        step_values: (T,) one scalar per step, in step order.

    Returns:
        (row_mask.sum(),) float tensor of token-level values, masked order.
    """
    mask_bool = row_mask.bool()
    n_masked = int(mask_bool.sum().item())
    T = len(step_values)

    # Empty guards: no masked tokens or no steps -> nothing to broadcast.
    if n_masked == 0 or T == 0:
        return torch.zeros(n_masked, dtype=torch.float32, device=row_mask.device)

    # --- Recover per-step completion lengths via run-length encoding of 1-runs ---
    # Find indices where the mask value changes, then keep only the runs whose
    # value is 1 (completion tokens). Loops only over the (small) number of runs.
    m = mask_bool.to(torch.int8)
    change = m[1:] != m[:-1]
    cut = torch.nonzero(change, as_tuple=False).flatten() + 1
    bounds = [0] + cut.tolist() + [m.numel()]
    seg_lengths = [
        bounds[k + 1] - bounds[k]
        for k in range(len(bounds) - 1)
        if m[bounds[k]].item() == 1  # keep mask=1 runs only
    ]
    n_segments = len(seg_lengths)

    result = torch.zeros(n_masked, dtype=torch.float32, device=row_mask.device)

    # --- Strict alignment: #1-runs from mask must equal #step_values ---
    if n_segments == T:
        pos = 0
        for s in range(T):
            L = seg_lengths[s]
            result[pos:pos + L] = float(step_values[s])
            pos += L
        return result

    # --- Safety fallback (should be rare) ---
    # Only reachable if two steps' completions touched with NO gap token in
    # between (mask ran 1->1 across a boundary), which the multi-turn chat
    # template normally prevents. We log and fall back to the old uniform split
    # so training never crashes, while the warning surfaces any real violation.
    import logging
    logging.getLogger(__name__).warning(
        "[_broadcast_steps_to_tokens] mask 1-runs (%d) != step_values (%d); "
        "falling back to uniform split (possible zero-gap step boundary).",
        n_segments, T,
    )
    tokens_per_step = n_masked / T
    for s in range(T):
        start = int(round(s * tokens_per_step))
        end = min(int(round((s + 1) * tokens_per_step)), n_masked)
        result[start:end] = float(step_values[s])
    return result



def _compute_grpo_trajectory_advantages(
    traj_rewards: Tensor,
    uids: np.ndarray,
) -> Tensor:
    """
    Compute GRPO group z-score trajectory advantages.

    For each uid group, compute: (reward - group_mean) / group_std.
    Groups with std < 1e-6 get advantage = 0 (no contrast signal).

    Args:
        traj_rewards: (batch_size,) - per-trajectory total rewards
        uids: (batch_size,) - group identifiers

    Returns:
        (batch_size,) - GRPO z-scored trajectory advantages
    """
    grpo_advs = torch.zeros_like(traj_rewards)
    unique_uids = np.unique(uids)

    for uid in unique_uids:
        uid_mask = torch.tensor(uids == uid, dtype=torch.bool, device=traj_rewards.device)
        group_rewards = traj_rewards[uid_mask]
        if group_rewards.numel() <= 1:
            continue
        group_std = group_rewards.std()
        if group_std < 1e-6:
            continue  # No contrast signal
        group_mean = group_rewards.mean()
        grpo_advs[uid_mask] = (group_rewards - group_mean) / group_std

    return grpo_advs


def _oracle_transform(a: np.ndarray, name: str) -> np.ndarray:
    """Compression transform g(a) for oracle advantages (CAST ablation only).

    All variants are odd functions with g(0)=0, so oracle=0 always maps to zero
    shaping regardless of the choice — only the tail-compression behavior differs.

    - "asinh": g(a)=arcsinh(a). CAST default: near-zero linear, far-field log.
    - "identity": g(a)=a. No compression (main ablation baseline).
    - "signed_log": g(a)=sign(a)*log(1+|a|). Optional: compresses main range too.
    """
    if name == "asinh":
        return np.arcsinh(a)
    if name == "identity":
        return a
    if name == "signed_log":
        return np.sign(a) * np.log1p(np.abs(a))
    raise ValueError(
        f"Unknown oracle transform: '{name}'. "
        f"Available: 'asinh', 'identity', 'signed_log'."
    )


# =============================================================================
# CAST: Additive Asinh Oracle Shaping with Batch RMS Normalization
# =============================================================================

@register(MODE_CAST)
def mapping_cast(
    oracle_advs: Tensor,
    response_mask: Tensor,
    *,
    uids: np.ndarray,
    token_level_scores: Tensor,
    turn_advantages: np.ndarray,
    rollout_n: int,
    config,
    **kwargs,
) -> Tensor:
    """
    Additive Asinh Oracle Shaping with Batch RMS Normalization.

    Core formula: A_{final,t} = A_j^GRPO + alpha * h(a_t)
    where h(a) = asinh(a) / RMS_batch(asinh)

    Algorithm:
    1. Compute GRPO trajectory advantages (z-score within each uid group)
    2. Collect ALL turn-level oracle advantages in batch (including 0)
    3. Apply asinh transform: g(a) = arcsinh(a)
    4. Compute batch-level RMS over all g-values (including g(0)=0)
    5. Normalize: h(a) = g(a) / (RMS + eps)
    6. Additive shaping: A_final = A_j^GRPO + alpha * h(a_t)
    7. Broadcast step-level values to token-level

    Properties:
    - Asinh: standard math function, near-zero linear, far-field logarithmic
    - g(0) = 0: dead/ineffective steps get zero shaping (pure GRPO preserved)
    - Monotonic: worse oracle → more negative shaping
    - Not excluding oracle=0 from RMS: provides free curriculum effect
      (more deadlocks → smaller RMS → stronger shaping on informative steps)
    - Single hyperparameter: alpha controls shaping strength
    - Direction: may flip for extreme oracle + near-zero GRPO (semantically valid)

    Based on:
    - Ng et al. (1999) Policy Invariance Under Reward Transformations
    - Wiewiora et al. (2003) Principled Methods for Advising RL Agents

    Config params:
        cast_alpha (default 0.3): shaping strength coefficient
        cast_eps (default 1e-8): RMS denominator protection
    """
    alpha = float(config.get("cast_alpha", 0.3))
    eps = float(config.get("cast_eps", 1e-8))

    mask = response_mask.bool()
    mapped = torch.zeros_like(oracle_advs)

    # --- Step 1: Compute GRPO trajectory advantages ---
    traj_rewards = token_level_scores.sum(dim=-1)
    grpo_advs = _compute_grpo_trajectory_advantages(traj_rewards, uids)

    batch_size = oracle_advs.shape[0]

    # --- Step 2: Collect ALL turn-level oracle advantages in batch ---
    all_turn_advs = []
    for i in range(batch_size):
        step_advs = turn_advantages[i] if i < len(turn_advantages) else None
        if step_advs is not None and hasattr(step_advs, '__len__') and len(step_advs) > 0:
            all_turn_advs.extend(step_advs)

    # --- Step 3 & 4: Asinh transform + Batch-level RMS ---
    if len(all_turn_advs) > 0:
        all_g = np.arcsinh(np.array(all_turn_advs, dtype=np.float64))
        rms = np.sqrt(np.mean(all_g ** 2))
    else:
        # Fallback: no oracle info in batch → no shaping possible
        rms = 1.0

    # --- Step 5 & 6 & 7: Normalize + Additive Shaping + Broadcast ---
    for i in range(batch_size):
        row_mask = mask[i]
        if not row_mask.any():
            continue

        base = grpo_advs[i].item()

        # Get per-step oracle advantages
        step_advs = turn_advantages[i] if i < len(turn_advantages) else None
        if step_advs is None or (hasattr(step_advs, '__len__') and len(step_advs) == 0):
            # No oracle info → pure GRPO
            mapped[i][row_mask] = base
            continue

        step_advs = np.array(step_advs, dtype=np.float64)
        T = len(step_advs)
        if T == 0:
            mapped[i][row_mask] = base
            continue

        # Asinh transform
        g = np.arcsinh(step_advs)

        # RMS normalization
        h = g / (rms + eps)

        # Additive shaping: A_final = A_j^GRPO + alpha * h(a_t)
        final_step_values = base + alpha * h

        # Broadcast to token level (step boundaries from response_mask 1-runs).
        token_values = _broadcast_steps_to_tokens(row_mask, final_step_values)
        mapped[i][row_mask] = token_values

    return mapped


# =============================================================================
# CAST-ablate: CAST with configurable compression transform (asinh ablation)
# =============================================================================

@register(MODE_CAST_ABLATE)
def mapping_cast_ablate(
    oracle_advs: Tensor,
    response_mask: Tensor,
    *,
    uids: np.ndarray,
    token_level_scores: Tensor,
    turn_advantages: np.ndarray,
    rollout_n: int,
    config,
    **kwargs,
) -> Tensor:
    """CAST with a configurable compression transform g(.) for asinh ablation.

    Identical to ``mapping_cast`` in every respect EXCEPT the compression
    function is read from ``config.cast_transform`` (default "asinh") instead of
    being hardcoded to asinh. This isolates the effect of the asinh tail
    compression while keeping additive shaping, batch turn-level RMS norm, alpha,
    and oracle=0 inclusion fixed.

    With ``cast_transform="asinh"`` this is numerically equivalent to ``mapping_cast``.

    NOTE: This is a deliberate copy of ``mapping_cast`` so the production CAST path
    stays byte-for-byte unchanged. If ``mapping_cast`` is modified, mirror the
    change here.

    Config params:
        cast_transform (default "asinh"): compression function g(a).
            One of "asinh" / "identity" / "signed_log". See ``_oracle_transform``.
        cast_alpha (default 0.3): shaping strength coefficient.
        cast_eps (default 1e-8): RMS denominator protection.
    """
    transform = str(config.get("cast_transform", "asinh"))
    norm = str(config.get("cast_norm", "rms"))
    if norm not in ("rms", "none"):
        raise ValueError(
            f"Unknown cast_norm: '{norm}'. Available: 'rms', 'none'."
        )
    alpha = float(config.get("cast_alpha", 0.3))
    eps = float(config.get("cast_eps", 1e-8))

    mask = response_mask.bool()
    mapped = torch.zeros_like(oracle_advs)

    # --- Step 1: Compute GRPO trajectory advantages ---
    traj_rewards = token_level_scores.sum(dim=-1)
    grpo_advs = _compute_grpo_trajectory_advantages(traj_rewards, uids)

    batch_size = oracle_advs.shape[0]

    # --- Step 2: Collect ALL turn-level oracle advantages in batch ---
    all_turn_advs = []
    for i in range(batch_size):
        step_advs = turn_advantages[i] if i < len(turn_advantages) else None
        if step_advs is not None and hasattr(step_advs, '__len__') and len(step_advs) > 0:
            all_turn_advs.extend(step_advs)

    # --- Step 3 & 4: Compression transform + Batch-level RMS ---
    if len(all_turn_advs) > 0:
        all_g = _oracle_transform(np.array(all_turn_advs, dtype=np.float64), transform)
        rms = np.sqrt(np.mean(all_g ** 2))
    else:
        # Fallback: no oracle info in batch → no shaping possible
        rms = 1.0

    # --- Step 5 & 6 & 7: Normalize + Additive Shaping + Broadcast ---
    for i in range(batch_size):
        row_mask = mask[i]
        if not row_mask.any():
            continue

        base = grpo_advs[i].item()

        # Get per-step oracle advantages
        step_advs = turn_advantages[i] if i < len(turn_advantages) else None
        if step_advs is None or (hasattr(step_advs, '__len__') and len(step_advs) == 0):
            # No oracle info → pure GRPO
            mapped[i][row_mask] = base
            continue

        step_advs = np.array(step_advs, dtype=np.float64)
        T = len(step_advs)
        if T == 0:
            mapped[i][row_mask] = base
            continue

        # Compression transform
        g = _oracle_transform(step_advs, transform)

        # Normalization: "rms" divides by batch RMS (default, identical to
        # mapping_cast); "none" skips it (h = g) for the no-norm ablation.
        h = g / (rms + eps) if norm == "rms" else g

        # Additive shaping: A_final = A_j^GRPO + alpha * h(a_t)
        final_step_values = base + alpha * h

        # Broadcast to token level (step boundaries from response_mask 1-runs).
        token_values = _broadcast_steps_to_tokens(row_mask, final_step_values)
        mapped[i][row_mask] = token_values

    return mapped
