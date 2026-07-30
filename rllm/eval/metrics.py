"""Unified metrics computation for evaluation.

This module provides metrics computation functions for evaluating
agent performance across different environments.

Includes logprob-based metrics:
- Sequence-Level Perplexity (SLP)
- Token-Level Perplexity (TLP)
- Sequence-Level Entropy (SLE)
- Token-Level Entropy (TLE)
- Variance of Acc@K
- Step/Turn/Sequence level trajectory quality metrics
"""

import hashlib
import json
import logging
import math
from collections import defaultdict
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types (ndarray, integer, float, bool)."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


def _task_to_hashable_str(task: dict) -> str:
    """Convert a task dict to a deterministic, hashable JSON string.

    Handles numpy arrays and other non-standard types that appear in
    environment task dicts (e.g. minesweeper's ``mine_positions``).
    """
    task_for_hash = {k: v for k, v in task.items() if k != "_rollout_id"}
    return json.dumps(task_for_hash, sort_keys=True, cls=_NumpyEncoder)


def compute_task_key(task: dict) -> str:
    """Stable identity for a (task, rollout) pair, used for resume/checkpointing.

    Combines the rollout-independent task hash with the ``_rollout_id`` so that
    the N rollouts of the same prompt (see portal.load_data expansion) get
    distinct keys and are NOT treated as duplicates when resuming.

    This is the single source of truth shared by the progress-jsonl writer
    (engine) and the resume filter (portal); keeping it in one place avoids
    the two sides drifting out of sync.
    """
    if not isinstance(task, dict):
        base = json.dumps(str(task), sort_keys=True)
        rollout = 0
    else:
        base = _task_to_hashable_str(task)
        rollout = task.get("_rollout_id", 0)
    return f"{base}::rollout={rollout}"


def _supports_success_metrics(env_name: str) -> bool:
    """Return whether an env should use info['success'] when available."""
    env_name_lower = env_name.lower()
    return (
        "alfworld" in env_name_lower
        or "sokoban" in env_name_lower
        or "babyai" in env_name_lower
        or "twenty_forty_eight" in env_name_lower
        or "2048" in env_name_lower
        or "minihack" in env_name_lower
        or "minesweeper" in env_name_lower
        or "sudoku" in env_name_lower
        or "webshop" in env_name_lower
        or "rush_hour" in env_name_lower
        or "aime" in env_name_lower
        or "hmmt" in env_name_lower
    )


def _extract_success_array(results: list) -> list[bool | None]:
    """Extract trajectory-level success labels from trajectory.info."""
    success_array = []
    for trajectory in results:
        traj_info = getattr(trajectory, "info", {})
        success = traj_info.get("success", None) if isinstance(traj_info, dict) else None
        success_array.append(success)
    return success_array


def _compute_pass_at_k_unbiased(n: int, c: int, k: int) -> float:
    """Compute pass@k using unbiased estimator.

    Formula: pass@k = 1 - C(n-c, k) / C(n, k)
    where C(n, k) is the binomial coefficient "n choose k".

    Args:
        n: Total number of samples for the problem
        c: Number of correct samples
        k: Number of samples to consider

    Returns:
        Unbiased estimate of pass@k for this problem
    """
    if n < k:
        return 0.0
    if c == 0:
        return 0.0
    if n - c < k:
        # If there are fewer incorrect samples than k, we're guaranteed to get
        # at least one correct sample when sampling k items
        return 1.0

    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))


def _extract_final_score(trajectory: Any) -> float | None:
    """Extract final score from trajectory.info or last step info."""
    traj_info = getattr(trajectory, "info", {})
    if isinstance(traj_info, dict):
        score = traj_info.get("score", None)
        if isinstance(score, (int, float)):
            return float(score)

    steps = getattr(trajectory, "steps", [])
    if steps:
        last_info = getattr(steps[-1], "info", {})
        if isinstance(last_info, dict):
            score = last_info.get("score", None)
            if isinstance(score, (int, float)):
                return float(score)
    return None


def _extract_scores(results: list) -> list[float]:
    """Extract valid scores from trajectories."""
    scores = []
    for trajectory in results:
        score = _extract_final_score(trajectory)
        if score is not None:
            scores.append(score)
    return scores


def compute_pass_at_k(
    results: list, return_dict: bool = False, success_threshold: float = 0.0, env_name: str = ""
) -> tuple[float, float] | dict[str, float]:
    """Compute pass@k metrics using unbiased estimator.

    This function computes pass@k for k = 1, 2, 4, 8, ... (powers of 2) using
    the unbiased estimator from the Codex paper:
        pass@k = E[1 - C(n-c, k) / C(n, k)]
    where n is the total number of samples per problem, c is the number of
    correct samples, and C(n, k) is the binomial coefficient.

    Args:
        results: List of Trajectory objects with task and reward attributes.
        return_dict: If True, return a dictionary instead of tuple.
        success_threshold: Reward threshold for success. A trajectory is
            considered successful if reward > success_threshold. Default 0.0
            (i.e., any positive reward is a success). For environments like
            Sokoban where cumulative rewards can be negative even with partial
            progress, set a higher threshold (e.g., 5.0).
        env_name: Environment name. Used to determine if success-based metrics
            should be used (for Sokoban environments).

    Returns:
        Tuple of (pass_at_1, pass_at_k) or dict with pass@k metrics for all
        powers of 2 up to the maximum number of samples per problem.
    """
    env_name_lower = env_name.lower()
    is_2048_env = ("twenty_forty_eight" in env_name_lower) or ("2048" in env_name_lower)

    if not results:
        empty = {
            "pass_at_1": 0.0,
            "pass_at_k": 0.0,
            "total_problems": 0,
            "total_attempts": 0,
            "avg_score": 0.0,
            "max_score": 0.0,
        }
        if is_2048_env:
            empty["score_ge_512_rate"] = 0.0
            empty["score_ge_1024_rate"] = 0.0
            empty["score_ge_2048_rate"] = 0.0
        if return_dict:
            return empty
        return 0.0, 0.0

    success_array = _extract_success_array(results)
    use_success_metrics = any(s is not None for s in success_array) and _supports_success_metrics(
        env_name
    )

    if use_success_metrics:
        print(
            f"[EVAL] Using success-based metrics (info['success']) for environment: {env_name}"
        )
    else:
        print(
            f"[EVAL] Using reward threshold metrics (threshold={success_threshold}) "
            f"for environment: {env_name}"
        )

    # Create maps to store correct answers per problem
    problem_correct_map: defaultdict[str, int] = defaultdict(int)
    problem_total_map: defaultdict[str, int] = defaultdict(int)

    # Count correct answers for each problem
    for idx, trajectory in enumerate(results):
        task = getattr(trajectory, "task", None)

        # Generate hash of problem dict/string
        # Exclude _rollout_id from hash so multiple rollouts of the same problem
        # are grouped together for pass@k computation
        if isinstance(task, dict):
            problem_str = _task_to_hashable_str(task)
        else:
            problem_str = str(task)
        problem_hash = hashlib.md5(problem_str.encode()).hexdigest()

        # Determine if trajectory is correct
        if use_success_metrics:
            # Use success boolean from environment
            success = success_array[idx]
            is_correct = 1 if success is True else 0
        else:
            # Fallback to reward threshold
            is_correct = 1 if getattr(trajectory, "reward", 0) > success_threshold else 0

        problem_correct_map[problem_hash] += is_correct
        problem_total_map[problem_hash] += 1

    # Calculate pass@1 and pass@k
    total_problems = len(problem_correct_map)
    total_attempts = sum(problem_total_map.values())

    # Compute pass@1 as attempt-level success rate.
    # NOTE: This is intentionally different from problem-level pass@1 in
    # pass_at_k_metrics (k=1), and should remain stable for both return modes.
    pass_at_1 = (
        sum(problem_correct_map.values()) / total_attempts if total_attempts > 0 else 0.0
    )

    # Determine maximum k (largest power of 2 <= max samples per problem)
    max_samples_per_problem = max(problem_total_map.values()) if problem_total_map else 1
    k_values = []
    k = 1
    while k <= max_samples_per_problem:
        k_values.append(k)
        k *= 2

    # Compute pass@k for each k using unbiased estimator
    pass_at_k_metrics = {}
    for k in k_values:
        pass_at_k_sum = 0.0
        valid_problems = 0

        for problem_hash in problem_correct_map:
            n = problem_total_map[problem_hash]
            c = problem_correct_map[problem_hash]

            if n >= k:
                pass_at_k_sum += _compute_pass_at_k_unbiased(n, c, k)
                valid_problems += 1

        pass_at_k_value = pass_at_k_sum / valid_problems if valid_problems > 0 else 0.0
        pass_at_k_metrics[f"pass_at_{k}"] = pass_at_k_value

    # For backward compatibility, also compute the old pass@k metric
    # (fraction of problems solved at least once)
    pass_at_k_any = (
        sum(1 for correct in problem_correct_map.values() if correct > 0) / total_problems
        if total_problems > 0
        else 0.0
    )

    scores = _extract_scores(results)
    avg_score = float(np.mean(scores)) if scores else 0.0
    max_score = float(np.max(scores)) if scores else 0.0

    if return_dict:
        metrics = {
            "pass_at_1": pass_at_1,
            "pass_at_k": pass_at_k_any,  # Legacy metric for backward compatibility
            "total_problems": total_problems,
            "total_attempts": total_attempts,
            "avg_score": avg_score,
            "max_score": max_score,
        }
        # Add all pass@k metrics (pass@1, pass@2, pass@4, ...)
        metrics.update(pass_at_k_metrics)
        # Keep pass_at_1 as attempt-level even after merging pass_at_k_metrics.
        metrics["pass_at_1"] = pass_at_1

        if is_2048_env and scores:
            score_arr = np.array(scores)
            metrics["score_ge_512_rate"] = float(np.mean(score_arr >= 512))
            metrics["score_ge_1024_rate"] = float(np.mean(score_arr >= 1024))
            metrics["score_ge_2048_rate"] = float(np.mean(score_arr >= 2048))
        elif is_2048_env:
            metrics["score_ge_512_rate"] = 0.0
            metrics["score_ge_1024_rate"] = 0.0
            metrics["score_ge_2048_rate"] = 0.0
        return metrics
    return pass_at_1, pass_at_k_any


def compute_success_rate(
    results: list, success_threshold: float = 0.0, env_name: str = ""
) -> float:
    """Compute overall success rate.

    Args:
        results: List of Trajectory objects with reward attribute.
        success_threshold: Reward threshold for success. A trajectory is
            considered successful if reward > success_threshold. Default 0.0.
        env_name: Environment name. Used to determine if success-based metrics
            should be used (for Sokoban environments).

    Returns:
        Fraction of trajectories with reward > success_threshold or success=True.
    """
    if not results:
        return 0.0

    success_array = _extract_success_array(results)
    use_success_metrics = any(s is not None for s in success_array) and _supports_success_metrics(
        env_name
    )

    if use_success_metrics:
        # Use success boolean from environment
        successes = sum(1 for s in success_array if s is True)
    else:
        # Fallback to reward threshold
        successes = sum(1 for t in results if getattr(t, "reward", 0) > success_threshold)

    return successes / len(results)


def compute_avg_reward(results: list) -> float:
    """Compute average reward.

    Args:
        results: List of Trajectory objects with reward attribute.

    Returns:
        Average reward across all trajectories.
    """
    if not results:
        return 0.0

    total_reward = sum(getattr(t, "reward", 0) for t in results)
    return total_reward / len(results)


def compute_score(results: list) -> float:
    """Compute average final score from trajectory info."""
    if not results:
        return 0.0
    scores = _extract_scores(results)
    if not scores:
        return 0.0
    return float(np.mean(scores))


def compute_score_stats(
    results: list, success_threshold: float = 0.0, env_name: str = ""
) -> dict[str, float]:
    """Compute score statistics, with 2048 threshold rates when applicable."""
    del success_threshold  # Unused, accepted for compute_metrics compatibility.

    if not results:
        return {
            "avg_score": 0.0,
            "max_score": 0.0,
            "min_score": 0.0,
            "score_p50": 0.0,
        }

    scores = _extract_scores(results)
    if not scores:
        return {
            "avg_score": 0.0,
            "max_score": 0.0,
            "min_score": 0.0,
            "score_p50": 0.0,
        }

    score_arr = np.array(scores)
    metrics = {
        "avg_score": float(np.mean(score_arr)),
        "max_score": float(np.max(score_arr)),
        "min_score": float(np.min(score_arr)),
        "score_p50": float(np.percentile(score_arr, 50)),
    }

    env_name_lower = env_name.lower()
    if ("twenty_forty_eight" in env_name_lower) or ("2048" in env_name_lower):
        metrics["score_ge_512_rate"] = float(np.mean(score_arr >= 512))
        metrics["score_ge_1024_rate"] = float(np.mean(score_arr >= 1024))
        metrics["score_ge_2048_rate"] = float(np.mean(score_arr >= 2048))

    return metrics


def compute_avg_steps(results: list) -> float:
    """Compute average steps taken.

    Args:
        results: List of Trajectory objects with steps attribute.

    Returns:
        Average number of steps across all trajectories.
    """
    if not results:
        return 0.0

    total_steps = sum(len(getattr(t, "steps", [])) for t in results)
    return total_steps / len(results)


def compute_completion_rate(results: list) -> float:
    """Compute completion rate (trajectories that ended with done=True).

    Args:
        results: List of Trajectory objects with steps attribute.

    Returns:
        Fraction of trajectories that completed (done=True).
    """
    if not results:
        return 0.0

    completed = 0
    for t in results:
        steps = getattr(t, "steps", [])
        if steps and getattr(steps[-1], "done", False):
            completed += 1

    return completed / len(results)


def compute_f1_score(results: list, gold_key: str = "answer") -> float:
    """Compute average F1 score for text-based tasks.

    Args:
        results: List of Trajectory objects with task and final answer.
        gold_key: Key in task dict containing the gold answer.

    Returns:
        Average F1 score across all trajectories.
    """
    if not results:
        return 0.0

    def _compute_f1(pred: str, gold: str) -> float:
        """Compute F1 score between prediction and gold."""
        pred_tokens = set(pred.lower().split())
        gold_tokens = set(gold.lower().split())

        if not pred_tokens or not gold_tokens:
            return 0.0

        common = pred_tokens & gold_tokens
        if not common:
            return 0.0

        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gold_tokens)

        return 2 * precision * recall / (precision + recall)

    total_f1 = 0.0
    count = 0

    for t in results:
        task = getattr(t, "task", {})
        if not isinstance(task, dict):
            continue

        gold = task.get(gold_key, "")
        if not gold:
            continue

        # Get the final answer from the last step
        steps = getattr(t, "steps", [])
        if not steps:
            continue

        # Try to get the action from the last step
        last_step = steps[-1]
        pred = getattr(last_step, "action", "") or ""

        if isinstance(pred, str):
            total_f1 += _compute_f1(pred, gold)
            count += 1

    return total_f1 / count if count > 0 else 0.0


def compute_exact_match(results: list, gold_key: str = "answer") -> float:
    """Compute exact match accuracy for text-based tasks.

    Args:
        results: List of Trajectory objects with task and final answer.
        gold_key: Key in task dict containing the gold answer.

    Returns:
        Fraction of trajectories with exact match.
    """
    if not results:
        return 0.0

    def _normalize(text: str) -> str:
        """Normalize text for comparison."""
        return " ".join(text.lower().split())

    matches = 0
    count = 0

    for t in results:
        task = getattr(t, "task", {})
        if not isinstance(task, dict):
            continue

        gold = task.get(gold_key, "")
        if not gold:
            continue

        # Get the final answer from the last step
        steps = getattr(t, "steps", [])
        if not steps:
            continue

        last_step = steps[-1]
        pred = getattr(last_step, "action", "") or ""

        if isinstance(pred, str):
            if _normalize(pred) == _normalize(gold):
                matches += 1
            count += 1

    return matches / count if count > 0 else 0.0


def compute_reward_distribution(results: list) -> dict[str, Any]:
    """Compute reward distribution statistics.

    Args:
        results: List of Trajectory objects with reward attribute.

    Returns:
        Dictionary with min, max, mean, std, and histogram of rewards.
    """
    if not results:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "histogram": {},
        }

    rewards = [getattr(t, "reward", 0) for t in results]

    import statistics

    return {
        "min": min(rewards),
        "max": max(rewards),
        "mean": statistics.mean(rewards),
        "std": statistics.stdev(rewards) if len(rewards) > 1 else 0.0,
        "histogram": _compute_histogram(rewards),
    }


def _compute_histogram(values: list[float], bins: int = 10) -> dict[str, int]:
    """Compute a simple histogram of values.

    Args:
        values: List of numeric values.
        bins: Number of bins.

    Returns:
        Dictionary mapping bin ranges to counts.
    """
    if not values:
        return {}

    min_val = min(values)
    max_val = max(values)

    if min_val == max_val:
        return {f"{min_val:.2f}": len(values)}

    bin_width = (max_val - min_val) / bins
    histogram = {}

    for i in range(bins):
        bin_start = min_val + i * bin_width
        bin_end = bin_start + bin_width
        bin_label = f"{bin_start:.2f}-{bin_end:.2f}"
        histogram[bin_label] = sum(
            1 for v in values if bin_start <= v < bin_end or (i == bins - 1 and v == max_val)
        )

    return histogram


def compute_exploration_coverage(results: list) -> dict[str, Any]:
    """Compute exploration coverage metrics.

    Measures how thoroughly the agent explored the environment by counting
    unique states visited and unique actions tried.

    Args:
        results: List of Trajectory objects from exploration.

    Returns:
        Dictionary with exploration coverage metrics.
    """
    if not results:
        return {
            "unique_observations": 0,
            "unique_actions": 0,
            "avg_steps_per_trajectory": 0.0,
            "total_trajectories": 0,
        }

    all_observations = set()
    all_actions = set()
    total_steps = 0

    for trajectory in results:
        steps = getattr(trajectory, "steps", [])
        total_steps += len(steps)

        for step in steps:
            # Track unique observations
            obs = getattr(step, "observation", None)
            if obs:
                # Use hash of observation string for uniqueness
                obs_str = str(obs)[:500]  # Truncate for efficiency
                all_observations.add(hash(obs_str))

            # Track unique actions
            action = getattr(step, "action", None)
            if action:
                all_actions.add(str(action))

    return {
        "unique_observations": len(all_observations),
        "unique_actions": len(all_actions),
        "avg_steps_per_trajectory": total_steps / len(results) if results else 0.0,
        "total_trajectories": len(results),
    }


def compute_world_modeling_completeness(results: list) -> dict[str, Any]:
    """Compute world modeling completeness metrics.

    Checks if <world_modeling> tags are present in the exploration output
    and measures the quality of the world modeling.

    Args:
        results: List of Trajectory objects from exploration.

    Returns:
        Dictionary with world modeling completeness metrics.
    """
    import re

    if not results:
        return {
            "has_world_modeling": 0,
            "world_modeling_rate": 0.0,
            "avg_content_length": 0.0,
            "total_trajectories": 0,
        }

    has_world_modeling_count = 0
    total_content_length = 0

    for trajectory in results:
        steps = getattr(trajectory, "steps", [])
        trajectory_has_wm = False

        for step in steps:
            response = getattr(step, "model_response", "")
            if response:
                # Check for world_modeling tags
                pattern = r"<world_modeling>(.*?)</world_modeling>"
                match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
                if match:
                    trajectory_has_wm = True
                    total_content_length += len(match.group(1).strip())
                    break

        if trajectory_has_wm:
            has_world_modeling_count += 1

    return {
        "has_world_modeling": has_world_modeling_count,
        "world_modeling_rate": has_world_modeling_count / len(results) if results else 0.0,
        "avg_content_length": total_content_length / has_world_modeling_count if has_world_modeling_count > 0 else 0.0,
        "total_trajectories": len(results),
    }


# =============================================================================
# Logprob-based Metrics for Trajectory Quality Evaluation
# =============================================================================


def _extract_trajectory_logprobs(trajectory) -> tuple[list[float], list[list[float]]]:
    """Extract logprobs from a trajectory.

    Args:
        trajectory: Trajectory object with steps containing logprobs.

    Returns:
        Tuple of (all_logprobs, step_logprobs) where:
        - all_logprobs: Flattened list of all token logprobs in the trajectory
        - step_logprobs: List of logprob lists, one per step
    """
    all_logprobs = []
    step_logprobs = []

    steps = getattr(trajectory, "steps", [])
    for step in steps:
        # Try to get logprobs from step directly
        logprobs = getattr(step, "logprobs", [])
        if not logprobs:
            # Try to get from model_output
            model_output = getattr(step, "model_output", None)
            if model_output is not None:
                logprobs = getattr(model_output, "logprobs", []) or []

        # Filter out None values and ensure floats
        valid_logprobs = [
            float(lp) for lp in logprobs
            if lp is not None and not (isinstance(lp, float) and math.isnan(lp))
        ]

        step_logprobs.append(valid_logprobs)
        all_logprobs.extend(valid_logprobs)

    return all_logprobs, step_logprobs


def _extract_trajectory_entropies(trajectory) -> tuple[list[float], list[list[float]]]:
    """Extract per-token entropies from a trajectory.

    Entropies should be computed from full logits distribution when available.
    Falls back to empty lists if not present.
    """
    all_entropies = []
    step_entropies = []

    steps = getattr(trajectory, "steps", [])
    for step in steps:
        entropies = getattr(step, "token_entropies", [])
        if not entropies:
            model_output = getattr(step, "model_output", None)
            if model_output is not None:
                entropies = getattr(model_output, "token_entropies", []) or []

        valid_entropies = [
            float(e) for e in entropies
            if e is not None and not (isinstance(e, float) and math.isnan(e))
        ]

        step_entropies.append(valid_entropies)
        all_entropies.extend(valid_entropies)

    return all_entropies, step_entropies


def compute_sequence_level_perplexity(results: list) -> dict[str, Any]:
    """Compute Sequence-Level Perplexity (SLP).

    SLP measures the average perplexity of entire trajectories.
    Formula: exp(-sum(logprobs) / num_tokens)

    Lower perplexity indicates the model is more confident in its outputs.

    Args:
        results: List of Trajectory objects with logprobs.

    Returns:
        Dictionary with SLP metrics including mean, std, min, max.
    """
    if not results:
        return {
            "slp_mean": float("nan"),
            "slp_std": float("nan"),
            "slp_min": float("nan"),
            "slp_max": float("nan"),
            "slp_median": float("nan"),
            "trajectories_with_logprobs": 0,
        }

    perplexities = []

    for trajectory in results:
        all_entropies, _ = _extract_trajectory_entropies(trajectory)
        if len(all_entropies) > 0:
            # Full-distribution perplexity: exp(avg entropy)
            avg_entropy = sum(all_entropies) / len(all_entropies)
            perplexity = math.exp(avg_entropy)
            perplexities.append(perplexity)
            continue

        all_logprobs, _ = _extract_trajectory_logprobs(trajectory)
        if len(all_logprobs) > 0:
            # Fallback: exp(-avg(logprobs))
            avg_log_prob = sum(all_logprobs) / len(all_logprobs)
            perplexity = math.exp(-avg_log_prob)
            perplexities.append(perplexity)

    if not perplexities:
        return {
            "slp_mean": float("nan"),
            "slp_std": float("nan"),
            "slp_min": float("nan"),
            "slp_max": float("nan"),
            "slp_median": float("nan"),
            "trajectories_with_logprobs": 0,
        }

    return {
        "slp_mean": float(np.mean(perplexities)),
        "slp_std": float(np.std(perplexities)) if len(perplexities) > 1 else 0.0,
        "slp_min": float(np.min(perplexities)),
        "slp_max": float(np.max(perplexities)),
        "slp_median": float(np.median(perplexities)),
        "trajectories_with_logprobs": len(perplexities),
    }


def compute_token_level_perplexity(results: list) -> dict[str, Any]:
    """Compute Token-Level Perplexity (TLP).

    TLP computes perplexity per token and provides statistics.
    For each token: perplexity = exp(-logprob)

    Args:
        results: List of Trajectory objects with logprobs.

    Returns:
        Dictionary with TLP metrics including mean, std, and percentiles.
    """
    if not results:
        return {
            "tlp_mean": float("nan"),
            "tlp_std": float("nan"),
            "tlp_min": float("nan"),
            "tlp_max": float("nan"),
            "tlp_p25": float("nan"),
            "tlp_p50": float("nan"),
            "tlp_p75": float("nan"),
            "tlp_p95": float("nan"),
            "total_tokens": 0,
        }

    all_token_perplexities = []

    for trajectory in results:
        all_entropies, _ = _extract_trajectory_entropies(trajectory)
        if all_entropies:
            for entropy in all_entropies:
                token_ppl = math.exp(entropy)
                all_token_perplexities.append(token_ppl)
            continue

        all_logprobs, _ = _extract_trajectory_logprobs(trajectory)
        for logprob in all_logprobs:
            # Fallback: exp(-logprob)
            token_ppl = math.exp(-logprob)
            all_token_perplexities.append(token_ppl)

    if not all_token_perplexities:
        return {
            "tlp_mean": float("nan"),
            "tlp_std": float("nan"),
            "tlp_min": float("nan"),
            "tlp_max": float("nan"),
            "tlp_p25": float("nan"),
            "tlp_p50": float("nan"),
            "tlp_p75": float("nan"),
            "tlp_p95": float("nan"),
            "total_tokens": 0,
        }

    perplexities_arr = np.array(all_token_perplexities)

    return {
        "tlp_mean": float(np.mean(perplexities_arr)),
        "tlp_std": float(np.std(perplexities_arr)),
        "tlp_min": float(np.min(perplexities_arr)),
        "tlp_max": float(np.max(perplexities_arr)),
        "tlp_p25": float(np.percentile(perplexities_arr, 25)),
        "tlp_p50": float(np.percentile(perplexities_arr, 50)),
        "tlp_p75": float(np.percentile(perplexities_arr, 75)),
        "tlp_p95": float(np.percentile(perplexities_arr, 95)),
        "total_tokens": len(all_token_perplexities),
    }


def compute_sequence_level_entropy(results: list) -> dict[str, Any]:
    """Compute Sequence-Level Entropy (SLE).

    SLE measures the average entropy (uncertainty) of entire trajectories.
    Formula: -sum(logprobs) / num_tokens = avg(-logprob)

    Higher entropy indicates more uncertainty in the model's outputs.
    Note: SLE is the log of SLP (entropy = log(perplexity))

    Args:
        results: List of Trajectory objects with logprobs.

    Returns:
        Dictionary with SLE metrics including mean, std, min, max.
    """
    if not results:
        return {
            "sle_mean": float("nan"),
            "sle_std": float("nan"),
            "sle_min": float("nan"),
            "sle_max": float("nan"),
            "sle_median": float("nan"),
            "trajectories_with_logprobs": 0,
        }

    entropies = []

    for trajectory in results:
        all_entropies, _ = _extract_trajectory_entropies(trajectory)
        if len(all_entropies) > 0:
            entropy = sum(all_entropies) / len(all_entropies)
            entropies.append(entropy)
            continue

        all_logprobs, _ = _extract_trajectory_logprobs(trajectory)
        if len(all_logprobs) > 0:
            # Fallback: avg(-logprob) = -avg(logprob)
            entropy = -sum(all_logprobs) / len(all_logprobs)
            entropies.append(entropy)

    if not entropies:
        return {
            "sle_mean": float("nan"),
            "sle_std": float("nan"),
            "sle_min": float("nan"),
            "sle_max": float("nan"),
            "sle_median": float("nan"),
            "trajectories_with_logprobs": 0,
        }

    return {
        "sle_mean": float(np.mean(entropies)),
        "sle_std": float(np.std(entropies)) if len(entropies) > 1 else 0.0,
        "sle_min": float(np.min(entropies)),
        "sle_max": float(np.max(entropies)),
        "sle_median": float(np.median(entropies)),
        "trajectories_with_logprobs": len(entropies),
    }


def compute_token_level_entropy(results: list) -> dict[str, Any]:
    """Compute Token-Level Entropy (TLE).

    TLE computes entropy per token and provides statistics.
    For each token: entropy = -logprob

    Args:
        results: List of Trajectory objects with logprobs.

    Returns:
        Dictionary with TLE metrics including mean, std, and percentiles.
    """
    if not results:
        return {
            "tle_mean": float("nan"),
            "tle_std": float("nan"),
            "tle_min": float("nan"),
            "tle_max": float("nan"),
            "tle_p25": float("nan"),
            "tle_p50": float("nan"),
            "tle_p75": float("nan"),
            "tle_p95": float("nan"),
            "total_tokens": 0,
        }

    all_token_entropies = []

    for trajectory in results:
        all_entropies, _ = _extract_trajectory_entropies(trajectory)
        if all_entropies:
            all_token_entropies.extend(all_entropies)
            continue

        all_logprobs, _ = _extract_trajectory_logprobs(trajectory)
        for logprob in all_logprobs:
            # Fallback: per-token entropy = -logprob
            token_entropy = -logprob
            all_token_entropies.append(token_entropy)

    if not all_token_entropies:
        return {
            "tle_mean": float("nan"),
            "tle_std": float("nan"),
            "tle_min": float("nan"),
            "tle_max": float("nan"),
            "tle_p25": float("nan"),
            "tle_p50": float("nan"),
            "tle_p75": float("nan"),
            "tle_p95": float("nan"),
            "total_tokens": 0,
        }

    entropies_arr = np.array(all_token_entropies)

    return {
        "tle_mean": float(np.mean(entropies_arr)),
        "tle_std": float(np.std(entropies_arr)),
        "tle_min": float(np.min(entropies_arr)),
        "tle_max": float(np.max(entropies_arr)),
        "tle_p25": float(np.percentile(entropies_arr, 25)),
        "tle_p50": float(np.percentile(entropies_arr, 50)),
        "tle_p75": float(np.percentile(entropies_arr, 75)),
        "tle_p95": float(np.percentile(entropies_arr, 95)),
        "total_tokens": len(all_token_entropies),
    }


def compute_variance_of_acc_at_k(
    results: list, success_threshold: float = 0.0, env_name: str = ""
) -> dict[str, Any]:
    """Compute Variance of Acc@K.

    This metric measures the variance in accuracy across different problems
    when multiple rollouts are performed per problem.

    Higher variance indicates inconsistent performance across different inputs.

    Args:
        results: List of Trajectory objects. Each trajectory should have:
            - task: The task/problem (used for grouping)
            - reward: The reward (> success_threshold means success)
        success_threshold: Reward threshold for success. Default 0.0.
        env_name: Environment name. Used to determine if success-based metrics
            should be used (for Sokoban environments).

    Returns:
        Dictionary with variance metrics for Acc@K.
    """
    if not results:
        return {
            "acc_at_k_variance": float("nan"),
            "acc_at_k_std": float("nan"),
            "per_problem_accuracy_mean": float("nan"),
            "per_problem_accuracy_std": float("nan"),
            "total_problems": 0,
            "total_rollouts": 0,
        }

    success_array = _extract_success_array(results)
    use_success_metrics = any(s is not None for s in success_array) and _supports_success_metrics(
        env_name
    )

    # Group trajectories by problem
    problem_results: defaultdict[str, list[float]] = defaultdict(list)

    for idx, trajectory in enumerate(results):
        task = getattr(trajectory, "task", None)

        # Generate hash of problem dict/string
        if isinstance(task, dict):
            problem_str = _task_to_hashable_str(task)
        else:
            problem_str = str(task)
        problem_hash = hashlib.md5(problem_str.encode()).hexdigest()

        # Determine if trajectory is correct
        if use_success_metrics:
            success = success_array[idx]
            is_correct = 1.0 if success is True else 0.0
        else:
            is_correct = 1.0 if getattr(trajectory, "reward", 0) > success_threshold else 0.0

        problem_results[problem_hash].append(is_correct)

    if not problem_results:
        return {
            "acc_at_k_variance": float("nan"),
            "acc_at_k_std": float("nan"),
            "per_problem_accuracy_mean": float("nan"),
            "per_problem_accuracy_std": float("nan"),
            "total_problems": 0,
            "total_rollouts": 0,
        }

    # Compute per-problem accuracy
    per_problem_accuracies = []
    for problem_hash, correctness_list in problem_results.items():
        problem_accuracy = sum(correctness_list) / len(correctness_list)
        per_problem_accuracies.append(problem_accuracy)

    accuracies_arr = np.array(per_problem_accuracies)

    return {
        "acc_at_k_variance": float(np.var(accuracies_arr)),
        "acc_at_k_std": float(np.std(accuracies_arr)),
        "per_problem_accuracy_mean": float(np.mean(accuracies_arr)),
        "per_problem_accuracy_std": float(np.std(accuracies_arr)),
        "total_problems": len(problem_results),
        "total_rollouts": len(results),
        "avg_rollouts_per_problem": len(results) / len(problem_results) if problem_results else 0,
    }


def compute_step_level_metrics(results: list) -> dict[str, Any]:
    """Compute Step-Level trajectory quality metrics.

    Analyzes metrics at the step (turn) level within trajectories.

    Args:
        results: List of Trajectory objects.

    Returns:
        Dictionary with step-level metrics including:
        - Per-step perplexity/entropy statistics
        - Step confidence progression
        - Step length statistics
    """
    if not results:
        return {
            "step_ppl_mean": float("nan"),
            "step_ppl_std": float("nan"),
            "step_entropy_mean": float("nan"),
            "step_entropy_std": float("nan"),
            "step_tokens_mean": float("nan"),
            "step_tokens_std": float("nan"),
            "confidence_progression": [],
            "total_steps": 0,
        }

    all_step_perplexities = []
    all_step_entropies = []
    all_step_token_counts = []

    # Track confidence progression (avg entropy per step index)
    step_entropies_by_idx: defaultdict[int, list[float]] = defaultdict(list)

    for trajectory in results:
        _, step_entropies = _extract_trajectory_entropies(trajectory)
        _, step_logprobs = _extract_trajectory_logprobs(trajectory)

        max_steps = max(len(step_entropies), len(step_logprobs))
        for step_idx in range(max_steps):
            entropies = step_entropies[step_idx] if step_idx < len(step_entropies) else []
            logprobs = step_logprobs[step_idx] if step_idx < len(step_logprobs) else []

            if entropies:
                step_entropy = sum(entropies) / len(entropies)
                token_count = len(entropies)
            elif logprobs:
                step_entropy = -sum(logprobs) / len(logprobs)
                token_count = len(logprobs)
            else:
                continue

            all_step_entropies.append(step_entropy)
            step_entropies_by_idx[step_idx].append(step_entropy)

            # Step-level perplexity (from entropy)
            step_ppl = math.exp(step_entropy)
            all_step_perplexities.append(step_ppl)

            # Token count per step
            all_step_token_counts.append(token_count)

    if not all_step_entropies:
        return {
            "step_ppl_mean": float("nan"),
            "step_ppl_std": float("nan"),
            "step_entropy_mean": float("nan"),
            "step_entropy_std": float("nan"),
            "step_tokens_mean": float("nan"),
            "step_tokens_std": float("nan"),
            "confidence_progression": [],
            "total_steps": 0,
        }

    # Compute confidence progression (average entropy at each step index)
    max_step_idx = max(step_entropies_by_idx.keys()) if step_entropies_by_idx else 0
    confidence_progression = []
    for step_idx in range(max_step_idx + 1):
        if step_idx in step_entropies_by_idx:
            avg_entropy = float(np.mean(step_entropies_by_idx[step_idx]))
            confidence_progression.append({
                "step": step_idx,
                "avg_entropy": avg_entropy,
                "avg_confidence": 1.0 / (1.0 + avg_entropy),  # Higher entropy = lower confidence
                "num_trajectories": len(step_entropies_by_idx[step_idx]),
            })

    return {
        "step_ppl_mean": float(np.mean(all_step_perplexities)),
        "step_ppl_std": float(np.std(all_step_perplexities)) if len(all_step_perplexities) > 1 else 0.0,
        "step_entropy_mean": float(np.mean(all_step_entropies)),
        "step_entropy_std": float(np.std(all_step_entropies)) if len(all_step_entropies) > 1 else 0.0,
        "step_tokens_mean": float(np.mean(all_step_token_counts)),
        "step_tokens_std": float(np.std(all_step_token_counts)) if len(all_step_token_counts) > 1 else 0.0,
        "confidence_progression": confidence_progression,
        "total_steps": len(all_step_entropies),
    }


def compute_trajectory_confidence_metrics(
    results: list, success_threshold: float = 0.0, env_name: str = ""
) -> dict[str, Any]:
    """Compute confidence-based trajectory quality metrics.

    Analyzes the relationship between model confidence and task success.

    Args:
        results: List of Trajectory objects.
        success_threshold: Reward threshold for success. Default 0.0.
        env_name: Environment name. Used to determine if success-based metrics
            should be used (for Sokoban environments).

    Returns:
        Dictionary with confidence-related metrics including:
        - Avg confidence for successful vs failed trajectories
        - Confidence calibration (correlation with success)
        - Low/high confidence thresholds
    """
    if not results:
        return {
            "avg_confidence_successful": float("nan"),
            "avg_confidence_failed": float("nan"),
            "confidence_gap": float("nan"),
            "confidence_success_correlation": float("nan"),
            "high_confidence_accuracy": float("nan"),
            "low_confidence_accuracy": float("nan"),
            "total_trajectories": 0,
        }

    success_array = _extract_success_array(results)
    use_success_metrics = any(s is not None for s in success_array) and _supports_success_metrics(
        env_name
    )

    successful_confidences = []
    failed_confidences = []
    all_confidences = []
    all_successes = []

    for idx, trajectory in enumerate(results):
        all_entropies, _ = _extract_trajectory_entropies(trajectory)
        if len(all_entropies) > 0:
            avg_entropy = sum(all_entropies) / len(all_entropies)
        else:
            all_logprobs, _ = _extract_trajectory_logprobs(trajectory)
            if not all_logprobs:
                continue
            avg_entropy = -sum(all_logprobs) / len(all_logprobs)

        # Compute confidence as 1 / (1 + avg_entropy)
        confidence = 1.0 / (1.0 + avg_entropy)

        # Determine if trajectory is successful
        if use_success_metrics:
            success = success_array[idx]
            is_success = success is True
        else:
            is_success = getattr(trajectory, "reward", 0) > success_threshold

        all_confidences.append(confidence)
        all_successes.append(1.0 if is_success else 0.0)

        if is_success:
            successful_confidences.append(confidence)
        else:
            failed_confidences.append(confidence)

    if not all_confidences:
        return {
            "avg_confidence_successful": float("nan"),
            "avg_confidence_failed": float("nan"),
            "confidence_gap": float("nan"),
            "confidence_success_correlation": float("nan"),
            "high_confidence_accuracy": float("nan"),
            "low_confidence_accuracy": float("nan"),
            "total_trajectories": 0,
        }

    avg_conf_success = float(np.mean(successful_confidences)) if successful_confidences else float("nan")
    avg_conf_failed = float(np.mean(failed_confidences)) if failed_confidences else float("nan")

    # Compute correlation between confidence and success
    if len(all_confidences) > 1 and np.std(all_confidences) > 0 and np.std(all_successes) > 0:
        correlation = float(np.corrcoef(all_confidences, all_successes)[0, 1])
    else:
        correlation = float("nan")

    # Compute accuracy for high/low confidence trajectories
    confidence_threshold = np.median(all_confidences)
    high_conf_mask = np.array(all_confidences) >= confidence_threshold
    low_conf_mask = ~high_conf_mask

    high_conf_acc = float(np.mean(np.array(all_successes)[high_conf_mask])) if np.sum(high_conf_mask) > 0 else float("nan")
    low_conf_acc = float(np.mean(np.array(all_successes)[low_conf_mask])) if np.sum(low_conf_mask) > 0 else float("nan")

    return {
        "avg_confidence_successful": avg_conf_success,
        "avg_confidence_failed": avg_conf_failed,
        "confidence_gap": avg_conf_success - avg_conf_failed if not (math.isnan(avg_conf_success) or math.isnan(avg_conf_failed)) else float("nan"),
        "confidence_success_correlation": correlation,
        "high_confidence_accuracy": high_conf_acc,
        "low_confidence_accuracy": low_conf_acc,
        "confidence_threshold": float(confidence_threshold),
        "total_trajectories": len(all_confidences),
    }


def compute_response_length_metrics(results: list) -> dict[str, Any]:
    """Compute response length metrics at trajectory and step levels.

    Args:
        results: List of Trajectory objects.

    Returns:
        Dictionary with length-related metrics.
    """
    if not results:
        return {
            "avg_trajectory_tokens": float("nan"),
            "avg_step_tokens": float("nan"),
            "trajectory_tokens_std": float("nan"),
            "step_tokens_std": float("nan"),
            "total_tokens": 0,
            "total_steps": 0,
        }

    trajectory_token_counts = []
    step_token_counts = []

    for trajectory in results:
        all_logprobs, step_logprobs = _extract_trajectory_logprobs(trajectory)

        trajectory_token_counts.append(len(all_logprobs))

        for logprobs in step_logprobs:
            step_token_counts.append(len(logprobs))

    if not trajectory_token_counts:
        return {
            "avg_trajectory_tokens": float("nan"),
            "avg_step_tokens": float("nan"),
            "trajectory_tokens_std": float("nan"),
            "step_tokens_std": float("nan"),
            "total_tokens": 0,
            "total_steps": 0,
        }

    return {
        "avg_trajectory_tokens": float(np.mean(trajectory_token_counts)),
        "avg_step_tokens": float(np.mean(step_token_counts)) if step_token_counts else float("nan"),
        "trajectory_tokens_std": float(np.std(trajectory_token_counts)) if len(trajectory_token_counts) > 1 else 0.0,
        "step_tokens_std": float(np.std(step_token_counts)) if len(step_token_counts) > 1 else 0.0,
        "total_tokens": sum(trajectory_token_counts),
        "total_steps": len(step_token_counts),
    }


def compute_all_logprob_metrics(
    results: list, success_threshold: float = 0.0, env_name: str = ""
) -> dict[str, Any]:
    """Compute all logprob-based metrics at once.

    This is a convenience function that computes all logprob-based metrics
    and returns them in a single dictionary.

    Args:
        results: List of Trajectory objects with logprobs.
        success_threshold: Reward threshold for success. Default 0.0.
        env_name: Environment name. Used to determine if success-based metrics
            should be used (for Sokoban environments).

    Returns:
        Dictionary containing all logprob-based metrics:
        - Sequence-Level Perplexity (SLP)
        - Token-Level Perplexity (TLP)
        - Sequence-Level Entropy (SLE)
        - Token-Level Entropy (TLE)
        - Variance of Acc@K
        - Step-Level metrics
        - Confidence metrics
        - Response length metrics
    """
    metrics = {}

    # Core perplexity and entropy metrics
    slp_metrics = compute_sequence_level_perplexity(results)
    metrics.update({f"slp/{k}": v for k, v in slp_metrics.items()})

    tlp_metrics = compute_token_level_perplexity(results)
    metrics.update({f"tlp/{k}": v for k, v in tlp_metrics.items()})

    sle_metrics = compute_sequence_level_entropy(results)
    metrics.update({f"sle/{k}": v for k, v in sle_metrics.items()})

    tle_metrics = compute_token_level_entropy(results)
    metrics.update({f"tle/{k}": v for k, v in tle_metrics.items()})

    # Variance of Acc@K
    acc_variance_metrics = compute_variance_of_acc_at_k(
        results, success_threshold=success_threshold, env_name=env_name
    )
    metrics.update({f"acc_variance/{k}": v for k, v in acc_variance_metrics.items()})

    # Step-level metrics
    step_metrics = compute_step_level_metrics(results)
    metrics.update({f"step/{k}": v for k, v in step_metrics.items()})

    # Confidence metrics
    confidence_metrics = compute_trajectory_confidence_metrics(
        results, success_threshold=success_threshold, env_name=env_name
    )
    metrics.update({f"confidence/{k}": v for k, v in confidence_metrics.items()})

    # Response length metrics
    length_metrics = compute_response_length_metrics(results)
    metrics.update({f"length/{k}": v for k, v in length_metrics.items()})

    return metrics


# Mapping of metric names to their computation functions
METRIC_FUNCTIONS = {
    # Core performance metrics
    "pass_at_k": compute_pass_at_k,
    "success_rate": compute_success_rate,
    "avg_reward": compute_avg_reward,
    "score": compute_score,
    "score_stats": compute_score_stats,
    "avg_steps": compute_avg_steps,
    "completion_rate": compute_completion_rate,
    "f1_score": compute_f1_score,
    "exact_match": compute_exact_match,
    "reward_distribution": compute_reward_distribution,
    "exploration_coverage": compute_exploration_coverage,
    "world_modeling_completeness": compute_world_modeling_completeness,
    # Logprob-based metrics
    "sequence_level_perplexity": compute_sequence_level_perplexity,
    "token_level_perplexity": compute_token_level_perplexity,
    "sequence_level_entropy": compute_sequence_level_entropy,
    "token_level_entropy": compute_token_level_entropy,
    "variance_of_acc_at_k": compute_variance_of_acc_at_k,
    "step_level_metrics": compute_step_level_metrics,
    "trajectory_confidence": compute_trajectory_confidence_metrics,
    "response_length": compute_response_length_metrics,
    "all_logprob_metrics": compute_all_logprob_metrics,
}


def compute_metrics(
    results: list,
    metrics_list: list[str] | None = None,
    include_logprob_metrics: bool = False,
    success_threshold: float = 0.0,
    env_name: str = "",
) -> dict[str, Any]:
    """Compute all requested metrics.

    Args:
        results: List of Trajectory objects.
        metrics_list: List of metric names to compute. If None, computes
            pass_at_k, success_rate, avg_reward, and avg_steps.
        include_logprob_metrics: If True, include all logprob-based metrics
            (SLP, TLP, SLE, TLE, etc.) in addition to the requested metrics.
        success_threshold: Reward threshold for success determination.
            Default 0.0 (any positive reward is success). For Sokoban, use 5.0.
        env_name: Environment name. Used to determine if success-based metrics
            should be used (for Sokoban environments).

    Returns:
        Dictionary mapping metric names to their values.
    """
    if metrics_list is None:
        metrics_list = ["pass_at_k", "success_rate", "avg_reward", "avg_steps"]

    # Functions that accept success_threshold and env_name
    _threshold_env_fns = {
        "pass_at_k",
        "success_rate",
        "score_stats",
        "variance_of_acc_at_k",
        "trajectory_confidence",
        "all_logprob_metrics",
    }

    computed = {}

    for metric in metrics_list:
        if metric not in METRIC_FUNCTIONS:
            logger.warning(f"Unknown metric '{metric}', skipping")
            continue

        func = METRIC_FUNCTIONS[metric]

        if metric == "pass_at_k":
            # Special handling for pass_at_k: request full metric dictionary.
            pass_metrics = func(
                results,
                return_dict=True,
                success_threshold=success_threshold,
                env_name=env_name,
            )
            computed.update(pass_metrics)
        elif metric == "all_logprob_metrics":
            # Flatten all_logprob_metrics into the computed dict
            logprob_metrics = func(
                results, success_threshold=success_threshold, env_name=env_name
            )
            computed.update(logprob_metrics)
        elif metric in _threshold_env_fns:
            result = func(results, success_threshold=success_threshold, env_name=env_name)
            if isinstance(result, dict):
                for k, v in result.items():
                    computed[f"{metric}/{k}"] = v
            else:
                computed[metric] = result
        else:
            result = func(results)
            # Handle dict returns by prefixing keys with metric name
            if isinstance(result, dict):
                for k, v in result.items():
                    computed[f"{metric}/{k}"] = v
            else:
                computed[metric] = result

    # Add total count
    computed["total_trajectories"] = len(results)

    # Optionally include all logprob-based metrics
    if include_logprob_metrics and "all_logprob_metrics" not in metrics_list:
        logprob_metrics = compute_all_logprob_metrics(
            results, success_threshold=success_threshold, env_name=env_name
        )
        computed.update(logprob_metrics)

    return computed


def print_metrics(metrics: dict[str, Any], title: str = "EVALUATION RESULTS", compact: bool = False) -> None:
    """Print metrics in a formatted way.

    Args:
        metrics: Dictionary of computed metrics.
        title: Title to display.
        compact: If True, skip detailed sub-metrics like confidence_progression.
    """
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

    # Group metrics by prefix for better organization
    grouped_metrics: defaultdict[str, dict[str, Any]] = defaultdict(dict)
    ungrouped_metrics: dict[str, Any] = {}

    for key, value in metrics.items():
        if "/" in key:
            prefix, suffix = key.split("/", 1)
            grouped_metrics[prefix][suffix] = value
        else:
            ungrouped_metrics[key] = value

    # Print ungrouped metrics first
    for key, value in ungrouped_metrics.items():
        _print_metric_value(key, value, indent=2, compact=compact)

    # Print grouped metrics
    for group_name, group_metrics in grouped_metrics.items():
        print(f"\n  [{group_name.upper()}]")
        for key, value in group_metrics.items():
            _print_metric_value(key, value, indent=4, compact=compact)

    print("=" * 70)


def _print_metric_value(key: str, value: Any, indent: int = 2, compact: bool = False) -> None:
    """Print a single metric value with proper formatting.

    Args:
        key: Metric name.
        value: Metric value.
        indent: Number of spaces for indentation.
        compact: If True, skip detailed lists like confidence_progression.
    """
    prefix = " " * indent

    if isinstance(value, float):
        if math.isnan(value):
            print(f"{prefix}{key}: N/A")
        else:
            print(f"{prefix}{key}: {value:.4f}")
    elif isinstance(value, dict):
        print(f"{prefix}{key}:")
        for k, v in value.items():
            _print_metric_value(k, v, indent=indent + 2, compact=compact)
    elif isinstance(value, list):
        if compact and len(value) > 5:
            print(f"{prefix}{key}: [{len(value)} items]")
        elif all(isinstance(item, dict) for item in value):
            print(f"{prefix}{key}:")
            for i, item in enumerate(value[:10] if compact else value):  # Limit to 10 items in compact mode
                print(f"{prefix}  [{i}]:")
                for k, v in item.items():
                    _print_metric_value(k, v, indent=indent + 4, compact=compact)
            if compact and len(value) > 10:
                print(f"{prefix}  ... and {len(value) - 10} more items")
        else:
            print(f"{prefix}{key}: {value}")


# =============================================================================
# Detailed Token/Step/Trajectory Level Metrics for Analysis
# =============================================================================


def compute_detailed_trajectory_metrics(
    results: list,
    tokenizer=None,
    include_token_text: bool = True,
) -> list[dict[str, Any]]:
    """Compute detailed token/step/trajectory level metrics paired with content.

    This function generates a detailed metrics structure that allows you to
    analyze specific tokens (e.g., "Wait") and understand model behavior
    at a granular level.

    Args:
        results: List of Trajectory objects with logprobs.
        tokenizer: Optional tokenizer to decode token IDs to text.
            If not provided, token text will not be included.
        include_token_text: Whether to include decoded token text.
            Requires tokenizer to be provided.

    Returns:
        List of detailed trajectory metrics, one per trajectory.
        Each contains:
        - trajectory_id: Unique identifier
        - task: Task information
        - reward: Trajectory reward
        - is_success: Whether trajectory succeeded
        - sequence_metrics: Sequence-level aggregated metrics
        - steps: List of step-level data with token details
    """
    detailed_results = []

    for traj_idx, trajectory in enumerate(results):
        traj_id = getattr(trajectory, "uid", f"traj_{traj_idx}")
        task = getattr(trajectory, "task", None)
        reward = getattr(trajectory, "reward", 0.0)
        is_success = reward > 0

        # Compute sequence-level metrics
        all_logprobs = []
        all_entropies = []
        steps_data = []

        steps = getattr(trajectory, "steps", [])
        for step_idx, step in enumerate(steps):
            step_data = _compute_step_metrics(
                step,
                step_idx,
                tokenizer=tokenizer,
                include_token_text=include_token_text,
            )
            steps_data.append(step_data)

            # Collect all logprobs for sequence-level metrics
            if step_data["tokens"]:
                all_logprobs.extend([t["logprob"] for t in step_data["tokens"] if t["logprob"] is not None])
                all_entropies.extend([t["entropy"] for t in step_data["tokens"] if t["entropy"] is not None])

        # Compute sequence-level metrics
        sequence_metrics = _compute_sequence_metrics(all_logprobs, entropies=all_entropies)
        sequence_metrics["num_steps"] = len(steps)
        sequence_metrics["num_tokens"] = len(all_logprobs)

        detailed_results.append({
            "trajectory_id": traj_id,
            "trajectory_idx": traj_idx,
            "task": _sanitize_task(task),
            "reward": float(reward),
            "is_success": is_success,
            "sequence_metrics": sequence_metrics,
            "steps": steps_data,
        })

    return detailed_results


def _compute_step_metrics(
    step,
    step_idx: int,
    tokenizer=None,
    include_token_text: bool = True,
) -> dict[str, Any]:
    """Compute detailed metrics for a single step.

    Args:
        step: Step object.
        step_idx: Index of the step in the trajectory.
        tokenizer: Optional tokenizer for decoding.
        include_token_text: Whether to include decoded token text.

    Returns:
        Dictionary with step-level metrics and token details.
    """
    # Extract basic step info
    observation = getattr(step, "observation", None)
    action = getattr(step, "action", None)
    model_response = getattr(step, "model_response", "")
    thought = getattr(step, "thought", "")
    step_reward = getattr(step, "reward", 0.0)
    done = getattr(step, "done", False)

    # Get token IDs, logprobs, and entropies
    logprobs = getattr(step, "logprobs", [])
    response_ids = getattr(step, "response_ids", [])
    token_entropies = getattr(step, "token_entropies", [])

    # Try to get from model_output if not directly available
    model_output = getattr(step, "model_output", None)
    if model_output is not None:
        if not logprobs:
            logprobs = getattr(model_output, "logprobs", []) or []
        if not token_entropies:
            token_entropies = getattr(model_output, "token_entropies", []) or []
        if not response_ids:
            response_ids = getattr(model_output, "completion_ids", []) or []

    # Compute token-level metrics
    tokens_data = []
    step_logprobs = []
    step_entropies = []

    for i, logprob in enumerate(logprobs):
        if logprob is None or (isinstance(logprob, float) and math.isnan(logprob)):
            continue

        logprob = float(logprob)
        step_logprobs.append(logprob)

        entropy_value = None
        if i < len(token_entropies):
            entropy_value = token_entropies[i]
            if entropy_value is not None and isinstance(entropy_value, float) and math.isnan(entropy_value):
                entropy_value = None
        if entropy_value is None:
            entropy_value = -logprob
        step_entropies.append(entropy_value)

        token_data = {
            "token_idx": i,
            "logprob": logprob,
            "entropy": entropy_value,
            "perplexity": math.exp(entropy_value),
            "confidence": 1.0 / (1.0 + entropy_value),  # Normalized confidence
        }

        # Add token ID if available
        if i < len(response_ids):
            token_data["token_id"] = response_ids[i]

            # Decode token text if tokenizer is provided
            if tokenizer is not None and include_token_text:
                try:
                    token_text = tokenizer.decode([response_ids[i]])
                    token_data["token_text"] = token_text
                except Exception:
                    token_data["token_text"] = None

        tokens_data.append(token_data)

    # Compute step-level aggregated metrics
    step_metrics = _compute_sequence_metrics(step_logprobs, entropies=step_entropies)
    step_metrics["num_tokens"] = len(step_logprobs)

    # Truncate observation for storage if it's too long
    if isinstance(observation, str) and len(observation) > 2000:
        observation_truncated = observation[:2000] + "... [truncated]"
    else:
        observation_truncated = observation

    return {
        "step_idx": step_idx,
        "observation": observation_truncated,
        "action": str(action) if action is not None else None,
        "model_response": model_response[:2000] + "..." if len(model_response) > 2000 else model_response,
        "thought": thought[:1000] + "..." if len(thought) > 1000 else thought,
        "reward": float(step_reward),
        "done": done,
        "step_metrics": step_metrics,
        "tokens": tokens_data,
    }


def _compute_sequence_metrics(
    logprobs: list[float],
    entropies: list[float] | None = None,
) -> dict[str, Any]:
    """Compute aggregated metrics from a list of logprobs.

    Args:
        logprobs: List of log probabilities.

    Returns:
        Dictionary with aggregated metrics.
    """
    if not logprobs and not entropies:
        return {
            "perplexity": float("nan"),
            "entropy_mean": float("nan"),
            "entropy_std": float("nan"),
            "entropy_min": float("nan"),
            "entropy_max": float("nan"),
            "confidence_mean": float("nan"),
            "confidence_std": float("nan"),
        }

    if entropies:
        entropies = [float(e) for e in entropies]
    else:
        entropies = [-lp for lp in logprobs]
    confidences = [1.0 / (1.0 + e) for e in entropies]

    avg_entropy = sum(entropies) / len(entropies)
    perplexity = math.exp(avg_entropy)

    return {
        "perplexity": perplexity,
        "entropy_mean": float(np.mean(entropies)),
        "entropy_std": float(np.std(entropies)) if len(entropies) > 1 else 0.0,
        "entropy_min": float(np.min(entropies)),
        "entropy_max": float(np.max(entropies)),
        "confidence_mean": float(np.mean(confidences)),
        "confidence_std": float(np.std(confidences)) if len(confidences) > 1 else 0.0,
    }


def _sanitize_task(task: Any) -> Any:
    """Sanitize task for JSON serialization by removing large/binary content."""
    if isinstance(task, dict):
        return {k: v for k, v in task.items() if k not in ("image", "images", "screenshot")}
    return task


def search_tokens_by_text(
    detailed_results: list[dict],
    search_text: str,
    case_sensitive: bool = False,
) -> list[dict[str, Any]]:
    """Search for specific tokens by text across all trajectories.

    This allows you to find all occurrences of tokens like "Wait" and
    analyze their metrics.

    Args:
        detailed_results: Output from compute_detailed_trajectory_metrics().
        search_text: Text to search for in tokens.
        case_sensitive: Whether search should be case-sensitive.

    Returns:
        List of matching tokens with their context:
        - trajectory_id
        - step_idx
        - token_idx
        - token_text
        - metrics (logprob, entropy, perplexity, confidence)
        - context (surrounding tokens)
    """
    matches = []

    for traj in detailed_results:
        traj_id = traj["trajectory_id"]
        traj_reward = traj["reward"]

        for step in traj["steps"]:
            step_idx = step["step_idx"]
            tokens = step["tokens"]

            for i, token in enumerate(tokens):
                token_text = token.get("token_text", "")
                if token_text is None:
                    continue

                # Check if search text matches
                if case_sensitive:
                    match = search_text in token_text
                else:
                    match = search_text.lower() in token_text.lower()

                if match:
                    # Get surrounding context (3 tokens before and after)
                    context_start = max(0, i - 3)
                    context_end = min(len(tokens), i + 4)
                    context_tokens = tokens[context_start:context_end]

                    matches.append({
                        "trajectory_id": traj_id,
                        "trajectory_reward": traj_reward,
                        "step_idx": step_idx,
                        "token_idx": token["token_idx"],
                        "token_text": token_text,
                        "logprob": token["logprob"],
                        "entropy": token["entropy"],
                        "perplexity": token["perplexity"],
                        "confidence": token["confidence"],
                        "context": [
                            {
                                "token_text": t.get("token_text"),
                                "logprob": t.get("logprob"),
                                "is_match": t["token_idx"] == token["token_idx"],
                            }
                            for t in context_tokens
                        ],
                    })

    return matches


def aggregate_token_metrics_by_text(
    detailed_results: list[dict],
) -> dict[str, dict[str, Any]]:
    """Aggregate metrics by unique token text.

    This provides statistics for each unique token across all trajectories.

    Args:
        detailed_results: Output from compute_detailed_trajectory_metrics().

    Returns:
        Dictionary mapping token text to aggregated metrics:
        - count: Number of occurrences
        - logprob_mean, logprob_std
        - entropy_mean, entropy_std
        - perplexity_mean, perplexity_std
        - confidence_mean, confidence_std
        - success_rate: Rate of occurrence in successful trajectories
    """
    token_stats: dict[str, dict[str, list]] = defaultdict(
        lambda: {
            "logprobs": [],
            "entropies": [],
            "perplexities": [],
            "confidences": [],
            "in_success": [],
        }
    )

    for traj in detailed_results:
        is_success = traj["is_success"]

        for step in traj["steps"]:
            for token in step["tokens"]:
                token_text = token.get("token_text")
                if token_text is None:
                    continue

                stats = token_stats[token_text]
                stats["logprobs"].append(token["logprob"])
                stats["entropies"].append(token["entropy"])
                stats["perplexities"].append(token["perplexity"])
                stats["confidences"].append(token["confidence"])
                stats["in_success"].append(1 if is_success else 0)

    # Aggregate statistics
    aggregated = {}
    for token_text, stats in token_stats.items():
        if not stats["logprobs"]:
            continue

        aggregated[token_text] = {
            "count": len(stats["logprobs"]),
            "logprob_mean": float(np.mean(stats["logprobs"])),
            "logprob_std": float(np.std(stats["logprobs"])) if len(stats["logprobs"]) > 1 else 0.0,
            "entropy_mean": float(np.mean(stats["entropies"])),
            "entropy_std": float(np.std(stats["entropies"])) if len(stats["entropies"]) > 1 else 0.0,
            "perplexity_mean": float(np.mean(stats["perplexities"])),
            "perplexity_std": float(np.std(stats["perplexities"])) if len(stats["perplexities"]) > 1 else 0.0,
            "confidence_mean": float(np.mean(stats["confidences"])),
            "confidence_std": float(np.std(stats["confidences"])) if len(stats["confidences"]) > 1 else 0.0,
            "success_rate": float(np.mean(stats["in_success"])),
        }

    return aggregated


def get_high_entropy_tokens(
    detailed_results: list[dict],
    threshold: float | None = None,
    top_k: int = 100,
) -> list[dict[str, Any]]:
    """Get tokens with highest entropy (most uncertain predictions).

    Args:
        detailed_results: Output from compute_detailed_trajectory_metrics().
        threshold: Minimum entropy threshold. If None, returns top_k tokens.
        top_k: Number of top tokens to return if threshold is not set.

    Returns:
        List of high-entropy tokens with context.
    """
    all_tokens = []

    for traj in detailed_results:
        traj_id = traj["trajectory_id"]
        traj_reward = traj["reward"]

        for step in traj["steps"]:
            step_idx = step["step_idx"]

            for token in step["tokens"]:
                all_tokens.append({
                    "trajectory_id": traj_id,
                    "trajectory_reward": traj_reward,
                    "step_idx": step_idx,
                    "token_idx": token["token_idx"],
                    "token_text": token.get("token_text"),
                    "token_id": token.get("token_id"),
                    "entropy": token["entropy"],
                    "logprob": token["logprob"],
                    "perplexity": token["perplexity"],
                    "confidence": token["confidence"],
                })

    # Sort by entropy (descending)
    all_tokens.sort(key=lambda x: x["entropy"], reverse=True)

    if threshold is not None:
        return [t for t in all_tokens if t["entropy"] >= threshold]
    else:
        return all_tokens[:top_k]


def get_low_confidence_tokens(
    detailed_results: list[dict],
    threshold: float = 0.5,
    top_k: int = 100,
) -> list[dict[str, Any]]:
    """Get tokens with lowest confidence.

    Args:
        detailed_results: Output from compute_detailed_trajectory_metrics().
        threshold: Maximum confidence threshold.
        top_k: Maximum number of tokens to return.

    Returns:
        List of low-confidence tokens with context.
    """
    all_tokens = []

    for traj in detailed_results:
        traj_id = traj["trajectory_id"]
        traj_reward = traj["reward"]

        for step in traj["steps"]:
            step_idx = step["step_idx"]

            for token in step["tokens"]:
                if token["confidence"] <= threshold:
                    all_tokens.append({
                        "trajectory_id": traj_id,
                        "trajectory_reward": traj_reward,
                        "step_idx": step_idx,
                        "token_idx": token["token_idx"],
                        "token_text": token.get("token_text"),
                        "token_id": token.get("token_id"),
                        "entropy": token["entropy"],
                        "logprob": token["logprob"],
                        "perplexity": token["perplexity"],
                        "confidence": token["confidence"],
                    })

    # Sort by confidence (ascending)
    all_tokens.sort(key=lambda x: x["confidence"])

    return all_tokens[:top_k]


def compare_success_vs_failure_tokens(
    detailed_results: list[dict],
) -> dict[str, Any]:
    """Compare token metrics between successful and failed trajectories.

    This helps identify tokens that are associated with success or failure.

    Args:
        detailed_results: Output from compute_detailed_trajectory_metrics().

    Returns:
        Dictionary with comparison statistics.
    """
    success_tokens = []
    failure_tokens = []

    for traj in detailed_results:
        is_success = traj["is_success"]
        token_list = success_tokens if is_success else failure_tokens

        for step in traj["steps"]:
            for token in step["tokens"]:
                token_list.append({
                    "entropy": token["entropy"],
                    "confidence": token["confidence"],
                    "perplexity": token["perplexity"],
                })

    def compute_stats(tokens):
        if not tokens:
            return {
                "count": 0,
                "entropy_mean": float("nan"),
                "entropy_std": float("nan"),
                "confidence_mean": float("nan"),
                "confidence_std": float("nan"),
                "perplexity_mean": float("nan"),
                "perplexity_std": float("nan"),
            }
        entropies = [t["entropy"] for t in tokens]
        confidences = [t["confidence"] for t in tokens]
        perplexities = [t["perplexity"] for t in tokens]
        return {
            "count": len(tokens),
            "entropy_mean": float(np.mean(entropies)),
            "entropy_std": float(np.std(entropies)),
            "confidence_mean": float(np.mean(confidences)),
            "confidence_std": float(np.std(confidences)),
            "perplexity_mean": float(np.mean(perplexities)),
            "perplexity_std": float(np.std(perplexities)),
        }

    success_stats = compute_stats(success_tokens)
    failure_stats = compute_stats(failure_tokens)

    return {
        "success": success_stats,
        "failure": failure_stats,
        "entropy_diff": success_stats["entropy_mean"] - failure_stats["entropy_mean"]
        if not (math.isnan(success_stats["entropy_mean"]) or math.isnan(failure_stats["entropy_mean"]))
        else float("nan"),
        "confidence_diff": success_stats["confidence_mean"] - failure_stats["confidence_mean"]
        if not (math.isnan(success_stats["confidence_mean"]) or math.isnan(failure_stats["confidence_mean"]))
        else float("nan"),
    }
