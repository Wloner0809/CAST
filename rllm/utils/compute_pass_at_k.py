import os
from collections import defaultdict

import torch


def _extract_final_score(trajectory):
    """Extract score from trajectory metadata, falling back to last step info."""
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


def compute_pass_at_k(results, success_threshold=0.0, env_name="", return_dict=False):
    import hashlib
    import json

    if not results:
        empty_metrics = {
            "pass_at_1": 0.0,
            "pass_at_k": 0.0,
            "total_problems": 0,
            "total_attempts": 0,
            "avg_score": 0.0,
            "max_score": 0.0,
        }
        print("[EVAL] No trajectories provided, metrics are all zero.")
        if return_dict:
            return empty_metrics
        return 0.0, 0.0

    env_name_lower = env_name.lower()
    # Check if we should use success-based metrics
    success_array = []
    for trajectory in results:
        traj_info = getattr(trajectory, "info", {})
        success = traj_info.get("success", None) if isinstance(traj_info, dict) else None
        success_array.append(success)

    # Use success-based metrics for environments that expose success booleans.
    supports_success_metrics = ("sokoban" in env_name_lower) or ("twenty_forty_eight" in env_name_lower) or ("2048" in env_name_lower)
    use_success_metrics = any(s is not None for s in success_array) and supports_success_metrics

    if use_success_metrics:
        print(f"[EVAL] Using success-based metrics (info['success']) for environment: {env_name}")
    else:
        print(
            f"[EVAL] Using reward threshold metrics (threshold={success_threshold}) "
            f"for environment: {env_name}"
        )

    # Create a map to store correct answers per problem
    problem_correct_map: defaultdict[str, int] = defaultdict(int)
    problem_total_map: defaultdict[str, int] = defaultdict(int)

    # Count correct answers for each problem
    for idx, trajectory in enumerate(results):
        task = trajectory.task

        # Generate hash of problem dict/string
        if isinstance(task, dict):
            task_for_hash = {k: v for k, v in task.items() if k != "_rollout_id"}
            problem_str = json.dumps(task_for_hash, sort_keys=True)
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
            is_correct = 1 if trajectory.reward > success_threshold else 0

        problem_correct_map[problem_hash] += is_correct
        problem_total_map[problem_hash] += 1

    # Calculate pass@1 and pass@k.
    total_problems = len(problem_correct_map)
    total_attempts = sum(problem_total_map.values())
    pass_at_1 = sum(problem_correct_map.values()) / total_attempts if total_attempts > 0 else 0.0
    pass_at_k = (
        sum(1 for _, correct in problem_correct_map.items() if correct > 0) / total_problems
        if total_problems > 0
        else 0.0
    )

    # Score metrics are important for sparse-success environments like 2048.
    score_array = []
    for trajectory in results:
        score = _extract_final_score(trajectory)
        if score is not None:
            score_array.append(score)

    avg_score = sum(score_array) / len(score_array) if score_array else 0.0
    max_score = max(score_array) if score_array else 0.0

    print("Total unique problems:", total_problems)
    print("Total attempts:", total_attempts)
    print("Average Pass@1 Accuracy:", pass_at_1)
    print("Average Pass@k Accuracy:", pass_at_k)
    if score_array:
        print("Average Score:", avg_score)
        print("Max Score:", max_score)

        if ("twenty_forty_eight" in env_name_lower) or ("2048" in env_name_lower):
            total_scores = len(score_array)
            reach_512 = sum(1 for s in score_array if s >= 512) / total_scores
            reach_1024 = sum(1 for s in score_array if s >= 1024) / total_scores
            reach_2048 = sum(1 for s in score_array if s >= 2048) / total_scores
            print("Score>=512 Rate:", reach_512)
            print("Score>=1024 Rate:", reach_1024)
            print("Score>=2048 Rate:", reach_2048)
    else:
        print("Average Score: N/A (score not found in trajectory info)")
        print("Max Score: N/A (score not found in trajectory info)")

    metrics = {
        "pass_at_1": pass_at_1,
        "pass_at_k": pass_at_k,
        "total_problems": total_problems,
        "total_attempts": total_attempts,
        "avg_score": avg_score,
        "max_score": max_score,
    }
    if return_dict:
        return metrics
    return pass_at_1, pass_at_k


def save_trajectories(results, save_dir="./trajectories", filename="trajectories.pt"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    torch.save(results, save_path)
    print(f"Trajectories saved to {save_path}")
    return save_path
