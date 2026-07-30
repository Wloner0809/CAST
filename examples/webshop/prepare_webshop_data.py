#!/usr/bin/env python3
"""Prepare WebShop datasets for rLLM training.

This script creates task entries for WebShop environment based on goal indices.
It supports three data splits (train/val/test) using direct sequential indexing
that is consistent with verl-agent and the original WebShop library.

Goal indices map directly to SimServer.goals[idx], so session_id == goal index.
The SimServer internally shuffles goals with seed=42 (its default), so the
mapping from goal index to actual shopping task is already randomized.

WebShop session index ranges (standard split):
  - Test:  goals[0:500)          (500 goals)
  - Val:   goals[500:1500)       (1000 goals)
  - Train: goals[1500:N)         (remaining goals; N ~= 6910 with synthetic goals)

Note: The SimServer loads ~6910 synthetic goals by default (human_goals=False).
      The exact count depends on the product catalog. All session_ids used in
      the datasets must be < len(SimServer.goals) to avoid modulo collisions.

Usage:
    python -m examples.webshop.prepare_webshop_data
    python -m examples.webshop.prepare_webshop_data --train-size 1000 --test-size 100
    python -m examples.webshop.prepare_webshop_data --use-full-data
"""

import argparse

import numpy as np

from rllm.data.dataset import DatasetRegistry


# Default environment configuration embedded in each task entry,
# mirroring how sokoban stores env config per sample.
DEFAULT_ENV_CONFIG = {
    "observation_mode": "text",
    "max_steps": 15,
    "human_goals": False,
    "data_source": "webshop",
}

# Session index ranges (standard WebShop split).
# These are direct goal indices into SimServer.goals[].
# The SimServer loads ~6910 synthetic goals with default config.
# IMPORTANT: TOTAL_GOALS must match the actual number of goals loaded by
# SimServer. If session_id >= len(goals), the modulo in SimServer.receive()
# will wrap around and cause train/test contamination.
TOTAL_GOALS = 6910  # Approximate; actual count printed at runtime.
TEST_START, TEST_END = 0, 500           # 500 goals (consistent with verl-agent)
VAL_START, VAL_END = 500, 1500          # 1000 goals
TRAIN_START, TRAIN_END = 1500, TOTAL_GOALS  # Remaining goals


def _create_task(
    session_id: int,
    split: str,
    idx: int,
    env_config: dict | None = None,
) -> dict:
    """Create a single task entry for a WebShop session.

    The task dict is stored as ``extra_info`` in verl format and later passed
    to ``WebShopEnv.from_dict()``.  Fields not in the env's ``allowed_keys``
    (e.g. ``data_source``, ``uid``) are silently dropped by ``from_dict``,
    so it is safe to include metadata here.

    Args:
        session_id: WebShop goal/session index.
        split: Data split name ("train", "val", "test").
        idx: Sequential index within the split.
        env_config: Optional env configuration to embed.

    Returns:
        Task dictionary.
    """
    task = {
        "session_id": session_id,
        "uid": f"webshop_{session_id}",
        "split": split,
        "index": idx,
    }
    if env_config:
        task.update(env_config)
    return task


def prepare_webshop_data(
    train_size: int = 1000,
    test_size: int = 500,
    val_size: int = 500,
    seed: int = 42,
    use_full_data: bool = False,
    env_config: dict | None = None,
):
    """Prepare and register WebShop datasets.

    Session IDs are direct goal indices into SimServer.goals[]. No permutation
    is applied because the SimServer already shuffles goals internally with its
    own seed (default 42). This matches verl-agent's convention:
      - test:  goals[0:500)
      - val:   goals[500:1500)
      - train: goals[1500:N)

    Args:
        train_size: Number of training examples (ignored if use_full_data=True).
        test_size: Number of test examples (ignored if use_full_data=True).
        val_size: Number of validation examples (ignored if use_full_data=True).
        seed: Random seed for sampling reproducibility.
        use_full_data: If True, use all available sessions per split.
        env_config: Optional environment configuration to embed in each task.
            Merged with DEFAULT_ENV_CONFIG.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    np.random.seed(seed)

    merged_env_config = {**DEFAULT_ENV_CONFIG}
    if env_config:
        merged_env_config.update(env_config)

    def _sample_indices(start: int, end: int, size: int) -> list[int]:
        """Sample or enumerate indices in [start, end)."""
        pool_size = end - start
        if use_full_data or size >= pool_size:
            return list(range(start, end))
        return np.random.choice(
            range(start, end), size=size, replace=False,
        ).tolist()

    # --- Build index lists (direct goal indices, no permutation) ---
    train_indices = _sample_indices(TRAIN_START, TRAIN_END, train_size)
    val_indices = _sample_indices(VAL_START, VAL_END, val_size)
    test_indices = _sample_indices(TEST_START, TEST_END, test_size)

    # --- Build task entries ---
    train_tasks = [
        _create_task(sid, "train", idx, merged_env_config)
        for idx, sid in enumerate(train_indices)
    ]
    val_tasks = [
        _create_task(sid, "val", idx, merged_env_config)
        for idx, sid in enumerate(val_indices)
    ]
    test_tasks = [
        _create_task(sid, "test", idx, merged_env_config)
        for idx, sid in enumerate(test_indices)
    ]

    # --- Register datasets ---
    train_dataset = DatasetRegistry.register_dataset("webshop", train_tasks, "train")
    val_dataset = DatasetRegistry.register_dataset("webshop", val_tasks, "val")
    test_dataset = DatasetRegistry.register_dataset("webshop", test_tasks, "test")

    return train_dataset, val_dataset, test_dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare WebShop datasets for rLLM")
    parser.add_argument(
        "--train-size",
        type=int,
        default=TRAIN_END - TRAIN_START,
        help="Number of training examples (ignored if --use-full-data is set)",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=TEST_END - TEST_START,
        help="Number of test examples (ignored if --use-full-data is set)",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=VAL_END - VAL_START,
        help="Number of validation examples (ignored if --use-full-data is set)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling reproducibility",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="webshop",
        help="Base name for the registered datasets",
    )
    parser.add_argument(
        "--use-full-data",
        action="store_true",
        help="Use all available sessions instead of sampling (ignores size arguments)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=15,
        help="Max steps per episode (stored in task metadata)",
    )
    parser.add_argument(
        "--observation-mode",
        type=str,
        default="text",
        choices=["text", "html", "text_rich"],
        help="Observation mode (stored in task metadata)",
    )
    args = parser.parse_args()

    env_config = {
        "observation_mode": args.observation_mode,
        "max_steps": args.max_steps,
    }

    train_dataset, val_dataset, test_dataset = prepare_webshop_data(
        train_size=args.train_size,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
        use_full_data=args.use_full_data,
        env_config=env_config,
    )

    sep = "=" * 60
    print(sep)
    print("  WEBSHOP DATASET PREPARATION")
    print(sep)
    print(f"  Train: {len(train_dataset.get_data()):>6} tasks  (goals[{TRAIN_START}:{TRAIN_END}))")
    print(f"  Val:   {len(val_dataset.get_data()):>6} tasks  (goals[{VAL_START}:{VAL_END}))")
    print(f"  Test:  {len(test_dataset.get_data()):>6} tasks  (goals[{TEST_START}:{TEST_END}))")
    print()
    print(f"  TOTAL_GOALS: {TOTAL_GOALS} (SimServer loads ~6910 synthetic goals)")
    print(f"  Indexing: direct (session_id == goal index, no permutation)")
    print(f"  Sampling seed: {args.seed}")
    print(f"  Full data mode: {args.use_full_data}")
    print(f"  Env config: observation_mode={args.observation_mode}, max_steps={args.max_steps}")
    print(sep)

    if train_dataset.get_data():
        print(f"\nSample train task: {train_dataset.get_data()[0]}")
    if val_dataset.get_data():
        print(f"Sample val task:   {val_dataset.get_data()[0]}")
    if test_dataset.get_data():
        print(f"Sample test task:  {test_dataset.get_data()[0]}")

    # Verify no train/test overlap
    train_sids = {t["session_id"] for t in train_dataset.get_data()}
    val_sids = {t["session_id"] for t in val_dataset.get_data()}
    test_sids = {t["session_id"] for t in test_dataset.get_data()}

    train_test_overlap = train_sids & test_sids
    train_val_overlap = train_sids & val_sids
    val_test_overlap = val_sids & test_sids

    print(f"\n  Overlap check:")
    print(f"    Train-Test overlap: {len(train_test_overlap)} session_ids")
    print(f"    Train-Val overlap:  {len(train_val_overlap)} session_ids")
    print(f"    Val-Test overlap:   {len(val_test_overlap)} session_ids")

    if train_test_overlap or train_val_overlap or val_test_overlap:
        print("  WARNING: Split contamination detected!")
    else:
        print("  All splits are clean (no overlap).")


if __name__ == "__main__":
    main()
