#!/usr/bin/env python3
"""Validate train/test data separation for all game environments.

Usage:
    python3 tests/data_inspect/validate_train_test_separation.py

    # Validate a single environment
    python3 tests/data_inspect/validate_train_test_separation.py --env sokoban

    # Limit samples for quick smoke testing
    python3 tests/data_inspect/validate_train_test_separation.py --limit 500
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "datasets"

# ---------------------------------------------------------------------------
# Helper: load train/test parquets for a given env
# ---------------------------------------------------------------------------


def _load_splits(env_name: str, limit: int | None = None):
    """Load train and test parquet files for *env_name*.

    Returns ``(train_df, test_df)`` or ``(None, None)`` if data is missing.
    """
    env_dir = DATA_DIR / env_name
    train_path = env_dir / "train.parquet"
    test_path = env_dir / "test.parquet"

    if not train_path.exists() or not test_path.exists():
        print(f"  [SKIP] {env_name}: missing parquet files at {env_dir}")
        return None, None

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    if limit is not None:
        train_df = train_df.head(limit)
        test_df = test_df.head(limit)

    return train_df, test_df


# ---------------------------------------------------------------------------
# Seed overlap check (applicable to ALL environments)
# ---------------------------------------------------------------------------


def check_seed_overlap(train_df: pd.DataFrame, test_df: pd.DataFrame, env_name: str) -> bool:
    """Return True if no seed overlap is found."""
    train_seeds = set(train_df["seed"].tolist())
    test_seeds = set(test_df["seed"].tolist())
    overlap = train_seeds & test_seeds

    if overlap:
        print(f"  [FAIL] {env_name}: {len(overlap)} seed(s) overlap between train/test")
        print(f"         Overlapping seeds (first 10): {sorted(overlap)[:10]}")
        return False

    print(f"  [PASS] {env_name}: no seed overlap ({len(train_seeds)} train, {len(test_seeds)} test)")
    return True


# ---------------------------------------------------------------------------
# Observation overlap check (for deterministic environments)
# ---------------------------------------------------------------------------


def _hash_sokoban_obs(row) -> str:
    """Instantiate a SokobanEnv and hash its initial observation."""
    from rllm.environments.sokoban.sokoban import SokobanEnv

    env = SokobanEnv(
        seed=int(row["seed"]),
        dim_room=(int(row["dim_x"]), int(row["dim_y"])),
        num_boxes=int(row["num_boxes"]),
        max_steps=int(row.get("max_steps", 100)),
    )
    obs, _ = env.reset()
    return hashlib.md5(obs.encode()).hexdigest()


def _to_python(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    import numpy as np

    if isinstance(obj, np.ndarray):
        # Handle object-dtype arrays (e.g., array of arrays from parquet)
        if obj.dtype == object:
            return [_to_python(x) for x in obj]
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, list):
        return [_to_python(x) for x in obj]
    return obj


def _hash_minesweeper_puzzle(row) -> str:
    """Hash the Minesweeper puzzle state (mine_positions + first_click).

    Unlike Sokoban where the initial observation is the full puzzle,
    Minesweeper hides its mines.  Two boards with the same visible observation
    but different hidden mine layouts are genuinely different puzzles.  We
    therefore hash the *full* puzzle state rather than the visible observation.
    """
    mine_positions = row.get("mine_positions")
    first_click = row.get("first_click")

    # Handle serialized JSON strings from parquet
    if isinstance(mine_positions, str):
        mine_positions = json.loads(mine_positions)
    if isinstance(first_click, str):
        first_click = json.loads(first_click)

    # Convert numpy types to native Python for JSON serialization
    mine_positions = _to_python(mine_positions)
    first_click = _to_python(first_click)

    canonical = json.dumps({
        "m": mine_positions,
        "fc": first_click,
        "r": int(row["rows"]),
        "c": int(row["cols"]),
    }, sort_keys=True)
    return hashlib.md5(canonical.encode()).hexdigest()


def check_observation_overlap(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    env_name: str,
    hash_fn,
) -> bool:
    """Return True if no observation overlap is found between train and test."""
    from tqdm import tqdm

    print(f"  Computing observation hashes for {env_name}...")

    train_hashes = set()
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc=f"  Hashing {env_name} train obs"):
        try:
            h = hash_fn(row)
            train_hashes.add(h)
        except Exception as e:
            print(f"    Warning: failed to hash train row {row.get('seed', '?')}: {e}")

    unique_train_rate = len(train_hashes) / len(train_df) * 100 if len(train_df) > 0 else 0
    print(f"  Train: {len(train_df)} samples -> {len(train_hashes)} unique observations ({unique_train_rate:.1f}%)")

    test_hashes = set()
    overlap_count = 0
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"  Hashing {env_name} test obs"):
        try:
            h = hash_fn(row)
            test_hashes.add(h)
            if h in train_hashes:
                overlap_count += 1
        except Exception as e:
            print(f"    Warning: failed to hash test row {row.get('seed', '?')}: {e}")

    unique_test_rate = len(test_hashes) / len(test_df) * 100 if len(test_df) > 0 else 0
    print(f"  Test:  {len(test_df)} samples -> {len(test_hashes)} unique observations ({unique_test_rate:.1f}%)")

    if overlap_count > 0:
        pct = overlap_count / len(test_df) * 100
        print(f"  [FAIL] {env_name}: {overlap_count} test observation(s) ({pct:.1f}%) collide with train set")
        return False

    print(f"  [PASS] {env_name}: no observation overlap between train and test")
    return True


# ---------------------------------------------------------------------------
# Per-environment validators
# ---------------------------------------------------------------------------

HASH_FUNCTIONS = {
    "sokoban": _hash_sokoban_obs,
    "minesweeper": _hash_minesweeper_puzzle,
}

STOCHASTIC_ENVS = set()

ALL_ENVS = ["sokoban", "minesweeper"]


def validate_env(env_name: str, limit: int | None = None) -> bool:
    """Validate a single environment. Returns True if all checks pass."""
    print(f"\n{'='*60}")
    print(f"  Validating: {env_name}")
    print(f"{'='*60}")

    train_df, test_df = _load_splits(env_name, limit=limit)
    if train_df is None:
        return True  # Skip, not fail

    all_ok = True

    # Seed overlap check (all envs)
    if "seed" in train_df.columns:
        if not check_seed_overlap(train_df, test_df, env_name):
            all_ok = False
    else:
        print(f"  [SKIP] {env_name}: no 'seed' column found")

    # Observation overlap check (deterministic envs only)
    if env_name in HASH_FUNCTIONS:
        hash_fn = HASH_FUNCTIONS[env_name]
        if not check_observation_overlap(train_df, test_df, env_name, hash_fn):
            all_ok = False
    elif env_name in STOCHASTIC_ENVS:
        print(f"  [INFO] {env_name}: stochastic environment, skipping observation overlap check")
        print(f"         (seed uniqueness is sufficient for meaningful train/test separation)")
    else:
        print(f"  [SKIP] {env_name}: no hash function registered, skipping observation check")

    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Validate train/test data separation for game environments"
    )
    parser.add_argument(
        "--env",
        type=str,
        choices=ALL_ENVS,
        default=None,
        help="Validate a single environment (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples per split for quick smoke testing",
    )
    args = parser.parse_args()

    envs = [args.env] if args.env else ALL_ENVS
    results = {}

    for env_name in envs:
        try:
            results[env_name] = validate_env(env_name, limit=args.limit)
        except Exception as e:
            print(f"\n  [ERROR] {env_name}: {e}")
            import traceback
            traceback.print_exc()
            results[env_name] = False

    # Summary
    print(f"\n{'='*60}")
    print("  VALIDATION SUMMARY")
    print(f"{'='*60}")
    all_passed = True
    for env_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {env_name:25s} [{status}]")
        if not passed:
            all_passed = False
    print(f"{'='*60}")

    if all_passed:
        print("\n  All checks PASSED.")
    else:
        print("\n  Some checks FAILED. See details above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
