# Silence gym's unconditional "unmaintained since 2022" notice printed to stderr
# at import time. gym prints it via gym_notices (no env-var/warnings switch), and
# with forkserver/spawn each worker re-imports this module, so it would repeat
# once per worker. Clearing the notice table BEFORE any import that pulls in gym
# (rush_hour.board imports gym) suppresses it in the parent and every worker.
try:
    import gym_notices.notices as _gym_notices

    _gym_notices.notices = {}
except Exception:  # pragma: no cover - gym_notices absent is fine
    pass

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from rllm.data.dataset import DatasetRegistry
# Reuse the env's difficulty presets so dataset and env never drift apart.
from rllm.environments.rush_hour.rush_hour_env import DIFFICULTY_CONFIGS
from rllm.environments.rush_hour.generator import generate_board
# Share the puzzle-identity hash with the oracle sidecar so dataset dedup and
# sidecar lookups key on byte-identical md5s (they must never drift).
from rllm.environments.rush_hour.oracle import puzzle_md5 as _puzzle_hash

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


def _build_record(seed, difficulty_name, idx):
    """Generate one puzzle record, pinning the exact board via board_config.

    ``generate_board`` is CPU-heavy and may raise RuntimeError when it cannot
    find a puzzle in the difficulty window; callers handle retries by pulling
    fresh seeds from the spare pool.

    This is a PURE function of ``(seed, difficulty_name)`` — it seeds an isolated
    ``random.Random(seed)`` inside ``generate_board`` and touches no module-level
    mutable state — so it is safe to run across processes and yields output
    identical to a serial run regardless of execution order.
    """
    # Normalize the difficulty to a plain str: it may arrive as a numpy.str_
    # (from np.random.choice), and storing that verbatim in the record would
    # serialize differently than a plain-str record, diverging by path.
    difficulty_name = str(difficulty_name)
    config = DIFFICULTY_CONFIGS[difficulty_name]
    # num_walls is optional in the presets; default to 0 when absent so the
    # generator and the env (which rebuilds from board_config) stay aligned.
    num_walls = config.get("num_walls", 0)
    board, min_moves_optimal = generate_board(
        seed=int(seed),
        width=config["width"],
        height=config["height"],
        num_vehicles=config["num_vehicles"],
        num_walls=num_walls,
        min_moves=config["min_moves"],
        max_moves=config["max_moves"],
    )
    return {
        "seed": int(seed),
        "width": config["width"],
        "height": config["height"],
        "num_vehicles": config["num_vehicles"],
        "num_walls": num_walls,
        "min_moves": config["min_moves"],
        "max_moves": config["max_moves"],
        "max_steps": config["max_steps"],
        "observation_format": config["observation_format"],
        "difficulty": difficulty_name,
        # Pin the EXACT puzzle so the env rebuilds it deterministically
        # (the rush_hour analogue of minesweeper storing mine_positions).
        "board_config": board.to_string(),
        # Solver-returned optimal move count (the true difficulty label).
        "min_moves_optimal": int(min_moves_optimal),
        "index": idx,
        "uid": f"{seed}_{config['width']}x{config['height']}_{config['num_vehicles']}v_{difficulty_name}",
    }


def _build_record_safe(task):
    """Process-pool worker: build a record, mapping generation failure to None.

    ``task`` is a ``(seed, difficulty_name, idx)`` tuple. Returns the record, or
    None when generation fails. ``RuntimeError`` is the expected
    "no puzzle in window" signal; any other exception is unexpected, so we print
    it (it would otherwise be hidden inside a worker) and still return None so a
    single bad seed degrades to a spare pull instead of aborting the whole build.
    Must be module-level so it is picklable for ``ProcessPoolExecutor``.
    """
    seed, difficulty_name, idx = task
    try:
        return _build_record(seed, difficulty_name, idx)
    except RuntimeError:
        return None
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[prepare_rush_hour_data] Unexpected error building seed={seed} "
              f"difficulty={difficulty_name} idx={idx}: {type(exc).__name__}: {exc}")
        return None


def _resolve_num_workers(num_workers):
    """Default to min(16, cores-2); never below 1. Capped at 16 (RAM/return)."""
    if num_workers is not None:
        return max(1, int(num_workers))
    cores = os.cpu_count() or 2
    return min(16, max(1, cores - 2))


def prepare_rush_hour_data(
    train_size=20000,
    test_size=200,
    difficulty="medium",
    difficulty_weights=None,
    deduplicate_puzzles=True,
    num_workers=None,
):
    """
    Prepare and register Rush Hour datasets for training and testing.

    Seeds are drawn without replacement from [0, 1_000_000) so that no seed
    appears in both the train and test splits.  Each record pins the EXACT
    generated puzzle via ``board_config`` (the flat 36-char board string) so
    every rollout of the same data sample starts from an identical board.

    When ``deduplicate_puzzles`` is True (default), board strings are hashed and
    any test sample whose puzzle already appears in the train set is replaced
    with a fresh sample drawn from the spare seed pool.

    Generation of the PRIMARY records (the CPU-heavy bulk) is parallelized across
    ``num_workers`` processes. Because ``_build_record`` is a pure function of
    ``(seed, difficulty)``, parallel results are collected in original index
    order and the deduplication + spare-seed consumption stay SERIAL in this
    process, so the output is identical to a single-process run.

    Args:
        train_size (int): Number of training examples to generate.
        test_size (int): Number of test examples to generate.
        difficulty (str): Single difficulty level to use ("easy", "medium", "hard").
        difficulty_weights (dict): Optional dict mapping difficulty names to
            weights for mixed sampling.
            Example: {"easy": 0.5, "medium": 0.3, "hard": 0.2}
            If provided, overrides the `difficulty` parameter.
        deduplicate_puzzles (bool): If True, ensure no puzzle overlap between
            and within splits.
        num_workers (int): Number of worker processes for primary-record
            generation. None auto-selects ``min(16, cores - 2)``; 1 forces a
            serial build (useful for debugging / determinism checks).

    Returns:
        tuple: (train_dataset, test_dataset)
    """
    np.random.seed(42)
    num_workers = _resolve_num_workers(num_workers)

    def sample_difficulty(n_samples):
        """Sample difficulty levels for each example."""
        if difficulty_weights is not None:
            difficulties = list(difficulty_weights.keys())
            weights = [difficulty_weights[d] for d in difficulties]
            # Normalize weights
            weights = np.array(weights) / sum(weights)
            return np.random.choice(difficulties, size=n_samples, p=weights)
        else:
            return np.array([difficulty] * n_samples)

    train_difficulties = sample_difficulty(train_size)
    test_difficulties = sample_difficulty(test_size)

    # --- Seed generation: draw without replacement to guarantee no overlap ---
    total_size = train_size + test_size
    # Draw extra seeds so we can recover from generation failures (RuntimeError)
    # and replace duplicate puzzles.  The 1M pool is always large enough.
    pool_size = min(1_000_000, max(total_size * 3, 100_000))
    all_seeds = np.random.choice(1_000_000, size=pool_size, replace=False)

    # Split into non-overlapping train / test / spare pools.
    train_seeds = list(all_seeds[:train_size])
    test_seeds = list(all_seeds[train_size:train_size + test_size])
    spare_seeds = list(all_seeds[train_size + test_size:])  # for replacements
    spare_idx = 0

    # Verify no seed overlap.
    assert len(set(train_seeds) & set(test_seeds)) == 0, "Seed overlap detected between train and test"
    print(f"Seed overlap check passed: {len(train_seeds)} train + {len(test_seeds)} test, no overlap")
    print(f"Generating primary records with {num_workers} worker process(es)")

    def _build_primary(seeds, difficulties, desc):
        """Build the primary record for every seed, preserving index order.

        Returns a list aligned with ``seeds`` where each entry is a record dict
        or None (generation failed for that seed). The pure ``_build_record_safe``
        runs in a process pool when ``num_workers > 1``; ``chunksize=1`` keeps the
        high per-board time variance well load-balanced.
        """
        tasks = [
            (int(seeds[i]), str(difficulties[i]), i) for i in range(len(seeds))
        ]
        if num_workers <= 1:
            return [
                _build_record_safe(t)
                for t in tqdm(tasks, desc=desc, unit="sample")
            ]
        # executor.map preserves input order, so the returned list is already
        # aligned with ``tasks`` / ``seeds`` — no manual index bookkeeping needed.
        #
        # Use a "forkserver" start method instead of the Linux default "fork".
        # The parent imports gym/numpy and runs a multi-threaded executor manager,
        # and fork() copies only the calling thread while inheriting locks held by
        # the others — those locks have no owner in the child, so workers can
        # deadlock on the call-queue lock (futex_wait) and never run a single task.
        # forkserver spawns workers from a clean single-threaded server process,
        # avoiding the inherited-lock deadlock while keeping full parallelism.
        # Fall back to "spawn" (also single-threaded-safe) if forkserver is
        # unavailable on this platform.
        try:
            ctx = mp.get_context("forkserver")
        except ValueError:
            ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
            return list(
                tqdm(
                    executor.map(_build_record_safe, tasks, chunksize=1),
                    total=len(tasks),
                    desc=desc,
                    unit="sample",
                )
            )

    def _pull_spare(difficulty_name, idx):
        """Return the next successfully-built spare record, advancing the cursor.

        Mirrors the serial spare-seed fallback: pull spare seeds in order, skip
        any that fail to generate (RuntimeError), and return the first board that
        builds. Returns None when the spare pool is exhausted. Kept serial so the
        shared ``spare_idx`` cursor advances in a single, deterministic order.
        """
        nonlocal spare_idx
        while spare_idx < len(spare_seeds):
            spare_seed = spare_seeds[spare_idx]
            spare_idx += 1
            record = _build_record_safe((int(spare_seed), difficulty_name, idx))
            if record is not None:
                return record
        return None

    # --- Build train records with within-train puzzle deduplication ---
    primary_train = _build_primary(train_seeds, train_difficulties, "Building train records")
    train_data = []
    train_puzzle_hashes = set()
    train_within_replaced = 0
    train_within_dropped = 0
    train_failed = 0
    for idx in range(len(train_seeds)):
        record = primary_train[idx]
        if record is None:
            # Primary seed failed to generate; fall back to spare seeds.
            record = _pull_spare(train_difficulties[idx], idx)
        if record is None:
            train_failed += 1
            continue
        h = _puzzle_hash(record["board_config"])

        if deduplicate_puzzles and h in train_puzzle_hashes:
            # Within-train duplicate; try spare seeds for a distinct puzzle.
            resolved = False
            while True:
                new_record = _pull_spare(train_difficulties[idx], idx)
                if new_record is None:
                    break
                new_h = _puzzle_hash(new_record["board_config"])
                if new_h not in train_puzzle_hashes:
                    record = new_record
                    h = new_h
                    train_within_replaced += 1
                    resolved = True
                    break
            if not resolved:
                # Spare pool exhausted without a distinct puzzle. Drop rather than
                # admit a duplicate — deduplication is a hard guarantee.
                train_within_dropped += 1
                continue

        train_data.append(record)
        train_puzzle_hashes.add(h)

    if deduplicate_puzzles:
        print(f"  Train: {len(train_data)} records -> {len(train_puzzle_hashes)} unique puzzles"
              f" (replaced {train_within_replaced} within-train duplicates,"
              f" dropped {train_within_dropped} unresolved duplicates,"
              f" {train_failed} generation failures)")

    # --- Build test records with cross-split and within-test deduplication ---
    primary_test = _build_primary(test_seeds, test_difficulties, "Building test records")
    test_data = []
    test_puzzle_hashes = set()
    test_cross_replaced = 0
    test_within_replaced = 0
    test_dropped = 0
    test_failed = 0
    for idx in range(len(test_seeds)):
        record = primary_test[idx]
        if record is None:
            record = _pull_spare(test_difficulties[idx], idx)
        if record is None:
            test_failed += 1
            continue
        h = _puzzle_hash(record["board_config"])

        collision_type = None
        if deduplicate_puzzles:
            if h in train_puzzle_hashes:
                collision_type = "cross"
            elif h in test_puzzle_hashes:
                collision_type = "within"

        if collision_type is not None:
            resolved = False
            while True:
                new_record = _pull_spare(test_difficulties[idx], idx)
                if new_record is None:
                    break
                new_h = _puzzle_hash(new_record["board_config"])
                if new_h not in train_puzzle_hashes and new_h not in test_puzzle_hashes:
                    record = new_record
                    h = new_h
                    if collision_type == "cross":
                        test_cross_replaced += 1
                    else:
                        test_within_replaced += 1
                    resolved = True
                    break
            if not resolved:
                # Spare pool exhausted without a distinct puzzle. Drop rather than
                # admit a puzzle that collides with train or another test sample —
                # train/test disjointness is a hard guarantee.
                test_dropped += 1
                continue

        test_data.append(record)
        test_puzzle_hashes.add(h)

    if deduplicate_puzzles:
        print(f"  Test: {len(test_data)} records -> {len(test_puzzle_hashes)} unique puzzles"
              f" (replaced {test_cross_replaced} cross-split duplicates,"
              f" {test_within_replaced} within-test duplicates,"
              f" dropped {test_dropped} unresolved duplicates,"
              f" {test_failed} generation failures)")

    # Final hard guarantee: verify the dedup invariant on the actual records
    # before persisting, so a logic regression can never silently ship a dataset
    # with overlapping puzzles.
    if deduplicate_puzzles:
        train_hashes = {_puzzle_hash(r["board_config"]) for r in train_data}
        test_hashes = {_puzzle_hash(r["board_config"]) for r in test_data}
        assert len(train_hashes) == len(train_data), (
            f"Duplicate puzzles within train split: "
            f"{len(train_data) - len(train_hashes)} collisions"
        )
        assert len(test_hashes) == len(test_data), (
            f"Duplicate puzzles within test split: "
            f"{len(test_data) - len(test_hashes)} collisions"
        )
        overlap = train_hashes & test_hashes
        assert not overlap, (
            f"Puzzle overlap between train and test splits: {len(overlap)} puzzles"
        )
        print(f"Dedup invariant verified: {len(train_hashes)} train + "
              f"{len(test_hashes)} test puzzles, no overlap")

    train_dataset = DatasetRegistry.register_dataset("rush_hour", train_data, "train")
    test_dataset = DatasetRegistry.register_dataset("rush_hour", test_data, "test")

    return train_dataset, test_dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare Rush Hour datasets")
    parser.add_argument("--train-size", type=int, default=20000, help="Number of training examples")
    parser.add_argument("--test-size", type=int, default=200, help="Number of test examples")
    parser.add_argument(
        "--difficulty",
        type=str,
        default="medium",
        choices=list(DIFFICULTY_CONFIGS.keys()),
        help="Difficulty level",
    )
    parser.add_argument(
        "--mixed",
        action="store_true",
        help="Use mixed difficulty sampling across all difficulties (equal weights)",
    )
    parser.add_argument(
        "--no-puzzle-dedup",
        action="store_true",
        help="Disable puzzle deduplication between and within splits",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Worker processes for generation (default: min(16, cores - 2); 1 = serial)",
    )
    args = parser.parse_args()

    difficulty_weights = None
    if args.mixed:
        # Equal weight across every available difficulty.
        difficulty_weights = {name: 1.0 for name in DIFFICULTY_CONFIGS}

    train_dataset, test_dataset = prepare_rush_hour_data(
        train_size=args.train_size,
        test_size=args.test_size,
        difficulty=args.difficulty,
        difficulty_weights=difficulty_weights,
        deduplicate_puzzles=not args.no_puzzle_dedup,
        num_workers=args.num_workers,
    )

    print(f"Train dataset: {len(train_dataset.get_data())} examples")
    print(f"Test dataset: {len(test_dataset.get_data())} examples")
    print("\nDifficulty configurations available:")
    for name, config in DIFFICULTY_CONFIGS.items():
        print(f"  {name}: {config['width']}x{config['height']}, {config['num_vehicles']} vehicles, "
              f"moves=[{config['min_moves']},{config['max_moves']}], {config['max_steps']} max_steps")
    sample = train_dataset.get_data()[0]
    print("\nSample train example:", sample)
    print(f"  board_config: {sample['board_config']}")
    print(f"  min_moves_optimal: {sample['min_moves_optimal']}")
    print("\nSample test example:", test_dataset.get_data()[0])
