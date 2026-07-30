import hashlib

import numpy as np

from rllm.data.dataset import DatasetRegistry
from rllm.environments.sokoban.utils import (
    derive_retry_seed,
    generate_room,
    seed_context,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


DIFFICULTY_CONFIGS = {
    "id": {
        "dim_x": 6,
        "dim_y": 6,
        "num_boxes": 2,
        "search_depth": 300,
        "max_steps": 100,
        "observation_format": "grid_coord",
    },
    "unseen": {
        "dim_x": 7,
        "dim_y": 7,
        "num_boxes": 3,
        "search_depth": 300,
        "max_steps": 100,
        "observation_format": "grid_coord",
    },
}


def _compute_obs_hash(seed, config, max_seed_validation_retries=50):
    """Compute a hash of the initial observation for a given seed and config.

    Sokoban is fully deterministic after reset -- same initial observation means
    the agent faces the exact same puzzle.  We hash the initial observation to
    detect and remove cross-split duplicates.
    """
    from rllm.environments.sokoban.sokoban import SokobanEnv

    env = SokobanEnv(
        seed=int(seed),
        dim_room=(config["dim_x"], config["dim_y"]),
        num_boxes=config["num_boxes"],
        max_steps=config["max_steps"],
    )
    obs, _ = env.reset()
    return hashlib.md5(obs.encode()).hexdigest()


def prepare_sokoban_data(
    train_size=20000,
    test_size=100,
    difficulty="id",
    difficulty_weights=None,
    max_seed_validation_retries=50,
    deduplicate_observations=True,
):
    """
    Prepare and register Sokoban datasets for training and testing.

    Seeds are drawn without replacement from [0, 1_000_000) so that no seed
    appears in both the train and test splits.

    When ``deduplicate_observations`` is True (default), initial observations
    are hashed and any test sample whose observation already appears in the
    train set is replaced with a fresh sample.  This is important because
    Sokoban is fully deterministic -- same initial observation = same puzzle.

    Args:
        train_size (int): Number of training examples to generate
        test_size (int): Number of test examples to generate
        difficulty (str): Single difficulty level to use ("id", "unseen")
        difficulty_weights (dict): Optional dict mapping difficulty names to weights for mixed sampling.
                                   Example: {"id": 0.7, "unseen": 0.3}
                                   If provided, overrides the `difficulty` parameter.
        max_seed_validation_retries (int): Max retries to find a valid seed per sample.
        deduplicate_observations (bool): If True, ensure no initial-observation overlap between train/test.

    Returns:
        tuple: (train_dataset, test_dataset)
    """
    np.random.seed(42)
    if max_seed_validation_retries < 1:
        raise ValueError(
            f"max_seed_validation_retries must be >= 1, got {max_seed_validation_retries}"
        )

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

    def validate_seed_for_difficulty(initial_seed, difficulty_name):
        config = DIFFICULTY_CONFIGS[difficulty_name]
        current_seed = int(initial_seed)
        last_exc = None
        for _ in range(max_seed_validation_retries):
            try:
                with seed_context(current_seed) as (py_rng, np_rng):
                    generate_room(
                        dim=(config["dim_x"], config["dim_y"]),
                        num_boxes=config["num_boxes"],
                        search_depth=config["search_depth"],
                        py_rng=py_rng,
                        np_rng=np_rng,
                    )
                return current_seed
            except (RuntimeError, RuntimeWarning) as exc:
                last_exc = exc
                current_seed = derive_retry_seed(current_seed)
        raise RuntimeError(
            "Failed to find a valid Sokoban seed for difficulty "
            f"'{difficulty_name}' after {max_seed_validation_retries} retries "
            f"(initial_seed={initial_seed}, last_seed={current_seed})."
        ) from last_exc

    # --- Seed generation: draw without replacement to guarantee no overlap ---
    total_size = train_size + test_size
    # We draw extra seeds to allow for validation failures and observation
    # deduplication replacements.  The pool is large enough (1M) that this
    # is always sufficient.
    pool_size = min(1_000_000, max(total_size * 3, 100_000))
    all_seeds = np.random.choice(1_000_000, size=pool_size, replace=False)

    # Split into non-overlapping train / test / spare pools
    train_initial_seeds = all_seeds[:train_size]
    test_initial_seeds = all_seeds[train_size:train_size + test_size]
    spare_seeds = list(all_seeds[train_size + test_size:])  # for replacements
    spare_idx = 0

    def build_validated_seeds(initial_seeds, sampled_difficulties):
        size = len(initial_seeds)
        validated = np.empty(size, dtype=np.int64)
        progress_desc = f"Validating seeds ({size} samples)"
        for idx, seed in enumerate(
            tqdm(initial_seeds, desc=progress_desc, unit="seed")
        ):
            validated[idx] = validate_seed_for_difficulty(seed, sampled_difficulties[idx])
        return validated

    train_seeds = build_validated_seeds(train_initial_seeds, train_difficulties)
    test_seeds = build_validated_seeds(test_initial_seeds, test_difficulties)

    # --- Observation deduplication for deterministic environments ---
    # Sokoban is fully deterministic: same initial observation = same puzzle.
    # Different seeds can produce identical room layouts due to procedural
    # generation collisions.  We remove duplicates both within each split
    # and across splits (test vs train).
    if deduplicate_observations:
        print("Computing observation hashes for deduplication...")

        # Build train observation hash set with within-train dedup
        train_obs_hashes = set()
        train_within_replaced = 0
        for idx in tqdm(range(len(train_seeds)), desc="Deduplicating train observations"):
            config = DIFFICULTY_CONFIGS[train_difficulties[idx]]
            obs_hash = _compute_obs_hash(train_seeds[idx], config, max_seed_validation_retries)

            if obs_hash in train_obs_hashes:
                # Within-train duplicate; try spare seeds
                found = False
                while spare_idx < len(spare_seeds):
                    new_seed = spare_seeds[spare_idx]
                    spare_idx += 1
                    try:
                        validated = validate_seed_for_difficulty(new_seed, train_difficulties[idx])
                        new_hash = _compute_obs_hash(validated, config, max_seed_validation_retries)
                        if new_hash not in train_obs_hashes:
                            train_seeds[idx] = validated
                            obs_hash = new_hash
                            train_within_replaced += 1
                            found = True
                            break
                    except RuntimeError:
                        continue
                if not found:
                    pass  # exhausted spares; keep duplicate (will be logged)

            train_obs_hashes.add(obs_hash)

        print(f"  Train: {len(train_seeds)} seeds -> {len(train_obs_hashes)} unique observations"
              f" (replaced {train_within_replaced} within-train duplicates)")

        # Check test observations against train set AND within-test, replace collisions
        test_obs_hashes = set()
        test_cross_replaced = 0
        test_within_replaced = 0
        for idx in tqdm(range(len(test_seeds)), desc="Deduplicating test observations"):
            config = DIFFICULTY_CONFIGS[test_difficulties[idx]]
            obs_hash = _compute_obs_hash(test_seeds[idx], config, max_seed_validation_retries)

            collision_type = None
            if obs_hash in train_obs_hashes:
                collision_type = "cross"
            elif obs_hash in test_obs_hashes:
                collision_type = "within"

            if collision_type is not None:
                found = False
                while spare_idx < len(spare_seeds):
                    new_seed = spare_seeds[spare_idx]
                    spare_idx += 1
                    try:
                        validated = validate_seed_for_difficulty(new_seed, test_difficulties[idx])
                        new_hash = _compute_obs_hash(validated, config, max_seed_validation_retries)
                        if new_hash not in train_obs_hashes and new_hash not in test_obs_hashes:
                            test_seeds[idx] = validated
                            obs_hash = new_hash
                            if collision_type == "cross":
                                test_cross_replaced += 1
                            else:
                                test_within_replaced += 1
                            found = True
                            break
                    except RuntimeError:
                        continue
                if not found:
                    pass  # exhausted spares; keep duplicate (will be logged)

            test_obs_hashes.add(obs_hash)

        print(f"  Test: {len(test_seeds)} seeds -> {len(test_obs_hashes)} unique observations"
              f" (replaced {test_cross_replaced} cross-split duplicates,"
              f" {test_within_replaced} within-test duplicates)")

    # Verify no seed overlap (seeds may have been modified by validation retries,
    # but the initial pools were non-overlapping so collisions are unlikely)
    train_seed_set = set(int(s) for s in train_seeds)
    test_seed_set = set(int(s) for s in test_seeds)
    overlap = train_seed_set & test_seed_set
    if overlap:
        print(f"WARNING: {len(overlap)} seed(s) overlap between train/test after validation: {list(overlap)[:10]}")
    else:
        print("Seed overlap check passed: no seed overlap between train and test")

    def sokoban_process_fn(seed, difficulty_name, idx):
        config = DIFFICULTY_CONFIGS[difficulty_name]
        return {
            "seed": int(seed),
            "dim_x": config["dim_x"],
            "dim_y": config["dim_y"],
            "num_boxes": config["num_boxes"],
            "max_steps": config["max_steps"],
            "search_depth": config["search_depth"],
            "observation_format": config["observation_format"],
            "difficulty": difficulty_name,
            "index": idx,
            "uid": f"{seed}_{config['dim_x']}_{config['num_boxes']}_{difficulty_name}",
        }

    train_data = []
    for idx, seed in enumerate(
        tqdm(train_seeds, desc="Building train records", unit="sample")
    ):
        train_data.append(sokoban_process_fn(seed, train_difficulties[idx], idx))

    test_data = []
    for idx, seed in enumerate(
        tqdm(test_seeds, desc="Building test records", unit="sample")
    ):
        test_data.append(sokoban_process_fn(seed, test_difficulties[idx], idx))

    train_dataset = DatasetRegistry.register_dataset("sokoban", train_data, "train")
    test_dataset = DatasetRegistry.register_dataset("sokoban", test_data, "test")

    return train_dataset, test_dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare Sokoban datasets")
    parser.add_argument("--train-size", type=int, default=20000, help="Number of training examples")
    parser.add_argument("--test-size", type=int, default=200, help="Number of test examples")
    parser.add_argument(
        "--difficulty",
        type=str,
        default="id",
        choices=list(DIFFICULTY_CONFIGS.keys()),
        help="Difficulty level",
    )
    parser.add_argument(
        "--mixed",
        action="store_true",
        help="Use mixed difficulty sampling (id:0.7, unseen:0.3)",
    )
    parser.add_argument(
        "--max-seed-validation-retries",
        type=int,
        default=50,
        help="Max retries when validating each generated seed",
    )
    parser.add_argument(
        "--no-obs-dedup",
        action="store_true",
        help="Disable initial-observation deduplication between train and test",
    )
    args = parser.parse_args()

    difficulty_weights = None
    if args.mixed:
        difficulty_weights = {"id": 0.7, "unseen": 0.3}

    train_dataset, test_dataset = prepare_sokoban_data(
        train_size=args.train_size,
        test_size=args.test_size,
        difficulty=args.difficulty,
        difficulty_weights=difficulty_weights,
        max_seed_validation_retries=args.max_seed_validation_retries,
        deduplicate_observations=not args.no_obs_dedup,
    )

    print(f"Train dataset: {len(train_dataset.get_data())} examples")
    print(f"Test dataset: {len(test_dataset.get_data())} examples")
    print("\nDifficulty configurations available:")
    for name, config in DIFFICULTY_CONFIGS.items():
        print(f"  {name}: {config['dim_x']}x{config['dim_y']}, {config['num_boxes']} boxes, search_depth={config['search_depth']}")
    print("\nSample train example:", train_dataset.get_data()[0])
    print("Sample test example:", test_dataset.get_data()[0])
