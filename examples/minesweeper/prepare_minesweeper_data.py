import hashlib
import json
import random

import numpy as np

from rllm.data.dataset import DatasetRegistry

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


# Difficulty configurations for Minesweeper
DIFFICULTY_CONFIGS = {
    "id": {
        "rows": 6,
        "cols": 6,
        "num_mines": 7,
        "max_steps": 100,
    },
    "unseen": {
        # Unseen-difficulty step on top of the ID tier: a larger board with a
        # slightly higher mine density, which lengthens the reasoning horizon
        # while keeping per-step difficulty close to ID. max_steps stays at 100,
        # which is not a binding constraint at this size.
        "rows": 7,
        "cols": 7,
        "num_mines": 10,
        "max_steps": 100,
    },
}


def _generate_mine_layout(
    seed: int,
    rows: int,
    cols: int,
    num_mines: int,
) -> tuple[list[list[int]], list[int]]:
    """Pre-generate mine positions and a safe first-click for a board.

    Uses the same 3x3 safe-zone logic as ``MinesweeperEnv._setup_mines`` so
    that the resulting layout is valid and consistent with the environment.

    Args:
        seed: RNG seed for deterministic generation.
        rows: Number of rows.
        cols: Number of columns.
        num_mines: Number of mines to place.

    Returns:
        ``(mine_positions, first_click)`` where *mine_positions* is a list of
        ``[r, c]`` pairs and *first_click* is ``[r, c]``.
    """
    rng = random.Random(seed)

    # Pick a random first_click position.
    fc_r = rng.randint(0, rows - 1)
    fc_c = rng.randint(0, cols - 1)
    first_click = [fc_r, fc_c]

    # Build safe zone (3x3 around first_click), same logic as _setup_mines.
    total_cells = rows * cols
    safe_3x3 = {
        (fc_r + dr, fc_c + dc)
        for dr in range(-1, 2)
        for dc in range(-1, 2)
        if 0 <= fc_r + dr < rows and 0 <= fc_c + dc < cols
    }
    available_with_3x3 = total_cells - len(safe_3x3)

    if available_with_3x3 >= num_mines:
        safe_zone = safe_3x3
    elif total_cells - 1 >= num_mines:
        safe_zone = {(fc_r, fc_c)}
    else:
        safe_zone = {(fc_r, fc_c)}
        num_mines = total_cells - 1

    mines: set[tuple[int, int]] = set()
    while len(mines) < num_mines:
        r = rng.randint(0, rows - 1)
        c = rng.randint(0, cols - 1)
        if (r, c) in mines or (r, c) in safe_zone:
            continue
        mines.add((r, c))

    mine_positions = [[r, c] for r, c in sorted(mines)]
    return mine_positions, first_click


def _puzzle_hash(mine_positions: list, first_click: list, rows: int, cols: int) -> str:
    """Hash the full puzzle state (mine layout + first click) for deduplication.

    Minesweeper is deterministic after reveal: same mine_positions + first_click
    = same puzzle.  Different seeds CAN produce identical layouts on small grids.
    """
    canonical = json.dumps({"m": mine_positions, "fc": first_click, "r": rows, "c": cols}, sort_keys=True)
    return hashlib.md5(canonical.encode()).hexdigest()


def prepare_minesweeper_data(
    train_size=20000,
    test_size=200,
    difficulty="id",
    difficulty_weights=None,
    deduplicate_puzzles=True,
):
    """
    Prepare and register Minesweeper datasets for training and testing.

    Seeds are drawn without replacement from [0, 1_000_000) so that no seed
    appears in both the train and test splits.

    When ``deduplicate_puzzles`` is True (default), puzzle hashes (mine_positions
    + first_click) are computed and any test sample whose puzzle already appears
    in the train set is replaced with a fresh sample.

    Each record includes pre-generated ``mine_positions`` and ``first_click``
    so that every rollout of the same data sample starts from an identical
    partially-revealed board.

    Args:
        train_size (int): Number of training examples to generate
        test_size (int): Number of test examples to generate
        difficulty (str): Single difficulty level to use ("id", "unseen")
        difficulty_weights (dict): Optional dict mapping difficulty names to weights
                                   for mixed sampling.
                                   Example: {"id": 0.7, "unseen": 0.3}
                                   If provided, overrides the `difficulty` parameter.
        deduplicate_puzzles (bool): If True, ensure no puzzle overlap between train/test.

    Returns:
        tuple: (train_dataset, test_dataset)
    """
    np.random.seed(42)

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
    pool_size = min(1_000_000, max(total_size * 3, 100_000))
    all_seeds = np.random.choice(1_000_000, size=pool_size, replace=False)
    train_seeds = list(all_seeds[:train_size])
    test_seeds = list(all_seeds[train_size:train_size + test_size])
    spare_seeds = list(all_seeds[train_size + test_size:])

    # Verify no seed overlap
    assert len(set(train_seeds) & set(test_seeds)) == 0, "Seed overlap detected between train and test"
    print(f"Seed overlap check passed: {len(train_seeds)} train + {len(test_seeds)} test, no overlap")

    def _build_record(seed, difficulty_name, idx):
        config = DIFFICULTY_CONFIGS[difficulty_name]
        mine_positions, first_click = _generate_mine_layout(
            seed=int(seed),
            rows=config["rows"],
            cols=config["cols"],
            num_mines=config["num_mines"],
        )
        return {
            "seed": int(seed),
            "rows": config["rows"],
            "cols": config["cols"],
            "num_mines": config["num_mines"],
            "max_steps": config["max_steps"],
            "difficulty": difficulty_name,
            "mine_positions": mine_positions,
            "first_click": first_click,
            "index": idx,
            "uid": f"{seed}_{config['rows']}x{config['cols']}_{config['num_mines']}mines_{difficulty_name}",
        }

    # Build train records with within-train puzzle deduplication
    train_data = []
    train_puzzle_hashes = set()
    spare_idx = 0
    train_within_replaced = 0
    for idx, seed in enumerate(
        tqdm(train_seeds, desc="Building train records", unit="sample")
    ):
        record = _build_record(seed, train_difficulties[idx], idx)
        h = _puzzle_hash(record["mine_positions"], record["first_click"], record["rows"], record["cols"])

        if deduplicate_puzzles and h in train_puzzle_hashes:
            # Within-train duplicate; try spare seeds
            found = False
            while spare_idx < len(spare_seeds):
                spare_seed = spare_seeds[spare_idx]
                spare_idx += 1
                new_record = _build_record(spare_seed, train_difficulties[idx], idx)
                new_h = _puzzle_hash(new_record["mine_positions"], new_record["first_click"], new_record["rows"], new_record["cols"])
                if new_h not in train_puzzle_hashes:
                    record = new_record
                    h = new_h
                    train_within_replaced += 1
                    found = True
                    break
            if not found:
                pass  # exhausted spares; keep duplicate (will be logged)

        train_data.append(record)
        train_puzzle_hashes.add(h)

    if deduplicate_puzzles:
        print(f"  Train: {len(train_data)} records -> {len(train_puzzle_hashes)} unique puzzles"
              f" (replaced {train_within_replaced} within-train duplicates)")

    # Build test records with cross-split and within-test puzzle deduplication
    test_data = []
    test_puzzle_hashes = set()
    test_cross_replaced = 0
    test_within_replaced = 0
    for idx, seed in enumerate(
        tqdm(test_seeds, desc="Building test records", unit="sample")
    ):
        record = _build_record(seed, test_difficulties[idx], idx)
        h = _puzzle_hash(record["mine_positions"], record["first_click"], record["rows"], record["cols"])

        collision_type = None
        if deduplicate_puzzles:
            if h in train_puzzle_hashes:
                collision_type = "cross"
            elif h in test_puzzle_hashes:
                collision_type = "within"

        if collision_type is not None:
            found = False
            while spare_idx < len(spare_seeds):
                spare_seed = spare_seeds[spare_idx]
                spare_idx += 1
                new_record = _build_record(spare_seed, test_difficulties[idx], idx)
                new_h = _puzzle_hash(new_record["mine_positions"], new_record["first_click"], new_record["rows"], new_record["cols"])
                if new_h not in train_puzzle_hashes and new_h not in test_puzzle_hashes:
                    record = new_record
                    h = new_h
                    if collision_type == "cross":
                        test_cross_replaced += 1
                    else:
                        test_within_replaced += 1
                    found = True
                    break
            if not found:
                pass  # exhausted spares; keep duplicate (will be logged)

        test_data.append(record)
        test_puzzle_hashes.add(h)

    if deduplicate_puzzles:
        print(f"  Test: {len(test_data)} records -> {len(test_puzzle_hashes)} unique puzzles"
              f" (replaced {test_cross_replaced} cross-split duplicates,"
              f" {test_within_replaced} within-test duplicates)")

    train_dataset = DatasetRegistry.register_dataset("minesweeper", train_data, "train")
    test_dataset = DatasetRegistry.register_dataset("minesweeper", test_data, "test")

    return train_dataset, test_dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare Minesweeper datasets")
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
        "--no-puzzle-dedup",
        action="store_true",
        help="Disable puzzle deduplication between and within splits",
    )
    args = parser.parse_args()

    difficulty_weights = None
    if args.mixed:
        difficulty_weights = {"id": 0.7, "unseen": 0.3}

    train_dataset, test_dataset = prepare_minesweeper_data(
        train_size=args.train_size,
        test_size=args.test_size,
        difficulty=args.difficulty,
        difficulty_weights=difficulty_weights,
        deduplicate_puzzles=not args.no_puzzle_dedup,
    )

    print(f"Train dataset: {len(train_dataset.get_data())} examples")
    print(f"Test dataset: {len(test_dataset.get_data())} examples")
    print("\nDifficulty configurations available:")
    for name, config in DIFFICULTY_CONFIGS.items():
        print(f"  {name}: {config['rows']}x{config['cols']}, {config['num_mines']} mines, {config['max_steps']} max_steps")
    sample = train_dataset.get_data()[0]
    print("\nSample train example:", sample)
    print(f"  mine_positions ({len(sample['mine_positions'])} mines): {sample['mine_positions']}")
    print(f"  first_click: {sample['first_click']}")
    print("\nSample test example:", test_dataset.get_data()[0])
