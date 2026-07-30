#!/usr/bin/env python3
"""Prepare ALFWorld datasets for rLLM training.

This script loads ALFWorld game files and registers them as datasets
for training and evaluation.

Usage:
    python -m examples.alfworld.prepare_alfworld_data --config-path <path_to_config>
    python -m examples.alfworld.prepare_alfworld_data --config-path <path> --split eval_in_distribution
"""

import argparse
import json
import os

from rllm.data.dataset import DatasetRegistry


# Task type mapping
TASK_TYPES = {
    1: "pick_and_place_simple",
    2: "look_at_obj_in_light",
    3: "pick_clean_then_place_in_recep",
    4: "pick_heat_then_place_in_recep",
    5: "pick_cool_then_place_in_recep",
    6: "pick_two_obj_and_place",
}


def load_alfworld_tasks(config_path: str, split: str = "train", task_types: list[int] | None = None) -> list[dict]:
    """Load ALFWorld tasks from game files.

    Args:
        config_path: Path to ALFWorld config YAML file.
        split: Data split ("train", "eval_in_distribution", "eval_out_of_distribution").
        task_types: List of task type IDs to include (1-6). If None, includes all.

    Returns:
        List of task dictionaries.
    """
    import yaml
    from tqdm import tqdm

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Get data path based on split
    if split == "train":
        data_path = os.path.expandvars(config["dataset"]["data_path"])
    elif split == "eval_in_distribution":
        data_path = os.path.expandvars(config["dataset"]["eval_id_data_path"])
    elif split == "eval_out_of_distribution":
        data_path = os.path.expandvars(config["dataset"]["eval_ood_data_path"])
    else:
        raise ValueError(f"Unknown split: {split}")

    # Get task types to include
    if task_types is None:
        task_types = list(TASK_TYPES.keys())
    task_type_names = [TASK_TYPES[tt] for tt in task_types if tt in TASK_TYPES]

    tasks = []
    print(f"Loading ALFWorld tasks from {data_path}...")

    for root, dirs, files in tqdm(list(os.walk(data_path, topdown=False))):
        if "traj_data.json" not in files:
            continue

        # Skip movable and sliced trajectories
        if "movable" in root or "Sliced" in root:
            continue

        json_path = os.path.join(root, "traj_data.json")
        game_file_path = os.path.join(root, "game.tw-pddl")

        # Load trajectory data
        try:
            with open(json_path, "r") as f:
                traj_data = json.load(f)
        except Exception as e:
            print(f"Error loading {json_path}: {e}")
            continue

        # Check task type
        if traj_data.get("task_type") not in task_type_names:
            continue

        # Check game file exists
        if not os.path.exists(game_file_path):
            continue

        # Check if solvable
        try:
            with open(game_file_path, "r") as f:
                gamedata = json.load(f)
            if not gamedata.get("solvable", False):
                continue
        except Exception:
            continue

        # Create task entry
        task = {
            "game_file": game_file_path,
            "task_type": traj_data.get("task_type"),
            "task_id": os.path.basename(root),
            "uid": os.path.basename(root),
            "split": split,
        }
        tasks.append(task)

    print(f"Loaded {len(tasks)} tasks for split={split}")
    return tasks


def prepare_alfworld_data(
    config_path: str,
    train_size: int | None = None,
    test_size: int | None = None,
    task_types: list[int] | None = None,
):
    """Prepare and register ALFWorld datasets.

    Args:
        config_path: Path to ALFWorld config YAML file.
        train_size: Maximum number of training examples. None for all.
        test_size: Maximum number of test examples. None for all.
        task_types: List of task type IDs to include (1-6). If None, includes all.

    Returns:
        Tuple of (train_dataset, test_dataset).
    """
    # Load tasks for each split
    train_tasks = load_alfworld_tasks(config_path, "train", task_types)
    eval_id_tasks = load_alfworld_tasks(config_path, "eval_in_distribution", task_types)
    eval_ood_tasks = load_alfworld_tasks(config_path, "eval_out_of_distribution", task_types)

    # Limit sizes if specified
    if train_size is not None and len(train_tasks) > train_size:
        train_tasks = train_tasks[:train_size]
    if test_size is not None:
        if len(eval_id_tasks) > test_size:
            eval_id_tasks = eval_id_tasks[:test_size]
        if len(eval_ood_tasks) > test_size:
            eval_ood_tasks = eval_ood_tasks[:test_size]

    # Register datasets
    train_dataset = DatasetRegistry.register_dataset("alfworld", train_tasks, "train")
    eval_id_dataset = DatasetRegistry.register_dataset("alfworld", eval_id_tasks, "eval_id")
    eval_ood_dataset = DatasetRegistry.register_dataset("alfworld", eval_ood_tasks, "eval_ood")

    # Also register combined test dataset
    test_tasks = eval_id_tasks + eval_ood_tasks
    test_dataset = DatasetRegistry.register_dataset("alfworld", test_tasks, "test")

    return train_dataset, test_dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare ALFWorld datasets for rLLM")
    parser.add_argument(
        "--config-path",
        type=str,
        default="rllm/environments/alfworld/alfworld_pkg/configs/config_tw.yaml",
        help="Path to ALFWorld config YAML file",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "eval_in_distribution", "eval_out_of_distribution"],
        help="Specific split to load. If not specified, loads all splits.",
    )
    parser.add_argument(
        "--task-types",
        type=int,
        nargs="+",
        default=None,
        help="Task type IDs to include (1-6). If not specified, includes all.",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=None,
        help="Maximum number of training examples",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=None,
        help="Maximum number of test examples",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="alfworld",
        help="Base name for the registered datasets",
    )
    args = parser.parse_args()

    if args.split:
        # Load only specific split
        tasks = load_alfworld_tasks(args.config_path, args.split, args.task_types)
        name = f"{args.dataset_name}_{args.split}"
        dataset = DatasetRegistry.register_dataset(args.dataset_name, tasks, args.split)
        print(f"Registered dataset {name} with {len(dataset)} tasks")
        print(f"\nSample task: {tasks[0] if tasks else 'No tasks'}")
    else:
        # Load all splits
        train_dataset, test_dataset = prepare_alfworld_data(
            config_path=args.config_path,
            train_size=args.train_size,
            test_size=args.test_size,
            task_types=args.task_types,
        )

        print(f"\nDatasets registered:")
        print(f"  Train: {len(train_dataset.get_data())} tasks")
        print(f"  Test: {len(test_dataset.get_data())} tasks")

        print("\nTask types available:")
        for tt_id, tt_name in TASK_TYPES.items():
            print(f"  {tt_id}: {tt_name}")

        if train_dataset.get_data():
            print(f"\nSample train task: {train_dataset.get_data()[0]}")


if __name__ == "__main__":
    main()
