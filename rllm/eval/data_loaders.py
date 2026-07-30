"""Data loaders for all supported environments.

This module provides unified data loading functions for all environments
supported by the rLLM evaluation portal.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Get the rLLM project root directory
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _PROJECT_ROOT / "data" / "datasets"


def _load_parquet_data(
    env_name: str, split: str = "test", limit: int | None = None,
    data_file: str | None = None,
) -> list[dict[str, Any]]:
    """Load data from a parquet file.

    Args:
        env_name: Name of the environment.
        split: Data split to load (train, test, etc.).
        limit: Maximum number of examples to load.
        data_file: Optional specific file name or relative path under the env data dir
            (e.g. "test_verl", "test_verl.parquet", "medium/test.parquet").
            Overrides the default ``{split}.parquet`` when provided.

    Returns:
        List of dictionaries containing the data.
    """
    if data_file is not None:
        p = Path(data_file)
        if p.is_absolute():
            data_path = p
        else:
            if not p.suffix:
                p = p.with_suffix(".parquet")
            data_path = _DATA_DIR / env_name / p
    else:
        data_path = _DATA_DIR / env_name / f"{split}.parquet"
    if not data_path.exists():
        logger.warning(f"Data file not found: {data_path}")
        return []

    df = pd.read_parquet(data_path)
    data = df.to_dict("records")

    if limit is not None and limit > 0:
        data = data[:limit]

    logger.info(f"Loaded {len(data)} examples from {data_path}")
    return data


def load_webshop_data(
    split: str = "test", limit: int | None = None, **kwargs
) -> list[dict[str, Any]]:
    """Load WebShop dataset.

    Args:
        split: Data split to load (train, test).
        limit: Maximum number of examples to load.
        **kwargs: Additional arguments (ignored).

    Returns:
        List of task dictionaries with session_id and instruction.
    """
    # Try loading from DatasetRegistry first
    try:
        from rllm.data.dataset import DatasetRegistry

        if DatasetRegistry.dataset_exists("webshop", split):
            dataset = DatasetRegistry.load_dataset("webshop", split)
            if dataset is not None:
                data = dataset.get_data()
                if limit is not None and limit > 0:
                    data = data[:limit]
                return data
    except Exception as e:
        logger.debug(f"Could not load from DatasetRegistry: {e}")

    # Fall back to parquet file
    return _load_parquet_data("webshop", split, limit)


def load_alfworld_data(
    split: str = "test", limit: int | None = None, **kwargs
) -> list[dict[str, Any]]:
    """Load ALFWorld dataset.

    Args:
        split: Data split to load (train, test).
        limit: Maximum number of examples to load.
        **kwargs: Additional arguments (ignored).

    Returns:
        List of task dictionaries with task_id, task_type, and game_file.
    """
    # Try loading from DatasetRegistry first
    try:
        from rllm.data.dataset import DatasetRegistry

        if DatasetRegistry.dataset_exists("alfworld", split):
            dataset = DatasetRegistry.load_dataset("alfworld", split)
            if dataset is not None:
                data = dataset.get_data()
                if limit is not None and limit > 0:
                    data = data[:limit]
                return data
    except Exception as e:
        logger.debug(f"Could not load from DatasetRegistry: {e}")

    # Fall back to parquet file
    return _load_parquet_data("alfworld", split, limit)


def load_sokoban_data(
    split: str = "test", limit: int | None = None,
    data_file: str | None = None, **kwargs
) -> list[dict[str, Any]]:
    """Load Sokoban dataset.

    Args:
        split: Data split to load.
        limit: Maximum number of examples to load.
        data_file: Optional specific file name or relative path under
            ``data/datasets/sokoban/`` (e.g. "test_verl", "medium/test.parquet").
            Overrides the default ``{split}.parquet`` when provided.
        **kwargs: Additional arguments (ignored).

    Returns:
        List of task dictionaries with level information.
    """
    # DatasetRegistry path does not support data_file; skip when data_file is set
    if data_file is None:
        try:
            from rllm.data.dataset import DatasetRegistry

            if DatasetRegistry.dataset_exists("sokoban", split):
                dataset = DatasetRegistry.load_dataset("sokoban", split)
                if dataset is not None:
                    data = dataset.get_data()
                    if limit is not None and limit > 0:
                        data = data[:limit]
                    return data
        except Exception as e:
            logger.debug(f"Could not load from DatasetRegistry: {e}")

    # Fall back to parquet file
    return _load_parquet_data("sokoban", split, limit, data_file=data_file)


def load_minesweeper_data(
    split: str = "test", limit: int | None = None,
    data_file: str | None = None, **kwargs
) -> list[dict[str, Any]]:
    """Load Minesweeper dataset.

    Args:
        split: Data split to load.
        limit: Maximum number of examples to load.
        data_file: Optional specific file name or relative path under
            ``data/datasets/minesweeper/`` (e.g. "test_verl", "medium/test.parquet").
            Overrides the default ``{split}.parquet`` when provided.
        **kwargs: Additional arguments (ignored).

    Returns:
        List of task dictionaries with seed, rows, cols, num_mines, etc.
    """
    # DatasetRegistry path does not support data_file; skip when data_file is set
    if data_file is None:
        try:
            from rllm.data.dataset import DatasetRegistry

            if DatasetRegistry.dataset_exists("minesweeper", split):
                dataset = DatasetRegistry.load_dataset("minesweeper", split)
                if dataset is not None:
                    data = dataset.get_data()
                    if limit is not None and limit > 0:
                        data = data[:limit]
                    return data
        except Exception as e:
            logger.debug(f"Could not load from DatasetRegistry: {e}")

    # Fall back to parquet file
    return _load_parquet_data("minesweeper", split, limit, data_file=data_file)


def load_rush_hour_data(
    split: str = "test", limit: int | None = None,
    data_file: str | None = None, **kwargs
) -> list[dict[str, Any]]:
    """Load Rush Hour dataset.

    Args:
        split: Data split to load.
        limit: Maximum number of examples to load.
        data_file: Optional specific file name or relative path under
            ``data/datasets/rush_hour/`` (e.g. "test_verl", "medium/test.parquet").
            Overrides the default ``{split}.parquet`` when provided.
        **kwargs: Additional arguments (ignored).

    Returns:
        List of task dictionaries with seed, width, height, num_vehicles, board_config, etc.
    """
    # DatasetRegistry path does not support data_file; skip when data_file is set
    if data_file is None:
        try:
            from rllm.data.dataset import DatasetRegistry

            if DatasetRegistry.dataset_exists("rush_hour", split):
                dataset = DatasetRegistry.load_dataset("rush_hour", split)
                if dataset is not None:
                    data = dataset.get_data()
                    if limit is not None and limit > 0:
                        data = data[:limit]
                    return data
        except Exception as e:
            logger.debug(f"Could not load from DatasetRegistry: {e}")

    # Fall back to parquet file
    return _load_parquet_data("rush_hour", split, limit, data_file=data_file)


# Mapping of environment names to their data loaders
DATA_LOADERS = {
    "webshop": load_webshop_data,
    "alfworld": load_alfworld_data,
    "sokoban": load_sokoban_data,
    "minesweeper": load_minesweeper_data,
    "rush_hour": load_rush_hour_data,
}


def get_data_loader(env_name: str):
    """Get the data loader function for an environment.

    Args:
        env_name: Name of the environment.

    Returns:
        Data loader function.

    Raises:
        ValueError: If the environment is not supported.
    """
    if env_name not in DATA_LOADERS:
        available = ", ".join(DATA_LOADERS.keys())
        raise ValueError(
            f"Unknown environment '{env_name}'. Available: {available}"
        )
    return DATA_LOADERS[env_name]
