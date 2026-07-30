"""Build the offline N(s) sidecar for a Rush Hour dataset.

The sidecar maps ``puzzle_md5(board_config) -> exact N(s) table`` for every board
in a dataset parquet, so the online oracle (``oracle_backend="precomputed"``)
becomes a lookup with no per-step solver call.

Run this AFTER generating the dataset parquet (``prepare_rush_hour_data.py``):
    PYTHONPATH=. python3 examples/rush_hour/build_oracle_sidecar.py \
        --parquet data/datasets/rush_hour/medium/train.parquet \
        --output  data/datasets/rush_hour/medium/train_oracle.pkl.gz

The board geometry (width/height) is read from the dataset rows themselves so
the sidecar always matches the boards it was built for.
"""

import argparse
import os

import pandas as pd

from rllm.environments.rush_hour.oracle import build_sidecar, save_sidecar

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Rush Hour oracle N(s) sidecar")
    parser.add_argument("--parquet", required=True, help="Dataset parquet with a board_config column")
    parser.add_argument("--output", required=True, help="Output sidecar path (.pkl.gz)")
    parser.add_argument(
        "--max-states",
        type=int,
        default=1_000_000,
        help="Component-size cap; boards exceeding it are stored as None (online IDA* fallback)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Worker processes for the per-board BFS (default: min(16, cores - 2); 1 = serial)",
    )
    parser.add_argument(
        "--oracle-potential",
        choices=["moves", "cells"],
        default="moves",
        help="Potential function N(s) the table encodes (moves=slide count, cells=slid cell count)",
    )
    args = parser.parse_args()

    num_workers = args.num_workers
    if num_workers is None:
        cores = os.cpu_count() or 2
        num_workers = min(16, max(1, cores - 2))

    df = pd.read_parquet(args.parquet)
    if "board_config" not in df.columns:
        raise ValueError(
            f"{args.parquet} has no 'board_config' column (columns: {list(df.columns)})"
        )

    # The sidecar build applies ONE geometry to every board, so all rows must
    # share it — otherwise off-size boards would be parsed at the wrong width and
    # tabulated incorrectly. Enforce uniformity rather than silently mis-build.
    if "width" in df.columns and df["width"].nunique() > 1:
        raise ValueError(f"{args.parquet} mixes board widths {sorted(df['width'].unique())}")
    if "height" in df.columns and df["height"].nunique() > 1:
        raise ValueError(f"{args.parquet} mixes board heights {sorted(df['height'].unique())}")

    board_configs = [str(b) for b in df["board_config"].tolist()]
    width = int(df["width"].iloc[0]) if "width" in df.columns else 6
    height = int(df["height"].iloc[0]) if "height" in df.columns else 6

    unique = len(set(board_configs))
    print(
        f"Building sidecar from {args.parquet}: {len(board_configs)} rows "
        f"({unique} unique puzzles), geometry {width}x{height}, "
        f"{num_workers} worker process(es)"
    )

    payload = build_sidecar(
        board_configs,
        width=width,
        height=height,
        max_states=args.max_states,
        progress=lambda it: tqdm(it, total=unique, desc="Tabulating puzzles", unit="puzzle"),
        num_workers=num_workers,
        potential=args.oracle_potential,
    )
    save_sidecar(payload, args.output)

    tables = payload["tables"]
    too_large = sum(1 for t in tables.values() if t is None)
    total_states = sum(len(t) for t in tables.values() if t is not None)
    print(
        f"Wrote {args.output}: {len(tables)} puzzles "
        f"(potential={args.oracle_potential}, {too_large} too large -> None, "
        f"{total_states} total states tabulated)"
    )


if __name__ == "__main__":
    main()
