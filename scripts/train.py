from __future__ import annotations

import argparse
import os
import sys

from house_price_prediction.config import load_settings


def _resolve_trainer(model_type: str):
    if model_type == "lightgbm":
        from house_price_prediction.model import train_and_save_model as lightgbm_trainer
        return lightgbm_trainer
    if model_type in {"random_forest", "random-forest", "rf"}:
        from house_price_prediction.model_random_forest import (
            train_and_save_model as random_forest_trainer,
        )
        return random_forest_trainer
    raise ValueError(
        f"Unsupported MODEL_TYPE '{model_type}'. Use 'lightgbm' or 'random_forest'."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the house price prediction model.")
    parser.add_argument(
        "--min-rows",
        type=int,
        default=None,
        help="Minimum number of training rows required. Overrides TRAINING_MIN_ROWS env var.",
    )
    args = parser.parse_args()

    # Allow --min-rows CLI flag to override the env var before loading settings
    if args.min_rows is not None:
        os.environ["TRAINING_MIN_ROWS"] = str(args.min_rows)

    load_settings.cache_clear()
    settings = load_settings()

    # Guard: count rows before training
    from pathlib import Path
    import json

    raw_path = settings.raw_data_path
    min_rows = settings.training_min_rows
    if min_rows and min_rows > 0:
        suffix = raw_path.suffix.lower()
        if suffix == ".jsonl":
            # Count non-empty lines
            try:
                with raw_path.open() as f:
                    row_count = sum(1 for line in f if line.strip())
            except FileNotFoundError:
                print(f"ERROR: Training data not found at {raw_path}", file=sys.stderr)
                sys.exit(1)
        elif suffix == ".csv":
            import pandas as pd
            try:
                row_count = len(pd.read_csv(raw_path, usecols=[0]))
            except FileNotFoundError:
                print(f"ERROR: Training data not found at {raw_path}", file=sys.stderr)
                sys.exit(1)
        else:
            row_count = min_rows  # unknown format, skip guard

        if row_count < min_rows:
            print(
                f"ERROR: Training data has only {row_count} rows at {raw_path}; "
                f"minimum required is {min_rows}. "
                "Run 'make ingest-csv' or 'make refresh-live-pipeline' first.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Training rows check: {row_count} >= {min_rows}")

    train_and_save_model = _resolve_trainer(settings.model_type)
    metrics = train_and_save_model(settings)

    print("Training complete")
    print(f"Saved model: {settings.model_path}")
    print("Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")