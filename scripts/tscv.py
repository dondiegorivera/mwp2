# scripts/tscv.py
# Usage:
#   poetry run python scripts/tscv.py \
#     --processed data/processed/processed_data.eqonly.win.parquet \
#     --folds 5 --embargo 45 --epochs 5
#
# This will run 5 folds with non-overlapping validation windows at the end of the timeline,
# training on all data before the val window start minus an embargo.

import argparse
import subprocess
import sys
from pathlib import Path
import pandas as pd
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True, type=Path)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--embargo", type=int, default=45)
    ap.add_argument("--epochs", type=int, default=5)
    args = ap.parse_args()

    df = pd.read_parquet(args.processed, columns=["date"])
    dates = pd.to_datetime(df["date"].unique())
    dates.sort_values(inplace=True)

    if len(dates) < 300:
        print("Not enough dates for CV.")
        sys.exit(1)

    # carve out last K slices as validation windows
    K = int(args.folds)
    cuts = np.linspace(
        int(len(dates) * 0.5), len(dates), K + 1, dtype=int
    )  # use 2nd half for CV windows
    fold_ranges = [(dates[cuts[i]], dates[cuts[i + 1] - 1]) for i in range(K)]

    print("CV windows:")
    for i, (a, b) in enumerate(fold_ranges, 1):
        print(f"  Fold {i}: val ∈ [{a.date()} .. {b.date()}]")

    for i, (vstart, vend) in enumerate(fold_ranges, 1):
        cutoff = vstart - pd.Timedelta(days=args.embargo)
        exp_id = f"cv5_fold{i}"
        cmd = [
            sys.executable,
            "src/market_prediction_workbench/train.py",
            f"experiment_id={exp_id}",
            f"trainer.epochs={args.epochs}",
            "data.split.use_date_split=true",
            f"data.split.cutoff_date={cutoff.date()}",
            f"data.split.val_end_date={vend.date()}",
            f"data.split.embargo_days={args.embargo}",
        ]
        print("\n==> Running", " ".join(cmd))
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"Fold {i} failed with code {rc}")
            sys.exit(rc)

    print("\nAll folds finished.")


if __name__ == "__main__":
    main()
