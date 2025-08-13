# scripts/winsorize_targets.py
# Usage:
# poetry run python scripts/winsorize_targets.py \
#   --in data/processed/processed_data.eqonly.parquet \
#   --out data/processed/processed_data.eqonly.win.parquet \
#   --train-cutoff 0.8

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

TARGET_COLS = ["target_1d", "target_5d", "target_20d"]
# per-horizon tail caps (two-sided). Tweak if needed.
CAPS = {
    "target_1d": 99.5,  # clip |1d| at 99.5th abs-quantile
    "target_5d": 99.2,
    "target_20d": 99.0,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, type=Path)
    ap.add_argument("--out", dest="out", required=True, type=Path)
    ap.add_argument(
        "--train-cutoff",
        type=float,
        default=0.8,
        help="fractional split by time_idx per ticker",
    )
    args = ap.parse_args()

    df = pd.read_parquet(args.inp)
    assert "time_idx" in df.columns and "ticker_id" in df.columns

    out_chunks = []
    for tid, g in df.groupby("ticker_id", sort=False):
        g = g.sort_values("time_idx")
        if len(g) < 50:
            out_chunks.append(g)
            continue

        cut = int(np.floor(len(g) * args.train_cutoff))
        train_g = g.iloc[:cut]

        # compute caps on TRAIN only
        caps_for_tid = {}
        for col in TARGET_COLS:
            if col not in train_g.columns:
                continue
            x = train_g[col].to_numpy()
            x = x[np.isfinite(x)]
            if x.size == 0:
                continue
            q = CAPS[col]
            hi = np.nanpercentile(np.abs(x), q)
            caps_for_tid[col] = (-hi, hi)

        # apply those caps to ALL rows of this ticker
        for col, (lo, hi) in caps_for_tid.items():
            g[col] = g[col].clip(lower=lo, upper=hi)

        out_chunks.append(g)

    df2 = pd.concat(out_chunks, axis=0, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df2.to_parquet(args.out, index=False)

    print(f"Input rows: {len(df):,}  ->  Output rows: {len(df2):,}")
    print(f"Capped columns: {', '.join(TARGET_COLS)}")
    # quick sanity check
    for c in TARGET_COLS:
        if c in df2.columns:
            q = np.nanpercentile(np.abs(df2[c].to_numpy()), [95, 99, 99.9])
            print(f"{c}: |q95|={q[0]:.4f}, |q99|={q[1]:.4f}, |q99.9|={q[2]:.4f}")


if __name__ == "__main__":
    main()
