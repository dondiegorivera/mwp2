# scripts/clean_targets.py
# Usage:
#   poetry run python scripts/clean_targets.py \
#     --in data/processed/processed_data.parquet \
#     --out data/processed/processed_data.cleaned.parquet \
#     --train-cutoff 0.8 \
#     --mode drop         # or: clip
#
# Writes:
#   <out> cleaned parquet
#   <out>.clean_report.json with caps and drop counts

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

TARGETS = ["target_1d", "target_5d", "target_20d"]


def compute_caps(df: pd.DataFrame, train_cutoff: float) -> dict:
    """
    Use the earliest `train_cutoff` fraction of time to estimate robust caps:
    cap_t = 99.9th percentile of |target_t|.
    """
    # Infer time axis
    if "time_idx" not in df.columns:
        raise RuntimeError("expected a 'time_idx' column in parquet")
    tmin, tmax = df["time_idx"].min(), df["time_idx"].max()
    split = int(tmin + (tmax - tmin) * train_cutoff)

    train = df[df["time_idx"] <= split]
    caps = {}
    for col in TARGETS:
        if col in train.columns:
            caps[col] = float(train[col].abs().quantile(0.999))
        else:
            caps[col] = None
    return caps


def clean_df(df: pd.DataFrame, caps: dict, mode: str):
    """
    If mode == 'drop': remove rows beyond caps.
    If mode == 'clip': winsorize to +/- caps.
    Returns cleaned df and a small report.
    """
    report = {"caps": caps, "dropped": {}, "clipped": {}}

    mask_ok = np.ones(len(df), dtype=bool)
    for col in TARGETS:
        cap = caps.get(col)
        if cap is None or col not in df.columns:
            continue
        if mode == "drop":
            bad = df[col].abs() > cap
            report["dropped"][col] = int(bad.sum())
            mask_ok &= ~bad
        else:  # clip
            before = df[col].copy()
            df[col] = df[col].clip(lower=-cap, upper=cap)
            report["clipped"][col] = int((before != df[col]).sum())

    if mode == "drop":
        df = df[mask_ok].copy()

    # Optional: flag missing rows
    if "is_missing" in df.columns:
        # leave as-is
        pass

    return df, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, type=Path)
    ap.add_argument("--out", dest="out", required=True, type=Path)
    ap.add_argument(
        "--train-cutoff",
        type=float,
        default=0.8,
        help="fraction of earliest time_idx for cap estimation",
    )
    ap.add_argument("--mode", choices=["drop", "clip"], default="drop")
    args = ap.parse_args()

    df = pd.read_parquet(args.inp)
    for req in ["time_idx"] + TARGETS:
        if req not in df.columns:
            print(f"Warning: '{req}' not in data.")

    caps = compute_caps(df, args.train_cutoff)
    cleaned, report = clean_df(df, caps, args.mode)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(args.out, index=False)
    with open(str(args.out) + ".clean_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== Clean targets ===")
    print(f"Input rows:  {len(df):,}")
    print(f"Output rows: {len(cleaned):,}")
    print("Caps (|value| ≤ cap):")
    for k, v in caps.items():
        print(f"  {k}: {v}")
    if args.mode == "drop":
        print("Dropped beyond caps:")
        for k, v in report["dropped"].items():
            print(f"  {k}: {v}")
    else:
        print("Clipped beyond caps:")
        for k, v in report["clipped"].items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
