# scripts/filter_universe.py
# Usage:
# poetry run python scripts/filter_universe.py \
#   --in data/processed/processed_data.cleaned.parquet \
#   --ticker-map data/processed/ticker_map.parquet \
#   --out data/processed/processed_data.eqonly.parquet

import argparse
from pathlib import Path
import pandas as pd
import polars as pl
import numpy as np
import re

# Symbols to exclude (substr matches)
EXCLUDE_SUBSTRINGS = [
    "=F",           # futures
    "-USD", "-EUR", "-USDT",  # common crypto quote suffixes
    "BTC", "ETH", "DOGE", "SHIB",
]

# Hard regexes
EXCLUDE_REGEXES = [
    r".*=X$",       # Yahoo FX pairs, e.g. EURUSD=X
]

# optional: very volatile names to remove in a first pass
OPTIONAL_EXCLUDE = {"MARA","RIOT","CLSK"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, type=Path)
    ap.add_argument("--ticker-map", dest="tickermap", required=True, type=Path)
    ap.add_argument("--out", dest="out", required=True, type=Path)
    args = ap.parse_args()

    df = pd.read_parquet(args.inp)
    tmap = pl.read_parquet(args.tickermap)

    # id -> symbol (ids as str)
    id2sym = dict(zip(tmap["ticker_id"].cast(pl.Utf8).to_list(),
                      tmap["ticker"].to_list()))
    df["ticker_symbol"] = df["ticker_id"].astype(str).map(id2sym).fillna("UNK")

    sym = df["ticker_symbol"].astype(str)

    bad = np.zeros(len(df), dtype=bool)
    for pat in EXCLUDE_SUBSTRINGS:
        bad |= sym.str.contains(pat, regex=False)

    for rx in EXCLUDE_REGEXES:
        bad |= sym.str.contains(rx, regex=True)

    bad |= sym.isin(OPTIONAL_EXCLUDE)

    keep = df.loc[~bad].copy()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    keep.to_parquet(args.out, index=False)

    print(f"Input rows: {len(df):,}")
    print(f"Removed rows: {int(bad.sum()):,}")
    print(f"Output rows: {len(keep):,}")
    print("Examples removed:", sorted(set(sym[bad]))[:20])

if __name__ == "__main__":
    main()
