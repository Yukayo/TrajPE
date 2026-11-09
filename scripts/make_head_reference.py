import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from trajpe.data import load_split

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nyc")
    ap.add_argument("--k", type=int, default=500)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    df = load_split(args.dataset, "train").sort_values(["trail_id", "pos"])
    head = set(df["venue_id"].value_counts().index[: args.k])
    keep = df["venue_id"].isin(head)
    seg = (~keep).groupby(df["trail_id"]).cumsum()
    df = df[keep].assign(trail_id=lambda d: d["trail_id"] + "~" + seg[keep].astype(str))
    n = df.groupby("trail_id")["venue_id"].transform("size")
    df = df[n >= 2].copy()
    df["pos"] = df.groupby("trail_id").cumcount()
    df.to_parquet(args.out, index=False)
    print(f"{args.dataset} head-{args.k}: {df['trail_id'].nunique()} trails, {len(df)} check-ins")
