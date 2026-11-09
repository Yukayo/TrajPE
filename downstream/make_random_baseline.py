from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from trajpe.paths import PROCESSED

REFERENCE_MONDAY = pd.Timestamp("2024-01-01")
SYNTHETIC_RELEASE_TRAILS = 50_000


def make_random(dataset: str, seed: int = 0, n_trails: int = SYNTHETIC_RELEASE_TRAILS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pois = pd.read_parquet(PROCESSED / dataset / "pois.parquet").sort_values("venue_id").reset_index(drop=True)
    real = pd.read_parquet(PROCESSED / dataset / "train.parquet", columns=["trail_id"])
    lengths = rng.choice(real.groupby("trail_id").size().to_numpy(), n_trails, replace=True)
    n, total = len(lengths), int(lengths.sum())

    trail = np.repeat(np.arange(n), lengths)
    pos = np.arange(total) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    start = rng.integers(0, 7 * 24 * 60, n)
    gap = np.where(pos == 0, 0, rng.integers(5, 241, total))
    minute = start[trail] + np.cumsum(gap) - np.repeat(np.cumsum(gap)[np.cumsum(lengths) - lengths], lengths)
    venue = pois.iloc[rng.integers(0, len(pois), total)]
    return pd.DataFrame({
        "trail_id": [f"rand_{i}" for i in trail],
        "user_id": "synthetic",
        "venue_id": venue["venue_id"].to_numpy(),
        "lat": venue["lat"].to_numpy(),
        "lon": venue["lon"].to_numpy(),
        "category": venue["category"].to_numpy(),
        "t_local": REFERENCE_MONDAY + pd.to_timedelta(minute, unit="m"),
        "pos": pos,
    })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-trails", type=int, default=SYNTHETIC_RELEASE_TRAILS)
    args = ap.parse_args()
    df = make_random(args.dataset, args.seed, args.n_trails)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"wrote {df['trail_id'].nunique()} trails / {len(df)} check-ins -> {args.out}")


if __name__ == "__main__":
    main()
