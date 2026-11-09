from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from downstream.data import load_catalog
from trajpe.paths import PROCESSED

VARIANTS = ("sub25", "geocat10", "catshuffle")


def degrade(dataset: str, variant: str, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    real = pd.read_parquet(PROCESSED / dataset / "train.parquet")
    if variant == "sub25":
        trails = real["trail_id"].unique()
        return real[real["trail_id"].isin(rng.permutation(trails)[: len(trails) // 4])]

    cat = load_catalog(PROCESSED / dataset)
    row = cat.index.get_indexer(real["venue_id"].astype(str))
    new = row.copy()
    for c in range(cat.n_cats):
        members = np.flatnonzero(cat.cat_idx == c)
        sel = np.flatnonzero(cat.cat_idx[row] == c)
        if len(sel) == 0:
            continue
        if variant == "catshuffle":
            new[sel] = members[rng.integers(0, len(members), len(sel))]
        elif len(members) > 1:
            k = min(11, len(members))
            _, nn = cKDTree(cat.xy[members]).query(cat.xy[row[sel]], k=k)
            new[sel] = members[nn[np.arange(len(sel)), rng.integers(1, k, len(sel))]]
    pois = pd.read_parquet(PROCESSED / dataset / "pois.parquet").assign(venue_id=lambda d: d["venue_id"].astype(str))
    p = pois.set_index("venue_id").loc[cat.venue_ids[new]]
    return real.assign(venue_id=p.index.to_numpy(), lat=p["lat"].to_numpy(), lon=p["lon"].to_numpy(),
                       category=p["category"].to_numpy())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for v in args.variants:
        out = args.out_dir / f"degraded_{args.dataset}_{v}.parquet"
        df = degrade(args.dataset, v, args.seed)
        df.to_parquet(out, index=False)
        print(f"wrote {df['trail_id'].nunique()} trails / {len(df)} check-ins -> {out}")


if __name__ == "__main__":
    main()
