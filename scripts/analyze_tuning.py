import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from trajpe.metrics import HIGHER_IS_BETTER, LOWER_IS_BETTER

SCORE_METRICS = [m for m in LOWER_IS_BETTER if m not in ("cat_trans_jsd", "dow_jsd", "root_jsd")] + HIGHER_IS_BETTER

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()
    recs = [json.loads(line) for f in args.files for line in open(f)]
    df = pd.json_normalize(recs)
    hp_cols = [c for c in df.columns if c.startswith("hp.")]
    for eps, d in df.groupby("eps"):
        d = d.copy()
        mets = [m for m in SCORE_METRICS if f"metrics.{m}" in d.columns]
        ranks = pd.DataFrame({m: d[f"metrics.{m}"].rank(ascending=m not in HIGHER_IS_BETTER) for m in mets})
        d["mean_rank"] = ranks.mean(axis=1)
        d = d.sort_values("mean_rank")
        print(f"\n=== eps={eps}  ({len(d)} trials)")
        show = ["mean_rank", "sec"] + hp_cols + [f"metrics.{m}" for m in
                                                 ["density_jsd", "query_error", "hop_jsd", "cat_jsd", "gap_jsd",
                                                  "root_trans_jsd", "poi_kendall"]]
        print(d[show].head(args.top).rename(columns=lambda c: c.split(".")[-1]).round(3).to_string(index=False))
        print("mean rank by value:")
        for c in hp_cols:
            print(f"  {c[3:]:>16}: " + "  ".join(f"{k}={v:.1f}" for k, v in d.groupby(c)["mean_rank"].mean().items()))
