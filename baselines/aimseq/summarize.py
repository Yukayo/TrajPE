from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

TERMS = ["density_jsd", "od_jsd", "query_error", "travel_jsd", "hop_jsd", "gyration_jsd", "length_jsd", "hour_jsd",
         "gap_jsd", "cat_jsd", "root_trans_jsd", "hour_root_jsd", "weekhour_jsd", "pattern_f1", "poi_kendall"]


def ms(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return float("nan"), float("nan"), 0
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return m, sd, len(xs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summaries", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", nargs="*", default=None, help="order / subset of methods")
    args = ap.parse_args()
    rows = []
    for f in args.summaries:
        rows += json.loads(Path(f).read_text())
    seen = {}
    for r in rows:
        seen[r["name"]] = r
    groups = collections.defaultdict(list)
    for r in seen.values():
        groups[(r["method"], float(r["eps"]))].append(r)
    methods = args.methods or sorted({m for m, _ in groups})
    out = ["| method | eps | n | composite | Markov MRR | runtime s | " + " | ".join(t.replace("_jsd", "") for t in TERMS) + " |",
           "|---|---|---|---|---|---|" + "---|" * len(TERMS)]
    for eps in sorted({e for _, e in groups}):
        for m in methods:
            g = groups.get((m, eps))
            if not g:
                continue
            c, csd, n = ms([r.get("composite") for r in g])
            mr, mrsd, _ = ms([r.get("markov_mrr") for r in g])
            rt, _, _ = ms([r.get("runtime_sec") for r in g])
            terms = [ms([r.get("fidelity_vs_train", {}).get(t) for r in g])[0] for t in TERMS]
            out.append(f"| {m} | {eps:g} | {n} | {c:.4f} ± {csd:.4f} | {mr:.4f} ± {mrsd:.4f} | {rt:.0f} | "
                       + " | ".join(f"{v:.4f}" for v in terms) + " |")
    Path(args.out).write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
