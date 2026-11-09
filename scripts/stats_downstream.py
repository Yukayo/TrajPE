import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.stats import binomtest, ttest_ind

from trajpe.paths import RESULTS

CITIES = ["istanbul", "tokyo", "sao_paulo", "kuwait_city", "nyc"]
OURS = ("F_hybriddiv", "TrajPE")
OTHERS = [("privtrace", "PrivTrace"), ("adatrace", "AdaTrace"), ("dplora", "DP-LoRA"), ("dploravp", "DP-LoRA + DP venue prior"),
          ("F_simdiv", "TrajPE-sim")]


def runs(city, prefix, eps, model="transformer"):
    vals = []
    for s in range(3):
        for t, suf in ((0, ""), (1, "_t1"), (2, "_t2")):
            f = RESULTS / "downstream" / city / f"{prefix}_eps{eps}_s{s}_{model}{suf}.json"
            if f.exists():
                vals.append(json.loads(f.read_text())["metrics"]["mrr"])
    return np.array(vals)


def main():
    lines = ["# Downstream significance (next-POI MRR, transformer, protocol v2.1)", ""]
    for other, oname in OTHERS:
        lines += [f"## {OURS[1]} vs {oname}", "", "| city | eps | ours (n) | other (n) | diff | Welch p |",
                  "|---|---|---|---|---|---|"]
        diffs = []
        for city in CITIES:
            for eps in (1, 4):
                a, b = runs(city, OURS[0], eps), runs(city, other, eps)
                if len(a) == 0 or len(b) == 0:
                    continue
                p = ttest_ind(a, b, equal_var=False).pvalue if len(a) > 1 and len(b) > 1 else float("nan")
                diffs.append(a.mean() - b.mean())
                lines.append(f"| {city} | {eps} | {a.mean():.4f}±{a.std():.4f} ({len(a)}) | {b.mean():.4f}±{b.std():.4f} "
                             f"({len(b)}) | {a.mean() - b.mean():+.4f} | {p:.2g} |")
        if diffs:
            wins = int(sum(d > 0 for d in diffs))
            lines += ["", f"Sign test over {len(diffs)} cells: {wins} wins, p = "
                      f"{binomtest(wins, len(diffs), 0.5).pvalue:.3g}", ""]
    out = RESULTS / "tables" / "stats.md"
    out.write_text("\n".join(lines) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    main()
