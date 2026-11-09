from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

METRICS = ("acc@1", "acc@5", "acc@10", "ndcg@10", "mrr")
DATASET_ORDER = {"nyc": 0, "istanbul": 1, "tokyo": 2, "sao_paulo": 3, "kuwait_city": 4, "jakarta": 5}
MODEL_ORDER = {"transformer": 0, "markov": 1}


def train_label(r: dict) -> str:
    name = "real" if r["train_is_real_split"] else Path(r["train_path"]).stem
    if r["train_data"].get("subsampled_to"):
        name += f" ({r['train_data']['subsampled_to'] // 1000}k trails)"
    if r["model"] == "transformer" and r["train_is_real_split"] and r["early_stop"] != "val":
        name += f" ({r['early_stop']} early stop)"
    return name


def fmt(vals: list[float], digits: int = 4) -> str:
    mean = statistics.fmean(vals)
    if len(vals) == 1:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {statistics.stdev(vals):.{digits}f}"


def collect(paths: list[str]) -> dict:
    groups = defaultdict(list)
    for arg in paths:
        p = Path(arg)
        for f in sorted(p.glob("*.json")) if p.is_dir() else [p]:
            r = json.loads(f.read_text())
            if "metrics" in r:
                groups[(r["dataset"], train_label(r), r["model"], r["eval_split"])].append(r)
    return groups


def table(groups: dict) -> str:
    head = ("| dataset | train data | model | eval | seeds | Acc@1 | Acc@5 | Acc@10 | NDCG@10 | MRR "
            "| #targets | vocab | in-vocab | cat Acc@1 | cat MRR | train s | total s | epochs |")
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    key = lambda k: (DATASET_ORDER.get(k[0], 9), k[1] != "real", k[1], MODEL_ORDER.get(k[2], 9), k[3])
    for k in sorted(groups, key=key):
        rs = groups[k]
        epochs = [r["train_info"].get("selected_epoch") for r in rs]
        cells = [k[0], k[1], k[2], k[3], ",".join(str(r["seed"]) for r in rs)]
        cells += [fmt([r["metrics"][m] for r in rs]) for m in METRICS]
        cells += [str(rs[0]["metrics"]["n_targets"]), str(rs[0].get("vocab_size", "-")),
                  f"{rs[0]['eval_target_vocab_coverage']:.4f}" if "eval_target_vocab_coverage" in rs[0] else "-"]
        cells += [fmt([r["metrics"]["category"][m] for r in rs]) if all("category" in r["metrics"] for r in rs)
                  else "-" for m in ("acc@1", "mrr")]
        cells += [fmt([r["runtime_s"]["train"] for r in rs], 0), fmt([r["runtime_s"]["total"] for r in rs], 0),
                  "-" if None in epochs else ",".join(map(str, epochs))]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="result JSON files or directories of them")
    ap.add_argument("--md", type=Path, default=None, help="also write the table to this markdown file")
    args = ap.parse_args()
    out = table(collect(args.paths))
    print(out)
    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(out + "\n")


if __name__ == "__main__":
    main()
