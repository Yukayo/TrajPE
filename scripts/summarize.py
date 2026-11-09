import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trajpe.metrics import composite_score
from trajpe.paths import RESULTS

COLS = ["density_jsd", "od_jsd", "query_error", "travel_jsd", "hop_jsd", "gyration_jsd", "length_jsd",
        "hour_jsd", "gap_jsd", "cat_jsd", "root_trans_jsd", "pattern_f1", "poi_kendall", "unique_frac"]
SHORT = ["dens", "od", "query", "travel", "hop", "gyr", "len", "hour", "gap", "cat", "rtrans", "patF1", "kend", "uniq"]


def rows(dataset, split):
    for sub in ["reference", "pe", "baselines"]:
        d = RESULTS / sub / dataset if sub != "reference" else RESULTS / "reference"
        for f in sorted(d.glob("*.json")):
            if sub == "reference" and not f.name.startswith(dataset + "_"):
                continue
            r = json.loads(f.read_text())
            m = r.get(split) or r.get("vs_train")
            name = f"{sub}/{f.stem}"
            yield name, m, r.get("runtime_sec")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nyc")
    ap.add_argument("--split", default="vs_train", choices=["vs_train", "vs_test"])
    ap.add_argument("--filter", default="")
    args = ap.parse_args()
    print("| run | " + " | ".join(SHORT) + " | score | sec |")
    print("|---|" + "---|" * (len(SHORT) + 2))
    for name, m, sec in rows(args.dataset, args.split):
        if args.filter and not re.search(args.filter, name):
            continue
        vals = [f"{m.get(c, float('nan')):.3f}" for c in COLS]
        try:
            sc = f"{composite_score(m):.3f}"
        except KeyError:
            sc = "n/a"
        print(f"| {name} | " + " | ".join(vals) + f" | {sc} | {sec if sec is not None else ''} |")
