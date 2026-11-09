from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from trajpe.metrics import SCORE_METRICS, composite_score


def main():
    for f in sys.argv[1:]:
        f = Path(f)
        ev = json.loads(f.with_suffix(".eval.json").read_text())
        res = {"composite_vs_train": composite_score(ev["vs_train"]),
               "composite_vs_test": composite_score(ev["vs_test"]) if "vs_test" in ev else None}
        side_p = f.with_suffix(".json")
        side = json.loads(side_p.read_text()) if side_p.exists() else {}
        side.update(res)
        if "train_total_sec" in side:
            side["runtime_sec"] = round((side.get("train_total_sec") or 0) + (side.get("sample_total_sec") or 0)
                                        + (side.get("grounding", {}).get("ground_sec") or 0), 1)
        side_p.write_text(json.dumps(side, indent=1, default=str))
        m = ev["vs_train"]
        print(f"{f.name}: composite vs train {res['composite_vs_train']:.4f} | "
              + " ".join(f"{k.replace('_jsd', '')}={m[k]:.4f}" for k in SCORE_METRICS))


if __name__ == "__main__":
    main()
