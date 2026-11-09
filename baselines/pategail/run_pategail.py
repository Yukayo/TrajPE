from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import common
import pategail


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=common.CITIES)
    ap.add_argument("--eps", required=True, help="total budget (1, 2, 4, 8); 'inf' = Jakarta noise-free reference")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm", choices=["main", "uniform"], default="main")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    eps = math.inf if a.eps.lower() == "inf" else float(a.eps)
    if Path(a.out).exists():
        raise SystemExit(f"{a.out} exists; refusing to overwrite a release")
    pategail.run(a.dataset, eps, a.seed, a.arm, Path(a.out), a.device, a.threads)


if __name__ == "__main__":
    main()
