import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from trajpe.catalog import Catalog
from trajpe.data import load_split
from trajpe.privacy_eval import evaluate
from trajpe.trails import from_frame

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nyc")
    ap.add_argument("--synth", help="synthetic check-ins parquet")
    ap.add_argument("--reference", action="store_true",
                    help="release the real train split itself (worst case, for calibrating the scores)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    c = Catalog(args.dataset)
    train, _ = from_frame(load_split(args.dataset, "train"), c)
    test, _ = from_frame(load_split(args.dataset, "test"), c)
    if args.reference:
        syn = train
    else:
        syn, _ = from_frame(pd.read_parquet(args.synth), c)
    res = {"dataset": args.dataset, "synth": args.synth, "reference": args.reference,
           **evaluate(syn, train, test, c)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res))
