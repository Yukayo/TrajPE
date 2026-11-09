import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trajpe.catalog import Catalog
from trajpe.data import load_split
from trajpe.metrics import Fidelity
from trajpe.trails import from_frame

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nyc")
    ap.add_argument("--synth", help="synthetic check-ins parquet (standard schema)")
    ap.add_argument("--reference", action="store_true", help="evaluate the real test split instead")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    catalog = Catalog(args.dataset)
    priv, _ = from_frame(load_split(args.dataset, "train"), catalog)
    test, _ = from_frame(load_split(args.dataset, "test"), catalog)
    syn = test if args.reference else from_frame(pd.read_parquet(args.synth), catalog)[0]
    fid = Fidelity(catalog)
    res = {"dataset": args.dataset, "synth": args.synth, "reference": args.reference,
           "vs_train": fid.compare(fid.stats(priv), fid.stats(syn))}
    if not args.reference:
        res["vs_test"] = fid.compare(fid.stats(test), fid.stats(syn))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res["vs_train"], default=float))


if __name__ == "__main__":
    main()
