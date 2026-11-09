from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np

import common
from accounting import budget
from encoding import Decoder, PublicDomain, Spec, spatial_tilt
from run_aimseq import parse_strata


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="parquet of a run made with --save_records")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tilt", type=int, choices=[0, 1], default=0, help="spatial tilt of hop venues (spatial_tilt)")
    ap.add_argument("--end_any", type=int, choices=[0, 1], default=0,
                    help="last venue: class venue anywhere in the end cell before dropping the end cell")
    ap.add_argument("--use_end", type=int, choices=[0, 1], default=1, help="0: ignore the records' end cell")
    ap.add_argument("--dec_seed", type=int, default=None, help="decoder RNG seed (default: the run's)")
    args = ap.parse_args()
    side = json.loads(args.run.with_suffix(".json").read_text())
    z = np.load(args.run.with_suffix(".records.npz"))
    cfg, seed = side["config"], side["seed"]
    t0 = time.time()
    s_prior, s_mech, s_synth, s_dec = common.seeds(seed, 4)
    ds = common.load_dataset(side["dataset"], side["user_level"], side["user_seed"])
    bud = budget(side["eps"], side["delta"], cfg["prior_frac"])
    prior = (common.VenuePrior.fit_central(ds, bud["eps_venue"], bud["eps_cat"], np.random.default_rng(s_prior),
                                           cfg["venue_floor"], cfg["venue_fp"])
             if cfg["venue_prior"] == "dp" else common.VenuePrior.uniform())
    assert prior.info == {k: v for k, v in side["venue_prior"].items()}, "prior differs from the run's"
    spec = Spec(grid=cfg["grid"], od_grid=cfg["od_grid"], k_sub=cfg["k_sub"], max_pos=cfg["max_pos"],
                time_split=bool(cfg["time_split"]), strata=parse_strata(cfg["strata"]))
    dom = PublicDomain(ds.pois, ds.bbox, spec)
    recs = {}
    for key in z.files:
        st, a = key.split("/")
        recs.setdefault(st, {})[a] = z[key]
    tilt = spatial_tilt(dom, prior.weights, np.concatenate([r["sc"] for r in recs.values()])) if args.tilt else None
    dec = Decoder(dom, prior.weights, tilt, bool(args.end_any), bool(args.use_end))
    rng = np.random.default_rng(s_dec if args.dec_seed is None else args.dec_seed)
    stats: dict = {}
    trails = []
    for lo, hi in spec.strata:
        rec = recs.get(f"{lo}-{hi}")
        if rec is None:
            continue
        v, L, t, _ = dec.decode(rec, rng, spec.min_gap, (lo, hi), stats)
        o = np.concatenate([[0], np.cumsum(L)])
        trails += [(v[o[i]:o[i + 1]], t[o[i]:o[i + 1]]) for i in range(len(L))]
    perm = rng.permutation(len(trails))
    vidx = np.concatenate([trails[i][0] for i in perm])
    tmin = np.concatenate([trails[i][1] for i in perm])
    lengths = np.array([len(trails[i][0]) for i in perm], np.int64)
    df = common.build_frame(vidx, lengths, tmin, ds.pois, prefix="aimseq")
    common.write_outputs(df, args.out, {**{k: side[k] for k in ("method", "dataset", "eps", "delta", "user_level",
                                                                 "seed", "config", "budget", "strata")},
                                        "redecoded_from": str(args.run),
                                        "decoder": {"tilt": bool(args.tilt), "end_any": bool(args.end_any),
                                                    "use_end": bool(args.use_end)},
                                        "decode": stats, "runtime_sec": side["runtime_sec"],
                                        "redecode_sec": time.time() - t0})


if __name__ == "__main__":
    main()
