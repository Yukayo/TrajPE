import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from trajpe.api_sim import RuleVariation
from trajpe.catalog import Catalog
from trajpe.data import load_split
from trajpe.marginals import dp_marginals
from trajpe.mechanisms import noise_levels, vote_weights
from trajpe.metrics import Fidelity
from trajpe.paths import PROCESSED, RESULTS, RUNS
from trajpe.pe import PEConfig
from trajpe.trails import TrailSet, from_frame, to_frame

STREAM = 7919
METRICS_DEFAULT = False


def private_metrics_enabled(requested):
    if os.environ.get("TRAJPE_GRID_NOMETRICS") == "1":
        if requested:
            sys.exit("--metrics conflicts with TRAJPE_GRID_NOMETRICS=1 (release mode: no private-data metric)")
        return False
    return METRICS_DEFAULT if requested is None else bool(requested)


def out_tag_for(tag: str) -> str:
    parts = tag.split("_")
    if parts[0] == "F" and len(parts) > 2:
        return "_".join([parts[0], parts[1] + "div"] + parts[2:])
    return tag + "_div"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tag", required=True, help="input run under RUNS/<dataset>/<tag>")
    ap.add_argument("--p", type=float, default=0.5)
    ap.add_argument("--radius_km", type=float, default=0.5)
    ap.add_argument("--out_tag", default="")
    ap.add_argument("--metrics", action=argparse.BooleanOptionalAction, default=None,
                    help="compute vs_train / vs_test on the private splits (internal evaluation only, never part of a "
                         "release; default: METRICS_DEFAULT, always off with TRAJPE_GRID_NOMETRICS=1)")
    args = ap.parse_args()
    nometrics = not private_metrics_enabled(args.metrics)

    src = RUNS / args.dataset / args.tag
    res_in = json.loads((src / "result.json").read_text())
    cfg = PEConfig(**res_in["pe"])
    catalog = Catalog(args.dataset)
    priv, users = from_frame(load_split(args.dataset, "train"), catalog)
    test = None if nometrics else from_frame(load_split(args.dataset, "test"), catalog)[0]

    rng = np.random.default_rng(cfg.seed)
    _, sigma_m = noise_levels(cfg.eps, cfg.delta, cfg.T, cfg.use_marginals, cfg.marg_frac)
    marg = dp_marginals(priv, vote_weights(users, cfg.unit), catalog.cat, len(catalog.categories), sigma_m,
                        cfg.pop_threshold_sigmas, rng)
    got = {"marg_nonzero_venues": int((marg.pop > 0).sum()),
           "marg_cat_transitions": int((marg.cat_trans > 0).sum()),
           "marg_venue_transitions": int((marg.venue_trans > 0).sum()) if marg.venue_trans.size else 0}
    want = {k: res_in["privacy"][k] for k in got}
    if got != want:
        sys.exit(f"regenerated marginals differ from the run's ({got} vs {want}); not post-processing")
    if "marg_sha256" in res_in["privacy"]:
        got["marg_sha256"] = marg.digest()
        if got["marg_sha256"] != res_in["privacy"]["marg_sha256"]:
            sys.exit("regenerated marginals differ from the run's (SHA-256); not post-processing")
    else:
        train_file = PROCESSED / args.dataset / "train.parquet"
        if train_file.stat().st_mtime >= (src / "result.json").stat().st_mtime:
            sys.exit("the train split changed after the run; regenerated marginals would reuse noise")

    var = RuleVariation(catalog)
    var.set_venue_weights(marg.venue_weights(cfg.pop_alpha_share, catalog.cat)
                          if "var" in cfg.marg_use.split(",") else None)
    syn = var.diversify(TrailSet.load(src / "synth.npz"), args.p, args.radius_km,
                        np.random.default_rng([cfg.seed, STREAM]))

    tag = args.out_tag or out_tag_for(args.tag)
    out = RUNS / args.dataset / tag
    out.mkdir(parents=True, exist_ok=True)
    syn.save(out / "synth.npz")
    to_frame(syn, catalog).to_parquet(out / "synth.parquet", index=False)
    res = {k: res_in[k] for k in ("dataset", "api", "privacy", "pe", "embed") if k in res_in}
    res.update(tag=tag, source=args.tag, args=res_in.get("args"),
               postprocess={"diversify_p": args.p, "radius_km": args.radius_km, "stream": [cfg.seed, STREAM],
                            "marginals_check": got})
    if nometrics:
        res["metrics"] = ("not computed (TRAJPE_GRID_NOMETRICS=1)" if os.environ.get("TRAJPE_GRID_NOMETRICS") == "1"
                          else "not computed (release mode)")
        (out / "result.json").write_text(json.dumps(res, indent=1, default=float))
        print(json.dumps({"tag": tag, "metrics": res["metrics"]}))
        return
    fid = Fidelity(catalog)
    res.update(vs_train=fid.compare(fid.stats(priv), fid.stats(syn)),
               vs_test=fid.compare(fid.stats(test), fid.stats(syn)))
    (out / "result.json").write_text(json.dumps(res, indent=1, default=float))
    dest = RESULTS / "pe" / args.dataset
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"{tag}.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({"tag": tag, **res["vs_train"]}, default=float))


if __name__ == "__main__":
    main()
