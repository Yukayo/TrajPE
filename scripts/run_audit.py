import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from trajpe.audit import canary_audit, concat, make_canaries, user_disjoint_split, user_mia
from trajpe.catalog import Catalog
from trajpe.data import load_split
from trajpe.paths import RESULTS
from trajpe.pe import PEConfig, run_pe
from trajpe.trails import from_frame

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nyc")
    ap.add_argument("--api", default="sim", choices=["sim", "llm", "copy"], help="copy = release the private data itself (positive control)")
    ap.add_argument("--eps", type=float, default=4.0)
    ap.add_argument("--unit", default="trail", choices=["trail", "user"])
    ap.add_argument("--config", default=None, help="JSON with PEConfig fields (e.g. the frozen Jakarta config)")
    ap.add_argument("--m_canaries", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    c = Catalog(args.dataset)
    train, users = from_frame(load_split(args.dataset, "train"), c)
    mem_mask, non_mask = user_disjoint_split(users, 0.2, args.seed)
    members, nonmembers = train.subset(np.flatnonzero(mem_mask)), train.subset(np.flatnonzero(non_mask))
    canaries = make_canaries(c, args.m_canaries, args.seed + 1)
    inserted = np.random.default_rng(args.seed + 2).random(args.m_canaries) < 0.5
    priv = concat(members, canaries.subset(np.flatnonzero(inserted)))
    priv_users = np.concatenate([users[mem_mask], np.array([f"canary_{i}" for i in np.flatnonzero(inserted)])])

    full = json.loads(Path(args.config).read_text()) if args.config else {}
    hp = {k: v for k, v in full.items() if k in PEConfig.__dataclass_fields__ and k not in ("eps", "unit", "seed")}
    cfg = PEConfig(eps=args.eps, unit=args.unit, seed=args.seed, **hp)
    if args.api == "copy":
        release, privacy, cfg_d = priv, {"eps": float("inf"), "note": "release = private data"}, {}
    else:
        if args.api == "sim":
            from trajpe.api_sim import SimAPI
            api = SimAPI(c)
        else:
            from trajpe.api_llm import LLMAPI
            api = LLMAPI(c, sim_frac=full.get("sim_frac", 0.0))
        release, _, privacy, cfg_d = run_pe(priv, priv_users, api, c, cfg, log=lambda *_: None)
    res = {"dataset": args.dataset, "api": args.api, "privacy": privacy, "pe": cfg_d,
           **user_mia(release, members, nonmembers, c, seed=args.seed),
           "canary": canary_audit(release, canaries, inserted, c)}
    tag = args.tag or f"{args.api}_eps{args.eps:g}_{args.unit}_s{args.seed}"
    out = RESULTS / "audit" / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{tag}.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({"tag": tag, "user_mia_auc": res["user_mia_auc"], **res["canary"]}, default=float))
