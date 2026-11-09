import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from trajpe.api_sim import SimAPI
from trajpe.catalog import Catalog
from trajpe.data import load_split
from trajpe.metrics import Fidelity
from trajpe.paths import RESULTS
from trajpe.pe import PEConfig, run_pe
from trajpe.trails import from_frame

SPACE = {
    "n_syn": [2000, 5000, 10000, 20000],
    "T": [5, 10, 20],
    "threshold_sigmas": [1.0, 2.0, 3.0],
    "marg_frac": [0.0, 0.1, 0.2, 0.4],
    "pop_alpha_share": [0.1, 0.3, 0.5],
    "degree_start": [0.5, 1.0],
    "degree_end": [0.05, 0.1, 0.2],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="jakarta")
    ap.add_argument("--api", default="sim", choices=["sim", "llm"])
    ap.add_argument("--eps", type=float, nargs="+", default=[1.0, 4.0])
    ap.add_argument("--n_trials", type=int, default=40)
    ap.add_argument("--n_out", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--space", default=None, help="JSON dict overriding (parts of) the search space")
    args = ap.parse_args()
    assert args.dataset == "jakarta", "tune only on the held-out tuning city"
    space = dict(SPACE, **(json.loads(args.space) if args.space else {}))

    catalog = Catalog(args.dataset)
    priv, users = from_frame(load_split(args.dataset, "train"), catalog)
    fid = Fidelity(catalog)
    real = fid.stats(priv)
    if args.api == "sim":
        api = SimAPI(catalog)
    else:
        from trajpe.api_llm import LLMAPI
        api = LLMAPI(catalog)
    out = Path(args.out or RESULTS / "tuning" / f"{args.dataset}_{args.api}_s{args.seed}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    with open(out, "a") as f:
        for trial in range(args.n_trials):
            hp = {k: v[rng.integers(len(v))] for k, v in space.items()}
            hp = {k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, np.floating) else v)
                  for k, v in hp.items()}
            if hp.get("vote_k", 1) > 1 and hp.get("cluster_c", 0) > 0:
                hp["cluster_c"] = 0.0
            for eps in args.eps:
                cfg = PEConfig(eps=eps, n_out=args.n_out, seed=args.seed, use_marginals=hp["marg_frac"] > 0,
                               **{k: v for k, v in hp.items() if k != "marg_frac"},
                               marg_frac=hp["marg_frac"] if hp["marg_frac"] > 0 else 0.2)
                t0 = time.time()
                syn, _, privacy, _ = run_pe(priv, users, api, catalog, cfg, log=lambda *_: None)
                rec = {"trial": trial, "eps": eps, "hp": hp, "sec": round(time.time() - t0, 1),
                       "metrics": fid.compare(real, fid.stats(syn))}
                f.write(json.dumps(rec, default=float) + "\n")
                f.flush()
                print(trial, eps, hp, rec["sec"], round(rec["metrics"]["query_error"], 3), flush=True)


if __name__ == "__main__":
    main()
