from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT))

import numpy as np

N_REF = 74_685
HIST_SHARE = 0.1
N_HIST = 2
ALPHA_SHARE = 0.1
MAX_POPULAR = 2000
DELTA = 1e-6
PRV_MARGIN = 0.015


def composed_eps(sigma_sgd: float, q: float, steps: int, sigma_hist: float, delta: float = DELTA) -> float:
    import dp_accounting
    from dp_accounting.pld import pld_privacy_accountant
    acc = pld_privacy_accountant.PLDAccountant(
        neighboring_relation=dp_accounting.NeighboringRelation.ADD_OR_REMOVE_ONE, value_discretization_interval=1e-4)
    acc.compose(dp_accounting.PoissonSampledDpEvent(q, dp_accounting.GaussianDpEvent(sigma_sgd)), steps)
    if sigma_hist > 0:
        acc.compose(dp_accounting.GaussianDpEvent(sigma_hist), N_HIST)
    return float(acc.get_epsilon(delta))


def calibrate(eps: float, batch: int, epochs: float, delta: float = DELTA, tol: float = 0.005,
              margin: float = PRV_MARGIN) -> dict:
    from trajpe.dp import mu_for
    q = batch / N_REF
    steps = int(round(epochs / q))
    mu = mu_for(eps, delta)
    sigma_hist = math.sqrt(N_HIST / (HIST_SHARE * mu ** 2))
    target = eps - margin
    lo, hi = 0.05, 1.0
    while composed_eps(hi, q, steps, sigma_hist, delta) > target:
        lo, hi = hi, 2 * hi
    while True:
        mid = 0.5 * (lo + hi)
        if composed_eps(mid, q, steps, sigma_hist, delta) > target:
            lo = mid
        else:
            hi = mid
        if target - composed_eps(hi, q, steps, sigma_hist, delta) < tol and hi - lo < 1e-4:
            break
    return {"eps_total": eps, "delta": delta, "pld_target": target, "mu_total": mu, "hist_mu2_share": HIST_SHARE, "n_hist": N_HIST,
            "sigma_hist": sigma_hist, "hist_sensitivity_l2": 1.0, "sample_rate": q, "steps": steps,
            "sigma_sgd": hi, "eps_composed_pld": composed_eps(hi, q, steps, sigma_hist, delta),
            "eps_sgd_only_pld": composed_eps(hi, q, steps, 0.0, delta),
            "eps_hist_only_pld": None if sigma_hist <= 0 else _hist_only(sigma_hist, delta),
            "accountant": "dp_accounting PLD, add-or-remove-one, value_discretization_interval=1e-4"}


def _hist_only(sigma_hist: float, delta: float) -> float:
    import dp_accounting
    from dp_accounting.pld import pld_privacy_accountant
    acc = pld_privacy_accountant.PLDAccountant(
        neighboring_relation=dp_accounting.NeighboringRelation.ADD_OR_REMOVE_ONE, value_discretization_interval=1e-4)
    acc.compose(dp_accounting.GaussianDpEvent(sigma_hist), N_HIST)
    return float(acc.get_epsilon(delta))


def prior(dataset: str, calib: dict, seed: int) -> tuple[dict, dict]:
    from trajpe.catalog import Catalog
    from trajpe.data import load_split
    from trajpe.marginals import Marginals, threshold_sigmas
    from trajpe.trails import from_frame
    cat = Catalog(dataset)
    priv, _ = from_frame(load_split(dataset, "train"), cat)
    sigma = float(calib["sigma_hist"])
    rng = np.random.default_rng([seed, 20261020])
    w_check = (1.0 / priv.lengths)[priv.trail_of]
    n_venues, n_cat = len(cat), len(cat.categories)
    pop = np.bincount(priv.poi, weights=w_check, minlength=n_venues).astype(np.float64)
    cats = np.bincount(cat.cat[priv.poi], weights=w_check, minlength=n_cat).astype(np.float64)
    del priv
    z = threshold_sigmas(n_venues)
    pop = pop + rng.normal(0.0, sigma, n_venues)
    cats = cats + rng.normal(0.0, sigma, n_cat)
    pop = np.where(pop > z * sigma, pop, 0.0)
    cats = np.maximum(cats, 0.0)
    m = Marginals(pop=pop, cat=cats, start=np.zeros(168), sigma=sigma)
    venue_w = m.venue_weights(ALPHA_SHARE, cat.cat)
    nz = np.flatnonzero(pop > 0)
    popular = nz[np.argsort(-pop[nz], kind="stable")[:MAX_POPULAR]]
    arrays = {"pop_idx": nz.astype(np.int64), "pop_val": pop[nz], "cat": cats, "venue_w": venue_w,
              "popular": popular.astype(np.int64)}
    info = {"sigma_hist": sigma, "threshold_sigmas": z, "threshold": z * sigma, "venues_passing": int(len(nz)),
            "expected_noise_venues": 100.0, "alpha_share": ALPHA_SHARE, "n_popular": int(len(popular)),
            "max_popular": MAX_POPULAR, "n_venues": n_venues, "n_categories": n_cat, "seed": seed}
    return arrays, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("--eps", type=float, required=True)
    c.add_argument("--batch", type=int, required=True)
    c.add_argument("--epochs", type=float, required=True)
    c.add_argument("--delta", type=float, default=DELTA)
    c.add_argument("--out", required=True)
    p = sub.add_parser("prior")
    p.add_argument("--dataset", required=True)
    p.add_argument("--calib", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    if args.cmd == "calibrate":
        res = calibrate(args.eps, args.batch, args.epochs, args.delta)
        res["calibrate_sec"] = round(time.time() - t0, 1)
        Path(args.out).write_text(json.dumps(res, indent=1))
        print(json.dumps(res))
    else:
        calib = json.loads(Path(args.calib).read_text())
        arrays, info = prior(args.dataset, calib, args.seed)
        np.savez_compressed(args.out, **arrays)
        info["prior_sec"] = round(time.time() - t0, 1)
        Path(args.out).with_suffix(".json").write_text(json.dumps(info, indent=1))
        print(json.dumps(info))


if __name__ == "__main__":
    main()
