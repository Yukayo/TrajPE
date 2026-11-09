from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pathfmt as PF

sys.path.insert(0, str(PF.PROJECT))
hybrid = PF.load_module("dplora_hybrid", PF.PROJECT / "baselines" / "dplora" / "hybrid.py")
from trajpe.dp import eps_for, mu_for

INF = float("inf")


def bisect(f, target: float, tol: float = PF.CALIB_TOL) -> float:
    lo, hi = 0.05, 1.0
    while f(hi) > target:
        lo, hi = hi, 2 * hi
    while True:
        mid = 0.5 * (lo + hi)
        if f(mid) > target:
            lo = mid
        else:
            hi = mid
        if target - f(hi) < tol and hi - lo < 1e-4:
            return hi


def calibrate(eps: float, delta: float = PF.DELTA, literal: bool = True) -> dict:
    q, T = PF.sample_rate(), PF.n_steps()
    es = PF.eps_select(eps)
    mu_sel = mu_for(es, delta / 2)
    sigma_sel = 1.0 / mu_sel
    target = eps - PF.PRV_MARGIN

    def joint(s):
        return hybrid.composed_eps(s, q, T, mu_sel, delta)

    if hybrid.composed_eps(INF, q, T, mu_sel, delta) >= target:
        raise ValueError("the selection alone exhausts the budget")
    sigma = bisect(joint, target)
    out = {
        "method": "path", "eps_total": eps, "delta": delta, "unit": "trail", "relation": "add/remove one trail",
        "composition": "joint PLD", "pld_target": target,
        "eps_select": es, "delta_select": delta / 2, "mu_sel": mu_sel, "sigma_sel": sigma_sel,
        "eps_select_check_gdp": eps_for(mu_sel, delta / 2), "vote_k": PF.VOTE_K,
        "vote_l2_sensitivity": math.sqrt(PF.VOTE_K), "vote_noise_std": math.sqrt(PF.VOTE_K) * sigma_sel,
        "sigma_hist": sigma_sel, "n_hist": 1, "hist_sensitivity_l2": 1.0,
        "sample_rate": q, "steps": T, "batch_ref": PF.BATCH, "epochs": PF.EPOCHS, "n_ref": PF.N_REF,
        "sigma_sgd": sigma, "eps_composed_pld": joint(sigma),
        "eps_sgd_only_pld": hybrid.composed_eps(sigma, q, T, 0.0, delta),
        "eps_sel_only_pld": hybrid.composed_eps(INF, q, T, mu_sel, delta),
        "accountant": "dp_accounting PLD, add-or-remove-one, value_discretization_interval=1e-4 "
                      "(baselines/dplora/hybrid.composed_eps, selection as Gaussian(1/mu_sel))",
    }
    if literal:
        et = eps - es
        s_basic = bisect(lambda s: hybrid.composed_eps(s, q, T, 0.0, delta / 2), et - PF.PRV_MARGIN)
        out["paper_literal"] = {
            "note": "report only, never used to calibrate; loose bound, not the guarantee",
            "composition": "basic", "eps_train": et, "delta_train": delta / 2, "eps_select": es,
            "delta_select": delta / 2, "sigma_sgd_basic": s_basic,
            "eps_sgd_basic_pld_at_delta_half": hybrid.composed_eps(s_basic, q, T, 0.0, delta / 2),
            "eps_joint_pld_of_sigma_sgd_basic": hybrid.composed_eps(s_basic, q, T, mu_sel, delta),
            "eps_basic_reading_of_release": hybrid.composed_eps(sigma, q, T, 0.0, delta / 2) + es,
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eps", type=float, required=True)
    ap.add_argument("--delta", type=float, default=PF.DELTA)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        print(out.read_text())
        return
    t0 = time.time()
    res = calibrate(args.eps, args.delta)
    res["calibrate_sec"] = round(time.time() - t0, 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(res, indent=1))
    tmp.rename(out)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
