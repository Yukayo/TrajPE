from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ctcl_conf as C

ACCOUNTANT = ("dp_accounting PLD, add-or-remove-one, value_discretization_interval=1e-4; events composed in the order "
              "PoissonSampled(q, Gaussian(sigma_sgd)) x T, Gaussian(sigma_topic_hist) x n_topic_hist, "
              "Gaussian(sigma_prior) x n_prior")


def composed_eps(sigma_sgd, q, steps, sigma_topic_hist=C.SIGMA_TOPIC_HIST, n_topic_hist=C.N_TOPIC_HIST,
                 sigma_prior=None, n_prior=0, delta=C.DELTA) -> float:
    import dp_accounting
    from dp_accounting.pld import pld_privacy_accountant
    acc = pld_privacy_accountant.PLDAccountant(
        neighboring_relation=dp_accounting.NeighboringRelation.ADD_OR_REMOVE_ONE,
        value_discretization_interval=C.PLD_DISCRETIZATION)
    if sigma_sgd is not None:
        acc.compose(dp_accounting.PoissonSampledDpEvent(q, dp_accounting.GaussianDpEvent(sigma_sgd)), int(steps))
    if n_topic_hist:
        acc.compose(dp_accounting.GaussianDpEvent(sigma_topic_hist), int(n_topic_hist))
    if n_prior:
        acc.compose(dp_accounting.GaussianDpEvent(sigma_prior), int(n_prior))
    return float(acc.get_epsilon(delta))


def sigma_prior_for(eps: float, delta: float = C.DELTA) -> float:
    sys.path.insert(0, str(C.PROJECT))
    from trajpe.dp import mu_for
    mu = mu_for(eps, delta)
    return math.sqrt(C.N_PRIOR / (C.PRIOR_MU2_SHARE * mu ** 2))


def sigma_eq(sigma_prior=None) -> float:
    inv = 1.0 / C.SIGMA_TOPIC_HIST ** 2 + (C.N_PRIOR / sigma_prior ** 2 if sigma_prior else 0.0)
    return 1.0 / math.sqrt(inv)


def calibrate(eps: float, variant: str, config: str, delta: float = C.DELTA, margin: float = C.PRV_MARGIN,
              tol: float = C.CALIB_EPS_TOL) -> dict:
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("CTCL needs a finite eps > 0")
    cfg = C.CONFIGS[config]
    q, T = cfg["batch"] / C.N_REF, cfg["steps"]
    s_prior = sigma_prior_for(eps, delta) if variant == "vp" else None
    n_prior = C.N_PRIOR if variant == "vp" else 0
    target = eps - margin

    def f(s):
        return composed_eps(s, q, T, sigma_prior=s_prior, n_prior=n_prior, delta=delta)
    if composed_eps(None, q, T, sigma_prior=s_prior, n_prior=n_prior, delta=delta) >= target:
        raise ValueError("the Gaussian histograms alone exhaust the budget")
    lo, hi = 0.05, 1.0
    while f(hi) > target:
        lo, hi = hi, 2 * hi
    while True:
        mid = 0.5 * (lo + hi)
        if f(mid) > target:
            lo = mid
        else:
            hi = mid
        if target - f(hi) < tol and hi - lo < C.CALIB_SIGMA_TOL:
            break
    e = f(hi)
    res = {"method": "CTCL", "variant": variant, "config": config, "eps_total": eps, "delta": delta,
           "pld_target": target, "sample_rate": q, "steps": T, "b_nom": cfg["batch"], "clip": cfg["clip"],
           "sigma_sgd": hi, "sigma_topic_hist": C.SIGMA_TOPIC_HIST, "n_topic_hist": C.N_TOPIC_HIST,
           "K_topics": C.K_TOPICS, "sigma_prior": s_prior, "n_prior": n_prior,
           "prior_mu2_share": C.PRIOR_MU2_SHARE if s_prior else None, "sigma_eq": sigma_eq(s_prior),
           "eps_composed_pld": e,
           "eps_sgd_only_pld": composed_eps(hi, q, T, n_topic_hist=0, delta=delta),
           "eps_topic_hist_only_pld": composed_eps(None, q, T, delta=delta),
           "eps_prior_only_pld": composed_eps(None, q, T, n_topic_hist=0, sigma_prior=s_prior, n_prior=n_prior,
                                              delta=delta) if s_prior else None,
           "accountant": ACCOUNTANT}
    ref = C.SIGMA_REF.get((config, variant), {}).get(int(eps)) if float(eps).is_integer() else None
    if ref is not None:
        res["sigma_sgd_reference"] = ref
        if abs(hi - ref) > C.SIGMA_REF_TOL:
            raise AssertionError(f"sigma {hi} differs from the spec's reference {ref} by more than 0.002")
    if not (math.isfinite(e) and e <= target + 1e-12):
        raise AssertionError(f"composed eps {e} > target {target}")
    return res


def prior_calib(calib: dict) -> dict:
    if calib["variant"] != "vp" or not calib["sigma_prior"]:
        raise ValueError("prior_calib only for the vp variant")
    return {"sigma_hist": calib["sigma_prior"], "n_hist": C.N_PRIOR, "eps_total": calib["eps_total"],
            "delta": calib["delta"], "hist_sensitivity_l2": 1.0,
            "note": "CTCL ctclvp: dploravp's prior noise sigma_prior (vp.py reads sigma_hist); see calib.json"}


def check_prior_calib(pc: dict, calib: dict) -> None:
    want = sigma_prior_for(calib["eps_total"], calib["delta"])
    if abs(pc["sigma_hist"] - want) > 1e-9 or abs(calib["sigma_prior"] - want) > 1e-9 or pc["n_hist"] != C.N_PRIOR:
        raise AssertionError(f"prior_calib sigma_hist {pc['sigma_hist']} != sigma_prior(eps) {want}")
    if abs(pc["sigma_hist"] - calib["sigma_eq"]) < 1e-9:
        raise AssertionError("prior_calib carries sigma_eq")


def cached(eps: float, variant: str, config: str) -> dict:
    C.CALIB_CACHE.mkdir(parents=True, exist_ok=True)
    p = C.CALIB_CACHE / f"{config}_{variant}_eps{C.eps_tag(eps)}.json"
    code = {"calibrate.py": C.sha256_file(HERE / "calibrate.py"), "ctcl_conf.py": C.sha256_file(HERE / "ctcl_conf.py")}
    if p.exists():
        res = json.loads(p.read_text())
        if res.get("code_sha256") == code:
            return res
    t0 = time.time()
    res = calibrate(eps, variant, config)
    res["calibrate_sec"] = round(time.time() - t0, 1)
    res["code_sha256"] = code
    C.write_json_atomic(p, res)
    return json.loads(p.read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eps", type=float, required=True)
    ap.add_argument("--variant", choices=["plain", "vp"], required=True)
    ap.add_argument("--config", choices=list(C.CONFIGS), default="code")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prior_out", default="")
    ap.add_argument("--no_cache", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        res = json.loads(out.read_text())
    else:
        res = calibrate(a.eps, a.variant, a.config) if a.no_cache else cached(a.eps, a.variant, a.config)
        C.write_json_atomic(out, res)
    if (res["variant"], res["config"], res["eps_total"]) != (a.variant, a.config, a.eps):
        raise SystemExit(f"{out} is the calibration of another (variant, config, eps)")
    if a.prior_out:
        pc = prior_calib(res)
        check_prior_calib(pc, res)
        pp = Path(a.prior_out)
        if pp.exists():
            old = json.loads(pp.read_text())
            if old != pc:
                raise SystemExit(f"{pp} differs from the calibration")
        else:
            C.write_json_atomic(pp, pc)
    print(json.dumps({k: res[k] for k in ("variant", "config", "eps_total", "sigma_sgd", "sigma_prior", "sigma_eq",
                                          "eps_composed_pld", "eps_sgd_only_pld")}), flush=True)


if __name__ == "__main__":
    main()
