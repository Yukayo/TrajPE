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

N_REF = 74_685
DELTA = 1e-6
PRV_MARGIN = 0.015
MU_SAFETY = 1e-9
SCRUB_PRIVATE_DEFAULT = True
PRIVATE_KEYS = ("vs_train", "vs_test")
PRIVATE_NOTE = "removed (release mode: un-noised private-data metrics are never part of a release)"


def scrub_private(obj):
    if isinstance(obj, dict):
        out, n = {}, 0
        for k, v in obj.items():
            if k in PRIVATE_KEYS:
                n += 1
                continue
            out[k], m = scrub_private(v)
            n += m
        return out, n
    if isinstance(obj, list):
        out, n = [], 0
        for v in obj:
            c, m = scrub_private(v)
            out.append(c)
            n += m
        return out, n
    return obj, 0


def scrub_record(res: dict) -> tuple[dict, int]:
    clean, n = scrub_private(res)
    if n:
        clean["metrics"] = PRIVATE_NOTE
    return clean, n


def scrub_file(p: Path) -> int:
    if not p.exists():
        return 0
    clean, n = scrub_record(json.loads(p.read_text()))
    if n:
        tmp = p.with_name(p.name + ".scrub.tmp")
        tmp.write_text(json.dumps(clean, indent=1, default=float))
        tmp.replace(p)
    return n


def _acc():
    import dp_accounting
    from dp_accounting.pld import pld_privacy_accountant
    return dp_accounting, pld_privacy_accountant.PLDAccountant(
        neighboring_relation=dp_accounting.NeighboringRelation.ADD_OR_REMOVE_ONE, value_discretization_interval=1e-4)


def composed_eps(sigma_sgd: float, q: float, steps: int, mu_pe: float, delta: float = DELTA) -> float:
    dpa, acc = _acc()
    if math.isfinite(sigma_sgd) and steps > 0:
        acc.compose(dpa.PoissonSampledDpEvent(q, dpa.GaussianDpEvent(sigma_sgd)), steps)
    if mu_pe > 0:
        acc.compose(dpa.GaussianDpEvent(1.0 / mu_pe), 1)
    return float(acc.get_epsilon(delta))


def calibrate(eps: float, sgd_share: float, batch: int, epochs: float, delta: float = DELTA, tol: float = 0.005,
              margin: float = PRV_MARGIN) -> dict:
    from trajpe.dp import eps_for, mu_for
    if not (0.0 < sgd_share < 1.0) or not math.isfinite(eps):
        raise ValueError("the hybrid splits a finite budget: 0 < sgd_share < 1")
    q = batch / N_REF
    steps = int(round(epochs / q))
    mu = mu_for(eps, delta)
    mu_pe = math.sqrt(1.0 - sgd_share) * mu
    target = eps - margin
    if composed_eps(float("inf"), q, steps, mu_pe, delta) >= target:
        raise ValueError("the PE share alone exhausts the budget")
    lo, hi = 0.05, 1.0
    while composed_eps(hi, q, steps, mu_pe, delta) > target:
        lo, hi = hi, 2 * hi
    while True:
        mid = 0.5 * (lo + hi)
        if composed_eps(mid, q, steps, mu_pe, delta) > target:
            lo = mid
        else:
            hi = mid
        if target - composed_eps(hi, q, steps, mu_pe, delta) < tol and hi - lo < 1e-4:
            break
    return {"method": "hybrid", "eps_total": eps, "delta": delta, "pld_target": target, "mu_total": mu,
            "sgd_share": sgd_share, "pe_mu2_share": 1.0 - sgd_share, "mu_pe": mu_pe,
            "mu_pe_run": mu_pe * (1.0 - MU_SAFETY), "eps_pe_alone_gdp": eps_for(mu_pe, delta),
            "eps_pe_run": eps_for(mu_pe * (1.0 - MU_SAFETY), delta),
            "sigma_hist": 1.0 / mu_pe, "n_hist": 1, "hist_sensitivity_l2": 1.0,
            "sample_rate": q, "steps": steps, "sigma_sgd": hi,
            "eps_composed_pld": composed_eps(hi, q, steps, mu_pe, delta),
            "eps_sgd_only_pld": composed_eps(hi, q, steps, 0.0, delta),
            "eps_pe_only_pld": composed_eps(float("inf"), q, steps, mu_pe, delta),
            "accountant": "dp_accounting PLD, add-or-remove-one, value_discretization_interval=1e-4"}


def check(calib: dict, mu_pe_spent: float) -> dict:
    if mu_pe_spent > calib["mu_pe"] * (1 + 1e-12):
        raise SystemExit(f"PE spent mu {mu_pe_spent} > planned {calib['mu_pe']}")
    e = composed_eps(calib["sigma_sgd"], calib["sample_rate"], calib["steps"], mu_pe_spent, calib["delta"])
    if e > calib["eps_total"]:
        raise SystemExit(f"composed PLD epsilon {e} exceeds {calib['eps_total']}")
    return {"eps_composed_pld_spent": e, "mu_pe_spent": mu_pe_spent, "eps_total": calib["eps_total"],
            "sigma_sgd": calib["sigma_sgd"], "sample_rate": calib["sample_rate"], "steps": calib["steps"]}


def sidecar(dataset: str, tag: str, runs: Path | None = None, results: Path | None = None,
            scrub: bool | None = None) -> dict:
    from trajpe.paths import RESULTS, RUNS
    runs, results = Path(runs or RUNS), Path(results or RESULTS)
    scrub = SCRUB_PRIVATE_DEFAULT if scrub is None else bool(scrub)
    out = runs / dataset / tag / "result.json"
    res = json.loads(out.read_text())
    src = res.get("source")
    if not src or src == tag:
        raise SystemExit(f"{tag} is not a post-processed release (no source run)")
    sres = json.loads((runs / dataset / src / "result.json").read_text())
    hy = sres.get("hybrid")
    if not hy or "diagnostic" in hy or "calib" not in hy:
        raise SystemExit(f"source run {src} is not a hybrid release run")
    for k in ("privacy", "pe", "dataset"):
        if res.get(k) != sres.get(k):
            raise SystemExit(f"{tag}: '{k}' differs from its source run {src}; not its post-processing")
    mc = (res.get("postprocess") or {}).get("marginals_check") or {}
    if mc.get("marg_sha256") != sres["privacy"].get("marg_sha256"):
        raise SystemExit(f"{tag}: DP-marginal digest does not match the source run")
    work = Path(hy["generator"]["work"])
    tj = json.loads((work / "train.json").read_text())
    cal = json.loads((work / "calib.json").read_text())
    if cal != hy["calib"]:
        raise SystemExit(f"{work}/calib.json differs from the calibration recorded by the run")
    sigma, q, steps = float(tj["noise_multiplier"]), float(tj["sample_rate"]), int(tj["steps"])
    if (sigma, q, steps) != (float(cal["sigma_sgd"]), float(cal["sample_rate"]), int(cal["steps"])) or \
            int(tj.get("steps_run", steps)) != steps or tj.get("complete") is False:
        raise SystemExit("train.json (sigma, q, T, steps run) does not match the calibration")
    delta, eps_total = float(hy["delta"]), float(hy["eps_total"])
    if abs(float(tj["eps_target"]) - eps_total) > 1e-12 or abs(float(tj["delta"]) - delta) > 0:
        raise SystemExit("train.json eps / delta do not match the run")
    mu_pe = float(sres["privacy"]["mu"])
    if mu_pe > float(cal["mu_pe"]) * (1 + 1e-12):
        raise SystemExit(f"PE spent mu {mu_pe} > planned {cal['mu_pe']}")
    e = composed_eps(sigma, q, steps, mu_pe, delta)
    if e > eps_total:
        raise SystemExit(f"composed PLD epsilon {e} exceeds {eps_total}")
    block = {
        "eps_total": eps_total, "delta": delta, "eps_composed_pld": e,
        "dp_sgd": {"noise_multiplier": sigma, "sample_rate": q, "steps": steps, "clip": tj.get("clip"),
                   "model": tj.get("model"), "sampling": tj.get("sampling"), "work": str(work),
                   "eps_sgd_only_pld": composed_eps(sigma, q, steps, 0.0, delta),
                   "eps_prv_composed_train": tj.get("eps_prv"), "eps_rdp_composed_train": tj.get("eps_rdp")},
        "pe": {"mu_pe_spent": mu_pe, "mu_pe_planned": float(cal["mu_pe"]), "eps_pe_arg": hy.get("eps_pe_arg"),
               "eps_pe_only_pld": composed_eps(float("inf"), q, steps, mu_pe, delta),
               "eps_check_gdp": sres["privacy"].get("eps_check")},
        "sgd_share": cal.get("sgd_share"), "calib": cal, "pool": hy.get("pool"), "source_run": src,
        "accountant": "dp_accounting PLD, add-or-remove-one, value_discretization_interval=1e-4 (recomputed by "
                      "baselines/dplora/hybrid.py sidecar)",
        "guarantee": hy.get("guarantee"),
        "note": "the 'privacy' block above is the PE part only (mu_PE-GDP); this block is the release's total "
                "accounting (DP-SGD of the generator composed with PE, PLD); the release post-processing adds nothing"}
    if abs(e - float(hy["check"]["eps_composed_pld_spent"])) > 1e-9:
        raise SystemExit(f"recomputed eps {e} differs from the run's check {hy['check']['eps_composed_pld_spent']}")
    res["hybrid"] = block
    removed = 0
    if scrub:
        res, removed = scrub_record(res)
    txt = json.dumps(res, indent=1, default=float)
    out.write_text(txt)
    dest = results / "pe" / dataset / f"{tag}.json"
    if dest.exists():
        if scrub:
            removed += scrub_private(json.loads(dest.read_text()))[1]
        dest.write_text(txt)
    if scrub:
        removed += scrub_file(runs / dataset / src / "result.json") + scrub_file(results / "pe" / dataset / f"{src}.json")
        print(f"release mode: {removed} private-data metric entr{'y' if removed == 1 else 'ies'} (vs_train / vs_test) "
              f"removed from {tag}, {src} and their results/pe copies", file=sys.stderr)
    return block


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("--eps", type=float, required=True)
    c.add_argument("--sgd_share", type=float, required=True, help="share of mu^2 NOT given to PE")
    c.add_argument("--batch", type=int, required=True)
    c.add_argument("--epochs", type=float, required=True)
    c.add_argument("--delta", type=float, default=DELTA)
    c.add_argument("--out", required=True)
    k = sub.add_parser("check")
    k.add_argument("--calib", required=True)
    k.add_argument("--mu_pe", type=float, required=True)
    sc = sub.add_parser("sidecar")
    sc.add_argument("--dataset", required=True)
    sc.add_argument("--tag", required=True, help="the released (post-processed) run, e.g. hyb_V0.3_e4_s0div")
    sc.add_argument("--scrub_private", action=argparse.BooleanOptionalAction, default=None,
                    help="release mode: remove vs_train / vs_test from the release, its source run and their results/pe "
                         "copies (default: SCRUB_PRIVATE_DEFAULT)")
    args = ap.parse_args()
    t0 = time.time()
    if args.cmd == "calibrate":
        out = Path(args.out)
        if out.exists():
            print(out.read_text())
            return
        res = calibrate(args.eps, args.sgd_share, args.batch, args.epochs, args.delta)
        res["calibrate_sec"] = round(time.time() - t0, 1)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1))
        print(json.dumps(res))
    elif args.cmd == "sidecar":
        b = sidecar(args.dataset, args.tag, scrub=args.scrub_private)
        print(json.dumps({"tag": args.tag, "eps_composed_pld": b["eps_composed_pld"], "eps_total": b["eps_total"],
                          "sigma_sgd": b["dp_sgd"]["noise_multiplier"], "q": b["dp_sgd"]["sample_rate"],
                          "T": b["dp_sgd"]["steps"], "mu_pe": b["pe"]["mu_pe_spent"]}))
    else:
        print(json.dumps(check(json.loads(Path(args.calib).read_text()), args.mu_pe)))


if __name__ == "__main__":
    main()
