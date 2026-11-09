from __future__ import annotations
import os as _tp_os
def _tp_root():
    p = _tp_os.path.dirname(_tp_os.path.abspath(__file__))
    while not _tp_os.path.isfile(_tp_os.path.join(p, "trajpe", "__init__.py")) and _tp_os.path.dirname(p) != p:
        p = _tp_os.path.dirname(p)
    return p
_TP_ROOT = _tp_os.environ.get("TRAJPE_ROOT", _tp_root())
_TP_WORK = _tp_os.environ.get("TRAJPE_WORK", _tp_os.path.join(_TP_ROOT, "work"))
_TP_PY = _tp_os.environ.get("TRAJPE_PYTHON", "python")
_TP_PY_DPLORA = _tp_os.environ.get("TRAJPE_PYTHON_DPLORA", _TP_PY)
_TP_PY_AIMSEQ = _tp_os.environ.get("TRAJPE_PYTHON_AIMSEQ", _TP_PY)
_TP_PY_GRID = _tp_os.environ.get("TRAJPE_PYTHON_GRID", _TP_PY)

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import mpmath as mp
import numpy as np
from dp_accounting.rdp import rdp_privacy_accountant as rdp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pg_accounting as A

DELTA = 1e-6
R, N_MIN, N_MAX = 50, 100, 118
ORDERS = sorted(set(rdp.DEFAULT_RDP_ORDERS) | set(np.round(np.exp(np.linspace(np.log(1.05), np.log(1024), 300)),
                                                              6).tolist()))
RUNS = f"{_TP_WORK}/runs/baselines_v2"
mp.mp.dps = 60


def concave_ok(a: float, eps_max: float, n_grid: int = 200) -> bool:
    a = mp.mpf(a)
    c = (a - 1) / (2 * a - 1)
    k = 2 * a - 1
    for le in mp.linspace(mp.log(mp.mpf("1e-8")), mp.log(mp.mpf(eps_max)), n_grid):
        e = mp.e ** le
        ex = mp.e ** (-k * e)
        u = 1 + c * (ex - 1)
        u1 = -c * k * ex
        u2 = c * k * k * ex
        f1 = 1 + u1 / u / (a - 1)
        f2 = (u2 * u - u1 * u1) / (u * u) / (a - 1)
        if e * f2 - f1 > 0:
            return False
    return True


def lap(eps: float) -> np.ndarray:
    return np.array([rdp._laplace_rdp(eps, o) for o in ORDERS])


def budgets(eps: float) -> tuple:
    rho_total = A.cdp_rho(eps, DELTA)
    eps_h = 0.1 * eps / 6
    rho_r = (rho_total - 6 * eps_h ** 2 / 2) / R
    return rho_total, eps_h, rho_r


def worst_case_curve(eps: float, arm: str) -> np.ndarray:
    _, eps_h, rho_r = budgets(eps)
    per_n = []
    for n in range(N_MIN, N_MAX + 1):
        if arm == "main":
            per_n.append(n * lap(math.sqrt(0.2 * rho_r / n)) + 2 * n * lap(math.sqrt(0.9 * rho_r / n)))
        else:
            per_n.append(2 * n * lap(math.sqrt(rho_r / n)))
    return 6 * lap(eps_h) + R * np.max(np.array(per_n), axis=0)


def realised_curve(side: dict) -> np.ndarray:
    b, a = side["budget"], side["accounting"]
    curve = 6 * lap(b["eps_h"])
    for r in a["per_round"]:
        n = r["n_r"]
        if r.get("eps_p") is not None:
            curve = curve + n * lap(r["eps_p"])
        curve = curve + 2 * n * lap(math.sqrt(r["sum_eps_q_sq"] / n))
    return curve


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sidecars", nargs="*")
    ap.add_argument("--json_out", default="")
    a = ap.parse_args()
    paths = a.sidecars or sorted(glob.glob(f"{RUNS}/pategail/*_s0.json") + glob.glob(f"{RUNS}/pategail_u/*_s0.json")
                                 + glob.glob(f"{RUNS}/pategail/_rerun/*_s0.json"))
    dom = math.sqrt(0.9 * budgets(8.0)[2]) * 1.05
    bad = [o for o in ORDERS if not concave_ok(o, dom)]
    out = {"orders": len(ORDERS), "order_range": [ORDERS[0], ORDERS[-1]], "delta": DELTA,
           "concavity_domain_max_eps": dom, "orders_not_concave": bad, "part1_data_independent": {}}
    for eps in (1.0, 2.0, 4.0, 8.0):
        for arm in ("main", "uniform"):
            e, o = rdp.compute_epsilon(ORDERS, worst_case_curve(eps, arm).tolist(), DELTA)
            out["part1_data_independent"][f"{arm} eps {eps:g}"] = {
                "eps_exact_laplace_rdp": e, "order": o, "le_target": e <= eps,
                "eps_zcdp_cks": A.cdp_eps(A.cdp_rho(eps, DELTA), DELTA)}
    rows, worst = [], {}
    for p in paths:
        s = json.load(open(p))
        if not math.isfinite(float(s["eps"])):
            continue
        e, o = rdp.compute_epsilon(ORDERS, realised_curve(s).tolist(), DELTA)
        k = f"{s['method']} eps {float(s['eps']):g}"
        worst[k] = max(worst.get(k, 0.0), e)
        rows.append({"file": p, "method": s["method"], "eps": s["eps"], "eps_cks_sidecar": s["accounting"]["eps_spent"],
                     "eps_exact_rdp_realised": e, "order": o, "ok": e <= float(s["eps"])})
    out["part2_realised"] = {"n_releases": len(rows), "all_ok": all(r["ok"] for r in rows),
                             "worst_per_method_eps": worst}
    out["all_ok"] = (not bad) and out["part2_realised"]["all_ok"] and all(
        v["le_target"] for v in out["part1_data_independent"].values())
    print(json.dumps(out, indent=1))
    if a.json_out:
        Path(a.json_out).write_text(json.dumps({"summary": out, "releases": rows}, indent=1))
    sys.exit(0 if out["all_ok"] else 1)


if __name__ == "__main__":
    main()
