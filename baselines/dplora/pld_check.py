from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import dp_accounting
from dp_accounting.pld import pld_privacy_accountant


def pld_eps(sigma: float, q: float, steps: int, delta: float, sigma_hist: float = 0.0, n_hist: int = 0) -> float:
    acc = pld_privacy_accountant.PLDAccountant(
        neighboring_relation=dp_accounting.NeighboringRelation.ADD_OR_REMOVE_ONE,
        value_discretization_interval=1e-4)
    acc.compose(dp_accounting.PoissonSampledDpEvent(q, dp_accounting.GaussianDpEvent(sigma)), steps)
    if n_hist:
        acc.compose(dp_accounting.GaussianDpEvent(sigma_hist), n_hist)
    return float(acc.get_epsilon(delta))


def main():
    rows, bad = [], []
    for f in sys.argv[1:]:
        if f.endswith(".eval.json"):
            continue
        s = json.loads(Path(f).read_text())
        if "noise_multiplier" not in s:
            continue
        ps = s.get("private_stats")
        if ps is not None and (ps.get("oom_retries", 0) != 0 or s.get("complete") is False):
            bad.append(f"{f}: release gate (oom_retries={ps.get('oom_retries')}, complete={s.get('complete')})")
        e_sgd = pld_eps(s["noise_multiplier"], s["sample_rate"], s["steps"], s["delta"])
        v = s.get("vp_calibration")
        e = e_sgd if not v else pld_eps(s["noise_multiplier"], s["sample_rate"], s["steps"], s["delta"],
                                        float(v["sigma_hist"]), int(v["n_hist"]))
        rows.append({"file": Path(f).name, "eps_target": s["eps_target"], "sigma": s["noise_multiplier"],
                     "q": s["sample_rate"], "T": s["steps"], "eps_prv": s["eps_prv"], "eps_rdp": s["eps_rdp"],
                     "eps_pld": e, "eps_pld_sgd_part": e_sgd if v else None,
                     "sigma_hist": v["sigma_hist"] if v else None, "n_hist": v["n_hist"] if v else 0})
        extra = f"  (DP-SGD part {e_sgd:.4f}; + {v['n_hist']} x Gaussian(sigma_hist={v['sigma_hist']:.4f}))" if v else ""
        print(f"{Path(f).name:40s} target {s['eps_target']:g}: sigma={s['noise_multiplier']:.4f} q={s['sample_rate']:.6f} "
              f"T={s['steps']} | PRV {s['eps_prv']:.4f}  PLD {e:.4f}  RDP {s['eps_rdp']:.4f}{extra}")
        if math.isfinite(e) and e > s["eps_target"]:
            bad.append(f"{f}: composed PLD epsilon {e:.4f} exceeds the target {s['eps_target']}")
    for b in bad:
        print("FAIL", b)
    assert not bad, "accounting / release-gate check failed"
    return rows


if __name__ == "__main__":
    main()
