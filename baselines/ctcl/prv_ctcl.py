from __future__ import annotations

import argparse
import json
import math
import warnings


def prv_rdp(sigma: float, q: float, steps: int, gauss: list, delta: float = 1e-6) -> dict:
    from opacus.accountants import PRVAccountant, RDPAccountant
    if not (math.isfinite(sigma) and sigma > 0 and 0 < q <= 1 and steps > 0) or \
            any(not (math.isfinite(s) and s > 0 and c >= 1) for s, c in gauss):
        return {"eps_prv": float("inf"), "eps_rdp": float("inf")}
    hist = [(float(sigma), float(q), int(steps))] + [(float(s), 1.0, int(c)) for s, c in gauss]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prv, rdp = PRVAccountant(), RDPAccountant()
        prv.history = list(hist)
        rdp.history = list(hist)
        return {"eps_prv": float(prv.get_epsilon(delta=delta)), "eps_rdp": float(rdp.get_epsilon(delta=delta))}


def parse_gauss(items) -> list:
    out = []
    for it in items or []:
        s, c = it.split(":")
        out.append((float(s), int(c)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sigma", type=float, required=True)
    ap.add_argument("--q", type=float, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--gauss", action="append", default=[], help="sigma:count of plain Gaussian mechanisms")
    ap.add_argument("--delta", type=float, default=1e-6)
    a = ap.parse_args()
    print(json.dumps(prv_rdp(a.sigma, a.q, a.steps, parse_gauss(a.gauss), a.delta)), flush=True)


if __name__ == "__main__":
    main()
