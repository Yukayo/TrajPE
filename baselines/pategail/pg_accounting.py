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

import hashlib
import math
import types
from pathlib import Path

import numpy as np

DELTA = 1e-6
REL_TOL = 1e-12
CDP2ADP = Path(f"{_TP_WORK}/baselines_src/private-pgm/repo/mechanisms/cdp2adp.py")
CDP2ADP_SHA256 = "6fc56098f6f8af0f43d68fb74a5d4c1d4ddb5a228d9ce1d976d26bca4233f3d8"
CONVERSION = ("Laplace eps'-DP -> (eps'^2/2)-zCDP (Bun & Steinke 2016, Prop. 1.4); additive zCDP composition with a "
              "zCDP (Renyi) filter for the adaptive per-query budgets (Feldman & Zrnic 2021); rho-zCDP -> (eps, delta)"
              "-DP by Canonne-Kamath-Steinke 2020 (official mbi cdp2adp.cdp_rho / cdp_eps)")

_cdp: dict = {}


def _cdp2adp() -> types.ModuleType:
    if "m" not in _cdp:
        b = CDP2ADP.read_bytes()
        if hashlib.sha256(b).hexdigest() != CDP2ADP_SHA256:
            raise RuntimeError(f"{CDP2ADP}: sha256 differs from the pinned {CDP2ADP_SHA256}")
        import matplotlib
        matplotlib.use("Agg")
        mod = types.ModuleType("pategail_cdp2adp")
        exec(compile(b.decode(), str(CDP2ADP), "exec"), mod.__dict__)
        _cdp["m"] = mod
    return _cdp["m"]


def cdp_rho(eps: float, delta: float = DELTA) -> float:
    return _cdp2adp().cdp_rho(eps, delta)


def cdp_eps(rho: float, delta: float = DELTA) -> float:
    return _cdp2adp().cdp_eps(rho, delta)


def cdp_delta(rho: float, eps: float) -> float:
    return _cdp2adp().cdp_delta(rho, eps)


def calibrate(eps: float, K: int, rounds: int, aux_frac: float = 0.1, n_aux: int = 6, pilot_share: float = 0.1,
              delta: float = DELTA) -> dict:
    if not math.isfinite(eps):
        return {"eps_total": math.inf, "rho_total": math.inf, "eps_h": math.inf, "rho_aux": math.inf,
                "rho_pate": math.inf, "rounds": rounds, "pilot_share": pilot_share, "K": K}
    if not (eps > 0 and 0 < delta < 1 and 0 < aux_frac < 1 and 0 <= pilot_share < 1):
        raise ValueError("bad budget parameters")
    rho_total = cdp_rho(eps, delta)
    eps_h = aux_frac * eps / n_aux
    rho_aux = n_aux * eps_h ** 2 / 2.0
    rho_pate = rho_total - rho_aux
    if rho_pate <= 0:
        raise ValueError("auxiliary share exhausts the budget")
    return {"eps_total": eps, "rho_total": rho_total, "eps_h": eps_h, "rho_aux": rho_aux, "rho_pate": rho_pate,
            "rounds": rounds, "pilot_share": pilot_share, "K": K}


def pilot_eps(rho_round: float, n_r: int, pilot_share: float) -> float:
    return math.sqrt(2.0 * pilot_share * rho_round / n_r)


def weights(pilot: np.ndarray, delta_w: float) -> np.ndarray:
    return 1.0 - np.clip(pilot, 0.0, 1.0) + delta_w


def main_eps(w: np.ndarray, rho_round: float, pilot_share: float) -> np.ndarray:
    w = np.asarray(w, np.float64)
    return w * math.sqrt((1.0 - pilot_share) * rho_round / float(np.sum(w * w)))


def uniform_eps(rho_round: float, n_r: int) -> np.ndarray:
    return np.full(n_r, math.sqrt(rho_round / n_r))


class BudgetExceeded(RuntimeError):
    pass


class Ledger:

    def __init__(self, rho_total: float, rho_aux: float):
        self.rho_total = float(rho_total)
        self.charges: list = [float(rho_aux)]
        self.log: list = [("M1-M6", 0, float(rho_aux))]
        self.rho_aux = float(rho_aux)
        self.n_releases = 0

    @property
    def rho(self) -> float:
        return math.fsum(self.charges)

    def charge(self, mech: str, rnd: int, eps: np.ndarray) -> float:
        eps = np.asarray(eps, np.float64).ravel()
        if not np.all(np.isfinite(eps)) or np.any(eps <= 0):
            raise BudgetExceeded(f"{mech} round {rnd}: invalid per-release budget")
        cost = math.fsum((eps * eps / 2.0).tolist())
        new = math.fsum(self.charges + [cost])
        if new > self.rho_total * (1.0 + REL_TOL):
            raise BudgetExceeded(f"{mech} round {rnd}: rho {new!r} would exceed rho_total {self.rho_total!r}")
        self.charges.append(cost)
        self.log.append((mech, rnd, cost))
        self.n_releases += len(eps)
        return cost


def _zcdp_orders(rho: float, delta: float) -> np.ndarray:
    grid = np.exp(np.linspace(np.log(1.02), np.log(1e5), 20001))
    f = grid * rho + np.log1p(-1.0 / grid) - (np.log(delta) + np.log(grid)) / (grid - 1.0)
    i = int(np.argmin(f))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    for _ in range(200):
        m1, m2 = lo + (hi - lo) * 0.381966, lo + (hi - lo) * 0.618034
        f1 = m1 * rho + math.log1p(-1 / m1) - (math.log(delta) + math.log(m1)) / (m1 - 1)
        f2 = m2 * rho + math.log1p(-1 / m2) - (math.log(delta) + math.log(m2)) / (m2 - 1)
        if f1 < f2:
            hi = m2
        else:
            lo = m1
    return np.concatenate([grid, [0.5 * (lo + hi)]])


def dpacc_eps_from_zcdp(rho: float, delta: float = DELTA) -> float:
    from dp_accounting.rdp import rdp_privacy_accountant as rdp
    orders = _zcdp_orders(rho, delta)
    eps, _ = rdp.compute_epsilon(orders.tolist(), (orders * rho).tolist(), delta)
    return float(eps)


def exact_laplace_rdp_eps(events: list, delta: float = DELTA) -> float:
    import dp_accounting
    from dp_accounting.rdp import rdp_privacy_accountant as rdp
    orders = sorted(set(rdp.DEFAULT_RDP_ORDERS) | set(np.round(np.exp(np.linspace(np.log(1.2), np.log(2000), 400)),
                                                                   6).tolist()))
    acc = rdp.RdpAccountant(orders=orders)
    for e, c in events:
        acc.compose(dp_accounting.LaplaceDpEvent(1.0 / float(e)), int(c))
    return float(acc.get_epsilon(delta))


def pld_eps(events: list, delta: float = DELTA, discretization: float = 1e-4) -> float:
    import dp_accounting
    from dp_accounting.pld import pld_privacy_accountant as pld
    acc = pld.PLDAccountant(value_discretization_interval=discretization)
    for e, c in events:
        acc.compose(dp_accounting.LaplaceDpEvent(1.0 / float(e)), int(c))
    return float(acc.get_epsilon(delta))


def rho_from_sidecar(acc: dict, rho_aux: float) -> float:
    parts = [float(rho_aux)]
    for r in acc["per_round"]:
        eps_p = r.get("eps_p")
        if eps_p is not None:
            parts.append(r["n_r"] * eps_p * eps_p / 2.0)
        parts.append(r["sum_eps_q_sq"])
    return math.fsum(parts)
