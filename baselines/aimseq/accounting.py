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

import math
import sys
from pathlib import Path

PGM_REPO = Path(f"{_TP_WORK}/baselines_src/private-pgm/repo")
if str(PGM_REPO / "mechanisms") not in sys.path:
    sys.path.insert(0, str(PGM_REPO / "mechanisms"))

import matplotlib

matplotlib.use("Agg")
from cdp2adp import cdp_delta, cdp_eps, cdp_rho


def budget(eps: float, delta: float, prior_frac: float = 0.1) -> dict:
    if not (eps > 0 and 0 < delta < 1 and 0 <= prior_frac < 1):
        raise ValueError("need eps > 0, 0 < delta < 1, 0 <= prior_frac < 1")
    rho_total = cdp_rho(eps, delta)
    eps_v = eps_c = 0.5 * prior_frac * eps
    rho_prior = 0.5 * eps_v ** 2 + 0.5 * eps_c ** 2
    rho_mech = rho_total - rho_prior
    if rho_mech <= 0:
        raise ValueError("prior share exhausts the budget")
    return {"eps": eps, "delta": delta, "prior_frac": prior_frac, "rho_total": rho_total,
            "eps_venue": eps_v, "eps_cat": eps_c, "rho_prior": rho_prior, "rho_mech": rho_mech,
            "rho_mech_share": rho_mech / rho_total,
            "conversion": "rho-zCDP -> (eps, delta)-DP: Canonne-Kamath-Steinke 2020 (mbi cdp2adp.cdp_rho); "
                          "Laplace eps'-DP -> eps'^2/2-zCDP; additive zCDP composition",
            "delta_check": cdp_delta(rho_total, eps), "eps_check": cdp_eps(rho_total, delta)}


class Ledger:

    def __init__(self):
        self.gauss: list = []
        self.em: list = []

    def add_gauss(self, sigma: float, size: int) -> None:
        self.gauss.append((float(sigma), int(size)))

    def add_em(self, eps_em: float, sensitivity: float, n: int) -> None:
        self.em.append((float(eps_em), float(sensitivity), int(n)))

    @property
    def rho(self) -> float:
        return math.fsum(0.5 / s ** 2 for s, _ in self.gauss) + math.fsum(e ** 2 / 8.0 for e, _, _ in self.em)

    def summary(self) -> dict:
        return {"n_gaussian": len(self.gauss), "n_em": len(self.em), "rho_spent": self.rho,
                "rho_gaussian": math.fsum(0.5 / s ** 2 for s, _ in self.gauss),
                "rho_em": math.fsum(e ** 2 / 8.0 for e, _, _ in self.em),
                "sigma_min": min((s for s, _ in self.gauss), default=None),
                "sigma_max": max((s for s, _ in self.gauss), default=None)}
