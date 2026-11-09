from __future__ import annotations

import math
import warnings

from opacus.accountants import PRVAccountant, RDPAccountant
from opacus.accountants.utils import get_noise_multiplier

DELTA = 1e-6
EPS_TOL = 0.01


def calibrate_sigma(eps: float, q: float, steps: int, delta: float = DELTA) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(get_noise_multiplier(target_epsilon=eps, target_delta=delta, sample_rate=q, steps=steps,
                                          accountant="prv", epsilon_tolerance=EPS_TOL))


def epsilons_composed(sigma: float, q: float, steps: int, sigma_hist: float, n_hist: int,
                      delta: float = DELTA) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prv, rdp = PRVAccountant(), RDPAccountant()
        prv.history = [(sigma, q, steps), (sigma_hist, 1.0, n_hist)]
        rdp.history = [(sigma, q, steps), (sigma_hist, 1.0, n_hist)]
        return {"eps_prv": float(prv.get_epsilon(delta=delta)), "eps_rdp": float(rdp.get_epsilon(delta=delta))}


def epsilons(sigma: float, q: float, steps: int, delta: float = DELTA) -> dict:
    if not math.isfinite(sigma) or sigma <= 0:
        return {"eps_prv": float("inf"), "eps_rdp": float("inf")}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prv, rdp = PRVAccountant(), RDPAccountant()
        prv.history = [(sigma, q, steps)]
        rdp.history = [(sigma, q, steps)]
        return {"eps_prv": float(prv.get_epsilon(delta=delta)), "eps_rdp": float(rdp.get_epsilon(delta=delta))}
