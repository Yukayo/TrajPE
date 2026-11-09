from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def gdp_delta(mu: float, eps: float) -> float:
    return float(norm.cdf(-eps / mu + mu / 2) - np.exp(eps) * norm.cdf(-eps / mu - mu / 2))


def mu_for(eps: float, delta: float) -> float:
    return brentq(lambda m: gdp_delta(m, eps) - delta, 1e-8, 500.0, xtol=1e-12)


def eps_for(mu: float, delta: float) -> float:
    return brentq(lambda e: gdp_delta(mu, e) - delta, 0.0, 500.0, xtol=1e-12)


def sigma_for(eps: float, delta: float, n_releases: int, sensitivity: float = 1.0,
              reserved_mu: float = 0.0) -> float:
    mu = mu_for(eps, delta)
    rest = mu ** 2 - reserved_mu ** 2
    if rest <= 0:
        raise ValueError("reserved budget exceeds total budget")
    return sensitivity * np.sqrt(n_releases / rest)
