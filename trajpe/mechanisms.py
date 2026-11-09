from __future__ import annotations

import numpy as np

from .dp import mu_for
from .embed import nearest_index, nearest_k
from .marginals import N_MARGINALS


def vote_weights(users: np.ndarray, unit: str) -> np.ndarray:
    if unit == "trail":
        return np.ones(len(users))
    if unit == "user":
        _, inv, cnt = np.unique(users, return_inverse=True, return_counts=True)
        return 1.0 / cnt[inv]
    raise ValueError(unit)


def noise_levels(eps: float, delta: float, T: int, use_marginals: bool, marg_frac: float) -> tuple[float, float]:
    if not np.isfinite(eps):
        return 0.0, 0.0
    mu2 = mu_for(eps, delta) ** 2
    marg_mu2 = (marg_frac if T > 0 else 1.0) * mu2 if use_marginals else 0.0
    sigma = float(np.sqrt(T / (mu2 - marg_mu2))) if T > 0 else 0.0
    sigma_m = float(np.sqrt(N_MARGINALS / marg_mu2)) if marg_mu2 > 0 else 0.0
    return sigma, sigma_m


def dp_nn_histogram(e_priv, e_syn, weights, sigma, threshold, rng, vote_k=1, labels=None, n_bins=None):
    if vote_k > 1:
        idx = nearest_k(e_priv, e_syn, vote_k)
        bins, w, nn = idx.ravel(), np.repeat(weights / np.sqrt(vote_k), vote_k), idx[:, 0]
    else:
        nn = nearest_index(e_priv, e_syn)
        bins, w = nn, weights
    if labels is not None:
        assert vote_k == 1, "clustered voting with top-k would exceed sensitivity 1"
        bins = labels[bins]
    counts = np.bincount(bins, weights=w, minlength=n_bins or len(e_syn))
    noisy = counts + rng.normal(0.0, sigma, len(counts)) if sigma > 0 else counts.astype(np.float64)
    noisy = np.where(noisy > threshold, noisy, 0.0)
    nn_dist = np.sqrt(np.maximum(((e_priv - e_syn[nn]) ** 2).sum(1), 0))
    return counts, noisy, nn_dist


def noise_levels_w(eps: float, delta: float, T: int, use_marginals: bool, marg_frac: float, work_frac: float,
                   n_work: int, extra_frac: float = 0.0) -> tuple[float, float, float]:
    if extra_frac > 0:
        if not np.isfinite(eps):
            return 0.0, 0.0, 0.0
        mu2 = mu_for(eps, delta) ** 2
        marg_mu2 = marg_frac * mu2 if use_marginals else 0.0
        work_mu2 = work_frac * mu2 if n_work > 0 else 0.0
        vote_mu2 = mu2 - marg_mu2 - work_mu2 - extra_frac * mu2
        if T <= 0 or vote_mu2 <= 0:
            raise ValueError("no budget left for voting")
        return (float(np.sqrt(T / vote_mu2)), float(np.sqrt(N_MARGINALS / marg_mu2)) if marg_mu2 > 0 else 0.0,
                float(np.sqrt(n_work / work_mu2)) if work_mu2 > 0 else 0.0)
    if work_frac <= 0 or n_work == 0:
        s, sm = noise_levels(eps, delta, T, use_marginals, marg_frac)
        return s, sm, 0.0
    if T <= 0:
        raise ValueError("workload marginals need T > 0 (their share is taken from the vote budget)")
    if not np.isfinite(eps):
        return 0.0, 0.0, 0.0
    mu2 = mu_for(eps, delta) ** 2
    marg_mu2 = marg_frac * mu2 if use_marginals else 0.0
    work_mu2 = work_frac * mu2
    vote_mu2 = mu2 - marg_mu2 - work_mu2
    if vote_mu2 <= 0:
        raise ValueError("no budget left for voting")
    sigma = float(np.sqrt(T / vote_mu2))
    sigma_m = float(np.sqrt(N_MARGINALS / marg_mu2)) if marg_mu2 > 0 else 0.0
    sigma_w = float(np.sqrt(n_work / work_mu2))
    return sigma, sigma_m, sigma_w


def extra_sigma(eps: float, delta: float, frac: float) -> float:
    return float(np.sqrt(1.0 / (frac * mu_for(eps, delta) ** 2))) if np.isfinite(eps) and frac > 0 else 0.0
