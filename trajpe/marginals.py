from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .trails import TrailSet

N_MARGINALS = 5
TOP_CATEGORIES = 200
TOP_VENUES = 500


@dataclass
class Marginals:
    pop: np.ndarray
    cat: np.ndarray
    start: np.ndarray
    sigma: float
    cat_dom: np.ndarray = None
    cat_trans: np.ndarray = None
    venue_dom: np.ndarray = None
    venue_trans: np.ndarray = None
    pop_raw: np.ndarray = None

    def cat_share(self) -> np.ndarray:
        s = self.cat.sum()
        return self.cat / s if s > 0 else np.full(len(self.cat), 1.0 / len(self.cat))

    def venue_weights(self, alpha_share: float, venue_cat: np.ndarray) -> np.ndarray:
        pop = np.maximum(self.pop, 0.0)
        total = pop.sum()
        floor_mass = alpha_share / (1.0 - alpha_share) * total if total > 0 else 1.0
        n_in_cat = np.bincount(venue_cat, minlength=len(self.cat)).astype(np.float64)
        per_venue = self.cat_share() / np.maximum(n_in_cat, 1.0)
        floor = per_venue[venue_cat]
        floor = floor_mass * floor / floor.sum() if floor.sum() > 0 else np.full(len(pop), floor_mass / len(pop))
        return pop + floor + 1e-12

    def start_probs(self, floor: float = 1e-3) -> np.ndarray:
        p = np.maximum(self.start, 0.0)
        p = p / p.sum() if p.sum() > 0 else np.ones_like(p) / len(p)
        p = p + floor / len(p)
        return p / p.sum()

    def digest(self) -> str:
        import hashlib
        h = hashlib.sha256()
        for a in (self.pop, self.cat, self.start, self.cat_dom, self.cat_trans, self.venue_dom, self.venue_trans):
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()

    def transitions(self):
        return self.cat_dom, self.cat_trans, self.venue_dom, self.venue_trans


def start_bins(ts: TrailSet) -> np.ndarray:
    t = ts.tmin[ts.first]
    return np.clip(t // 1440, 0, 6) * 24 + (t % 1440) // 60


def threshold_sigmas(n_bins: int, fp_target: float = 100.0) -> float:
    return float(max(2.0, norm.isf(fp_target / n_bins)))


popularity_threshold_sigmas = threshold_sigmas


def _transition_hist(priv: TrailSet, codes: np.ndarray, dom: np.ndarray, n_codes: int, w_trail: np.ndarray):
    pos = np.full(n_codes, -1, dtype=np.int64)
    pos[dom] = np.arange(len(dom))
    has_next = np.ones(len(priv.poi), bool)
    has_next[priv.last] = False
    i = np.flatnonzero(has_next)
    a, b = pos[codes[i]], pos[codes[i + 1]]
    keep = (a >= 0) & (b >= 0)
    w = (w_trail / np.maximum(priv.lengths - 1, 1))[priv.trail_of[i]]
    k = len(dom)
    return np.bincount(a[keep] * k + b[keep], weights=w[keep], minlength=k * k).reshape(k, k).astype(np.float64)


def _noisy_threshold(counts: np.ndarray, sigma: float, rng) -> np.ndarray:
    noisy = counts + rng.normal(0.0, sigma, counts.shape)
    return np.where(noisy > threshold_sigmas(counts.size) * sigma, noisy, 0.0)


def dp_marginals(priv: TrailSet, trail_weights: np.ndarray, venue_cat: np.ndarray, n_categories: int,
                 sigma: float, pop_threshold_sigmas: float | None, rng) -> Marginals:
    n_venues = len(venue_cat)
    w_check = (trail_weights / priv.lengths)[priv.trail_of]
    pop = np.bincount(priv.poi, weights=w_check, minlength=n_venues).astype(np.float64)
    cat = np.bincount(venue_cat[priv.poi], weights=w_check, minlength=n_categories).astype(np.float64)
    start = np.bincount(start_bins(priv), weights=trail_weights, minlength=168).astype(np.float64)
    if sigma > 0:
        z = pop_threshold_sigmas if pop_threshold_sigmas is not None else threshold_sigmas(n_venues)
        pop = pop + rng.normal(0.0, sigma, n_venues)
        cat = cat + rng.normal(0.0, sigma, n_categories)
        start = start + rng.normal(0.0, sigma, 168)
        pop_raw = pop
        pop = np.where(pop > z * sigma, pop, 0.0)
    else:
        pop_raw = pop
    cat, start = np.maximum(cat, 0.0), np.maximum(start, 0.0)

    cat_dom = np.argsort(-cat, kind="stable")[:TOP_CATEGORIES]
    nz = np.flatnonzero(pop > 0)
    venue_dom = nz[np.argsort(-pop[nz], kind="stable")[:TOP_VENUES]]
    cat_trans = _transition_hist(priv, venue_cat[priv.poi], cat_dom, n_categories, trail_weights)
    venue_trans = _transition_hist(priv, priv.poi, venue_dom, n_venues, trail_weights)
    if sigma > 0:
        cat_trans = _noisy_threshold(cat_trans, sigma, rng)
        if venue_trans.size:
            venue_trans = _noisy_threshold(venue_trans, sigma, rng)
    return Marginals(pop=pop, cat=cat, start=start, sigma=sigma, cat_dom=cat_dom, cat_trans=cat_trans,
                     venue_dom=venue_dom, venue_trans=venue_trans, pop_raw=pop_raw)


def dp_venue_popularity(priv: TrailSet, trail_weights: np.ndarray, n_venues: int, sigma: float, rng) -> np.ndarray:
    w_check = (trail_weights / priv.lengths)[priv.trail_of]
    h = np.bincount(priv.poi, weights=w_check, minlength=n_venues).astype(np.float64)
    return h + rng.normal(0.0, sigma, n_venues) if sigma > 0 else h


def combine_popularity(marg: Marginals, second: np.ndarray, sigma2: float, pop_threshold_sigmas: float | None):
    from dataclasses import replace
    s1, s2 = marg.sigma, sigma2
    prec = 1.0 / s1 ** 2 + 1.0 / s2 ** 2
    comb = (marg.pop_raw / s1 ** 2 + second / s2 ** 2) / prec
    sc = float(np.sqrt(1.0 / prec))
    z = pop_threshold_sigmas if pop_threshold_sigmas is not None else threshold_sigmas(len(comb))
    return replace(marg, pop=np.where(comb > z * sc, comb, 0.0), pop_raw=comb), sc
