from __future__ import annotations

import numpy as np
from scipy.stats import binom

from .embed import EmbedConfig, embed
from .privacy_eval import _min_dist, auc
from .trails import WEEK_MIN, TrailSet


def user_disjoint_split(users: np.ndarray, holdout_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    uniq = np.unique(users)
    held = set(rng.choice(uniq, int(round(holdout_frac * len(uniq))), replace=False))
    non = np.array([u in held for u in users])
    return ~non, non


def make_canaries(catalog, m: int, seed: int, length: int = 6) -> TrailSet:
    rng = np.random.default_rng(seed)
    trails = []
    for _ in range(m):
        pois = rng.integers(0, len(catalog), length)
        start = int(rng.integers(0, 7)) * 1440 + 120 + int(rng.integers(0, 60))
        trails.append((pois, (start + 30 * np.arange(length)) % WEEK_MIN))
    return TrailSet.from_lists(trails)


def concat(a: TrailSet, b: TrailSet) -> TrailSet:
    return TrailSet.from_lists(a.to_lists() + b.to_lists())


def eps_lower_bound(r: int, v: int, beta: float = 0.05) -> float:
    def tail(e):
        return binom.sf(v - 1, r, np.exp(e) / (1 + np.exp(e)))
    if tail(0.0) > beta:
        return 0.0
    lo, hi = 0.0, 20.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if tail(mid) <= beta else (lo, mid)
    return lo


def canary_audit(release: TrailSet, canaries: TrailSet, inserted: np.ndarray, catalog,
                 ecfg: EmbedConfig = EmbedConfig(), beta: float = 0.05) -> dict:
    d = _min_dist(embed(canaries, catalog, ecfg), embed(release, catalog, ecfg))
    score = -d
    order = np.argsort(-score)
    m = len(score)
    out = {"m": int(m), "n_inserted": int(inserted.sum()),
           "auc": auc(score[inserted], score[~inserted]) if 0 < inserted.sum() < m else float("nan")}
    best = 0.0
    for r in [m // 10, m // 4, m // 2, m]:
        r -= r % 2
        if r < 2:
            continue
        top, bottom = order[: r // 2], order[m - r // 2:]
        v = int(inserted[top].sum() + (~inserted[bottom]).sum())
        lb = eps_lower_bound(r, v, beta)
        out[f"r{r}_correct"] = v
        best = max(best, lb)
    out["eps_lower_bound"] = best
    return out


def user_mia(release: TrailSet, members: TrailSet, nonmembers: TrailSet, catalog, n: int = 10000,
             seed: int = 0, ecfg: EmbedConfig = EmbedConfig()) -> dict:
    rng = np.random.default_rng(seed)
    k = min(n, len(members), len(nonmembers))
    mem = members.subset(rng.choice(len(members), k, replace=False))
    non = nonmembers.subset(rng.choice(len(nonmembers), k, replace=False))
    e_rel = embed(release, catalog, ecfg)
    s_mem = -_min_dist(embed(mem, catalog, ecfg), e_rel)
    s_non = -_min_dist(embed(non, catalog, ecfg), e_rel)
    return {"user_mia_auc": auc(s_mem, s_non), "n_each": int(k)}
