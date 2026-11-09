from __future__ import annotations

import numpy as np

from .trails import TrailSet

UNION_STREAM = 220013
CLUSTER_MIN = 64


def concat(a: TrailSet, b: TrailSet) -> TrailSet:
    off = np.concatenate([a.offsets, a.offsets[-1] + b.offsets[1:]]).astype(np.int64)
    return TrailSet(np.concatenate([a.poi, b.poi.astype(a.poi.dtype)]),
                    np.concatenate([a.tmin, b.tmin.astype(a.tmin.dtype)]), off)


def js_shrink(y: np.ndarray, target: np.ndarray, sigma: float) -> tuple[np.ndarray, float]:
    y = np.asarray(y, np.float64)
    t = np.asarray(target, np.float64)
    B = len(y)
    if B < 3 or sigma <= 0:
        return y.copy(), 1.0
    d = y - t
    ss = float(d @ d)
    c = max(0.0, 1.0 - (B - 2) * sigma ** 2 / ss) if ss > 0 else 0.0
    return t + c * d, c


def shrink_work(work: dict, names, pool: TrailSet, catalog, n_hat: float, sigma_w: float, hist_fn) -> tuple[dict, dict]:
    out, cs = dict(work), {}
    ones = np.ones(len(pool))
    for k in names:
        if k not in work:
            continue
        g = hist_fn(pool, ones, catalog, k)
        g = g / max(g.sum(), 1e-12)
        out[k], cs[k] = js_shrink(work[k], n_hat * g, sigma_w)
    return out, cs


def cluster_k(n_hat: float, sigma: float, kappa: float, n_syn: int) -> int:
    hi = max(CLUSTER_MIN, n_syn // 8)
    return int(np.clip(np.floor(n_hat / (kappa * sigma)), CLUSTER_MIN, hi))


def spread_cluster_votes(noisy_c: np.ndarray, labels: np.ndarray) -> np.ndarray:
    size = np.bincount(labels, minlength=len(noisy_c)).astype(np.float64)
    return np.asarray(noisy_c, np.float64)[labels] / np.maximum(size[labels], 1.0)
