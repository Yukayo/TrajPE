from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau

from .embed import return_flags
from .trails import TrailSet

DIST_BINS = np.array([0, .25, .5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 50, np.inf])
GAP_BINS = np.array([0, 5, 10, 20, 30, 45, 60, 90, 120, 180, 240, 360, 481, np.inf])
LOWER_IS_BETTER = ["density_jsd", "query_error", "od_jsd", "travel_jsd", "hop_jsd", "gyration_jsd",
                   "length_jsd", "hour_jsd", "dow_jsd", "gap_jsd", "cat_jsd", "root_jsd",
                   "root_trans_jsd", "cat_trans_jsd", "hour_root_jsd", "weekhour_jsd"]
HIGHER_IS_BETTER = ["pattern_f1", "poi_kendall"]


def jsd(p, q) -> float:
    p, q = np.asarray(p, np.float64), np.asarray(q, np.float64)
    if p.sum() == 0 or q.sum() == 0:
        return 1.0
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)

    def kl(a):
        nz = a > 0
        return np.sum(a[nz] * np.log2(a[nz] / m[nz]))
    return float(0.5 * kl(p) + 0.5 * kl(q))


def sparse_jsd(keys_a, keys_b) -> float:
    ua, ca = np.unique(keys_a, return_counts=True)
    ub, cb = np.unique(keys_b, return_counts=True)
    allk = np.union1d(ua, ub)
    pa = np.zeros(len(allk))
    pb = np.zeros(len(allk))
    pa[np.searchsorted(allk, ua)] = ca
    pb[np.searchsorted(allk, ub)] = cb
    return jsd(pa, pb)


class Fidelity:
    def __init__(self, catalog, grid: int = 16, od_grid: int = 8, n_queries: int = 1000,
                 query_res_km: float = 0.1, sanity_frac: float = 0.01, top_k: int = 100, seed: int = 0):
        self.c = catalog
        self.grid, self.od_grid, self.top_k = grid, od_grid, top_k
        self.sanity_frac = sanity_frac
        rng = np.random.default_rng(seed)
        centers = catalog.xy[rng.integers(0, len(catalog), n_queries)]
        half = rng.uniform(0.25, 2.5, size=(n_queries, 2))
        self.res = query_res_km
        self.nx = int(np.ceil(catalog.width_km / query_res_km)) + 1
        self.ny = int(np.ceil(catalog.height_km / query_res_km)) + 1
        lo = np.floor((centers - half) / query_res_km).astype(int)
        hi = np.floor((centers + half) / query_res_km).astype(int)
        self.q_lo = np.clip(lo, 0, [self.nx - 1, self.ny - 1])
        self.q_hi = np.clip(hi, 0, [self.nx - 1, self.ny - 1])

    def _cell(self, xy, g):
        gx = np.clip((xy[:, 0] / self.c.width_km * g).astype(np.int64), 0, g - 1)
        gy = np.clip((xy[:, 1] / self.c.height_km * g).astype(np.int64), 0, g - 1)
        return gy * g + gx

    def stats(self, ts: TrailSet) -> dict:
        c = self.c
        xy = c.xy[ts.poi]
        lens = ts.lengths
        first, last = ts.first, ts.last
        same = np.ones(len(ts.poi), bool)
        same[last] = False
        nxt = np.flatnonzero(same)
        hop = np.linalg.norm(xy[nxt + 1] - xy[nxt], axis=1)
        travel = np.bincount(ts.trail_of[nxt], weights=hop, minlength=len(ts))
        cent = np.add.reduceat(xy, first, axis=0) / lens[:, None]
        gyr = np.sqrt(np.add.reduceat(((xy - cent[ts.trail_of]) ** 2).sum(1), first) / lens)
        gaps = (ts.tmin[nxt + 1] - ts.tmin[nxt]).astype(np.float64)
        cell = self._cell(xy, self.grid)
        g = self.od_grid
        root, cat = c.root[ts.poi], c.cat[ts.poi]
        ix = np.clip((xy[:, 0] / self.res).astype(int), 0, self.nx - 1)
        iy = np.clip((xy[:, 1] / self.res).astype(int), 0, self.ny - 1)
        img = np.zeros((self.nx + 1, self.ny + 1))
        np.add.at(img, (ix + 1, iy + 1), 1)
        integ = img.cumsum(0).cumsum(1)
        lo, hi = self.q_lo, self.q_hi + 1
        q = integ[hi[:, 0], hi[:, 1]] - integ[lo[:, 0], hi[:, 1]] - integ[hi[:, 0], lo[:, 1]] + integ[lo[:, 0], lo[:, 1]]
        rep, ret = return_flags(ts)
        return {
            "n_trails": len(ts), "n_checkins": len(ts.poi),
            "density": np.bincount(cell, minlength=self.grid ** 2),
            "od": np.bincount(self._cell(xy[first], g) * g * g + self._cell(xy[last], g), minlength=g ** 4),
            "travel": np.histogram(travel, DIST_BINS)[0],
            "hop": np.histogram(hop, DIST_BINS)[0],
            "gyration": np.histogram(gyr, DIST_BINS)[0],
            "length": np.bincount(np.minimum(lens, 20), minlength=21),
            "hour": np.bincount(((ts.tmin % 1440) // 60).astype(int), minlength=24),
            "dow": np.bincount(np.clip(ts.tmin[first] // 1440, 0, 6), minlength=7),
            "gap": np.histogram(gaps, GAP_BINS)[0],
            "cat": np.bincount(cat, minlength=len(c.categories)),
            "root": np.bincount(root, minlength=c.n_roots),
            "root_trans": np.bincount(root[nxt] * c.n_roots + root[nxt + 1], minlength=c.n_roots ** 2),
            "hour_root": np.bincount(((ts.tmin % 1440) // 60).astype(np.int64) * c.n_roots + root,
                                     minlength=24 * c.n_roots),
            "weekhour": np.bincount(np.clip(ts.tmin // 60, 0, 7 * 24 * 2 - 1) % 168, minlength=168),
            "cat_trans_keys": cat[nxt].astype(np.int64) * len(c.categories) + cat[nxt + 1],
            "cell_trans_keys": cell[nxt] * self.grid ** 2 + cell[nxt + 1],
            "poi_counts": np.bincount(ts.poi, minlength=len(c)),
            "queries": q,
            "n_unique_trails": len({(tuple(p), tuple(t)) for p, t in ts.to_lists()}) if len(ts) <= 300000 else -1,
            "n_hops": len(nxt), "n_repeat": int(rep.sum()), "n_return": int(ret.sum()),
            "n_unique_seqs": len({tuple(p) for p, _ in ts.to_lists()}),
        }

    def _top(self, keys):
        u, cnt = np.unique(keys, return_counts=True)
        return set(u[np.argsort(-cnt, kind="stable")[: self.top_k]])

    def compare(self, real: dict, syn: dict) -> dict:
        out = {k + "_jsd": jsd(real[k], syn[k]) for k in
               ["density", "od", "travel", "hop", "gyration", "length", "hour", "dow", "gap", "cat", "root",
                "root_trans", "hour_root", "weekhour"]}
        out["cat_trans_jsd"] = sparse_jsd(real["cat_trans_keys"], syn["cat_trans_keys"])
        scale = real["n_checkins"] / max(syn["n_checkins"], 1)
        b = self.sanity_frac * real["n_checkins"]
        out["query_error"] = float(np.mean(np.abs(real["queries"] - scale * syn["queries"])
                                           / np.maximum(real["queries"], b)))
        ra, sa = self._top(real["cell_trans_keys"]), self._top(syn["cell_trans_keys"])
        out["pattern_f1"] = len(ra & sa) / max(len(ra), 1)
        top = np.argsort(-real["poi_counts"], kind="stable")[: self.top_k]
        out["poi_kendall"] = float(kendalltau(real["poi_counts"][top], syn["poi_counts"][top]).statistic)
        out["n_trails"] = syn["n_trails"]
        out["unique_frac"] = syn["n_unique_trails"] / max(syn["n_trails"], 1)
        if "n_unique_seqs" in syn:
            out["unique_seq_frac"] = syn["n_unique_seqs"] / max(syn["n_trails"], 1)
        if "n_hops" in syn and "n_hops" in real:
            for k in ("repeat", "return"):
                r, s_ = real[f"n_{k}"] / max(real["n_hops"], 1), syn[f"n_{k}"] / max(syn["n_hops"], 1)
                out[f"{k}_rate"], out[f"{k}_rate_err"] = s_, abs(s_ - r)
        return out


SCORE_METRICS = ["density_jsd", "od_jsd", "query_error", "travel_jsd", "hop_jsd", "gyration_jsd", "length_jsd",
                 "hour_jsd", "gap_jsd", "cat_jsd", "root_trans_jsd", "hour_root_jsd", "weekhour_jsd",
                 "pattern_f1", "poi_kendall"]


def composite_score(m: dict, floor: float = 1e-3) -> float:
    vals = []
    for k in SCORE_METRICS:
        v = m[k]
        if k == "poi_kendall" and not np.isfinite(v):
            v = 0.0
        vals.append(np.log10(max(1.0 - v if k in ("pattern_f1", "poi_kendall") else v, floor)))
    return float(np.mean(vals))
