from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .trails import WEEK_MIN, TrailSet


def normalize_times(t: np.ndarray) -> np.ndarray:
    t = np.maximum.accumulate(np.round(t).astype(np.int64))
    return (t - (t[0] // WEEK_MIN) * WEEK_MIN).astype(np.int32)


def weighted_rows(cand: np.ndarray, weights: np.ndarray, rng) -> np.ndarray:
    w = weights[cand]
    cdf = np.cumsum(w, axis=1)
    u = rng.random(len(cand))[:, None] * cdf[:, -1:]
    return cand[np.arange(len(cand)), np.minimum((cdf < u).sum(1), cand.shape[1] - 1)]


class PopularIndex:

    def __init__(self, catalog, popular, radius_km: float, k: int = 8, scale_km: float = 0.5):
        idx, w = popular
        self.idx, self.w = np.asarray(idx, dtype=np.int64), np.asarray(w, dtype=np.float64)
        self.tree = cKDTree(catalog.xy[self.idx]) if len(self.idx) else None
        self.r, self.k, self.scale = radius_km, min(k, max(len(self.idx), 1)), scale_km

    def draw(self, points, rng):
        out = np.full(len(points), -1, dtype=np.int64)
        if self.tree is None or len(points) == 0:
            return out
        d, j = self.tree.query(points, k=self.k, distance_upper_bound=self.r)
        d, j = d.reshape(len(points), -1), j.reshape(len(points), -1)
        valid = np.isfinite(d)
        jj = np.where(valid, j, 0)
        wts = np.where(valid, self.w[jj] * np.exp(-np.where(valid, d, 0.0) / self.scale), 0.0)
        has = wts.sum(1) > 0
        cdf = np.cumsum(wts, 1)
        u = rng.random(len(points))[:, None] * cdf[:, -1:]
        pick = np.minimum((cdf < u).sum(1), jj.shape[1] - 1)
        out[has] = self.idx[jj[np.arange(len(points)), pick]][has]
        return out


class TransitionModel:

    def __init__(self, catalog, transitions, radius_km: float = 2.0, k: int = 10):
        cat_dom, cat_trans, venue_dom, venue_trans = transitions
        self.c, self.r, self.k = catalog, radius_km, k
        self.venue_w = None
        self.v_pos = np.full(len(catalog), -1, dtype=np.int64)
        self.v_dom = np.asarray(venue_dom, dtype=np.int64)
        self.v_pos[self.v_dom] = np.arange(len(self.v_dom))
        self.v_trans = np.asarray(venue_trans, dtype=np.float64)
        self.c_pos = np.full(len(catalog.categories), -1, dtype=np.int64)
        self.c_dom = np.asarray(cat_dom, dtype=np.int64)
        self.c_pos[self.c_dom] = np.arange(len(self.c_dom))
        self.c_trans = np.asarray(cat_trans, dtype=np.float64)

    def venue_successor(self, v: int, rng) -> int:
        i = self.v_pos[v]
        if i >= 0 and self.v_trans.size and self.v_trans[i].sum() > 0:
            row = self.v_trans[i]
            return int(self.v_dom[rng.choice(len(row), p=row / row.sum())])
        return -1

    def successor(self, v: int, rng) -> int:
        c = self.c
        i = self.v_pos[v]
        if i >= 0 and self.v_trans.size and self.v_trans[i].sum() > 0:
            row = self.v_trans[i]
            return int(self.v_dom[rng.choice(len(row), p=row / row.sum())])
        j = self.c_pos[c.cat[v]]
        if j >= 0 and self.c_trans[j].sum() > 0:
            row = self.c_trans[j]
            nxt_cat = int(self.c_dom[rng.choice(len(row), p=row / row.sum())])
            cand = c.nearest(c.xy[v], cat=nxt_cat, k=self.k)
            if cand is None:
                return -1
            cand = np.atleast_1d(cand)
            cand = cand[c.dist_km(cand, v) <= self.r]
            if len(cand) == 0:
                return -1
            if self.venue_w is None:
                return int(cand[rng.integers(len(cand))])
            w = self.venue_w[cand]
            return int(cand[rng.choice(len(cand), p=w / w.sum())])
        return -1


N_BLOCKS = 8


def _draw_rows(P: np.ndarray, rng) -> np.ndarray:
    cdf = np.cumsum(P, axis=1)
    u = rng.random(len(P)) * cdf[:, -1]
    return np.minimum((cdf < u[:, None]).sum(1), P.shape[1] - 1)


class SemanticModel:

    def __init__(self, catalog, venue_w: np.ndarray, T: np.ndarray, H: np.ndarray, k: int = 10, scale_km: float = 0.5):
        self.c, self.venue_w = catalog, np.asarray(venue_w, np.float64)
        self.T = np.asarray(T, np.float64)
        H = np.maximum(np.asarray(H, np.float64), 0.0) + 1e-12
        self.start_p = H / H.sum(1, keepdims=True)
        marg = H.sum(0) / H.sum()
        self.tau = np.clip(self.start_p / np.maximum(marg, 1e-12)[None, :], 0.05, 20.0)
        self.k, self.scale = int(k), float(scale_km)
        self._cdf = {}

    @staticmethod
    def blocks(tmin) -> np.ndarray:
        return ((np.asarray(tmin) % 1440) // 180).astype(np.int64)

    @classmethod
    def from_dp(cls, catalog, marg, work: dict, sigma_w: float, n_hat: float, venue_w, k=10, scale_km=0.5,
                known_sigmas: float = 2.0):
        C, R = len(catalog.categories), catalog.n_roots
        root_of = np.asarray(catalog.root_of_cat, np.int64)
        share = marg.cat_share()
        rshare = np.bincount(root_of, weights=share, minlength=R)
        if "root_trans" in work:
            RT = np.maximum(np.asarray(work["root_trans"], np.float64).reshape(R, R), 0.0) + 1e-9
            RT = RT / RT.sum(1, keepdims=True)
        else:
            RT = np.tile(rshare / max(rshare.sum(), 1e-12), (R, 1))
        base = share[None, :] * RT[root_of][:, root_of] / np.maximum(rshare[root_of], 1e-12)[None, :]
        base = base / np.maximum(base.sum(1, keepdims=True), 1e-300)
        T = base.copy()
        if marg.cat_trans is not None and marg.cat_trans.size:
            dom = np.asarray(marg.cat_dom, np.int64)
            for i in np.flatnonzero(marg.cat_trans.sum(1) > 0):
                row = np.zeros(C)
                row[dom] = marg.cat_trans[i]
                cf = dom[i]
                total = max(n_hat * share[cf], row.sum())
                b = base[cf] * (row == 0)
                b = b / b.sum() if b.sum() > 0 else b
                T[cf] = (row + (total - row.sum()) * b) / total
        if "hb_cat" in work:
            Hn = np.asarray(work["hb_cat"], np.float64).reshape(N_BLOCKS, C)
            kept = Hn > known_sigmas * sigma_w
            Hk = np.where(kept, Hn, 0.0)
            if "weekhour" in work:
                wh = np.maximum(np.asarray(work["weekhour"], np.float64), 0.0)
                tot = np.bincount((np.arange(168) % 24) // 3, weights=wh, minlength=N_BLOCKS)
            else:
                tot = np.maximum(Hn.sum(1), 0.0)
            fill = share[None, :] * ~kept
            fill = fill / np.maximum(fill.sum(1, keepdims=True), 1e-300)
            H = Hk + np.maximum(tot - Hk.sum(1), 0.0)[:, None] * fill
        elif "hour_root" in work:
            HR = np.maximum(np.asarray(work["hour_root"], np.float64).reshape(24, R), 0.0)
            HB = HR.reshape(N_BLOCKS, 3, R).sum(1) + 1e-9
            pr_b = HB / HB.sum(1, keepdims=True)
            H = (pr_b[:, root_of] / np.maximum(rshare[root_of], 1e-12)[None, :]) * share[None, :]
        else:
            H = np.tile(share, (N_BLOCKS, 1))
        return cls(catalog, venue_w, T, H, k, scale_km)

    def start_cat(self, blk, rng) -> np.ndarray:
        return _draw_rows(self.start_p[blk], rng)

    def next_cat(self, cprev, blk, rng, chunk: int = 4096) -> np.ndarray:
        out = np.empty(len(cprev), np.int64)
        for s in range(0, len(cprev), chunk):
            out[s:s + chunk] = _draw_rows(self.T[cprev[s:s + chunk]] * self.tau[blk[s:s + chunk]], rng)
        return out

    def venue_global(self, cats, rng) -> np.ndarray:
        out = np.empty(len(cats), np.int64)
        for cc in np.unique(cats):
            m = np.flatnonzero(cats == cc)
            if cc not in self._cdf:
                idx = np.flatnonzero(self.c.cat == cc)
                self._cdf[cc] = (idx, np.cumsum(self.venue_w[idx]))
            idx, cdf = self._cdf[cc]
            out[m] = idx[np.minimum(np.searchsorted(cdf, rng.random(len(m)) * cdf[-1], side="right"), len(idx) - 1)]
        return out

    def venue_near(self, cats, points, rng) -> np.ndarray:
        out = np.empty(len(cats), np.int64)
        for cc in np.unique(cats):
            m = np.flatnonzero(cats == cc)
            cand = self.c.nearest(points[m], cat=int(cc), k=self.k)
            cand = np.asarray(cand).reshape(len(m), -1)
            d = np.linalg.norm(self.c.xy[cand] - points[m][:, None, :], axis=2)
            out[m] = cand[np.arange(len(m)), _draw_rows(self.venue_w[cand] * np.exp(-d / self.scale), rng)]
        return out


@dataclass
class VariationConfig:
    p_replace: float = 0.6
    radius_km: float = 3.0
    k_neigh: int = 32
    time_shift_min: float = 180.0
    gap_logstd: float = 0.6
    p_insert: float = 0.25
    p_delete: float = 0.25
    insert_gap_min: float = 60.0
    insert_k: int = 10
    p_short_gap: float = 0.3
    short_gap_max: float = 10.0
    p_same_cat: float = 0.3
    same_cat_radius_km: float = 2.0
    p_snap: float = 0.3
    snap_radius_km: float = 1.5
    p_successor: float = 0.4
    p_return: float = 0.0
    max_len: int = 20
    max_gap_min: int = 480


class RuleVariation:

    def __init__(self, catalog, cfg: VariationConfig = VariationConfig()):
        self.c, self.cfg = catalog, cfg
        self.venue_w = None
        self.popular = None
        self.trans = None
        self.sem, self.sem_var = None, 0.0
        n, k = len(catalog), cfg.k_neigh
        self.neigh = np.tile(np.arange(n)[:, None], (1, k)).astype(np.int64)
        self.neigh_d = np.full((n, k), np.inf, np.float32)
        for r in range(catalog.n_roots):
            idx = np.flatnonzero(catalog.root == r)
            if len(idx) == 0:
                continue
            kk = min(k, len(idx))
            d, j = cKDTree(catalog.xy[idx]).query(catalog.xy[idx], k=kk)
            d, j = d.reshape(len(idx), kk), j.reshape(len(idx), kk)
            self.neigh[idx, :kk] = idx[j]
            self.neigh_d[idx, :kk] = d

    def set_venue_weights(self, w):
        self.venue_w = w

    def set_popular(self, popular):
        self.popular = PopularIndex(self.c, popular, self.cfg.snap_radius_km) if popular is not None else None

    def set_transitions(self, transitions):
        self.trans = TransitionModel(self.c, transitions) if transitions is not None else None
        if self.trans is not None:
            self.trans.venue_w = self.venue_w

    def _pick(self, cand, rng):
        if self.venue_w is None:
            return cand[rng.integers(len(cand))]
        p = self.venue_w[cand]
        return cand[rng.choice(len(cand), p=p / p.sum())]

    def vary_one(self, poi, t, d, rng):
        cfg, c = self.cfg, self.c
        poi, t = poi.astype(np.int64).copy(), t.astype(np.float64).copy()
        L = len(poi)
        for i in np.flatnonzero(rng.random(L) < cfg.p_replace * d):
            cand, dist = self.neigh[poi[i]], self.neigh_d[poi[i]]
            ok = cand[(dist <= cfg.radius_km * d) & (cand != poi[i])]
            if len(ok):
                poi[i] = self._pick(ok, rng)
        if self.trans is not None and L >= 2 and rng.random() < cfg.p_successor * d:
            i = int(rng.integers(1, L))
            if self.sem is not None and rng.random() < self.sem_var:
                nxt = self.trans.venue_successor(int(poi[i - 1]), rng)
                if nxt < 0:
                    cn = self.sem.next_cat(np.array([c.cat[poi[i - 1]]], np.int64),
                                           self.sem.blocks(np.array([t[i]])), rng)
                    nxt = int(self.sem.venue_near(cn, c.xy[poi[i]][None, :], rng)[0])
            else:
                nxt = self.trans.successor(int(poi[i - 1]), rng)
            if nxt >= 0:
                poi[i] = nxt
        if self.popular is not None:
            snap = np.flatnonzero(rng.random(L) < cfg.p_snap * d)
            if len(snap):
                new = self.popular.draw(c.xy[poi[snap]], rng)
                poi[snap] = np.where(new >= 0, new, poi[snap])
        if cfg.p_return > 0 and L >= 3 and rng.random() < cfg.p_return * d:
            i = int(rng.integers(2, L))
            poi[i] = poi[int(rng.integers(0, i - 1))]
        if L > 1:
            gaps = np.clip(np.diff(t) * np.exp(rng.normal(0, cfg.gap_logstd * d, L - 1)), 1, cfg.max_gap_min)
            if rng.random() < cfg.p_short_gap * d:
                gaps[rng.integers(L - 1)] = rng.uniform(1, cfg.short_gap_max)
            t = t[0] + np.concatenate([[0.0], np.cumsum(gaps)])
        t = t + rng.normal(0, cfg.time_shift_min * d)
        if rng.random() < cfg.p_same_cat * d:
            poi, t, L = self._same_category_hop(poi, t, L, rng)
        if L > 2 and rng.random() < cfg.p_delete * d:
            j = rng.integers(L)
            poi, t, L = np.delete(poi, j), np.delete(t, j), L - 1
        if L < cfg.max_len and rng.random() < cfg.p_insert * d:
            j = rng.integers(L + 1)
            gap = 5 + rng.exponential(cfg.insert_gap_min)
            if j == 0:
                anchor, tt = c.xy[poi[0]], t[0] - gap
            elif j == L:
                anchor, tt = c.xy[poi[-1]], t[-1] + gap
            else:
                anchor, tt = 0.5 * (c.xy[poi[j - 1]] + c.xy[poi[j]]), 0.5 * (t[j - 1] + t[j])
            p = c.clip_xy(anchor + rng.normal(0, max(cfg.radius_km * d, 0.2), 2))
            k = cfg.insert_k if self.venue_w is not None else 1
            new = self._pick(np.atleast_1d(c.tree.query(p, k=k)[1]), rng)
            poi, t = np.insert(poi, j, new), np.insert(t, j, tt)
        return poi.astype(np.int32), normalize_times(t)

    def _same_category_hop(self, poi, t, L, rng):
        c, cfg = self.c, self.cfg
        if L >= 2 and rng.random() < 0.5:
            i = int(rng.integers(1, L))
            cand = c.nearest(c.xy[poi[i]], cat=int(c.cat[poi[i - 1]]), k=10)
            if cand is not None:
                cand = np.atleast_1d(cand)
                cand = cand[(c.dist_km(cand, poi[i]) <= cfg.same_cat_radius_km) & (cand != poi[i - 1])]
                if len(cand):
                    poi[i] = self._pick(cand, rng)
            return poi, t, L
        if L < cfg.max_len:
            i = int(rng.integers(0, L))
            cand = c.nearest(c.xy[poi[i]], cat=int(c.cat[poi[i]]), k=10)
            if cand is not None:
                cand = np.atleast_1d(cand)
                cand = cand[(c.dist_km(cand, poi[i]) <= cfg.same_cat_radius_km) & (cand != poi[i])]
                if len(cand):
                    poi = np.insert(poi, i + 1, self._pick(cand, rng))
                    t = np.insert(t, i + 1, t[i] + rng.uniform(2, 30))
                    L += 1
        return poi, t, L

    def __call__(self, ts: TrailSet, degree: float, rng) -> TrailSet:
        return TrailSet.from_lists([self.vary_one(p, t, degree, rng) for p, t in ts.to_lists()])

    def semantic_refine(self, ts: TrailSet, sem, p: float, radius_km: float, rng, scale_km: float = 0.5,
                        mode: str = "snap", rho: np.ndarray | None = None) -> TrailSet:
        c = self.c
        w = self.venue_w if self.venue_w is not None else np.ones(len(c))
        poi = ts.poi.astype(np.int64).copy()
        blk = sem.blocks(ts.tmin)
        cat = np.asarray(c.cat, np.int64)
        off = ts.offsets
        draw = rng.random(len(poi)) < p
        u = rng.random(len(poi))
        for tr in range(len(ts)):
            a, b = int(off[tr]), int(off[tr + 1])
            for i in range(a, b):
                if not draw[i]:
                    continue
                cand, d = self.neigh[poi[i]], self.neigh_d[poi[i]]
                ok = d <= radius_km
                if i > a:
                    ok &= cand != poi[i - 1]
                if i + 1 < b:
                    ok &= cand != poi[i + 1]
                cand, d = cand[ok], d[ok].astype(np.float64)
                if len(cand) < 2:
                    continue
                cc = cat[cand]
                m = w[cand] * np.exp(-d / scale_km)
                if mode == "gibbs":
                    _, inv = np.unique(cc, return_inverse=True)
                    m = m / np.bincount(inv, weights=m)[inv]
                m = m * (sem.start_p[blk[i], cc] if i == a else sem.T[cat[poi[i - 1]], cc] * sem.tau[blk[i], cc])
                if i + 1 < b:
                    m = m * sem.T[cc, cat[poi[i + 1]]]
                if rho is not None:
                    m = m * rho[cc]
                tot = m.sum()
                if tot > 0:
                    poi[i] = cand[min(int(np.searchsorted(np.cumsum(m), u[i] * tot, side="right")), len(cand) - 1)]
        return TrailSet(poi.astype(np.int32), ts.tmin.copy(), ts.offsets.copy())

    def diversify(self, ts: TrailSet, p: float, radius_km: float, rng, k: int = 20) -> TrailSet:
        c = self.c
        poi = ts.poi.astype(np.int64).copy()
        orig = poi.copy()
        start = np.zeros(len(poi), bool)
        start[ts.offsets[:-1]] = True
        end = np.zeros(len(poi), bool)
        end[ts.offsets[1:] - 1] = True
        for i in np.flatnonzero(rng.random(len(poi)) < p):
            cand = c.nearest(c.xy[poi[i]], cat=int(c.cat[poi[i]]), k=k)
            if cand is None:
                continue
            cand = np.atleast_1d(cand)
            cand = cand[c.dist_km(cand, poi[i]) <= radius_km]
            for j in ([] if start[i] else [i - 1]) + ([] if end[i] else [i + 1]):
                if orig[j] != orig[i]:
                    cand = cand[(cand != poi[j]) & (cand != orig[j])]
            if len(cand) > 1:
                w = self.venue_w[cand] if self.venue_w is not None else np.ones(len(cand))
                poi[i] = cand[rng.choice(len(cand), p=w / w.sum())]
        return TrailSet(poi.astype(np.int32), ts.tmin.copy(), ts.offsets.copy())


class SimulatorRandom:

    def __init__(self, catalog, p_len: float = 0.5, hop_km: float = 2.0, gap_mean_min: float = 90.0,
                 max_len: int = 20, max_gap_min: int = 480, k_next: int = 10):
        self.c, self.p_len, self.hop_km, self.gap = catalog, p_len, hop_km, gap_mean_min
        self.max_len, self.max_gap, self.k_next = max_len, max_gap_min, k_next
        self.venue_w = None
        self.start_p = None
        self.next_w = None
        self.popular = None
        self.q_popular = 0.5
        self.trans = None
        self.q_trans = 0.5
        self.q_return = 0.0
        self.sem = None
        self.sem_frac = 0.0
        self.hop_law = None

    def set_semantic(self, sem, frac: float):
        self.sem, self.sem_frac = sem, float(frac)

    def set_hop_law(self, edges, probs, cap_km: float):
        p = np.maximum(np.asarray(probs, np.float64), 0.0)
        self.hop_law = (np.minimum(np.asarray(edges, np.float64), cap_km), p / p.sum()) if p.sum() > 0 else None

    def _hop(self, n, rng):
        if self.hop_law is None:
            return rng.exponential(self.hop_km, n)
        e, p = self.hop_law
        b = rng.choice(len(p), n, p=p)
        return e[b] + rng.random(n) * (e[b + 1] - e[b])

    def set_transitions(self, transitions):
        self.trans = TransitionModel(self.c, transitions) if transitions is not None else None
        if self.trans is not None:
            self.trans.venue_w = self.next_w

    def set_popular(self, popular, radius_km: float = 1.5):
        self.popular = PopularIndex(self.c, popular, radius_km) if popular is not None else None

    def set_marginals(self, venue_w, start_p, next_w=None):
        self.venue_w, self.start_p, self.next_w = venue_w, start_p, next_w

    def sample_starts(self, n, rng):
        c = self.c
        start = (rng.choice(len(c), n, p=self.venue_w / self.venue_w.sum()) if self.venue_w is not None
                 else rng.integers(0, len(c), n))
        if self.start_p is not None:
            b = rng.choice(168, n, p=self.start_p)
            t0 = (b // 24) * 1440 + (b % 24) * 60 + rng.integers(0, 60, n)
        else:
            t0 = rng.integers(0, WEEK_MIN, n)
        return start, t0

    def _call_semantic(self, n: int, rng) -> TrailSet:
        c, sem = self.c, self.sem
        L = np.minimum(1 + rng.geometric(self.p_len, n), self.max_len)
        poi = np.zeros((n, self.max_len), np.int64)
        t = np.zeros((n, self.max_len), np.float64)
        poi[:, 0], t[:, 0] = self.sample_starts(n, rng)
        semantic = rng.random(n) < self.sem_frac
        s0 = np.flatnonzero(semantic)
        if len(s0):
            poi[s0, 0] = sem.venue_global(sem.start_cat(sem.blocks(t[s0, 0]), rng), rng)
        for k in range(1, int(L.max())):
            act = np.flatnonzero(L > k)
            t[act, k] = t[act, k - 1] + np.minimum(5 + rng.exponential(self.gap, len(act)), self.max_gap)
            ang = rng.uniform(0, 2 * np.pi, len(act))
            dist = self._hop(len(act), rng)
            p = c.clip_xy(c.xy[poi[act, k - 1]] + dist[:, None] * np.stack([np.cos(ang), np.sin(ang)], 1))
            sm = semantic[act]
            plain, sa = act[~sm], np.flatnonzero(sm)
            pp = p[~sm]
            if len(plain):
                if self.next_w is None:
                    poi[plain, k] = c.tree.query(pp)[1]
                else:
                    poi[plain, k] = weighted_rows(c.tree.query(pp, k=self.k_next)[1], self.next_w, rng)
                if self.popular is not None:
                    use = np.flatnonzero(rng.random(len(plain)) < self.q_popular)
                    new = self.popular.draw(pp[use], rng)
                    ok = new >= 0
                    poi[plain[use[ok]], k] = new[ok]
                if self.trans is not None:
                    for a in plain[rng.random(len(plain)) < self.q_trans]:
                        nxt = self.trans.successor(int(poi[a, k - 1]), rng)
                        if nxt >= 0:
                            poi[a, k] = nxt
            if len(sa):
                rows = act[sa]
                vs = np.full(len(rows), -1, np.int64)
                if self.trans is not None:
                    for j in np.flatnonzero(rng.random(len(rows)) < self.q_trans):
                        vs[j] = self.trans.venue_successor(int(poi[rows[j], k - 1]), rng)
                need = np.flatnonzero(vs < 0)
                if len(need):
                    cn = sem.next_cat(c.cat[poi[rows[need], k - 1]].astype(np.int64),
                                      sem.blocks(t[rows[need], k]), rng)
                    vs[need] = sem.venue_near(cn, p[sa[need]], rng)
                poi[rows, k] = vs
            if self.q_return > 0 and k >= 2:
                back = act[rng.random(len(act)) < self.q_return]
                poi[back, k] = poi[back, rng.integers(0, k - 1, len(back))]
        return TrailSet.from_lists([(poi[i, :L[i]], normalize_times(t[i, :L[i]])) for i in range(n)])

    def __call__(self, n: int, rng) -> TrailSet:
        c = self.c
        if n == 0:
            return TrailSet.from_lists([])
        if self.sem is not None and self.sem_frac > 0:
            return self._call_semantic(n, rng)
        L = np.minimum(1 + rng.geometric(self.p_len, n), self.max_len)
        poi = np.zeros((n, self.max_len), np.int64)
        t = np.zeros((n, self.max_len), np.float64)
        poi[:, 0], t[:, 0] = self.sample_starts(n, rng)
        for k in range(1, int(L.max())):
            act = np.flatnonzero(L > k)
            ang = rng.uniform(0, 2 * np.pi, len(act))
            dist = self._hop(len(act), rng)
            p = c.clip_xy(c.xy[poi[act, k - 1]] + dist[:, None] * np.stack([np.cos(ang), np.sin(ang)], 1))
            if self.next_w is None:
                poi[act, k] = c.tree.query(p)[1]
            else:
                poi[act, k] = weighted_rows(c.tree.query(p, k=self.k_next)[1], self.next_w, rng)
            if self.popular is not None:
                use = np.flatnonzero(rng.random(len(act)) < self.q_popular)
                new = self.popular.draw(p[use], rng)
                ok = new >= 0
                poi[act[use[ok]], k] = new[ok]
            if self.trans is not None:
                for a in act[rng.random(len(act)) < self.q_trans]:
                    nxt = self.trans.successor(int(poi[a, k - 1]), rng)
                    if nxt >= 0:
                        poi[a, k] = nxt
            if self.q_return > 0 and k >= 2:
                back = act[rng.random(len(act)) < self.q_return]
                poi[back, k] = poi[back, rng.integers(0, k - 1, len(back))]
            t[act, k] = t[act, k - 1] + np.minimum(5 + rng.exponential(self.gap, len(act)), self.max_gap)
        return TrailSet.from_lists([(poi[i, :L[i]], normalize_times(t[i, :L[i]])) for i in range(n)])


class SimAPI:
    def __init__(self, catalog, var_cfg: VariationConfig = VariationConfig()):
        self.random = SimulatorRandom(catalog)
        self.variation = RuleVariation(catalog, var_cfg)

    def diversify(self, ts, p, radius_km, rng):
        return self.variation.diversify(ts, p, radius_km, rng)

    def set_semantic(self, sem, frac: float, var: float = 0.0):
        self.random.set_semantic(sem, frac)
        self.variation.sem, self.variation.sem_var = (sem, float(var)) if var > 0 else (None, 0.0)
        self.sem = sem

    def semantic_refine(self, ts, p, radius_km, rng, mode="snap", rho=None):
        return self.variation.semantic_refine(ts, self.sem, p, radius_km, rng, mode=mode, rho=rho)

    def set_hop_law(self, edges, probs, cap_km: float):
        self.random.set_hop_law(edges, probs, cap_km)

    def set_marginals(self, venue_w, start_p, cat_share=None, use=("prompt", "ground", "var"), popular=None,
                      transitions=None):
        self.random.set_marginals(venue_w if "prompt" in use else None, start_p if "prompt" in use else None,
                                  venue_w if "ground" in use else None)
        self.random.set_popular(popular if "ground" in use else None)
        self.random.set_transitions(transitions if "ground" in use else None)
        self.variation.set_venue_weights(venue_w if "var" in use else None)
        self.variation.set_popular(popular if "var" in use else None)
        self.variation.set_transitions(transitions if "var" in use else None)
