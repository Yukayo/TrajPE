from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
from scipy import sparse

from .trails import TrailSet

PROTOCOL_PATH = Path(__file__).resolve().parents[1] / "configs" / "benchmark_protocol_v1.json"
WEEK_MIN = 7 * 24 * 60
DAY_MIN = 24 * 60
LN2 = math.log(2.0)


def load_protocol(path=None) -> dict:
    return json.loads(Path(path or PROTOCOL_PATH).read_text())


def jsd_skip0(p, q) -> float:
    p, q = np.asarray(p, np.float64), np.asarray(q, np.float64)
    m = 0.5 * (p + q)
    a, b = p > 0, q > 0
    return float(0.5 * np.sum(p[a] * np.log(p[a] / m[a])) + 0.5 * np.sum(q[b] * np.log(q[b] / m[b])))


def jsd_eps(p, q, eps: float = 1e-8) -> float:
    p, q = np.asarray(p, np.float64), np.asarray(q, np.float64)
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(np.log((p + eps) / (m + eps)) * p) + 0.5 * np.sum(np.log((q + eps) / (m + eps)) * q))


def jsd_eps_rows(P, Q, eps: float = 1e-8) -> np.ndarray:
    M = 0.5 * (P + Q)
    return 0.5 * np.sum(np.log((P + eps) / (M + eps)) * P, 1) + 0.5 * np.sum(np.log((Q + eps) / (M + eps)) * Q, 1)


def jsd_nats(p, q) -> float:
    p, q = np.asarray(p, np.float64), np.asarray(q, np.float64)
    sp, sq = p.sum(), q.sum()
    if sp <= 0 or sq <= 0:
        return 0.0 if sp <= 0 and sq <= 0 else LN2
    p, q = p / sp, q / sq
    m = 0.5 * (p + q)
    a, b = p > 0, q > 0
    return float(0.5 * np.sum(p[a] * np.log(p[a] / m[a])) + 0.5 * np.sum(q[b] * np.log(q[b] / m[b])))


def sparse_jsd_nats(keys_a, keys_b) -> float:
    ua, ca = np.unique(keys_a, return_counts=True)
    ub, cb = np.unique(keys_b, return_counts=True)
    allk = np.union1d(ua, ub)
    pa, pb = np.zeros(len(allk)), np.zeros(len(allk))
    pa[np.searchsorted(allk, ua)] = ca
    pb[np.searchsorted(allk, ub)] = cb
    return jsd_nats(pa, pb)


def bernoulli_jsd(p, q) -> np.ndarray:
    p, q = np.asarray(p, np.float64), np.asarray(q, np.float64)
    m = 0.5 * (p + q)

    def term(x, y):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(x > 0, x * np.log(x / np.where(y > 0, y, 1.0)), 0.0)
    return 0.5 * (term(p, m) + term(1 - p, 1 - m)) + 0.5 * (term(q, m) + term(1 - q, 1 - m))


def _cells(xy, W, H, g):
    ix = np.clip(np.floor(xy[:, 0] / W * g).astype(np.int64), 0, g - 1)
    iy = np.clip(np.floor(xy[:, 1] / H * g).astype(np.int64), 0, g - 1)
    return ix, iy


def grid_trajectories(ix, iy, off, g, mode: str):
    n = len(off) - 1
    keep = np.ones(len(ix), bool)
    if len(ix) > 1:
        same = (ix[1:] == ix[:-1]) & (iy[1:] == iy[:-1])
        keep[1:] = ~same
    keep[off[:-1]] = True
    kidx = np.flatnonzero(keep)
    klen = np.add.reduceat(keep.astype(np.int64), off[:-1]) if n else np.zeros(0, np.int64)
    dix, diy = ix[kidx], iy[kidx]
    if mode == "ada":
        single = np.flatnonzero(klen == 1)
        if len(single):
            doff0 = np.concatenate([[0], np.cumsum(klen)])
            rep = np.ones(len(dix), np.int64)
            rep[doff0[single]] = 2
            dix, diy = np.repeat(dix, rep), np.repeat(diy, rep)
            klen = klen.copy()
            klen[single] = 2
    doff = np.concatenate([[0], np.cumsum(klen)]).astype(np.int64)
    if mode == "dedup":
        return dix * g + diy, doff
    m = len(dix)
    has_next = np.ones(m, bool)
    has_next[doff[1:] - 1] = False
    pi = np.flatnonzero(has_next)
    dx, dy = dix[pi + 1] - dix[pi], diy[pi + 1] - diy[pi]
    emit = np.ones(m, np.int64)
    emit[pi] = np.maximum(np.maximum(np.abs(dx), np.abs(dy)), 1)
    sgx, sgy, adx, ady = (np.zeros(m, np.int64) for _ in range(4))
    sgx[pi], sgy[pi], adx[pi], ady[pi] = np.sign(dx), np.sign(dy), np.abs(dx), np.abs(dy)
    src = np.repeat(np.arange(m), emit)
    s = np.arange(int(emit.sum())) - np.repeat(np.cumsum(emit) - emit, emit)
    gx = dix[src] + sgx[src] * np.minimum(s, adx[src])
    gy = diy[src] + sgy[src] * np.minimum(s, ady[src])
    tl = np.add.reduceat(emit, doff[:-1]) if n else np.zeros(0, np.int64)
    return gx * g + gy, np.concatenate([[0], np.cumsum(tl)]).astype(np.int64)


def ngrams(cell, off, sizes, base: int, ts=None, no_repeat_above2: bool = False):
    assert float(base) ** max(sizes) < 2.0 ** 62, "n-gram keys would overflow int64"
    L = np.diff(off)
    tr = np.repeat(np.arange(len(L)), L)
    pos = np.arange(len(cell)) - np.repeat(off[:-1], L)
    room = L[tr] - pos
    keys, trails, t0, t1 = [], [], [], []
    cell = cell.astype(np.int64)
    for n in sizes:
        ok = np.flatnonzero(room >= n)
        if not len(ok):
            continue
        k = np.zeros(len(ok), np.int64)
        for j in range(n):
            k = k * base + (cell[ok + j] + 1)
        if no_repeat_above2 and n > 2:
            rep = np.zeros(len(ok), bool)
            for j in range(n - 1):
                rep |= cell[ok + j] == cell[ok + j + 1]
            ok, k = ok[~rep], k[~rep]
        keys.append(k)
        trails.append(tr[ok])
        if ts is not None:
            t0.append(ts[ok])
            t1.append(ts[ok + n - 1])
    cat = (lambda a: np.concatenate(a) if a else np.zeros(0, np.int64))
    return cat(keys), cat(trails), (cat(t0) if ts is not None else None), (cat(t1) if ts is not None else None)


def count_keys(keys):
    return np.unique(keys, return_counts=True)


def top_keys(u, c, k: int):
    order = np.lexsort((u, -c))[:k]
    return u[order], c[order]


def lookup(u_ref, u, c) -> np.ndarray:
    out = np.zeros(len(u_ref), np.float64)
    if len(u) == 0 or len(u_ref) == 0:
        return out
    pos = np.searchsorted(u, u_ref)
    ok = pos < len(u)
    ok[ok] &= u[pos[ok]] == u_ref[ok]
    out[ok] = c[pos[ok]]
    return out


def kendall_ldp(a, s) -> float:
    a, s = np.asarray(a, np.float64), np.asarray(s, np.float64)
    n = len(a)
    iu = np.triu_indices(n, 1)
    A = (a[:, None] - a[None, :])[iu]
    S = (s[:, None] - s[None, :])[iu]
    C = np.sum((A > 0) & (S > 0)) + np.sum((A < 0) & (S < 0))
    D = np.sum((A > 0) & (S <= 0)) + np.sum((A < 0) & (S >= 0))
    return float((C - D) / (n * (n - 1) / 2.0))


def kendall_ada(a, s) -> float:
    a, s = np.asarray(a, np.float64), np.asarray(s, np.float64)
    n = len(a)
    iu = np.triu_indices(n, 1)
    A = (a[:, None] - a[None, :])[iu]
    S = (s[:, None] - s[None, :])[iu]
    C = np.sum((A > 0) & (S > 0)) + np.sum((A < 0) & (S <= 0))
    D = np.sum((A > 0) & (S <= 0)) + np.sum((A < 0) & (S > 0))
    return float((C - D) / (n * (n - 1) / 2.0))


def hotspot_ndcg(orig_counts, syn_counts, k: int) -> float:
    o = np.asarray(orig_counts, np.float64)
    s = np.asarray(syn_counts, np.float64)
    if o.sum() <= 0:
        return float("nan")
    if s.sum() <= 0:
        return 0.0
    otop = np.argsort(-o / o.sum(), kind="stable")[:k]
    stop = np.argsort(-s / s.sum(), kind="stable")[:k]
    rank = {int(c): i for i, c in enumerate(otop)}
    r = np.array([1.0 / (rank[int(c)] + 1) if int(c) in rank else 0.0 for c in stop] + [0.0] * (k - len(stop)))
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    idcg = np.sum((1.0 / np.arange(1, k + 1)) * disc)
    return float(np.sum(r * disc) / idcg) if idcg else 0.0


def are(real, syn_scaled, bound) -> float:
    real = np.asarray(real, np.float64)
    return float(np.mean(np.abs(real - np.asarray(syn_scaled, np.float64)) / np.maximum(real, bound)))


def bucket_counts_ada(v, bs: float, n: int) -> np.ndarray:
    return np.array([np.sum((v >= i * bs) & (v < (i + 1) * bs)) for i in range(n)], np.float64)


def bucket_counts_ldp(v, bs: float, n: int) -> np.ndarray:
    out = np.zeros(n)
    for i in range(n):
        start = i * bs
        end = start + bs
        out[i] = np.sum((v >= start) & (v <= end))
    return out


def movesim_hist(arr, lo: float, hi: float, bins: int) -> np.ndarray:
    return np.histogram(arr, np.arange(lo, hi, float(hi - lo) / bins))[0].astype(np.float64)


def llmob_hist(arr, lo: float, hi: float, bins: int) -> np.ndarray:
    arr = np.asarray(arr, np.float64)
    h = np.histogram(arr[arr <= hi], bins=bins, range=(lo, hi))[0]
    return np.append(h, np.sum(arr > hi)).astype(np.float64)


def quantized_ids(keys_r, keys_s, bins: int):
    u = np.union1d(np.unique(keys_r), np.unique(keys_s))
    span = max(len(u) - 1, 1)
    xr = np.searchsorted(u, keys_r) / span
    xs = np.searchsorted(u, keys_s) / span
    return llmob_hist(xr, 0.0, 1.0, bins), llmob_hist(xs, 0.0, 1.0, bins)


class Prep:

    def __init__(self, ts: TrailSet, catalog):
        self.ts = ts
        self.cat = catalog
        self.W, self.H = float(catalog.width_km), float(catalog.height_km)
        self.n = len(ts)
        self.L = ts.lengths.astype(np.int64)
        self.off = ts.offsets.astype(np.int64)
        self.tr = np.repeat(np.arange(self.n), self.L)
        self.first, self.last = self.off[:-1], self.off[1:] - 1
        self.poi = ts.poi.astype(np.int64)
        self.t = ts.tmin.astype(np.int64)
        self.xy = np.asarray(catalog.xy, np.float64)[self.poi]
        has = np.ones(len(self.poi), bool)
        has[self.last] = False
        self.nxt = np.flatnonzero(has)
        self.hop_off = np.concatenate([[0], np.cumsum(self.L - 1)]).astype(np.int64)
        self._cache: dict = {}

    def cached(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    @property
    def n_checkins(self) -> int:
        return len(self.poi)

    @property
    def hop(self) -> np.ndarray:
        return self.cached("hop", lambda: np.linalg.norm(self.xy[self.nxt + 1] - self.xy[self.nxt], axis=1))

    @property
    def travel(self) -> np.ndarray:
        return self.cached("travel", lambda: np.bincount(self.tr[self.nxt], weights=self.hop, minlength=self.n))

    @property
    def diameter(self) -> np.ndarray:
        def f():
            out = np.zeros(self.n)
            for Lv in np.unique(self.L):
                idx = np.flatnonzero(self.L == Lv)
                step = max(1, int(4e6 // (Lv * Lv)))
                for i in range(0, len(idx), step):
                    j = idx[i:i + step]
                    p = self.xy[self.off[j][:, None] + np.arange(Lv)[None, :]]
                    d2 = ((p[:, :, None, :] - p[:, None, :, :]) ** 2).sum(-1)
                    out[j] = np.sqrt(d2.reshape(len(j), -1).max(1))
            return out
        return self.cached("diameter", f)

    @property
    def gyr2(self) -> np.ndarray:
        def f():
            cent = np.add.reduceat(self.xy, self.first, axis=0) / self.L[:, None]
            return np.add.reduceat(((self.xy - cent[self.tr]) ** 2).sum(1), self.first) / self.L
        return self.cached("gyr2", f)

    def cells(self, g):
        return self.cached(("cells", g), lambda: _cells(self.xy, self.W, self.H, g))

    def cell(self, g):
        def f():
            ix, iy = self.cells(g)
            return ix * g + iy
        return self.cached(("cell", g), f)

    def gridtraj(self, g, mode):
        return self.cached(("gt", g, mode), lambda: grid_trajectories(*self.cells(g), self.off, g, mode))

    def slots(self, minutes: int, n_slots: int):
        return self.cached(("slots", minutes), lambda: np.minimum((self.t % WEEK_MIN) // minutes, n_slots - 1))

    def coverage(self, g, mode):
        def f():
            c, o = (self.gridtraj(g, mode) if mode else (self.cell(g), self.off))
            tr = np.repeat(np.arange(len(o) - 1), np.diff(o))
            pairs = np.unique(tr * (g * g) + c)
            return np.bincount(pairs % (g * g), minlength=g * g).astype(np.float64)
        return self.cached(("cov", g, mode), f)


def _rng(seed, stream):
    return np.random.default_rng([int(seed), int(stream)])


def trails_in_circles(prep: Prep, centers, radii, chunk_elems: int = 4_000_000) -> np.ndarray:
    P = prep.xy[prep.nxt]
    D = prep.xy[prep.nxt + 1] - P
    dd = (D ** 2).sum(1)
    safe = np.where(dd > 0, dd, 1.0)
    out = np.zeros(len(centers))
    if len(P) == 0:
        return out
    m = max(1, chunk_elems // len(P))
    for i in range(0, len(centers), m):
        C = centers[i:i + m]
        wx = C[:, 0:1] - P[None, :, 0]
        wy = C[:, 1:2] - P[None, :, 1]
        u = np.clip((wx * D[None, :, 0] + wy * D[None, :, 1]) / safe[None, :], 0.0, 1.0)
        u[:, dd == 0] = 0.0
        d2 = (wx - u * D[None, :, 0]) ** 2 + (wy - u * D[None, :, 1]) ** 2
        mn = np.minimum.reduceat(d2, prep.hop_off[:-1], axis=1)
        out[i:i + m] = np.sum(mn <= (radii[i:i + m, None] ** 2), axis=1)
    return out


def points_in_boxes(xy, boxes, mask=None, chunk_elems: int = 8_000_000) -> np.ndarray:
    out = np.zeros(len(boxes))
    if len(xy) == 0:
        return out
    m = max(1, chunk_elems // len(xy))
    for i in range(0, len(boxes), m):
        B = boxes[i:i + m]
        inside = ((xy[None, :, 0] >= B[:, 0:1]) & (xy[None, :, 0] <= B[:, 1:2])
                  & (xy[None, :, 1] >= B[:, 2:3]) & (xy[None, :, 1] <= B[:, 3:4]))
        if mask is not None:
            inside &= mask(i, i + len(B))
        out[i:i + m] = inside.sum(1)
    return out


METRICS: dict = {}


def metric(name):
    def deco(fn):
        METRICS[name] = fn
        return fn
    return deco


def _scale(R: Prep, S: Prep) -> float:
    return R.n / max(S.n, 1)


@metric("ada_query_avre")
def _ada_query(B, R, S, p):
    q = B.fixed("ada_query", lambda: B.circles(p, constant_radius=True))
    ans_r = R.cached(("circ", "ada_query", id(B)), lambda: trails_in_circles(R, *q))
    ans_s = trails_in_circles(S, *q)
    return are(ans_r, _scale(R, S) * ans_s, p["sanity_bound_frac_of_trails"] * R.n)


@metric("ada_kendall_tau")
def _ada_kendall(B, R, S, p):
    return kendall_ada(R.coverage(p["grid"], "ada"), S.coverage(p["grid"], "ada"))


def _ada_patterns(P: Prep, p):
    g = p["grid"]
    key = ("ada_fp", g, p["min_size"], p["max_size"])

    def f():
        c, o = P.gridtraj(g, "ada")
        k = ngrams(c, o, range(p["min_size"], p["max_size"] + 1), g * g + 1)[0]
        return count_keys(k)
    return P.cached(key, f)


@metric("ada_fp_f1")
def _ada_fp_f1(B, R, S, p):
    ur, cr = _ada_patterns(R, p)
    us, cs = _ada_patterns(S, p)
    o = set(top_keys(ur, cr, p["top_k"])[0].tolist())
    s = set(top_keys(us, cs, p["top_k"])[0].tolist())
    hits = len(o & s)
    if hits == 0:
        return 0.0
    prec, rec = hits / len(s), hits / len(o)
    return float(2 * prec * rec / (prec + rec))


@metric("ada_fp_support_are")
def _ada_fp_support(B, R, S, p):
    ur, cr = _ada_patterns(R, p)
    us, cs = _ada_patterns(S, p)
    tk, tc = top_keys(ur, cr, p["top_k"])
    if not len(tk):
        return float("nan")
    syn = _scale(R, S) * lookup(tk, us, cs)
    return float(np.mean(np.abs(tc - syn) / tc))


def _trip_hist(P: Prep, g):
    c = P.cell(g)
    return np.bincount(c[P.first] * (g * g) + c[P.last], minlength=g ** 4).astype(np.float64)


@metric("ada_trip_error")
def _ada_trip(B, R, S, p):
    g = p["grid"]
    hr = R.cached(("trip", g), lambda: _trip_hist(R, g))
    hs = _trip_hist(S, g)
    return jsd_skip0(hr / max(hr.sum(), 1), hs / max(hs.sum(), 1))


def _ada_bucket_jsd(vr, vs, n):
    if len(vr) == 0:
        return float("nan")
    bs = (vr.max() - vr.min()) / n
    return jsd_skip0(bucket_counts_ada(vr, bs, n) / len(vr), bucket_counts_ada(vs, bs, n) / max(len(vs), 1))


@metric("ada_diameter_error")
def _ada_diam(B, R, S, p):
    return _ada_bucket_jsd(R.diameter, S.diameter, p["n_buckets"])


@metric("ada_length_error")
def _ada_len(B, R, S, p):
    return _ada_bucket_jsd(R.travel, S.travel, p["n_buckets"])


def _ldp_density(P: Prep, g):
    return P.cached(("ldp_dens", g), lambda: np.bincount(P.gridtraj(g, "ldp")[0], minlength=g * g).astype(np.float64))


def _norm(h):
    s = h.sum()
    return h / s if s > 0 else h


@metric("ldp_density_error")
def _ldp_dens(B, R, S, p):
    return jsd_eps(_norm(_ldp_density(R, p["grid"])), _norm(_ldp_density(S, p["grid"])))


@metric("ldp_hotspot_error")
def _ldp_hot(B, R, S, p):
    return 1.0 - hotspot_ndcg(_ldp_density(R, p["grid"]), _ldp_density(S, p["grid"]), p["k"])


@metric("ldp_query_error")
def _ldp_query(B, R, S, p):
    boxes = B.fixed("ldp_query", lambda: B.squares(p, inside=False))
    ans_r = R.cached(("box", "ldp_query", id(B)), lambda: points_in_boxes(R.xy, boxes))
    ans_s = points_in_boxes(S.xy, boxes)
    return are(ans_r, _scale(R, S) * ans_s, p["sanity_bound_frac_of_points"] * R.n_checkins)


@metric("ldp_kendall_tau")
def _ldp_kendall(B, R, S, p):
    return kendall_ldp(R.coverage(p["grid"], "ldp"), S.coverage(p["grid"], "ldp"))


@metric("ldp_trip_error")
def _ldp_trip(B, R, S, p):
    g = p["grid"]
    hr = R.cached(("trip", g), lambda: _trip_hist(R, g))
    return jsd_eps(_norm(hr), _norm(_trip_hist(S, g)))


def _ldp_bucket_jsd(vr, vs, n, lo_hi=None):
    if len(vr) == 0:
        return float("nan")
    lo, hi = lo_hi if lo_hi is not None else (vr.min(), vr.max())
    bs = (hi - lo) / n
    cr, cs = bucket_counts_ldp(vr, bs, n), bucket_counts_ldp(vs, bs, n)
    if cr.sum() <= 0 or cs.sum() <= 0:
        return LN2
    return jsd_eps(cr / cr.sum(), cs / cs.sum())


@metric("ldp_length_error")
def _ldp_len(B, R, S, p):
    return _ldp_bucket_jsd(R.travel, S.travel, p["n_buckets"])


@metric("ldp_diameter_error")
def _ldp_diam(B, R, S, p):
    return _ldp_bucket_jsd(R.diameter, S.diameter, p["n_buckets"])


def _ldp_patterns(P: Prep, p):
    g = p["grid"]

    def f():
        c, o = P.gridtraj(g, "ldp")
        return count_keys(ngrams(c, o, range(p["min_size"], p["max_size"] + 1), g * g + 1)[0])
    return P.cached(("ldp_fp", g, p["min_size"], p["max_size"]), f)


@metric("ldp_pattern_f1")
def _ldp_pf1(B, R, S, p):
    ur, cr = _ldp_patterns(R, p)
    us, cs = _ldp_patterns(S, p)
    k = p["top_k"]
    hits = len(set(top_keys(ur, cr, k)[0].tolist()) & set(top_keys(us, cs, k)[0].tolist()))
    return float(hits / k)


@metric("ldp_pattern_error")
def _ldp_perr(B, R, S, p):
    ur, cr = _ldp_patterns(R, p)
    us, cs = _ldp_patterns(S, p)
    tk, tc = top_keys(ur, cr, p["top_k"])
    syn = _scale(R, S) * lookup(tk, us, cs)
    return float(np.sum(np.abs(tc - syn) / tc) / p["top_k"])


def _retra_counts(P: Prep, p):
    g, T = p["grid"], p["n_timestamps"]

    def f():
        t = P.slots(p["timestamp_min"], T)
        return np.bincount(t * (g * g) + P.cell(g), minlength=T * g * g).reshape(T, g * g).astype(np.float64)
    return P.cached(("retra_counts", g, T, p["timestamp_min"]), f)


@metric("retra_density_error")
def _retra_dens(B, R, S, p):
    cr, cs = _retra_counts(R, p), _retra_counts(S, p)
    pr = cr / (cr.sum(1, keepdims=True) + 1e-10)
    ps = cs / (cs.sum(1, keepdims=True) + 1e-10)
    return float(np.mean(jsd_eps_rows(pr, ps)))


def _retra_trans(P: Prep, p):
    g, T = p["grid"], p["n_timestamps"]
    G2 = g * g

    def f():
        t = P.slots(p["timestamp_min"], T)[P.nxt]
        c = P.cell(g)
        ok = t < T - 1
        idx = t[ok] * (G2 * G2) + c[P.nxt][ok] * G2 + c[P.nxt + 1][ok]
        return sparse.csr_matrix((np.ones(len(idx)), (idx // (G2 * G2), idx % (G2 * G2))), shape=(T - 1, G2 * G2))
    return P.cached(("retra_trans", g, T, p["timestamp_min"]), f)


@metric("retra_transition_error")
def _retra_tr(B, R, S, p):
    A, Bm = _retra_trans(R, p), _retra_trans(S, p)
    sa = np.asarray(A.sum(1)).ravel() + 1e-10
    sb = np.asarray(Bm.sum(1)).ravel() + 1e-10
    out = np.zeros(A.shape[0])
    for t in range(A.shape[0]):
        a = A.getrow(t).toarray().ravel() / sa[t]
        b = Bm.getrow(t).toarray().ravel() / sb[t]
        if a.any() or b.any():
            out[t] = jsd_eps(a, b)
    return float(np.mean(out))


@metric("retra_query_error")
def _retra_query(B, R, S, p):
    q = B.fixed("retra_query", lambda: B.st_squares(p))
    boxes, t0 = q
    T, phi = p["n_timestamps"], p["phi"]

    def answers(P):
        ts = P.slots(p["timestamp_min"], T)
        return points_in_boxes(P.xy, boxes, mask=lambda a, b: (ts[None, :] >= t0[a:b, None])
                               & (ts[None, :] <= t0[a:b, None] + phi - 1))
    ans_r = R.cached(("box", "retra_query", id(B)), lambda: answers(R))
    upt = R.n_checkins / T
    return are(ans_r, _scale(R, S) * answers(S), p["sanity_bound_frac"] * upt * phi)


def _retra_ngrams(P: Prep, p):
    g = p["grid"]

    def f():
        ts = P.t // p["timestamp_min"]
        k, _, t0, t1 = ngrams(P.cell(g), P.off, range(p["min_size"], p["max_size"] + 1), g * g + 1, ts=ts,
                              no_repeat_above2=True)
        u, inv = np.unique(k, return_inverse=True)
        return u, inv.ravel(), t0, t1
    return P.cached(("retra_fp", g, p["min_size"], p["max_size"]), f)


@metric("retra_pattern_f1")
def _retra_pf1(B, R, S, p):
    t0w = B.fixed("retra_windows", lambda: B.windows(p))
    ur, ir, r0, r1 = _retra_ngrams(R, p)
    us, is_, s0, s1 = _retra_ngrams(S, p)
    k, phi = p["top_k"], p["phi"]
    out = []
    for a in t0w:
        b = a + phi - 1
        mr = (r0 >= a) & (r1 <= b)
        ms = (s0 >= a) & (s1 <= b)
        cr = np.bincount(ir[mr], minlength=len(ur))
        cs = np.bincount(is_[ms], minlength=len(us))
        nzr, nzs = np.flatnonzero(cr), np.flatnonzero(cs)
        o = set(top_keys(ur[nzr], cr[nzr], k)[0].tolist())
        s = set(top_keys(us[nzs], cs[nzs], k)[0].tolist())
        out.append(len(o & s) / k)
    return float(np.mean(out))


@metric("retra_kendall_tau")
def _retra_kendall(B, R, S, p):
    return kendall_ldp(R.coverage(p["grid"], None), S.coverage(p["grid"], None))


@metric("retra_length_error")
def _retra_len(B, R, S, p):
    vr, vs = R.travel, S.travel
    if len(vr) == 0 or len(vs) == 0:
        return float("nan")
    lo, hi = min(vr.min(), vs.min()), max(vr.max(), vs.max())
    bs = (hi - lo) / p["n_buckets"]
    cr, cs = bucket_counts_ldp(vr, bs, p["n_buckets"]), bucket_counts_ldp(vs, bs, p["n_buckets"])
    if cr.sum() <= 0 or cs.sum() <= 0:
        return LN2
    return jsd_eps(cr / cr.sum(), cs / cs.sum())


@metric("retra_trip_error")
def _retra_trip(B, R, S, p):
    g = p["grid"]
    hr = R.cached(("trip", g), lambda: _trip_hist(R, g))
    return jsd_eps(_norm(hr), _norm(_trip_hist(S, g)))


@metric("retra_hotspot_ndcg")
def _retra_hot(B, R, S, p):
    t0w = B.fixed("retra_windows", lambda: B.windows(p))
    cr, cs = _retra_counts(R, p), _retra_counts(S, p)
    ccr, ccs = np.cumsum(np.vstack([np.zeros(cr.shape[1]), cr]), 0), np.cumsum(np.vstack([np.zeros(cs.shape[1]), cs]), 0)
    out = []
    for a in t0w:
        b = a + p["phi"]
        wr, ws = ccr[b] - ccr[a], ccs[b] - ccs[a]
        if wr.sum() <= 0:
            continue
        out.append(hotspot_ndcg(wr, ws, p["k"]))
    return float(np.mean(out)) if out else float("nan")


def _clip_hist_jsd(vr, vs, n):
    if len(vr) == 0:
        return float("nan")
    hi = max(float(vr.max()), 1e-12)
    hr = np.histogram(np.minimum(vr, hi), bins=n, range=(0.0, hi))[0]
    hs = np.histogram(np.minimum(vs, hi), bins=n, range=(0.0, hi))[0]
    return jsd_nats(hr, hs)


@metric("priv_length_jsd")
def _priv_len(B, R, S, p):
    return _clip_hist_jsd(R.travel, S.travel, p["n_bins"])


@metric("priv_diameter_jsd")
def _priv_diam(B, R, S, p):
    return _clip_hist_jsd(R.diameter, S.diameter, p["n_bins"])


@metric("priv_density_are")
def _priv_dens(B, R, S, p):
    q = B.fixed("priv_density", lambda: B.circles(p, constant_radius=False))
    ans_r = R.cached(("circ", "priv_density", id(B)), lambda: trails_in_circles(R, *q))
    return are(ans_r, _scale(R, S) * trails_in_circles(S, *q), p["sanity_bound_frac_of_trails"] * R.n)


def _priv_patterns(P: Prep, p):
    g = p["grid"]

    def f():
        c, o = P.gridtraj(g, "dedup")
        return count_keys(ngrams(c, o, range(p["min_size"], p["max_size"] + 1), g * g + 1)[0])
    return P.cached(("priv_fp", g, p["min_size"], p["max_size"]), f)


@metric("priv_pattern_are")
def _priv_pat(B, R, S, p):
    ur, cr = _priv_patterns(R, p)
    us, cs = _priv_patterns(S, p)
    tk, tc = top_keys(ur, cr, p["top_k"])
    return are(tc, _scale(R, S) * lookup(tk, us, cs), p["sanity_bound_frac_of_trails"] * R.n)


def _hrnet_starts(R: Prep, g, n_start):
    def f():
        c0 = R.cell(g)[R.first]
        cnt = np.bincount(c0, minlength=g * g)
        nz = np.flatnonzero(cnt)
        return nz[np.lexsort((nz, -cnt[nz]))][:n_start]
    return R.cached(("hrnet_starts", g, n_start), f)


def _hrnet_cond(P: Prep, starts, g, which):
    G2 = g * g
    c = P.cell(g)
    pos = np.full(G2, -1, np.int64)
    pos[starts] = np.arange(len(starts))
    r = pos[c[P.first]]
    ok = r >= 0
    tgt = c[P.last] if which == "last" else c[P.first + 1]
    h = np.bincount(r[ok] * G2 + tgt[ok], minlength=len(starts) * G2).reshape(len(starts), G2).astype(np.float64)
    return h, np.bincount(r[ok], minlength=len(starts)).astype(np.float64)


def _hrnet_cond_jsd(B, R, S, p, which):
    g = p["grid"]
    starts = _hrnet_starts(R, g, p["n_start"])
    hr, nr = R.cached(("hrnet_cond", g, which, p["n_start"]), lambda: _hrnet_cond(R, starts, g, which))
    hs, ns = _hrnet_cond(S, starts, g, which)
    vals = [jsd_nats(hr[i], hs[i] if ns[i] > 0 else np.ones(g * g)) for i in range(len(starts))]
    return float(np.mean(vals)) if vals else float("nan")


@metric("hrnet_destination")
def _hrnet_dest(B, R, S, p):
    return _hrnet_cond_jsd(B, R, S, p, "last")


@metric("hrnet_transition")
def _hrnet_trans(B, R, S, p):
    return _hrnet_cond_jsd(B, R, S, p, "second")


def _hrnet_waypoints(P: Prep, starts, g):
    G2 = g * g
    c = P.cell(g)
    pos = np.full(G2, -1, np.int64)
    pos[starts] = np.arange(len(starts))
    start_of = c[P.first][P.tr]
    later = np.ones(len(c), bool)
    later[P.first] = False
    later &= c != start_of
    r = pos[start_of]
    ok = later & (r >= 0)
    pairs = np.unique(P.tr[ok] * G2 + c[ok])
    tr_of_pair = pairs // G2
    rr = pos[c[P.first][tr_of_pair]]
    h = np.bincount(rr * G2 + pairs % G2, minlength=len(starts) * G2).reshape(len(starts), G2).astype(np.float64)
    r0 = pos[c[P.first]]
    n = np.bincount(r0[r0 >= 0], minlength=len(starts)).astype(np.float64)
    return h, n


@metric("hrnet_waypoint")
def _hrnet_wp(B, R, S, p):
    g = p["grid"]
    starts = _hrnet_starts(R, g, p["n_start"])
    hr, nr = R.cached(("hrnet_wp", g, p["n_start"]), lambda: _hrnet_waypoints(R, starts, g))
    hs, ns = _hrnet_waypoints(S, starts, g)
    vals = []
    for i in range(len(starts)):
        pr = hr[i] / nr[i]
        qs = hs[i] / ns[i] if ns[i] > 0 else np.full(g * g, 1.0 / (g * g))
        vals.append(float(np.mean(bernoulli_jsd(pr, qs))))
    return float(np.mean(vals)) if vals else float("nan")


@metric("hrnet_travel_distance")
def _hrnet_td(B, R, S, p):
    return _clip_hist_jsd(R.travel, S.travel, p["n_bins"])


@metric("hrnet_diameter")
def _hrnet_diam(B, R, S, p):
    return _clip_hist_jsd(R.diameter, S.diameter, p["n_bins"])


def _hrnet_occupancy(P: Prep, g, step, n_times):
    G2 = g * g
    a, b = P.t[P.nxt], P.t[P.nxt + 1]
    h0 = -(-a // step)
    h1 = -(-b // step)
    n = np.maximum(h1 - h0, 0)
    src = np.repeat(np.arange(len(a)), n)
    hh = np.repeat(h0, n) + (np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n))
    cell = P.cell(g)[P.nxt][src]
    return np.bincount((hh % n_times) * G2 + cell, minlength=n_times * G2).reshape(n_times, G2).astype(np.float64)


@metric("hrnet_density_t")
def _hrnet_dt(B, R, S, p):
    g, step, nt = p["grid"], p["time_step_min"], p["n_times"]
    hr = R.cached(("hrnet_occ", g, step, nt), lambda: _hrnet_occupancy(R, g, step, nt))
    hs = _hrnet_occupancy(S, g, step, nt)
    vals = [jsd_nats(hr[t], hs[t] if hs[t].sum() > 0 else np.ones(g * g)) for t in range(nt) if hr[t].sum() > 0]
    return float(np.mean(vals)) if vals else float("nan")


def _presence(P: Prep, g):
    def f():
        G2 = g * g
        pairs = np.unique(P.tr * G2 + P.cell(g))
        return sparse.csr_matrix((np.ones(len(pairs), np.float32), (pairs // G2, pairs % G2)), shape=(P.n, G2))
    return P.cached(("presence", g), f)


@metric("hrnet_traj_density")
def _hrnet_tdens(B, R, S, p):
    g = p["grid"]
    Q = B.fixed("hrnet_sets", lambda: B.cell_sets(p))

    def answers(P):
        A = _presence(P, g)
        out = np.zeros(Q.shape[1])
        for i in range(0, Q.shape[1], 100):
            out[i:i + 100] = np.asarray(((A @ Q[:, i:i + 100]) > 0).sum(0)).ravel()
        return out
    ans_r = R.cached(("sets", "hrnet", id(B)), lambda: answers(R))
    return are(ans_r, _scale(R, S) * answers(S), p["sanity_bound_frac_of_trails"] * R.n)


def _hrnet_patterns(P: Prep, p):
    g = p["grid"]

    def f():
        k, tr, _, _ = ngrams(P.cell(g), P.off, range(p["min_size"], p["max_size"] + 1), g * g + 1)
        o = np.lexsort((k, tr))
        k, tr = k[o], tr[o]
        first = np.ones(len(k), bool)
        first[1:] = (k[1:] != k[:-1]) | (tr[1:] != tr[:-1])
        return count_keys(k[first])
    return P.cached(("hrnet_fp", g, p["min_size"], p["max_size"]), f)


@metric("hrnet_traj_pattern")
def _hrnet_tpat(B, R, S, p):
    ur, cr = _hrnet_patterns(R, p)
    us, cs = _hrnet_patterns(S, p)
    tk, tc = top_keys(ur, cr, p["top_k"])
    return are(tc, _scale(R, S) * lookup(tk, us, cs), p["sanity_bound_frac_of_trails"] * R.n)


def _ms_jsd(h1, h2):
    return jsd_nats(h1, h2)


@metric("movesim_distance")
def _ms_dist(B, R, S, p):
    D2 = B.diag ** 2
    hr = R.cached(("ms_dist", p["n_bins"]), lambda: movesim_hist(R.hop ** 2, 0.0, D2, p["n_bins"]))
    return _ms_jsd(hr, movesim_hist(S.hop ** 2, 0.0, D2, p["n_bins"]))


@metric("movesim_radius")
def _ms_rad(B, R, S, p):
    D2 = B.diag ** 2
    hr = R.cached(("ms_rad", p["n_bins"]), lambda: movesim_hist(R.gyr2, 0.0, D2, p["n_bins"]))
    return _ms_jsd(hr, movesim_hist(S.gyr2, 0.0, D2, p["n_bins"]))


def stay_durations(P: Prep) -> np.ndarray:
    start = np.ones(len(P.poi), bool)
    start[1:] = P.poi[1:] != P.poi[:-1]
    start[P.first] = True
    idx = np.flatnonzero(start)
    tr = P.tr[idx]
    has_next = np.zeros(len(idx), bool)
    has_next[:-1] = tr[1:] == tr[:-1]
    return (P.t[idx[1:]] - P.t[idx[:-1]])[has_next[:-1]].astype(np.float64)


@metric("movesim_duration")
def _ms_dur(B, R, S, p):
    spd = p["slots_per_day"]
    hr = R.cached("ms_dur", lambda: movesim_hist(stay_durations(R) / DAY_MIN, 0.0, 1.0, spd))
    return _ms_jsd(hr, movesim_hist(stay_durations(S) / DAY_MIN, 0.0, 1.0, spd))


def distinct_per_trail(P: Prep) -> np.ndarray:
    base = int(P.poi.max()) + 1 if len(P.poi) else 1
    pairs = np.unique(P.tr * base + P.poi)
    return np.bincount(pairs // base, minlength=P.n).astype(np.float64)


@metric("movesim_dailyloc")
def _ms_daily(B, R, S, p):
    spd = p["slots_per_day"]
    hr = R.cached("ms_daily", lambda: movesim_hist(distinct_per_trail(R) / spd, 0.0, 1.0, spd))
    return _ms_jsd(hr, movesim_hist(distinct_per_trail(S) / spd, 0.0, 1.0, spd))


def topk_visit_probs(P: Prep, k: int, n_locs: int) -> np.ndarray:
    c = np.bincount(P.poi, minlength=n_locs).astype(np.float64)
    c = c / max(c.sum(), 1.0)
    top = np.sort(c[np.flatnonzero(c)])[::-1][:k]
    return np.concatenate([top, np.zeros(k - len(top))])


def rank_profile(P: Prep, k: int) -> np.ndarray:
    base = int(P.poi.max()) + 1 if len(P.poi) else 1
    pairs, cnt = np.unique(P.tr * base + P.poi, return_counts=True)
    tr = pairs // base
    o = np.lexsort((-cnt, tr))
    tr, cnt = tr[o], cnt[o]
    start = np.concatenate([[0], np.flatnonzero(tr[1:] != tr[:-1]) + 1])
    rank = np.arange(len(tr)) - np.repeat(start, np.diff(np.append(start, len(tr))))
    ok = rank < k
    prof = np.bincount(rank[ok], weights=cnt[ok], minlength=k).astype(np.float64)
    return prof / max(prof.sum(), 1e-300)


@metric("movesim_grank")
def _ms_grank(B, R, S, p):
    k, nl = p["top_k"], len(B.cat.xy)
    hr = R.cached(("ms_grank", k), lambda: movesim_hist(topk_visit_probs(R, k, nl), 0.0, 1.0, 100))
    return _ms_jsd(hr, movesim_hist(topk_visit_probs(S, k, nl), 0.0, 1.0, 100))


@metric("movesim_irank")
def _ms_irank(B, R, S, p):
    k = p["top_k"]
    hr = R.cached(("ms_irank", k), lambda: movesim_hist(rank_profile(R, k), 0.0, 1.0, 100))
    return _ms_jsd(hr, movesim_hist(rank_profile(S, k), 0.0, 1.0, 100))


@metric("movesim_grank_rank")
def _ms_grank_r(B, R, S, p):
    k, nl = p["top_k"], len(B.cat.xy)
    return jsd_nats(R.cached(("ms_topk", k), lambda: topk_visit_probs(R, k, nl)), topk_visit_probs(S, k, nl))


@metric("movesim_irank_rank")
def _ms_irank_r(B, R, S, p):
    k = p["top_k"]
    return jsd_nats(R.cached(("ms_prof", k), lambda: rank_profile(R, k)), rank_profile(S, k))


@metric("llmob_sd")
def _llm_sd(B, R, S, p):
    hr = R.cached(("llm_sd", p["max_km"], p["n_bins"]),
                  lambda: llmob_hist(np.round(R.hop, 3), 0.0, p["max_km"], p["n_bins"]))
    return jsd_nats(hr, llmob_hist(np.round(S.hop, 3), 0.0, p["max_km"], p["n_bins"]))


def step_intervals(P: Prep, slot_min: int, unit_min: float) -> np.ndarray:
    s = P.t // slot_min
    return np.round((s[P.nxt + 1] - s[P.nxt]) * slot_min) / unit_min


@metric("llmob_si")
def _llm_si(B, R, S, p):
    f = (lambda P: llmob_hist(step_intervals(P, p["slot_min"], p["unit_min"]), 0.0, p["max_value"], p["n_bins"]))
    return jsd_nats(R.cached(("llm_si", p["unit_min"]), lambda: f(R)), f(S))


@metric("llmob_si_code")
def _llm_si_code(B, R, S, p):
    return _llm_si(B, R, S, p)


def _dard_keys(P: Prep, slot_min):
    slot = (P.t % DAY_MIN) // slot_min
    return slot * int(P.cat.n_roots) + np.asarray(P.cat.root, np.int64)[P.poi]


def _stvd_keys(P: Prep, slot_min):
    slot = (P.t % DAY_MIN) // slot_min
    return slot * len(P.cat.xy) + P.poi


@metric("llmob_dard")
def _llm_dard(B, R, S, p):
    return sparse_jsd_nats(_dard_keys(R, p["slot_min"]), _dard_keys(S, p["slot_min"]))


@metric("llmob_dard_code")
def _llm_dard_code(B, R, S, p):
    return jsd_nats(*quantized_ids(_dard_keys(R, p["slot_min"]), _dard_keys(S, p["slot_min"]), p["n_bins"]))


@metric("llmob_stvd")
def _llm_stvd(B, R, S, p):
    return sparse_jsd_nats(_stvd_keys(R, p["slot_min"]), _stvd_keys(S, p["slot_min"]))


@metric("llmob_stvd_code")
def _llm_stvd_code(B, R, S, p):
    return jsd_nats(*quantized_ids(_stvd_keys(R, p["slot_min"]), _stvd_keys(S, p["slot_min"]), p["n_bins"]))


class Benchmark:

    def __init__(self, catalog, protocol: dict | None = None, metrics=None):
        self.cat = catalog
        self.P = protocol or load_protocol()
        self.W, self.H = float(catalog.width_km), float(catalog.height_km)
        self.diag = math.hypot(self.W, self.H)
        self.seed = int(self.P["common"]["seed"])
        self.specs = {m["name"]: m for m in self.P["metrics"]}
        missing = [m for m in self.specs if m not in METRICS]
        if missing:
            raise KeyError(f"protocol metrics without an implementation: {missing}")
        self.names = list(metrics) if metrics is not None else list(self.specs)
        self._fixed: dict = {}

    def fixed(self, key, fn):
        if key not in self._fixed:
            self._fixed[key] = fn()
        return self._fixed[key]

    def circles(self, p, constant_radius: bool):
        rng = _rng(self.seed, p["stream"])
        n = p["n_queries"]
        c = np.stack([rng.random(n) * self.W, rng.random(n) * self.H], 1)
        longer = max(self.W, self.H)
        if constant_radius:
            r = np.full(n, longer * p["radius_frac_of_longer_side"])
        else:
            r = (1.0 - rng.random(n)) * longer * p["max_radius_frac_of_longer_side"]
        return c, r

    def squares(self, p, inside: bool):
        rng = _rng(self.seed, p["stream"])
        n = p["n_queries"]
        e = math.sqrt(self.W * self.H / p["size_factor"])
        if inside:
            cx = rng.random(n) * (self.W - e) + e / 2
            cy = rng.random(n) * (self.H - e) + e / 2
        else:
            cx, cy = rng.random(n) * self.W, rng.random(n) * self.H
        return np.stack([cx - e / 2, cx + e / 2, cy - e / 2, cy + e / 2], 1)

    def st_squares(self, p):
        boxes = self.squares(p, inside=True)
        rng = _rng(self.seed, p["stream"] * 1000 + 1)
        t0 = rng.integers(0, p["n_timestamps"] - p["phi"] + 1, size=p["n_queries"])
        return boxes, t0

    def windows(self, p):
        rng = _rng(self.seed, p["stream"])
        return rng.integers(0, p["n_timestamps"] - p["phi"] + 1, size=p["n_windows"])

    def cell_sets(self, p):
        rng = _rng(self.seed, p["stream"])
        G2 = p["grid"] ** 2
        rows, cols = [], []
        for i in range(p["n_queries"]):
            k = int(rng.integers(1, G2 + 1))
            cells = rng.choice(G2, size=k, replace=False)
            rows.append(cells)
            cols.append(np.full(k, i))
        r, c = np.concatenate(rows), np.concatenate(cols)
        return sparse.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(G2, p["n_queries"]))

    def prepare(self, ts: TrailSet) -> Prep:
        return Prep(ts, self.cat)

    def compare(self, ref: Prep, rel: Prep, timings: dict | None = None) -> dict:
        out = {}
        for name in self.names:
            t0 = time.perf_counter()
            out[name] = float(METRICS[name](self, ref, rel, self.specs[name]["params"]))
            if timings is not None:
                timings[name] = timings.get(name, 0.0) + time.perf_counter() - t0
        return out


def v3_measurements(v3_config: dict, protocol: dict | None = None) -> list[str]:
    from .marginals import N_MARGINALS
    from .workload import FAMILIES, MARGINAL_FAMILIES
    P = protocol or load_protocol()
    fams = [f for f in str(v3_config.get("work_families", "")).split(",") if f]
    unknown = [f for f in fams if f not in FAMILIES]
    if unknown:
        raise ValueError(f"unknown workload families {unknown}")
    assert len(MARGINAL_FAMILIES) == N_MARGINALS
    names = fams + list(MARGINAL_FAMILIES)
    described = P["v3_split"]["families"]
    missing = [f for f in names if f not in described]
    if missing:
        raise KeyError(f"the v3 config uses families without a feature descriptor in the protocol: {missing} "
                       "(describing them is a protocol amendment)")
    return names


def _closure(vars_, closure: dict) -> set:
    out = set(vars_)
    todo = list(vars_)
    while todo:
        v = todo.pop()
        for w in closure.get(v, []):
            if w not in out:
                out.add(w)
                todo.append(w)
    return out


def classify(feature: dict, measurements: list[str], protocol: dict | None = None) -> tuple[str, str]:
    P = protocol or load_protocol()
    S = P["v3_split"]
    fam = S["families"]
    unit, vars_ = feature["unit"], set(feature["vars"])
    if unit == "task":
        return "unmeasured", "downstream task"
    for f in measurements:
        d = fam[f]
        if not d.get("restricted") and d["unit"] == unit and vars_ <= _closure(d["vars"], S["closure"]):
            return "measured", f
    unit_r = S["relax_unit"].get(unit, unit)
    vars_r = {S["relax_vars"].get(v, v) for v in vars_}
    for f in measurements:
        d = fam[f]
        if d["unit"] == unit_r and vars_r <= _closure(d["vars"], S["closure"]):
            return "near", f"{f} (relaxed)"
    same = [f for f in measurements if fam[f]["unit"] == unit_r]
    joint = set().union(*[_closure(fam[f]["vars"], S["closure"]) for f in same]) if same else set()
    if vars_r <= joint:
        used = [f for f in same if _closure(fam[f]["vars"], S["closure"]) & vars_r]
        return "near", "joint of " + "+".join(used)
    return "unmeasured", "no v3 measurement of " + ",".join(sorted(vars_r - joint))


def v3_split(v3_config: dict, protocol: dict | None = None, metric_names=None) -> dict:
    P = protocol or load_protocol()
    meas = v3_measurements(v3_config, P)
    out = {m["name"]: classify(m["feature"], meas, P) for m in P["metrics"]}
    for name in metric_names or []:
        if name.startswith("nextpoi_"):
            out[name] = classify(P["next_poi"]["feature"], meas, P)
    return out


def metric_info(protocol: dict | None = None) -> dict:
    P = protocol or load_protocol()
    info = {m["name"]: {k: m[k] for k in ("direction", "role", "family", "source_paper")} for m in P["metrics"]}
    for m in P["metrics"]:
        if m.get("duplicate_of"):
            info[m["name"]]["role"] = "duplicate"
            info[m["name"]]["duplicate_of"] = m["duplicate_of"]
    np_ = P["next_poi"]
    for model in np_["models"]:
        for met in np_["metrics"]:
            for scope in np_["scopes"]:
                info[f"nextpoi_{model}_{met}_{scope}"] = {
                    "direction": "higher", "role": "primary" if scope == "last" else "variant",
                    "family": "downstream", "source_paper": "Massive-STEPS"}
    return info
