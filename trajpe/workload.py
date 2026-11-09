from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .marginals import start_bins
from .trails import TrailSet

GAP_EDGES = np.array([0, 2, 4, 6, 9, 12, 16, 20, 25, 30, 40, 50, 60, 75, 90, 120, 150, 180, 240, 300, 360, 481],
                     np.float64)
N_GAP = len(GAP_EDGES) - 1
N_HB = 4
N_LC = 2
HOP_EDGES = np.array([0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 7.5, 10, 15, 20, 30, np.inf])
GYR_EDGES = np.array([0, 0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 7.5, 10, 15, np.inf])
LEN7 = np.array([0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5] + [6] * 10)
TD_EDGES = np.array([0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 24, 27, 30, 35, 40,
                     50, 60, 80, 100, 130, np.inf])
DIAM_EDGES = np.array([0, 0.25, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 25,
                       np.inf])
N_TD, N_DIAM = len(TD_EDGES) - 1, len(DIAM_EDGES) - 1
LC4 = np.array([0, 0, 0, 1, 2] + [3] * 1000)
FAMILIES = ("gap", "hour_root", "root_trans", "hop", "len_gyr", "cell", "cell16", "od", "od8", "weekhour",
            "ctrans8", "ctrans16", "hb_cat", "tdist", "diam", "od6", "ctrans6", "tcell6",
            "hcell6", "tctrans6", "gram6")
DEFAULT_FAMILIES = "gap,hour_root,root_trans,hop,len_gyr,cell16,od8,weekhour"
MARGINAL_FAMILIES = ("start", "cat", "venue", "cat_trans", "venue_trans")
CELL_GRID, OD_GRID = 8, 4
POOL_MIN_BINS = 1000
KNOWN_SIGMAS = 2.0


def _cells(xy, catalog, g):
    gx = np.clip((xy[:, 0] / catalog.width_km * g).astype(np.int64), 0, g - 1)
    gy = np.clip((xy[:, 1] / catalog.height_km * g).astype(np.int64), 0, g - 1)
    return gy * g + gx


def _bin(edges, x):
    return np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)


def _hops(ts: TrailSet):
    has_next = np.ones(len(ts.poi), bool)
    has_next[ts.last] = False
    return np.flatnonzero(has_next)


def n_bins(name: str, catalog) -> int:
    return {"gap": N_LC * N_HB * N_GAP, "hour_root": 24 * catalog.n_roots, "root_trans": catalog.n_roots ** 2,
            "hop": len(HOP_EDGES) - 1, "len_gyr": 7 * (len(GYR_EDGES) - 1), "cell": CELL_GRID ** 2,
            "od": OD_GRID ** 4, "od8": 8 ** 4, "ctrans16": 16 ** 4, "ctrans8": 8 ** 4, "cell16": 256,
            "weekhour": 168, "hb_cat": 8 * len(catalog.categories), "tdist": 4 * N_TD, "diam": 1 + 3 * N_DIAM,
            "od6": 6 ** 4, "ctrans6": 6 ** 4, "tcell6": 56 * 36,
            "hcell6": 24 * 36, "tctrans6": N_HB * 6 ** 4, "gram6": 6 ** 4 + 6 ** 6}[name]


def trail_geometry(ts: TrailSet, catalog):
    xy = catalog.xy[ts.poi]
    nxt = _hops(ts)
    td = np.bincount(ts.trail_of[nxt], weights=np.linalg.norm(xy[nxt + 1] - xy[nxt], axis=1), minlength=len(ts))
    diam = np.zeros(len(ts))
    pos = np.arange(len(ts.poi)) - np.repeat(ts.first, ts.lengths)
    rem = np.repeat(ts.lengths, ts.lengths) - pos - 1
    for k in range(1, int(ts.lengths.max()) if len(ts) else 1):
        i = np.flatnonzero(rem >= k)
        if len(i) == 0:
            break
        np.maximum.at(diam, ts.trail_of[i], np.linalg.norm(xy[i + k] - xy[i], axis=1))
    return td, diam


def gap_cells(ts: TrailSet):
    nxt = _hops(ts)
    L = ts.lengths[ts.trail_of[nxt]]
    lc = (L >= 3).astype(np.int64)
    hb = ((ts.tmin[nxt] % 1440) // 360).astype(np.int64)
    g = (ts.tmin[nxt + 1] - ts.tmin[nxt]).astype(np.float64)
    return nxt, lc * N_HB + hb, g, 1.0 / (L - 1)


def gram_rows(ts: TrailSet, catalog, g: int = 6):
    G = g * g
    c = _cells(catalog.xy[ts.poi], catalog, g)
    keep = np.ones(len(c), bool)
    keep[1:] = c[1:] != c[:-1]
    keep[ts.first] = True
    k = np.flatnonzero(keep)
    kc, ktr = c[k], ts.trail_of[k]
    klen = np.bincount(ktr, minlength=len(ts))
    last = np.ones(len(k), bool)
    last[:-1] = ktr[1:] != ktr[:-1]
    i2 = np.flatnonzero(~last)
    last2 = np.ones(len(k), bool)
    last2[:-2] = ktr[2:] != ktr[:-2]
    i3 = np.flatnonzero(~last2)
    single = np.flatnonzero(klen == 1)
    m = np.where(klen == 1, 1, 2 * klen - 3).astype(np.float64)
    sc = c[ts.first[single]]
    tr = np.concatenate([single, ktr[i2], ktr[i3]])
    col = np.concatenate([sc * G + sc, kc[i2] * G + kc[i2 + 1], G * G + kc[i3] * G * G + kc[i3 + 1] * G + kc[i3 + 2]])
    return tr, col, 1.0 / m[tr]


def rows(ts: TrailSet, catalog, name: str):
    tr = ts.trail_of
    L = ts.lengths
    if name == "gap":
        nxt, cell, g, w = gap_cells(ts)
        return tr[nxt], cell * N_GAP + _bin(GAP_EDGES, g), w
    if name == "hour_root":
        hour = ((ts.tmin % 1440) // 60).astype(np.int64)
        return tr, hour * catalog.n_roots + catalog.root[ts.poi], (1.0 / L)[tr]
    if name in ("cell", "cell16"):
        return tr, _cells(catalog.xy[ts.poi], catalog, CELL_GRID if name == "cell" else 16), (1.0 / L)[tr]
    if name == "weekhour":
        wh = (np.clip(ts.tmin // 60, 0, 7 * 24 * 2 - 1) % 168).astype(np.int64)
        return tr, wh, (1.0 / L)[tr]
    if name == "tcell6":
        wb = ((ts.tmin % 10080) // 180).astype(np.int64)
        return tr, wb * 36 + _cells(catalog.xy[ts.poi], catalog, 6), (1.0 / L)[tr]
    if name == "hcell6":
        hour = ((ts.tmin % 1440) // 60).astype(np.int64)
        return tr, hour * 36 + _cells(catalog.xy[ts.poi], catalog, 6), (1.0 / L)[tr]
    if name == "gram6":
        return gram_rows(ts, catalog, 6)
    if name == "hb_cat":
        hb = ((ts.tmin % 1440) // 180).astype(np.int64)
        return tr, hb * len(catalog.categories) + np.asarray(catalog.cat)[ts.poi].astype(np.int64), (1.0 / L)[tr]
    if name in ("tdist", "diam"):
        td, diam = trail_geometry(ts, catalog)
        lc = LC4[L]
        if name == "tdist":
            return np.arange(len(ts)), lc * N_TD + _bin(TD_EDGES, td), np.ones(len(ts))
        return np.arange(len(ts)), np.where(lc == 0, 0, 1 + (lc - 1) * N_DIAM + _bin(DIAM_EDGES, diam)), np.ones(len(ts))
    if name in ("od", "od8", "od6"):
        g = {"od": OD_GRID, "od8": 8, "od6": 6}[name]
        a = _cells(catalog.xy[ts.poi[ts.first]], catalog, g)
        b = _cells(catalog.xy[ts.poi[ts.last]], catalog, g)
        return np.arange(len(ts)), a * g * g + b, np.ones(len(ts))
    nxt = _hops(ts)
    if name in ("ctrans16", "ctrans8", "ctrans6"):
        g = {"ctrans16": 16, "ctrans8": 8, "ctrans6": 6}[name]
        c = _cells(catalog.xy[ts.poi], catalog, g)
        return tr[nxt], c[nxt] * g * g + c[nxt + 1], (1.0 / (L - 1))[tr[nxt]]
    if name == "tctrans6":
        c = _cells(catalog.xy[ts.poi], catalog, 6)
        hb = ((ts.tmin[nxt] % 1440) // 360).astype(np.int64)
        return tr[nxt], hb * 6 ** 4 + c[nxt] * 36 + c[nxt + 1], (1.0 / (L - 1))[tr[nxt]]
    if name == "root_trans":
        r = catalog.root[ts.poi]
        return tr[nxt], r[nxt] * catalog.n_roots + r[nxt + 1], (1.0 / (L - 1))[tr[nxt]]
    if name == "hop":
        xy = catalog.xy[ts.poi]
        d = np.linalg.norm(xy[nxt + 1] - xy[nxt], axis=1)
        return tr[nxt], _bin(HOP_EDGES, d), (1.0 / (L - 1))[tr[nxt]]
    if name == "len_gyr":
        xy = catalog.xy[ts.poi]
        cent = np.add.reduceat(xy, ts.first, axis=0) / L[:, None]
        gyr = np.sqrt(np.add.reduceat(((xy - cent[tr]) ** 2).sum(1), ts.first) / L)
        return np.arange(len(ts)), LEN7[np.minimum(L, 20)] * (len(GYR_EDGES) - 1) + _bin(GYR_EDGES, gyr), np.ones(len(ts))
    raise ValueError(name)


def feature_matrix(ts: TrailSet, catalog, name: str) -> sparse.csr_matrix:
    r, c, v = rows(ts, catalog, name)
    return sparse.csr_matrix((v, (r, c)), shape=(len(ts), n_bins(name, catalog)))


def histogram(ts: TrailSet, trail_weights: np.ndarray, catalog, name: str) -> np.ndarray:
    r, c, v = rows(ts, catalog, name)
    return np.bincount(c, weights=v * trail_weights[r], minlength=n_bins(name, catalog)).astype(np.float64)


def dp_workload(priv: TrailSet, trail_weights: np.ndarray, catalog, sigma: float, rng, families) -> dict:
    out = {}
    for name in families:
        h = histogram(priv, trail_weights, catalog, name)
        if sigma > 0:
            h = h + rng.normal(0.0, sigma, h.shape)
        out[name] = h
    return out


def digest(work: dict) -> str:
    import hashlib
    h = hashlib.sha256()
    for k in sorted(work):
        h.update(k.encode())
        h.update(np.ascontiguousarray(work[k]).tobytes())
    return h.hexdigest()


@dataclass
class DPOutputs:
    work: dict
    sigma_w: float
    marg: object = None
    sigma_m: float = 0.0
    popular: np.ndarray = None
    n_hat: float = 0.0
    sigma_pop: float = 0.0
    known_sigmas: float = KNOWN_SIGMAS


@dataclass
class Family:
    name: str
    Phi: sparse.csr_matrix
    t: np.ndarray
    y: np.ndarray
    sigma: np.ndarray
    calib: bool = True


def _pool_matrix(B: int, known: np.ndarray) -> sparse.csr_matrix:
    col = np.full(B, len(known), dtype=np.int64)
    col[known] = np.arange(len(known))
    return sparse.csr_matrix((np.ones(B), (np.arange(B), col)), shape=(B, len(known) + 1))


def pooled_family(Phi: sparse.csr_matrix, target: np.ndarray, known: np.ndarray, total: float):
    t = np.append(target[known], max(total - target[known].sum(), 0.0))
    return (Phi @ _pool_matrix(Phi.shape[1], known)).tocsr(), t


def _pooled(name, Phi, raw, sigma, known, n_hat, var_n_hat):
    P, t = pooled_family(Phi, np.maximum(raw, 0.0), known, n_hat)
    y = np.append(raw[known], n_hat - raw[known].sum())
    s = np.append(np.full(len(known), sigma), np.sqrt(var_n_hat + len(known) * sigma ** 2))
    return Family(name, P, t, y, s)


def _transition_family(name, ts, codes, dom, n_codes, counts, sigma, n_hat, var_n_hat):
    pos = np.full(n_codes, -1, dtype=np.int64)
    pos[dom] = np.arange(len(dom))
    i = _hops(ts)
    a, b = pos[codes[i]], pos[codes[i + 1]]
    k = len(dom)
    flat = np.asarray(counts, np.float64).ravel()
    bins = np.flatnonzero(flat > 0)
    col = np.full(k * k, len(bins), np.int64)
    col[bins] = np.arange(len(bins))
    inside = (a >= 0) & (b >= 0)
    c = np.where(inside, col[np.where(inside, a * k + b, 0)], len(bins))
    w = (1.0 / np.maximum(ts.lengths - 1, 1))[ts.trail_of[i]]
    Phi = sparse.csr_matrix((w, (ts.trail_of[i], c)), shape=(len(ts), len(bins) + 1))
    kept = flat[bins]
    t = np.append(kept, max(n_hat - kept.sum(), 0.0))
    y = np.append(kept, n_hat - kept.sum())
    s = np.append(np.full(len(bins), sigma), np.sqrt(var_n_hat + len(bins) * sigma ** 2))
    return Family(name, Phi, t, y, s)


def families(ts: TrailSet, catalog, names, dp: DPOutputs) -> list:
    out = []
    work, sw, marg = dp.work or {}, dp.sigma_w, dp.marg
    var_n_hat = 168 * dp.sigma_m ** 2
    for name in names:
        if name in work:
            Phi = feature_matrix(ts, catalog, name)
            raw = work[name]
            if Phi.shape[1] > POOL_MIN_BINS:
                out.append(_pooled(name, Phi, raw, sw, np.flatnonzero(raw > dp.known_sigmas * sw), dp.n_hat, var_n_hat))
            else:
                out.append(Family(name, Phi, np.maximum(raw, 0.0), raw, np.full(len(raw), sw)))
        elif marg is None:
            continue
        elif name == "start":
            Phi = sparse.csr_matrix((np.ones(len(ts)), (np.arange(len(ts)), start_bins(ts))), shape=(len(ts), 168))
            out.append(Family(name, Phi, marg.start, marg.start, np.full(168, dp.sigma_m)))
        elif name == "cat":
            tr = ts.trail_of
            Phi = sparse.csr_matrix(((1.0 / ts.lengths)[tr], (tr, catalog.cat[ts.poi].astype(np.int64))),
                                    shape=(len(ts), len(catalog.categories)))
            known = (np.flatnonzero(marg.cat > KNOWN_SIGMAS * dp.sigma_m) if dp.sigma_m > 0
                     else np.flatnonzero(marg.cat > 0))
            out.append(_pooled(name, Phi, marg.cat, dp.sigma_m, known, dp.n_hat, var_n_hat))
        elif name == "venue" and dp.popular is not None and len(dp.popular):
            tr = ts.trail_of
            col = np.full(len(catalog), len(dp.popular), dtype=np.int64)
            col[dp.popular] = np.arange(len(dp.popular))
            Phi = sparse.csr_matrix(((1.0 / ts.lengths)[tr], (tr, col[ts.poi])), shape=(len(ts), len(dp.popular) + 1))
            kept = marg.pop[dp.popular]
            t = np.append(kept, max(dp.n_hat - kept.sum(), 0.0))
            y = np.append(kept, dp.n_hat - kept.sum())
            sp = dp.sigma_pop if dp.sigma_pop > 0 else dp.sigma_m
            s = np.append(np.full(len(kept), sp), np.sqrt(var_n_hat + len(kept) * sp ** 2))
            out.append(Family(name, Phi, t, y, s))
        elif name == "cat_trans" and marg.cat_trans is not None and marg.cat_trans.size:
            out.append(_transition_family(name, ts, catalog.cat[ts.poi].astype(np.int64), marg.cat_dom,
                                          len(catalog.categories), marg.cat_trans, dp.sigma_m, dp.n_hat, var_n_hat))
        elif name == "venue_trans" and marg.venue_trans is not None and marg.venue_trans.size:
            out.append(_transition_family(name, ts, ts.poi.astype(np.int64), marg.venue_dom, len(catalog),
                                          marg.venue_trans, dp.sigma_m, dp.n_hat, var_n_hat))
    return out


def calibrate(p: np.ndarray, fams, iters: int = 20, eta: float = 1.0, cap: float = 0.0, floor: float = 0.0):
    act = np.flatnonzero(p > 0)
    q0 = p[act] / p[act].sum()
    q = q0.copy()
    F = []
    for Phi, t in fams:
        A = Phi[act]
        t = np.maximum(np.asarray(t, np.float64), 0.0)
        F.append((A, A.T.tocsr(), t / max(t.sum(), 1e-12)))
    for _ in range(iters):
        for A, At, t in F:
            mu = At @ q
            reach = mu > 1e-15
            tt = np.where(reach, t, 0.0)
            if tt.sum() <= 0:
                continue
            tt = tt / tt.sum() + floor
            tt = np.where(reach, tt, 0.0)
            tt = tt / tt.sum()
            r = np.zeros_like(mu)
            r[reach] = np.log(np.maximum(tt[reach], 1e-12) / mu[reach])
            q = q * np.exp(eta * (A @ r))
            q = q / q.sum()
        if cap > 0:
            q = q0 * np.clip(q / q0, 1.0 / cap, cap)
            q = q / q.sum()
    out = np.zeros(len(p))
    out[act] = q
    return out


def calibrate_parallel(p: np.ndarray, fams, iters: int = 50, eta: float = 1.0, cap: float = 0.0,
                       lam=None) -> np.ndarray:
    act = np.flatnonzero(p > 0)
    q0 = p[act] / p[act].sum()
    q = q0.copy()
    lam = np.ones(len(fams)) if lam is None else np.asarray(lam, np.float64)
    F = []
    for Phi, t in fams:
        A = Phi[act]
        t = np.maximum(np.asarray(t, np.float64), 0.0)
        F.append((A, A.T.tocsr(), t / max(t.sum(), 1e-12)))
    for _ in range(iters):
        step = np.zeros(len(q))
        for (A, At, t), l in zip(F, lam):
            mu = At @ q
            reach = mu > 1e-15
            tt = np.where(reach, t, 0.0)
            if tt.sum() <= 0:
                continue
            tt = tt / tt.sum()
            r = np.zeros_like(mu)
            r[reach] = np.log(np.maximum(tt[reach], 1e-12) / mu[reach])
            step += l * (A @ r)
        q = q * np.exp(eta * step / lam.sum())
        q = q / q.sum()
        if cap > 0:
            q = q0 * np.clip(q / q0, 1.0 / cap, cap)
            q = q / q.sum()
    out = np.zeros(len(p))
    out[act] = q
    return out


def family_error(Phi, t, q) -> float:
    from .metrics import jsd
    return jsd(np.maximum(np.asarray(t), 0.0), np.asarray(Phi.T @ q))


def time_ground(ts: TrailSet, gap_counts: np.ndarray, rng, strength: float = 1.0) -> TrailSet:
    target = np.maximum(np.asarray(gap_counts, np.float64), 0.0).reshape(N_LC * N_HB, N_GAP)
    nxt, cell, g, w = gap_cells(ts)
    new = g.copy()
    for c in range(N_LC * N_HB):
        m = np.flatnonzero(cell == c)
        tc = target[c]
        if len(m) == 0 or tc.sum() <= 0:
            continue
        order = np.lexsort((rng.random(len(m)), g[m]))
        cw = np.cumsum(w[m][order])
        u = np.empty(len(m))
        u[order] = (cw - 0.5 * w[m][order]) / cw[-1]
        C = np.concatenate([[0.0], np.cumsum(tc) / tc.sum()])
        b = np.clip(np.searchsorted(C, u, side="left") - 1, 0, N_GAP - 1)
        f = np.clip((u - C[b]) / np.maximum(C[b + 1] - C[b], 1e-12), 0.0, 1.0 - 1e-9)
        lo, hi = GAP_EDGES[b], GAP_EDGES[b + 1]
        new[m] = np.floor(lo + f * (hi - lo))
    if strength < 1:
        keep = rng.random(len(g)) >= strength
        new[keep] = g[keep]
    t = ts.tmin.astype(np.int64).copy()
    gaps = np.zeros(len(t), np.int64)
    gaps[nxt + 1] = new.astype(np.int64)
    gaps[ts.first] = 0
    cs = np.cumsum(gaps)
    base = np.repeat(cs[ts.first], ts.lengths)
    t = np.repeat(t[ts.first], ts.lengths) + (cs - base)
    return TrailSet(ts.poi.copy(), t.astype(np.int32), ts.offsets.copy())
