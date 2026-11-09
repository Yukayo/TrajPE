from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .trails import TrailSet

LEN_BUCKETS = np.array([0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5] + [6] * 10)
DIST_CENTERS_KM = np.array([0.05, 0.2, 0.5, 1.0, 2.0, 3.5, 6.0, 10.0, 16.0, 25.0, 40.0])
GAP_CENTERS_MIN = np.array([2, 7, 15, 30, 50, 80, 120, 180, 270, 400])


@dataclass
class EmbedConfig:
    grids: tuple = (4, 8, 16, 32)
    w_grids: tuple = (1.0, 1.0, 1.0, 0.7)
    od_grid: int = 8
    w_od: float = 1.0
    w_hour: float = 1.0
    w_start_hour: float = 1.0
    hour_sigma: float = 1.0
    w_dow: float = 0.5
    w_root: float = 1.0
    w_cat: float = 1.0
    w_len: float = 1.0
    w_hop: float = 1.0
    w_travel: float = 0.5
    w_gyr: float = 0.5
    w_gap: float = 0.5
    w_seq: float = 1.0
    seq_k: int = 4
    w_cbigram: float = 1.0
    cbigram_buckets: int = 256
    w_ret: float = 0.0
    w_venue: float = 1.5
    popular: object = None
    text_model: str = ""


def _cells(xy, catalog, g):
    gx = np.clip((xy[:, 0] / catalog.width_km * g).astype(np.int64), 0, g - 1)
    gy = np.clip((xy[:, 1] / catalog.height_km * g).astype(np.int64), 0, g - 1)
    return gy * g + gx


def _hist(rows, cols, weights, n_rows, n_cols):
    flat = np.bincount(rows * n_cols + cols, weights=weights, minlength=n_rows * n_cols)
    return flat.reshape(n_rows, n_cols).astype(np.float32)


def _soft_hours(hours, sigma):
    centers = np.arange(24) + 0.5
    d = np.abs(hours[:, None] - centers[None, :])
    d = np.minimum(d, 24 - d)
    w = np.exp(-0.5 * (d / sigma) ** 2)
    return (w / w.sum(1, keepdims=True)).astype(np.float32)


def _soft_log_bins(values, centers):
    lv, lc = np.log(np.maximum(values, 1e-6)), np.log(centers)
    u = np.interp(lv, lc, np.arange(len(centers)))
    lo = np.floor(u).astype(np.int64)
    hi = np.minimum(lo + 1, len(centers) - 1)
    return lo, hi, (u - lo)


def _soft_hist(rows, values, weights, n_rows, centers):
    lo, hi, f = _soft_log_bins(values, centers)
    k = len(centers)
    return _hist(rows, lo, weights * (1 - f), n_rows, k) + _hist(rows, hi, weights * f, n_rows, k)


def return_flags(ts: TrailSet) -> tuple[np.ndarray, np.ndarray]:
    tr = ts.trail_of.astype(np.int64)
    key = tr * (int(ts.poi.max()) + 1 if len(ts.poi) else 1) + ts.poi
    _, first, inv = np.unique(key, return_index=True, return_inverse=True)
    seen_before = first[inv] < np.arange(len(key))
    rep = np.zeros(len(key), bool)
    rep[1:] = (ts.poi[1:] == ts.poi[:-1]) & (tr[1:] == tr[:-1])
    return rep, seen_before & ~rep


def embed(ts: TrailSet, catalog, cfg: EmbedConfig = EmbedConfig()) -> np.ndarray:
    if cfg.text_model:
        from .text_embed import serialize, text_embed
        return text_embed(serialize(ts, catalog), cfg.text_model)
    n = len(ts)
    lens = ts.lengths
    tr = ts.trail_of
    w = (1.0 / lens)[tr]
    xy = catalog.xy[ts.poi]
    hours = (ts.tmin % 1440) / 60.0
    rows = np.arange(n)
    ones = np.ones(n)
    blocks = []

    for g, wg in zip(cfg.grids, cfg.w_grids):
        blocks.append(wg * _hist(tr, _cells(xy, catalog, g), w, n, g * g))

    first, last = ts.first, ts.last
    g = cfg.od_grid
    blocks.append(cfg.w_od * _hist(rows, _cells(xy[first], catalog, g), ones, n, g * g))
    blocks.append(cfg.w_od * _hist(rows, _cells(xy[last], catalog, g), ones, n, g * g))

    soft = _soft_hours(hours, cfg.hour_sigma)
    blocks.append(cfg.w_hour * np.add.reduceat(soft * w[:, None].astype(np.float32), first, axis=0))
    blocks.append(cfg.w_start_hour * soft[first])
    blocks.append(cfg.w_dow * _hist(rows, np.clip(ts.tmin[first] // 1440, 0, 6), ones, n, 7))

    root = catalog.root[ts.poi]
    blocks.append(cfg.w_root * _hist(tr, root, w, n, catalog.n_roots))
    blocks.append(cfg.w_cat * _hist(tr, catalog.cat[ts.poi].astype(np.int64), w, n, len(catalog.categories)))
    blocks.append(cfg.w_len * _hist(rows, LEN_BUCKETS[np.minimum(lens, 20)], ones, n, 7))

    has_next = np.ones(len(ts.poi), bool)
    has_next[last] = False
    nxt = np.flatnonzero(has_next)
    hop = np.linalg.norm(xy[nxt + 1] - xy[nxt], axis=1)
    hw = (1.0 / (lens - 1))[tr[nxt]]
    blocks.append(cfg.w_hop * _soft_hist(tr[nxt], hop, hw, n, DIST_CENTERS_KM))
    travel = np.bincount(tr[nxt], weights=hop, minlength=n)
    blocks.append(cfg.w_travel * _soft_hist(rows, travel, ones, n, DIST_CENTERS_KM))
    cent = np.add.reduceat(xy, first, axis=0) / lens[:, None]
    gyr = np.sqrt(np.add.reduceat(((xy - cent[tr]) ** 2).sum(1), first) / lens)
    blocks.append(cfg.w_gyr * _soft_hist(rows, gyr, ones, n, DIST_CENTERS_KM))
    gaps = (ts.tmin[nxt + 1] - ts.tmin[nxt]).astype(np.float64)
    blocks.append(cfg.w_gap * _soft_hist(tr[nxt], gaps, hw, n, GAP_CENTERS_MIN))

    if cfg.w_cbigram > 0:
        ca, cb = catalog.cat[ts.poi[nxt]].astype(np.int64), catalog.cat[ts.poi[nxt + 1]].astype(np.int64)
        key = (ca * 1000003 + cb * 9176) % cfg.cbigram_buckets
        blocks.append(cfg.w_cbigram * _hist(tr[nxt], key, hw, n, cfg.cbigram_buckets))

    if cfg.w_ret > 0:
        rep, ret = return_flags(ts)
        f_rep = np.bincount(tr[nxt], weights=hw * rep[nxt + 1], minlength=n)
        f_ret = np.bincount(tr[nxt], weights=hw * ret[nxt + 1], minlength=n)
        blocks.append(cfg.w_ret * np.stack([f_rep, f_ret, f_rep > 0, f_ret > 0], 1).astype(np.float32))

    if cfg.popular is not None and len(cfg.popular) and cfg.w_venue > 0:
        col = np.full(len(catalog), -1, dtype=np.int64)
        col[np.asarray(cfg.popular)] = np.arange(len(cfg.popular))
        hit = col[ts.poi] >= 0
        blocks.append(cfg.w_venue * _hist(tr[hit], col[ts.poi[hit]], w[hit], n, len(cfg.popular)))

    k, nr = cfg.seq_k, catalog.n_roots
    seq = np.zeros((n, k, 4 + nr), np.float32)
    pos = ts.pos
    m = pos < k
    ang = 2 * np.pi * hours[m] / 24
    seq[tr[m], pos[m], 0] = xy[m, 0] / catalog.width_km
    seq[tr[m], pos[m], 1] = xy[m, 1] / catalog.height_km
    seq[tr[m], pos[m], 2] = np.sin(ang)
    seq[tr[m], pos[m], 3] = np.cos(ang)
    seq[tr[m], pos[m], 4 + root[m]] = 1.0
    blocks.append(cfg.w_seq * seq.reshape(n, -1) / np.sqrt(k))

    return np.concatenate(blocks, axis=1).astype(np.float32)


def compute_device(min_free_gb: float = 6.0) -> str:
    import torch
    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        if free > min_free_gb * 2**30:
            return "cuda"
    return "cpu"


def nearest_k(query: np.ndarray, base: np.ndarray, k: int, chunk: int = 8192) -> np.ndarray:
    import torch
    dev = compute_device()
    b = torch.from_numpy(base).to(dev)
    bn = (b * b).sum(1)
    out = []
    for s in range(0, len(query), chunk):
        q = torch.from_numpy(query[s:s + chunk]).to(dev)
        out.append(torch.topk(2.0 * q @ b.T - bn[None, :], k, dim=1).indices.cpu().numpy())
    return np.concatenate(out)


def kmeans(x: np.ndarray, k: int, iters: int = 15, seed: int = 0) -> np.ndarray:
    import torch
    dev = compute_device()
    g = torch.Generator(device="cpu").manual_seed(seed)
    xt = torch.from_numpy(x).to(dev)
    cent = xt[torch.randperm(len(x), generator=g)[:k].to(dev)].clone()
    xn = (xt * xt).sum(1, keepdim=True)
    for _ in range(iters):
        lab = (xn - 2.0 * xt @ cent.T + (cent * cent).sum(1)[None, :]).argmin(1)
        sums = torch.zeros_like(cent).index_add_(0, lab, xt)
        cnt = torch.bincount(lab, minlength=k).float()[:, None]
        cent = torch.where(cnt > 0, sums / cnt.clamp(min=1), cent)
    return lab.cpu().numpy()


def nearest_index(query: np.ndarray, base: np.ndarray, chunk: int = 8192) -> np.ndarray:
    import torch
    dev = compute_device()
    b = torch.from_numpy(base).to(dev)
    bn = (b * b).sum(1)
    out = []
    for s in range(0, len(query), chunk):
        q = torch.from_numpy(query[s:s + chunk]).to(dev)
        out.append((bn[None, :] - 2.0 * q @ b.T).argmin(1).cpu().numpy())
    return np.concatenate(out)
