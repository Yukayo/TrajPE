from __future__ import annotations

import numpy as np
import torch

from .data import Catalog, Trails

DIST_EDGES_KM = np.geomspace(0.025, 80.0, 35)


def time_slot(hour, dow, n_slots: int):
    if n_slots == 1:
        return hour * 0
    return hour // 3 + 8 * (dow >= 5)


class MarkovBackoff:
    def __init__(self, catalog: Catalog, vocab: np.ndarray, device: torch.device, alpha: float = 2.0,
                 beta: float = 30.0, gamma: float = 200.0, n_slots: int = 16, kernel_iters: int = 10,
                 kernel_events: int = 50_000, seed: int = 0):
        assert n_slots in (1, 16)
        self.cat, self.device = catalog, device
        self.alpha, self.beta, self.gamma, self.n_slots = alpha, beta, gamma, n_slots
        self.kernel_iters, self.kernel_events, self.seed = kernel_iters, kernel_events, seed
        self.venue_xy = torch.as_tensor(catalog.xy, device=device)
        self.venue_cat = torch.as_tensor(catalog.cat_idx, device=device)
        rows = torch.as_tensor(vocab, device=device)
        self.n_cand = len(vocab)
        self.to_cand = torch.full((catalog.n_venues,), -1, dtype=torch.long, device=device)
        self.to_cand[rows] = torch.arange(self.n_cand, device=device)
        self.cand_xy, self.cand_cat = self.venue_xy[rows], self.venue_cat[rows]
        self.edges = torch.as_tensor(DIST_EDGES_KM, dtype=torch.float32, device=device)

    def hparams(self) -> dict:
        return {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma, "n_slots": self.n_slots,
                "kernel_iters": self.kernel_iters, "kernel_events": self.kernel_events,
                "n_dist_bins": len(DIST_EDGES_KM) + 1}

    def fit(self, trails: Trails) -> "MarkovBackoff":
        Vt, C, S, dev = self.n_cand, self.cat.n_cats, self.n_slots, self.device
        tr = trails.transitions()
        to_cand = self.to_cand.cpu().numpy()
        src, dst = to_cand[tr["src"]], to_cand[tr["dst"]]
        if (src < 0).any() or (dst < 0).any():
            raise ValueError("training transitions must lie inside the vocabulary built from the same trails")

        visits = np.bincount(to_cand[trails.checkin_venues()], minlength=Vt)
        self.pop = torch.as_tensor(visits + self.alpha, dtype=torch.float32, device=dev)

        uniq, cnt = np.unique(src * Vt + dst, return_counts=True)
        rows = uniq // Vt
        self.crow = torch.as_tensor(np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=Vt))]), device=dev)
        self.col = torch.as_tensor(uniq % Vt, device=dev)
        self.val = torch.as_tensor(cnt, dtype=torch.float32, device=dev)
        self.n_out = torch.as_tensor(np.bincount(src, minlength=Vt), dtype=torch.float32, device=dev)

        cu, cv = self.cat.cat_idx[tr["src"]], self.cat.cat_idx[tr["dst"]]
        s = time_slot(tr["hour"], tr["dow"], S)
        n3 = np.bincount((cu * S + s) * C + cv, minlength=C * S * C).reshape(C, S, C).astype(np.float64)
        n2, n1 = n3.sum(1), n3.sum(0)
        n0 = n1.sum(0)
        g = self.gamma
        p0 = (n0 + 1.0) / (n0.sum() + C)
        p1 = (n1 + g * p0) / (n1.sum(1, keepdims=True) + g)
        p2 = (n2[:, None, :] + g * p1[None]) / (n2.sum(1)[:, None, None] + g)
        p3 = (n3 + g * p2) / (n3.sum(2, keepdims=True) + g)
        p3 *= np.bincount(self.cand_cat.cpu().numpy(), minlength=C) > 0
        p3 /= p3.sum(2, keepdims=True)
        self.pcat = torch.as_tensor(p3.reshape(C * S, C), dtype=torch.float32, device=dev)

        self.kernel = self._fit_kernel(tr["src"], dst)
        return self

    def _dist_bins(self, src_rows: torch.Tensor) -> torch.Tensor:
        xy = self.venue_xy[src_rows]
        d = torch.hypot(xy[:, :1] - self.cand_xy[:, 0], xy[:, 1:] - self.cand_xy[:, 1])
        return torch.bucketize(d, self.edges)

    @torch.no_grad()
    def _fit_kernel(self, src_rows: np.ndarray, dst: np.ndarray, max_elems: int = 2**27) -> torch.Tensor:
        chunk = int(min(1024, max(1, max_elems // self.n_cand)))
        if len(src_rows) > self.kernel_events:
            sel = np.random.default_rng(self.seed).choice(len(src_rows), self.kernel_events, replace=False)
            src_rows, dst = src_rows[sel], dst[sel]
        src_rows, dst = torch.as_tensor(src_rows, device=self.device), torch.as_tensor(dst, device=self.device)
        nb = len(self.edges) + 1
        d_obs = torch.hypot(*(self.venue_xy[src_rows] - self.cand_xy[dst]).unbind(1))
        obs = torch.bincount(torch.bucketize(d_obs, self.edges), minlength=nb).float()
        log_k = torch.zeros(nb, device=self.device)
        for _ in range(self.kernel_iters):
            expected = torch.zeros(nb, device=self.device)
            for i in range(0, len(dst), chunk):
                b = self._dist_bins(src_rows[i:i + chunk])
                w = self.pop * log_k.exp()[b] * (self.cand_cat == self.cand_cat[dst[i:i + chunk]][:, None])
                expected += torch.bincount(b.flatten(), weights=(w / w.sum(1, keepdim=True)).flatten(), minlength=nb)
            log_k += torch.log((obs + 1.0) / (expected + 1.0))
            log_k = (log_k - log_k.max()).clamp(min=-30.0)
        return log_k.exp()

    def _count_rows(self, src: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(len(src), self.n_cand, device=self.device)
        start = self.crow[src.clamp(min=0)]
        lens = torch.where(src >= 0, self.crow[src.clamp(min=0) + 1] - start, 0)
        if int(lens.sum()) == 0:
            return out
        row = torch.repeat_interleave(torch.arange(len(src), device=self.device), lens)
        first = torch.repeat_interleave(torch.cumsum(lens, 0) - lens, lens)
        idx = torch.repeat_interleave(start, lens) + torch.arange(len(row), device=self.device) - first
        out[row, self.col[idx]] = self.val[idx]
        return out

    @torch.no_grad()
    def score(self, src_rows: torch.Tensor, hour: torch.Tensor, dow: torch.Tensor) -> torch.Tensor:
        C, S = self.cat.n_cats, self.n_slots
        w = self.pop * self.kernel[self._dist_bins(src_rows)]
        cc = self.cand_cat.expand(len(src_rows), -1)
        z = torch.zeros(len(src_rows), C, device=self.device).scatter_add_(1, cc, w)
        pc = self.pcat[self.venue_cat[src_rows] * S + time_slot(hour, dow, S)].gather(1, cc)
        src = self.to_cand[src_rows]
        n_u = torch.where(src >= 0, self.n_out[src.clamp(min=0)], 0.0)
        return (self._count_rows(src) + self.beta * pc * w / z.gather(1, cc)) / (n_u[:, None] + self.beta)

    def iter_scores(self, trails: Trails, max_elems: int = 2**27):
        tr = {k: torch.as_tensor(v, device=self.device) for k, v in trails.transitions().items()}
        chunk = int(min(4096, max(1, max_elems // self.n_cand)))
        for i in range(0, len(tr["dst"]), chunk):
            sl = slice(i, i + chunk)
            yield self.score(tr["src"][sl], tr["hour"][sl], tr["dow"][sl]).log(), tr["dst"][sl], tr["k"][sl]
