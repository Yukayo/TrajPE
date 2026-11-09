from __future__ import annotations

import copy
import math
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import MAX_LEN, Catalog, Trails
from .markov import DIST_EDGES_KM


@dataclass
class TransformerConfig:
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 512
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 0.05
    batch_trails: int = 256
    max_epochs: int = 50
    patience: int = 5
    fine_eval_epochs: int = 2
    fine_eval_per_epoch: int = 4
    warmup_steps: int = 200
    grad_clip: float = 1.0
    label_smoothing: float = 0.0
    id_dropout: float = 0.2
    lr_schedule: str = "plateau"
    time_budget_min: float = 10.0
    logit_budget: int = 2**27


class GeoEncoder(nn.Module):

    def __init__(self, d: int, n_freq: int = 11, max_wavelength_km: float = 100.0):
        super().__init__()
        freq = 2 * math.pi * 2.0 ** torch.arange(n_freq) / max_wavelength_km
        self.register_buffer("freq", freq, persistent=False)
        self.mlp = nn.Sequential(nn.Linear(4 * n_freq, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        ang = (xy[..., None] * self.freq).flatten(-2)
        return self.mlp(torch.cat([ang.sin(), ang.cos()], -1))


class Block(nn.Module):

    def __init__(self, d: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.n_heads, self.p_attn = n_heads, dropout
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, L, 3, self.n_heads, d // self.n_heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.p_attn if self.training else 0.0)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, L, d)))
        return x + self.drop(self.ff(self.ln2(x)))


class _BinnedBias(torch.autograd.Function):
    LANES = 128

    @staticmethod
    def forward(ctx, table: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(bins)
        ctx.n_bins = table.shape[1]
        return table.gather(1, bins)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        (bins,) = ctx.saved_tensors
        lanes = _BinnedBias.LANES
        lane = torch.arange(bins.shape[1], device=bins.device) % lanes
        acc = torch.zeros(grad.shape[0], ctx.n_bins * lanes, device=grad.device, dtype=grad.dtype)
        acc.scatter_add_(1, bins * lanes + lane, grad)
        return acc.view(grad.shape[0], ctx.n_bins, lanes).sum(-1), None


class NextPOITransformer(nn.Module):
    def __init__(self, catalog: Catalog, vocab: np.ndarray, cfg: TransformerConfig):
        super().__init__()
        C, d, Vt = catalog.n_cats, cfg.d_model, len(vocab)
        rows = torch.as_tensor(vocab)
        to_cand = torch.full((catalog.n_venues,), Vt, dtype=torch.long)
        to_cand[rows] = torch.arange(Vt)
        for name, val in [("venue_cat", torch.as_tensor(catalog.cat_idx)), ("venue_xy", torch.as_tensor(catalog.xy)),
                          ("to_cand", to_cand), ("cand_rows", rows),
                          ("dist_edges", torch.as_tensor(DIST_EDGES_KM, dtype=torch.float32))]:
            self.register_buffer(name, val, persistent=False)
        self.n_cand = Vt
        self.id_emb = nn.Embedding(Vt + 1, d, padding_idx=Vt)
        self.cat_emb = nn.Embedding(C, d)
        self.geo = GeoEncoder(d)
        self.hour_emb, self.dow_emb, self.pos_emb = nn.Embedding(24, d), nn.Embedding(7, d), nn.Embedding(MAX_LEN, d)
        self.emb_drop = nn.Dropout(cfg.dropout)
        self.id_dropout = cfg.id_dropout
        self.blocks = nn.ModuleList(Block(d, cfg.n_heads, cfg.d_ff, cfg.dropout) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(d)
        self.to_query = nn.Linear(d, d)
        self.to_dist = nn.Linear(d, len(DIST_EDGES_KM) + 1)
        self.cand_bias = nn.Parameter(torch.zeros(Vt))
        nn.init.zeros_(self.id_emb.weight)
        for emb in (self.cat_emb, self.hour_emb, self.dow_emb, self.pos_emb):
            nn.init.normal_(emb.weight, std=0.02)
        nn.init.zeros_(self.to_dist.weight)
        nn.init.zeros_(self.to_dist.bias)

    def content(self, rows: torch.Tensor) -> torch.Tensor:
        return self.cat_emb(self.venue_cat[rows]) + self.geo(self.venue_xy[rows])

    def candidate_table(self) -> torch.Tensor:
        return self.id_emb.weight[:-1] + self.content(self.cand_rows)

    def encode(self, venue, hour, dow) -> torch.Tensor:
        rows = venue.clamp(min=0)
        ids = torch.where(venue >= 0, self.to_cand[rows], self.n_cand)
        if self.training and self.id_dropout > 0:
            ids = torch.where(torch.rand(ids.shape, device=ids.device) < self.id_dropout, self.n_cand, ids)
        x = self.id_emb(ids) + self.content(rows) + self.hour_emb(hour) + self.dow_emb(dow)
        x = self.emb_drop(x + self.pos_emb.weight[: venue.shape[1]])
        for blk in self.blocks:
            x = blk(x)
        return self.ln_f(x)

    def logits(self, h: torch.Tensor, last: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            xy, cxy = self.venue_xy[last], self.venue_xy[self.cand_rows]
            dist_bin = torch.bucketize(torch.hypot(xy[:, :1] - cxy[:, 0], xy[:, 1:] - cxy[:, 1]), self.dist_edges)
        out = torch.addmm(self.cand_bias, self.to_query(h), table.t())
        return out + _BinnedBias.apply(self.to_dist(h), dist_bin)


class DeviceTrails:

    def __init__(self, trails: Trails, device: torch.device):
        self.venue = torch.as_tensor(trails.venue, device=device)
        self.hour = torch.as_tensor(trails.hour, device=device)
        self.dow = torch.as_tensor(trails.dow, device=device)
        self.length = torch.as_tensor(trails.length, device=device)

    def __len__(self) -> int:
        return len(self.length)

    def batch(self, idx: torch.Tensor):
        L = int(self.length[idx].max())
        return self.venue[idx, :L], self.hour[idx, :L], self.dow[idx, :L], self.length[idx]


def _prefix_states(model, venue, hour, dow, length):
    h = model.encode(venue[:, :-1], hour[:, :-1], dow[:, :-1])
    k = torch.arange(1, venue.shape[1], device=venue.device).expand(venue.shape[0], -1)
    m = k < length[:, None]
    return h[m], venue[:, :-1][m], venue[:, 1:][m], k[m]


@contextmanager
def _exact_matmul():
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


TARGET_TYPES = ("novel", "revisit", "repeat")


def target_groups(venue: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
    L = venue.shape[1]
    eq = venue[:, :, None] == venue[:, None, :]
    before = torch.ones(L, L, dtype=torch.bool, device=venue.device).tril(-1)
    in_prefix = (eq & before).any(2)[:, 1:]
    repeat = venue[:, 1:] == venue[:, :-1]
    g = torch.where(repeat, 2, torch.where(in_prefix, 1, 0))
    k = torch.arange(1, L, device=venue.device).expand(venue.shape[0], -1)
    return g[k < length[:, None]]


@torch.no_grad()
def iter_scores(model: NextPOITransformer, trails: DeviceTrails, logit_budget: int = 2**27, batch_trails: int = 1024,
                with_types: bool = False):
    model.eval()
    rows = int(min(4096, max(1, logit_budget // model.n_cand)))
    with _exact_matmul():
        table = model.candidate_table()
        for i in range(0, len(trails), batch_trails):
            idx = torch.arange(i, min(i + batch_trails, len(trails)), device=table.device)
            venue, hour, dow, length = trails.batch(idx)
            h, last, tgt, k = _prefix_states(model, venue, hour, dow, length)
            types = target_groups(venue, length) if with_types else None
            for j in range(0, len(tgt), rows):
                sl = slice(j, j + rows)
                out = (model.logits(h[sl], last[sl], table).log_softmax(1), tgt[sl], k[sl])
                yield out + (types[sl],) if with_types else out


def train_transformer(catalog: Catalog, vocab: np.ndarray, train: Trails, stop_set: Trails | None,
                      cfg: TransformerConfig, device: torch.device, seed: int, evaluate, log=print):
    torch.manual_seed(seed)
    model = NextPOITransformer(catalog, vocab, cfg).to(device)
    tr = DeviceTrails(train, device)
    st = DeviceTrails(stop_set, device) if stop_set is not None else None
    if (model.to_cand[tr.venue[tr.venue >= 0]] == model.n_cand).any():
        raise ValueError("training venues must lie inside the vocabulary")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": [p for p in params if p.ndim >= 2], "weight_decay": cfg.weight_decay},
                             {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0}],
                            lr=cfg.lr, betas=(0.9, 0.98))
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    max_targets = max(1, cfg.logit_budget // model.n_cand)
    steps_per_epoch = math.ceil(len(tr) / cfg.batch_trails)
    fine = set()
    if st is not None and cfg.fine_eval_per_epoch > 1:
        for e in range(cfg.fine_eval_epochs):
            fine |= {e * steps_per_epoch + round(k * steps_per_epoch / cfg.fine_eval_per_epoch)
                     for k in range(1, cfg.fine_eval_per_epoch)}
    best = {"mrr": -1.0, "epoch": 0, "state": None}
    best_end = -1.0
    history, step, bad, t0 = [], 0, 0, time.time()
    stop_reason, lr_scale = "max_epochs", 1.0
    total_steps = cfg.max_epochs * math.ceil(len(tr) / cfg.batch_trails)
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        perm = torch.randperm(len(tr), device=device, generator=gen)
        loss_sum = torch.zeros((), device=device)
        n_sum = 0
        for i in range(0, len(tr), cfg.batch_trails):
            idx = perm[i:i + cfg.batch_trails]
            n_tgt = int((tr.length[idx] - 1).sum())
            if cfg.lr_schedule == "cosine":
                lr_scale = 0.5 * (1 + math.cos(math.pi * min(1.0, step / total_steps)))
            for g in opt.param_groups:
                g["lr"] = cfg.lr * lr_scale * min(1.0, (step + 1) / cfg.warmup_steps)
            opt.zero_grad(set_to_none=True)
            for part in idx.tensor_split(min(len(idx), math.ceil(n_tgt / max_targets))):
                h, last, tgt, _ = _prefix_states(model, *tr.batch(part))
                logits = model.logits(h, last, model.candidate_table())
                loss = F.cross_entropy(logits, model.to_cand[tgt], reduction="sum",
                                       label_smoothing=cfg.label_smoothing) / n_tgt
                loss.backward()
                loss_sum += loss.detach() * n_tgt
            nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            step += 1
            n_sum += n_tgt
            if step in fine:
                m = evaluate(iter_scores(model, st, cfg.logit_budget))
                prog = round(step / steps_per_epoch, 3)
                history.append({"epoch": prog, "checkpoint": "within-epoch", "stop_mrr": round(m["mrr"], 5)})
                if m["mrr"] > best["mrr"]:
                    best = {"mrr": m["mrr"], "epoch": prog, "state": copy.deepcopy(model.state_dict())}
                model.train()
        rec = {"epoch": epoch, "train_loss": round(float(loss_sum) / n_sum, 4)}
        if st is not None:
            m = evaluate(iter_scores(model, st, cfg.logit_budget))
            rec.update({f"stop_{k}": round(v, 5) for k, v in m.items() if isinstance(v, float)})
            if m["mrr"] > best["mrr"]:
                best = {"mrr": m["mrr"], "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
            if m["mrr"] > best_end:
                best_end = m["mrr"]
                bad = 0
            else:
                bad += 1
                if cfg.lr_schedule == "plateau":
                    lr_scale *= 0.5
        rec["elapsed_s"] = round(time.time() - t0, 1)
        history.append(rec)
        log("  " + "  ".join(f"{k}={v}" for k, v in rec.items()))
        if st is not None and bad >= cfg.patience:
            stop_reason = "early_stopping"
            break
        if time.time() - t0 > 60 * cfg.time_budget_min:
            stop_reason = "time_budget"
            log(f"  time budget of {cfg.time_budget_min} min reached")
            break
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    info = {"config": asdict(cfg), "epochs_run": epoch, "stop_reason": stop_reason,
            "selected_epoch": best["epoch"] if st is not None else epoch,
            "n_params": sum(p.numel() for p in params), "history": history}
    return model, info
