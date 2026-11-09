from __future__ import annotations

import numpy as np

from .embed import EmbedConfig, embed
from .trails import TrailSet


def _seq_keys(ts: TrailSet) -> set:
    return {tuple(p) for p, _ in ts.to_lists()}


def _min_dist(query: np.ndarray, base: np.ndarray, chunk: int = 8192) -> np.ndarray:
    import torch

    from .embed import compute_device
    dev = compute_device()
    b = torch.from_numpy(base).to(dev)
    bn = (b * b).sum(1)
    out = []
    for s in range(0, len(query), chunk):
        q = torch.from_numpy(query[s:s + chunk]).to(dev)
        d = (bn[None, :] - 2.0 * q @ b.T).min(1).values + (q * q).sum(1)
        out.append(torch.sqrt(torch.clamp(d, min=0)).cpu().numpy())
    return np.concatenate(out)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    scores = np.concatenate([pos, neg])
    ranks = scores.argsort().argsort() + 1.0
    n_pos, n_neg = len(pos), len(neg)
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def evaluate(syn: TrailSet, train: TrailSet, test: TrailSet, catalog, n: int = 10000, seed: int = 0,
             ecfg: EmbedConfig = EmbedConfig()) -> dict:
    rng = np.random.default_rng(seed)
    m_all = min(len(train), len(test))
    syn_keys = [tuple(p) for p, _ in syn.to_lists()]
    tr_keys = _seq_keys(train.subset(rng.choice(len(train), m_all, replace=False)))
    te_keys = _seq_keys(test.subset(rng.choice(len(test), m_all, replace=False)))
    copy_train = float(np.mean([k in tr_keys for k in syn_keys]))
    copy_test = float(np.mean([k in te_keys for k in syn_keys]))

    m = min(n, m_all)
    tr = train.subset(rng.choice(len(train), m, replace=False))
    te = test.subset(rng.choice(len(test), m, replace=False))
    e_tr, e_te = embed(tr, catalog, ecfg), embed(te, catalog, ecfg)
    e_all = embed(syn, catalog, ecfg)
    e_sy = e_all[rng.choice(len(syn), min(n, len(syn)), replace=False)]
    d_tr, d_te = _min_dist(e_sy, e_tr), _min_dist(e_sy, e_te)
    dcr_share = float(np.mean(d_tr < d_te) + 0.5 * np.mean(d_tr == d_te))
    mia = auc(-_min_dist(e_tr, e_all), -_min_dist(e_te, e_all))
    return {"copy_rate_train": copy_train, "copy_rate_test": copy_test, "dcr_train_share": dcr_share,
            "mia_auc": mia, "dcr_median_train": float(np.median(d_tr)), "dcr_median_test": float(np.median(d_te))}
