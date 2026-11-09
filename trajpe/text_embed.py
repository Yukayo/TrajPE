from __future__ import annotations

import numpy as np

from .hf_revisions import revision as hf_revision
from .paths import HF_HOME
from .trails import TrailSet

_CACHE = {}
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def serialize(ts: TrailSet, catalog) -> list[str]:
    out = []
    for p, t in ts.to_lists():
        parts = [f"{DAYS[int(m // 1440) % 7]} {(m % 1440) // 60:02d}:{m % 60:02d} {catalog.categories[catalog.cat[v]]} "
                 f"({catalog.lat[v]:.3f},{catalog.lon[v]:.3f})" for v, m in zip(p, t)]
        out.append("; ".join(parts))
    return out


def text_embed(texts: list[str], model_name: str, batch: int = 512) -> np.ndarray:
    import os

    import torch
    from transformers import AutoModel, AutoTokenizer

    from .embed import compute_device
    os.environ.setdefault("HF_HOME", str(HF_HOME))
    if model_name not in _CACHE:
        tok = AutoTokenizer.from_pretrained(model_name, revision=hf_revision(model_name))
        mdl = AutoModel.from_pretrained(model_name, revision=hf_revision(model_name)).eval()
        _CACHE[model_name] = (tok, mdl)
    tok, mdl = _CACHE[model_name]
    dev = compute_device(2.0)
    mdl = mdl.to(dev)
    outs = []
    with torch.no_grad():
        for s in range(0, len(texts), batch):
            enc = tok(texts[s:s + batch], padding=True, truncation=True, max_length=256, return_tensors="pt").to(dev)
            h = mdl(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            e = (h * m).sum(1) / m.sum(1).clamp(min=1)
            outs.append(torch.nn.functional.normalize(e, dim=1).cpu().numpy())
    return np.concatenate(outs).astype(np.float32)
