"""Pinned Hugging Face revisions of the models the code loads by Hub id (the commits the README downloads).

`hf download <id> --revision <commit>` stores only snapshots/<commit> in the cache and no refs/main, so a load that
asks for the default revision finds nothing offline (HF_HUB_OFFLINE=1). Every loader therefore passes
revision=revision(<id>): the pinned commit for these ids, None (the library default) for a local directory or another
model id."""
from __future__ import annotations

import os

PINNED = {
    "Qwen/Qwen2.5-0.5B-Instruct": "7ae557604adf67be50417f59c2c2f167def9a775",
    "Qwen/Qwen2.5-7B-Instruct": "a09a35458c702b33eeacc393d103063234e8bc28",
    "sentence-transformers/all-MiniLM-L6-v2": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    "Qwen/Qwen3-Embedding-0.6B": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
}


def revision(model) -> str | None:
    """Pinned commit of a Hub model id; None for a local model directory or an id that is not pinned."""
    name = str(model)
    if os.path.isdir(name):
        return None
    return PINNED.get(name)
