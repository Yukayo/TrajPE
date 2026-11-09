from __future__ import annotations
import os as _tp_os
def _tp_root():
    p = _tp_os.path.dirname(_tp_os.path.abspath(__file__))
    while not _tp_os.path.isfile(_tp_os.path.join(p, "trajpe", "__init__.py")) and _tp_os.path.dirname(p) != p:
        p = _tp_os.path.dirname(p)
    return p
_TP_ROOT = _tp_os.environ.get("TRAJPE_ROOT", _tp_root())
_TP_WORK = _tp_os.environ.get("TRAJPE_WORK", _tp_os.path.join(_TP_ROOT, "work"))
_TP_PY = _tp_os.environ.get("TRAJPE_PYTHON", "python")
_TP_PY_DPLORA = _tp_os.environ.get("TRAJPE_PYTHON_DPLORA", _TP_PY)
_TP_PY_AIMSEQ = _tp_os.environ.get("TRAJPE_PYTHON_AIMSEQ", _TP_PY)
_TP_PY_GRID = _tp_os.environ.get("TRAJPE_PYTHON_GRID", _TP_PY)

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", f"{_TP_WORK}/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))
import ctcl_testlib as TL

C = TL.C
sys.path.insert(0, str(TL.CTCL))
import topics as TP
import train_ctcl as TR

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_time", type=int, default=150)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = torch.device("cuda", 0)
    tok = AutoTokenizer.from_pretrained(C.MODEL, revision=C.MODEL_COMMIT)
    end = tok.convert_tokens_to_ids("<|im_end|>")
    table = TP.load_table(C.table_path("bandung"))
    df = TL.synthetic_trails(TL.public_catalog("bandung"), 600, seed=21)
    texts = C.textfmt.serialize_frame(df, C.COORD_DECIMALS)
    rng = np.random.default_rng(0)
    ks = rng.integers(0, C.K_TOPICS, len(texts))
    ex = [(torch.tensor(C.prompt_ids(tok, "bandung", table["rows"][int(k)]["condition"]), device=dev),
           torch.tensor((tok(t, add_special_tokens=False)["input_ids"] + [end])[:C.MAX_RESP_TOKENS], device=dev))
          for t, k in zip(texts, ks)]
    res = {"model": C.MODEL, "commit": C.MODEL_COMMIT, "gpu": torch.cuda.get_device_name(0)}
    model = AutoModelForCausalLM.from_pretrained(C.MODEL, revision=C.MODEL_COMMIT, dtype=torch.float32,
                                                 attn_implementation="sdpa").to(dev)
    model.config.use_cache = False
    model.requires_grad_(True)
    model.train()
    core, head, params = TR.model_parts(model, "full")
    names = [n for n, _ in model.named_parameters()]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    worst, worst_name = 0.0, None
    for i in (0, 1):
        prompt, resp = ex[i]
        cs = TR.ClippedSum(params, 1e9)
        cs.add(TR.example_loss(core, head, prompt, resp))
        for p in params:
            p.grad = None
        ids = torch.cat([prompt, resp]).unsqueeze(0)
        labels = torch.cat([torch.full_like(prompt, -100), resp]).unsqueeze(0)
        loss = model(input_ids=ids, labels=labels, use_cache=False).loss
        ref = torch.autograd.grad(loss, params)
        for n, x, y in zip(names, cs.acc, ref):
            e = float((x - y).norm() / (y.norm() + 1e-30))
            if e > worst:
                worst, worst_name = e, n
        del cs, ref
    res["exact_worst_rel_err"], res["exact_worst_param"] = worst, worst_name
    res["exact_ok"] = worst < 1e-3
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cfg = C.CONFIGS["code"]
    opt = torch.optim.AdamW(params, lr=0.0, betas=cfg["betas"], eps=cfg["adam_eps"], weight_decay=cfg["weight_decay"],
                            foreach=True)
    cs = TR.ClippedSum(params, 1.0)
    gen = torch.Generator(device=dev).manual_seed(7919)
    for i in range(10):
        cs.add(TR.example_loss(core, head, *ex[i]))
    TR.noisy_update(opt, cs, 0.778, 128.0, 1e-4, gen)
    cs.zero()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    n_tok = 0
    for i in range(a.n_time):
        p, r = ex[10 + i % (len(ex) - 10)]
        n_tok += len(p) + len(r)
        cs.add(TR.example_loss(core, head, p, r))
    torch.cuda.synchronize()
    per_ex = (time.time() - t0) / a.n_time
    t1 = time.time()
    for _ in range(5):
        TR.noisy_update(opt, cs, 0.778, 128.0, 1e-4, gen)
        assert TR.all_finite(params)
    torch.cuda.synchronize()
    res.update({"sec_per_example": round(per_ex, 4), "tokens_per_example": round(n_tok / a.n_time, 1),
                "sec_per_noisy_step": round((time.time() - t1) / 5, 4),
                "max_mem_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                "max_mem_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
                "est_jakarta_train_h": round((per_ex * 128 * 1166 + 1166 * (time.time() - t1) / 5) / 3600, 2)})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)
    if not res["exact_ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
