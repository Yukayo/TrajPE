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
import collections
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", f"{_TP_WORK}/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")

TESTS = Path(__file__).resolve().parent
CTCL = TESTS.parent
sys.path.insert(0, str(CTCL))
import ctcl_conf as C

import numpy as np


def train_nonprivate(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import topics as TP
    import train_ctcl as TR
    if C.DATA == C.DEFAULT_DATA:
        raise SystemExit("set TRAJPE_DATA to a synthetic data dir: this diagnostic never reads private trails")
    dev = torch.device("cuda", 0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(C.MODEL, revision=C.MODEL_COMMIT)
    table = TP.load_table(C.table_path(a.dataset))
    prompts, resp, taus = TR.load_examples(a.dataset, tok, table)
    data = TR.DeviceExamples(prompts, resp, taus, dev)
    model = AutoModelForCausalLM.from_pretrained(C.MODEL, revision=C.MODEL_COMMIT, dtype=torch.float32,
                                                 attn_implementation="sdpa").to(dev)
    model.config.use_cache = False
    model.requires_grad_(True)
    model.train()
    core, head, params = TR.model_parts(model, "full")
    cfg = C.CONFIGS["code"]
    opt = torch.optim.AdamW(params, lr=0.0, betas=cfg["betas"], eps=cfg["adam_eps"], weight_decay=cfg["weight_decay"],
                            foreach=True)
    cs = TR.ClippedSum(params, cfg["clip"])
    rng = np.random.default_rng(0)
    gen = torch.Generator(device=dev).manual_seed(7919)
    q = C.sample_rate("code")
    t0, losses = time.time(), []
    for s in range(a.steps):
        sel = np.flatnonzero(rng.random(data.n) < q)
        cs.zero()
        for i in sel:
            loss = TR.example_loss(core, head, *data.get(int(i)))
            cs.add(loss)
            losses.append(float(loss.detach()))
        TR.noisy_update(opt, cs, 0.0, float(cfg["batch"]), C.lr_at(s, "code"), gen)
        if s % 50 == 0 or s == a.steps - 1:
            print(f"step {s + 1}/{a.steps} mean loss (last 200 examples) {np.mean(losses[-200:]):.3f} "
                  f"{time.time() - t0:.0f}s", flush=True)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out / "model", safe_serialization=True)
    (out / "train_nonprivate.json").write_text(json.dumps({
        "steps": a.steps, "n_trails": data.n, "q": q, "sigma": 0.0, "first_losses": losses[:20],
        "mean_loss_last_200": float(np.mean(losses[-200:])), "sec": round(time.time() - t0, 1)}, indent=1))


def sample(a):
    import pandas as pd
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    import sample_ctcl as SC
    import topics as TP
    tok = AutoTokenizer.from_pretrained(C.MODEL, revision=C.MODEL_COMMIT)
    hist = json.loads(Path(a.hist).read_text())
    table = TP.load_table(C.table_path(a.dataset))
    quota = np.asarray(hist["quota"])
    top = [int(k) for k in np.argsort(-quota, kind="stable")[:a.topics]]
    prompts = {k: C.prompt_ids(tok, a.dataset, table["rows"][k]["condition"]) for k in top}
    if a.model_dir == "base":
        model, extra = C.MODEL, {"revision": C.MODEL_COMMIT, "tokenizer_revision": C.MODEL_COMMIT}
    else:
        model, extra = str(Path(a.model_dir).resolve()), {"tokenizer_revision": C.MODEL_COMMIT}
    llm = LLM(model=model, tokenizer=C.MODEL, dtype="bfloat16", seed=0, max_model_len=C.MAX_MODEL_LEN,
              gpu_memory_utilization=0.05, kv_cache_memory_bytes=int(C.KV_GB * 2**30), enable_prefix_caching=True,
              generation_config="vllm", max_num_seqs=C.MAX_NUM_SEQS, **extra)
    end_ids = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    sp = SamplingParams(n=1, temperature=C.TEMPERATURE, top_p=C.TOP_P, top_k=C.TOP_K, min_p=C.MIN_P,
                        repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0,
                        max_tokens=C.MAX_RESP_TOKENS, stop_token_ids=end_ids, skip_special_tokens=False)
    generate = SC.generator_vllm(llm, sp, prompts, None)
    cats = set(pd.read_parquet(C.DATA / a.dataset / "pois.parquet", columns=["category"])["category"].astype(str))
    t0 = time.time()
    outs = generate([(k, a.n) for k in top])
    reasons, lens, dropped, ex = collections.Counter(), [], 0, []
    for j, (k, text, fin) in enumerate(outs):
        res, why, nd = C.textfmt.clean(text, cats, bool(fin))
        reasons[why] += 1
        if res is not None:
            lens.append(res.count("\n"))
            dropped += nd
        if j % (a.n * max(1, a.topics // 8)) == 0 and len(ex) < 8:
            ex.append({"topic": k, "finished": fin, "why": why, "text": text[:700]})
    r = {"model": a.model_dir, "n": len(outs), "valid_rate": reasons["ok"] / len(outs), "reasons": dict(reasons),
         "mean_len_kept": float(np.mean(lens)) if lens else None, "dropped_lines_in_kept": dropped,
         "gen_tokens": generate.tokens, "sec": round(time.time() - t0, 1), "examples": ex}
    Path(a.out).write_text(json.dumps(r, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in r.items() if k != "examples"}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["train_nonprivate", "sample"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--model_dir", default="")
    ap.add_argument("--hist", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--topics", type=int, default=20)
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    C.check_dataset(a.dataset)
    if a.cmd == "train_nonprivate":
        train_nonprivate(a)
    else:
        sample(a)


if __name__ == "__main__":
    main()
