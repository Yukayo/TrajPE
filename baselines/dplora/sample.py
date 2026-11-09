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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE.parents[1]))
from trajpe.hf_revisions import revision as hf_revision

import pandas as pd
import torch

import textfmt

PROJECT = HERE.parents[1]
DATA = PROJECT / "data" / "processed"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
N_OUT = 50_000


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=list(textfmt.CITY_NAMES))
    ap.add_argument("--adapter", required=True, help="PEFT adapter dir, or 'none' for the base model (zero-shot)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_out", type=int, default=N_OUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gen_batch", type=int, default=1024)
    ap.add_argument("--max_new", type=int, default=640, help="token limit per trail (20 check-ins need <= ~530)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--mem_cap_gb", type=float, default=0.0)
    ap.add_argument("--max_batches", type=int, default=0, help="debug: stop after this many batches")
    ap.add_argument("--model", default=MODEL, help="v2: base model of the adapter (reject decoding, top-p 1 only)")
    args = ap.parse_args()
    t0 = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    dev = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    if dev.type == "cuda" and args.mem_cap_gb > 0:
        total = torch.cuda.get_device_properties(dev).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.mem_cap_gb / total), dev)
    tok = AutoTokenizer.from_pretrained(args.model, revision=hf_revision(args.model))
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=hf_revision(args.model), dtype=dtype, attn_implementation="sdpa").to(dev)
    if args.adapter != "none":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    end_ids = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    gc = GenerationConfig(do_sample=True, temperature=args.temperature, top_k=0, top_p=1.0, min_p=None,
                          typical_p=1.0, repetition_penalty=1.0, no_repeat_ngram_size=0,
                          max_new_tokens=args.max_new, eos_token_id=end_ids, pad_token_id=end_ids[1])
    core = model.get_base_model() if hasattr(model, "get_base_model") else model
    core.generation_config = gc
    eff, _ = core._prepare_generation_config(gc)
    eff_d = {k: getattr(eff, k, None) for k in ["do_sample", "temperature", "top_k", "top_p", "min_p", "typical_p",
                                                 "repetition_penalty", "no_repeat_ngram_size", "max_new_tokens"]}
    log(f"effective generation config: {eff_d}")
    assert eff.temperature == args.temperature and eff.top_k in (0, None) and eff.top_p == 1.0 \
        and eff.repetition_penalty == 1.0

    cats = set(pd.read_parquet(DATA / args.dataset / "pois.parquet", columns=["category"])["category"].astype(str))
    prompt = tok.apply_chat_template(textfmt.messages(args.dataset), add_generation_prompt=True, tokenize=False)
    pid = tok(prompt, add_special_tokens=False)["input_ids"]
    P = len(pid)
    inp = torch.tensor([pid], device=dev)
    end_set = set(end_ids)
    kept, reasons = [], collections.Counter()
    n_gen = n_tok = b = 0
    t_gen = time.time()
    while len(kept) < args.n_out:
        if args.max_batches and b >= args.max_batches:
            break
        torch.manual_seed(args.seed * 1_000_003 + b)
        bs = args.gen_batch
        with torch.inference_mode():
            out = model.generate(input_ids=inp.expand(bs, -1), attention_mask=torch.ones((bs, P), dtype=torch.long,
                                 device=dev), generation_config=gc)
        for row in out[:, P:].tolist():
            end = next((k for k, t in enumerate(row) if t in end_set), None)
            finished = end is not None
            toks = row[:end] if finished else row
            n_tok += len(toks) + int(finished)
            text = tok.decode(toks, skip_special_tokens=False)
            res, why = textfmt.parse(text, cats, finished)
            reasons[why] += 1
            n_gen += 1
            if res is not None and len(kept) < args.n_out:
                kept.append((text, finished))
        b += 1
        log(f"batch {b}: generated {n_gen}, kept {len(kept)}/{args.n_out}, valid rate {reasons['ok'] / n_gen:.4f}, "
            f"{time.time() - t_gen:.0f}s")
        if n_gen >= 5000 and reasons["ok"] / n_gen < 0.02:
            log("valid rate < 0.02: model is broken")
            sys.exit(3)
    t_gen = time.time() - t_gen
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"text": [t for t, _ in kept], "finished": [f for _, f in kept]}).to_parquet(outp, index=False)
    info = {"model": args.model, "adapter": args.adapter, "dataset": args.dataset, "seed": args.seed,
            "n_out": args.n_out, "n_kept": len(kept), "n_generated": n_gen, "valid_rate": reasons["ok"] / max(n_gen, 1),
            "reasons": dict(reasons), "generated_tokens": n_tok, "batches": b, "gen_batch": args.gen_batch,
            "max_new_tokens": args.max_new, "generation": eff_d, "prompt_tokens": P,
            "sample_gen_sec": round(t_gen, 1), "sample_total_sec": round(time.time() - t0, 1), "device": str(dev),
            "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
            "max_mem_gb": round(torch.cuda.max_memory_allocated(dev) / 2**30, 2) if dev.type == "cuda" else None}
    outp.with_suffix(".json").write_text(json.dumps(info, indent=1))
    log(f"wrote {outp}: {len(kept)} texts ({n_gen} generated, valid rate {info['valid_rate']:.4f}) in {t_gen:.0f}s")


if __name__ == "__main__":
    main()
