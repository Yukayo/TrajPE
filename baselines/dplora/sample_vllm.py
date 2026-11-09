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
os.environ.setdefault("DO_NOT_TRACK", "1")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE.parents[1]))
from trajpe.hf_revisions import revision as hf_revision

import pandas as pd

import textfmt

PROJECT = HERE.parents[1]
DATA = PROJECT / "data" / "processed"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=list(textfmt.CITY_NAMES))
    ap.add_argument("--adapter", required=True, help="PEFT adapter dir, or 'none' for the base model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_out", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_new", type=int, default=640)
    ap.add_argument("--kv_gb", type=float, default=8.0, help="fixed KV-cache size (GiB)")
    ap.add_argument("--util", type=float, default=0.05, help="vLLM start-up check only (see docstring)")
    ap.add_argument("--max_num_seqs", type=int, default=1024)
    ap.add_argument("--overshoot", type=float, default=1.03, help="requests per missing valid text")
    ap.add_argument("--max_rounds", type=int, default=20)
    ap.add_argument("--min_valid", type=float, default=0.02, help="abort below this valid rate (>= 20k samples)")
    ap.add_argument("--model", default=MODEL, help="v2: base model of the adapter")
    ap.add_argument("--decode", choices=["reject", "repair", "constrained"], default="reject", help="v2 (docstring)")
    ap.add_argument("--coord_decimals", type=int, default=3, choices=[3, 4], help="v2: decimals in the grammar")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--min_p", type=float, default=0.0)
    args = ap.parse_args()
    t0 = time.time()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt
    from vllm.lora.request import LoRARequest

    tok = AutoTokenizer.from_pretrained(args.model, revision=hf_revision(args.model))
    lora, rank = None, 0
    if args.adapter != "none":
        acfg = json.loads((Path(args.adapter) / "adapter_config.json").read_text())
        rank = int(acfg["r"])
        if acfg.get("base_model_name_or_path") not in (None, args.model):
            raise SystemExit(f"adapter base model {acfg['base_model_name_or_path']} != --model {args.model}")
        lora = LoRARequest("dplora", 1, str(Path(args.adapter).resolve()))
    llm = LLM(model=args.model, tokenizer=args.model, revision=hf_revision(args.model), tokenizer_revision=hf_revision(args.model), dtype="bfloat16", seed=args.seed, max_model_len=1024,
              gpu_memory_utilization=args.util, kv_cache_memory_bytes=int(args.kv_gb * 2**30),
              enable_prefix_caching=True, generation_config="vllm", max_num_seqs=args.max_num_seqs,
              enable_lora=lora is not None, max_lora_rank=max(8, rank) if lora else 16, max_loras=1)
    end_ids = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    cats = set(pd.read_parquet(DATA / args.dataset / "pois.parquet", columns=["category"])["category"].astype(str))
    extra = {}
    if args.decode == "constrained":
        from vllm.sampling_params import StructuredOutputsParams
        extra["structured_outputs"] = StructuredOutputsParams(grammar=textfmt.ebnf(cats, args.coord_decimals))
    sp = SamplingParams(n=1, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, min_p=args.min_p,
                        repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0, max_tokens=args.max_new,
                        stop_token_ids=end_ids, skip_special_tokens=False, **extra)
    prompt = tok.apply_chat_template(textfmt.messages(args.dataset), add_generation_prompt=True, tokenize=False)
    pid = tok(prompt, add_special_tokens=False)["input_ids"]
    t_load = time.time() - t0
    kept, reasons = [], collections.Counter()
    n_gen = n_tok = rounds = 0
    n_dropped_lines = 0
    t_gen = time.time()
    while len(kept) < args.n_out and rounds < args.max_rounds:
        missing = args.n_out - len(kept)
        rate = reasons["ok"] / n_gen if n_gen else 0.9
        if n_gen >= 20_000 and rate < args.min_valid:
            print(f"valid rate {rate:.4f} < {args.min_valid} after {n_gen} samples: model is broken", flush=True)
            sys.exit(3)
        n_req = min(3 * args.n_out, max(256, int(missing / max(rate, 0.05) * args.overshoot)))
        outs = llm.generate([TokensPrompt(prompt_token_ids=pid)] * n_req, sp, use_tqdm=False, lora_request=lora)
        for o in outs:
            c = o.outputs[0]
            finished = c.finish_reason == "stop"
            n_tok += len(c.token_ids)
            if args.decode == "repair":
                res, why, nd = textfmt.clean(c.text, cats, finished)
                if res is not None and len(kept) < args.n_out:
                    n_dropped_lines += nd
                    kept.append((res, True))
                reasons[why] += 1
                n_gen += 1
                continue
            res, why = textfmt.parse(c.text, cats, finished)
            reasons[why] += 1
            n_gen += 1
            if res is not None and len(kept) < args.n_out:
                kept.append((c.text, finished))
        rounds += 1
        log(f"round {rounds}: {n_req} requests, kept {len(kept)}/{args.n_out}, valid rate "
            f"{reasons['ok'] / n_gen:.4f}, {time.time() - t_gen:.0f}s")
    t_gen = time.time() - t_gen
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"text": [t for t, _ in kept], "finished": [f for _, f in kept]}).to_parquet(outp, index=False)
    import torch
    info = {"model": args.model, "adapter": args.adapter, "dataset": args.dataset, "seed": args.seed,
            "sampler": "vllm",
            "n_out": args.n_out, "n_kept": len(kept), "n_generated": n_gen,
            "valid_rate": reasons["ok"] / max(n_gen, 1), "reasons": dict(reasons), "generated_tokens": n_tok,
            "batches": rounds, "gen_batch": args.max_num_seqs, "max_new_tokens": args.max_new,
            "generation": {"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
                           "min_p": args.min_p, "repetition_penalty": 1.0,
                           "lora_at_runtime": lora is not None, "kv_gb": args.kv_gb},
            "prompt_tokens": len(pid), "sample_load_sec": round(t_load, 1), "sample_gen_sec": round(t_gen, 1),
            "sample_total_sec": round(time.time() - t0, 1), "device": "cuda",
            "gpu": torch.cuda.get_device_name(0), "max_mem_gb": None}
    if args.decode != "reject":
        info["generation"]["decode"] = args.decode
        info["generation"]["coord_decimals"] = args.coord_decimals
        if args.decode == "repair":
            info["generation"]["dropped_lines_in_kept"] = n_dropped_lines
    outp.with_suffix(".json").write_text(json.dumps(info, indent=1))
    log(f"wrote {outp}: {len(kept)} texts ({n_gen} generated, valid rate {info['valid_rate']:.4f}) in {t_gen:.0f}s")
    if len(kept) < args.n_out:
        print(f"only {len(kept)} valid texts after {rounds} rounds", flush=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
