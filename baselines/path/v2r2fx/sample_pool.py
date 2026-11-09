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

import pandas as pd

import pathfmt as PF


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def gpu_mem_gb() -> float | None:
    import subprocess
    try:
        import psutil
        me = psutil.Process()
        pids = {me.pid} | {c.pid for c in me.children(recursive=True)}
        q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30).stdout
        mib = sum(float(m) for pid, m in (ln.split(",") for ln in q.strip().splitlines() if ln.strip())
                  if int(pid) in pids)
        return round(mib / 1024, 2)
    except Exception:
        return None


def n_requests(missing: int, rate: float, m: int = PF.POOL_M) -> int:
    return min(3 * m, max(256, int(missing / max(rate, 0.05) * PF.OVERSHOOT)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model_key", default=PF.DEFAULT_MODEL, choices=list(PF.MODELS))
    ap.add_argument("--seed", type=int, default=PF.SEED)
    ap.add_argument("--m", type=int, default=PF.POOL_M, help="pool size (debug only; releases need M = 100,000)")
    ap.add_argument("--kv_gb", type=float, default=8.0, help="fixed KV-cache size (GiB; operational)")
    ap.add_argument("--util", type=float, default=0.05, help="vLLM start-up check only (operational)")
    ap.add_argument("--max_num_seqs", type=int, default=1024, help="operational")
    args = ap.parse_args()
    PF.check_dataset(args.dataset)
    t0 = time.time()
    mk = PF.MODELS[args.model_key]

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt
    from vllm.lora.request import LoRARequest

    tok = AutoTokenizer.from_pretrained(mk["id"], revision=mk["revision"])
    acfg = json.loads((Path(args.adapter) / "adapter_config.json").read_text())
    if int(acfg["r"]) != PF.RANK or acfg.get("base_model_name_or_path") != mk["id"]:
        raise SystemExit(f"adapter r={acfg['r']} base={acfg.get('base_model_name_or_path')} != {PF.RANK} / {mk['id']}")
    lora = LoRARequest("path", 1, str(Path(args.adapter).resolve()))
    llm = LLM(model=mk["id"], revision=mk["revision"], tokenizer=mk["id"], tokenizer_revision=mk["revision"],
              dtype="bfloat16", seed=args.seed, max_model_len=1024, gpu_memory_utilization=args.util,
              kv_cache_memory_bytes=int(args.kv_gb * 2**30), enable_prefix_caching=True, generation_config="vllm",
              max_num_seqs=args.max_num_seqs, enable_lora=True, max_lora_rank=PF.RANK, max_loras=1)
    end_ids = PF.stop_ids(tok)
    cats = set(pd.read_parquet(PF.DATA / args.dataset / "pois.parquet", columns=["category"])["category"].astype(str))
    sp = SamplingParams(n=1, temperature=PF.TEMPERATURE, top_p=PF.TOP_P, top_k=PF.TOP_K, min_p=PF.MIN_P,
                        repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0, max_tokens=PF.MAX_NEW,
                        stop_token_ids=end_ids, skip_special_tokens=False)
    pid = tok(PF.header(args.dataset, tok), add_special_tokens=False)["input_ids"]
    t_load = time.time() - t0
    kept, rows_kept = [], []
    reasons, ends, ends_kept = collections.Counter(), collections.Counter(), collections.Counter()
    n_gen = n_tok = rounds = n_cut20 = 0
    per_round = []
    t_gen = time.time()
    status = "ok"
    peak_mem = None
    while len(kept) < args.m and rounds < PF.MAX_ROUNDS:
        rate = reasons["ok"] / n_gen if n_gen else 0.9
        if n_gen >= PF.MIN_VALID_AFTER and rate < PF.MIN_VALID:
            status = f"abort: valid rate {rate:.4f} < {PF.MIN_VALID} after {n_gen} samples"
            break
        n_req = n_requests(args.m - len(kept), rate, args.m)
        outs = llm.generate([TokensPrompt(prompt_token_ids=pid)] * n_req, sp, use_tqdm=False, lora_request=lora)
        for o in outs:
            c = o.outputs[0]
            finished = c.finish_reason == "stop"
            n_tok += len(c.token_ids)
            text = c.text
            for t in PF.STOP_TOKENS:
                text = text.replace(t, "")
            res, why, info = PF.truncate(text, cats, finished)
            reasons[why] += 1
            ends[info["end"]] += 1
            n_gen += 1
            if res is not None and len(kept) < args.m:
                kept.append(res)
                rows_kept.append(min(info["rows"], PF.MAX_LEN))
                ends_kept[info["end"]] += 1
                n_cut20 += int(info["rows"] > PF.MAX_LEN)
        rounds += 1
        per_round.append(n_req)
        mem = gpu_mem_gb()
        peak_mem = max(peak_mem or 0.0, mem or 0.0) if mem is not None else peak_mem
        log(f"round {rounds}: {n_req} requests, kept {len(kept)}/{args.m}, valid rate {reasons['ok'] / n_gen:.4f}, "
            f"{n_tok} tokens, {time.time() - t_gen:.0f}s")
    t_gen = time.time() - t_gen
    if status == "ok" and len(kept) < args.m:
        status = f"cap: only {len(kept)} valid candidates after {rounds} rounds"
    import torch
    info = {"method": "path", "version": PF.VERSION, "model": mk["id"], "model_revision": mk["revision"],
            "adapter": str(args.adapter),
            "dataset": args.dataset, "seed": args.seed, "sampler": "vllm", "m": args.m, "n_kept": len(kept),
            "n_generated": n_gen, "valid_rate": reasons["ok"] / max(n_gen, 1), "reasons": dict(reasons),
            "table_end": dict(ends), "table_end_kept": dict(ends_kept), "cut_at_20": n_cut20,
            "generated_tokens": n_tok, "rounds": rounds, "requests_per_round": per_round, "status": status,
            "generation": {"temperature": PF.TEMPERATURE, "top_p": PF.TOP_P, "top_k": PF.TOP_K, "min_p": PF.MIN_P,
                           "repetition_penalty": 1.0, "max_new_tokens": PF.MAX_NEW, "stop": PF.STOP_TOKENS,
                           "prompt": "chat-template header H(d) (O2)",
                           "lora_at_runtime": True, "max_lora_rank": PF.RANK, "kv_gb": args.kv_gb,
                           "max_num_seqs": args.max_num_seqs, "generation_config": "vllm (Hub config kept out)",
                           "conditioning": "header H(d) only (A6)", "decode": PF.DECODE,
                           "coord_decimals": PF.DECIMALS},
            "rule": {"max_rounds": PF.MAX_ROUNDS, "min_valid": PF.MIN_VALID, "min_valid_after": PF.MIN_VALID_AFTER,
                     "overshoot": PF.OVERSHOOT, "max_requests_per_round": 3 * args.m},
            "prompt_tokens": len(pid), "sample_load_sec": round(t_load, 1), "sample_gen_sec": round(t_gen, 1),
            "sample_total_sec": round(time.time() - t0, 1), "gpu": torch.cuda.get_device_name(0),
            "gpu_mem_gb_after_round_max": peak_mem,
            "vllm_version": __import__("vllm").__version__}
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    if status != "ok":
        outp.with_name(outp.stem + ".failed.json").write_text(json.dumps(info, indent=1))
        log(f"FAILED ({status}); details in {outp.with_name(outp.stem + '.failed.json')}")
        sys.exit(3)
    tmp = outp.with_name(outp.name + ".tmp")
    pd.DataFrame({"text": kept, "n_rows": rows_kept}).to_parquet(tmp, index=False)
    outp.with_suffix(".json").write_text(json.dumps(info, indent=1))
    tmp.rename(outp)
    log(f"wrote {outp}: {len(kept)} candidates ({n_gen} generated, valid rate {info['valid_rate']:.4f}, "
        f"{n_tok} tokens) in {t_gen:.0f}s")


if __name__ == "__main__":
    main()
