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
import math
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
import ctcl_conf as C

import numpy as np

textfmt = C.textfmt


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class BrokenModel(Exception):

    def __init__(self, why: str, stats: dict):
        super().__init__(why)
        self.why, self.stats = why, stats


def _apportion(w, n: int) -> np.ndarray:
    sys.path.insert(0, str(C.PROJECT))
    from trajpe.ablations import apportion
    return apportion(np.asarray(w, np.float64), n)


def sizing(m: dict, gen: dict, ok: dict, v: float) -> dict:
    req = {}
    for k, mk in m.items():
        rk = (ok[k] + v) / (gen[k] + 1)
        req[k] = math.ceil(C.OVERSHOOT * mk / max(rk, C.RATE_FLOOR))
    tot = sum(req.values())
    if tot > C.ROUND_CAP:
        f = C.ROUND_CAP / tot
        req = {k: math.ceil(x * f) for k, x in req.items()}
    return req


def run_rounds(n: np.ndarray, w: np.ndarray, generate, categories: set, log_fn=log) -> tuple:
    K = len(n)
    gen, ok, reasons = collections.Counter(), collections.Counter(), collections.Counter()
    kept = {k: [] for k in range(K)}
    surplus = {k: [] for k in range(K)}
    extra = {k: [] for k in range(K)}
    st = {"rounds": [], "extra_rounds": []}
    t0 = time.time()

    def vhat() -> float:
        g = sum(gen.values())
        return sum(ok.values()) / g if g else C.V_INIT

    def stats() -> dict:
        return {"n_generated": int(sum(gen.values())), "n_ok": int(sum(ok.values())), "valid_rate": vhat(),
                "reasons": dict(reasons), "rounds": list(st["rounds"]), "extra_rounds": list(st["extra_rounds"])}

    def one_round(sink: dict, want: dict, label: str, r: int) -> bool:
        m = {k: want[k] - len(sink[k]) for k in sorted(want) if want[k] > len(sink[k])}
        if not m:
            return False
        n_gen, v = int(sum(gen.values())), vhat()
        if n_gen >= C.MIN_VALID_AFTER and v < C.MIN_VALID:
            raise BrokenModel(f"valid rate {v:.4f} < {C.MIN_VALID} after {n_gen} samples: model is broken", stats())
        req = sizing(m, gen, ok, v)
        outs = generate([(k, req[k]) for k in sorted(req)])
        if len(outs) != sum(req.values()):
            raise RuntimeError("the generator returned a different number of outputs")
        for k, text, finished in outs:
            gen[k] += 1
            res, why, nd = textfmt.clean(text, categories, bool(finished))
            reasons[why] += 1
            if res is None:
                continue
            ok[k] += 1
            (sink[k] if len(sink[k]) < want[k] else surplus[k]).append((res, nd))
        row = {"round": r, "requests": int(sum(req.values())), "topics": len(req), "valid_rate": round(vhat(), 5),
               "sec": round(time.time() - t0, 1)}
        st["rounds" if label == "main" else "extra_rounds"].append(row)
        log_fn(f"{label} round {r}: {row['requests']} requests over {row['topics']} topics, overall valid rate "
               f"{row['valid_rate']:.4f}")
        return True

    want = {k: int(n[k]) for k in range(K) if n[k] > 0}
    for r in range(1, C.MAX_ROUNDS + 1):
        if not one_round(kept, want, "main", r):
            break
    unfilled = {k: want[k] - len(kept[k]) for k in want if want[k] > len(kept[k])}
    D = int(sum(unfilled.values()))
    reap = {"deficit": D, "unfilled_topics": len(unfilled), "unfilled_quota": int(sum(want[k] for k in unfilled)),
            "unfilled_texts_kept": int(sum(len(kept[k]) for k in unfilled)), "from_surplus": 0,
            "from_extra_rounds": 0, "topics_receiving": 0}
    if D > 0:
        F = [k for k in sorted(want) if k not in unfilled]
        if not F:
            raise BrokenModel("no filled topic to re-apportion the deficit to (safety net)", stats())
        add = _apportion([float(w[k]) for k in F], D)
        want_x = {}
        for k, a in zip(F, add):
            a = int(a)
            if a <= 0:
                continue
            extra[k], surplus[k] = surplus[k][:a], surplus[k][a:]
            reap["from_surplus"] += len(extra[k])
            want_x[k] = a
        for r in range(1, C.EXTRA_ROUNDS + 1):
            before = sum(len(extra[k]) for k in want_x)
            if not one_round(extra, want_x, "extra", r):
                break
            reap["from_extra_rounds"] += sum(len(extra[k]) for k in want_x) - before
        missing = int(sum(want_x[k] - len(extra[k]) for k in want_x))
        if missing > 0:
            raise BrokenModel(f"{missing} texts still missing after the re-apportionment rounds (safety net)",
                              stats())
        reap["topics_receiving"] = len(want_x)
    texts = [(t, k) for k in range(K) for t, _ in kept[k]] + [(t, k) for k in range(K) for t, _ in extra[k]]
    if len(texts) != int(np.sum(n)):
        raise RuntimeError(f"{len(texts)} texts != n_out {int(np.sum(n))}")
    out = stats()
    out["reapportionment"] = reap
    out["dropped_lines_in_kept"] = int(sum(nd for k in range(K) for _, nd in kept[k] + extra[k]))
    return texts, out


def gpu_mem_tree_gb():
    import subprocess
    try:
        pids, stack = {os.getpid()}, [os.getpid()]
        while stack:
            p = stack.pop()
            for t in Path(f"/proc/{p}/task").iterdir():
                for c in (t / "children").read_text().split():
                    if int(c) not in pids:
                        pids.add(int(c))
                        stack.append(int(c))
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        tot = 0.0
        for line in out.splitlines():
            pid, mem = (x.strip() for x in line.split(","))
            if int(pid) in pids:
                tot += float(mem)
        return round(tot / 1024, 2)
    except Exception:
        return None


def generator_vllm(llm, sp, prompts: dict, lora):
    from vllm import TokensPrompt

    def generate(reqs):
        batch, owner = [], []
        for k, c in reqs:
            batch += [TokensPrompt(prompt_token_ids=prompts[k])] * c
            owner += [k] * c
        outs = llm.generate(batch, sp, use_tqdm=False, lora_request=lora)
        res = []
        for k, o in zip(owner, outs):
            c = o.outputs[0]
            res.append((k, c.text, c.finish_reason == "stop"))
            generate.tokens += len(c.token_ids)
        return res
    generate.tokens = 0
    return generate


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model_dir", default="", help="full fine-tuned weights (train_ctcl.py model/)")
    ap.add_argument("--adapter", default="", help="Q5 (ii) fallback: LoRA adapter dir")
    ap.add_argument("--hist", required=True)
    ap.add_argument("--table", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_out", type=int, default=C.N_OUT)
    ap.add_argument("--kv_gb", type=float, default=C.KV_GB)
    ap.add_argument("--util", type=float, default=0.05, help="vLLM start-up check only (see sample_vllm.py)")
    a = ap.parse_args(argv)
    C.check_dataset(a.dataset)
    if bool(a.model_dir) == bool(a.adapter):
        ap.error("give exactly one of --model_dir / --adapter")
    return a


def failed_marker(a) -> Path:
    outp = Path(a.out)
    return outp.with_name(outp.stem + ".failed.json")


def load_inputs(a) -> tuple:
    import pandas as pd
    sys.path.insert(0, str(HERE))
    import topics as TP
    hist = json.loads(Path(a.hist).read_text())
    table_file = Path(a.table or C.table_path(a.dataset))
    table = TP.load_table(table_file)
    if hist["dataset"] != a.dataset or table["dataset"] != a.dataset or \
            hist["table_sha256"] != C.sha256_file(table_file):
        raise SystemExit("hist.json / topic table mismatch")
    n, w = np.asarray(hist["quota"], np.int64), np.asarray(hist["w"], np.int64)
    if int(n.sum()) != a.n_out:
        raise SystemExit(f"quotas sum to {int(n.sum())}, not {a.n_out}")
    cats = set(pd.read_parquet(C.DATA / a.dataset / "pois.parquet", columns=["category"])["category"].astype(str))
    return hist, table, n, w, cats


def synthesize(a, hist: dict, n, w, cats: set, generate, meta: dict, t0: float, final=lambda: {}) -> int:
    import pandas as pd
    outp = Path(a.out)
    t_gen = time.time()
    try:
        texts, st = run_rounds(n, w, generate, cats)
    except BrokenModel as e:
        C.write_json_atomic(failed_marker(a), dict(e.stats, why=e.why, dataset=a.dataset, seed=a.seed,
                                                   sample_gen_sec=round(time.time() - t_gen, 1), **final()))
        print(e.why, flush=True)
        return 3
    t_gen = time.time() - t_gen
    order = np.random.default_rng([a.seed, C.ORDER_STREAM]).permutation(len(texts))
    df = pd.DataFrame({"text": [texts[i][0] for i in order], "finished": True,
                       "topic": np.asarray([texts[i][1] for i in order], np.int64)})
    outp.parent.mkdir(parents=True, exist_ok=True)
    tmp = outp.with_name(outp.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, outp)
    info = {"method": "ctcl", "dataset": a.dataset, "seed": a.seed, "n_out": a.n_out, "n_kept": len(df),
            "n_generated": st["n_generated"], "valid_rate": st["valid_rate"], "reasons": st["reasons"],
            "batches": len(st["rounds"]) + len(st["extra_rounds"]), "gen_batch": C.MAX_NUM_SEQS,
            "max_new_tokens": C.MAX_RESP_TOKENS,
            "generation": {"temperature": C.TEMPERATURE, "top_p": C.TOP_P, "top_k": C.TOP_K, "min_p": C.MIN_P,
                           "repetition_penalty": 1.0, "decode": "repair", "coord_decimals": C.COORD_DECIMALS,
                           "lora_at_runtime": bool(a.adapter), "kv_gb": a.kv_gb, "dtype": "bfloat16",
                           "dropped_lines_in_kept": st["dropped_lines_in_kept"],
                           "ctcl": {"quota_sha256": hist["quota_sha256"], "topic_table_sha256": hist["table_sha256"],
                                    "n_topics_with_quota": int((n > 0).sum()), "rounds": st["rounds"],
                                    "extra_rounds": st["extra_rounds"], "reapportionment": st["reapportionment"],
                                    "constants": {"overshoot": C.OVERSHOOT, "rate_floor": C.RATE_FLOOR,
                                                  "max_rounds": C.MAX_ROUNDS, "extra_rounds": C.EXTRA_ROUNDS,
                                                  "round_cap": C.ROUND_CAP, "min_valid": C.MIN_VALID,
                                                  "min_valid_after": C.MIN_VALID_AFTER, "v_init": C.V_INIT}}},
            "sample_gen_sec": round(t_gen, 1), **meta}
    info.update(final())
    info["sample_total_sec"] = round(time.time() - t0, 1)
    C.write_json_atomic(outp.with_suffix(".json"), info)
    log(f"wrote {outp}: {len(df)} texts ({st['n_generated']} generated, valid rate {st['valid_rate']:.4f}) "
        f"in {t_gen:.0f}s")
    return 0


def main(argv=None):
    a = parse(argv)
    t0 = time.time()
    if failed_marker(a).exists():
        print(f"{failed_marker(a)} exists: this sampling failed for good (R13); not re-sampling", flush=True)
        sys.exit(3)
    hist, table, n, w, cats = load_inputs(a)
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    tok = AutoTokenizer.from_pretrained(C.MODEL, revision=C.MODEL_COMMIT)
    lora, rank = None, 0
    if a.adapter:
        rank = int(json.loads((Path(a.adapter) / "adapter_config.json").read_text())["r"])
        lora = LoRARequest("ctcl_lora", 1, str(Path(a.adapter).resolve()))
        model, extra = C.MODEL, {"revision": C.MODEL_COMMIT, "tokenizer_revision": C.MODEL_COMMIT}
    else:
        model, extra = str(Path(a.model_dir).resolve()), {"tokenizer_revision": C.MODEL_COMMIT}
    llm = LLM(model=model, tokenizer=C.MODEL, dtype="bfloat16", seed=a.seed, max_model_len=C.MAX_MODEL_LEN,
              gpu_memory_utilization=a.util, kv_cache_memory_bytes=int(a.kv_gb * 2**30), enable_prefix_caching=True,
              generation_config="vllm", max_num_seqs=C.MAX_NUM_SEQS, enable_lora=lora is not None,
              max_lora_rank=max(8, rank) if lora else 16, max_loras=1, **extra)
    end_ids = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    sp = SamplingParams(n=1, temperature=C.TEMPERATURE, top_p=C.TOP_P, top_k=C.TOP_K, min_p=C.MIN_P,
                        repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0,
                        max_tokens=C.MAX_RESP_TOKENS, stop_token_ids=end_ids, skip_special_tokens=False)
    rows = table["rows"]
    prompts = {k: C.prompt_ids(tok, a.dataset, rows[k]["condition"]) for k in np.flatnonzero(n > 0).tolist()}
    plen = max(len(v) for v in prompts.values())
    if plen + C.MAX_RESP_TOKENS > C.MAX_MODEL_LEN:
        raise SystemExit("prompt + 640 tokens exceed max_model_len")
    generate = generator_vllm(llm, sp, prompts, lora)
    meta = {"model": model, "adapter": a.adapter or None, "sampler": "vllm", "prompt_tokens_max": plen,
            "sample_load_sec": round(time.time() - t0, 1), "device": "cuda", "gpu": torch.cuda.get_device_name(0)}

    def final():
        return {"generated_tokens": generate.tokens, "max_mem_gb": gpu_mem_tree_gb()}
    sys.exit(synthesize(a, hist, n, w, cats, generate, meta, t0, final))


if __name__ == "__main__":
    main()
