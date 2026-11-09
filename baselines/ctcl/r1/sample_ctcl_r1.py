from __future__ import annotations

import collections
import json
import os
import sys
import time
from pathlib import Path

R1 = Path(__file__).resolve().parent
CTCL = R1.parent
sys.path.insert(0, str(CTCL))
import ctcl_conf as C
import sample_ctcl as SC

import numpy as np

textfmt = C.textfmt
REV = "r1"
BrokenModel = SC.BrokenModel
log = SC.log


def global_fill(M: int, w, avail: dict) -> dict | None:
    avail = {k: int(a) for k, a in avail.items() if a > 0}
    if M <= 0:
        return {}
    if sum(avail.values()) < M:
        return None
    take = {k: 0 for k in avail}
    rem = M
    while rem > 0:
        act = [k for k in sorted(avail) if take[k] < avail[k]]
        add = SC._apportion([float(w[k]) for k in act], rem)
        got = 0
        for k, a in zip(act, add):
            t = min(int(a), avail[k] - take[k])
            take[k] += t
            got += t
        if got <= 0:
            raise RuntimeError("global fill made no progress")
        rem -= got
    return {k: t for k, t in take.items() if t > 0}


def rule_i_from_rounds(rounds: list) -> dict | None:
    n = 0
    for r in rounds:
        n += int(r["requests"])
        if n >= C.MIN_VALID_AFTER:
            v = float(r["valid_rate"])
            return {"rounds_in_prefix": int(r["round"]), "n_samples": n, "first_round_valid_rate": v,
                    "threshold": C.MIN_VALID, "passed": v >= C.MIN_VALID}
    return None


def run_rounds(n: np.ndarray, w: np.ndarray, generate, categories: set, log_fn=log) -> tuple:
    K = len(n)
    gen, ok, reasons = collections.Counter(), collections.Counter(), collections.Counter()
    kept = {k: [] for k in range(K)}
    surplus = {k: [] for k in range(K)}
    extra = {k: [] for k in range(K)}
    fill = {k: [] for k in range(K)}
    st = {"rounds": [], "extra_rounds": []}
    gate = {"evaluated": False}
    t0 = time.time()

    def vhat() -> float:
        g = sum(gen.values())
        return sum(ok.values()) / g if g else C.V_INIT

    def stats() -> dict:
        return {"n_generated": int(sum(gen.values())), "n_ok": int(sum(ok.values())), "valid_rate": vhat(),
                "reasons": dict(reasons), "rounds": list(st["rounds"]), "extra_rounds": list(st["extra_rounds"]),
                "r1_gate": dict(gate)}

    def one_round(sink: dict, want: dict, label: str, r: int) -> bool:
        m = {k: want[k] - len(sink[k]) for k in sorted(want) if want[k] > len(sink[k])}
        if not m:
            return False
        n_gen, v = int(sum(gen.values())), vhat()
        if not gate["evaluated"] and n_gen >= C.MIN_VALID_AFTER:
            gate.update(evaluated=True, n_samples=n_gen, first_round_valid_rate=v, threshold=C.MIN_VALID,
                        passed=v >= C.MIN_VALID, before=f"{label} round {r}")
            if v < C.MIN_VALID:
                raise BrokenModel(f"r1 rule (i): first-round valid rate {v:.4f} < {C.MIN_VALID} after {n_gen} "
                                  f"samples: model is broken", stats())
        req = SC.sizing(m, gen, ok, v)
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
    r1fill = {"applied": False, "missing_after_extra_rounds": 0, "from_global_surplus": 0, "topics_giving": 0,
              "global_surplus_available": 0}
    if D > 0:
        F = [k for k in sorted(want) if k not in unfilled]
        if not F:
            raise BrokenModel("no filled topic to re-apportion the deficit to (safety net)", stats())
        add = SC._apportion([float(w[k]) for k in F], D)
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
            avail = {k: len(surplus[k]) for k in range(K) if surplus[k]}
            r1fill.update(applied=True, missing_after_extra_rounds=missing,
                          global_surplus_available=int(sum(avail.values())))
            take = global_fill(missing, w, avail)
            if take is None:
                raise BrokenModel(f"r1 rule (ii): {missing} texts still missing after the re-apportionment rounds and "
                                  f"only {sum(avail.values())} surplus texts exist (safety net)",
                                  dict(stats(), r1_global_fill=r1fill))
            for k, a in take.items():
                fill[k], surplus[k] = surplus[k][:a], surplus[k][a:]
            r1fill.update(from_global_surplus=int(sum(take.values())), topics_giving=len(take))
            log_fn(f"r1 (ii): {missing} missing texts filled from the global surplus of {len(take)} topics")
        reap["topics_receiving"] = len(want_x)
    texts = ([(t, k) for k in range(K) for t, _ in kept[k]] + [(t, k) for k in range(K) for t, _ in extra[k]]
             + [(t, k) for k in range(K) for t, _ in fill[k]])
    if len(texts) != int(np.sum(n)):
        raise RuntimeError(f"{len(texts)} texts != n_out {int(np.sum(n))}")
    out = stats()
    out["reapportionment"] = reap
    out["r1_global_fill"] = r1fill
    out["dropped_lines_in_kept"] = int(sum(nd for k in range(K) for _, nd in kept[k] + extra[k] + fill[k]))
    return texts, out


def failed_marker(a) -> Path:
    outp = Path(a.out)
    return outp.with_name(outp.stem + ".failed_r1.json")


def synthesize(a, hist: dict, n, w, cats: set, generate, meta: dict, t0: float, final=lambda: {}) -> int:
    import pandas as pd
    outp = Path(a.out)
    t_gen = time.time()
    try:
        texts, st = run_rounds(n, w, generate, cats)
    except BrokenModel as e:
        C.write_json_atomic(failed_marker(a), dict(e.stats, why=e.why, dataset=a.dataset, seed=a.seed, rev=REV,
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
                           "ctcl": {"rev": REV, "quota_sha256": hist["quota_sha256"],
                                    "topic_table_sha256": hist["table_sha256"],
                                    "n_topics_with_quota": int((n > 0).sum()), "rounds": st["rounds"],
                                    "extra_rounds": st["extra_rounds"], "reapportionment": st["reapportionment"],
                                    "r1_gate": st["r1_gate"], "r1_global_fill": st["r1_global_fill"],
                                    "constants": {"overshoot": C.OVERSHOOT, "rate_floor": C.RATE_FLOOR,
                                                  "max_rounds": C.MAX_ROUNDS, "extra_rounds": C.EXTRA_ROUNDS,
                                                  "round_cap": C.ROUND_CAP, "min_valid": C.MIN_VALID,
                                                  "min_valid_after": C.MIN_VALID_AFTER, "v_init": C.V_INIT,
                                                  "broken_model_gate": "r1 (i): first-round valid rate",
                                                  "safety_net": "r1 (ii): global surplus fill"}}},
            "sample_gen_sec": round(t_gen, 1), **meta}
    info.update(final())
    info["sample_total_sec"] = round(time.time() - t0, 1)
    C.write_json_atomic(outp.with_suffix(".json"), info)
    log(f"wrote {outp}: {len(df)} texts ({st['n_generated']} generated, valid rate {st['valid_rate']:.4f}) "
        f"in {t_gen:.0f}s (r1)")
    return 0


def main(argv=None):
    a = SC.parse(argv)
    t0 = time.time()
    if failed_marker(a).exists():
        print(f"{failed_marker(a)} exists: this sampling failed for good under r1; not re-sampling", flush=True)
        sys.exit(3)
    if SC.failed_marker(a).exists():
        raise SystemExit(f"{SC.failed_marker(a)} (v1 failure marker) exists: run through pipeline_ctcl_r1.py, which "
                         f"applies rule (i) to it and moves it aside")
    hist, table, n, w, cats = SC.load_inputs(a)
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
    generate = SC.generator_vllm(llm, sp, prompts, lora)
    meta = {"model": model, "adapter": a.adapter or None, "sampler": "vllm", "sampler_rev": REV,
            "prompt_tokens_max": plen, "sample_load_sec": round(time.time() - t0, 1), "device": "cuda",
            "gpu": torch.cuda.get_device_name(0)}

    def final():
        return {"generated_tokens": generate.tokens, "max_mem_gb": SC.gpu_mem_tree_gb()}
    sys.exit(synthesize(a, hist, n, w, cats, generate, meta, t0, final))


if __name__ == "__main__":
    main()
