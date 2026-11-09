from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

R2 = Path(__file__).resolve().parent
sys.path.insert(0, str(R2))
import r2conf
import sample_ctcl as SC
import sample_ctcl_r1 as S1

import numpy as np

C = r2conf.C
REV = r2conf.REV
log = SC.log


class RejectFormat:

    def __init__(self, textfmt):
        self.tf = textfmt

    def clean(self, text: str, categories: set, finished: bool = True):
        res, why = self.tf.parse(text, categories, finished)
        if res is None:
            return None, why, 0
        if not finished:
            text = "\n".join(text.split("\n")[:-1])
        return text, "ok", 0


S1.textfmt = RejectFormat(C.textfmt)


def failed_marker(a) -> Path:
    outp = Path(a.out)
    return outp.with_name(outp.stem + ".failed_r2.json")


def synthesize(a, hist: dict, n, w, cats: set, generate, meta: dict, t0: float, final=lambda: {}) -> int:
    import pandas as pd
    outp = Path(a.out)
    t_gen = time.time()
    try:
        texts, st = S1.run_rounds(n, w, generate, cats)
    except S1.BrokenModel as e:
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
                           "repetition_penalty": 1.0, "decode": r2conf.DECODE, "coord_decimals": C.COORD_DECIMALS,
                           "lora_at_runtime": bool(a.adapter), "kv_gb": a.kv_gb, "dtype": "bfloat16",
                           "dropped_lines_in_kept": st["dropped_lines_in_kept"],
                           "ctcl": {"rev": REV, "r2": r2conf.summary(), "quota_sha256": hist["quota_sha256"],
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
        f"in {t_gen:.0f}s (r2)")
    return 0


def main(argv=None):
    a = SC.parse(argv)
    assert C.COORD_DECIMALS == r2conf.COORD_DECIMALS and C.MAX_RESP_TOKENS == r2conf.MAX_RESP_TOKENS
    t0 = time.time()
    if failed_marker(a).exists():
        print(f"{failed_marker(a)} exists: this sampling failed for good under r2; not re-sampling", flush=True)
        sys.exit(3)
    for old in (SC.failed_marker(a), S1.failed_marker(a)):
        if old.exists():
            raise SystemExit(f"{old} (a v1/r1 failure marker) exists in an r2 work dir: refusing")
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
