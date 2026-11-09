from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

R2 = Path(__file__).resolve().parent
sys.path.insert(0, str(R2))
sys.path.insert(0, str(R2.parent / "tests"))
import sample_ctcl_r2 as S2
import ctcl_testlib as TL

import numpy as np

C = S2.C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    import topics as TP
    res = {}
    d = "bandung"
    table_f = C.table_path(d)
    w, n = TP.quotas(np.full(C.K_TOPICS, 10.0), 2000)
    hist = {"dataset": d, "table_sha256": C.sha256_file(table_f), "quota": np.asarray(n).tolist(),
            "w": np.asarray(w).tolist(), "quota_sha256": C.sha256_bytes(np.asarray(n, np.int64).tobytes()),
            "note": "PUBLIC test quota (constant vector), not a DP histogram"}
    hf = out / "hist_public.json"
    hf.write_text(json.dumps(hist))
    pois = TL.public_catalog(d)
    cats = set(pois["category"])
    fa = argparse.Namespace(out=str(out / "fake" / "texts.parquet"), dataset=d, seed=0, n_out=2000, adapter="",
                            kv_gb=C.KV_GB)
    rc = S2.synthesize(fa, hist, np.asarray(n), np.asarray(w), cats, TL.FakeGenerator({k: 0.7 for k in range(1440)},
                                                                                       pois, seed=1), {}, time.time())
    import pandas as pd
    df = pd.read_parquet(fa.out)
    info = json.loads(Path(fa.out).with_suffix(".json").read_text())
    g = info["generation"]
    assert rc == 0 and len(df) == 2000 and g["decode"] == "reject" and g["coord_decimals"] == 3
    assert g["ctcl"]["rev"] == "r2" and g["ctcl"]["r2"]["max_resp_tokens"] == C.MAX_RESP_TOKENS
    assert all(C.textfmt.parse(t, cats, True)[1] == "ok" for t in df["text"])
    res["synthesize_fake"] = {"rc": rc, "n": len(df), "valid_rate": info["valid_rate"]}
    C.MIN_VALID_AFTER = 1000
    snap = Path(C.HF_HOME) / "hub" / "models--Qwen--Qwen2.5-0.5B-Instruct" / "snapshots" / C.MODEL_COMMIT
    argv = ["--dataset", d, "--model_dir", str(snap), "--hist", str(hf), "--table", str(table_f), "--out",
            str(out / "vllm" / "texts.parquet"), "--n_out", "2000"]
    (out / "vllm").mkdir(exist_ok=True)
    t0 = time.time()
    try:
        S2.main(argv)
        code = 0
    except SystemExit as e:
        code = e.code
    fm = out / "vllm" / "texts.failed_r2.json"
    rec = json.loads(fm.read_text()) if fm.exists() else None
    res["vllm_base"] = {"exit": code, "sec": round(time.time() - t0, 1), "failed_r2": rec and {
        k: rec.get(k) for k in ("why", "n_generated", "valid_rate", "reasons", "r1_gate", "generated_tokens",
                                "max_mem_gb", "rev")}}
    ok = code == 3 and rec is not None and "rule (i)" in rec["why"] and rec["r1_gate"]["evaluated"]
    res["ok"] = bool(ok)
    (out / "gpu_check_r2.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, default=str))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
