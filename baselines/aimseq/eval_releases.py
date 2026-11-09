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
import subprocess
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
TRAJPE_PY = f"{_TP_PY}"
DOWN_PY = f"{_TP_PY_GRID}"
SCORE_METRICS = ["density_jsd", "od_jsd", "query_error", "travel_jsd", "hop_jsd", "gyration_jsd", "length_jsd",
                 "hour_jsd", "gap_jsd", "cat_jsd", "root_trans_jsd", "hour_root_jsd", "weekhour_jsd",
                 "pattern_f1", "poi_kendall"]


def composite(m: dict) -> float:
    vals = []
    for k in SCORE_METRICS:
        v = m[k]
        if k == "poi_kendall" and not np.isfinite(v):
            v = 0.0
        vals.append(np.log10(max(1.0 - v if k in ("pattern_f1", "poi_kendall") else v, 1e-3)))
    return float(np.mean(vals))


def run(cmd, log: Path, cores: str, threads: int, env_extra=None):
    env = {**os.environ, "OMP_NUM_THREADS": str(threads), "OPENBLAS_NUM_THREADS": str(threads),
           "MKL_NUM_THREADS": str(threads), **(env_extra or {})}
    t = time.time()
    with open(log, "w") as f:
        r = subprocess.run(["taskset", "-c", cores] + cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    if r.returncode:
        raise RuntimeError(f"failed ({r.returncode}): {' '.join(cmd)} -> see {log}")
    return time.time() - t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--cores", default="76-79")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--skip", default="", help="comma list of steps to skip: fid,markov,bench")
    args = ap.parse_args()
    skip = set(filter(None, args.skip.split(",")))
    out = args.out_dir
    (out / "bench").mkdir(parents=True, exist_ok=True)
    jobs = json.loads(args.jobs.read_text())
    env_t = {"PATH": f"{os.environ.get('PATH', '')}", "PYTHONDONTWRITEBYTECODE": "1"}
    rows = []
    for j in jobs:
        name, synth = j["name"], j["synth"]
        ev, mk, bj = out / f"{name}.eval.json", out / f"{name}.markov.json", out / "bench" / f"{name}.json"
        if "fid" not in skip and not ev.exists():
            dt = run([TRAJPE_PY, "-I", "-B", str(PROJECT / "scripts" / "eval_synth.py"), "--dataset", args.dataset,
                      "--synth", synth, "--out", str(ev)], out / f"{name}.eval.log", args.cores, args.threads, env_t)
            print(f"{name}: eval_synth {dt:.0f}s", flush=True)
        if "markov" not in skip and not mk.exists():
            dt = run([DOWN_PY, "-I", str(PROJECT / "downstream" / "train_eval.py"), "--dataset", args.dataset,
                      "--train", synth, "--model", "markov", "--out", str(mk), "--bench-metrics"],
                     out / f"{name}.markov.log", args.cores, args.threads, {"CUDA_VISIBLE_DEVICES": ""})
            print(f"{name}: markov {dt:.0f}s", flush=True)
        if "bench" not in skip and not bj.exists():
            dt = run([TRAJPE_PY, "-I", "-B", str(PROJECT / "scripts" / "eval_bench.py"), "--dataset", args.dataset,
                      "--synth", synth, "--method", j["method"], "--eps", str(j["eps"]), "--seed", str(j["seed"]),
                      "--name", name, "--out-dir", str(out / "bench")] + (["--downstream", str(mk)] if mk.exists()
                                                                            else []),
                     out / "bench" / f"{name}.log", args.cores, args.threads, env_t)
            print(f"{name}: bench {dt:.0f}s", flush=True)
        row = {k: j.get(k) for k in ("name", "method", "eps", "seed", "synth")}
        if ev.exists():
            e = json.loads(ev.read_text())
            row["composite"] = composite(e["vs_train"])
            row["fidelity_vs_train"] = {k: e["vs_train"][k] for k in SCORE_METRICS + ["cat_jsd", "root_jsd"]}
        if mk.exists():
            m = json.loads(mk.read_text())
            row["markov_mrr"] = m["metrics"]["mrr"]
            row["markov_cat_mrr"] = m["metrics"].get("cat_mrr", m["metrics"].get("category", {}).get("mrr"))
        side = Path(synth).with_suffix(".json")
        if side.exists():
            s = json.loads(side.read_text())
            row["runtime_sec"] = s.get("runtime_sec")
            row["n_checkins"] = s.get("n_checkins")
        rows.append(row)
    (out / f"summary_{args.jobs.stem}.json").write_text(json.dumps(rows, indent=1, default=float))
    lines = ["| name | method | eps | seed | composite | markov MRR | ck/trail | runtime s |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ck = (r.get("n_checkins") or 0) / 50000 if r.get("n_checkins") else float("nan")
        lines.append(f"| {r['name']} | {r['method']} | {r['eps']} | {r['seed']} | {r.get('composite', float('nan')):.4f} "
                     f"| {r.get('markov_mrr', float('nan')):.4f} | {ck:.3f} | {r.get('runtime_sec') or float('nan'):.0f} |")
    (out / f"summary_{args.jobs.stem}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
