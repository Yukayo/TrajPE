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
import hashlib
import inspect
import itertools
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from downstream.data import data_version
from downstream.markov import MarkovBackoff
from downstream.transformer import TransformerConfig

PYTHON = f"{_TP_PY_GRID}"
DATA = HERE.parent / "data" / "processed" / "jakarta"
METRICS = ("mrr", "acc@1", "acc@5", "acc@10", "ndcg@10")


def defaults(model: str) -> dict:
    if model == "transformer":
        return asdict(TransformerConfig())
    sig = inspect.signature(MarkovBackoff.__init__).parameters
    return {k: p.default for k, p in sig.items() if p.default is not inspect.Parameter.empty and k != "seed"}


def effective(model: str, overrides: dict) -> dict:
    return {**defaults(model), **overrides}


def run_one(model: str, overrides: dict, out_dir: Path, seed: int, max_gpu_gb: float) -> dict:
    cfg = effective(model, overrides)
    key = {"model": model, "config": cfg, "data": data_version(DATA)}
    tag = hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10]
    out = out_dir / f"{model}_{tag}_s{seed}.json"
    if not out.exists():
        cmd = [PYTHON, "-I", str(HERE / "train_eval.py"), "--dataset", "jakarta", "--train", str(DATA / "train.parquet"),
               "--model", model, "--eval-split", "val", "--seed", str(seed), "--out", str(out),
               "--max-gpu-gb", str(max_gpu_gb), "--set"] + [f"{k}={json.dumps(v)}" for k, v in cfg.items()]
        if model == "transformer":
            cmd += ["--early-stop", "holdout"]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "1"))
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        (out_dir / f"{out.stem}.log").write_text(proc.stdout + proc.stderr)
        if proc.returncode != 0:
            return {"delta": overrides, "error": proc.stderr.strip().splitlines()[-1:]}
    return summarize(model, json.loads(out.read_text()))


def summarize(model: str, res: dict) -> dict:
    info = res["train_info"]
    used = info["config"] if model == "transformer" else info["hparams"]
    delta = {k: v for k, v in used.items() if defaults(model).get(k, v) != v}
    delta.update({k: "<absent>" for k in defaults(model) if k not in used and k not in ("kernel_iters", "kernel_events")})
    return {"delta": delta, "seed": res["seed"], "epoch": info.get("selected_epoch"),
            "train_s": res["runtime_s"]["train"], **{k: res["metrics"][k] for k in METRICS}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["transformer", "markov"])
    ap.add_argument("--grid", default="{}", help="JSON dict: hyper-parameter -> list of values (full product)")
    ap.add_argument("--configs", default=None, help="JSON list of override dicts (instead of --grid)")
    ap.add_argument("--report", action="store_true", help="only list all cached runs of --model")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--max-gpu-gb", type=float, default=5.0, help="per-run GPU cap (sum must stay <= 20 GB)")
    ap.add_argument("--out-dir", type=Path, default=Path(f"{_TP_WORK}/runs/downstream/tune"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.report:
        rows = []
        for f in sorted(args.out_dir.glob(f"{args.model}_*.json")):
            res = json.loads(f.read_text())
            if res.get("dataset") == "jakarta" and res.get("eval_split") == "val" \
                    and res.get("data_version") == data_version(DATA):
                rows.append(summarize(args.model, res))
    else:
        grid = json.loads(args.grid)
        configs = json.loads(args.configs) if args.configs else \
            [dict(zip(grid, vals)) for vals in itertools.product(*grid.values())]
        with ThreadPoolExecutor(args.parallel) as pool:
            rows = list(pool.map(lambda c: run_one(args.model, c, args.out_dir, args.seed, args.max_gpu_gb), configs))
    rows.sort(key=lambda r: -r.get("mrr", -1))
    print(f"jakarta val (data {data_version(DATA)}), {args.model}; 'delta' = difference from the current defaults "
          f"{json.dumps(defaults(args.model))}")
    for r in rows:
        if "error" in r:
            print(f"ERROR {r['delta']}: {r['error']}")
            continue
        print("  ".join(f"{k}={r[k]:.4f}" for k in METRICS) +
              f"  seed={r['seed']} epoch={r['epoch']} train_s={r['train_s']}  {json.dumps(r['delta'])}")


if __name__ == "__main__":
    main()
