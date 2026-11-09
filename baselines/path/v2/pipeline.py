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
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
DPLORA_PY = f"{_TP_PY_DPLORA}"
TRAJPE_PY = f"{_TP_PY}"
BASE_PY = f"{_TP_PY_GRID}"
RUNS = Path(f"{_TP_WORK}/runs/baselines_v2")
ROWS = {"0.5b": "path", "7b": "path_7b"}
SMOKE, SMOKE_DIR = "jakarta", "_smoke_v2"
NEVER = {"gw_austin", "gw_nyc"}


class Genuine(Exception):
    pass


def run(cmd, log, extra_env=None, genuine_rc=None):
    print(f"[{time.strftime('%H:%M:%S')}] $ {' '.join(map(str, cmd))}", flush=True)
    log.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] $ {' '.join(map(str, cmd))}\n")
    log.flush()
    env = dict(os.environ)
    env.update({"HF_HOME": f"{_TP_WORK}/hf_cache", "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
                "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"})
    env.update(extra_env or {})
    env["PATH"] = str(Path(str(cmd[0])).parent) + os.pathsep + env.get("PATH", "")
    r = subprocess.run([str(c) for c in cmd], stdout=log, stderr=subprocess.STDOUT, env=env, cwd="/tmp")
    if r.returncode != 0:
        if genuine_rc is not None and r.returncode in genuine_rc:
            raise Genuine(f"{Path(str(cmd[3])).name} exit {r.returncode}")
        raise subprocess.CalledProcessError(r.returncode, cmd)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--eps", type=float, required=True)
    ap.add_argument("--model_key", default="0.5b", choices=list(ROWS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--work", default="")
    ap.add_argument("--mem_cap_gb", type=float, default=0.0, help="training: cap on the process's CUDA memory")
    ap.add_argument("--kv_gb", type=float, default=8.0, help="vLLM KV cache (GiB)")
    ap.add_argument("--keep_adapter", action="store_true", help="default for the Jakarta smoke runs")
    args = ap.parse_args()
    if args.dataset in NEVER:
        raise SystemExit(f"{args.dataset} is never read or run")
    smoke = args.dataset == SMOKE
    e = f"{args.eps:g}"
    base = RUNS / ROWS[args.model_key] / SMOKE_DIR if smoke else RUNS / ROWS[args.model_key]
    out = Path(args.out) if args.out else base / f"{args.dataset}_eps{e}_s{args.seed}.parquet"
    work = Path(args.work) if args.work else out.parent / "_work" / out.stem
    work.mkdir(parents=True, exist_ok=True)
    failed = work / "FAILED"
    if failed.exists():
        print(f"genuine failure recorded earlier, not repeated: {failed.read_text().strip()}", flush=True)
        sys.exit(3)
    log = open(work / "pipeline.log", "a")
    calib, tj, pool, adapter = work / "calib.json", work / "train.json", work / "pool.parquet", work / "adapter"
    common = ["--dataset", args.dataset, "--model_key", args.model_key]
    try:
        if not calib.exists():
            run([TRAJPE_PY, "-I", "-B", HERE / "calibrate.py", "--eps", args.eps, "--out", calib], log)
        done = tj.exists() and json.loads(tj.read_text()).get("complete") is True and \
            (adapter / "adapter_model.safetensors").exists()
        if done and (json.loads(tj.read_text()).get("private_stats") or {}).get("oom_retries", 0):
            stamp = time.strftime("%m%d_%H%M%S")
            tj.rename(work / f"train.json.oom_{stamp}")
            adapter.rename(work / f"adapter.oom_{stamp}")
            pool.unlink(missing_ok=True)
            done = False
        if not done:
            pool.unlink(missing_ok=True)
            run([DPLORA_PY, "-I", "-B", HERE / "train_dp.py", "--eps", args.eps, "--calib", calib, "--out_dir", work,
                 "--seed", args.seed, "--mem_cap_gb", args.mem_cap_gb] + common
                + (["--log_private"] if smoke else []), log,
                extra_env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
        if not pool.exists():
            run([TRAJPE_PY, "-I", "-B", HERE / "sample_pool.py", "--adapter", adapter, "--out", pool,
                 "--seed", args.seed, "--kv_gb", args.kv_gb] + common, log, genuine_rc={3})
        if not out.exists():
            run([TRAJPE_PY, "-I", "-B", HERE / "release.py", "--eps", args.eps, "--work", work, "--out", out,
                 "--seed", args.seed] + common, log)
        run([TRAJPE_PY, "-I", "-B", HERE / "pld_check.py", out.with_suffix(".json")], log, genuine_rc={3})
        run([BASE_PY, "-I", "-B", PROJECT / "scripts" / "queue_path.py", "validity", out], log, genuine_rc={3})
    except Genuine as ex:
        failed.write_text(f"{ex} {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        print(f"GENUINE FAILURE: {ex} (see {work / 'pipeline.log'})", flush=True)
        sys.exit(3)
    finally:
        log.close()
    if not (args.keep_adapter or smoke) and (adapter / "adapter_model.safetensors").exists():
        man = {"note": "adapter weights deleted after the release passed every check (disk); kept: config",
               "files": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in sorted(adapter.iterdir())
                         if p.is_file()},
               "deleted": "adapter_model.safetensors", "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        rel = json.loads(out.with_suffix(".json").read_text())
        if man["files"]["adapter_model.safetensors"]["sha256"] != rel["adapter_sha256"]:
            raise SystemExit("adapter differs from the one recorded in the release sidecar")
        (work / "adapter_manifest.json").write_text(json.dumps(man, indent=1))
        (adapter / "adapter_model.safetensors").unlink()
    print(f"[{time.strftime('%H:%M:%S')}] PATH release done: {out}", flush=True)


if __name__ == "__main__":
    main()
