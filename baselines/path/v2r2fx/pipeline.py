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
ROWS = {"0.5b": "path_fx", "7b": "path_7b_fx"}
SMOKE, SMOKE_DIR = "jakarta", "_smoke_fx"
QUEUE_SCRIPT = "queue_path_fx.py"
NEVER = {"gw_austin", "gw_nyc"}
REUSE_ROWS = {"0.5b": "path_v2r2", "7b": "path_7b_v2r2"}
REUSE_SMOKE_DIR = "_smoke_v2r2"
REUSE_VERSION = "v2r2"
REUSE_DIR = PROJECT / "baselines" / "path" / "v2r2"
REUSE_FREEZE = PROJECT / "frozen" / "path_v2r2.sha256"


def frozen_sha(freeze: Path, rel: str):
    for ln in freeze.read_text().splitlines():
        if ln.strip() and not ln.startswith("#"):
            h, p = ln.split(maxsplit=1)
            if p.strip() == rel:
                return h
    return None


def reuse_source(dataset: str, eps: float, seed: int, model_key: str) -> tuple[Path, Path]:
    base = RUNS / REUSE_ROWS[model_key]
    if dataset == SMOKE:
        base = base / REUSE_SMOKE_DIR
    stem = f"{dataset}_eps{eps:g}_s{seed}"
    return base / "_work" / stem, base / f"{stem}.json"


def try_reuse(args, work: Path, calib: Path) -> tuple[bool, str]:
    src, srel = reuse_source(args.dataset, args.eps, args.seed, args.model_key)
    checks = []

    def need(cond, what):
        if not cond:
            raise ValueError(f"not reused: {what} fails")
        checks.append(what)
    try:
        for f in ("train_dp.py", "calibrate.py"):
            need(sha256(HERE / f) == sha256(REUSE_DIR / f) == frozen_sha(REUSE_FREEZE, f"baselines/path/v2r2/{f}"),
                 f"{f} byte-identical to the frozen v2r2 file")
        st = json.loads((src / "train.json").read_text())
        ps = st.get("private_stats") or {}
        need(st.get("version") == REUSE_VERSION and st.get("complete") is True and
             st.get("steps_run") == st.get("steps") == 875 and ps.get("oom_retries") == 0,
             "complete v2r2 training run (875 steps, oom_retries 0)")
        need(st.get("dataset") == args.dataset and float(st.get("eps_target")) == float(args.eps) and
             int(st.get("seed")) == args.seed, "same dataset, eps and seed")
        fc = json.loads(calib.read_text())
        sc = json.loads((src / "calib.json").read_text())
        skip = {"version", "calibrate_sec"}
        need({k: v for k, v in sc.items() if k not in skip} == {k: v for k, v in fc.items() if k not in skip},
             "calib.json identical up to the version tag and calibrate_sec")
        need((float(st["noise_multiplier"]), float(st["sample_rate"]), int(st["steps"])) ==
             (float(fc["sigma_sgd"]), float(fc["sample_rate"]), int(fc["steps"])), "train.json matches the calibration")
        need((st.get("text") or {}).get("decimals") == 3, "3-decimal training text")
        rel = json.loads(srel.read_text())
        need(rel.get("version") == REUSE_VERSION and bool(rel.get("adapter_sha256")) and
             (rel.get("training") or {}).get("model") == st.get("model") and
             (rel.get("training") or {}).get("model_revision") == st.get("model_revision"),
             "v2r2 release sidecar with adapter_sha256 and the same backbone")
        cand = [src / "adapter" / "adapter_model.safetensors", src / "adapter_preserved" / "adapter_model.safetensors"]
        wf = next((p for p in cand if p.exists()), None)
        need(wf is not None, "adapter weights kept")
        a = sha256(wf)
        need(a == rel["adapter_sha256"], "adapter SHA-256 equals the v2r2 release sidecar's adapter_sha256")
        cfg = wf.parent / "adapter_config.json"
        cfg = cfg if cfg.exists() else src / "adapter" / "adapter_config.json"
        need(cfg.exists() and json.loads(cfg.read_text()).get("base_model_name_or_path") == st.get("model"),
             "adapter_config.json of the same backbone")
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as ex:
        return False, str(ex)
    ad = work / "adapter"
    ad.mkdir(parents=True, exist_ok=True)
    (ad / "adapter_config.json").write_bytes(cfg.read_bytes())
    target = ad / "adapter_model.safetensors"
    if target.is_symlink() or target.exists():
        target.unlink()
    os.symlink(wf.resolve(), target)
    info = {"ok": True, "rule": "SPEC appendix FX1/F2: identical v2r2 training run reused; only sampling, grounding "
                                "and selection are re-run", "source_work": str(src), "source_release": str(srel),
            "adapter_path": str(wf.resolve()), "adapter_sha256": a, "train_json_sha256": sha256(src / "train.json"),
            "checks": checks, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    (work / "reuse.json").write_text(json.dumps(info, indent=1))
    tmp = work / "train.json.tmp"
    tmp.write_bytes((src / "train.json").read_bytes())
    tmp.rename(work / "train.json")
    return True, f"reused {src} ({len(checks)} checks)"


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
    ap.add_argument("--no_reuse", action="store_true", help="tests only: never reuse a v2r2 training run")
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
        if not tj.exists() and not args.no_reuse:
            ok, why = try_reuse(args, work, calib)
            msg = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] training reuse: {why}"
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()
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
        run([BASE_PY, "-I", "-B", PROJECT / "scripts" / QUEUE_SCRIPT, "validity", out], log, genuine_rc={3})
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
