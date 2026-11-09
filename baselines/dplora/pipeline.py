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
import os
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
DPLORA_PY = f"{_TP_PY_DPLORA}"
TRAJPE_PY = f"{_TP_PY}"


def run(cmd, log):
    print(f"[{time.strftime('%H:%M:%S')}] $ {' '.join(map(str, cmd))}", flush=True)
    env = dict(os.environ)
    py_bin = str(Path(str(cmd[0])).parent)
    env["PATH"] = py_bin + os.pathsep + env.get("PATH", "")
    subprocess.run([str(c) for c in cmd], check=True, stdout=log, stderr=subprocess.STDOUT, env=env)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--eps", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--epochs", type=float, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--out", required=True, help="release parquet")
    ap.add_argument("--work", default="", help="work dir (default: <out dir>/_work/<out stem>)")
    ap.add_argument("--stage", choices=["gpu", "cpu", "all"], default="all")
    ap.add_argument("--mem_cap_gb", type=float, default=0.0)
    ap.add_argument("--gen_batch", type=int, default=1024, help="HF sampler batch size")
    ap.add_argument("--sampler", choices=["vllm", "hf"], default="vllm",
                    help="vllm: sample_vllm.py (trajpe env, continuous batching, LoRA at run time); hf: sample.py")
    ap.add_argument("--kv_gb", type=float, default=8.0, help="vLLM KV-cache size (GiB)")
    ap.add_argument("--jitter", type=int, default=0)
    ap.add_argument("--n_out", type=int, default=50_000)
    ap.add_argument("--log_private", action="store_true")
    ap.add_argument("--variant", choices=["plain", "vp"], default="plain",
                    help="vp = dploravp: DP venue prior for grounding, sigma_SGD recalibrated (vp.py)")
    ap.add_argument("--extra_train", default="", help="extra train.py args (debug), space separated")
    ap.add_argument("--extra_sample", default="", help="extra sample.py args (debug), space separated")
    ap.add_argument("--model", default="")
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--coord_decimals", type=int, default=3)
    ap.add_argument("--inf_steps", default="data")
    ap.add_argument("--inf_clip", type=float, default=0.0)
    ap.add_argument("--tok_budget", type=int, default=8192)
    ap.add_argument("--tgt_budget", type=int, default=4096)
    ap.add_argument("--decode", default="reject")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--min_p", type=float, default=0.0)
    ap.add_argument("--max_new", type=int, default=640)
    ap.add_argument("--ground_k", type=int, default=10)
    ap.add_argument("--ground_scale_km", type=float, default=0.5)
    ap.add_argument("--pop_radius_km", type=float, default=1.0)
    args = ap.parse_args()

    def nondefault(names):
        out = []
        for n in names:
            v = getattr(args, n)
            if v != ap.get_default(n):
                out += [f"--{n}"] if v is True else [f"--{n}", v]
        return out
    train_v2 = nondefault(["model", "grad_ckpt", "coord_decimals", "inf_steps", "inf_clip", "tok_budget",
                           "tgt_budget"])
    sample_v2 = nondefault(["model", "decode", "coord_decimals", "temperature", "top_p", "top_k", "min_p", "max_new"])
    hf_ok = not nondefault(["decode", "top_p", "top_k", "min_p"])
    hf_v2 = nondefault(["model", "temperature", "max_new"])
    ground_v2 = nondefault(["ground_k", "ground_scale_km", "pop_radius_km"])
    out = Path(args.out)
    work = Path(args.work) if args.work else out.parent / "_work" / out.stem
    work.mkdir(parents=True, exist_ok=True)
    log = open(work / "pipeline.log", "a")
    tj, texts = work / "train.json", work / "texts.parquet"

    vp = args.variant == "vp"
    calib, prior = work / "calib.json", work / "prior.npz"
    if args.stage in ("gpu", "all"):
        if vp and not calib.exists():
            run([TRAJPE_PY, "-I", "-B", HERE / "vp.py", "calibrate", "--eps", args.eps, "--batch", args.batch,
                 "--epochs", args.epochs, "--out", calib], log)
        done = tj.exists() and json.loads(tj.read_text()).get("complete") and (work / "adapter").exists()
        if done and (json.loads(tj.read_text()).get("private_stats") or {}).get("oom_retries", 0):
            stamp = time.strftime("%m%d_%H%M%S")
            tj.rename(work / f"train.json.oom_{stamp}")
            (work / "adapter").rename(work / f"adapter.oom_{stamp}")
            texts.unlink(missing_ok=True)
            done = False
        if not done:
            run([DPLORA_PY, "-I", "-B", HERE / "train.py", "--dataset", args.dataset, "--eps", args.eps,
                 "--seed", args.seed, "--batch", args.batch, "--epochs", args.epochs, "--lr", args.lr,
                 "--rank", args.rank, "--clip", args.clip, "--out_dir", work, "--mem_cap_gb", args.mem_cap_gb]
                + (["--log_private"] if args.log_private else []) + (["--calib", calib] if vp else [])
                + train_v2 + args.extra_train.split(), log)
        if not texts.exists() and args.sampler == "vllm":
            try:
                run([TRAJPE_PY, "-I", "-B", HERE / "sample_vllm.py", "--dataset", args.dataset,
                     "--adapter", work / "adapter", "--out", texts, "--seed", args.seed, "--n_out", args.n_out,
                     "--kv_gb", args.kv_gb] + sample_v2 + args.extra_sample.split(), log)
            except subprocess.CalledProcessError as e:
                texts.unlink(missing_ok=True)
                if e.returncode == 3 or not hf_ok:
                    raise
                print("vLLM sampling failed; falling back to HF generate (sample.py)", flush=True)
                log.write("vLLM sampling failed; falling back to HF generate (sample.py)\n")
                log.flush()
        if not texts.exists():
            if not hf_ok:
                raise SystemExit("v2 decoding needs sample_vllm.py (the HF fallback has v1 decoding only)")
            run([DPLORA_PY, "-I", "-B", HERE / "sample.py", "--dataset", args.dataset, "--adapter", work / "adapter",
                 "--out", texts, "--seed", args.seed, "--n_out", args.n_out, "--gen_batch", args.gen_batch,
                 "--mem_cap_gb", args.mem_cap_gb] + hf_v2, log)
    if args.stage in ("cpu", "all"):
        if vp and not prior.exists():
            run([TRAJPE_PY, "-I", "-B", HERE / "vp.py", "prior", "--dataset", args.dataset, "--calib", calib,
                 "--seed", args.seed, "--out", prior], log)
        if not out.exists():
            run([TRAJPE_PY, "-I", "-B", HERE / "ground.py", "--dataset", args.dataset, "--texts", texts,
                 "--train_json", tj, "--out", out, "--seed", args.seed, "--jitter", args.jitter,
                 "--n_out", args.n_out] + (["--prior", prior, "--method", "dploravp"] + ground_v2 if vp else []), log)
        ev = out.with_suffix(".eval.json")
        if not ev.exists():
            run([TRAJPE_PY, "-I", "-B", PROJECT / "scripts" / "eval_synth.py", "--dataset", args.dataset,
                 "--synth", out, "--out", ev], log)
        run([TRAJPE_PY, "-I", "-B", HERE / "score.py", out], log)
    log.close()


if __name__ == "__main__":
    main()
