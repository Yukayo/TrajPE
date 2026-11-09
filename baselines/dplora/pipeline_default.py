from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pipeline

TRAIN_PY = HERE / "train.py"
TRAIN_DEFAULT = HERE / "train_default.py"


def own_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--lora_alpha", type=int, required=True)
    ap.add_argument("--lr_schedule", choices=["constant", "train_py"], required=True)
    ap.add_argument("--max_target_tokens", type=int, required=True)
    ap.add_argument("--defaults_cfg", required=True)
    return ap.parse_known_args(argv)


def substitute(cmd: list, extra: list[str]) -> list:
    if len(cmd) > 3 and Path(str(cmd[3])).resolve() == TRAIN_PY.resolve():
        return list(cmd[:3]) + [TRAIN_DEFAULT] + list(cmd[4:]) + extra
    return list(cmd)


def work_dir(rest: list[str]) -> Path:
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="")
    a, _ = ap.parse_known_args(rest)
    out = Path(a.out)
    return Path(a.work) if a.work else out.parent / "_work" / out.stem


def guard_stale(work: Path, cfg_sha: str) -> None:
    tj = work / "train.json"
    if not tj.exists():
        return
    t = json.loads(tj.read_text())
    if not t.get("complete") or (t.get("defaults") or {}).get("config_sha256") == cfg_sha:
        return
    stamp = time.strftime("%m%d_%H%M%S")
    tj.rename(work / f"train.json.stale_{stamp}")
    if (work / "adapter").exists():
        (work / "adapter").rename(work / f"adapter.stale_{stamp}")
    for f in ("texts.parquet", "texts.json"):
        (work / f).unlink(missing_ok=True)
    print(f"pipeline_default: moved aside a train.json without this config's defaults block ({stamp})", flush=True)


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    a, rest = own_args(argv)
    cfg = Path(a.defaults_cfg).resolve()
    cfg_sha = hashlib.sha256(cfg.read_bytes()).hexdigest()
    extra = ["--lora_alpha", str(a.lora_alpha), "--lr_schedule", a.lr_schedule, "--max_target_tokens",
             str(a.max_target_tokens), "--defaults_cfg", str(cfg)]
    work = work_dir(rest)
    work.mkdir(parents=True, exist_ok=True)
    guard_stale(work, cfg_sha)
    base_run = getattr(pipeline.run, "_dplora_default_base", pipeline.run)

    def run(cmd, log):
        return base_run(substitute(cmd, extra), log)
    run._dplora_default_base = base_run
    pipeline.run = run
    sys.argv = [str(HERE / "pipeline.py")] + rest
    pipeline.main()


if __name__ == "__main__":
    main()
