from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import train

SCHEDULE_NOTE = "constant learning rate (no warm-up, no decay): Adam with a fixed learning rate (Kurakin et al. 2023)"
OPTIMIZER_NOTE = ("AdamW(betas=(0.9,0.999), eps=1e-8, wd=0) = Adam without weight decay (Kurakin et al. 2023, App. "
                  "'Details of LoRA'); betas / eps are the torch defaults")


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def own_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--lora_alpha", type=int, required=True)
    ap.add_argument("--lr_schedule", choices=["constant", "train_py"], required=True,
                    help="constant = fixed lr; train_py = train.py's warm-up + linear decay (not used by any job)")
    ap.add_argument("--max_target_tokens", type=int, required=True)
    ap.add_argument("--defaults_cfg", required=True, help="frozen_default.json (its SHA-256 goes into train.json)")
    return ap.parse_known_args(argv)


def apply_patches(lora_alpha: int, lr_schedule: str, max_target_tokens: int) -> None:
    import peft
    if lr_schedule == "constant":
        def lr_lambda(T: int, warmup_frac: float = 0.03):
            return lambda s: 1.0
        train.lr_lambda = lr_lambda
    base_load = getattr(train.load_data, "_dplora_default_base", train.load_data)

    def load_data(city, tok, limit=0, decimals=3):
        prompt_ids, resp = base_load(city, tok, limit, decimals)
        if max_target_tokens > 0:
            resp = [r[:max_target_tokens] for r in resp]
        return prompt_ids, resp
    load_data._dplora_default_base = base_load
    train.load_data = load_data
    base_cfg = getattr(peft.LoraConfig, "_dplora_default_base", peft.LoraConfig)

    def LoraConfig(*a, **k):
        k["lora_alpha"] = int(lora_alpha)
        return base_cfg(*a, **k)
    LoraConfig._dplora_default_base = base_cfg
    peft.LoraConfig = LoraConfig


def out_dir_of(rest: list[str]) -> Path:
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--out_dir", required=True)
    a, _ = ap.parse_known_args(rest)
    return Path(a.out_dir)


def record(out_dir: Path, a: argparse.Namespace) -> None:
    tj = out_dir / "train.json"
    info = json.loads(tj.read_text())
    info["lora_alpha"] = int(a.lora_alpha)
    if a.lr_schedule == "constant":
        info["schedule"] = SCHEDULE_NOTE
    info["optimizer"] = OPTIMIZER_NOTE
    info["max_target_tokens"] = int(a.max_target_tokens)
    cfg = Path(a.defaults_cfg).resolve()
    info["defaults"] = {"config": str(cfg), "config_sha256": sha256(cfg), "wrapper": str(Path(__file__).resolve()),
                        "wrapper_sha256": sha256(Path(__file__)), "train_py_sha256": sha256(HERE / "train.py"),
                        "lora_alpha": int(a.lora_alpha), "lr_schedule": a.lr_schedule,
                        "max_target_tokens": int(a.max_target_tokens), "doc": "docs/baselines/DPLORA_DEFAULTS.md"}
    tmp = tj.with_name("train.json.tmp_default")
    tmp.write_text(json.dumps(info, indent=1, default=str))
    os.replace(tmp, tj)


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    a, rest = own_args(argv)
    if not Path(a.defaults_cfg).exists():
        raise SystemExit(f"--defaults_cfg {a.defaults_cfg} does not exist")
    out_dir = out_dir_of(rest)
    apply_patches(a.lora_alpha, a.lr_schedule, a.max_target_tokens)
    sys.argv = [str(HERE / "train.py")] + rest
    train.main()
    record(out_dir, a)
    print(f"train_default: recorded lora_alpha={a.lora_alpha}, schedule={a.lr_schedule}, "
          f"max_target_tokens={a.max_target_tokens} in {out_dir / 'train.json'}", flush=True)


if __name__ == "__main__":
    main()
