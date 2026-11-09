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

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("HF_HOME", f"{_TP_WORK}/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT / "scripts"))

import pipeline_default as PD
import train
import train_default as TD

CFG = HERE / "frozen_default.json"


def ok(msg):
    print("PASS", msg, flush=True)


def test_args():
    argv = ["--dataset", "jakarta", "--eps", "4", "--lr", "0.0003", "--rank", "8", "--lora_alpha", "32",
            "--lr_schedule", "constant", "--max_target_tokens", "512", "--defaults_cfg", str(CFG), "--out_dir", "/x"]
    for mod in (TD, PD):
        a, rest = mod.own_args(argv)
        assert (a.lora_alpha, a.lr_schedule, a.max_target_tokens, a.defaults_cfg) == (32, "constant", 512, str(CFG))
        assert rest == ["--dataset", "jakarta", "--eps", "4", "--lr", "0.0003", "--rank", "8", "--out_dir", "/x"], rest
    ok("(a) own flags split off; --lr passed on unabbreviated; order kept")


def test_substitute():
    extra = ["--lora_alpha", "32", "--lr_schedule", "constant", "--max_target_tokens", "512", "--defaults_cfg", "C"]
    tr = ["/py", "-I", "-B", HERE / "train.py", "--dataset", "d", "--eps", 1.0]
    got = PD.substitute(tr, extra)
    assert got == ["/py", "-I", "-B", HERE / "train_default.py", "--dataset", "d", "--eps", 1.0] + extra, got
    for other in (["/py", "-I", "-B", HERE / "sample_vllm.py", "--dataset", "d"],
                  ["/py", "-I", "-B", HERE / "vp.py", "calibrate", "--eps", 1.0],
                  ["/py", "-I", "-B", HERE / "ground.py", "--dataset", "d"]):
        assert PD.substitute(other, extra) == other
    ok("(b) only the train.py command is replaced")


def test_patches():
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    TD.apply_patches(32, "constant", 512)
    TD.apply_patches(32, "constant", 512)

    def build(rank):
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0, target_modules=train.LORA_TARGETS,
                         bias="none", task_type="CAUSAL_LM")
        return cfg, get_peft_model
    cfg, get_peft_model = build(8)
    assert cfg.lora_alpha == 32 and cfg.r == 8 and cfg.lora_dropout == 0.0 and cfg.bias == "none"
    assert set(cfg.target_modules) == set(train.LORA_TARGETS)
    torch.manual_seed(0)
    m = Qwen2ForCausalLM(Qwen2Config(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                     num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128))
    pm = get_peft_model(m, cfg)
    seen = set()
    for n, mod in pm.named_modules():
        if hasattr(mod, "scaling") and isinstance(mod.scaling, dict) and "default" in mod.scaling:
            assert abs(mod.scaling["default"] - 4.0) < 1e-12, (n, mod.scaling)
            seen.add(n.split(".")[-1])
    assert seen == set(train.LORA_TARGETS), seen
    p = torch.nn.Parameter(torch.zeros(3))
    opt = torch.optim.AdamW([p], lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, train.lr_lambda(729))
    lrs = []
    for _ in range(729):
        lrs.append(sched.get_last_lr()[0])
        opt.step()
        sched.step()
    assert max(lrs) == min(lrs) == 3e-4, (min(lrs), max(lrs))
    ok("(c) lora_alpha 32 (scaling 4 on all 7 projection types), rank/dropout/targets unchanged, lr fixed over 729 steps")


def test_truncation():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(train.MODEL, revision=train.hf_revision(train.MODEL))
    base = train.load_data._dplora_default_base
    p0, r0 = base("jakarta", tok, 400, 3)
    TD.apply_patches(32, "constant", 60)
    p1, r1 = train.load_data("jakarta", tok, 400, 3)
    assert p1 == p0 and len(r1) == len(r0)
    assert all(len(b) <= 60 and b == a[:60] for a, b in zip(r0, r1))
    n_cut, n_kept = sum(len(a) > 60 for a in r0), sum(len(a) <= 60 for a in r0)
    assert n_cut > 0 and n_kept > 0 and all(a == b for a, b in zip(r0, r1) if len(a) <= 60)
    TD.apply_patches(32, "constant", 512)
    p2, r2 = train.load_data("jakarta", tok, 0, 3)
    assert max(len(r) for r in r2) <= 512
    _, rfull = base("jakarta", tok, 0, 3)
    assert r2 == rfull, "512 must leave the Jakarta targets unchanged (longest = 498 tokens)"
    ok(f"(d) truncation: L=60 cuts {n_cut}/400 targets and keeps {n_kept} unchanged; L=512 leaves all "
       f"{len(rfull)} Jakarta targets unchanged (max {max(len(r) for r in rfull)} tokens)")


def test_e2e():
    import hashlib
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
        out = Path(d) / "work"
        argv = ["--dataset", "jakarta", "--eps", "4", "--seed", "0", "--batch", "1024", "--epochs", "10",
                "--lr", "0.0003", "--rank", "8", "--clip", "1.0", "--out_dir", str(out), "--limit", "64",
                "--max_steps", "2", "--lora_alpha", "32", "--lr_schedule", "constant", "--max_target_tokens", "512",
                "--defaults_cfg", str(CFG)]
        TD.main(argv)
        t = json.loads((out / "train.json").read_text())
        assert t["lora_alpha"] == 32 and t["max_target_tokens"] == 512 and t["schedule"] == TD.SCHEDULE_NOTE
        assert t["optimizer"] == TD.OPTIMIZER_NOTE and t["lora_dropout"] == 0.0 and t["rank"] == 8
        assert t["defaults"]["config_sha256"] == hashlib.sha256(CFG.read_bytes()).hexdigest()
        assert t["defaults"]["train_py_sha256"] == hashlib.sha256((HERE / "train.py").read_bytes()).hexdigest()
        assert abs(t["sample_rate"] - 1024 / 74685) < 1e-15 and t["steps"] == 729 and t["steps_run"] == 2
        assert t["complete"] is False and t["batch_ref"] == 1024 and t["epochs"] == 10 and t["lr"] == 0.0003
        assert abs(t["noise_multiplier"] - 0.8319) < 5e-4 and t["eps_prv"] <= 4.0, (t["noise_multiplier"], t["eps_prv"])
        ac = json.loads((out / "adapter" / "adapter_config.json").read_text())
        assert ac["lora_alpha"] == 32 and ac["r"] == 8 and ac["lora_dropout"] == 0.0
        assert set(ac["target_modules"]) == set(train.LORA_TARGETS)
    ok("(e) end to end (CPU, 2 of 729 steps): train.json and adapter_config.json record alpha 32, constant "
       f"schedule, max_target_tokens 512, defaults block; sigma {t['noise_multiplier']:.4f}, PRV {t['eps_prv']:.4f}")




def main():
    test_args()
    test_substitute()
    test_patches()
    test_truncation()
    if "--skip_e2e" not in sys.argv:
        test_e2e()
    print("all dplora_def checks passed")


if __name__ == "__main__":
    main()
