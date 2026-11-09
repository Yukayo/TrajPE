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

import copy
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", f"{_TP_WORK}/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))
import ctcl_testlib as TL

C = TL.C
sys.path.insert(0, str(TL.CTCL))
import train_ctcl as TR

import numpy as np
import torch
import torch.nn.functional as F

TINY = TL.SCRATCH / "tiny_model"
V = 151936


def tiny_model(seed=0):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    cfg = Qwen2Config(vocab_size=V, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, tie_word_embeddings=True, max_position_embeddings=2048, rms_norm_eps=1e-6,
                      attention_dropout=0.0)
    torch.manual_seed(seed)
    m = Qwen2ForCausalLM(cfg)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n:
                p.add_(0.3 * torch.randn_like(p))
    m.config._attn_implementation = "sdpa"
    return m.float().train()


def rand_example(g, P, R):
    return (torch.randint(0, V, (P,), generator=g), torch.randint(0, V, (R,), generator=g))


def reference_grad(model, prompt, resp):
    ref = copy.deepcopy(model).double()
    assert ref.lm_head.weight is ref.model.embed_tokens.weight
    ids = torch.cat([prompt, resp]).unsqueeze(0)
    labels = torch.cat([torch.full_like(prompt, -100), resp]).unsqueeze(0)
    loss = ref(input_ids=ids, labels=labels, use_cache=False).loss
    names = [n for n, _ in ref.named_parameters()]
    gr = torch.autograd.grad(loss, [p for _, p in ref.named_parameters()])
    return dict(zip(names, gr)), float(loss)


def impl_grad(model, prompt, resp, clip=1e9):
    core, head, params = TR.model_parts(model, "full")
    cs = TR.ClippedSum(params, clip)
    loss = TR.example_loss(core, head, prompt, resp)
    norm = cs.add(loss)
    names = [n for n, _ in model.named_parameters()]
    return dict(zip(names, cs.acc)), float(loss), float(norm), cs


def test_exact():
    m = tiny_model()
    g = torch.Generator().manual_seed(1)
    worst = 0.0
    for P, R in ((17, 23), (40, 5), (9, 64)):
        prompt, resp = rand_example(g, P, R)
        ref, lref = reference_grad(m, prompt, resp)
        imp, limp, _, _ = impl_grad(m, prompt, resp)
        assert abs(lref - limp) < 1e-5, (lref, limp)
        assert set(ref) == set(imp) and "lm_head.weight" not in imp
        for n in ref:
            a, b = imp[n].double(), ref[n]
            err = float((a - b).norm() / (b.norm() + 1e-30))
            worst = max(worst, err)
            assert err < 2e-4, (n, err)
        for n in ("model.embed_tokens.weight", "model.norm.weight", "model.layers.0.input_layernorm.weight",
                  "model.layers.1.post_attention_layernorm.weight", "model.layers.0.self_attn.q_proj.bias"):
            assert float(imp[n].norm()) > 0, f"{n} got no gradient"
    prompt, resp = rand_example(g, 12, 10)
    full, _, _, _ = impl_grad(m, prompt, resp)
    core, head = m.model, m.lm_head
    for p in m.parameters():
        p.grad = None
    emb = core.embed_tokens(torch.cat([prompt, resp[:-1]]).unsqueeze(0)).detach()
    hid = core(inputs_embeds=emb, use_cache=False).last_hidden_state
    F.cross_entropy(head(hid[:, len(prompt) - 1:]).float()[0], resp).backward()
    head_only = m.model.embed_tokens.weight.grad.clone()
    assert float((full["model.embed_tokens.weight"] - head_only).norm()) > 1e-6
    print(f"per-example gradients exact for all {len(ref)} parameters (worst relative error {worst:.1e})")


def test_clip_batching_neighbours():
    m = tiny_model(3)
    g = torch.Generator().manual_seed(4)
    exs = [rand_example(g, P, R) for P, R in ((10, 30), (25, 7), (14, 50), (31, 12))]
    core, head, params = TR.model_parts(m, "full")
    raw = [impl_grad(m, p, r)[2] for p, r in exs]
    C_ = 0.5 * min(raw)
    alone = []
    for p, r in exs:
        cs = TR.ClippedSum(params, C_)
        cs.add(TR.example_loss(core, head, p, r))
        n = float(torch.linalg.vector_norm(torch.stack(torch._foreach_norm(cs.acc))))
        assert n <= C_ * (1 + 1e-6), (n, C_)
        alone.append([a.clone() for a in cs.acc])
    big = TR.ClippedSum(params, 1e9)
    big.add(TR.example_loss(core, head, *exs[0]))
    TL.assert_close(float(torch.linalg.vector_norm(torch.stack(torch._foreach_norm(big.acc)))), raw[0], 1e-4 * raw[0],
                    "unclipped norm")

    def summed(order):
        cs = TR.ClippedSum(params, C_)
        for i in order:
            cs.add(TR.example_loss(core, head, *exs[i]))
        return [a.clone() for a in cs.acc]
    ref = [sum(al[j] for al in alone) for j in range(len(params))]
    for order in ((0, 1, 2, 3), (3, 1, 0, 2), (2, 3, 1, 0)):
        s = summed(order)
        err = max(float((a - b).abs().max()) for a, b in zip(s, ref))
        assert err < 1e-6, err
    s3 = summed((0, 1, 2))
    s4 = summed((0, 1, 2, 3))
    diff = [a - b for a, b in zip(s4, s3)]
    dn = float(torch.linalg.vector_norm(torch.stack(torch._foreach_norm(diff))))
    assert dn <= C_ * (1 + 1e-6)
    assert max(float((d - a).abs().max()) for d, a in zip(diff, alone[3])) < 1e-6
    rec = {}

    class Rec:
        param_groups = [{"lr": 0.0}]

        def __init__(self, tag):
            self.tag = tag

        def step(self):
            rec[self.tag] = [p.grad.clone() for p in params]
    for tag, order in (("without", (0, 1, 2)), ("with", (0, 1, 2, 3))):
        cs = TR.ClippedSum(params, C_)
        for i in order:
            cs.add(TR.example_loss(core, head, *exs[i]))
        gen = torch.Generator().manual_seed(7919)
        TR.noisy_update(Rec(tag), cs, 0.7, 128.0, 1e-4, gen)
    d = [a - b for a, b in zip(rec["with"], rec["without"])]
    assert max(float((x * 128.0 - a).abs().max()) for x, a in zip(d, alone[3])) < 1e-5
    print(f"clipping (<= C over all parameters), order/batching invariance and the neighbouring-dataset bound hold")


def test_noise_and_schedule():
    m = tiny_model(5)
    _, _, params = TR.model_parts(m, "full")
    out = {}

    class Rec:
        param_groups = [{"lr": 0.0}]

        def step(self):
            out["g"] = torch.cat([p.grad.flatten() for p in params]).clone()
            out["lr"] = self.param_groups[0]["lr"]
    for seed in (11, 11, 12):
        cs = TR.ClippedSum(params, 1.0)
        gen = torch.Generator().manual_seed(seed)
        TR.noisy_update(Rec(), cs, 0.55, 128.0, 3e-4, gen)
        out.setdefault("all", []).append(out["g"])
    g = out["all"][0]
    TL.assert_close(float(g.std()), 0.55 / 128.0, 0.01 * 0.55 / 128.0, "noise std sigma C / B_nom")
    TL.assert_close(float(g.mean()), 0.0, 5e-6, "noise mean")
    assert torch.equal(out["all"][0], out["all"][1]) and not torch.equal(out["all"][0], out["all"][2])
    assert out["lr"] == 3e-4
    lr = [C.lr_at(s) for s in range(1166)]
    assert lr[0] == 0.0 and abs(lr[29] - 4e-4 * 29 / 58) < 1e-15 and lr[58] == 4e-4
    assert abs(lr[1165] - 4e-4 / 1108) < 1e-15 and all(x > 0 for x in lr[1:])
    assert max(lr) == 4e-4 and lr.index(max(lr)) == 58
    assert all(lr[s + 1] >= lr[s] for s in range(57)) and all(lr[s + 1] <= lr[s] for s in range(58, 1165))
    lp = [C.lr_at(s, "paper") for s in (0, 50, 100, 1999)]
    assert lp == [0.0, 2e-4, 4e-4, 4e-4 / 1900]
    p0 = torch.tensor([0.3, -0.2, 0.05])
    g1, g2 = torch.tensor([0.1, 0.2, -0.3]), torch.tensor([-0.05, 0.4, 0.1])
    w = torch.nn.Parameter(p0.clone())
    opt = torch.optim.AdamW([w], lr=0.0, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, foreach=True)
    for grad, lr_ in ((g1, 0.0), (g2, 2e-4)):
        w.grad = grad.clone()
        opt.param_groups[0]["lr"] = lr_
        opt.step()
    m1, v1 = 0.1 * g1, 0.001 * g1 ** 2
    m2, v2 = 0.9 * m1 + 0.1 * g2, 0.999 * v1 + 0.001 * g2 ** 2
    mh, vh = m2 / (1 - 0.9 ** 2), v2 / (1 - 0.999 ** 2)
    want = p0 - 2e-4 * (mh / (vh.sqrt() + 1e-8) + 0.01 * p0)
    assert torch.allclose(w.detach(), want, atol=1e-9), (w, want)
    print("noise N(0, sigma^2 C^2) / B_nom seeded; schedule and AdamW as the code's optax")


def test_lora():
    from peft import LoraConfig, get_peft_model
    m = get_peft_model(tiny_model(6), LoraConfig(r=C.LORA["r"], lora_alpha=C.LORA["alpha"],
                                                 lora_dropout=C.LORA["dropout"], target_modules=C.LORA["targets"],
                                                 bias="none", task_type="CAUSAL_LM"))
    core, head, params = TR.model_parts(m, "lora")
    names = [n for n, p in m.named_parameters() if p.requires_grad]
    assert names and all("lora_" in n for n in names) and len(names) == len(params)
    assert not m.base_model.model.model.embed_tokens.weight.requires_grad
    cs = TR.ClippedSum(params, 1.0)
    g = torch.Generator().manual_seed(9)
    cs.add(TR.example_loss(core, head, *rand_example(g, 12, 9)))
    assert float(torch.linalg.vector_norm(torch.stack(torch._foreach_norm(cs.acc)))) <= 1.0 + 1e-6
    sys.path.insert(0, str(C.DPLORA))
    import train as dtrain
    fz = json.loads((C.DPLORA / "frozen_v2.json").read_text())
    assert C.LORA["targets"] == dtrain.LORA_TARGETS and C.LORA["r"] == fz["eps"]["1"]["rank"] == fz["eps"]["4"]["rank"]
    assert C.LORA["alpha"] == 2 * C.LORA["r"] and C.LORA["dropout"] == 0.0
    print(f"LoRA fallback: {len(names)} adapter tensors trainable, base frozen")


def test_cli_and_full_run():
    shutil.rmtree(TINY, ignore_errors=True)
    tiny_model(0).save_pretrained(TINY)
    root = TL.SCRATCH / "train_run"
    shutil.rmtree(root, ignore_errors=True)
    TL.fake_dataset(root / "data", "bandung", 500, seed=3)
    work = root / "work"
    work.mkdir(parents=True)
    calib = work / "calib.json"
    r = subprocess.run([C.TRAJPE_PY, "-I", "-B", str(TL.CTCL / "calibrate.py"), "--eps", "4", "--variant", "plain",
                        "--config", "code", "--out", str(calib), "--no_cache"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1500:]
    env = dict(os.environ, TRAJPE_DATA=str(root / "data"), CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="8",
               MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8")
    base = [C.DPLORA_PY, "-I", "-B", str(TL.CTCL / "train_ctcl.py"), "--eps", "4", "--variant", "plain",
            "--calib", str(calib), "--model", str(TINY), "--device", "cpu"]
    r = subprocess.run(base + ["--dataset", "bandung", "--out_dir", str(root / "x"), "--log_private"], env=env,
                       capture_output=True, text=True)
    assert r.returncode != 0 and "jakarta" in (r.stdout + r.stderr)
    for d in ("gw_austin", "gw_nyc"):
        r = subprocess.run(base + ["--dataset", d, "--out_dir", str(root / "x")], env=env, capture_output=True,
                           text=True)
        assert r.returncode != 0 and "never" in (r.stdout + r.stderr), d
    wrong = work / "calib_wrong.json"
    cw = json.loads(calib.read_text())
    cw["steps"] = 1000
    wrong.write_text(json.dumps(cw))
    r = subprocess.run(base[:-6] + ["--calib", str(wrong), "--model", str(TINY), "--device", "cpu", "--dataset",
                                    "bandung", "--out_dir", str(root / "x")], env=env, capture_output=True, text=True)
    assert r.returncode != 0 and "does not match" in (r.stdout + r.stderr)
    t0 = time.time()
    r = subprocess.run(base + ["--dataset", "bandung", "--out_dir", str(work)], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    tj = json.loads((work / "train.json").read_text())
    assert tj["complete"] is True and tj["steps_run"] == tj["steps"] == 1166 and tj["noise_multiplier"] == cw["sigma_sgd"]
    assert tj["vp_calibration"] == {"sigma_hist": 10.0, "n_hist": 1, "note": tj["vp_calibration"]["note"]}
    assert tj["sigma_eq"] == 10.0 and tj["sigma_prior"] is None and tj["n_prior"] == 0
    assert tj["private_stats"] == {"oom_retries": 0} and "log" not in tj
    forbidden = ("loss", "gnorm", "grad_norm", "clipped", "batch_size", "realised", "empty", "truncat", "response_tok")
    flat = json.dumps(tj).lower()
    keys = set()

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                keys.add(k.lower())
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(tj)
    bad = [k for k in keys for f in forbidden if f in k]
    assert not bad, bad
    out = (r.stdout + r.stderr).lower()
    assert not any(f in out for f in ("loss", "gnorm", "clipped", "\"batch\"")), out[-1500:]
    files = tj["weights"]["files"]
    assert files and all(C.sha256_file(work / "model" / n) == h for n, h in files.items())
    assert "train.json" in os.listdir(work) and not (work / "train.json.tmp").exists()
    r = subprocess.run(base + ["--dataset", "bandung", "--out_dir", str(work)], env=env, capture_output=True, text=True)
    assert r.returncode != 0 and "exists" in (r.stdout + r.stderr)
    print(f"full synthetic run: T=1166 complete in {time.time() - t0:.0f}s; train.json has no private statistic; "
          f"{flat.count('sha256')} hashes recorded; forbidden datasets and a mismatched calibration refused")


def test_real_param_count():
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(C.MODEL, revision=C.MODEL_COMMIT, dtype=torch.float32)
    _, _, params = TR.model_parts(m, "full")
    n = sum(p.numel() for p in params)
    assert n == C.N_PARAMS, n
    assert m.lm_head.weight is m.model.embed_tokens.weight
    print(f"Qwen2.5-0.5B-Instruct @ {C.MODEL_COMMIT[:7]}: {n:,} trainable parameters (tied head counted once)")


if __name__ == "__main__":
    t0 = time.time()
    torch.set_num_threads(8)
    test_exact()
    test_clip_batching_neighbours()
    test_noise_and_schedule()
    test_lora()
    test_real_param_count()
    test_cli_and_full_run()
    print(f"test_train: all passed ({time.time() - t0:.0f}s)")
