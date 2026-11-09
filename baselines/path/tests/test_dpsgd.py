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
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HOME", f"{_TP_WORK}/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
import torch

import pathfmt as PF
import train_dp as TD

CITY = "jakarta"
MK = PF.MODELS["1.5b"]
_TOK = {}


def tokenizer():
    if "t" not in _TOK:
        from transformers import AutoTokenizer
        _TOK["t"] = AutoTokenizer.from_pretrained(MK["id"], revision=MK["revision"])
    return _TOK["t"]


def synth_texts(n: int, seed: int = 0, lmax: int = 8) -> list[str]:
    pois = pd.read_parquet(PF.DATA / CITY / "pois.parquet", columns=["category", "lat", "lon"])
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        L = int(rng.integers(2, lmax + 1))
        idx = rng.integers(0, len(pois), L)
        mins = np.sort(rng.integers(0, 1440, L))
        out.append(PF.textfmt.render(int(rng.integers(0, 7)), [(int(m) // 60, int(m) % 60) for m in mins],
                                     pois["category"].iloc[idx].astype(str).tolist(), pois["lat"].iloc[idx].to_numpy(),
                                     pois["lon"].iloc[idx].to_numpy(), PF.DECIMALS))
    return out


def test_windows_tokens():
    tok = tokenizer()
    texts = synth_texts(300, seed=1, lmax=20)
    seqs, starts, info = TD.encode(texts, CITY, tok)
    H = PF.header(CITY)
    assert info["straddling_boundaries"] == 0 and info["eos_id"] == tok.eos_token_id
    nh = info["header_tokens"]
    for t, s, st in zip(texts, seqs, starts):
        lines = t.split("\n")
        assert len(st) == len(lines) - 1 and st[0] == nh and s[-1] == tok.eos_token_id
        assert list(st) == sorted(st) and st[-1] < len(s) - 1
        for k, b in enumerate(st):
            ctx = H + ("\n".join(lines[:k + 1]) + "\n" if k else "")
            tgt = "\n".join(lines[k + 1:]) if k else t
            assert tok.decode(s[:b]) == ctx, (k, tok.decode(s[:b]), ctx)
            assert tok.decode(s[b:-1]) == tgt and int(s[-1]) == tok.eos_token_id


def tiny_peft(grad_ckpt: bool = False, seed: int = 0):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, Qwen2Config
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=len(tokenizer()), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
                      tie_word_embeddings=True)
    model = AutoModelForCausalLM.from_config(cfg, attn_implementation="sdpa").float()
    model.config.use_cache = False
    if grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
        model.enable_input_require_grads()
    pm = get_peft_model(model, LoraConfig(r=8, lora_alpha=8, lora_dropout=0.0, target_modules=PF.TARGETS,
                                          bias="none", task_type="CAUSAL_LM"))
    torch.manual_seed(seed + 1)
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.05)
    return pm


def examples(n_tr: int = 6, seed: int = 2):
    tok = tokenizer()
    seqs, starts, _ = TD.encode(synth_texts(n_tr, seed=seed, lmax=7), CITY, tok)
    rng = np.random.default_rng(seed)
    ks = [int(rng.integers(0, len(s))) for s in starts]
    return [seqs[i] for i in range(n_tr)], [int(starts[i][ks[i]]) for i in range(n_tr)]


def per_example(pm, ex, batches):
    from opacus import GradSampleModule
    params = [p for p in pm.parameters() if p.requires_grad]
    fn = TD.WindowLoss(pm, tokenizer().pad_token_id, torch.device("cpu"))
    gsm = GradSampleModule(pm, batch_first=True, loss_reduction="sum", strict=True)
    gsm.train()
    out = {}
    for mb in batches:
        fn([ex[0][i] for i in mb], [ex[1][i] for i in mb]).sum().backward()
        for j, i in enumerate(mb):
            out[i] = [p.grad_sample[j].detach().clone() for p in params]
        for p in params:
            p.grad_sample, p.grad = None, None
    left = sum(len(getattr(m, "activations", []) or []) for m in pm.modules())
    gsm._close()
    return out, params, left


def alone(pm, ex):
    params = [p for p in pm.parameters() if p.requires_grad]
    fn = TD.WindowLoss(pm, tokenizer().pad_token_id, torch.device("cpu"))
    pm.train()
    res = []
    for i in range(len(ex[0])):
        fn([ex[0][i]], [ex[1][i]]).sum().backward()
        res.append([p.grad.detach().clone() for p in params])
        for p in params:
            p.grad = None
    return res


def rel(a, b):
    return float((a - b).abs().max() / (b.abs().max() + 1e-12))


def test_per_example_grads_batched_vs_alone_and_ckpt():
    ex = examples()
    ref = alone(tiny_peft(), ex)
    lens = np.array([len(s) for s in ex[0]])
    mbs = TD.pack(lens, 4, 10_000)
    assert sum(len(m) for m in mbs) == len(lens) and max(len(m) for m in mbs) <= 4
    for ckpt in (False, True):
        got, params, left = per_example(tiny_peft(grad_ckpt=ckpt), ex, mbs)
        err = max(rel(got[i][k], ref[i][k]) for i in range(len(lens)) for k in range(len(params)))
        nrm = [float(torch.sqrt(sum((g ** 2).sum() for g in got[i]))) for i in range(len(lens))]
        nrm_ref = [float(torch.sqrt(sum((g ** 2).sum() for g in ref[i]))) for i in range(len(lens))]
        print(f"   grad_ckpt={ckpt}: max rel diff {err:.2e}; per-sample norms {np.round(nrm, 5).tolist()}", flush=True)
        assert err < 1e-4 and np.allclose(nrm, nrm_ref, rtol=1e-5) and left == 0


def test_clipped_sum_sensitivity_and_noise():
    from opacus.optimizers import DPOptimizer
    ex = examples(5, seed=3)
    ref = alone(tiny_peft(), ex)
    norms = np.array([float(torch.sqrt(sum((g ** 2).sum() for g in r))) for r in ref])
    C, B = float(np.median(norms)), 7.0
    pm = tiny_peft()
    from opacus import GradSampleModule
    params = [p for p in pm.parameters() if p.requires_grad]
    fn = TD.WindowLoss(pm, tokenizer().pad_token_id, torch.device("cpu"))
    gsm = GradSampleModule(pm, batch_first=True, loss_reduction="sum", strict=True)
    gsm.train()
    opt = DPOptimizer(torch.optim.SGD(params, lr=0.0), noise_multiplier=0.0, max_grad_norm=C, expected_batch_size=B,
                      loss_reduction="mean")
    got = {}
    opt.attach_step_hook(lambda o: got.update(g=[p.grad.detach().clone() for p in o.params]))

    def clipped_sum(idx):
        mbs = [idx[:2], idx[2:]] if len(idx) > 2 else [idx]
        for j, mb in enumerate(mbs):
            fn([ex[0][i] for i in mb], [ex[1][i] for i in mb]).sum().backward()
            if j < len(mbs) - 1:
                opt.signal_skip_step(do_skip=True)
            opt.step()
            opt.zero_grad(set_to_none=True)
        return [g * B for g in got["g"]]

    full = clipped_sum([0, 1, 2, 3, 4])
    hand = [sum(min(1.0, C / (norms[i] + 1e-6)) * ref[i][k] for i in range(5)) for k in range(len(params))]
    assert max(rel(a, b) for a, b in zip(full, hand)) < 1e-4
    for drop in range(5):
        rest = clipped_sum([i for i in range(5) if i != drop])
        diff = float(torch.sqrt(sum(((a - b) ** 2).sum() for a, b in zip(full, rest))))
        clip_norm = min(norms[drop], C)
        assert diff <= C * (1 + 1e-4) and abs(diff - clip_norm) < 1e-4 * C, (drop, diff, C)
    gsm._close()
    pm2 = tiny_peft()
    params2 = [p for p in pm2.parameters() if p.requires_grad]
    sigma, C2, B2 = 0.6, 0.01, 256.0
    opt2 = DPOptimizer(torch.optim.SGD(params2, lr=0.0), noise_multiplier=sigma, max_grad_norm=C2,
                       expected_batch_size=B2, loss_reduction="mean", generator=torch.Generator().manual_seed(7919))
    opt2.attach_step_hook(lambda o: got.update(g=[p.grad.detach().clone() for p in o.params]))
    for p in params2:
        p.grad_sample = torch.zeros((0, *p.shape), dtype=p.dtype)
    opt2.step()
    z = torch.cat([g.flatten() for g in got["g"]])
    assert abs(float(z.std()) / (sigma * C2 / B2) - 1) < 0.02, float(z.std())


def test_pack_schedule_and_streams():
    lens = np.array([400, 30, 31, 200, 50, 50, 50, 600, 45] * 7)
    for mm, tb in ((16, 8192), (8, 4096), (1, 100)):
        mbs = TD.pack(lens, mm, tb)
        flat = np.sort(np.concatenate(mbs))
        assert np.array_equal(flat, np.arange(len(lens)))
        for m in mbs:
            assert len(m) <= mm and (len(m) == 1 or lens[m].max() * len(m) <= tb)
    T = PF.n_steps()
    f = [PF.lr_factor(s, T) for s in range(T)]
    w = int(round(0.03 * T))
    assert w == 26 and abs(f[0] - 1 / 26) < 1e-12 and f[25] == 1.0 and f[26] == 1.0
    assert all(a >= b for a, b in zip(f[26:], f[27:])) and f[-1] < 1e-4
    N, q = 5000, PF.sample_rate()
    L = np.random.default_rng(9).integers(2, 21, N)

    def draws(with_windows):
        rp, rw = np.random.default_rng(0), np.random.default_rng([0, PF.STREAM_WINDOWS])
        out = []
        for _ in range(20):
            sel = np.flatnonzero(rp.random(N) < q)
            out.append((sel, rw.integers(0, L[sel]) if with_windows and len(sel) else None))
        return out
    a, b, c = draws(True), draws(True), draws(False)
    assert all(np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1]) for x, y in zip(a, b))
    assert all(np.array_equal(x[0], y[0]) for x, y in zip(a, c))
    ks = np.concatenate([k for _, k in a])
    sl = np.concatenate([s for s, _ in a])
    assert (ks >= 0).all() and (ks < L[sl]).all()


def test_real_model_production_config():
    from opacus import GradSampleModule
    tok = tokenizer()
    pm = TD.build_model(MK["id"], MK["revision"], torch.device("cpu"), torch.float32, True, 0)
    params = [p for p in pm.parameters() if p.requires_grad]
    assert sum(p.numel() for p in params) == MK["trainable_params"] == 147_718_144
    assert all(p.dtype == torch.float32 for p in params)
    lora_layers = [m for m in pm.modules() if hasattr(m, "scaling") and isinstance(m.scaling, dict)]
    assert lora_layers and all(abs(m.scaling["default"] - 1.0) < 1e-12 for m in lora_layers)
    assert {n.split(".")[-1] for n, m in pm.named_modules() if hasattr(m, "lora_A")} == set(PF.TARGETS)
    with torch.no_grad():
        g = torch.Generator().manual_seed(5)
        for n, p in pm.named_parameters():
            if "lora_B" in n:
                p.copy_(torch.randn(p.shape, generator=g) * 1e-3)
    seqs, starts, _ = TD.encode(synth_texts(2, seed=4, lmax=3), CITY, tok)
    ex = ([seqs[0], seqs[1]], [int(starts[0][1]), int(starts[1][0])])
    fn = TD.WindowLoss(pm, tok.pad_token_id, torch.device("cpu"))
    ref = []
    pm.train()
    for i in range(2):
        fn([ex[0][i]], [ex[1][i]]).sum().backward()
        ref.append([p.grad.detach().clone() for p in params])
        for p in params:
            p.grad = None
    gsm = GradSampleModule(pm, batch_first=True, loss_reduction="sum", strict=True)
    gsm.train()
    fn(ex[0], ex[1]).sum().backward()
    err = max(rel(p.grad_sample[i], ref[i][k]) for i in range(2) for k, p in enumerate(params))
    nb = [float(torch.sqrt(sum((p.grad_sample[i] ** 2).sum() for p in params))) for i in range(2)]
    na = [float(torch.sqrt(sum((g ** 2).sum() for g in ref[i]))) for i in range(2)]
    print(f"   Qwen2.5-1.5B r=128: max rel diff {err:.2e}; per-sample norms batched {nb} alone {na}", flush=True)
    assert err < 1e-3 and np.allclose(nb, na, rtol=1e-4)
    gsm._close()


def test_train_main_debug_synthetic():
    import json
    import tempfile
    cal = {"method": "path", "eps_total": 4.0, "delta": 1e-6, "sample_rate": PF.sample_rate(), "steps": PF.n_steps(),
           "sigma_sgd": 0.6059, "sigma_sel": 4.3652, "sigma_hist": 4.3652, "n_hist": 1, "eps_composed_pld": 3.9845,
           "eps_sgd_only_pld": 3.88}
    old = PF.DATA
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        try:
            pois = pd.read_parquet(PF.DATA / CITY / "pois.parquet", columns=["venue_id", "category", "lat", "lon"])
            rng = np.random.default_rng(11)
            rows = []
            for i in range(2500):
                L = int(rng.integers(2, 5))
                v = rng.integers(0, len(pois), L)
                t0 = pd.Timestamp("2013-01-07") + pd.Timedelta(minutes=int(rng.integers(0, 10080)))
                for p in range(L):
                    rows.append({"trail_id": f"t{i:05d}", "user_id": f"u{i}", "venue_id": str(pois["venue_id"].iloc[v[p]]),
                                 "lat": float(pois["lat"].iloc[v[p]]), "lon": float(pois["lon"].iloc[v[p]]),
                                 "category": str(pois["category"].iloc[v[p]]),
                                 "t_local": t0 + pd.Timedelta(minutes=37 * p), "pos": p})
            (td / "data" / CITY).mkdir(parents=True)
            pd.DataFrame(rows).to_parquet(td / "data" / CITY / "train.parquet", index=False)
            PF.DATA = td / "data"
            (td / "calib.json").write_text(json.dumps(cal))
            argv = sys.argv
            sys.argv = ["train_dp.py", "--dataset", CITY, "--eps", "4", "--calib", str(td / "calib.json"),
                        "--out_dir", str(td / "w"), "--max_steps", "2", "--log_private", "--log_every", "1"]
            try:
                TD.main()
            finally:
                sys.argv = argv
            info = json.loads((td / "w" / "train.json").read_text())
            assert (td / "w" / "adapter" / "adapter_model.safetensors").exists()
            acfg = json.loads((td / "w" / "adapter" / "adapter_config.json").read_text())
        finally:
            PF.DATA = old
    assert info["complete"] is False and info["steps_run"] == 2 and info["steps"] == 875
    assert info["eps_prv"] <= 4.0 and info["noise_multiplier"] == 0.6059 and info["trainable_params"] == 147_718_144
    assert info["private_stats"]["n_trails"] == 2500 and info["private_stats"]["oom_retries"] == 0
    assert len(info["log"]) == 2 and info["log"][0]["batch"] > 0
    assert acfg["r"] == 128 and acfg["lora_alpha"] == 128 and acfg["base_model_name_or_path"] == MK["id"]
    assert sorted(acfg["target_modules"]) == sorted(PF.TARGETS)


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    for name, f in list(globals().items()):
        if name.startswith("test_") and callable(f):
            f()
            print("ok", name, flush=True)
    print("all DP-SGD tests passed")
