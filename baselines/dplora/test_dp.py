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

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", f"{_TP_WORK}/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

import accounting
import train

DEV = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")


def main():
    from opacus import GradSampleModule
    from opacus.optimizers import DPOptimizer
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(train.MODEL, revision=train.hf_revision(train.MODEL))
    prompt_ids, resp = train.load_data("jakarta", tok, limit=400)
    rlen = np.array([len(r) for r in resp])
    idx = np.concatenate([np.argsort(-rlen)[:2], np.argsort(rlen)[:3], [7]])
    model = AutoModelForCausalLM.from_pretrained(train.MODEL, revision=train.hf_revision(train.MODEL), dtype=torch.float32, attn_implementation="sdpa").to(DEV)
    pm = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=train.LORA_TARGETS,
                                          bias="none", task_type="CAUSAL_LM"))
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.02)
    params = [p for p in pm.parameters() if p.requires_grad]
    fn = train.LossFn(pm, prompt_ids, resp, tok.pad_token_id, DEV)
    gsm = GradSampleModule(pm, batch_first=True, loss_reduction="sum", strict=True)
    gsm.train()

    fn(idx).sum().backward()
    gs_batched = [p.grad_sample.detach().clone() for p in params]
    for p in params:
        p.grad_sample, p.grad = None, None
    alone = []
    for i in idx:
        fn(np.array([i])).sum().backward()
        alone.append([p.grad.detach().clone() for p in params])
        for p in params:
            p.grad_sample, p.grad = None, None
    err = max(float((gs_batched[k][j] - alone[j][k]).abs().max() / (alone[j][k].abs().max() + 1e-12))
              for j in range(len(idx)) for k in range(len(params)))
    print(f"(a) max relative diff batched vs alone per-example grads: {err:.2e}")
    assert err < 1e-3, err

    norms = torch.stack([torch.sqrt(sum((g ** 2).sum() for g in a)) for a in alone])
    C, B = float(norms.median()), 10
    expect = [sum(min(1.0, C / (float(norms[j]) + 1e-6)) * alone[j][k] for j in range(len(idx))) / B
              for k in range(len(params))]
    opt = DPOptimizer(torch.optim.SGD(params, lr=0.0), noise_multiplier=0.0, max_grad_norm=C,
                      expected_batch_size=B, loss_reduction="mean")
    got = {}
    opt.attach_step_hook(lambda o: got.update(g=[p.grad.detach().clone() for p in o.params]))
    for j, mb in enumerate([idx[:2], idx[2:]]):
        fn(mb).sum().backward()
        if j == 0:
            opt.signal_skip_step(do_skip=True)
        opt.step()
        opt.zero_grad(set_to_none=True)
    err_b = max(float((g - e).abs().max() / (e.abs().max() + 1e-12)) for g, e in zip(got["g"], expect))
    print(f"(b) clipped-sum update vs hand computation: max relative diff {err_b:.2e} "
          f"(C={C:.3f}, {int((norms > C).sum())}/{len(idx)} clipped)")
    assert err_b < 1e-3, err_b

    sigma, C, B = 1.3, 0.7, 4.0
    opt = DPOptimizer(torch.optim.SGD(params, lr=0.0), noise_multiplier=sigma, max_grad_norm=C,
                      expected_batch_size=B, loss_reduction="mean", generator=torch.Generator(device=DEV).manual_seed(1))
    opt.attach_step_hook(lambda o: got.update(g=[p.grad.detach().clone() for p in o.params]))
    for p in params:
        p.grad_sample = torch.zeros((0, *p.shape), dtype=p.dtype, device=p.device)
    opt.step()
    opt.zero_grad(set_to_none=True)
    z = torch.cat([g.flatten() for g in got["g"]])
    print(f"(c) noise std {float(z.std()):.5f} (expected {sigma * C / B:.5f}), mean {float(z.mean()):+.1e}, "
          f"n={z.numel()}")
    assert abs(float(z.std()) / (sigma * C / B) - 1) < 0.01

    q, T = 2048 / train.N_REF, 365
    s = accounting.calibrate_sigma(4.0, q, T)
    e = accounting.epsilons(s, q, T)
    print(f"(d) eps=4, q={q:.5f}, T={T}: sigma={s:.4f} -> eps_prv={e['eps_prv']:.4f} eps_rdp={e['eps_rdp']:.4f}")
    assert 4.0 - accounting.EPS_TOL <= e["eps_prv"] <= 4.0 and e["eps_rdp"] >= e["eps_prv"]

    model2 = AutoModelForCausalLM.from_pretrained(train.MODEL, revision=train.hf_revision(train.MODEL), dtype=torch.float32, attn_implementation="sdpa").to(DEV)
    model2.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    model2.enable_input_require_grads()
    pm2 = get_peft_model(model2, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=train.LORA_TARGETS,
                                            bias="none", task_type="CAUSAL_LM"))
    with torch.no_grad():
        for (n1, p1), (n2, p2) in zip(pm.named_parameters(), pm2.named_parameters()):
            assert n1 == n2
            if p1.requires_grad:
                p2.copy_(p1)
    params2 = [p for p in pm2.parameters() if p.requires_grad]
    fn2 = train.LossFn(pm2, prompt_ids, resp, tok.pad_token_id, DEV)
    gsm2 = GradSampleModule(pm2, batch_first=True, loss_reduction="sum", strict=True)
    gsm2.train()
    err_e = 0.0
    for rep in range(2):
        fn2(idx).sum().backward()
        err_e = max(err_e, max(float((p.grad_sample - g).abs().max() / (g.abs().max() + 1e-12))
                               for p, g in zip(params2, gs_batched)))
        for p in params2:
            p.grad_sample, p.grad = None, None
    left = sum(len(getattr(m, "activations", []) or []) for m in pm2.modules())
    print(f"(e) grad checkpointing: max relative diff of per-example grads vs the plain batched ones of (a) "
          f"{err_e:.2e} (two passes), activations left {left}")
    assert err_e < 1e-4 and left == 0, (err_e, left)
    del gsm2, pm2, model2

    import pandas as pd
    import textfmt
    df = pd.read_parquet(train.DATA / "jakarta" / "train.parquet",
                         columns=["trail_id", "category", "lat", "lon", "t_local", "pos"]).head(3000)
    cats = set(pd.read_parquet(train.DATA / "jakarta" / "pois.parquet", columns=["category"])["category"].astype(str))
    t3, t4 = textfmt.serialize_frame(df), textfmt.serialize_frame(df, 4)
    assert all(ln.rsplit(" | ", 1)[1].split(",")[0].split(".")[1].__len__() == 3 for t in t3 for ln in t.split("\n")[1:])
    for t in t4:
        res, why = textfmt.parse(t, cats)
        assert why == "ok" and textfmt.clean(t, cats)[0] == t
    bad = t4[0] + "\n12:00 | Not A Category | -6.2000,106.8000"
    assert textfmt.parse(bad, cats)[1] == "bad_category" and textfmt.clean(bad, cats)[0] == t4[0]
    try:
        import xgrammar as xgr
        from transformers import AutoTokenizer as AT
        tk = AT.from_pretrained(train.MODEL, revision=train.hf_revision(train.MODEL))
        eos = tk.convert_tokens_to_ids("<|im_end|>")
        info = xgr.TokenizerInfo.from_huggingface(tk, stop_token_ids=[eos])
        comp = xgr.GrammarCompiler(info).compile_grammar(textfmt.ebnf(cats, 4))
        for t in t4[:200]:
            m = xgr.GrammarMatcher(comp)
            ok = all(m.accept_token(i) for i in tk(t, add_special_tokens=False)["input_ids"])
            assert ok and m.accept_token(eos) and m.is_terminated(), t
        m = xgr.GrammarMatcher(comp)
        assert not all(m.accept_token(i) for i in tk("Day: Monday\n12:00 | Not A Category", add_special_tokens=False)["input_ids"])
        print("(f) text format: v1 default, 4-decimal parse/clean round trip, grammar accepts 200 real trails and "
              "rejects an out-of-list category")
    except ImportError:
        print("(f) text format: v1 default, 4-decimal parse/clean round trip (xgrammar not importable here)")
    print("all DP checks passed")


if __name__ == "__main__":
    main()
