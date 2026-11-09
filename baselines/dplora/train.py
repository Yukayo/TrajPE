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
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", f"{_TP_WORK}/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE.parents[1]))
from trajpe.hf_revisions import revision as hf_revision

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import accounting
import textfmt

PROJECT = HERE.parents[1]
DATA = PROJECT / "data" / "processed"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
N_REF = 74_685
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
CE_CHUNK = 2048


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_data(city: str, tok, limit: int = 0, decimals: int = 3):
    df = pd.read_parquet(DATA / city / "train.parquet", columns=["trail_id", "category", "lat", "lon", "t_local", "pos"])
    texts = textfmt.serialize_frame(df, decimals)
    if limit:
        texts = texts[:limit]
    prompt = tok.apply_chat_template(textfmt.messages(city), add_generation_prompt=True, tokenize=False)
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    end = tok.convert_tokens_to_ids("<|im_end|>")
    resp = [r + [end] for r in tok(texts, add_special_tokens=False)["input_ids"]]
    return prompt_ids, resp


def pack(idx: np.ndarray, rlen: np.ndarray, plen: int, tok_budget: int, tgt_budget: int) -> list:
    order = idx[np.argsort(-rlen[idx], kind="stable")]
    mbs, cur, cur_max, cur_tgt = [], [], 0, 0
    for i in order:
        L = plen + int(rlen[i])
        if cur and (max(cur_max, L) * (len(cur) + 1) > tok_budget or cur_tgt + rlen[i] > tgt_budget):
            mbs.append(np.array(cur))
            cur, cur_max, cur_tgt = [], 0, 0
        cur.append(i)
        cur_max = max(cur_max, L)
        cur_tgt += int(rlen[i])
    if cur:
        mbs.append(np.array(cur))
    return mbs


class LossFn:

    def __init__(self, peft_model, prompt_ids, resp, pad_id, device):
        inner = peft_model.base_model.model
        self.core, self.head = inner.model, inner.lm_head
        self.prompt = torch.tensor(prompt_ids, dtype=torch.long)
        self.resp = [torch.tensor(r, dtype=torch.long) for r in resp]
        self.pad, self.dev = pad_id, device

    def _ce(self, h, tgt):
        return F.cross_entropy(self.head(h).float(), tgt, reduction="none")

    def __call__(self, idx) -> torch.Tensor:
        P = len(self.prompt)
        rs = [self.resp[i] for i in idx]
        rl = torch.tensor([len(r) for r in rs])
        n, Lmax = len(rs), P + int(rl.max())
        ids = torch.full((n, Lmax), self.pad, dtype=torch.long)
        ids[:, :P] = self.prompt
        for i, r in enumerate(rs):
            ids[i, P:P + len(r)] = r
        ids = ids.to(self.dev)
        hidden = self.core(input_ids=ids, use_cache=False).last_hidden_state
        rows = torch.repeat_interleave(torch.arange(n), rl).to(self.dev)
        cols = (P - 1 + torch.cat([torch.arange(len(r)) for r in rs])).to(self.dev)
        tgt = torch.cat(rs).to(self.dev)
        h = hidden[rows, cols]
        tl = torch.cat([checkpoint(self._ce, h[s:s + CE_CHUNK], tgt[s:s + CE_CHUNK], use_reentrant=False)
                        for s in range(0, len(tgt), CE_CHUNK)])
        per = torch.zeros(n, device=self.dev, dtype=tl.dtype).index_add_(0, rows, tl)
        return per / rl.to(self.dev, tl.dtype)


def lr_lambda(T: int, warmup_frac: float = 0.03):
    w = max(1, int(round(warmup_frac * T)))
    return lambda s: (s + 1) / w if s < w else max(0.0, (T - s) / max(1, T - w))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=list(textfmt.CITY_NAMES))
    ap.add_argument("--eps", type=float, required=True, help="target epsilon (inf = non-private LoRA)")
    ap.add_argument("--delta", type=float, default=accounting.DELTA)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, required=True, help="expected batch size at N_REF (Jakarta)")
    ap.add_argument("--epochs", type=float, required=True, help="expected passes over the data")
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--clip", type=float, default=1.0, help="per-example L2 clipping norm C (DP only)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tok_budget", type=int, default=8192, help="padded tokens per micro-batch")
    ap.add_argument("--tgt_budget", type=int, default=4096, help="target tokens per micro-batch")
    ap.add_argument("--mem_cap_gb", type=float, default=0.0, help="cap on this process's CUDA memory")
    ap.add_argument("--limit", type=int, default=0, help="debug: use only the first N trails")
    ap.add_argument("--max_steps", type=int, default=0, help="debug: stop after this many steps")
    ap.add_argument("--log_private", action="store_true", help="log losses / grad norms (tuning city only)")
    ap.add_argument("--calib", default="", help="dploravp: vp.py calibration JSON (sigma_SGD of the composition)")
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--model", default=MODEL, help="v2: base model (Hub id, cached under HF_HOME)")
    ap.add_argument("--grad_ckpt", action="store_true", help="v2: gradient checkpointing (needed for 7B)")
    ap.add_argument("--coord_decimals", type=int, default=3, choices=[3, 4], help="v2: text-format coordinates")
    ap.add_argument("--inf_steps", choices=["data", "nref"], default="data",
                    help="eps=inf step count: data = epochs*ceil(N/batch) (v1), nref = round(epochs*N_REF/batch)")
    ap.add_argument("--inf_clip", type=float, default=0.0, help="eps=inf: global gradient-norm clipping (0 = off)")
    args = ap.parse_args()
    t_start = time.time()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    private = math.isfinite(args.eps)

    from opacus import GradSampleModule
    from opacus.optimizers import DPOptimizer
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    if dev.type == "cuda" and args.mem_cap_gb > 0:
        total = torch.cuda.get_device_properties(dev).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.mem_cap_gb / total), dev)
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model, revision=hf_revision(args.model))
    prompt_ids, resp = load_data(args.dataset, tok, args.limit, args.coord_decimals)
    N = len(resp)
    rlen = np.array([len(r) for r in resp])
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=hf_revision(args.model), dtype=dtype, attn_implementation="sdpa").to(dev)
    model.config.use_cache = False
    if args.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
        model.enable_input_require_grads()
    lcfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0, target_modules=LORA_TARGETS,
                      bias="none", task_type="CAUSAL_LM")
    pm = get_peft_model(model, lcfg)
    params = [p for p in pm.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    loss_fn = LossFn(pm, prompt_ids, resp, tok.pad_token_id, dev)
    rng = np.random.default_rng(args.seed)
    info = {"method": "dplora", "model": args.model, "dataset": args.dataset, "eps_target": args.eps,
            "delta": args.delta,
            "seed": args.seed, "batch_ref": args.batch, "epochs": args.epochs, "lr": args.lr, "rank": args.rank,
            "lora_alpha": 2 * args.rank, "lora_targets": LORA_TARGETS, "lora_dropout": 0.0,
            "trainable_params": n_train, "optimizer": "AdamW(betas=(0.9,0.999), eps=1e-8, wd=0)",
            "schedule": "linear warmup 3% then linear decay to 0", "prompt": textfmt.messages(args.dataset),
            "device": str(dev), "dtype": str(dtype), "torch": torch.__version__}
    v2 = {k: v for k, v in [("coord_decimals", args.coord_decimals), ("grad_ckpt", args.grad_ckpt),
                            ("inf_steps", args.inf_steps), ("inf_clip", args.inf_clip)]
          if v != ap.get_default(k)}
    if v2:
        info["v2"] = v2
        info["response_tokens"] = {"mean": round(float(rlen.mean()), 1), "max": int(rlen.max())} \
            if args.log_private else None

    if private:
        q = args.batch / N_REF
        T = int(round(args.epochs / q))
        if args.calib:
            cal = json.loads(Path(args.calib).read_text())
            if abs(cal["sample_rate"] - q) > 1e-12 or cal["steps"] != T or abs(cal["eps_total"] - args.eps) > 1e-12:
                raise SystemExit(f"calibration {args.calib} does not match (q={q}, T={T}, eps={args.eps})")
            sigma = float(cal["sigma_sgd"])
            part = accounting.epsilons(sigma, q, T, args.delta)
            acc = accounting.epsilons_composed(sigma, q, T, cal["sigma_hist"], cal["n_hist"], args.delta)
            if acc["eps_prv"] > args.eps:
                raise SystemExit(f"composed PRV epsilon {acc['eps_prv']} exceeds {args.eps}")
            info.update({"vp_calibration": cal, "eps_pld": cal["eps_composed_pld"],
                         "eps_sgd_part_prv": part["eps_prv"], "eps_sgd_part_rdp": part["eps_rdp"],
                         "eps_sgd_part_pld": cal["eps_sgd_only_pld"]})
            accountant = ("PLD (dp_accounting) of DP-SGD composed with the prior's 2 Gaussians (calibration, target "
                          "eps - 0.015); eps_prv / eps_rdp = Opacus cross-checks of the same composition")
        else:
            sigma = accounting.calibrate_sigma(args.eps, q, T, args.delta)
            acc = accounting.epsilons(sigma, q, T, args.delta)
            accountant = "PRV (opacus, upper bound; calibration)"
        info.update({"privacy": "DP-SGD, trail-level, add/remove-one neighbours", "sampling": "Poisson",
                     "sample_rate": q, "steps": T, "noise_multiplier": sigma, "clip": args.clip,
                     "b_nom": args.batch, "accountant": accountant, **acc})
        log(f"DP-SGD: q={q:.6f} T={T} sigma={sigma:.4f} C={args.clip} -> eps_prv={acc['eps_prv']:.4f} "
            f"eps_rdp={acc['eps_rdp']:.4f} (delta={args.delta})")
        gsm = GradSampleModule(pm, batch_first=True, loss_reduction="sum", strict=True)
        gsm.train()
        base_opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
        gen = torch.Generator(device=dev)
        gen.manual_seed(args.seed + 7919)
        opt = DPOptimizer(base_opt, noise_multiplier=sigma, max_grad_norm=args.clip, expected_batch_size=args.batch,
                          loss_reduction="mean", generator=gen)
    else:
        if args.inf_steps == "nref":
            T = int(round(args.epochs * N_REF / args.batch))
        else:
            T = int(args.epochs * math.ceil(N / args.batch))
        info.update({"privacy": "none (eps = inf reference)", "sampling": "shuffled epochs", "steps": T,
                     "batch": args.batch})
        if v2:
            info["delta"] = None
        pm.train()
        base_opt = opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(base_opt, lr_lambda(T))
    steps = min(T, args.max_steps) if args.max_steps else T
    log(f"{args.dataset} eps={args.eps} T={T} (running {steps}) trainable={n_train} device={dev}")

    budgets = [args.tok_budget, args.tgt_budget]

    def one_step(sel):
        mbs = pack(sel, rlen, len(prompt_ids), *budgets)
        losses, norms = [], []
        if private and not mbs:
            for p in params:
                p.grad_sample = torch.zeros((0, *p.shape), device=p.device, dtype=p.dtype)
            opt.step()
            opt.zero_grad(set_to_none=True)
        for j, mb in enumerate(mbs):
            per = loss_fn(mb)
            if private:
                per.sum().backward()
                if args.log_private:
                    gs = opt.grad_samples
                    norms.append(torch.stack([g.reshape(len(g), -1).norm(2, dim=-1) for g in gs], 1).norm(2, 1))
                if j < len(mbs) - 1:
                    opt.signal_skip_step(do_skip=True)
                opt.step()
                opt.zero_grad(set_to_none=True)
            else:
                (per.sum() / len(sel)).backward()
            if args.log_private:
                losses.append(per.detach())
        if not private:
            if args.inf_clip > 0:
                gn = torch.nn.utils.clip_grad_norm_(params, args.inf_clip)
                if args.log_private:
                    norms.append(gn.detach().reshape(1).float())
            opt.step()
            opt.zero_grad(set_to_none=True)
        return losses, norms, len(mbs)

    def reset_after_oom():
        for p in params:
            p.grad = None
            for a in ("grad_sample", "summed_grad", "_current_grad_sample"):
                if hasattr(p, a):
                    setattr(p, a, None)
            if hasattr(p, "_forward_counter"):
                p._forward_counter = 0
        for m in pm.modules():
            if hasattr(m, "activations"):
                m.activations = []
            if hasattr(m, "max_batch_len"):
                del m.max_batch_len
        if private:
            opt._step_skip_queue = []
            opt._is_last_step_skipped = False
        torch.cuda.empty_cache()

    hist, t_loop, n_empty, n_oom = [], time.time(), 0, 0
    perm, pos = None, 0
    win_norms = []
    for step in range(steps):
        if private:
            sel = np.flatnonzero(rng.random(len(resp)) < q)
        else:
            if perm is None or pos >= len(perm):
                perm, pos = rng.permutation(N), 0
            sel, pos = perm[pos:pos + args.batch], pos + args.batch
        n_empty += int(len(sel) == 0)
        while True:
            try:
                losses, norms, n_micro = one_step(sel)
                break
            except torch.OutOfMemoryError:
                reset_after_oom()
                n_oom += 1
                budgets[:] = [budgets[0] // 2, budgets[1] // 2]
                log(f"CUDA OOM at step {step}: retrying the same batch with budgets {budgets}")
                if budgets[0] < 1024:
                    raise
        sched.step()
        if not all(bool(torch.isfinite(p).all()) for p in params):
            raise FloatingPointError(f"non-finite LoRA weights after step {step}")
        if not private and norms:
            win_norms.extend(float(x) for x in torch.cat(norms))
        if args.log_private and (step % args.log_every == 0 or step == steps - 1):
            L = torch.cat(losses).float() if losses else torch.zeros(0)
            row = {"step": step, "lr": sched.get_last_lr()[0], "batch": int(len(sel)), "n_micro": n_micro,
                   "loss": float(L.mean()) if len(L) else float("nan"),
                   "sec": round(time.time() - t_loop, 1)}
            if norms and private:
                nn_ = torch.cat(norms).float()
                row.update({"gnorm_med": float(nn_.median()), "gnorm_p10": float(nn_.quantile(0.1)),
                            "gnorm_p90": float(nn_.quantile(0.9)), "clipped": float((nn_ > args.clip).float().mean())})
            elif win_norms:
                w = np.array(win_norms)
                row.update({"gnorm_med": float(np.median(w)), "gnorm_max": float(w.max()),
                            "clipped": float((w > args.inf_clip).mean())})
                win_norms.clear()
            if dev.type == "cuda":
                row["max_mem_gb"] = round(torch.cuda.max_memory_allocated(dev) / 2**30, 2)
            hist.append(row)
            log(json.dumps(row))
    t_loop = time.time() - t_loop
    pstats = {"empty_batches": n_empty, "oom_retries": n_oom, "micro_budgets": budgets}
    if dev.type == "cuda":
        torch.cuda.synchronize()
        pstats["max_mem_gb"] = round(torch.cuda.max_memory_allocated(dev) / 2**30, 2)
        info["gpu"] = torch.cuda.get_device_name(dev)
    pm.save_pretrained(out / "adapter")
    info.update({"steps_run": steps, "train_loop_sec": round(t_loop, 1), "private_stats": pstats,
                 "train_total_sec": round(time.time() - t_start, 1), "complete": steps == T})
    if args.log_private:
        info["log"] = hist
    (out / "train.json").write_text(json.dumps(info, indent=1, default=str))
    log(f"done: loop {t_loop:.0f}s, total {time.time() - t_start:.0f}s -> {out}")


if __name__ == "__main__":
    main()
