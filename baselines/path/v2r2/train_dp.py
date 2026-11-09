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

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import pathfmt as PF

accounting = PF.load_module("dplora_accounting", PF.PROJECT / "baselines" / "dplora" / "accounting.py")
CE_CHUNK = 2048


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def encode(texts: list[str], city: str, tok) -> tuple[list[np.ndarray], list[np.ndarray], dict]:
    H = PF.header(city, tok)
    hid = tok(H, add_special_tokens=False)["input_ids"]
    eos = PF.eot_id(tok)
    seqs, starts, n_str, n_pref, n_dplora = [], [], 0, 0, 0
    for s in range(0, len(texts), 4096):
        chunk = texts[s:s + 4096]
        enc = tok([H + t for t in chunk], add_special_tokens=False, return_offsets_mapping=True)
        sep = tok(chunk, add_special_tokens=False)["input_ids"]
        for t, ids, off, own in zip(chunk, enc["input_ids"], enc["offset_mapping"], sep):
            if ids[:len(hid)] != hid:
                n_pref += 1
            if ids != hid + own:
                n_dplora += 1
            st, k = PF.target_starts(off, [len(H) + b for b in PF.window_char_bounds(t)])
            n_str += k
            seqs.append(np.asarray(ids + [eos], dtype=np.int64))
            starts.append(np.asarray(st, dtype=np.int64))
    if n_pref or n_dplora:
        raise SystemExit(f"{n_pref} sequences do not start with the header's tokens, {n_dplora} differ from "
                         "DP-LoRA v2's tokenization")
    return seqs, starts, {"header_tokens": len(hid), "straddling_boundaries": n_str, "eos_id": int(eos),
                          "eos": PF.EOT}


def load_texts(city: str, limit: int = 0) -> list[str]:
    df = pd.read_parquet(PF.DATA / city / "train.parquet", columns=["trail_id", "category", "lat", "lon", "t_local", "pos"])
    texts = PF.textfmt.serialize_frame(df, PF.DECIMALS)
    return texts[:limit] if limit else texts


def pack(lengths: np.ndarray, micro_max: int, tok_budget: int) -> list[np.ndarray]:
    order = np.argsort(-lengths, kind="stable")
    mbs, cur, cur_max = [], [], 0
    for i in order:
        L = int(lengths[i])
        if cur and (len(cur) + 1 > micro_max or max(cur_max, L) * (len(cur) + 1) > tok_budget):
            mbs.append(np.array(cur))
            cur, cur_max = [], 0
        cur.append(i)
        cur_max = max(cur_max, L)
    if cur:
        mbs.append(np.array(cur))
    return mbs


class WindowLoss:

    def __init__(self, peft_model, pad_id: int, device):
        inner = peft_model.base_model.model
        self.core, self.head = inner.model, inner.lm_head
        self.pad, self.dev = pad_id, device

    def _ce(self, h, tgt):
        return F.cross_entropy(self.head(h).float(), tgt, reduction="none")

    def __call__(self, seqs: list[np.ndarray], starts: list[int]) -> torch.Tensor:
        n = len(seqs)
        lens = [len(s) for s in seqs]
        ids = torch.full((n, max(lens)), self.pad, dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :lens[i]] = torch.from_numpy(s)
        ids = ids.to(self.dev)
        hidden = self.core(input_ids=ids, use_cache=False).last_hidden_state
        ntgt = torch.tensor([lens[i] - starts[i] for i in range(n)])
        if bool((ntgt <= 0).any()) or min(starts) < 1:
            raise ValueError("every example needs >= 1 target token after a non-empty context")
        rows = torch.repeat_interleave(torch.arange(n), ntgt).to(self.dev)
        cols = torch.cat([torch.arange(starts[i] - 1, lens[i] - 1) for i in range(n)]).to(self.dev)
        tgt = torch.cat([torch.from_numpy(seqs[i][starts[i]:]) for i in range(n)]).to(self.dev)
        h = hidden[rows, cols]
        tl = torch.cat([checkpoint(self._ce, h[s:s + CE_CHUNK], tgt[s:s + CE_CHUNK], use_reentrant=False)
                        for s in range(0, len(tgt), CE_CHUNK)])
        per = torch.zeros(n, device=self.dev, dtype=tl.dtype).index_add_(0, rows, tl)
        return per / ntgt.to(self.dev, tl.dtype)


def build_model(model_id: str, revision: str, device, dtype, grad_ckpt: bool, seed: int):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=dtype,
                                                 attn_implementation="sdpa").to(device)
    model.config.use_cache = False
    if grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
        model.enable_input_require_grads()
    lcfg = LoraConfig(r=PF.RANK, lora_alpha=PF.ALPHA, lora_dropout=PF.DROPOUT, target_modules=PF.TARGETS,
                      bias="none", task_type="CAUSAL_LM")
    return get_peft_model(model, lcfg)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--eps", type=float, required=True)
    ap.add_argument("--calib", required=True, help="calibrate.py output (sigma_SGD of the joint composition)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_key", default=PF.DEFAULT_MODEL, choices=list(PF.MODELS))
    ap.add_argument("--seed", type=int, default=PF.SEED)
    ap.add_argument("--mem_cap_gb", type=float, default=0.0, help="cap on this process's CUDA memory")
    ap.add_argument("--no_grad_ckpt", action="store_true", help="debug only")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N trails only (refused for releases)")
    ap.add_argument("--max_steps", type=int, default=0, help="debug: stop early (train.json complete=false)")
    ap.add_argument("--log_private", action="store_true", help="losses / grad norms / batches (tuning city only)")
    ap.add_argument("--log_every", type=int, default=25)
    args = ap.parse_args()
    PF.check_dataset(args.dataset)
    if args.log_private and args.dataset != PF.SMOKE:
        raise SystemExit("--log_private is allowed on the tuning city only")
    t_start = time.time()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mk = PF.MODELS[args.model_key]

    cal = json.loads(Path(args.calib).read_text())
    q, T = PF.sample_rate(), PF.n_steps()
    if abs(cal["sample_rate"] - q) > 1e-15 or cal["steps"] != T or abs(cal["eps_total"] - args.eps) > 1e-12 \
            or abs(cal["delta"] - PF.DELTA) > 0 or cal.get("method") != "path":
        raise SystemExit(f"calibration {args.calib} does not match (q={q}, T={T}, eps={args.eps})")
    sigma, sigma_sel = float(cal["sigma_sgd"]), float(cal["sigma_sel"])
    if float(cal["sigma_hist"]) != sigma_sel or int(cal["n_hist"]) != 1:
        raise SystemExit("calibration: the selection must enter as ONE Gaussian of noise sigma_sel")
    acc = accounting.epsilons_composed(sigma, q, T, sigma_sel, 1, PF.DELTA)
    part = accounting.epsilons(sigma, q, T, PF.DELTA)
    if not acc["eps_prv"] <= args.eps:
        raise SystemExit(f"Opacus PRV epsilon of the composition {acc['eps_prv']} exceeds {args.eps}: refusing to train")
    log(f"DP-SGD: q={q:.6f} T={T} sigma={sigma:.4f} C={PF.CLIP}; joint with Gaussian(sigma_sel={sigma_sel:.4f}): "
        f"PLD {cal['eps_composed_pld']:.4f}, PRV {acc['eps_prv']:.4f}, RDP {acc['eps_rdp']:.4f} (target {args.eps})")

    from opacus import GradSampleModule
    from opacus.optimizers import DPOptimizer
    from transformers import AutoTokenizer

    dev = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    if dev.type == "cuda" and args.mem_cap_gb > 0:
        total = torch.cuda.get_device_properties(dev).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.mem_cap_gb / total), dev)
    tok = AutoTokenizer.from_pretrained(mk["id"], revision=mk["revision"])
    t_tok = time.time()
    seqs, starts, enc_info = encode(load_texts(args.dataset, args.limit), args.dataset, tok)
    t_tok = time.time() - t_tok
    N = len(seqs)
    slen = np.array([len(s) for s in seqs])
    nrows = np.array([len(s) for s in starts])
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    pm = build_model(mk["id"], mk["revision"], dev, dtype, not args.no_grad_ckpt, args.seed)
    params = [p for p in pm.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    if n_train != mk["trainable_params"]:
        raise SystemExit(f"{n_train} trainable parameters, expected {mk['trainable_params']}")
    if any(p.dtype != torch.float32 for p in params):
        raise SystemExit("LoRA weights must be fp32")
    loss_fn = WindowLoss(pm, tok.pad_token_id, dev)
    rng_p = np.random.default_rng(args.seed)
    rng_w = np.random.default_rng([args.seed, PF.STREAM_WINDOWS])
    gsm = GradSampleModule(pm, batch_first=True, loss_reduction="sum", strict=True)
    gsm.train()
    base_opt = torch.optim.AdamW(params, lr=PF.LR, betas=PF.ADAM["betas"], eps=PF.ADAM["eps"],
                                 weight_decay=PF.ADAM["weight_decay"])
    gen = torch.Generator(device=dev)
    gen.manual_seed(args.seed + PF.NOISE_SEED_OFFSET)
    opt = DPOptimizer(base_opt, noise_multiplier=sigma, max_grad_norm=PF.CLIP, expected_batch_size=PF.BATCH,
                      loss_reduction="mean", generator=gen)
    sched = torch.optim.lr_scheduler.LambdaLR(base_opt, lambda s: PF.lr_factor(s, T))
    steps = min(T, args.max_steps) if args.max_steps else T
    budgets = [mk["micro_max"], mk["tok_budget"]]
    log(f"{args.dataset} eps={args.eps} model={mk['id']} T={T} (running {steps}) trainable={n_train} device={dev}")

    def one_step(sel, ks):
        mbs = pack(slen[sel], *budgets)
        losses, norms = [], []
        if not mbs:
            for p in params:
                p.grad_sample = torch.zeros((0, *p.shape), device=p.device, dtype=p.dtype)
            opt.step()
            opt.zero_grad(set_to_none=True)
        for j, mb in enumerate(mbs):
            per = loss_fn([seqs[sel[i]] for i in mb], [int(starts[sel[i]][ks[i]]) for i in mb])
            per.sum().backward()
            if args.log_private:
                gs = opt.grad_samples
                norms.append(torch.stack([g.reshape(len(g), -1).norm(2, dim=-1) for g in gs], 1).norm(2, 1))
                losses.append(per.detach())
            if j < len(mbs) - 1:
                opt.signal_skip_step(do_skip=True)
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
        opt._step_skip_queue = []
        opt._is_last_step_skipped = False
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    hist, n_empty, n_oom, n_examples, n_tok, n_tgt = [], 0, 0, 0, 0, 0
    t_loop = time.time()
    for step in range(steps):
        sel = np.flatnonzero(rng_p.random(N) < q)
        ks = rng_w.integers(0, nrows[sel]) if len(sel) else np.zeros(0, np.int64)
        n_empty += int(len(sel) == 0)
        n_examples += len(sel)
        n_tok += int(slen[sel].sum())
        n_tgt += int(sum(slen[i] - starts[i][k] for i, k in zip(sel, ks)))
        lr_used = sched.get_last_lr()[0]
        while True:
            try:
                losses, norms, n_micro = one_step(sel, ks)
                break
            except torch.OutOfMemoryError:
                reset_after_oom()
                n_oom += 1
                budgets[:] = [max(1, budgets[0] // 2), budgets[1] // 2]
                log(f"CUDA OOM at step {step}: retrying the same batch with budgets {budgets}")
                if budgets[1] < 512:
                    raise
        sched.step()
        if not all(bool(torch.isfinite(p).all()) for p in params):
            raise FloatingPointError(f"non-finite LoRA weights after step {step}")
        if args.log_private and (step % args.log_every == 0 or step == steps - 1):
            Lv = torch.cat(losses).float() if losses else torch.zeros(0)
            row = {"step": step, "lr": lr_used, "batch": int(len(sel)), "n_micro": n_micro,
                   "loss": float(Lv.mean()) if len(Lv) else float("nan"), "sec": round(time.time() - t_loop, 1)}
            if norms:
                nn_ = torch.cat(norms).float()
                row.update({"gnorm_med": float(nn_.median()), "gnorm_p10": float(nn_.quantile(0.1)),
                            "gnorm_p90": float(nn_.quantile(0.9)), "clipped": float((nn_ > PF.CLIP).float().mean())})
            if dev.type == "cuda":
                row["max_mem_gb"] = round(torch.cuda.max_memory_allocated(dev) / 2**30, 2)
            hist.append(row)
            log(json.dumps(row))
        elif step % 100 == 0:
            log(f"step {step}/{steps}")
    t_loop = time.time() - t_loop
    pstats = {"empty_batches": n_empty, "oom_retries": n_oom, "micro_budgets": budgets, "n_trails": N,
              "examples": n_examples, "tokens": n_tok, "target_tokens": n_tgt, "encode": enc_info,
              "tokenize_sec": round(t_tok, 1), "train_loop_sec": round(t_loop, 1)}
    info = {"method": "path", "version": PF.VERSION, "row": mk["row"], "model": mk["id"],
            "model_revision": mk["revision"],
            "dataset": args.dataset, "eps_target": args.eps, "delta": PF.DELTA, "seed": args.seed,
            "privacy": "DP-SGD, trail-level, add/remove-one neighbours", "sampling": "Poisson",
            "sample_rate": q, "steps": T, "noise_multiplier": sigma, "clip": PF.CLIP, "b_nom": PF.BATCH,
            "batch_ref": PF.BATCH, "epochs": PF.EPOCHS, "n_ref": PF.N_REF, "lr": PF.LR,
            "schedule": f"linear warm-up {PF.WARMUP_FRAC:.0%} of the steps, then cosine to 0",
            "optimizer": "AdamW(betas=(0.9, 0.999), eps=1e-8, weight_decay=0)", "lora_rank": PF.RANK,
            "lora_alpha": PF.ALPHA, "lora_dropout": PF.DROPOUT, "lora_targets": PF.TARGETS,
            "trainable_params": n_train, "lora_dtype": "float32", "base_dtype": str(dtype),
            "grad_ckpt": not args.no_grad_ckpt, "micro_max": mk["micro_max"], "tok_budget": mk["tok_budget"],
            "window": "k ~ Uniform{0..L-1} per draw, stream [seed, 220001]; loss on the target tokens only",
            "text": {"header": PF.header(args.dataset, tok), "decimals": PF.DECIMALS, "eos": PF.EOT,
                     "chat_template": True,
                     "tokenization": "tok(H(d) + text) + [<|im_end|>] == DP-LoRA v2 prompt_ids + tok(text) + "
                                     "[<|im_end|>] (checked for every trail)"},
            "calibration": cal, "eps_prv": acc["eps_prv"], "eps_rdp": acc["eps_rdp"],
            "eps_sgd_part_prv": part["eps_prv"], "eps_sgd_part_rdp": part["eps_rdp"],
            "accountant": "PLD (dp_accounting) of DP-SGD composed with the selection Gaussian (calibration, target "
                          "eps - 0.015); eps_prv / eps_rdp = Opacus cross-checks of the same composition",
            "torch": torch.__version__, "device": str(dev), "limit": args.limit}
    if dev.type == "cuda":
        torch.cuda.synchronize()
        pstats["max_mem_allocated_gb"] = round(torch.cuda.max_memory_allocated(dev) / 2**30, 2)
        pstats["max_mem_reserved_gb"] = round(torch.cuda.max_memory_reserved(dev) / 2**30, 2)
        info["gpu"] = torch.cuda.get_device_name(dev)
    pm.save_pretrained(out / "adapter")
    pstats["train_total_sec"] = round(time.time() - t_start, 1)
    info.update({"steps_run": steps, "private_stats": pstats,
                 "complete": steps == T and not args.limit})
    if args.log_private:
        info["log"] = hist
    tmp = out / "train.json.tmp"
    tmp.write_text(json.dumps(info, indent=1, default=str))
    tmp.rename(out / "train.json")
    log(f"done: loop {t_loop:.0f}s, total {time.time() - t_start:.0f}s -> {out}")


if __name__ == "__main__":
    main()
