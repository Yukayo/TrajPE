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
import ctcl_conf as C
import prv_ctcl
import topics as TP

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

textfmt = C.textfmt


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_examples(dataset: str, tok, table: dict, limit: int = 0):
    pois, bbox, files = TP.load_catalog(dataset)
    if files["pois_sha256"] != table["catalog"]["pois_sha256"]:
        raise SystemExit("the public catalog differs from the one the topic table was built from")
    lk = TP.VenueLookup(pois, bbox)
    del pois
    df = pd.read_parquet(C.DATA / dataset / "train.parquet",
                         columns=["trail_id", "venue_id", "category", "lat", "lon", "t_local", "pos"])
    texts = textfmt.serialize_frame(df, C.COORD_DECIMALS)
    taus = TP.assign_frame(df, lk)
    del df
    if len(texts) != len(taus):
        raise AssertionError("texts and topics are not aligned")
    if limit:
        texts, taus = texts[:limit], taus[:limit]
    end = tok.convert_tokens_to_ids("<|im_end|>")
    resp = [(r + [end])[:C.MAX_RESP_TOKENS] for r in tok(texts, add_special_tokens=False)["input_ids"]]
    del texts
    rows = table["rows"]
    prompts = {int(k): C.prompt_ids(tok, dataset, rows[int(k)]["condition"]) for k in np.unique(taus)}
    return prompts, resp, taus


class DeviceExamples:

    def __init__(self, prompts: dict, resp: list, taus: np.ndarray, device):
        self.n = len(resp)
        self.tau = np.asarray(taus, np.int64)
        self.prompt = {k: torch.tensor(v, dtype=torch.long, device=device) for k, v in prompts.items()}
        self.rlen = np.array([len(r) for r in resp], np.int64)
        self.rstart = np.concatenate([[0], np.cumsum(self.rlen)[:-1]])
        flat = np.fromiter((t for r in resp for t in r), dtype=np.int64, count=int(self.rlen.sum()))
        self.flat = torch.from_numpy(flat).to(device)

    def get(self, i: int):
        s = int(self.rstart[i])
        return self.prompt[int(self.tau[i])], self.flat[s:s + int(self.rlen[i])]


def model_parts(model, param: str):
    if param == "lora":
        inner = model.base_model.model
        params = [p for p in model.parameters() if p.requires_grad]
    else:
        inner = model
        params = list(model.parameters())
    return inner.model, inner.lm_head, params


def example_loss(core, head, prompt: torch.Tensor, resp: torch.Tensor) -> torch.Tensor:
    P, R = prompt.shape[0], resp.shape[0]
    ids = torch.cat([prompt, resp[:-1]]).unsqueeze(0)
    hidden = core(input_ids=ids, use_cache=False).last_hidden_state
    logits = head(hidden[:, P - 1:P - 1 + R, :]).float()
    return F.cross_entropy(logits.reshape(R, -1), resp, reduction="mean")


class ClippedSum:

    def __init__(self, params: list, clip: float):
        self.params, self.clip = params, float(clip)
        self.acc = [torch.zeros_like(p) for p in params]

    def zero(self):
        torch._foreach_zero_(self.acc)

    def add(self, loss: torch.Tensor) -> torch.Tensor:
        for p in self.params:
            p.grad = None
        loss.backward()
        grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in self.params]
        norm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads)))
        factor = (self.clip / (norm + 1e-6)).clamp(max=1.0)
        torch._foreach_mul_(grads, factor)
        torch._foreach_add_(self.acc, grads)
        for p in self.params:
            p.grad = None
        return norm.detach()


def noisy_update(opt, cs: ClippedSum, sigma: float, b_nom: float, lr: float, gen) -> None:
    for p, a in zip(cs.params, cs.acc):
        a.add_(torch.normal(0.0, sigma * cs.clip, size=tuple(p.shape), generator=gen, device=p.device,
                            dtype=p.dtype))
        a.div_(b_nom)
        p.grad = a
    for g in opt.param_groups:
        g["lr"] = lr
    opt.step()
    for p in cs.params:
        p.grad = None


def all_finite(params) -> bool:
    return bool(torch.isfinite(torch.stack(torch._foreach_norm(params))).all())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--eps", type=float, required=True)
    ap.add_argument("--variant", choices=["plain", "vp"], required=True)
    ap.add_argument("--config", choices=list(C.CONFIGS), default="code")
    ap.add_argument("--param", choices=["full", "lora"], default="full")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calib", required=True, help="calibrate.py output (calib.json)")
    ap.add_argument("--table", default="", help="public topic table (default baselines/ctcl/topics/<dataset>.json.gz)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--mem_cap_gb", type=float, default=0.0)
    ap.add_argument("--log_private", action="store_true", help="tuning city only")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--model", default=C.MODEL, help="test only: a local tiny model dir")
    ap.add_argument("--device", default="", help="test only: cpu")
    ap.add_argument("--max_steps", type=int, default=0, help="debug: stop early (run marked incomplete)")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N trails only (run marked incomplete)")
    args = ap.parse_args()
    t_start = time.time()
    C.check_dataset(args.dataset)
    if args.log_private and args.dataset != C.TUNING_CITY:
        raise SystemExit("--log_private is allowed on the tuning city (jakarta) only (R14)")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "train.json").exists():
        raise SystemExit(f"{out}/train.json exists: the pipeline renames an unusable run aside before retraining")
    cfg = C.CONFIGS[args.config]
    q, T, clip, b_nom = cfg["batch"] / C.N_REF, cfg["steps"], cfg["clip"], cfg["batch"]
    cal = json.loads(Path(args.calib).read_text())
    if (cal["variant"], cal["config"]) != (args.variant, args.config) or cal["sample_rate"] != q or \
            cal["steps"] != T or cal["eps_total"] != args.eps or cal["delta"] != C.DELTA or cal["b_nom"] != b_nom:
        raise SystemExit(f"calibration {args.calib} does not match (variant, config, q, T, eps, delta, B)")
    sigma = float(cal["sigma_sgd"])
    gauss = [(cal["sigma_topic_hist"], cal["n_topic_hist"])] + \
        ([(cal["sigma_prior"], cal["n_prior"])] if cal["n_prior"] else [])
    acc = prv_ctcl.prv_rdp(sigma, q, T, gauss, C.DELTA)
    if not (math.isfinite(acc["eps_prv"]) and acc["eps_prv"] <= args.eps):
        raise SystemExit(f"Opacus PRV epsilon {acc['eps_prv']} of the composition exceeds {args.eps}: refusing to train")
    log(f"DP-Adam: q={q:.7f} T={T} sigma={sigma:.5f} C={clip} B_nom={b_nom}; composed PLD "
        f"{cal['eps_composed_pld']:.5f}, PRV {acc['eps_prv']:.4f}, RDP {acc['eps_rdp']:.3f} (eps {args.eps:g})")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers
    dev = torch.device(args.device) if args.device else \
        (torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu"))
    if dev.type == "cuda":
        if args.mem_cap_gb > 0:
            total = torch.cuda.get_device_properties(dev).total_memory / 2**30
            torch.cuda.set_per_process_memory_fraction(min(1.0, args.mem_cap_gb / total), dev)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    hub = args.model == C.MODEL
    kw = {"revision": C.MODEL_COMMIT} if hub else {}
    tok = AutoTokenizer.from_pretrained(C.MODEL, revision=C.MODEL_COMMIT)
    table_file = Path(args.table or C.table_path(args.dataset))
    table = TP.load_table(table_file)
    if table["dataset"] != args.dataset:
        raise SystemExit("topic table of another dataset")
    prompts, resp, taus = load_examples(args.dataset, tok, table, args.limit)
    data = DeviceExamples(prompts, resp, taus, dev)
    del prompts, resp, taus
    N = data.n
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, attn_implementation="sdpa",
                                                 **kw).to(dev)
    model.config.use_cache = False
    if args.param == "lora":
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=C.LORA["r"], lora_alpha=C.LORA["alpha"],
                                                 lora_dropout=C.LORA["dropout"], target_modules=C.LORA["targets"],
                                                 bias="none", task_type="CAUSAL_LM"))
    else:
        model.requires_grad_(True)
    model.train()
    core, head, params = model_parts(model, args.param)
    n_train = sum(p.numel() for p in params)
    if hub and args.param == "full" and n_train != C.N_PARAMS:
        raise AssertionError(f"{n_train} trainable parameters, expected {C.N_PARAMS}")
    opt = torch.optim.AdamW(params, lr=0.0, betas=cfg["betas"], eps=cfg["adam_eps"], weight_decay=cfg["weight_decay"],
                            foreach=True)
    cs = ClippedSum(params, clip)
    rng = np.random.default_rng(args.seed)
    gen = torch.Generator(device=dev)
    gen.manual_seed(args.seed + C.NOISE_OFFSET)
    steps = min(T, args.max_steps) if args.max_steps else T
    log(f"{args.dataset} eps={args.eps:g} {args.variant}/{args.config}/{args.param}: T={T} (running {steps}), "
        f"{n_train} trainable parameters, device {dev}")
    plog, t_loop = [], time.time()
    for s in range(steps):
        sel = np.flatnonzero(rng.random(N) < q)
        cs.zero()
        losses, norms = [], []
        for i in sel:
            prompt, r = data.get(int(i))
            loss = example_loss(core, head, prompt, r)
            norm = cs.add(loss)
            if args.log_private:
                losses.append(loss.detach())
                norms.append(norm)
        noisy_update(opt, cs, sigma, b_nom, C.lr_at(s, args.config), gen)
        if not all_finite(params):
            raise FloatingPointError(f"non-finite weights after step {s}")
        if args.log_private and (s % args.log_every == 0 or s == steps - 1):
            row = {"step": s, "lr": C.lr_at(s, args.config), "batch": int(len(sel)),
                   "loss": float(torch.stack(losses).mean()) if losses else None,
                   "gnorm_med": float(torch.stack(norms).median()) if norms else None,
                   "clipped": float((torch.stack(norms) > clip).float().mean()) if norms else None,
                   "sec": round(time.time() - t_loop, 1)}
            plog.append(row)
            log(json.dumps(row))
        elif s % args.log_every == 0 or s == steps - 1:
            mem = round(torch.cuda.max_memory_allocated(dev) / 2**30, 2) if dev.type == "cuda" else None
            log(f"step {s + 1}/{steps}  {time.time() - t_loop:.0f}s  max_mem_gb {mem}")
    t_loop = time.time() - t_loop
    complete = steps == T and not args.limit
    info = {"method": "ctcl", "variant": args.variant, "config": args.config, "param": args.param,
            "model": C.MODEL if hub else args.model, "model_commit": C.MODEL_COMMIT if hub else None,
            "dataset": args.dataset, "eps_target": args.eps, "delta": C.DELTA, "seed": args.seed,
            "privacy": "DP-Adam (DP-SGD with AdamW), trail-level, add/remove-one neighbours",
            "sampling": "Poisson", "sample_rate": q, "steps": T, "b_nom": b_nom, "clip": clip,
            "noise_multiplier": sigma, "lr": cfg["lr"], "weight_decay": cfg["weight_decay"],
            "warmup_steps": cfg["warmup_steps"], "betas": list(cfg["betas"]), "adam_eps": cfg["adam_eps"],
            "schedule": "optax linear: eta_s = lr s / W for s < W, then lr (T - s) / (T - W) (eta_0 = 0)",
            "optimizer": "AdamW (decoupled weight decay on every parameter), fp32 state",
            "per_example_gradients": "exact: one forward/backward pass per example (batch of one), clipped over all "
                                     "trainable parameters, factor min(1, C / (norm + 1e-6))",
            "precision": "fp32 weights / gradients / clipping / noise / optimizer state; TF32 matmuls",
            "max_resp_tokens": C.MAX_RESP_TOKENS, "coord_decimals": C.COORD_DECIMALS,
            "prompt": C.messages(args.dataset, "<CTCL condition of the trail's topic>"),
            "topic_table": str(table_file), "topic_table_sha256": C.sha256_file(table_file),
            "trainable_params": n_train,
            "lora": ({"r": C.LORA["r"], "alpha": C.LORA["alpha"], "dropout": C.LORA["dropout"],
                      "targets": C.LORA["targets"]} if args.param == "lora" else None),
            "accountant": "PLD (dp_accounting) of the whole release, calibrated to eps - 0.015 by calibrate.py; "
                          "eps_prv / eps_rdp = Opacus cross-checks of the same composition",
            "eps_pld": cal["eps_composed_pld"], "eps_composed_pld": cal["eps_composed_pld"],
            "eps_sgd_part_pld": cal["eps_sgd_only_pld"], "eps_prv": acc["eps_prv"], "eps_rdp": acc["eps_rdp"],
            "sigma_topic_hist": cal["sigma_topic_hist"], "n_topic_hist": cal["n_topic_hist"], "K_topics": C.K_TOPICS,
            "sigma_prior": cal["sigma_prior"], "n_prior": cal["n_prior"], "sigma_eq": cal["sigma_eq"],
            "vp_calibration": {"sigma_hist": cal["sigma_eq"], "n_hist": 1,
                               "note": "sigma_hist here is sigma_eq (the Gaussian part as one Gaussian) for "
                                       "baselines/dplora/pld_check.py; the prior's own sigma is sigma_prior"},
            "calib": {k: cal[k] for k in ("sigma_sgd", "eps_composed_pld", "eps_sgd_only_pld", "pld_target")},
            "device": str(dev), "torch": torch.__version__, "transformers": transformers.__version__,
            "code_sha256": C.code_hashes(), "steps_run": steps, "train_loop_sec": round(t_loop, 1),
            "complete": complete}
    if dev.type == "cuda":
        torch.cuda.synchronize()
        info["gpu"] = torch.cuda.get_device_name(dev)
    pstats = {"oom_retries": 0}
    if dev.type == "cuda":
        pstats["max_mem_allocated_gb"] = round(torch.cuda.max_memory_allocated(dev) / 2**30, 2)
        pstats["max_mem_reserved_gb"] = round(torch.cuda.max_memory_reserved(dev) / 2**30, 2)
    info["private_stats"] = pstats
    if args.log_private:
        info["log"] = plog
    wdir = out / ("adapter" if args.param == "lora" else "model")
    model.save_pretrained(wdir, safe_serialization=True)
    files = sorted(p for p in wdir.iterdir() if p.is_file())
    info["weights"] = {"dir": wdir.name, "files": {p.name: C.sha256_file(p) for p in files},
                       "bytes": int(sum(p.stat().st_size for p in files))}
    info["train_total_sec"] = round(time.time() - t_start, 1)
    C.write_json_atomic(out / "train.json", info)
    log(f"done: loop {t_loop:.0f}s, total {time.time() - t_start:.0f}s, complete={complete} -> {out}")


if __name__ == "__main__":
    main()
