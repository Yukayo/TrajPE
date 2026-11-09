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
import hashlib
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

import pathfmt as PF

sys.path.insert(0, str(PF.PROJECT))
from trajpe.api_llm import CITY_NAMES, WEEKDAYS, Grounder
from trajpe.catalog import Catalog
from trajpe.trails import TrailSet, to_frame

OUT_COLS = ["trail_id", "user_id", "venue_id", "lat", "lon", "category", "t_local", "pos"]
PRIVATE_KEYS = {"private_stats", "log", "train_sec", "train_total_sec", "train_loop_sec", "n_private_trails",
                "n_private_rows", "embed_private_sec", "knn_sec", "vote_sec", "raw_votes", "tokenize_sec",
                "response_tokens", "losses"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ground(texts, catalog, seed: int):
    cats = set(catalog.categories)
    g = Grounder(catalog, jitter_min=0)
    if g.venue_w is not None or g.popular_by_cat:
        raise SystemExit("PATH grounds without a DP venue prior (A8)")
    rng = np.random.default_rng([seed, PF.STREAM_GROUND])
    trails, dist, n_ck = [], [], 0
    for text in texts:
        res, why = PF.textfmt.parse(text, cats, True)
        if res is None:
            raise SystemExit(f"unparsable pool text ({why}): {text[:200]!r}")
        day, cks = res
        poi, t = g(cks, day, rng)
        xy = catalog.clip_xy(catalog.to_xy(np.array([c["lat"] for c in cks]), np.array([c["lon"] for c in cks])))
        dist.append(np.linalg.norm(catalog.xy[poi] - xy, axis=1))
        trails.append((poi, t))
        n_ck += len(cks)
    d = np.concatenate(dist)
    return TrailSet.from_lists(trails), {"checkins": int(n_ck), "dist_km_median": float(np.median(d)),
                                         "dist_km_p90": float(np.quantile(d, 0.9)), "dist_km_mean": float(d.mean())}


class RowEmbedder:

    def __init__(self, device: str):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch, self.dev = torch, device
        e = PF.EMBEDDER
        self.tok = AutoTokenizer.from_pretrained(e["id"], revision=e["revision"])
        self.model = AutoModel.from_pretrained(e["id"], revision=e["revision"], dtype=torch.bfloat16,
                                               attn_implementation="sdpa").to(device).eval()
        self.eos = self.tok.convert_tokens_to_ids("<|endoftext|>")
        self.dim = int(self.model.config.hidden_size)
        self.n_truncated = 0

    def __call__(self, texts: list[str], batch_tokens: int = 65536) -> np.ndarray:
        torch = self.torch
        uniq, inv = np.unique(np.asarray(texts, dtype=object), return_inverse=True)
        enc = self.tok(list(uniq), add_special_tokens=True, truncation=True, max_length=PF.EMBED_MAX_TOKENS)["input_ids"]
        if any(ids[-1] != self.eos for ids in enc):
            raise SystemExit("the embedder's tokenizer did not append <|endoftext|>")
        lens = np.array([len(ids) for ids in enc])
        self.n_truncated += int((lens >= PF.EMBED_MAX_TOKENS).sum())
        out = np.empty((len(uniq), self.dim), np.float32)
        with torch.no_grad():
            for L in np.unique(lens):
                idx = np.flatnonzero(lens == L)
                step = max(1, batch_tokens // int(L))
                for s in range(0, len(idx), step):
                    b = idx[s:s + step]
                    ids = torch.tensor([enc[i] for i in b], dtype=torch.long, device=self.dev)
                    h = self.model(input_ids=ids).last_hidden_state[:, -1, :].float()
                    out[b] = torch.nn.functional.normalize(h, p=2, dim=1).cpu().numpy()
        return out[np.asarray(inv).ravel()]


def table_embeddings(frame: pd.DataFrame, emb: RowEmbedder) -> tuple[np.ndarray, np.ndarray]:
    codes, uniq = pd.factorize(frame["trail_id"], sort=False)
    rows = emb(PF.row_texts(frame))
    order = np.argsort(codes, kind="stable")
    lens = np.bincount(codes)
    starts = np.r_[0, np.cumsum(lens)[:-1]]
    e = np.add.reduceat(rows[order], starts, axis=0) / lens[:, None]
    return e.astype(np.float32), np.asarray(uniq)


def all_keys(o) -> set:
    if isinstance(o, dict):
        return set(o) | set().union(*(all_keys(v) for v in o.values()))
    if isinstance(o, (list, tuple)):
        return set().union(*(all_keys(v) for v in o)) if o else set()
    return set()


def vote_counts(e_priv: np.ndarray, e_cand: np.ndarray, k: int = PF.VOTE_K) -> np.ndarray:
    from trajpe.embed import nearest_k
    idx = nearest_k(np.ascontiguousarray(e_priv, np.float32), np.ascontiguousarray(e_cand, np.float32), k)
    if idx.shape[1] != k or any(len(set(r)) != k for r in idx[: min(len(idx), 1000)]):
        raise SystemExit("nearest_k must return k distinct candidates per private table")
    return np.bincount(idx.ravel(), minlength=len(e_cand)).astype(np.float64)


def noisy_votes(h: np.ndarray, sigma_sel: float, rng) -> np.ndarray:
    return h + rng.normal(0.0, math.sqrt(PF.VOTE_K) * sigma_sel, len(h))


def top_n(noisy: np.ndarray, n: int, rng) -> np.ndarray:
    perm = rng.permutation(len(noisy))
    return perm[np.argsort(-noisy[perm], kind="stable")[:n]]


def select(h: np.ndarray, sigma_sel: float, n_out: int, seed: int):
    rng = np.random.default_rng([seed, PF.STREAM_SELECT])
    noisy = noisy_votes(h, sigma_sel, rng)
    keep = top_n(noisy, n_out, rng)
    order = np.random.default_rng([seed, PF.STREAM_SHUFFLE]).permutation(keep)
    return noisy, order


def release_frame(Z: TrailSet, idx: np.ndarray, catalog) -> pd.DataFrame:
    df = to_frame(Z.subset(idx), catalog, prefix="path")
    for c in ["trail_id", "user_id", "venue_id", "category"]:
        df[c] = df[c].astype(object)
    df["t_local"] = df["t_local"].astype("datetime64[ns]")
    df["pos"] = df["pos"].astype(np.int64)
    df["lat"] = df["lat"].astype(np.float64)
    df["lon"] = df["lon"].astype(np.float64)
    return df[OUT_COLS]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def accounting_block(cal: dict, tj: dict) -> dict:
    k = PF.VOTE_K
    return {
        "eps_total": cal["eps_total"], "delta": cal["delta"], "unit": "trail", "relation": "add/remove one trail",
        "composition": "joint PLD", "pld_target": cal["pld_target"],
        "eps_joint_pld": cal["eps_composed_pld"], "eps_joint_prv": tj["eps_prv"], "eps_joint_rdp": tj["eps_rdp"],
        "dp_sgd": {"q": cal["sample_rate"], "T": cal["steps"], "sigma_sgd": cal["sigma_sgd"], "clip": PF.CLIP,
                   "b_nom": PF.BATCH, "sampling": "Poisson", "eps_sgd_only_pld": cal["eps_sgd_only_pld"],
                   "eps_sgd_only_prv": tj.get("eps_sgd_part_prv"), "eps_sgd_only_rdp": tj.get("eps_sgd_part_rdp"),
                   "l2_sensitivity": PF.CLIP},
        "selection": {"mechanism": "Gaussian", "k": k, "l2_sensitivity": math.sqrt(k),
                      "eps_select": cal["eps_select"], "delta_select": cal["delta_select"],
                      "sigma_unit": cal["sigma_sel"], "noise_std": math.sqrt(k) * cal["sigma_sel"],
                      "eps_sel_only_pld": cal["eps_sel_only_pld"],
                      "normalised": "unit votes with noise sqrt(k) sigma_unit = votes of weight 1/sqrt(k) with "
                                    "noise sigma_unit (one Gaussian of L2 sensitivity 1)"},
        "paper_literal": cal["paper_literal"],
        "accountant": "dp_accounting PLD (add-or-remove-one, value_discretization_interval 1e-4) of "
                      "PoissonSampled(q, Gaussian(sigma_sgd))^T composed with Gaussian(sigma_unit); Opacus PRV "
                      "(gate, before training) and RDP (report) of the same composition",
        "guarantee": "(eps_total, delta)-DP at trail level, add/remove one trail: M1 (DP-SGD) and M2 (noisy vote "
                     "histogram) composed by joint PLD; sampling, parsing, grounding, embedding the candidates, the "
                     "top-n_out choice, tie-breaking and shuffling are post-processing",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--eps", type=float, required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model_key", default=PF.DEFAULT_MODEL, choices=list(PF.MODELS))
    ap.add_argument("--seed", type=int, default=PF.SEED)
    args = ap.parse_args()
    d = PF.check_dataset(args.dataset)
    assert PF.textfmt.WEEKDAYS == WEEKDAYS and PF.textfmt.CITY_NAMES == CITY_NAMES
    work, out = Path(args.work), Path(args.out)
    mk = PF.MODELS[args.model_key]
    t0 = time.time()

    cal = json.loads((work / "calib.json").read_text())
    tj = json.loads((work / "train.json").read_text())
    ps = tj.get("private_stats") or {}
    if ps.get("oom_retries", 0) != 0 or tj.get("complete") is not True or int(tj.get("steps_run", -1)) != cal["steps"]:
        raise SystemExit(f"refusing the release: train.json oom_retries={ps.get('oom_retries')} "
                         f"complete={tj.get('complete')} steps_run={tj.get('steps_run')}")
    if (float(tj["noise_multiplier"]), float(tj["sample_rate"]), int(tj["steps"])) != \
            (float(cal["sigma_sgd"]), float(cal["sample_rate"]), int(cal["steps"])) or \
            abs(float(cal["eps_total"]) - args.eps) > 1e-12 or tj["dataset"] != d or tj["model"] != mk["id"] \
            or tj.get("model_revision") != mk["revision"] or int(tj.get("seed", -1)) != args.seed:
        raise SystemExit("train.json does not match the calibration / dataset / model / seed of this release")
    if not float(tj["eps_prv"]) <= args.eps:
        raise SystemExit(f"Opacus PRV epsilon {tj['eps_prv']} exceeds {args.eps}")
    pool = pd.read_parquet(work / "pool.parquet")
    pinfo = json.loads((work / "pool.json").read_text())
    if len(pool) != PF.POOL_M or pinfo.get("n_kept") != PF.POOL_M or pinfo.get("status") != "ok":
        raise SystemExit(f"the pool has {len(pool)} candidates, expected exactly {PF.POOL_M}")
    if pinfo.get("model") != mk["id"] or pinfo.get("model_revision") != mk["revision"]:
        raise SystemExit("pool.json was sampled from another model")

    cat = Catalog(d)
    t = time.time()
    Z, gstats = ground(pool["text"].tolist(), cat, args.seed)
    ground_sec = time.time() - t
    log(f"grounded {len(Z)} candidates ({gstats['checkins']} check-ins) in {ground_sec:.0f}s")

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t = time.time()
    emb = RowEmbedder(dev)
    e_cand, cand_ids = table_embeddings(to_frame(Z, cat, prefix="cand"), emb)
    if len(e_cand) != len(Z) or list(cand_ids[:3]) != ["cand_0", "cand_1", "cand_2"] or cand_ids[-1] != f"cand_{len(Z) - 1}":
        raise SystemExit("candidate embeddings are not in pool order")
    trunc_cand = emb.n_truncated
    embed_cand_sec = time.time() - t
    log(f"embedded {len(Z)} candidates in {embed_cand_sec:.0f}s")

    t = time.time()
    priv = pd.read_parquet(PF.DATA / d / "train.parquet", columns=["trail_id", "lat", "lon", "category", "t_local", "pos"])
    n_priv_rows = len(priv)
    e_priv, _ = table_embeddings(priv, emb)
    n_priv = len(e_priv)
    del priv
    embed_priv_sec = time.time() - t
    t = time.time()
    h = vote_counts(e_priv, e_cand)
    del e_priv
    knn_sec = time.time() - t
    noisy, order = select(h, float(cal["sigma_sel"]), PF.N_OUT, args.seed)
    del h
    trunc_priv = emb.n_truncated - trunc_cand

    df = release_frame(Z, order, cat)
    assert df["trail_id"].nunique() == PF.N_OUT and len(np.unique(order)) == PF.N_OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(work / "noisy_votes.npy", noisy)
    side = {
        "method": "path", "version": PF.VERSION, "row": mk["row"], "dataset": d, "eps": args.eps, "delta": PF.DELTA,
        "seed": args.seed,
        "n_out": PF.N_OUT, "n_trails": int(df["trail_id"].nunique()), "n_checkins": int(len(df)),
        "paper": "Rosenblatt, Liu, McKenna, Ponomareva, Privately Fine-Tuned LLMs Preserve Temporal Dynamics in "
                 "Tabular Data (PATH), ICML 2026, PMLR 306 (arXiv 2602.02766)",
        "spec": "docs/baselines/PATH_SPEC.md incl. the owner decisions of section 10 (pre-registration frozen in "
                "frozen/path_v2*.sha256)",
        "settings": PF.settings(args.model_key),
        "accounting": accounting_block(cal, tj),
        "training": {k: tj.get(k) for k in ("model", "model_revision", "sample_rate", "steps", "noise_multiplier",
                                             "clip", "b_nom", "batch_ref", "epochs", "lr", "schedule", "optimizer",
                                             "lora_rank", "lora_alpha", "lora_dropout", "lora_targets",
                                             "trainable_params", "lora_dtype", "base_dtype", "grad_ckpt",
                                             "micro_max", "tok_budget", "window", "text", "torch", "gpu", "steps_run",
                                             "complete")},
        "sampling": {k: pinfo.get(k) for k in ("n_generated", "valid_rate", "reasons", "table_end", "table_end_kept",
                                                "cut_at_20", "generated_tokens", "rounds", "requests_per_round",
                                                "generation", "rule", "prompt_tokens", "gpu", "vllm_version", "m")},
        "grounding": {"method": "trajpe.api_llm.Grounder(Catalog(city), jitter_min=0) without venue weights: "
                                "nearest public venue of the category (A8)", "dp_prior": None,
                      "stream": [args.seed, PF.STREAM_GROUND], "pool": gstats},
        "selection": {"embedding": f"{PF.EMBEDDER['id']}@{PF.EMBEDDER['revision']}: per check-in row text "
                                   "'{weekday} {HH:MM} | {category} | {lat:.4f},{lon:.4f}', no instruction, "
                                   "last-token pooling, L2-normalised, <= 64 tokens; table = mean row embedding",
                      "embedding_dtype": "bfloat16 weights, float32 pooled output",
                      "distance": "Euclidean, exact (trajpe.embed.nearest_k)", "k": PF.VOTE_K, "pool_m": PF.POOL_M,
                      "noisy_threshold": float(noisy[order].min()), "n_distinct_released": int(len(np.unique(order))),
                      "ties": "seeded permutation then stable sort (stream [seed, 220031], after the noise)",
                      "streams": {"noise_and_ties": [args.seed, PF.STREAM_SELECT],
                                  "release_shuffle": [args.seed, PF.STREAM_SHUFFLE]},
                      "candidate_rows_truncated": int(trunc_cand)},
        "models": {"generator": {"id": mk["id"], "revision": mk["revision"], "licence": mk["licence"],
                                 "backbone_rule": mk["backbone_rule"]},
                   "embedder": {"id": PF.EMBEDDER["id"], "revision": PF.EMBEDDER["revision"],
                                "licence": PF.EMBEDDER["licence"]}},
        "timings_public_sec": {"sample_load": pinfo.get("sample_load_sec"), "sample_gen": pinfo.get("sample_gen_sec"),
                               "ground": round(ground_sec, 1), "embed_candidates": round(embed_cand_sec, 1)},
        "adapter_sha256": sha256_file(work / "adapter" / "adapter_model.safetensors"),
        "pool_sha256": sha256_file(work / "pool.parquet"),
        "work": str(work), "no_metric": "no metric is computed by this job (no eval_synth / score / eval_bench)",
    }
    bad = PRIVATE_KEYS & all_keys(side)
    if bad:
        raise SystemExit(f"release sidecar would carry work-dir-only fields {bad}")
    tmp = out.with_name(out.stem + ".tmp.parquet")
    df.to_parquet(tmp, index=False)
    out.with_suffix(".json").write_text(json.dumps(side, indent=1, default=str))
    tmp.rename(out)
    (work / "release_private.json").write_text(json.dumps({
        "note": "work dir only: depends on the private data beyond the DP outputs (A12, R5)",
        "n_private_trails": n_priv, "n_private_rows": n_priv_rows, "private_rows_truncated": int(trunc_priv),
        "embed_private_sec": round(embed_priv_sec, 1), "knn_sec": round(knn_sec, 1),
        "gpu_max_mem_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2) if dev == "cuda" else None,
        "release_total_sec": round(time.time() - t0, 1)}, indent=1))
    log(f"wrote {out}: {side['n_trails']} trails, {side['n_checkins']} check-ins; noisy threshold "
        f"{side['selection']['noisy_threshold']:.2f} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
