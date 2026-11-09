from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import numpy as np
import torch

from downstream.data import data_version, load_catalog, load_trails, vocabulary
from downstream.markov import MarkovBackoff
from downstream.metrics import RankingMetrics
from downstream.transformer import (TARGET_TYPES, DeviceTrails, TransformerConfig, iter_scores,
                                    train_transformer)
from trajpe.paths import PROCESSED

TUNING_DATASET = "jakarta"
PROTOCOL = "v2.1"
PREFIX_GROUPS = ("k=1", "k=2", "k=3-4", "k>=5")
POP_GROUPS = ("head", "torso", "tail")


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


class Evaluator:

    def __init__(self, catalog, vocab: np.ndarray, device: torch.device, pop_group: np.ndarray | None = None):
        rows = torch.as_tensor(vocab, device=device)
        self.to_cand = torch.full((catalog.n_venues,), -1, dtype=torch.long, device=device)
        self.to_cand[rows] = torch.arange(len(vocab), device=device)
        self.venue_cat = torch.as_tensor(catalog.cat_idx, device=device)
        self.cand_cat, self.n_cats = self.venue_cat[rows], catalog.n_cats
        self.group_edges = torch.tensor([1, 2, 4], device=device)
        self.pop_group = torch.as_tensor(pop_group, device=device) if pop_group is not None else None

    def __call__(self, score_iter, category: bool = False) -> dict:
        acc = RankingMetrics(ks=(1, 5, 10), ndcg_k=10, group_names=list(PREFIX_GROUPS),
                             cand_cat=self.cand_cat if category else None, n_cats=self.n_cats)
        extra = {}
        if self.pop_group is not None:
            extra["by_target_popularity"] = [RankingMetrics(group_names=list(POP_GROUPS)), POP_GROUPS, None]
        for item in score_iter:
            scores, true_rows, k = item[:3]
            target = self.to_cand[true_rows]
            acc.update(scores, target, group=torch.bucketize(k, self.group_edges),
                       target_cat=self.venue_cat[true_rows] if category else None)
            groups = {}
            if self.pop_group is not None:
                groups["by_target_popularity"] = self.pop_group[true_rows]
            if len(item) > 3:
                if "by_target_type" not in extra:
                    extra["by_target_type"] = [RankingMetrics(group_names=list(TARGET_TYPES)), TARGET_TYPES, None]
                groups["by_target_type"] = item[3]
            for name, g in groups.items():
                e = extra[name]
                e[0].update(scores, target, group=g)
                n = len(e[1])
                cnt = torch.stack([torch.bincount(g, minlength=n), torch.bincount(g[target >= 0], minlength=n)])
                e[2] = cnt.cpu() if e[2] is None else e[2] + cnt.cpu()
        res = acc.result()
        res["mrr_in_vocab"] = res["mrr"] / max(res["target_in_vocab"], 1e-12)
        for name, (m, names, cnt) in extra.items():
            block = m.result()["by_group"]
            total = max(int(cnt[0].sum()), 1)
            for i, g in enumerate(names):
                block[g]["share"] = int(cnt[0, i]) / total
                block[g]["in_vocab"] = int(cnt[1, i]) / max(int(cnt[0, i]), 1)
                block[g]["mrr_in_vocab"] = block[g]["mrr"] / max(block[g]["in_vocab"], 1e-12)
            res[name] = block
        return res


def popularity_groups(catalog, real_train) -> np.ndarray:
    counts = np.bincount(real_train.checkin_venues(), minlength=catalog.n_venues)
    order = np.argsort(-counts, kind="stable")
    before = (np.cumsum(counts[order]) - counts[order]) / max(int(counts.sum()), 1)
    g = np.empty(catalog.n_venues, dtype=np.int64)
    g[order] = np.where(before < 1 / 3, 0, np.where(before < 2 / 3, 1, 2))
    g[counts == 0] = 2
    return g


def parse_overrides(items: list[str]) -> dict:
    out = {}
    for it in items:
        key, _, val = it.partition("=")
        try:
            out[key] = json.loads(val)
        except json.JSONDecodeError:
            out[key] = val
    return out


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="city directory under trajpe.paths.PROCESSED, e.g. nyc")
    ap.add_argument("--train", required=True, type=Path, help="trails parquet to train on (real or synthetic)")
    ap.add_argument("--out", required=True, type=Path, help="output JSON path")
    ap.add_argument("--model", default="transformer", choices=["transformer", "markov"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None,
                    help="transformer: max epochs under early stopping, exact epochs with --early-stop none")
    ap.add_argument("--early-stop", default="auto", choices=["auto", "val", "holdout", "none"],
                    help="auto = 'val' (real val split) for every training set, or 'holdout' when --eval-split "
                         "is val; holdout = 5%% of --train (protocol v1, grouped by venue sequence)")
    ap.add_argument("--holdout-frac", type=float, default=0.05)
    ap.add_argument("--holdout", default="grouped", choices=["grouped", "random"],
                    help="grouped: trails with the same venue sequence stay on one side of the holdout split")
    ap.add_argument("--max-trails", type=int, default=None,
                    help="randomly subsample (seeded) the training trails to at most N before anything else")
    ap.add_argument("--eval-split", default="test", choices=["test", "val"],
                    help="real split to evaluate on; 'val' is meant only for hyper-parameter tuning on jakarta")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override a model hyper-parameter (for tuning on jakarta / ablations; recorded in the JSON)")
    ap.add_argument("--max-gpu-gb", type=float, default=20.0, help="hard cap on this process's GPU memory")
    ap.add_argument("--bench-metrics", action="store_true",
                    help="also report the benchmark protocol v1 next-POI metrics (Acc@1/5/10/20, NDCG@5/10, MRR; last "
                         "check-in of each test trail and every position) under 'bench_metrics'; off: output unchanged")
    args = ap.parse_args(argv)
    overrides = parse_overrides(args.set)

    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_gpu_gb * 2**30 / total), device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    data_dir = PROCESSED / args.dataset
    if not all((data_dir / f).exists() for f in ("pois.parquet", "test.parquet", "meta.json")):
        ap.error(f"{data_dir} is not a processed city directory")
    real_train = args.train.resolve() == (data_dir / "train.parquet").resolve()
    early_stop = args.early_stop
    if args.model == "markov":
        early_stop = "n/a"
    elif early_stop == "auto":
        early_stop = "holdout" if args.eval_split == "val" else "val"
    if early_stop == "val" and args.eval_split == "val":
        ap.error("cannot early-stop on the split that is evaluated; use --early-stop holdout/none")
    if (overrides or args.eval_split == "val") and args.dataset != TUNING_DATASET:
        log(f"WARNING: hyper-parameter overrides / val evaluation outside the tuning dataset ({TUNING_DATASET})")

    log(f"dataset={args.dataset} model={args.model} seed={args.seed} train={args.train} early_stop={early_stop}")
    catalog = load_catalog(data_dir)
    train, train_stats = load_trails(args.train, catalog)
    if args.max_trails is not None and len(train) > args.max_trails:
        keep = np.random.default_rng(args.seed).choice(len(train), args.max_trails, replace=False)
        train = train.subset(np.sort(keep))
        train_stats.update(subsampled_to=len(train), trails=len(train), checkins=int(train.length.sum()),
                           targets=train.n_targets, venues_covered=int(np.unique(train.checkin_venues()).size))
    vocab = vocabulary(train)
    evaluator = Evaluator(catalog, vocab, device)
    evalset, eval_stats = load_trails(data_dir / f"{args.eval_split}.parquet", catalog)
    log(f"catalog: {catalog.n_venues} venues, {catalog.n_cats} categories | vocabulary: {len(vocab)} venues | "
        f"train: {train_stats} | {args.eval_split}: {eval_stats['trails']} trails, {eval_stats['targets']} targets")
    t_load = time.time()

    if args.model == "markov":
        model = MarkovBackoff(catalog, vocab, device, seed=args.seed, **overrides).fit(train)
        train_info = {"hparams": model.hparams()}
        scores = model.iter_scores(evalset)
    else:
        cfg = TransformerConfig(**overrides)
        if args.epochs is not None:
            cfg.max_epochs = args.epochs
        stop_set, stop_stats = None, None
        if early_stop == "val":
            stop_set, stop_stats = load_trails(data_dir / "val.parquet", catalog)
        elif early_stop == "holdout":
            train, stop_set = train.split_holdout(args.holdout_frac, args.seed, grouped=args.holdout == "grouped")
            stop_stats = {"trails": len(stop_set), "targets": stop_set.n_targets, "fit_trails": len(train),
                          "split": args.holdout}
        model, train_info = train_transformer(catalog, vocab, train, stop_set, cfg, device, args.seed, evaluator, log)
        train_info["stop_set"] = stop_stats
        scores = iter_scores(model, DeviceTrails(evalset, device), cfg.logit_budget, with_types=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_train = time.time()
    real_train_trails = train if real_train and args.max_trails is None else load_trails(data_dir / "train.parquet", catalog)[0]
    final_evaluator = Evaluator(catalog, vocab, device, pop_group=popularity_groups(catalog, real_train_trails))
    bench = None
    if args.bench_metrics:
        from downstream.bench_ranking import BenchRanking
        bench = BenchRanking(final_evaluator.to_cand, evalset.length)
        scores = bench.tap(scores)
    metrics = final_evaluator(scores, category=True)
    t_eval = time.time()

    result = {
        "dataset": args.dataset, "data_dir": str(data_dir), "data_version": data_version(data_dir),
        "model": args.model, "seed": args.seed, "max_trails": args.max_trails,
        "train_path": str(args.train.resolve()), "train_is_real_split": real_train,
        "eval_split": args.eval_split, "early_stop": early_stop, "protocol": PROTOCOL,
        "metrics": metrics,
        "vocab_size": int(len(vocab)), "eval_target_vocab_coverage": metrics["target_in_vocab"],
        "train_data": train_stats, "eval_data": eval_stats,
        "train_info": train_info, "hparam_overrides": overrides,
        "runtime_s": {"load": round(t_load - t0, 1), "train": round(t_train - t_load, 1),
                      "eval": round(t_eval - t_train, 1), "total": round(t_eval - t0, 1)},
        "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device.type == "cuda" else None,
        "env": {"torch": torch.__version__, "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu"},
        "created": datetime.now().isoformat(timespec="seconds"),
        "argv": sys.argv if argv is None else argv,
    }
    if bench is not None:
        result["bench_metrics"] = bench.result()
        log("bench_metrics: " + "  ".join(f"{s}:{k}={v:.4f}" for s, b in result["bench_metrics"].items()
                                           for k, v in b.items() if k in ("acc@1", "ndcg@5", "mrr")))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    fmt = lambda m: "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                              for k, v in m.items() if not isinstance(v, dict))
    log("metrics: " + fmt(metrics))
    for name, m in metrics["by_group"].items():
        log(f"  {name:8s} " + fmt(m))
    log("  category " + fmt(metrics["category"]))
    log(f"runtime: {result['runtime_s']}  gpu_peak_gb={result['gpu_peak_gb']}  -> {args.out}")
    return result


if __name__ == "__main__":
    main()
