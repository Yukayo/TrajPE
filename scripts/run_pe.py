import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from trajpe.api_sim import SimAPI, VariationConfig
from trajpe.catalog import Catalog
from trajpe.data import load_split
from trajpe.embed import EmbedConfig
from trajpe.metrics import Fidelity
from trajpe.paths import RESULTS, RUNS
from trajpe.pe import PEConfig, run_pe
from trajpe.trails import from_frame, to_frame

V3_FIELDS = ("work_frac", "work_families", "calib", "calib_families", "calib_iters", "calib_cap", "calib_cap_early",
             "calib_eta", "calib_floor", "calib_solver", "time_ground", "release_mode", "fuse_families", "fuse_gamma",
             "fuse_rho", "fuse_iters", "release_pool", "release_degree", "pool_diversify", "pool_solver",
             "fuse_gamma_refit", "release_sampler", "sem_frac", "sem_var", "sem_k", "sem_scale_km", "venue2_frac",
             "known_sigmas", "sem_refine", "sem_refine_km", "sem_refine_mode",
             "sem_refine_loop", "sem_refine_stage", "sem_refine_rake", "pool_time_jitter", "hop_law", "pool_stretch")
V5_FIELDS = ("release_union_raw", "vote_cluster_kappa", "work_shrink")
METRICS_DEFAULT = False


def private_metrics_enabled(requested):
    if os.environ.get("TRAJPE_GRID_NOMETRICS") == "1":
        if requested:
            sys.exit("--metrics conflicts with TRAJPE_GRID_NOMETRICS=1 (release mode: no private-data metric)")
        return False
    return METRICS_DEFAULT if requested is None else bool(requested)


def build_api(name, catalog, args):
    vcfg = VariationConfig(p_successor=args.p_successor, p_snap=args.p_snap, p_return=args.p_return,
                           p_same_cat=args.p_same_cat, time_shift_min=args.time_shift_min)
    if name == "sim":
        api = SimAPI(catalog, vcfg)
        sims = [api.random]
    elif name == "llm":
        from trajpe.api_llm import LLMAPI
        api = LLMAPI(catalog, variation=args.llm_variation, llm_frac=args.llm_frac, sim_frac=args.sim_frac,
                     var_cfg=vcfg)
        api.ground.consistent = args.consistent_ground == "on"
        sims = [api.fallback]
    else:
        raise ValueError(name)
    for s in sims:
        s.q_trans, s.q_popular, s.q_return = args.q_trans, args.q_popular, args.q_return
    return api


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nyc")
    ap.add_argument("--api", default="sim", choices=["sim", "llm"])
    ap.add_argument("--eps", type=float, default=4.0)
    ap.add_argument("--delta", type=float, default=1e-6)
    ap.add_argument("--T", type=int, default=10)
    ap.add_argument("--n_syn", type=int, default=10000)
    ap.add_argument("--n_out", type=int, default=0, help="0 -> DEFAULT_N_OUT (public constant)")
    ap.add_argument("--unit", default="trail", choices=["trail", "user"])
    ap.add_argument("--threshold_sigmas", type=float, default=2.0)
    ap.add_argument("--degree_start", type=float, default=1.0)
    ap.add_argument("--degree_end", type=float, default=0.1)
    ap.add_argument("--vote_k", type=int, default=1)
    ap.add_argument("--cluster_c", type=float, default=0.0)
    ap.add_argument("--marginals", default="on", choices=["on", "off"])
    ap.add_argument("--marg_frac", type=float, default=0.2)
    ap.add_argument("--pop_alpha_share", type=float, default=0.3)
    ap.add_argument("--pop_threshold_sigmas", type=float, default=None, help="None: ~100 pure-noise venues pass")
    ap.add_argument("--marg_use", default="prompt,ground,var", help="stages steered by the DP marginals")
    ap.add_argument("--venue_aware", default="on", choices=["on", "off"])
    ap.add_argument("--max_popular", type=int, default=2000)
    ap.add_argument("--transitions", default="on", choices=["on", "off"])
    ap.add_argument("--embed", default="mobility", choices=["mobility", "text"],
                    help="text = generic MiniLM embedding of serialized trails (naive-PE baseline)")
    ap.add_argument("--q_trans", type=float, default=0.5, help="simulator: share of next venues from DP transitions")
    ap.add_argument("--q_popular", type=float, default=0.5, help="simulator: share of next venues snapped to popular")
    ap.add_argument("--p_successor", type=float, default=0.4)
    ap.add_argument("--p_snap", type=float, default=0.3)
    ap.add_argument("--q_return", type=float, default=0.0, help="simulator: share of steps returning to an earlier venue")
    ap.add_argument("--p_return", type=float, default=0.0, help="variation: per-trail probability of a return (A-B-A)")
    ap.add_argument("--w_ret", type=float, default=0.0, help="embedding weight of the return/repeat block")
    ap.add_argument("--consistent_ground", default="off", choices=["on", "off"],
                    help="LLM grounding: the same category and place twice in a trail -> the same venue")
    ap.add_argument("--release_diversify", type=float, default=0.0)
    ap.add_argument("--diversify_radius_km", type=float, default=0.5)
    ap.add_argument("--llm_variation", default="rule", choices=["rule", "llm", "mix"])
    ap.add_argument("--llm_frac", type=float, default=0.2)
    ap.add_argument("--sim_frac", type=float, default=0.0, help="share of the initial population from the simulator")
    _C = PEConfig
    v3 = ap.add_argument_group("v3")
    v3.add_argument("--work_frac", type=float, default=_C.work_frac, help="share of mu^2 for the workload marginals")
    v3.add_argument("--work_families", default=_C.work_families)
    v3.add_argument("--calib", default=_C.calib, choices=["none", "final", "all"])
    v3.add_argument("--calib_families", default=_C.calib_families)
    v3.add_argument("--calib_iters", type=int, default=_C.calib_iters)
    v3.add_argument("--calib_cap", type=float, default=_C.calib_cap)
    v3.add_argument("--calib_cap_early", type=float, default=_C.calib_cap_early)
    v3.add_argument("--calib_eta", type=float, default=_C.calib_eta)
    v3.add_argument("--calib_floor", type=float, default=_C.calib_floor)
    v3.add_argument("--calib_solver", default=_C.calib_solver, choices=["cyclic", "parallel", "fusion"])
    v3.add_argument("--time_ground", default=_C.time_ground, choices=["none", "release", "all"])
    v3.add_argument("--release_mode", default=_C.release_mode, choices=["vote", "fusion"])
    v3.add_argument("--fuse_families", default=_C.fuse_families)
    v3.add_argument("--fuse_gamma", type=float, default=_C.fuse_gamma)
    v3.add_argument("--fuse_rho", type=float, default=_C.fuse_rho)
    v3.add_argument("--fuse_iters", type=int, default=_C.fuse_iters)
    v3.add_argument("--release_pool", type=int, default=_C.release_pool)
    v3.add_argument("--release_degree", type=float, default=_C.release_degree)
    v3.add_argument("--pool_diversify", type=float, default=_C.pool_diversify)
    v3.add_argument("--pool_solver", default=_C.pool_solver, choices=["calib", "fusion"])
    v3.add_argument("--fuse_gamma_refit", type=float, default=_C.fuse_gamma_refit)
    v3.add_argument("--release_sampler", default=_C.release_sampler, choices=["multinomial", "systematic"])
    v31 = ap.add_argument_group("v3.1 (defaults keep v3 / the frozen method bit for bit)")
    v31.add_argument("--sem_frac", type=float, default=_C.sem_frac, help="share of generated trails from the semantic model")
    v31.add_argument("--sem_var", type=float, default=_C.sem_var, help="share of variation successor steps from it")
    v31.add_argument("--sem_k", type=int, default=_C.sem_k)
    v31.add_argument("--sem_scale_km", type=float, default=_C.sem_scale_km)
    v31.add_argument("--w_venue", type=float, default=EmbedConfig.w_venue, help="embedding: DP-popular venue block")
    v31.add_argument("--w_cat", type=float, default=EmbedConfig.w_cat, help="embedding: fine-category histogram")
    v31.add_argument("--w_cbigram", type=float, default=EmbedConfig.w_cbigram, help="embedding: category bigrams")
    v31.add_argument("--sem_refine", type=float, default=_C.sem_refine,
                     help="pool release: share of check-ins re-drawn along the DP category chain within their root")
    v31.add_argument("--sem_refine_km", type=float, default=_C.sem_refine_km)
    v31.add_argument("--sem_refine_mode", default=_C.sem_refine_mode, choices=["snap", "gibbs"])
    v31.add_argument("--sem_refine_stage", default=_C.sem_refine_stage, choices=["post_div", "pre_div"])
    v31.add_argument("--sem_refine_rake", type=int, default=_C.sem_refine_rake,
                     help="pilot passes that rake the pool re-draw's categories to the DP category share (0 = off)")
    v31.add_argument("--sem_refine_loop", type=float, default=_C.sem_refine_loop,
                     help="the same re-draw on every generation before voting")
    v31.add_argument("--known_sigmas", type=float, default=_C.known_sigmas,
                     help="pooled workload families (> 1000 bins) keep the bins whose noisy count exceeds this * sigma_w")
    v31.add_argument("--venue2_frac", type=float, default=_C.venue2_frac,
                     help="share of mu^2 for a second venue-popularity release (from the votes)")
    v31.add_argument("--p_same_cat", type=float, default=VariationConfig.p_same_cat,
                     help="variation: per-trail probability of a same-category hop at degree 1")
    v31.add_argument("--time_shift_min", type=float, default=VariationConfig.time_shift_min,
                     help="variation: std of the whole-trail time shift at degree 1 (minutes)")
    v32 = ap.add_argument_group("v3.2 (defaults keep v3.1 / v3 / the frozen method bit for bit)")
    v32.add_argument("--pool_time_jitter", type=int, default=_C.pool_time_jitter,
                     help="pool release: shift each child's whole trail by U{-m..m} minutes before the pool fusion")
    v32.add_argument("--hop_law", default=_C.hop_law, choices=["exp", "dp"],
                     help="simulator hop distances: exponential (frozen) or the DP hop workload histogram")
    v32.add_argument("--pool_stretch", type=float, default=_C.pool_stretch,
                     help="pool release: share of the pool added as geometric children covering the DP travel-distance excess")
    v5 = ap.add_argument_group("v5 (defaults keep v4 / v3.2 bit for bit; results/tuning/jakarta_v5.md)")
    v5.add_argument("--release_union_raw", type=int, default=_C.release_union_raw,
                    help="pool release: fuse over the children UNION every untouched raw pool sample (PoolAPI)")
    v5.add_argument("--vote_cluster_kappa", type=float, default=_C.vote_cluster_kappa,
                    help="SNR-matched cluster votes: K = clip(n_hat / (kappa sigma), 64, n_syn / 8) (0 = off)")
    v5.add_argument("--work_shrink", default=_C.work_shrink, choices=["none", "js"],
                    help="James-Stein shrinkage of the calibration / fusion workload targets toward the raw pool")
    v3.add_argument("--save_final", default="off", choices=["on", "off"],
                    help="save the final population, its noisy votes and the DP workload in the run directory")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--metrics", action=argparse.BooleanOptionalAction, default=None,
                    help="compute vs_train / vs_test on the private splits (internal evaluation only, never part of a "
                         "release; default: METRICS_DEFAULT, always off with TRAJPE_GRID_NOMETRICS=1)")
    args = ap.parse_args()
    nometrics = not private_metrics_enabled(args.metrics)
    del args.metrics

    catalog = Catalog(args.dataset)
    priv, users = from_frame(load_split(args.dataset, "train"), catalog)
    test = None if nometrics else from_frame(load_split(args.dataset, "test"), catalog)[0]
    eps_s = "inf" if not np.isfinite(args.eps) else f"{args.eps:g}"
    marg_s = f"m{args.marg_frac:g}" if args.marginals == "on" else "m0"
    var_s = "" if args.api == "sim" or args.llm_variation == "rule" else f"_{args.llm_variation}"
    tag = args.tag or f"{args.api}{var_s}_eps{eps_s}_T{args.T}_n{args.n_syn}_{marg_s}_{args.unit}_s{args.seed}"
    out = RUNS / args.dataset / tag
    out.mkdir(parents=True, exist_ok=True)

    cfg = PEConfig(n_syn=args.n_syn, T=args.T, eps=args.eps, delta=args.delta, unit=args.unit,
                   threshold_sigmas=args.threshold_sigmas, degree_start=args.degree_start,
                   degree_end=args.degree_end, n_out=args.n_out, seed=args.seed,
                   use_marginals=args.marginals == "on", marg_frac=args.marg_frac, pop_alpha_share=args.pop_alpha_share, marg_use=args.marg_use,
                   pop_threshold_sigmas=args.pop_threshold_sigmas,
                   venue_aware=args.venue_aware == "on", max_popular=args.max_popular,
                   use_transitions=args.transitions == "on",
                   release_diversify=args.release_diversify, diversify_radius_km=args.diversify_radius_km,
                   vote_k=args.vote_k, cluster_c=args.cluster_c,
                   **{k: getattr(args, k) for k in V3_FIELDS + V5_FIELDS}, save_final=str(out) if args.save_final == "on" else "")
    ecfg = EmbedConfig(w_ret=args.w_ret, w_venue=args.w_venue, w_cat=args.w_cat, w_cbigram=args.w_cbigram,
                       text_model="sentence-transformers/all-MiniLM-L6-v2" if args.embed == "text" else "")
    api = build_api(args.api, catalog, args)
    t0 = time.time()
    with open(out / "log.txt", "w") as logf:
        def log(x):
            print(x, flush=True)
            logf.write(f"{x}\n")
            logf.flush()
        syn, history, privacy, cfg_d = run_pe(priv, users, api, catalog, cfg, ecfg, log)
    runtime = time.time() - t0

    syn.save(out / "synth.npz")
    to_frame(syn, catalog).to_parquet(out / "synth.parquet", index=False)
    if nometrics:
        res = {"dataset": args.dataset, "tag": tag, "api": args.api, "args": vars(args), "privacy": privacy,
               "pe": cfg_d, "embed": asdict(ecfg), "runtime_sec": round(runtime, 1), "history": history,
               "metrics": "not computed (TRAJPE_GRID_NOMETRICS=1)" if os.environ.get("TRAJPE_GRID_NOMETRICS") == "1"
               else "not computed (release mode)"}
        if hasattr(api, "usage"):
            res["llm_usage"] = api.usage()
        (out / "result.json").write_text(json.dumps(res, indent=1, default=float))
        print(json.dumps({"tag": tag, "runtime_sec": res["runtime_sec"]}))
        return
    fid = Fidelity(catalog)
    res = {"dataset": args.dataset, "tag": tag, "api": args.api, "args": vars(args), "privacy": privacy, "pe": cfg_d,
           "embed": asdict(ecfg), "runtime_sec": round(runtime, 1), "history": history,
           "vs_train": fid.compare(fid.stats(priv), fid.stats(syn)),
           "vs_test": fid.compare(fid.stats(test), fid.stats(syn))}
    if hasattr(api, "usage"):
        res["llm_usage"] = api.usage()
    (out / "result.json").write_text(json.dumps(res, indent=1, default=float))
    dest = RESULTS / "pe" / args.dataset
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"{tag}.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({"tag": tag, "runtime_sec": res["runtime_sec"], **res["vs_train"]}, default=float))


if __name__ == "__main__":
    main()
