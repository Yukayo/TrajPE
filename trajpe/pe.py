from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from . import fusion as FZ
from . import v5 as V5
from . import workload as W
from .dp import eps_for
from .embed import EmbedConfig, embed, kmeans
from .marginals import N_MARGINALS, combine_popularity, dp_marginals, dp_venue_popularity
from .mechanisms import dp_nn_histogram, extra_sigma, noise_levels_w, vote_weights
from .trails import TrailSet


DEFAULT_N_OUT = 50000
WORK_STREAM = 104729
VENUE2_STREAM = 130363
STRETCH_STREAM = 180437
RAKE_STREAM = 155921
V3_FUSE_FAMILIES = W.DEFAULT_FAMILIES + ",start,cat,venue,cat_trans,venue_trans"


@dataclass
class PEConfig:
    n_syn: int = 10000
    T: int = 10
    eps: float = 4.0
    delta: float = 1e-6
    unit: str = "trail"
    threshold_sigmas: float = 2.0
    degree_start: float = 1.0
    degree_end: float = 0.1
    n_out: int = 0
    out_degree: float = 0.05
    use_marginals: bool = True
    marg_frac: float = 0.2
    pop_threshold_sigmas: float | None = None
    pop_alpha_share: float = 0.3
    marg_use: str = "prompt,ground,var"
    venue_aware: bool = True
    max_popular: int = 2000
    use_transitions: bool = True
    release_diversify: float = 0.0
    diversify_radius_km: float = 0.5
    vote_k: int = 1
    cluster_c: float = 0.0
    seed: int = 0
    work_frac: float = 0.0
    work_families: str = W.DEFAULT_FAMILIES
    calib: str = "none"
    calib_families: str = W.DEFAULT_FAMILIES + ",start,cat,venue"
    calib_iters: int = 200
    calib_cap: float = 5.0
    calib_cap_early: float = 3.0
    calib_eta: float = 1.0
    calib_floor: float = 0.0
    calib_solver: str = "parallel"
    time_ground: str = "none"
    release_mode: str = "vote"
    fuse_families: str = V3_FUSE_FAMILIES
    fuse_gamma: float = 1.0
    fuse_rho: float = 0.0
    fuse_iters: int = 300
    release_pool: int = 0
    release_degree: float = 0.15
    pool_diversify: float = 0.0
    pool_solver: str = "calib"
    fuse_gamma_refit: float = 1.0
    release_sampler: str = "multinomial"
    save_final: str = ""
    sem_frac: float = 0.0
    sem_var: float = 0.0
    sem_k: int = 10
    sem_scale_km: float = 0.5
    known_sigmas: float = W.KNOWN_SIGMAS
    sem_refine: float = 0.0
    sem_refine_km: float = 1.0
    sem_refine_mode: str = "snap"
    sem_refine_loop: float = 0.0
    sem_refine_stage: str = "post_div"
    sem_refine_rake: int = 0
    venue2_frac: float = 0.0
    hop_law: str = "exp"
    pool_stretch: float = 0.0
    pool_time_jitter: int = 0
    release_union_raw: int = 0
    vote_cluster_kappa: float = 0.0
    work_shrink: str = "none"


def sample_from_clusters(labels, n_clusters, probs, n, rng):
    sizes = np.bincount(labels, minlength=n_clusters)
    p = probs * (sizes > 0)
    p = p / p.sum()
    order = np.argsort(labels, kind="stable")
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    c = rng.choice(n_clusters, n, p=p)
    return order[starts[c] + (rng.random(n) * sizes[c]).astype(np.int64)]


def jitter_times(ts: TrailSet, m: int, rng) -> TrailSet:
    sh = rng.integers(-m, m + 1, len(ts))
    t0 = ts.tmin[ts.first].astype(np.int64)
    sh = np.mod(t0 + sh, 10080) - t0
    t = ts.tmin.astype(np.int64) + np.repeat(sh, ts.lengths)
    return TrailSet(ts.poi.copy(), t.astype(ts.tmin.dtype), ts.offsets.copy())


def stretch_pool(pool: TrailSet, catalog, dp, frac: float, rng, top: float = 0.2) -> TrailSet:
    y = np.asarray(dp.work["tdist"], np.float64)
    td, _ = W.trail_geometry(pool, catalog)
    lc = W.LC4[pool.lengths]
    cellp = lc * W.N_TD + W._bin(W.TD_EDGES, td)
    have = np.bincount(cellp, minlength=len(y)) / len(pool) * dp.n_hat
    excess = np.maximum(y - 2.0 * dp.sigma_w - have, 0.0)
    n_add = int(round(frac * len(pool)))
    if excess.sum() <= 0 or n_add <= 0:
        return pool
    cells = rng.choice(len(y), n_add, p=excess / excess.sum())
    src = np.empty(n_add, np.int64)
    ok = np.ones(n_add, bool)
    for c in np.unique(cells):
        m = np.flatnonzero(cells == c)
        cand = np.flatnonzero((lc == c // W.N_TD) & (td > 0.1))
        if len(cand) == 0:
            ok[m] = False
            continue
        cand = cand[td[cand] >= np.quantile(td[cand], 1.0 - top)]
        src[m] = cand[rng.integers(0, len(cand), len(m))]
    cells, src = cells[ok], src[ok]
    if len(src) == 0:
        return pool
    b = cells % W.N_TD
    lo, hi = W.TD_EDGES[b], np.minimum(W.TD_EDGES[b + 1], W.TD_EDGES[-2] * 1.25)
    f = (lo + rng.random(len(b)) * (hi - lo)) / td[src]
    kids = pool.subset(src)
    tr = kids.trail_of
    xy = catalog.xy[kids.poi]
    x0 = xy[kids.first][tr]
    new = catalog.clip_xy(x0 + f[tr][:, None] * (xy - x0))
    poi = kids.poi.astype(np.int64).copy()
    cats = np.asarray(catalog.cat)[poi]
    for c in np.unique(cats):
        m = np.flatnonzero(cats == c)
        j = catalog.nearest(new[m], cat=int(c))
        if j is not None:
            poi[m] = np.asarray(j).reshape(-1)
    kids = TrailSet(poi.astype(pool.poi.dtype), kids.tmin.copy(), kids.offsets.copy())
    off = np.concatenate([pool.offsets, pool.offsets[-1] + kids.offsets[1:]])
    return TrailSet(np.concatenate([pool.poi, kids.poi]), np.concatenate([pool.tmin, kids.tmin]), off)


def v3_enabled(cfg: PEConfig) -> bool:
    return (cfg.work_frac > 0 or cfg.calib != "none" or cfg.time_ground != "none" or cfg.release_mode != "vote"
            or cfg.release_pool > 0)


def run_pe(priv: TrailSet, users: np.ndarray, api, catalog, cfg: PEConfig,
           ecfg: EmbedConfig = EmbedConfig(), log=print):
    rng = np.random.default_rng(cfg.seed)
    private = bool(np.isfinite(cfg.eps))
    v3 = v3_enabled(cfg)
    fams_w = tuple(f for f in cfg.work_families.split(",") if f) if cfg.work_frac > 0 else ()
    sigma, sigma_m, sigma_w = noise_levels_w(cfg.eps, cfg.delta, cfg.T, cfg.use_marginals, cfg.marg_frac,
                                             cfg.work_frac, len(fams_w), cfg.venue2_frac)
    sigma_v2 = extra_sigma(cfg.eps, cfg.delta, cfg.venue2_frac)
    threshold = cfg.threshold_sigmas * sigma
    weights = vote_weights(users, cfg.unit)
    mu2 = (cfg.T / sigma ** 2 if sigma > 0 else 0.0) + (N_MARGINALS / sigma_m ** 2 if sigma_m > 0 else 0.0)
    mu2 += len(fams_w) / sigma_w ** 2 if sigma_w > 0 else 0.0
    mu2 += 1.0 / sigma_v2 ** 2 if sigma_v2 > 0 else 0.0
    mu_total = np.sqrt(mu2) if private else np.inf
    privacy = {"eps": cfg.eps, "delta": cfg.delta, "T": cfg.T, "unit": cfg.unit, "sigma_vote": sigma,
               "threshold": threshold, "use_marginals": cfg.use_marginals, "sigma_marg": sigma_m,
               "mu": float(mu_total)}
    if fams_w:
        privacy.update(sigma_work=sigma_w, work_families=list(fams_w), work_frac=cfg.work_frac)
    if sigma_v2 > 0:
        privacy.update(sigma_venue2=sigma_v2, venue2_frac=cfg.venue2_frac)
    if private and (cfg.T > 0 or cfg.use_marginals):
        privacy["eps_check"] = eps_for(float(mu_total), cfg.delta)
    if (cfg.sem_frac > 0 or cfg.sem_var > 0 or cfg.sem_refine > 0 or cfg.sem_refine_loop > 0) and not v3:
        raise ValueError("the semantic model (sem_frac / sem_var / sem_refine) needs the v3 DP workload")
    if cfg.sem_refine > 0 and cfg.release_pool <= 0:
        raise ValueError("sem_refine acts on the release pool (release_pool > 0)")
    if v3:
        if not cfg.use_marginals or cfg.T == 0:
            raise ValueError("the v3 steps need the DP marginals and T > 0")
        if cfg.cluster_c > 0:
            raise ValueError("the v3 steps reweight candidates; clustered voting is not supported")
        if cfg.time_ground != "none" and "gap" not in fams_w:
            raise ValueError("time grounding needs the 'gap' workload family")
    if cfg.venue2_frac > 0 and (not v3 or not cfg.use_marginals or not private):
        raise ValueError("venue2_frac needs the v3 steps, the DP marginals and a finite eps")
    if cfg.release_union_raw > 0 and (cfg.release_pool <= 0 or not hasattr(api, "raw_pool")):
        raise ValueError("release_union_raw needs the pool release and an API with raw_pool (PoolAPI)")
    if cfg.work_shrink not in ("none", "js"):
        raise ValueError(f"work_shrink {cfg.work_shrink}")
    if cfg.work_shrink == "js" and (not v3 or cfg.work_frac <= 0 or not hasattr(api, "raw_pool")):
        raise ValueError("work_shrink js needs the v3 workload and an API with raw_pool (PoolAPI)")
    if cfg.vote_cluster_kappa > 0 and (cfg.cluster_c > 0 or cfg.vote_k != 1):
        raise ValueError("vote_cluster_kappa excludes cluster_c and top-k voting")

    n_out = cfg.n_out or DEFAULT_N_OUT
    n_hat = float(DEFAULT_N_OUT)
    if cfg.use_marginals:
        marg = dp_marginals(priv, weights, catalog.cat, len(catalog.categories), sigma_m,
                            cfg.pop_threshold_sigmas, rng)
        marg_g, sigma_pop = marg, 0.0
        if sigma_v2 > 0:
            v2 = dp_venue_popularity(priv, weights, len(catalog), sigma_v2,
                                     np.random.default_rng([cfg.seed, VENUE2_STREAM]))
            marg_g, sigma_pop = combine_popularity(marg, v2, sigma_v2, cfg.pop_threshold_sigmas)
            privacy["venue2_nonzero_venues"] = int((marg_g.pop > 0).sum())
        popular = None
        if cfg.venue_aware:
            nz = np.flatnonzero(marg_g.pop > 0)
            popular = nz[np.argsort(-marg_g.pop[nz], kind="stable")[: cfg.max_popular]]
            ecfg = replace(ecfg, popular=popular)
        api.set_marginals(marg_g.venue_weights(cfg.pop_alpha_share, catalog.cat), marg.start_probs(),
                          marg.cat_share(), use=tuple(cfg.marg_use.split(",")),
                          popular=(popular, marg_g.pop[popular]) if popular is not None else None,
                          transitions=marg.transitions() if cfg.use_transitions else None)
        privacy["marg_cat_transitions"] = int((marg.cat_trans > 0).sum())
        privacy["marg_venue_transitions"] = int((marg.venue_trans > 0).sum()) if marg.venue_trans.size else 0
        privacy["marg_nonzero_venues"] = int((marg.pop > 0).sum())
        privacy["marg_sha256"] = marg.digest()
        n_hat = float(marg.start.sum())

    dp = None
    if v3:
        work = (W.dp_workload(priv, weights, catalog, sigma_w, np.random.default_rng([cfg.seed, WORK_STREAM]), fams_w)
                if fams_w else {})
        if fams_w:
            privacy["work_sha256"] = W.digest(work)
            privacy["work_stream"] = [cfg.seed, WORK_STREAM]
        dp = W.DPOutputs(work=work, sigma_w=sigma_w, marg=marg_g, sigma_m=sigma_m, popular=popular, n_hat=n_hat,
                         sigma_pop=sigma_pop, known_sigmas=cfg.known_sigmas)
        if cfg.sem_frac > 0 or cfg.sem_var > 0 or cfg.sem_refine > 0 or cfg.sem_refine_loop > 0:
            from .api_sim import SemanticModel
            if not hasattr(api, "set_semantic"):
                raise ValueError("the semantic model needs an API with set_semantic")
            sem = SemanticModel.from_dp(catalog, marg, dp.work, sigma_w, n_hat,
                                        marg_g.venue_weights(cfg.pop_alpha_share, catalog.cat), cfg.sem_k,
                                        cfg.sem_scale_km, cfg.known_sigmas)
            api.set_semantic(sem, cfg.sem_frac, cfg.sem_var)
    if cfg.pool_stretch > 0 and (dp is None or "tdist" not in dp.work or cfg.release_pool <= 0):
        raise ValueError("pool_stretch needs the 'tdist' workload family and the pool release")
    if cfg.hop_law == "dp":
        if dp is None or "hop" not in dp.work or not hasattr(api, "set_hop_law"):
            raise ValueError("hop_law dp needs the v3 'hop' workload family and an API with set_hop_law")
        api.set_hop_law(W.HOP_EDGES, dp.work["hop"], float(np.hypot(catalog.width_km, catalog.height_km)))
    calib_names = tuple(n for n in cfg.calib_families.split(",") if n)
    fuse_names = tuple(n for n in cfg.fuse_families.split(",") if n)
    dp_fit = dp

    def ground_time(ts):
        return W.time_ground(ts, dp.work["gap"], rng) if dp is not None and "gap" in dp.work else ts

    def calibrated(p, ts, final=True):
        if cfg.calib_solver == "fusion":
            q, info = FZ.fuse(W.families(ts, catalog, calib_names, dp_fit), dp.n_hat, p, cfg.fuse_gamma, cfg.fuse_iters)
            log({"calib_fusion": info})
            return q
        fams = [f for f in W.families(ts, catalog, calib_names, dp_fit) if f.calib]
        cap = cfg.calib_cap if final or cfg.calib_cap_early <= 0 else cfg.calib_cap_early
        pairs = [(f.Phi, f.t) for f in fams]
        if cfg.calib_solver == "parallel":
            q = W.calibrate_parallel(p, pairs, cfg.calib_iters, cfg.calib_eta, cap)
        else:
            q = W.calibrate(p, pairs, cfg.calib_iters, cfg.calib_eta, cap, cfg.calib_floor)
        log({"calib_family_jsd_before_after": {f.name: (round(W.family_error(f.Phi, f.t, p), 4),
                                                        round(W.family_error(f.Phi, f.t, q), 4)) for f in fams}})
        return q

    def fused(p, ts, gamma):
        w, info = FZ.fuse(W.families(ts, catalog, fuse_names, dp_fit), dp.n_hat, p, gamma, cfg.fuse_iters)
        log({"fusion": info})
        return w, {k: info[k] for k in ("iters", "ess_prior", "ess", "kl")}

    def refine_pool(pool):
        rho = None
        if cfg.sem_refine_rake > 0:
            prng = np.random.default_rng([cfg.seed, RAKE_STREAM])
            sub = pool.subset(np.sort(prng.choice(len(pool), max(1, len(pool) // 5), replace=False)))
            target = dp.marg.cat_share()
            rho = np.ones(len(target))
            for _ in range(cfg.sem_refine_rake):
                r = api.semantic_refine(sub, cfg.sem_refine, cfg.sem_refine_km, prng, cfg.sem_refine_mode, rho)
                got = np.bincount(catalog.cat[r.poi], weights=(1.0 / r.lengths)[r.trail_of], minlength=len(target))
                got = got / max(got.sum(), 1e-12)
                rho = np.clip(rho * (target + 1e-6) / (got + 1e-6), 0.25, 4.0)
            log({"sem_refine_rake": {"rounds": cfg.sem_refine_rake, "rho_min": float(rho.min()),
                                     "rho_max": float(rho.max())}})
        return api.semantic_refine(pool, cfg.sem_refine, cfg.sem_refine_km, rng, cfg.sem_refine_mode, rho)

    def sample(q, n):
        return FZ.systematic(q, n, rng) if cfg.release_sampler == "systematic" else rng.choice(len(q), n, p=q)

    if cfg.T == 0:
        t0 = time.time()
        S = api.random(n_out, rng)
        return S, [{"step": "random", "sec": round(time.time() - t0, 1)}], privacy, asdict(cfg)

    e_priv = embed(priv, catalog, ecfg)
    t0 = time.time()
    S = api.random(cfg.n_syn, rng)
    if cfg.work_shrink == "js":
        raw0 = api.raw_pool()
        if len(raw0):
            shr, cks = V5.shrink_work(dp.work, sorted(set(calib_names) | set(fuse_names)), raw0, catalog, dp.n_hat,
                                      dp.sigma_w, W.histogram)
            dp_fit = replace(dp, work=shr)
            log({"work_shrink_js": {k: round(v, 4) for k, v in cks.items()}})
        else:
            log({"work_shrink_js": "empty raw pool: not applied"})
    if cfg.time_ground == "all":
        S = ground_time(S)
    if cfg.sem_refine_loop > 0:
        S = api.semantic_refine(S, cfg.sem_refine_loop, cfg.sem_refine_km, rng, cfg.sem_refine_mode)
    history = [{"step": "random", "sec": round(time.time() - t0, 1)}]
    degrees = (cfg.degree_start * (cfg.degree_end / cfg.degree_start) ** (np.arange(cfg.T - 1) / max(cfg.T - 2, 1))
               if cfg.T > 1 else [])
    for it in range(cfg.T):
        t0 = time.time()
        last = it == cfg.T - 1
        e_syn = embed(S, catalog, ecfg)
        labels, n_bins = None, None
        if cfg.cluster_c > 0 and sigma > 0:
            k = int(np.clip(n_hat / (cfg.cluster_c * sigma), 10, len(S)))
            if k < len(S):
                labels, n_bins = kmeans(e_syn, k, seed=cfg.seed * 1000 + it), k
        if cfg.vote_cluster_kappa > 0 and sigma > 0:
            kc = V5.cluster_k(n_hat, sigma, cfg.vote_cluster_kappa, len(S))
            lab_c = kmeans(e_syn, kc, seed=cfg.seed * 1000 + it)
            counts, noisy_c, nn_dist = dp_nn_histogram(e_priv, e_syn, weights, sigma, threshold, rng, 1, lab_c, kc)
            noisy = V5.spread_cluster_votes(noisy_c, lab_c)
            v5_clu = {"clusters": kc, "clusters_nonzero": int((noisy_c > 0).sum()),
                      "cluster_size_max": int(np.bincount(lab_c, minlength=kc).max())}
        else:
            counts, noisy, nn_dist = dp_nn_histogram(e_priv, e_syn, weights, sigma, threshold, rng,
                                                     cfg.vote_k, labels, n_bins)
            v5_clu = None
        if noisy.sum() <= 0:
            log(f"[it {it}] all noisy counts below threshold; falling back to uniform")
            noisy = np.ones_like(noisy)
        probs = noisy / noisy.sum()
        rec_v3 = {}
        if last and cfg.save_final:
            d = Path(cfg.save_final)
            d.mkdir(parents=True, exist_ok=True)
            S.save(d / "final_pop.npz")
            np.save(d / "final_votes.npy", noisy)
            if dp is not None and dp.work:
                np.savez_compressed(d / "dp_workload.npz", **dp.work)
        if last and cfg.release_mode == "fusion":
            probs, rec_v3["fusion"] = fused(FZ.vote_prior(noisy, cfg.fuse_rho), S, cfg.fuse_gamma)
        elif dp is not None and (cfg.calib == "all" or (cfg.calib == "final" and last)):
            p_vote = probs
            probs = calibrated(probs, S, final=last)
            rec_v3 = {"calib_ess": float(1.0 / (probs ** 2).sum()), "calib_kl": float(
                np.sum(probs[probs > 0] * np.log(probs[probs > 0] / p_vote[probs > 0])))}
        rec = {"step": it, "bins": len(probs), "nonzero": int((noisy > 0).sum()),
               "eff_samples": float(1.0 / (probs ** 2).sum())}
        if not private:
            rec.update(max_count=float(counts.max()), mean_nn_dist=float(nn_dist.mean()))
        rec.update(rec_v3)
        if v5_clu is not None:
            rec.update(v5_clu)

        def draw(n):
            return (sample_from_clusters(labels, n_bins, probs, n, rng) if labels is not None
                    else rng.choice(len(S), n, p=probs))
        if it < cfg.T - 1:
            S = api.variation(S.subset(draw(cfg.n_syn)), float(degrees[it]), rng)
            if cfg.time_ground == "all":
                S = ground_time(S)
            if cfg.sem_refine_loop > 0:
                S = api.semantic_refine(S, cfg.sem_refine_loop, cfg.sem_refine_km, rng, cfg.sem_refine_mode)
            rec["degree"] = float(degrees[it])
        elif cfg.release_pool > 0:
            pool = api.variation(S.subset(sample(probs, cfg.release_pool * n_out)), cfg.release_degree, rng)
            if cfg.time_ground in ("release", "all"):
                pool = ground_time(pool)
            if cfg.pool_time_jitter > 0:
                pool = jitter_times(pool, cfg.pool_time_jitter, rng)
            if cfg.pool_stretch > 0:
                pool = stretch_pool(pool, catalog, dp, cfg.pool_stretch, np.random.default_rng([cfg.seed, STRETCH_STREAM]))
            if cfg.sem_refine > 0 and cfg.sem_refine_stage == "pre_div":
                pool = refine_pool(pool)
            if cfg.pool_diversify > 0 and hasattr(api, "diversify"):
                pool = api.diversify(pool, cfg.pool_diversify, cfg.diversify_radius_km, rng)
            if cfg.sem_refine > 0 and cfg.sem_refine_stage == "post_div":
                pool = refine_pool(pool)
            if cfg.release_union_raw > 0:
                raw = api.raw_pool()
                if len(raw):
                    if cfg.time_ground in ("release", "all"):
                        raw = W.time_ground(raw, dp.work["gap"], np.random.default_rng([cfg.seed, V5.UNION_STREAM]))
                    rec["union_children"], rec["union_raw"] = len(pool), len(raw)
                    pool = V5.concat(pool, raw)
            u = np.full(len(pool), 1.0 / len(pool))
            if cfg.pool_solver == "fusion":
                qp, rec["pool_fusion"] = fused(u, pool, cfg.fuse_gamma_refit)
            else:
                qp = calibrated(u, pool)
            rec["pool_ess"] = float(1.0 / (qp ** 2).sum())
            S = pool.subset(sample(qp, n_out))
            if cfg.release_diversify > 0 and hasattr(api, "diversify"):
                S = api.diversify(S, cfg.release_diversify, cfg.diversify_radius_km, rng)
        else:
            S = S.subset(sample(probs, n_out) if v3 else draw(n_out))
            if cfg.out_degree > 0:
                S = api.variation(S, cfg.out_degree, rng)
            if cfg.time_ground in ("release", "all"):
                S = ground_time(S)
            if cfg.release_diversify > 0 and hasattr(api, "diversify"):
                S = api.diversify(S, cfg.release_diversify, cfg.diversify_radius_km, rng)
        rec["sec"] = round(time.time() - t0, 1)
        history.append(rec)
        log(rec)
    return S, history, privacy, asdict(cfg)
