from __future__ import annotations

import math

import numpy as np

N_OUT = 50_000
YU_K = 1000
YU_KMEANS_ITERS = 25
PATH_K = 10
GROUND_STREAM = 210011
AB0_STREAM = 210013
AB1_STREAM = 210017
AB2_STREAM = 210019
DIV_STREAM = 7919
ABLATIONS = ("ab0", "ab1", "ab2", "ab3")


def kmeans_centroids(x: np.ndarray, k: int, iters: int = YU_KMEANS_ITERS, seed: int = 0):
    import torch
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    k = min(k, len(x))
    cent = xt[torch.randperm(len(x), generator=g)[:k]].clone()
    xn = (xt * xt).sum(1, keepdim=True)
    lab = None
    for _ in range(max(iters, 1)):
        lab = _argmin_dist(xt, xn, cent)
        sums = torch.zeros_like(cent).index_add_(0, lab, xt)
        cnt = torch.bincount(lab, minlength=k).float()[:, None]
        cent = torch.where(cnt > 0, sums / cnt.clamp(min=1), cent)
    lab = _argmin_dist(xt, xn, cent)
    return cent.numpy(), lab.numpy()


def _argmin_dist(xt, xn, cent, chunk: int = 16384):
    import torch
    cn = (cent * cent).sum(1)[None, :]
    out = [(xn[s:s + chunk] - 2.0 * xt[s:s + chunk] @ cent.T + cn).argmin(1) for s in range(0, len(xt), chunk)]
    return torch.cat(out)


def nearest_centroid(e: np.ndarray, cent: np.ndarray) -> np.ndarray:
    import torch
    xt = torch.from_numpy(np.ascontiguousarray(e, dtype=np.float32))
    return _argmin_dist(xt, (xt * xt).sum(1, keepdim=True), torch.from_numpy(cent)).numpy()


def apportion(w: np.ndarray, n: int) -> np.ndarray:
    w = np.maximum(np.asarray(w, np.float64), 0.0)
    if w.sum() <= 0:
        w = np.ones_like(w)
    x = w / w.sum() * n
    c = np.floor(x).astype(np.int64)
    r = n - int(c.sum())
    if r > 0:
        c[np.argsort(-(x - c), kind="stable")[:r]] += 1
    return c


def yu_resample(labels: np.ndarray, counts: np.ndarray, rng):
    order = np.argsort(labels, kind="stable")
    sizes = np.bincount(labels, minlength=len(counts))
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    out, short, extra = [], 0, 0
    for i in np.flatnonzero(counts > 0):
        mem = order[starts[i]:starts[i] + sizes[i]]
        c = int(counts[i])
        if sizes[i] == 0:
            raise ValueError("a cluster with positive count has no member")
        if c <= sizes[i]:
            out.append(rng.choice(mem, c, replace=False))
        else:
            short += 1
            extra += c - sizes[i]
            out.append(np.concatenate([mem, rng.choice(mem, c - sizes[i], replace=True)]))
    idx = np.concatenate(out) if out else np.zeros(0, np.int64)
    return rng.permutation(idx), short, extra


def yu_counts(noisy_hist: np.ndarray, sizes: np.ndarray, n_out: int) -> np.ndarray:
    h = np.where(np.asarray(sizes) > 0, np.maximum(noisy_hist, 0.0), 0.0)
    return apportion(h, n_out)


def path_votes(e_priv: np.ndarray, e_syn: np.ndarray, k: int = PATH_K, weights=None) -> np.ndarray:
    from trajpe.embed import nearest_k
    idx = nearest_k(np.ascontiguousarray(e_priv, np.float32), np.ascontiguousarray(e_syn, np.float32), k)
    w = np.ones(len(e_priv)) if weights is None else np.asarray(weights, np.float64)
    return np.bincount(idx.ravel(), weights=np.repeat(w / np.sqrt(k), k), minlength=len(e_syn))


def top_n(noisy: np.ndarray, n: int, rng) -> np.ndarray:
    perm = rng.permutation(len(noisy))
    keep = perm[np.argsort(-noisy[perm], kind="stable")[:n]]
    return rng.permutation(keep)


def ab3_sigmas(mu_pe: float, marg_frac: float, work_frac: float, n_marg: int, n_work: int):
    a = marg_frac / (marg_frac + work_frac)
    m2 = mu_pe ** 2
    return float(np.sqrt(n_marg / (a * m2))), float(np.sqrt(n_work / ((1.0 - a) * m2)))


def mu_of(mechs) -> float:
    return float(math.sqrt(sum(c / s ** 2 for s, c in mechs))) if mechs else 0.0


def accounting(cal: dict, tj: dict, mechs, hy) -> dict:
    sigma, q, steps = float(tj["noise_multiplier"]), float(tj["sample_rate"]), int(tj["steps"])
    if (sigma, q, steps) != (float(cal["sigma_sgd"]), float(cal["sample_rate"]), int(cal["steps"])):
        raise SystemExit("train.json (sigma, q, T) does not match calib.json")
    if int(tj.get("steps_run", steps)) != steps or tj.get("complete") is False or \
            (tj.get("private_stats") or {}).get("oom_retries", 0):
        raise SystemExit("training release gate (steps_run / complete / oom_retries)")
    delta, eps_total = float(cal["delta"]), float(cal["eps_total"])
    mu = mu_of(mechs)
    if mu > float(cal["mu_pe"]) * (1 + 1e-12):
        raise SystemExit(f"ablation spends mu {mu} > mu_PE {cal['mu_pe']}")
    e = hy.composed_eps(sigma, q, steps, mu, delta)
    e_sgd = hy.composed_eps(sigma, q, steps, 0.0, delta)
    if e > eps_total:
        raise SystemExit(f"composed PLD epsilon {e} exceeds {eps_total}")
    ind = independent_check(sigma, q, steps, mechs, delta)
    if ind["eps_pld_per_mechanism"] > eps_total:
        raise SystemExit(f"independent PLD epsilon {ind['eps_pld_per_mechanism']} exceeds {eps_total}")
    return {"eps_budget": eps_total, "delta": delta, "eps_composed_pld": e, "eps_sgd_only_pld": e_sgd,
            "eps_release": e, "mu_pe_spent": mu, "mu_pe_budget": float(cal["mu_pe"]),
            "pe_mechanisms": [{"sigma": float(s), "count": int(c), "l2_sensitivity": 1.0} for s, c in mechs],
            "dp_sgd": {"noise_multiplier": sigma, "sample_rate": q, "steps": steps, "clip": tj.get("clip"),
                       "model": tj.get("model"), "eps_prv_composed_train": tj.get("eps_prv")},
            "independent_check": ind,
            "accountant": "dp_accounting PLD, add-or-remove-one, value_discretization_interval=1e-4 "
                          "(baselines/dplora/hybrid.composed_eps; PE part as one Gaussian of noise 1/mu)",
            "guarantee": "(eps_release, delta)-DP, trail level, add/remove-one; eps_release <= eps_budget. The "
                         "generator's texts, grounding, clustering, selection and release are post-processing."}


def independent_check(sigma: float, q: float, steps: int, mechs, delta: float) -> dict:
    import dp_accounting
    from dp_accounting.pld import pld_privacy_accountant
    from dp_accounting.rdp import rdp_privacy_accountant
    pld = pld_privacy_accountant.PLDAccountant(
        neighboring_relation=dp_accounting.NeighboringRelation.ADD_OR_REMOVE_ONE, value_discretization_interval=1e-4)
    rdp = rdp_privacy_accountant.RdpAccountant(
        neighboring_relation=dp_accounting.NeighboringRelation.ADD_OR_REMOVE_ONE)
    ev = dp_accounting.PoissonSampledDpEvent(q, dp_accounting.GaussianDpEvent(sigma))
    pld.compose(ev, steps)
    rdp.compose(ev, steps)
    for s, c in mechs:
        pld.compose(dp_accounting.GaussianDpEvent(float(s)), int(c))
        rdp.compose(dp_accounting.GaussianDpEvent(float(s)), int(c))
    return {"eps_pld_per_mechanism": float(pld.get_epsilon(delta)), "eps_rdp": float(rdp.get_epsilon(delta)),
            "route": "dp_accounting PLD with each Gaussian composed separately + RDP (reported only)"}


def pool_from_texts(catalog, recs, ground_k: int, ground_scale_km: float, pop_radius_km: float, seed: int,
                    marginals=None, api=None):
    from trajpe.api_pool import PoolAPI
    if api is None:
        api = PoolAPI(catalog, recs, None, None, pool_frac=1.0, refresh=0.0, ground_k=ground_k,
                      ground_scale_km=ground_scale_km, pop_radius_km=pop_radius_km)
    return api.random(len(recs), np.random.default_rng([seed, GROUND_STREAM])), api


def run_ab012(priv, catalog, pool, which, mu_pe: float, seed: int, n_out: int = N_OUT, log=print) -> dict:
    from trajpe.embed import EmbedConfig, embed
    out = {}
    if "ab0" in which:
        r = np.random.default_rng([seed, AB0_STREAM])
        idx = r.choice(len(pool), n_out, replace=len(pool) < n_out)
        out["ab0"] = (pool.subset(idx), [], {"sampled": "uniform without replacement from the grounded pool"})
    if not ({"ab1", "ab2"} & set(which)):
        return out
    sigma = 1.0 / mu_pe
    ecfg = EmbedConfig()
    e_syn = embed(pool, catalog, ecfg).astype(np.float32)
    e_priv = embed(priv, catalog, ecfg).astype(np.float32)
    log({"embed_dim": int(e_syn.shape[1]), "n_candidates": len(pool)})
    if "ab1" in which:
        r = np.random.default_rng([seed, AB1_STREAM])
        cent, lab = kmeans_centroids(e_syn, YU_K, YU_KMEANS_ITERS, seed)
        nn = nearest_centroid(e_priv, cent)
        h = np.bincount(nn, minlength=len(cent)).astype(np.float64)
        noisy = h + r.normal(0.0, sigma, len(h))
        del h, nn
        sizes = np.bincount(lab, minlength=len(cent))
        counts = yu_counts(noisy, sizes, n_out)
        idx, short, extra = yu_resample(lab, counts, r)
        out["ab1"] = (pool.subset(idx), [(sigma, 1)],
                      {"K": int(len(cent)), "kmeans_iters": YU_KMEANS_ITERS, "sigma_hist": sigma,
                       "noisy_positive_clusters": int((noisy > 0).sum()), "clusters_drawn": int((counts > 0).sum()),
                       "short_clusters": int(short), "drawn_with_replacement": int(extra),
                       "distinct_released": int(len(np.unique(idx)))})
        log({"ab1": out["ab1"][2]})
    if "ab2" in which:
        r = np.random.default_rng([seed, AB2_STREAM])
        v = path_votes(e_priv, e_syn, PATH_K)
        noisy = v + r.normal(0.0, sigma, len(v))
        del v
        idx = top_n(noisy, n_out, r)
        out["ab2"] = (pool.subset(idx), [(sigma, 1)],
                      {"k": PATH_K, "vote_weight": 1.0 / math.sqrt(PATH_K), "sigma": sigma,
                       "noisy_threshold_kept": float(np.min(noisy[idx])), "distinct_released": int(len(idx))})
        log({"ab2": out["ab2"][2]})
    return out


def run_ab3(priv, users, catalog, recs, cfg: dict, mu_pe: float, seed: int, n_out: int = N_OUT, log=print):
    from trajpe import fusion as FZ
    from trajpe import workload as W
    from trajpe.api_pool import PoolAPI
    from trajpe.api_sim import RuleVariation, SimAPI
    from trajpe.marginals import N_MARGINALS, dp_marginals
    from trajpe.mechanisms import vote_weights
    from trajpe.pe import WORK_STREAM
    if cfg.get("venue2_frac", 0) or cfg.get("unit", "trail") != "trail":
        raise ValueError("AB3 mirrors v4: no venue2 release, trail level")
    fams_w = tuple(f for f in cfg["work_families"].split(",") if f)
    sigma_m, sigma_w = ab3_sigmas(mu_pe, cfg["marg_frac"], cfg["work_frac"], N_MARGINALS, len(fams_w))
    rng = np.random.default_rng(seed)
    weights = vote_weights(users, "trail")
    marg = dp_marginals(priv, weights, catalog.cat, len(catalog.categories), sigma_m, None, rng)
    nz = np.flatnonzero(marg.pop > 0)
    popular = nz[np.argsort(-marg.pop[nz], kind="stable")[: int(cfg.get("max_popular", 2000))]]
    sim = SimAPI(catalog)
    api = PoolAPI(catalog, recs, sim.random, sim.variation, pool_frac=1.0, refresh=0.0, ground_k=cfg["ground_k"],
                  ground_scale_km=cfg["ground_scale_km"], pop_radius_km=cfg["pop_radius_km"])
    vw = marg.venue_weights(cfg["pop_alpha_share"], catalog.cat)
    api.set_marginals(vw, marg.start_probs(), marg.cat_share(), use=tuple(cfg.get("marg_use", "prompt,ground,var").split(",")),
                      popular=(popular, marg.pop[popular]), transitions=marg.transitions())
    work = W.dp_workload(priv, weights, catalog, sigma_w, np.random.default_rng([seed, WORK_STREAM]), fams_w)
    dp = W.DPOutputs(work=work, sigma_w=sigma_w, marg=marg, sigma_m=sigma_m, popular=popular,
                     n_hat=float(marg.start.sum()), sigma_pop=0.0, known_sigmas=float(cfg.get("known_sigmas", W.KNOWN_SIGMAS)))
    pool = api.random(int(cfg["pool_n"]), rng)
    if cfg["time_ground"] in ("release", "all"):
        pool = W.time_ground(pool, dp.work["gap"], rng)
    fuse_names = tuple(n for n in cfg["fuse_families"].split(",") if n)
    u = np.full(len(pool), 1.0 / len(pool))
    qp, info = FZ.fuse(W.families(pool, catalog, fuse_names, dp), dp.n_hat, u, cfg["fuse_gamma_refit"], cfg["fuse_iters"])
    if cfg.get("release_sampler") != "systematic":
        raise ValueError("v4 releases with the systematic sampler")
    S = pool.subset(FZ.systematic(qp, n_out, rng))
    pp = cfg["_hybrid"]["release_postprocess"]
    var = RuleVariation(catalog)
    var.set_venue_weights(vw if "var" in cfg.get("marg_use", "prompt,ground,var").split(",") else None)
    S = var.diversify(S, float(pp["p"]), float(pp["radius_km"]), np.random.default_rng([seed, DIV_STREAM]))
    mechs = [(sigma_m, N_MARGINALS), (sigma_w, len(fams_w))]
    inf = {"sigma_marg": sigma_m, "sigma_work": sigma_w, "work_families": list(fams_w),
           "share_marg": cfg["marg_frac"] / (cfg["marg_frac"] + cfg["work_frac"]),
           "fusion": {k: info[k] for k in ("iters", "ess_prior", "ess", "kl") if k in info},
           "pool_ess": float(1.0 / (qp ** 2).sum()), "ground_underflow_fallbacks": api.n_underflow,
           "work_sha256": W.digest(work), "marg_sha256": marg.digest()}
    log({"ab3": inf})
    return S, mechs, inf
