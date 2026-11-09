import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from trajpe import workload as W
from trajpe.api_sim import SimAPI, VariationConfig
from test_v31 import WF9, _SimCat, _trails

NEW = ("tdist", "diam", "od6", "ctrans6", "tcell6")
WF13 = WF9 + ",tdist,diam,od6,ctrans6,tcell6"


def test_geometry_is_exact():
    rng = np.random.default_rng(11)
    cat = _SimCat(rng)
    D = _trails(rng, cat, 150)
    td, diam = W.trail_geometry(D, cat)
    for i in range(len(D)):
        p = cat.xy[D.poi[D.offsets[i]:D.offsets[i + 1]]]
        assert abs(td[i] - np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) < 1e-9
        assert abs(diam[i] - np.sqrt(((p[:, None] - p[None]) ** 2).sum(-1)).max()) < 1e-9
    r, c, v = W.rows(D, cat, "diam")
    L2 = D.lengths == 2
    assert np.all(c[L2] == 0) and np.all(c[~L2] >= 1)


def test_new_families_rows_and_sensitivity():
    rng = np.random.default_rng(12)
    cat = _SimCat(rng)
    D = _trails(rng, cat, 120)
    users = rng.integers(0, 15, len(D))
    _, inv, cnt = np.unique(users, return_inverse=True, return_counts=True)
    for f in NEW:
        assert f in W.FAMILIES and f not in W.DEFAULT_FAMILIES.split(",")
        r, c, v = W.rows(D, cat, f)
        assert np.all(v >= 0) and c.min() >= 0 and c.max() < W.n_bins(f, cat)
        assert np.allclose(np.bincount(r, weights=v, minlength=len(D)), 1.0)
        for w, units in ((np.ones(len(D)), [[i] for i in range(len(D))]),
                         (1.0 / cnt[inv], [np.flatnonzero(users == u) for u in np.unique(users)])):
            full = W.histogram(D, w, cat, f)
            for drop in units:
                keep = np.setdiff1d(np.arange(len(D)), drop)
                d = full - W.histogram(D.subset(keep), w[keep], cat, f)
                assert np.all(d >= -1e-12) and abs(d.sum() - 1.0) < 1e-9 and np.linalg.norm(d) <= 1 + 1e-9


def test_budget_scaling_keeps_sigma_w():
    from trajpe.dp import eps_for
    from trajpe.mechanisms import noise_levels_w
    for eps in (1.0, 4.0):
        base = noise_levels_w(eps, 1e-6, 3, True, 0.4, 0.07875, 9)
        for k, frac in ((11, 0.09625), (12, 0.105), (13, 0.11375)):
            s, sm, sw = noise_levels_w(eps, 1e-6, 3, True, 0.4, frac, k)
            assert abs(sw - base[2]) < 1e-9 and sm == base[1] and s > base[0]
            assert abs(eps_for(np.sqrt(3 / s ** 2 + 5 / sm ** 2 + k / sw ** 2), 1e-6) - eps) < 1e-6


def test_hop_law_and_jitter():
    from trajpe.pe import jitter_times
    rng = np.random.default_rng(14)
    cat = _SimCat(rng)
    api = SimAPI(cat, VariationConfig())
    probs = np.zeros(len(W.HOP_EDGES) - 1)
    probs[[3, 9]] = [1.0, 3.0]
    api.set_hop_law(W.HOP_EDGES, probs, 28.3)
    d = api.random._hop(20000, rng)
    b = np.searchsorted(W.HOP_EDGES, d, side="right") - 1
    assert set(np.unique(b)) == {3, 9} and abs(np.mean(b == 9) - 0.75) < 0.02
    probs[:] = 0
    probs[-1] = 1.0
    api.set_hop_law(W.HOP_EDGES, probs, 28.3)
    d = api.random._hop(1000, rng)
    assert d.min() >= 20 and d.max() <= 28.3
    D = _trails(rng, cat, 300)
    J = jitter_times(D, 30, rng)
    assert np.array_equal(J.poi, D.poi) and np.array_equal(J.offsets, D.offsets)
    t0 = J.tmin[J.offsets[:-1]]
    assert t0.min() >= 0 and t0.max() < 10080
    gd, gj = np.diff(D.tmin.astype(np.int64)), np.diff(J.tmin.astype(np.int64))
    inner = np.ones(len(gd), bool)
    inner[D.offsets[1:-1] - 1] = False
    assert np.array_equal(gd[inner], gj[inner])


def test_stretch_pool_targets_dp_excess():
    from trajpe.pe import stretch_pool
    rng = np.random.default_rng(15)
    cat = _SimCat(rng)
    pool = _trails(rng, cat, 400)
    td, _ = W.trail_geometry(pool, cat)
    y = np.zeros(W.n_bins("tdist", cat))
    tgt = 1 * W.N_TD + int(np.searchsorted(W.TD_EDGES, 40.0, side="right") - 1)
    y[tgt] = 200.0
    dp = W.DPOutputs(work={"tdist": y}, sigma_w=1.0, n_hat=400.0)
    out = stretch_pool(pool, cat, dp, 0.1, np.random.default_rng(0))
    assert len(out) == 440 and np.array_equal(out.poi[: len(pool.poi)], pool.poi)
    kids = out.subset(np.arange(400, 440))
    assert np.all(kids.lengths == 3)
    tk, _ = W.trail_geometry(kids, cat)
    assert np.median(tk) > np.quantile(td[pool.lengths == 3], 0.9)
    dp0 = W.DPOutputs(work={"tdist": np.zeros_like(y)}, sigma_w=1.0, n_hat=400.0)
    assert stretch_pool(pool, cat, dp0, 0.1, np.random.default_rng(0)) is pool


def test_v32_run_reads_only_dp_outputs():
    from trajpe import pe as PE
    from test_workload import _ALLOW, _Guarded
    rng = np.random.default_rng(13)
    cat = _SimCat(rng)
    D1, D2 = _trails(rng, cat, 200), _trails(rng, cat, 150)
    u1, u2 = np.arange(200).astype(str), (np.arange(150) // 3).astype(str)
    fams = WF13 + ",start,cat,venue,cat_trans,venue_trans"
    cfg = PE.PEConfig(eps=4.0, T=3, n_syn=80, seed=4, marg_frac=0.3, calib_iters=20, fuse_iters=40, n_out=250,
                      work_frac=0.13, work_families=WF13, calib="all", time_ground="all", release_mode="fusion",
                      fuse_rho=0.2, release_pool=2, pool_solver="fusion", pool_diversify=1.0,
                      release_sampler="systematic", fuse_families=fams, calib_families=fams, sem_refine=0.5,
                      sem_refine_mode="gibbs", sem_refine_stage="pre_div", hop_law="dp", pool_time_jitter=20, pool_stretch=0.05)
    orig = {"dp_marginals": PE.dp_marginals, "dp_nn_histogram": PE.dp_nn_histogram, "dp_workload": W.dp_workload,
            "embed": PE.embed}

    def allowed(fn):
        def wrapped(*a, **k):
            _ALLOW[0] += 1
            try:
                return fn(*a, **k)
            finally:
                _ALLOW[0] -= 1
        return wrapped

    try:
        priv = _Guarded(D1.poi, D1.tmin, D1.offsets)
        PE.dp_marginals, W.dp_workload = allowed(orig["dp_marginals"]), allowed(orig["dp_workload"])
        PE.embed = lambda ts, *a, **k: (allowed(orig["embed"]) if isinstance(ts, _Guarded) else orig["embed"])(ts, *a, **k)
        S, hist, privacy, _ = PE.run_pe(priv, u1, SimAPI(cat, VariationConfig()), cat, cfg, log=lambda *_: None)
        assert len(S) == 250 and abs(privacy["eps_check"] - 4.0) < 1e-6
        assert set(NEW) <= set(privacy["work_families"])
    finally:
        PE.dp_marginals, W.dp_workload, PE.embed = orig["dp_marginals"], orig["dp_workload"], orig["embed"]

    e1 = {}

    def embed_rec(ts, catalog, ecfg):
        out = orig["embed"](ts, catalog, ecfg)
        if ts is D1:
            e1["e"] = out
        return out
    try:
        outs = []
        for D, u in ((D1, u1), (D2, u2)):
            PE.embed = embed_rec
            if D is D2:
                PE.dp_marginals = lambda priv, w, *a: orig["dp_marginals"](D1, np.ones(len(D1)), *a)
                PE.dp_nn_histogram = lambda ep, es, w, *a: orig["dp_nn_histogram"](e1["e"], es, np.ones(len(D1)), *a)
                W.dp_workload = lambda priv, w, *a: orig["dp_workload"](D1, np.ones(len(D1)), *a)
            S, _, _, _ = PE.run_pe(D, u, SimAPI(cat, VariationConfig()), cat, cfg, log=lambda *_: None)
            outs.append(S)
        assert all(np.array_equal(getattr(outs[0], f), getattr(outs[1], f)) for f in ("poi", "tmin", "offsets"))
    finally:
        PE.dp_marginals, PE.dp_nn_histogram, PE.embed = orig["dp_marginals"], orig["dp_nn_histogram"], orig["embed"]
        W.dp_workload = orig["dp_workload"]


if __name__ == "__main__":
    test_geometry_is_exact()
    test_new_families_rows_and_sensitivity()
    test_budget_scaling_keeps_sigma_w()
    test_hop_law_and_jitter()
    test_stretch_pool_targets_dp_excess()
    test_v32_run_reads_only_dp_outputs()
    print("all v3.2 tests passed")
