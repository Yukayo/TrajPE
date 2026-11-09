import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines" / "dplora"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import hybrid as HY
from test_hybrid import _GeoCat, _pool_api, _records
from test_v31 import WF9, _trails
from trajpe import v5 as V5
from trajpe import workload as W
from trajpe.dp import eps_for, mu_for


def test_js_shrink():
    y, t = np.array([10.0, 0.0, 5.0, -3.0]), np.array([8.0, 1.0, 4.0, 0.0])
    d = y - t
    c_hand = max(0.0, 1 - 2 * 1.5 ** 2 / float(d @ d))
    yh, c = V5.js_shrink(y, t, 1.5)
    assert abs(c - c_hand) < 1e-12 and np.allclose(yh, t + c_hand * d)
    yh, c = V5.js_shrink(y, t, 100.0)
    assert c == 0.0 and np.allclose(yh, t)
    yh, c = V5.js_shrink(y[:2], t[:2], 1.0)
    assert c == 1.0 and np.allclose(yh, y[:2])
    rng = np.random.default_rng(0)
    truth = rng.uniform(0, 20, 200)
    target = truth + rng.normal(0, 2, 200)
    e_raw = e_js = 0.0
    for _ in range(200):
        yy = truth + rng.normal(0, 4, 200)
        e_raw += float(((yy - truth) ** 2).sum())
        e_js += float(((V5.js_shrink(yy, target, 4.0)[0] - truth) ** 2).sum())
    assert e_js < e_raw


def test_cluster_rule_and_spread():
    assert V5.cluster_k(74685.0, 14.68, 3.0, 100000) == int(74685 // (3 * 14.68))
    assert V5.cluster_k(74685.0, 4.15, 3.0, 40000) == 40000 // 8
    assert V5.cluster_k(100.0, 14.68, 3.0, 100000) == 64
    lab = np.array([0, 0, 1, 2, 2, 2])
    noisy = np.array([4.0, 0.0, 9.0])
    p = V5.spread_cluster_votes(noisy, lab)
    assert np.allclose(p, [2, 2, 0, 3, 3, 3]) and abs(p.sum() - noisy.sum()) < 1e-12


def test_concat_and_refresh_share():
    rng = np.random.default_rng(5)
    cat = _GeoCat(rng)
    recs = _records(rng, cat, 400)
    vw = rng.uniform(0.1, 1.0, len(cat))
    a = _trails(rng, cat, 30)
    b = _trails(rng, cat, 20)
    c = V5.concat(a, b)
    assert len(c) == 50 and np.array_equal(c.subset(np.arange(30, 50)).poi, b.poi)
    assert np.array_equal(c.subset(np.arange(30)).tmin, a.tmin)
    for degree in (1.0, 0.05):
        api = _pool_api(cat, recs, refresh_share=0.5)
        api.set_marginals(vw, np.full(168, 1 / 168))
        S = api.random(100, np.random.default_rng(1))
        n0 = api.n_used
        out = api.variation(S, degree, np.random.default_rng(2))
        assert len(out) == 100 and 30 <= api.n_used - n0 <= 70
        assert api.usage()["refresh_share"] == 0.5
    api0, api1 = _pool_api(cat, recs), _pool_api(cat, recs, refresh_share=0.0)
    for x in (api0, api1):
        x.set_marginals(vw, np.full(168, 1 / 168))
    o = [x.variation(x.random(60, np.random.default_rng(1)), 1.0, np.random.default_rng(2)) for x in (api0, api1)]
    assert np.array_equal(o[0].poi, o[1].poi) and np.array_equal(o[0].tmin, o[1].tmin)
    assert "refresh_share" not in api1.usage() and len(api1.raw_pool()) == 400


def _cfg(PE, eps_pe, **kw):
    return PE.PEConfig(eps=eps_pe, T=3, n_syn=80, seed=3, marg_frac=0.4, calib_iters=20, fuse_iters=40, n_out=250,
                       work_frac=0.07875, work_families=WF9, calib="all", calib_solver="parallel", time_ground="all",
                       release_mode="fusion", fuse_rho=0.2, release_pool=2, pool_solver="fusion", pool_diversify=1.0,
                       release_sampler="systematic", fuse_families=WF9 + ",start,cat,venue,cat_trans,venue_trans",
                       calib_families=WF9 + ",start,cat,venue,cat_trans,venue_trans", sem_refine=1.0,
                       sem_refine_mode="gibbs", sem_refine_stage="pre_div", sem_k=4, **kw)


def test_v5_run_reads_only_dp_outputs_and_defaults_are_v4():
    from trajpe import pe as PE
    from test_workload import _ALLOW, _Guarded
    rng = np.random.default_rng(2)
    cat = _GeoCat(rng)
    D1 = _trails(rng, cat, 200)
    u1 = np.arange(200).astype(str)
    recs = _records(rng, cat, 300)
    mu_pe = math.sqrt(0.5) * mu_for(4.0, 1e-6)
    eps_pe = eps_for(mu_pe * (1 - HY.MU_SAFETY), 1e-6)
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
    logs = []
    try:
        priv = _Guarded(D1.poi, D1.tmin, D1.offsets)
        PE.dp_marginals, W.dp_workload = allowed(orig["dp_marginals"]), allowed(orig["dp_workload"])
        PE.dp_nn_histogram = allowed(orig["dp_nn_histogram"])
        PE.embed = lambda ts, *a, **k: (allowed(orig["embed"]) if isinstance(ts, _Guarded) else orig["embed"])(ts, *a, **k)
        api = _pool_api(cat, recs, refresh_share=0.5)
        cfg = _cfg(PE, eps_pe, release_union_raw=1, vote_cluster_kappa=3.0, work_shrink="js")
        S, hist, privacy, _ = PE.run_pe(priv, u1, api, cat, cfg, log=logs.append)
        assert len(S) == 250 and abs(privacy["mu"] - mu_pe) < 1e-6 and privacy["mu"] <= mu_pe
        assert any(isinstance(x, dict) and "work_shrink_js" in x for x in logs)
        assert all("clusters" in h for h in hist[1:]) and hist[-1]["union_raw"] == 300
    finally:
        PE.dp_marginals, PE.dp_nn_histogram, PE.embed = orig["dp_marginals"], orig["dp_nn_histogram"], orig["embed"]
        W.dp_workload = orig["dp_workload"]
    outs = []
    for kw, akw in (({}, {}), ({"release_union_raw": 0, "vote_cluster_kappa": 0.0, "work_shrink": "none"},
                              {"refresh_share": 0.0})):
        S, _, _, _ = PE.run_pe(D1, u1, _pool_api(cat, recs, refresh=0.5, **akw), cat, _cfg(PE, eps_pe, **kw),
                               log=lambda *_: None)
        outs.append(S)
    assert all(np.array_equal(getattr(outs[0], f), getattr(outs[1], f)) for f in ("poi", "tmin", "offsets"))


def test_v5_flags_on_generator_failed_fallback():
    from trajpe import pe as PE
    rng = np.random.default_rng(6)
    cat = _GeoCat(rng)
    D1 = _trails(rng, cat, 120)
    api = _pool_api(cat, [], pool_frac=0.0, refresh_share=0.0)
    S, hist, privacy, _ = PE.run_pe(D1, np.arange(120).astype(str), api, cat,
                                    _cfg(PE, 2.0, release_union_raw=1, vote_cluster_kappa=3.0), log=lambda *_: None)
    assert len(S) == 250 and "union_raw" not in hist[-1] and api.n_used == 0 and "clusters" in hist[-1]


def test_v5_flag_guards():
    from trajpe import pe as PE
    from trajpe.api_sim import SimAPI, VariationConfig
    rng = np.random.default_rng(4)
    cat = _GeoCat(rng)
    D1 = _trails(rng, cat, 60)
    for kw in ({"release_union_raw": 1}, {"work_shrink": "js"}):
        try:
            PE.run_pe(D1, np.arange(60).astype(str), SimAPI(cat, VariationConfig()), cat, _cfg(PE, 2.0, **kw),
                      log=lambda *_: None)
            raise AssertionError(f"{kw} accepted without a pool API")
        except ValueError:
            pass


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
