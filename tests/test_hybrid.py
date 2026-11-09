import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines" / "dplora"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import hybrid as HY
from test_v31 import WF9, _SimCat, _trails
from trajpe import workload as W
from trajpe.api_pool import PoolAPI
from trajpe.api_sim import SimAPI, VariationConfig
from trajpe.dp import eps_for, mu_for


class _GeoCat(_SimCat):

    def __init__(self, rng, **kw):
        super().__init__(rng, **kw)
        self.cat_index = {c: i for i, c in enumerate(self.categories)}
        self.lat, self.lon = self.xy[:, 1].copy(), self.xy[:, 0].copy()

    def to_xy(self, lat, lon):
        return np.stack([np.asarray(lon, np.float64), np.asarray(lat, np.float64)], -1)


def _records(rng, cat, n):
    out = []
    for _ in range(n):
        L = int(rng.integers(2, 6))
        mins = np.sort(rng.integers(0, 1440, L))
        cks = [{"time": f"{m // 60:02d}:{m % 60:02d}", "category": cat.categories[int(rng.integers(len(cat.categories)))],
                "lat": float(rng.uniform(0, 20)), "lon": float(rng.uniform(0, 20))} for m in mins]
        out.append((int(rng.integers(7)), cks))
    return out


def test_budget_split_and_composition():
    eps, f, batch, epochs = 2.0, 0.5, 1494, 0.6
    cal = HY.calibrate(eps, f, batch, epochs)
    mu = mu_for(eps, 1e-6)
    assert abs(cal["mu_pe"] ** 2 - (1 - f) * mu ** 2) < 1e-12 and cal["n_hist"] == 1
    assert abs(cal["sigma_hist"] - 1 / cal["mu_pe"]) < 1e-12
    q, T = cal["sample_rate"], cal["steps"]
    e = HY.composed_eps(cal["sigma_sgd"], q, T, cal["mu_pe"])
    assert e <= eps - HY.PRV_MARGIN + 1e-9 and e > eps - HY.PRV_MARGIN - 0.01 and abs(e - cal["eps_composed_pld"]) < 1e-12
    assert HY.composed_eps(cal["sigma_sgd"] * 1.1, q, T, cal["mu_pe"]) < e
    assert HY.composed_eps(cal["sigma_sgd"], q, T, cal["mu_pe"] * 0.9) < e
    assert cal["eps_sgd_only_pld"] < e and cal["eps_pe_only_pld"] < e
    e_gdp = eps_for(cal["mu_pe"], 1e-6)
    assert e_gdp <= cal["eps_pe_only_pld"] + 1e-6 and cal["eps_pe_only_pld"] - e_gdp < 5e-3
    assert mu_for(cal["eps_pe_run"], 1e-6) <= cal["mu_pe"]
    assert HY.check(cal, mu_for(cal["eps_pe_run"], 1e-6))["eps_composed_pld_spent"] <= eps
    for bad in (cal["mu_pe"] * 1.01,):
        try:
            HY.check(cal, bad)
            raise AssertionError("check accepted a larger PE mu")
        except SystemExit:
            pass
    try:
        HY.calibrate(eps, 1e-9, batch, epochs)
        raise AssertionError("calibrate accepted a split without room for DP-SGD")
    except ValueError:
        pass


def _pool_api(cat, recs, **kw):
    base = SimAPI(cat, VariationConfig())
    return PoolAPI(cat, recs, base.random, base.variation, **kw)


def test_pool_api_draws_grounding_refresh():
    rng = np.random.default_rng(5)
    cat = _GeoCat(rng)
    recs = _records(rng, cat, 120)
    api = _pool_api(cat, recs, ground_scale_km=0.5)
    vw = rng.uniform(0.1, 1.0, len(cat))
    pop = np.argsort(-vw)[:30]
    api.set_marginals(vw, np.full(168, 1 / 168), use=("prompt", "ground", "var"), popular=(pop, vw[pop]))
    assert api.ground.venue_w is vw and api.ground.popular_by_cat
    r = np.random.default_rng(0)
    a = api.random(50, r)
    b = api.random(70, r)
    assert len(a) == 50 and len(b) == 70 and api.n_wraps == 0 and api.n_used == 120
    used = np.concatenate([api._order[:120]])
    assert len(np.unique(used)) == 120
    i = int(api._order[0])
    assert np.array_equal(cat.cat[api.pool.trail(i)[0]], [cat.cat_index[c["category"]] for c in recs[i][1]])
    c = api.random(10, r)
    assert len(c) == 10 and api.n_wraps == 1
    try:
        api.set_marginals(vw, np.full(168, 1 / 168))
        raise AssertionError("set_marginals after grounding accepted")
    except RuntimeError:
        pass
    api2 = _pool_api(cat, recs, refresh=0.5)
    api2.set_marginals(vw, np.full(168, 1 / 168))
    S = api2.random(40, np.random.default_rng(1))
    n0 = api2.n_used
    out = api2.variation(S, 1.0, np.random.default_rng(2))
    assert len(out) == 40 and 8 <= api2.n_used - n0 <= 32
    api3 = _pool_api(cat, recs)
    api3.set_marginals(vw, np.full(168, 1 / 168))
    S3 = api3.random(40, np.random.default_rng(1))
    o1 = api3.variation(S3, 1.0, np.random.default_rng(2))
    o2 = api3.rule(S3, 1.0, np.random.default_rng(2))
    assert np.array_equal(o1.poi, o2.poi) and np.array_equal(o1.tmin, o2.tmin)
    api4 = _pool_api(cat, recs, pool_frac=0.5)
    api4.set_marginals(vw, np.full(168, 1 / 168))
    assert len(api4.random(40, np.random.default_rng(3))) == 40 and api4.n_used == 20


def test_hybrid_run_reads_only_dp_outputs():
    from trajpe import pe as PE
    from test_workload import _ALLOW, _Guarded
    rng = np.random.default_rng(2)
    cat = _GeoCat(rng)
    D1, D2 = _trails(rng, cat, 200), _trails(rng, cat, 150)
    u1, u2 = np.arange(200).astype(str), (np.arange(150) // 3).astype(str)
    recs = _records(rng, cat, 300)
    mu_pe = math.sqrt(0.5) * mu_for(4.0, 1e-6)
    eps_pe = eps_for(mu_pe * (1 - HY.MU_SAFETY), 1e-6)
    cfg = PE.PEConfig(eps=eps_pe, T=3, n_syn=80, seed=3, marg_frac=0.4, calib_iters=20, fuse_iters=40, n_out=250,
                      work_frac=0.07875, work_families=WF9, calib="all", calib_solver="parallel", time_ground="all",
                      release_mode="fusion", fuse_rho=0.2, release_pool=2, pool_solver="fusion", pool_diversify=1.0,
                      release_sampler="systematic", fuse_families=WF9 + ",start,cat,venue,cat_trans,venue_trans",
                      calib_families=WF9 + ",start,cat,venue,cat_trans,venue_trans", sem_refine=1.0,
                      sem_refine_mode="gibbs", sem_refine_stage="pre_div", sem_k=4)
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
        api = _pool_api(cat, recs, refresh=0.5)
        S, hist, privacy, _ = PE.run_pe(priv, u1, api, cat, cfg, log=lambda *_: None)
        assert len(S) == 250 and api.n_used >= 80 and api.ground.venue_w is not None
        assert privacy["mu"] <= mu_pe and abs(privacy["mu"] - mu_pe) < 1e-6
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
            S, _, _, _ = PE.run_pe(D, u, _pool_api(cat, recs, refresh=0.5), cat, cfg, log=lambda *_: None)
            outs.append(S)
        assert all(np.array_equal(getattr(outs[0], f), getattr(outs[1], f)) for f in ("poi", "tmin", "offsets"))
    finally:
        PE.dp_marginals, PE.dp_nn_histogram, PE.embed = orig["dp_marginals"], orig["dp_nn_histogram"], orig["embed"]
        W.dp_workload = orig["dp_workload"]


def test_pool_api_v32_e2_flags():
    from trajpe import pe as PE
    from test_workload import _ALLOW, _Guarded
    rng = np.random.default_rng(7)
    cat = _GeoCat(rng)
    D1 = _trails(rng, cat, 200)
    recs = _records(rng, cat, 300)
    mu_pe = math.sqrt(0.6) * mu_for(4.0, 1e-6)
    eps_pe = eps_for(mu_pe * (1 - HY.MU_SAFETY), 1e-6)
    wf = WF9 + ",tdist,diam,od6,ctrans6,tcell6"
    fams = wf + ",start,cat,venue,cat_trans,venue_trans"
    cfg = PE.PEConfig(eps=eps_pe, T=3, n_syn=80, seed=3, marg_frac=0.4, calib_iters=20, fuse_iters=40, n_out=250,
                      work_frac=0.245, work_families=wf, calib="all", calib_solver="parallel", time_ground="all",
                      release_mode="fusion", fuse_rho=0.2, release_pool=4, release_degree=0.2, pool_solver="fusion",
                      pool_diversify=1.0, release_sampler="systematic", fuse_families=fams, calib_families=fams,
                      sem_refine=1.0, sem_refine_mode="gibbs", sem_refine_stage="pre_div", sem_k=4, hop_law="dp",
                      pool_stretch=0.02)
    orig = {"dp_marginals": PE.dp_marginals, "dp_workload": W.dp_workload, "embed": PE.embed}

    def allowed(fn):
        def wrapped(*a, **k):
            _ALLOW[0] += 1
            try:
                return fn(*a, **k)
            finally:
                _ALLOW[0] -= 1
        return wrapped
    try:
        PE.dp_marginals, W.dp_workload = allowed(orig["dp_marginals"]), allowed(orig["dp_workload"])
        PE.embed = lambda ts, *a, **k: (allowed(orig["embed"]) if isinstance(ts, _Guarded) else orig["embed"])(ts, *a, **k)
        for frac in (1.0, 0.5):
            priv = _Guarded(D1.poi, D1.tmin, D1.offsets)
            api = _pool_api(cat, recs, pool_frac=frac)
            S, _, privacy, _ = PE.run_pe(priv, np.arange(200).astype(str), api, cat, cfg, log=lambda *_: None)
            assert len(S) == 250 and api.sim.hop_law is not None and api.n_used >= round(80 * frac)
            assert privacy["mu"] <= mu_pe and abs(privacy["mu"] - mu_pe) < 1e-6
    finally:
        PE.dp_marginals, W.dp_workload, PE.embed = orig["dp_marginals"], orig["dp_workload"], orig["embed"]


if __name__ == "__main__":
    test_budget_split_and_composition()
    test_pool_api_draws_grounding_refresh()
    test_hybrid_run_reads_only_dp_outputs()
    test_pool_api_v32_e2_flags()
    print("all hybrid tests passed")
