import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy import sparse

from trajpe import fusion as FZ
from trajpe import workload as W
from trajpe.dp import eps_for, mu_for
from trajpe.marginals import N_MARGINALS
from trajpe.trails import TrailSet

DAY, WEEK = 1440, 10080


class _Cat:
    def __init__(self, rng, n=200):
        self.xy = rng.uniform(0, 30, size=(n, 2))
        self.width_km = self.height_km = 30.0
        self.cat = rng.integers(0, 6, n).astype(np.int32)
        self.categories = list("abcdef")
        self.root = (self.cat % 3).astype(np.int32)
        self.n_roots = 4

    def __len__(self):
        return len(self.xy)


def _edge_trails(rng, n_venues):
    out = [(np.array([3, 7]), np.array([100, 130])),
           (rng.integers(0, n_venues, 20), np.cumsum(np.r_[500, rng.integers(0, 481, 19)])),
           (np.array([5, 5, 5, 5]), np.array([600, 600, 610, 1000])),
           (np.array([1, 2, 1, 2, 1]), np.array([1400, 1430, 1445, 1500, 1700])),
           (np.array([9, 8]), np.array([DAY - 1, DAY + 479])),
           (np.array([4, 6, 4]), np.array([WEEK - 30, WEEK + 10, WEEK + 200])),
           (rng.integers(0, n_venues, 20), np.r_[WEEK - 1, WEEK - 1 + np.cumsum(rng.integers(0, 481, 19))])]
    for _ in range(80):
        L = int(rng.integers(2, 21))
        out.append((rng.integers(0, n_venues, L), np.cumsum(rng.integers(0, 481, L)) + rng.integers(0, WEEK)))
    return TrailSet.from_lists(out)


def test_family_rows_have_mass_one():
    rng = np.random.default_rng(0)
    cat = _Cat(rng)
    D = _edge_trails(rng, len(cat))
    for f in W.FAMILIES:
        r, c, v = W.rows(D, cat, f)
        assert np.all(v >= 0) and np.all((c >= 0) & (c < W.n_bins(f, cat))), f
        mass = np.bincount(r, weights=v, minlength=len(D))
        assert np.allclose(mass, 1.0, atol=1e-12), (f, mass.min(), mass.max())


def test_workload_sensitivity_trail_and_user_level():
    rng = np.random.default_rng(1)
    cat = _Cat(rng)
    D = _edge_trails(rng, len(cat))
    n = len(D)
    full1 = W.dp_workload(D, np.ones(n), cat, 0.0, rng, W.FAMILIES)
    for i in list(range(7)) + list(range(7, n, 9)):
        keep = np.setdiff1d(np.arange(n), [i])
        part = W.dp_workload(D.subset(keep), np.ones(n - 1), cat, 0.0, rng, W.FAMILIES)
        for f in W.FAMILIES:
            d = full1[f] - part[f]
            assert np.all(d >= -1e-12), f
            assert abs(d.sum() - 1.0) < 1e-9 and np.linalg.norm(d) <= 1.0 + 1e-9, (f, i, d.sum())
    users = rng.integers(0, 12, n)
    _, inv, cnt = np.unique(users, return_inverse=True, return_counts=True)
    w = 1.0 / cnt[inv]
    full = W.dp_workload(D, w, cat, 0.0, rng, W.FAMILIES)
    for u in np.unique(users):
        keep = np.flatnonzero(users != u)
        part = W.dp_workload(D.subset(keep), w[keep], cat, 0.0, rng, W.FAMILIES)
        for f in W.FAMILIES:
            d = full[f] - part[f]
            assert np.all(d >= -1e-12), f
            assert abs(d.sum() - 1.0) < 1e-9 and np.linalg.norm(d) <= 1.0 + 1e-9, (f, u, d.sum())
    assert np.all(w <= 1.0) and np.allclose(np.bincount(inv, weights=w), 1.0)


def test_budget_split_sums_to_mu2():
    import dp_accounting
    from trajpe.mechanisms import noise_levels, noise_levels_w
    for eps, T, K, mf, wf in [(1.0, 3, 8, 0.4, 0.07), (4.0, 3, 8, 0.4, 0.07), (2.0, 5, 3, 0.2, 0.15),
                              (8.0, 3, 12, 0.4, 0.3)]:
        s, sm, sw = noise_levels_w(eps, 1e-6, T, True, mf, wf, K)
        mu2 = mu_for(eps, 1e-6) ** 2
        parts = (T / s ** 2, N_MARGINALS / sm ** 2, K / sw ** 2)
        assert abs(sum(parts) - mu2) < 1e-9 * mu2, (sum(parts), mu2)
        assert abs(parts[0] - (1 - mf - wf) * mu2) < 1e-9 * mu2 and abs(parts[2] - wf * mu2) < 1e-9 * mu2
        assert abs(eps_for(np.sqrt(sum(parts)), 1e-6) - eps) < 1e-6
        assert sm == noise_levels(eps, 1e-6, T, True, mf)[1]
        acc = dp_accounting.pld.PLDAccountant()
        acc.compose(dp_accounting.GaussianDpEvent(s), T)
        acc.compose(dp_accounting.GaussianDpEvent(sm), N_MARGINALS)
        acc.compose(dp_accounting.GaussianDpEvent(sw), K)
        assert abs(acc.get_epsilon(1e-6) - eps) / eps < 0.01
    assert noise_levels_w(4.0, 1e-6, 3, True, 0.4, 0.0, 8) == (*noise_levels(4.0, 1e-6, 3, True, 0.4), 0.0)


def test_calibration_invariants():
    rng = np.random.default_rng(2)
    n, B = 500, 6
    lab = rng.integers(0, B, n)
    Phi = sparse.csr_matrix((np.ones(n), (np.arange(n), lab)), shape=(n, B))
    p = rng.random(n) * (rng.random(n) > 0.3)
    p /= p.sum()
    t = np.bincount(lab[p > 0], minlength=B).astype(float)
    t = t / t.sum()
    for solver in (W.calibrate, W.calibrate_parallel):
        q = solver(p, [(Phi, t)], 200, 1.0, 0.0)
        assert np.all(q[p == 0] == 0)
        assert np.abs(Phi.T @ q - t).max() < 1e-6
        qc = solver(p, [(Phi, t)], 200, 1.0, 2.0)
        r = qc[p > 0] / p[p > 0]
        assert r.max() / r.min() <= 4.0 + 1e-6


def test_fusion_invariants():
    rng = np.random.default_rng(3)
    n, B, N = 400, 5, 1000.0
    lab = rng.integers(0, B, n)
    Phi = sparse.csr_matrix((np.ones(n), (np.arange(n), lab)), shape=(n, B))
    p = rng.random(n) * (rng.random(n) > 0.2)
    p /= p.sum()
    target = np.array([0.4, 0.3, 0.15, 0.1, 0.05])
    fam = W.Family("x", Phi, N * target, N * target, np.full(B, 2.0))
    w, _ = FZ.fuse([fam], N, p, gamma=1e6)
    assert np.abs(w - p).max() < 1e-4
    w, info = FZ.fuse([fam], N, p, gamma=1e-3, max_iter=500)
    assert np.abs(Phi.T @ w - target).max() < 2e-3, Phi.T @ w
    assert np.all(w[p == 0] == 0)
    pr = FZ.vote_prior(np.where(p > 0, p, 0.0), 0.2)
    assert np.all(pr > 0) and abs(pr.sum() - 1) < 1e-12
    idx = FZ.systematic(pr, 1000, rng)
    assert np.all(np.abs(np.bincount(idx, minlength=n) - 1000 * pr) < 1 + 1e-9)


def test_families_rows():
    from trajpe.marginals import dp_marginals
    rng = np.random.default_rng(4)
    cat = _Cat(rng)
    D = _edge_trails(rng, len(cat))
    marg = dp_marginals(D, np.ones(len(D)), cat.cat, len(cat.categories), 0.1, None, rng)
    work = W.dp_workload(D, np.ones(len(D)), cat, 0.1, rng, W.FAMILIES)
    popular = np.flatnonzero(marg.pop > 0)
    dp = W.DPOutputs(work, 0.1, marg, 0.1, popular, float(marg.start.sum()))
    names = list(W.FAMILIES) + list(W.MARGINAL_FAMILIES)
    fams = W.families(D, cat, names, dp)
    assert {f.name for f in fams} == set(names), {f.name for f in fams} ^ set(names)
    for f in fams:
        assert f.Phi.shape[0] == len(D) and f.Phi.shape[1] == len(f.t) == len(f.y) == len(f.sigma), f.name
        assert np.all(f.t >= 0) and f.Phi.min() >= 0, f.name
        rs = np.asarray(f.Phi.sum(1)).ravel()
        if f.calib:
            assert np.allclose(rs, 1.0), (f.name, rs.min(), rs.max())
        else:
            assert np.all(rs <= 1.0 + 1e-12), f.name


def test_time_ground_invariants():
    rng = np.random.default_rng(5)
    cat = _Cat(rng)
    D = _edge_trails(rng, len(cat))
    target = rng.random(W.N_LC * W.N_HB * W.N_GAP)
    G = W.time_ground(D, target, rng)
    assert np.array_equal(G.poi, D.poi) and np.array_equal(G.offsets, D.offsets)
    assert np.array_equal(G.tmin[G.first], D.tmin[D.first])
    g = np.diff(G.tmin.astype(np.int64))
    inside = np.ones(len(G.tmin) - 1, bool)
    inside[G.last[:-1]] = False
    assert np.all(g[inside] >= 0) and np.all(g[inside] <= 480)


class _FakeCatalog(_Cat):
    pass


class _FakeAPI:
    def __init__(self, n_venues):
        self.n = n_venues

    def set_marginals(self, *args, **kwargs):
        pass

    def random(self, n, rng):
        lens = rng.integers(2, 6, n)
        return TrailSet.from_lists([(rng.integers(0, self.n, L), np.sort(rng.integers(0, 5000, L))) for L in lens])

    def variation(self, ts, degree, rng):
        out = []
        for p, t in ts.to_lists():
            p = p.copy()
            p[rng.integers(len(p))] = rng.integers(0, self.n)
            out.append((p, t))
        return TrailSet.from_lists(out)

    def diversify(self, ts, p, radius_km, rng):
        return self.variation(ts, p, rng)


_ALLOW = [0]


class _Guarded(TrailSet):
    def __getattribute__(self, name):
        if name in ("poi", "tmin", "offsets") and not _ALLOW[0]:
            raise AssertionError(f"private trails read outside a DP mechanism ({name})")
        return object.__getattribute__(self, name)


V3_CONFIGS = [
    dict(work_frac=0.1, calib="all", time_ground="all", release_pool=2, n_out=300),
    dict(work_frac=0.1, calib="all", time_ground="all", release_mode="fusion", fuse_rho=0.2, release_pool=2,
         pool_solver="fusion", pool_diversify=1.0, release_sampler="systematic", n_out=300),
    dict(work_frac=0.1, release_mode="fusion", fuse_rho=0.2, n_out=300, release_diversify=0.5),
    dict(work_frac=0.1, calib="all", calib_solver="fusion", time_ground="all", release_mode="fusion", fuse_rho=0.2,
         calib_families="gap,hour_root,od8,ctrans16,start,cat,venue,cat_trans,venue_trans", release_pool=2,
         pool_solver="fusion", pool_diversify=1.0, release_sampler="systematic", n_out=300),
]


def _v3_cfg(**kw):
    from trajpe.pe import PEConfig
    base = dict(eps=4.0, T=3, n_syn=60, seed=3, marg_frac=0.3, calib_iters=30, fuse_iters=60,
                work_families="gap,hour_root,root_trans,hop,len_gyr,cell16,od8,weekhour,ctrans16")
    base.update(kw)
    return PEConfig(**base)


def test_postprocessing_reads_only_dp_outputs():
    from trajpe import pe as PE
    rng = np.random.default_rng(6)
    cat = _FakeCatalog(rng, 80)
    api = _FakeAPI(len(cat))
    plain = api.random(200, rng)
    priv = _Guarded(plain.poi, plain.tmin, plain.offsets)
    users = np.arange(200).astype(str)
    private_arrays = []

    def allowed(fn):
        def wrapped(*a, **k):
            _ALLOW[0] += 1
            try:
                out = fn(*a, **k)
            finally:
                _ALLOW[0] -= 1
            return out
        return wrapped

    def arrays(x):
        assert not isinstance(x, _Guarded), "private trails passed to post-processing"
        if isinstance(x, np.ndarray):
            return [x]
        if isinstance(x, TrailSet):
            return [x.poi, x.tmin, x.offsets]
        if isinstance(x, W.DPOutputs):
            return list(x.work.values()) + [a for a in (x.popular,) if a is not None]
        if isinstance(x, (list, tuple)):
            return [a for y in x for a in arrays(y)]
        return []

    def no_private(fn):
        def wrapped(*a, **k):
            for arr in arrays(list(a) + list(k.values())):
                for pa in private_arrays:
                    assert not np.shares_memory(arr, pa), fn.__name__
            return fn(*a, **k)
        return wrapped

    def embed_rec(ts, *a, **k):
        if isinstance(ts, _Guarded):
            e = allowed(orig["embed"])(ts, *a, **k)
            private_arrays.append(e)
            return e
        return orig["embed"](ts, *a, **k)

    def weights_rec(u, unit):
        w = orig["vote_weights"](u, unit)
        private_arrays.append(w)
        return w

    orig = {"dp_marginals": PE.dp_marginals, "embed": PE.embed, "vote_weights": PE.vote_weights,
            "dp_workload": W.dp_workload, "calibrate": W.calibrate, "calibrate_parallel": W.calibrate_parallel,
            "families": W.families, "time_ground": W.time_ground, "fuse": FZ.fuse}
    try:
        PE.dp_marginals = allowed(orig["dp_marginals"])
        PE.embed = embed_rec
        PE.vote_weights = weights_rec
        W.dp_workload = allowed(orig["dp_workload"])
        for name in ("calibrate", "calibrate_parallel", "families", "time_ground"):
            setattr(W, name, no_private(orig[name]))
        FZ.fuse = no_private(orig["fuse"])
        for kw in V3_CONFIGS:
            S, hist, privacy, _ = PE.run_pe(priv, users, api, cat, _v3_cfg(**kw), log=lambda *_: None)
            assert len(S) == kw["n_out"] and abs(privacy["eps_check"] - 4.0) < 1e-6, privacy
            for rec in hist:
                assert "max_count" not in rec and "mean_nn_dist" not in rec, rec
    finally:
        PE.dp_marginals, PE.embed, PE.vote_weights = orig["dp_marginals"], orig["embed"], orig["vote_weights"]
        for name in ("dp_workload", "calibrate", "calibrate_parallel", "families", "time_ground"):
            setattr(W, name, orig[name])
        FZ.fuse = orig["fuse"]


def test_release_depends_on_data_only_through_dp_outputs():
    from trajpe import pe as PE
    rng = np.random.default_rng(7)
    cat = _FakeCatalog(rng, 80)
    api = _FakeAPI(len(cat))
    D1, D2 = api.random(200, rng), api.random(150, rng)
    u1, u2 = np.arange(200).astype(str), (np.arange(150) // 3).astype(str)
    orig = {"dp_marginals": PE.dp_marginals, "dp_nn_histogram": PE.dp_nn_histogram, "dp_workload": W.dp_workload,
            "embed": PE.embed}
    e1 = {}

    def marg_as_d1(priv, weights, *a):
        return orig["dp_marginals"](D1, np.ones(len(D1)), *a)

    def work_as_d1(priv, weights, *a):
        return orig["dp_workload"](D1, np.ones(len(D1)), *a)

    def embed_rec(ts, catalog, ecfg):
        out = orig["embed"](ts, catalog, ecfg)
        if ts is D1:
            e1["e"] = out
        return out

    def votes_as_d1(e_priv, e_syn, weights, *a):
        return orig["dp_nn_histogram"](e1["e"], e_syn, np.ones(len(D1)), *a)

    try:
        for kw in V3_CONFIGS:
            outs = []
            for D, u in ((D1, u1), (D2, u2)):
                PE.embed = embed_rec
                if D is D2:
                    PE.dp_marginals, PE.dp_nn_histogram, W.dp_workload = marg_as_d1, votes_as_d1, work_as_d1
                S, _, _, _ = PE.run_pe(D, u, api, cat, _v3_cfg(**kw), log=lambda *_: None)
                outs.append(S)
                PE.dp_marginals, PE.dp_nn_histogram, W.dp_workload = (orig["dp_marginals"], orig["dp_nn_histogram"],
                                                                       orig["dp_workload"])
            assert all(np.array_equal(getattr(outs[0], f), getattr(outs[1], f)) for f in ("poi", "tmin", "offsets")), kw
    finally:
        PE.dp_marginals, PE.dp_nn_histogram, PE.embed = orig["dp_marginals"], orig["dp_nn_histogram"], orig["embed"]
        W.dp_workload = orig["dp_workload"]


def test_v3_off_is_frozen_path():
    from trajpe import pe as PE
    rng = np.random.default_rng(8)
    cat = _FakeCatalog(rng, 80)
    api = _FakeAPI(len(cat))
    D = api.random(120, rng)
    called = []
    orig = W.dp_workload, W.families
    try:
        W.dp_workload = lambda *a, **k: called.append("work")
        W.families = lambda *a, **k: called.append("families")
        cfg = PE.PEConfig(eps=4.0, T=2, n_syn=40, seed=1, n_out=100)
        assert not PE.v3_enabled(cfg)
        S, _, privacy, _ = PE.run_pe(D, np.arange(120).astype(str), api, cat, cfg, log=lambda *_: None)
    finally:
        W.dp_workload, W.families = orig
    assert called == [] and len(S) == 100 and "sigma_work" not in privacy


if __name__ == "__main__":
    test_family_rows_have_mass_one()
    test_workload_sensitivity_trail_and_user_level()
    test_budget_split_sums_to_mu2()
    test_calibration_invariants()
    test_fusion_invariants()
    test_families_rows()
    test_time_ground_invariants()
    test_postprocessing_reads_only_dp_outputs()
    test_release_depends_on_data_only_through_dp_outputs()
    test_v3_off_is_frozen_path()
    print("all workload tests passed")
