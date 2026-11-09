import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.spatial import cKDTree

from trajpe import fusion as FZ
from trajpe import workload as W
from trajpe.api_sim import N_BLOCKS, SemanticModel, SimAPI
from trajpe.marginals import Marginals, combine_popularity, dp_venue_popularity
from trajpe.trails import TrailSet

WF9 = "gap,hour_root,root_trans,hop,len_gyr,cell16,od8,weekhour,hb_cat"


class _SimCat:

    def __init__(self, rng, n=400, n_cat=12, n_roots=4):
        self.name = "test"
        self.xy = rng.uniform(0, 20, size=(n, 2))
        self.width_km = self.height_km = 20.0
        self.categories = [f"c{i}" for i in range(n_cat)]
        self.cat = rng.integers(0, n_cat, n).astype(np.int32)
        self.root_of_cat = (np.arange(n_cat) % (n_roots - 1)).astype(np.int32)
        self.root = self.root_of_cat[self.cat]
        self.n_roots = n_roots
        self.tree = cKDTree(self.xy)
        self._trees = {}

    def __len__(self):
        return len(self.xy)

    def clip_xy(self, xy):
        return np.clip(xy, 0.0, 20.0)

    def dist_km(self, a, b):
        return np.linalg.norm(self.xy[a] - self.xy[b], axis=-1)

    def nearest(self, xy, cat=None, root=None, k=1):
        codes, code = (self.cat, cat) if cat is not None else (self.root, root)
        if cat is None and root is None:
            return self.tree.query(xy, k=min(k, len(self)))[1]
        key = ("c" if cat is not None else "r", int(code))
        if key not in self._trees:
            idx = np.flatnonzero(codes == code)
            self._trees[key] = (cKDTree(self.xy[idx]) if len(idx) else None, idx)
        tree, idx = self._trees[key]
        if tree is None:
            return None
        return idx[tree.query(xy, k=min(k, tree.n))[1]]


def _trails(rng, cat, n):
    out = []
    for _ in range(n):
        L = int(rng.integers(2, 7))
        out.append((rng.integers(0, len(cat), L), np.cumsum(rng.integers(0, 300, L)) + rng.integers(0, 10080)))
    return TrailSet.from_lists(out)


def test_hb_cat_rows_and_sensitivity():
    rng = np.random.default_rng(0)
    cat = _SimCat(rng)
    D = _trails(rng, cat, 120)
    r, c, v = W.rows(D, cat, "hb_cat")
    assert np.all(v >= 0) and c.max() < W.n_bins("hb_cat", cat) == N_BLOCKS * len(cat.categories)
    assert np.allclose(np.bincount(r, weights=v, minlength=len(D)), 1.0)
    users = rng.integers(0, 15, len(D))
    _, inv, cnt = np.unique(users, return_inverse=True, return_counts=True)
    for w in (np.ones(len(D)), 1.0 / cnt[inv]):
        full = W.histogram(D, w, cat, "hb_cat")
        for u in np.unique(users):
            keep = np.flatnonzero(users != u) if w.max() < 1 else np.setdiff1d(np.arange(len(D)), [u])
            d = full - W.histogram(D.subset(keep), w[keep], cat, "hb_cat")
            assert np.all(d >= -1e-12) and abs(d.sum() - 1.0) < 1e-9 and np.linalg.norm(d) <= 1 + 1e-9


def test_venue2_sensitivity_budget_and_combination():
    from trajpe.dp import eps_for
    from trajpe.mechanisms import extra_sigma, noise_levels_w
    rng = np.random.default_rng(5)
    cat = _SimCat(rng)
    D = _trails(rng, cat, 90)
    users = rng.integers(0, 10, len(D))
    _, inv, cnt = np.unique(users, return_inverse=True, return_counts=True)
    for w, units in ((np.ones(len(D)), [[i] for i in range(len(D))]),
                     (1.0 / cnt[inv], [np.flatnonzero(users == u) for u in np.unique(users)])):
        full = dp_venue_popularity(D, w, len(cat), 0.0, rng)
        for drop in units:
            keep = np.setdiff1d(np.arange(len(D)), drop)
            d = full - dp_venue_popularity(D.subset(keep), w[keep], len(cat), 0.0, rng)
            assert np.all(d >= -1e-12) and abs(d.sum() - 1.0) < 1e-9 and np.linalg.norm(d) <= 1 + 1e-9
    for eps in (1.0, 4.0):
        s, sm, sw = noise_levels_w(eps, 1e-6, 3, True, 0.4, 0.07875, 9, 0.08)
        sv = extra_sigma(eps, 1e-6, 0.08)
        assert (s, sm, sw) != noise_levels_w(eps, 1e-6, 3, True, 0.4, 0.07875, 9)
        assert sm == noise_levels_w(eps, 1e-6, 3, True, 0.4, 0.07875, 9)[1]
        assert abs(eps_for(np.sqrt(3 / s ** 2 + 5 / sm ** 2 + 9 / sw ** 2 + 1 / sv ** 2), 1e-6) - eps) < 1e-6
    raw = rng.normal(0, 2.0, len(cat)) + 30.0 * (np.arange(len(cat)) < 5)
    marg = Marginals(pop=np.where(raw > 8, raw, 0.0), cat=np.ones(12), start=np.ones(168), sigma=2.0,
                     cat_dom=np.arange(3), cat_trans=np.zeros((3, 3)), venue_dom=np.zeros(0, np.int64),
                     venue_trans=np.zeros((0, 0)), pop_raw=raw)
    second = rng.normal(0, 2.0, len(cat)) + 30.0 * (np.arange(len(cat)) < 5)
    m2, sc = combine_popularity(marg, second, 2.0, None)
    assert abs(sc - 2.0 / np.sqrt(2)) < 1e-12 and np.allclose(m2.pop_raw, (raw + second) / 2)
    assert np.all(m2.pop[:5] > 0) and m2.digest() != marg.digest() and m2.cat is marg.cat


def _fake_dp(rng, cat):
    C = len(cat.categories)
    share = rng.uniform(1, 5, C)
    dom = np.argsort(-share)[:6]
    ct = np.zeros((6, 6))
    ct[0, 1], ct[2, 2], ct[3, 0] = 40.0, 25.0, 10.0
    marg = Marginals(pop=np.zeros(len(cat)), cat=share * 100, start=np.ones(168), sigma=1.0, cat_dom=dom,
                     cat_trans=ct, venue_dom=np.zeros(0, np.int64), venue_trans=np.zeros((0, 0)))
    work = {"root_trans": rng.uniform(0, 50, cat.n_roots ** 2), "weekhour": rng.uniform(0, 30, 168),
            "hb_cat": np.where(rng.random(N_BLOCKS * C) < 0.3, rng.uniform(0, 60, N_BLOCKS * C), rng.normal(0, 2, N_BLOCKS * C))}
    return marg, work


def test_semantic_model_tables_and_draws():
    rng = np.random.default_rng(1)
    cat = _SimCat(rng)
    marg, work = _fake_dp(rng, cat)
    vw = rng.uniform(0.1, 1.0, len(cat))
    for w in (work, {k: work[k] for k in ("root_trans",)}, {}):
        sem = SemanticModel.from_dp(cat, marg, w, 2.0, 1000.0, vw, k=5, scale_km=0.5)
        assert np.allclose(sem.T.sum(1), 1.0) and np.all(sem.T >= 0)
        assert np.allclose(sem.start_p.sum(1), 1.0) and np.all(np.isfinite(sem.tau)) and np.all(sem.tau > 0)
        dom = marg.cat_dom
        total = max(1000.0 * marg.cat_share()[dom[0]], 40.0)
        assert abs(sem.T[dom[0], dom[1]] - 40.0 / total) < 1e-9
        assert abs(sem.T[dom[2], dom[2]] - 25.0 / max(1000.0 * marg.cat_share()[dom[2]], 25.0)) < 1e-9
        blk = rng.integers(0, N_BLOCKS, 500)
        cs = sem.start_cat(blk, rng)
        nc = sem.next_cat(cs, blk, rng)
        assert cs.min() >= 0 and cs.max() < len(cat.categories) and nc.min() >= 0 and nc.max() < len(cat.categories)
        vg = sem.venue_global(nc, rng)
        assert np.array_equal(cat.cat[vg], nc)
        vn = sem.venue_near(nc, rng.uniform(0, 20, (len(nc), 2)), rng)
        assert np.array_equal(cat.cat[vn], nc)


def test_semantic_run_reads_only_dp_outputs():
    from trajpe import pe as PE
    from trajpe.api_sim import VariationConfig
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_workload import _ALLOW, _Guarded
    rng = np.random.default_rng(2)
    cat = _SimCat(rng)
    D1, D2 = _trails(rng, cat, 200), _trails(rng, cat, 150)
    u1, u2 = np.arange(200).astype(str), (np.arange(150) // 3).astype(str)
    cfg = PE.PEConfig(eps=4.0, T=3, n_syn=80, seed=3, marg_frac=0.3, calib_iters=20, fuse_iters=40, n_out=250,
                      work_frac=0.1, work_families=WF9, calib="all", time_ground="all", release_mode="fusion",
                      fuse_rho=0.2, release_pool=2, pool_solver="fusion", pool_diversify=1.0,
                      release_sampler="systematic", fuse_families=WF9 + ",start,cat,venue,cat_trans,venue_trans",
                      calib_families=WF9 + ",start,cat,venue,cat_trans,venue_trans", sem_frac=0.7, sem_var=0.5,
                      sem_k=4, venue2_frac=0.08, sem_refine=0.5)
    orig = {"dp_marginals": PE.dp_marginals, "dp_nn_histogram": PE.dp_nn_histogram, "dp_workload": W.dp_workload,
            "embed": PE.embed, "from_dp": SemanticModel.from_dp, "v2": PE.dp_venue_popularity}

    def allowed(fn):
        def wrapped(*a, **k):
            _ALLOW[0] += 1
            try:
                return fn(*a, **k)
            finally:
                _ALLOW[0] -= 1
        return wrapped

    seen = []

    def from_dp_rec(*a, **k):
        for x in a:
            assert not isinstance(x, _Guarded)
        seen.append(1)
        return orig["from_dp"](*a, **k)

    try:
        SemanticModel.from_dp = staticmethod(from_dp_rec)
        priv = _Guarded(D1.poi, D1.tmin, D1.offsets)
        PE.dp_marginals, W.dp_workload = allowed(orig["dp_marginals"]), allowed(orig["dp_workload"])
        PE.dp_venue_popularity = allowed(orig["v2"])
        PE.embed = lambda ts, *a, **k: (allowed(orig["embed"]) if isinstance(ts, _Guarded) else orig["embed"])(ts, *a, **k)
        S, hist, privacy, _ = PE.run_pe(priv, u1, SimAPI(cat, VariationConfig()), cat, cfg, log=lambda *_: None)
        assert len(S) == 250 and seen and abs(privacy["eps_check"] - 4.0) < 1e-6
        assert np.all(np.diff(S.offsets) >= 1) and "sigma_venue2" in privacy
    finally:
        PE.dp_marginals, W.dp_workload, PE.embed = orig["dp_marginals"], orig["dp_workload"], orig["embed"]
        SemanticModel.from_dp, PE.dp_venue_popularity = orig["from_dp"], orig["v2"]
    from trajpe.mechanisms import noise_levels
    regen = orig["dp_marginals"](D1, np.ones(len(D1)), cat.cat, len(cat.categories),
                                 noise_levels(4.0, 1e-6, 3, True, 0.3)[1], None, np.random.default_rng(3))
    assert regen.digest() == privacy["marg_sha256"] and int((regen.pop > 0).sum()) == privacy["marg_nonzero_venues"]

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
                PE.dp_venue_popularity = lambda priv, w, *a: orig["v2"](D1, np.ones(len(D1)), *a)
            S, _, _, _ = PE.run_pe(D, u, SimAPI(cat, VariationConfig()), cat, cfg, log=lambda *_: None)
            outs.append(S)
        assert all(np.array_equal(getattr(outs[0], f), getattr(outs[1], f)) for f in ("poi", "tmin", "offsets"))
    finally:
        PE.dp_marginals, PE.dp_nn_histogram, PE.embed = orig["dp_marginals"], orig["dp_nn_histogram"], orig["embed"]
        W.dp_workload, PE.dp_venue_popularity = orig["dp_workload"], orig["v2"]


def test_semantic_flags_off_build_nothing():
    from trajpe import pe as PE
    from trajpe.api_sim import VariationConfig
    rng = np.random.default_rng(4)
    cat = _SimCat(rng)
    D = _trails(rng, cat, 120)
    called = []
    orig = SemanticModel.from_dp
    try:
        SemanticModel.from_dp = staticmethod(lambda *a, **k: called.append(1))
        api = SimAPI(cat, VariationConfig())
        cfg = PE.PEConfig(eps=4.0, T=2, n_syn=50, seed=1, n_out=100, work_frac=0.1, work_families=WF9,
                          release_mode="fusion", fuse_iters=30)
        S, _, _, _ = PE.run_pe(D, np.arange(120).astype(str), api, cat, cfg, log=lambda *_: None)
    finally:
        SemanticModel.from_dp = orig
    assert called == [] and api.random.sem is None and api.variation.sem is None and len(S) == 100


if __name__ == "__main__":
    test_hb_cat_rows_and_sensitivity()
    test_venue2_sensitivity_budget_and_combination()
    test_semantic_model_tables_and_draws()
    test_semantic_run_reads_only_dp_outputs()
    test_semantic_flags_off_build_nothing()
    print("all v3.1 tests passed")
