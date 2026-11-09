import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from trajpe.dp import eps_for, gdp_delta, mu_for, sigma_for
from trajpe.trails import TrailSet


def test_gdp_matches_pld():
    import dp_accounting
    for eps, delta, T in [(1.0, 1e-5, 5), (4.0, 1e-5, 10), (8.0, 1e-5, 20)]:
        sigma = sigma_for(eps, delta, T)
        acc = dp_accounting.pld.PLDAccountant()
        acc.compose(dp_accounting.GaussianDpEvent(sigma), T)
        eps_pld = acc.get_epsilon(delta)
        assert abs(eps_pld - eps) / eps < 0.01, (eps, eps_pld)


def test_marginal_budget_split():
    import dp_accounting
    from trajpe.mechanisms import noise_levels
    for eps, T in [(1.0, 5), (4.0, 10)]:
        s, sm = noise_levels(eps, 1e-5, T, True, 0.2)
        from trajpe.marginals import N_MARGINALS
        assert abs(eps_for(np.sqrt(T / s**2 + N_MARGINALS / sm**2), 1e-5) - eps) < 1e-6
        acc = dp_accounting.pld.PLDAccountant()
        acc.compose(dp_accounting.GaussianDpEvent(s), T)
        acc.compose(dp_accounting.GaussianDpEvent(sm), N_MARGINALS)
        assert abs(acc.get_epsilon(1e-5) - eps) / eps < 0.01
    s, sm = noise_levels(2.0, 1e-5, 0, True, 0.2)
    from trajpe.marginals import N_MARGINALS
    assert s == 0 and abs(eps_for(np.sqrt(N_MARGINALS / sm**2), 1e-5) - 2.0) < 1e-6


def test_vote_sensitivity():
    from trajpe.embed import kmeans
    from trajpe.mechanisms import dp_nn_histogram
    rng = np.random.default_rng(0)
    e_syn = rng.normal(size=(200, 16)).astype(np.float32)
    labels = kmeans(e_syn, 20)
    for _ in range(20):
        e_one = rng.normal(size=(1, 16)).astype(np.float32)
        for k, lab in [(1, None), (4, None), (1, labels)]:
            counts, _, _ = dp_nn_histogram(e_one, e_syn, np.ones(1), 0.0, -1.0, rng, k, lab,
                                           20 if lab is not None else None)
            assert abs(np.linalg.norm(counts) - 1.0) < 1e-9, (k, np.linalg.norm(counts))


class _FakeCatalog:
    def __init__(self, rng):
        self.xy = rng.uniform(0, 10, size=(60, 2))
        self.width_km = self.height_km = 10.0
        self.cat = rng.integers(0, 4, 60).astype(np.int32)
        self.categories = ["a", "b", "c", "d"]
        self.root = (self.cat % 2).astype(np.int32)
        self.n_roots = 3

    def __len__(self):
        return 60


class _FakeAPI:
    def __init__(self, n_venues):
        self.n = n_venues

    def set_marginals(self, *args, **kwargs):
        pass

    def random(self, n, rng):
        lens = rng.integers(2, 5, n)
        return TrailSet.from_lists([(rng.integers(0, self.n, L), np.sort(rng.integers(0, 5000, L))) for L in lens])

    def variation(self, ts, degree, rng):
        out = []
        for p, t in ts.to_lists():
            p = p.copy()
            p[rng.integers(len(p))] = rng.integers(0, self.n)
            out.append((p, t))
        return TrailSet.from_lists(out)


def test_release_independent_of_dataset_size():
    from trajpe.pe import DEFAULT_N_OUT, PEConfig, run_pe
    rng = np.random.default_rng(0)
    cat = _FakeCatalog(rng)
    api = _FakeAPI(len(cat))
    D = api.random(300, rng)
    users = np.arange(300).astype(str)
    D_minus = D.subset(np.arange(299))
    sizes = []
    for data, u in [(D, users), (D_minus, users[:299])]:
        S, hist, privacy, _ = run_pe(data, u, api, cat, PEConfig(eps=4.0, T=2, n_syn=40, seed=1), log=lambda *_: None)
        sizes.append(len(S))
        for rec in hist:
            assert "max_count" not in rec and "mean_nn_dist" not in rec, rec
    assert sizes == [DEFAULT_N_OUT, DEFAULT_N_OUT], sizes


def test_transition_sensitivity():
    from trajpe.marginals import _transition_hist
    rng = np.random.default_rng(0)
    for _ in range(50):
        L = int(rng.integers(2, 21))
        one = TrailSet.from_lists([(rng.integers(0, 30, L), np.sort(rng.integers(0, 5000, L)))])
        h = _transition_hist(one, one.poi, np.arange(30), 30, np.ones(1))
        assert h.sum() <= 1.0 + 1e-9 and np.linalg.norm(h) <= 1.0 + 1e-9


def test_gdp_roundtrip():
    for mu in [0.3, 1.0, 2.5]:
        e = eps_for(mu, 1e-5)
        assert abs(gdp_delta(mu, e) - 1e-5) < 1e-9
        assert abs(mu_for(e, 1e-5) - mu) < 1e-6


def test_trailset_subset_roundtrip():
    rng = np.random.default_rng(0)
    trails = [(rng.integers(0, 100, L), np.sort(rng.integers(0, 2000, L))) for L in rng.integers(2, 9, 50)]
    ts = TrailSet.from_lists(trails)
    idx = rng.integers(0, 50, 80)
    sub = ts.subset(idx)
    for k, i in enumerate(idx):
        p, t = sub.trail(k)
        assert np.array_equal(p, trails[i][0]) and np.array_equal(t, trails[i][1])
    assert np.array_equal(ts.pos[:len(trails[0][0])], np.arange(len(trails[0][0])))


if __name__ == "__main__":
    test_gdp_matches_pld()
    test_marginal_budget_split()
    test_vote_sensitivity()
    test_release_independent_of_dataset_size()
    test_transition_sensitivity()
    test_gdp_roundtrip()
    test_trailset_subset_roundtrip()
    print("all tests passed")
