import os as _tp_os
def _tp_root():
    p = _tp_os.path.dirname(_tp_os.path.abspath(__file__))
    while not _tp_os.path.isfile(_tp_os.path.join(p, "trajpe", "__init__.py")) and _tp_os.path.dirname(p) != p:
        p = _tp_os.path.dirname(p)
    return p
_TP_ROOT = _tp_os.environ.get("TRAJPE_ROOT", _tp_root())
_TP_WORK = _tp_os.environ.get("TRAJPE_WORK", _tp_os.path.join(_TP_ROOT, "work"))
_TP_PY = _tp_os.environ.get("TRAJPE_PYTHON", "python")
_TP_PY_DPLORA = _tp_os.environ.get("TRAJPE_PYTHON_DPLORA", _TP_PY)
_TP_PY_AIMSEQ = _tp_os.environ.get("TRAJPE_PYTHON_AIMSEQ", _TP_PY)
_TP_PY_GRID = _tp_os.environ.get("TRAJPE_PYTHON_GRID", _TP_PY)
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "baselines" / "dplora"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import hybrid as HY
from trajpe import ablations as AB
from trajpe.dp import mu_for


def test_apportion_exact_and_proportional():
    c = AB.apportion(np.array([3.0, -2.0, 1.0, 0.0, 6.0]), 100)
    assert c.sum() == 100 and c[1] == 0 and c[3] == 0
    assert np.all(np.abs(c - np.array([30, 0, 10, 0, 60])) <= 1)
    assert AB.apportion(np.array([-1.0, -5.0]), 7).sum() == 7


def test_yu_resample_counts_and_replacement():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 20, 2000)
    noisy = rng.normal(50, 40, 20)
    sizes = np.bincount(labels, minlength=20)
    counts = AB.yu_counts(noisy, sizes, 500)
    assert counts.sum() == 500 and np.all(counts[noisy <= 0] == 0)
    idx, short, extra = AB.yu_resample(labels, counts, rng)
    assert len(idx) == 500 and short == 0 and extra == 0 and len(np.unique(idx)) == 500
    assert np.array_equal(np.bincount(labels[idx], minlength=20), counts)
    small = np.r_[np.zeros(3, int), np.ones(500, int)]
    idx, short, extra = AB.yu_resample(small, np.array([10, 5]), rng)
    assert short == 1 and extra == 7 and len(idx) == 15
    assert set(np.flatnonzero(small == 0)) <= set(idx.tolist())


def test_kmeans_labels_are_nearest_centroids():
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.normal(m, 0.1, (200, 3)) for m in (0.0, 5.0, 10.0)]).astype(np.float32)
    cent, lab = AB.kmeans_centroids(x, 3, 25, seed=0)
    assert cent.shape == (3, 3) and len(np.unique(lab)) == 3
    assert np.array_equal(AB.nearest_centroid(x, cent), lab)
    cent2, lab2 = AB.kmeans_centroids(x, 3, 25, seed=0)
    assert np.array_equal(lab, lab2)


def test_path_votes_l2_sensitivity_one_and_top_n():
    rng = np.random.default_rng(2)
    e_syn = rng.normal(size=(300, 8)).astype(np.float32)
    e_priv = rng.normal(size=(40, 8)).astype(np.float32)
    for i in range(5):
        v = AB.path_votes(e_priv[i:i + 1], e_syn, 10)
        assert math.isclose(float(np.sqrt((v ** 2).sum())), 1.0, rel_tol=1e-9) and (v > 0).sum() == 10
    v = AB.path_votes(e_priv, e_syn, 10)
    assert math.isclose(v.sum(), 40 * math.sqrt(10), rel_tol=1e-9)
    noisy = v + rng.normal(0, 0.5, len(v))
    keep = AB.top_n(noisy, 50, rng)
    assert len(np.unique(keep)) == 50 and noisy[keep].min() >= np.sort(noisy)[-50]


def test_ab3_budget_split():
    mu = 0.7
    sm, sw = AB.ab3_sigmas(mu, 0.4, 0.245, 5, 14)
    assert math.isclose(5 / sm ** 2 + 14 / sw ** 2, mu ** 2, rel_tol=1e-12)
    assert math.isclose((5 / sm ** 2) / (14 / sw ** 2), 0.4 / 0.245, rel_tol=1e-12)
    assert math.isclose(AB.mu_of([(sm, 5), (sw, 14)]), mu, rel_tol=1e-12)


def _raises_exit(f, *a):
    try:
        f(*a)
    except SystemExit:
        return True
    return False


def _jakarta_generator(eps):
    w = Path(f"{_TP_WORK}/runs/hybrid/jakarta/_work/h_f0.3_eps{eps}_s0")
    if not (w / "calib.json").exists():
        print(f"skip: Jakarta V0.3 generator eps {eps} not on this machine")
        return None, None
    return json.loads((w / "calib.json").read_text()), json.loads((w / "train.json").read_text())


def test_accounting_composes_within_budget():
    for eps in (1, 4):
        _accounting(eps)


def _accounting(eps):
    cal, tj = _jakarta_generator(eps)
    if cal is None:
        return
    mu = cal["mu_pe_run"]
    a0 = AB.accounting(cal, tj, [], HY)
    assert a0["mu_pe_spent"] == 0 and a0["eps_release"] == a0["eps_sgd_only_pld"] < eps
    a1 = AB.accounting(cal, tj, [(1.0 / mu, 1)], HY)
    assert a1["eps_release"] <= eps and abs(a1["eps_release"] - cal["eps_composed_pld"]) < 1e-6
    assert abs(a1["independent_check"]["eps_pld_per_mechanism"] - a1["eps_release"]) < 1e-3
    sm, sw = AB.ab3_sigmas(mu, 0.4, 0.245, 5, 14)
    a3 = AB.accounting(cal, tj, [(sm, 5), (sw, 14)], HY)
    assert a3["eps_release"] <= eps and abs(a3["eps_release"] - a1["eps_release"]) < 1e-6
    assert a3["independent_check"]["eps_pld_per_mechanism"] <= eps
    assert _raises_exit(AB.accounting, cal, tj, [(0.9 / mu, 1)], HY)
    bad = dict(tj, noise_multiplier=tj["noise_multiplier"] * 0.9)
    assert _raises_exit(AB.accounting, cal, bad, [], HY)


def _toy_catalog(rng, n=400):
    from trajpe import catalog as C
    from trajpe.public_catalog import ROOTS
    cats = [f"cat{i}" for i in range(8)]
    pois = pd.DataFrame({"venue_id": [f"v{i}" for i in range(n)], "lat": rng.uniform(-6.3, -6.1, n),
                         "lon": rng.uniform(106.7, 106.9, n), "name": [f"n{i}" for i in range(n)],
                         "category": [cats[i % 8] for i in range(n)], "root": [ROOTS[i % 8 % 4] for i in range(n)]})
    meta = {"bbox": {"lat_min": -6.35, "lat_max": -6.05, "lon_min": 106.65, "lon_max": 106.95}, "tz": "Asia/Jakarta"}
    old = C.load_pois, C.load_meta
    C.load_pois, C.load_meta = (lambda name: pois), (lambda name: meta)
    try:
        return C.Catalog("toy")
    finally:
        C.load_pois, C.load_meta = old


def _toy_data(cat, rng, n_rec=600, n_priv=500):
    recs = []
    for _ in range(n_rec):
        L = int(rng.integers(2, 8))
        h0 = int(rng.integers(6, 15))
        recs.append((int(rng.integers(0, 7)), [{"time": f"{h0 + j:02d}:{int(rng.integers(0, 60)):02d}",
                                                "category": cat.categories[int(rng.integers(0, 8))],
                                                "lat": float(rng.uniform(-6.3, -6.1)), "lon": float(rng.uniform(106.7, 106.9))}
                                               for j in range(L)]))
    from trajpe.trails import TrailSet
    trails = []
    for _ in range(n_priv):
        L = int(rng.integers(2, 8))
        t0 = int(rng.integers(0, 10080 - 600))
        trails.append((rng.integers(0, len(cat), L), t0 + np.cumsum(rng.integers(5, 90, L))))
    return recs, TrailSet.from_lists(trails), np.arange(n_priv)


def test_pipelines_end_to_end_toy():
    rng = np.random.default_rng(3)
    cat = _toy_catalog(rng)
    recs, priv, users = _toy_data(cat, rng)
    pool, _ = AB.pool_from_texts(cat, recs, 10, 0.05, 1.0, seed=0)
    assert len(pool) == len(recs)
    pool2, _ = AB.pool_from_texts(cat, recs, 10, 0.05, 1.0, seed=0)
    assert np.array_equal(pool.poi, pool2.poi)
    for i in range(len(pool)):
        p, _ = pool.trail(i)
        assert np.all(cat.cat[p] >= 0)
    res = AB.run_ab012(priv, cat, pool, ["ab0", "ab1", "ab2"], mu_pe=0.7, seed=0, n_out=100, log=lambda x: None)
    for ab in ("ab0", "ab1", "ab2"):
        S, mechs, info = res[ab]
        assert len(S) == 100 and S.poi.max() < len(cat)
        assert mechs == ([] if ab == "ab0" else [(1.0 / 0.7, 1)])
    assert len(np.unique(res["ab2"][0].offsets)) > 1
    cfg = json.loads((ROOT / "configs" / "v4_large_eps.json").read_text())
    cfg["pool_n"] = len(recs)
    S, mechs, info = AB.run_ab3(priv, users, cat, recs, cfg, mu_pe=0.7, seed=0, n_out=100, log=lambda x: None)
    assert len(S) == 100 and S.poi.max() < len(cat)
    assert math.isclose(AB.mu_of(mechs), 0.7, rel_tol=1e-12) and [c for _, c in mechs] == [5, 14]
    S2, _, info2 = AB.run_ab3(priv, users, cat, recs, cfg, mu_pe=0.7, seed=0, n_out=100, log=lambda x: None)
    assert np.array_equal(S.poi, S2.poi) and info["work_sha256"] == info2["work_sha256"]






if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_") and callable(f):
            f()
            print("ok", name, flush=True)
    print("all ablation tests passed")
