import copy
import json
import sys
from pathlib import Path

import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "baselines" / "dplora"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import hybrid as HY
from test_workload import _Cat, _edge_trails
from trajpe import workload as W
from trajpe.pe import PEConfig
from trajpe.trails import TrailSet

V4 = ("hcell6", "tctrans6", "gram6")


def test_v4_families_are_opt_in():
    assert all(f in W.FAMILIES for f in V4)
    assert not any(f in W.DEFAULT_FAMILIES.split(",") for f in V4)
    c = PEConfig()
    assert c.work_families == W.DEFAULT_FAMILIES and c.work_frac == 0.0
    assert not any(f in c.calib_families.split(",") + c.fuse_families.split(",") for f in V4)


def _cat_grid():
    rng = np.random.default_rng(0)
    cat = _Cat(rng, n=36)
    j = np.arange(36)
    cat.xy = np.stack([(j % 6 + 0.5) * 5.0, (j // 6 + 0.5) * 5.0], 1)
    return cat


def test_v4_family_bins_by_hand():
    cat = _cat_grid()
    G = 36
    ts = TrailSet.from_lists([(np.array([3, 3, 3]), np.array([60, 70, 80])),
                              (np.array([1, 1, 2, 7]), np.array([400, 430, 800, 1500])),
                              (np.array([5, 9]), np.array([25 * 60, 26 * 60]))])
    r, c, v = W.rows(ts, cat, "gram6")
    got = sorted(zip(r.tolist(), c.tolist(), np.round(v, 12).tolist()))
    want = sorted([(0, 3 * G + 3, 1.0),
                   (1, 1 * G + 2, 1 / 3), (1, 2 * G + 7, 1 / 3), (1, G * G + 1 * G * G + 2 * G + 7, 1 / 3),
                   (2, 5 * G + 9, 1.0)])
    assert got == [(a, b, round(x, 12)) for a, b, x in want]
    r, c, v = W.rows(ts, cat, "hcell6")
    assert c.tolist() == [1 * 36 + 3, 1 * 36 + 3, 1 * 36 + 3, 6 * 36 + 1, 7 * 36 + 1, 13 * 36 + 2, 1 * 36 + 7,
                          1 * 36 + 5, 2 * 36 + 9]
    r, c, v = W.rows(ts, cat, "tctrans6")
    assert c.tolist() == [0 * G * G + 3 * G + 3, 0 * G * G + 3 * G + 3, 1 * G * G + 1 * G + 1,
                          1 * G * G + 1 * G + 2, 2 * G * G + 2 * G + 7, 0 * G * G + 5 * G + 9]
    for f in V4:
        assert W.n_bins(f, cat) == {"hcell6": 864, "tctrans6": 5184, "gram6": 1296 + 46656}[f]


def test_v4_families_mass_and_sensitivity():
    rng = np.random.default_rng(3)
    cat = _Cat(rng)
    D = _edge_trails(rng, len(cat))
    n = len(D)
    for f in V4:
        r, c, v = W.rows(D, cat, f)
        assert np.all(v >= 0) and np.all((c >= 0) & (c < W.n_bins(f, cat))), f
        assert np.allclose(np.bincount(r, weights=v, minlength=n), 1.0, atol=1e-12), f
    full = W.dp_workload(D, np.ones(n), cat, 0.0, rng, V4)
    for i in range(n):
        keep = np.setdiff1d(np.arange(n), [i])
        part = W.dp_workload(D.subset(keep), np.ones(n - 1), cat, 0.0, rng, V4)
        for f in V4:
            d = full[f] - part[f]
            assert np.all(d >= -1e-12) and abs(d.sum() - 1.0) < 1e-9 and np.linalg.norm(d) <= 1 + 1e-9, (f, i)


def test_v4_families_pooled_in_families():
    rng = np.random.default_rng(4)
    cat = _Cat(rng)
    D = _edge_trails(rng, len(cat))
    work = W.dp_workload(D, np.ones(len(D)), cat, 0.5, rng, V4)
    dp = W.DPOutputs(work=work, sigma_w=0.5, n_hat=float(len(D)), sigma_m=1.0)
    fams = {f.name: f for f in W.families(D, cat, V4, dp)}
    assert fams["hcell6"].Phi.shape[1] == 864
    for f in ("tctrans6", "gram6"):
        assert fams[f].Phi.shape[1] < W.n_bins(f, cat)
    for f in fams.values():
        assert np.allclose(np.asarray(f.Phi.sum(1)).ravel(), 1.0)


_CAL = {}


def _calib():
    if not _CAL:
        _CAL.update(HY.calibrate(1.0, 0.3, 2048, 3.0))
    return dict(_CAL)


def _raises_exit(fn, *a):
    try:
        fn(*a)
    except SystemExit:
        return True
    return False


def _setup(tmp, cal):
    runs, results = tmp / "runs", tmp / "results"
    work = tmp / "work"
    work.mkdir(parents=True)
    (work / "calib.json").write_text(json.dumps(cal))
    tj = {"noise_multiplier": cal["sigma_sgd"], "sample_rate": cal["sample_rate"], "steps": cal["steps"],
          "steps_run": cal["steps"], "eps_target": 1.0, "delta": 1e-6, "complete": True, "clip": 1.0,
          "model": "m", "eps_prv": 0.99, "eps_rdp": 1.06, "sampling": "Poisson"}
    (work / "train.json").write_text(json.dumps(tj))
    mu = cal["mu_pe_run"]
    priv = {"eps": 0.8, "delta": 1e-6, "mu": mu, "marg_sha256": "abc", "eps_check": 0.8}
    chk = HY.check(cal, mu)
    src = {"dataset": "d", "api": "sim", "privacy": priv, "pe": {"seed": 0}, "embed": {},
           "hybrid": {"eps_total": 1.0, "delta": 1e-6, "eps_pe_arg": 0.8, "calib": cal,
                      "generator": {"work": str(work)}, "pool": {"n_texts": 10}, "check": chk, "guarantee": "g"}}
    for tag, res in (("run", src), ("rundiv", {"dataset": "d", "api": "sim", "privacy": copy.deepcopy(priv),
                                               "pe": {"seed": 0}, "embed": {}, "tag": "rundiv", "source": "run",
                                               "postprocess": {"marginals_check": {"marg_sha256": "abc"}}})):
        (runs / "d" / tag).mkdir(parents=True)
        (runs / "d" / tag / "result.json").write_text(json.dumps(res))
    (results / "pe" / "d").mkdir(parents=True)
    (results / "pe" / "d" / "rundiv.json").write_text("{}")
    return runs, results, work


def test_sidecar_writes_full_accounting():
    calib = _calib()
    runs, results, _ = _setup(Path(tempfile.mkdtemp(prefix="v4sc_")), calib)
    b = HY.sidecar("d", "rundiv", runs, results)
    res = json.loads((runs / "d" / "rundiv" / "result.json").read_text())
    assert res["hybrid"] == json.loads(json.dumps(b))
    assert json.loads((results / "pe" / "d" / "rundiv.json").read_text())["hybrid"] == res["hybrid"]
    h = res["hybrid"]
    assert h["dp_sgd"]["noise_multiplier"] == calib["sigma_sgd"] and h["dp_sgd"]["steps"] == calib["steps"]
    assert h["dp_sgd"]["sample_rate"] == calib["sample_rate"] and h["pe"]["mu_pe_spent"] == calib["mu_pe_run"]
    e = HY.composed_eps(calib["sigma_sgd"], calib["sample_rate"], calib["steps"], calib["mu_pe_run"], 1e-6)
    assert h["eps_composed_pld"] == e and e <= 1.0 and h["eps_total"] == 1.0
    assert h["dp_sgd"]["eps_sgd_only_pld"] < e and h["pe"]["eps_pe_only_pld"] < e
    assert res["privacy"]["mu"] == calib["mu_pe_run"]
    HY.sidecar("d", "rundiv", runs, results)
    assert json.loads((runs / "d" / "rundiv" / "result.json").read_text())["hybrid"] == h


def test_sidecar_refuses_non_hybrid_or_mismatch():
    for tamper in ("privacy", "diag", "train", "digest"):
        _refuses(tamper)


def _refuses(tamper):
    calib = _calib()
    runs, results, work = _setup(Path(tempfile.mkdtemp(prefix="v4sc_")), calib)
    p = runs / "d" / "rundiv" / "result.json"
    s = runs / "d" / "run" / "result.json"
    if tamper == "privacy":
        r = json.loads(p.read_text()); r["privacy"]["mu"] = 0.5; p.write_text(json.dumps(r))
    elif tamper == "diag":
        r = json.loads(s.read_text()); r["hybrid"] = {"diagnostic": "x"}; s.write_text(json.dumps(r))
    elif tamper == "train":
        t = json.loads((work / "train.json").read_text()); t["steps"] += 1
        (work / "train.json").write_text(json.dumps(t))
    else:
        r = json.loads(p.read_text()); r["postprocess"]["marginals_check"]["marg_sha256"] = "x"
        p.write_text(json.dumps(r))
    assert _raises_exit(HY.sidecar, "d", "rundiv", runs, results), tamper
    assert "hybrid" not in json.loads(p.read_text())


if __name__ == "__main__":
    test_v4_families_are_opt_in()
    test_v4_family_bins_by_hand()
    test_v4_families_mass_and_sensitivity()
    test_v4_families_pooled_in_families()
    test_sidecar_writes_full_accounting()
    test_sidecar_refuses_non_hybrid_or_mismatch()
    print("all v4 tests passed")
