from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import common
from accounting import budget, cdp_delta, cdp_rho
from encoding import LEN_MAX, POSITIONS, Decoder, PublicDomain, Spec

DELTA = 1e-6


def cks_delta_grid(rho: float, eps: float) -> float:
    a = np.exp(np.linspace(np.log(1.0001), np.log(1e5), 400000))
    logd = (a - 1) * (a * rho - eps) + a * np.log1p(-1 / a) - np.log(a - 1)
    return float(np.exp(logd.min()))


@pytest.mark.parametrize("eps", [0.5, 1.0, 2.0, 4.0, 8.0])
def test_budget_conversion(eps):
    b = budget(eps, DELTA, 0.1)
    assert math.isclose(b["rho_mech"] + b["rho_prior"], b["rho_total"], rel_tol=1e-12)
    assert math.isclose(b["eps_venue"], 0.05 * eps) and math.isclose(b["eps_cat"], 0.05 * eps)
    assert math.isclose(b["rho_prior"], (0.05 * eps) ** 2, rel_tol=1e-12)
    assert cdp_delta(b["rho_total"], eps) <= DELTA * (1 + 1e-9)
    assert cdp_delta(b["rho_total"] * 1.001, eps) > DELTA
    assert cks_delta_grid(b["rho_total"], eps) <= DELTA * 1.001
    assert b["rho_total"] + 2 * math.sqrt(b["rho_total"] * math.log(1 / DELTA)) >= eps


def test_reference_values():
    assert abs(cdp_rho(1.0, DELTA) - 0.02436) < 5e-5
    assert abs(cdp_rho(4.0, DELTA) - 0.31106) < 5e-5


def toy_dataset(n_trails=300, n_venues=400, seed=0, n_users=60) -> common.Dataset:
    rng = np.random.default_rng(seed)
    bbox = {"lat_min": 0.0, "lat_max": 0.2, "lon_min": 0.0, "lon_max": 0.2}
    roots = ["Dining and Drinking", "Retail", "Travel and Transportation", "Event"]
    leaves = {r: [f"{r[:4]}_{j}" for j in range(5)] for r in roots}
    root = rng.choice(roots, n_venues)
    cat = np.array([rng.choice(leaves[r]) for r in root])
    pois = pd.DataFrame({"venue_id": [f"v{i}" for i in range(n_venues)], "lat": rng.uniform(0, 0.2, n_venues),
                         "lon": rng.uniform(0, 0.2, n_venues), "category": cat, "root": root,
                         "name": "x", "closed": False})
    venue_cat, cats = pd.factorize(pois["category"].astype(str), sort=True)
    lengths = rng.choice(np.arange(2, LEN_MAX + 1), n_trails, p=np.r_[0.5, 0.2, np.full(17, 0.3 / 17)])
    checkin_venue = rng.integers(0, n_venues, lengths.sum())
    gaps = [rng.integers(1, 481, l - 1) for l in lengths]
    start_wh = rng.integers(0, 168, n_trails)
    off = np.concatenate([[0], np.cumsum(lengths)])
    lonlat = pois[["lon", "lat"]].to_numpy()
    trails = [lonlat[checkin_venue[off[i]:off[i + 1]]] for i in range(n_trails)]
    ds = common.Dataset("toy", bbox, trails, lengths, start_wh, gaps, pois, checkin_venue.astype(np.int64),
                        venue_cat.astype(np.int64), list(cats), {})
    ds.user_ids = rng.integers(0, n_users, n_trails).astype(str)
    ds.trail_ids = np.array([f"t{i}" for i in range(n_trails)])
    return ds


def drop_trail(ds: common.Dataset, j: int) -> common.Dataset:
    keep = np.r_[0:j, j + 1:len(ds.lengths)]
    off = np.concatenate([[0], np.cumsum(ds.lengths)])
    cv = np.concatenate([ds.checkin_venue[off[i]:off[i + 1]] for i in keep])
    return common.Dataset(ds.name, ds.bbox, [ds.trails[i] for i in keep], ds.lengths[keep], ds.start_wh[keep],
                          [ds.gaps[i] for i in keep], ds.pois, cv, ds.venue_cat, ds.categories, {})


STRATA = ((2, 2), (3, 3), (4, 20))


def stratum_view(ds: common.Dataset, st):
    keep = (ds.lengths >= st[0]) & (ds.lengths <= st[1])
    off = np.concatenate([[0], np.cumsum(ds.lengths)])
    idx = np.flatnonzero(keep)
    v = np.concatenate([ds.checkin_venue[off[i]:off[i + 1]] for i in idx]) if len(idx) else np.zeros(0, np.int64)
    g = np.concatenate([ds.gaps[i] for i in idx]) if len(idx) else np.zeros(0, np.int64)
    return v, ds.lengths[keep], ds.start_wh[keep], g, idx


def encode(dom, ds, st=(2, 20)):
    v, L, wh, g, _ = stratum_view(ds, st)
    return dom.encode(v, L, wh, g, st)


def marginal(rec, attrs, cl):
    idx = np.ravel_multi_index(tuple(rec[a] for a in cl), tuple(attrs[a] for a in cl))
    return np.bincount(idx, minlength=int(np.prod([attrs[a] for a in cl]))).astype(float)


def test_spec_strata_partition():
    with pytest.raises(ValueError):
        Spec(strata=((2, 3), (5, 20)))
    with pytest.raises(ValueError):
        Spec(strata=((2, 4), (4, 20)))
    sp = Spec(strata=STRATA)
    assert list(sp.stratum_of(np.array([2, 3, 4, 20]))) == [0, 1, 2, 2]


def test_allocate_sums_to_n_out():
    from run_aimseq import allocate
    for totals in ([46000.3, 15000.2, 13000.1], [1.0, -5.0, float("nan")], [0.0, 0.0]):
        n = allocate(50000, totals)
        assert n.sum() == 50000 and (n >= 0).all()


@pytest.mark.parametrize("k_sub,max_pos,time_split", [(0, 20, False), (3, 20, True), (3, 6, True)])
def test_add_remove_one_sensitivity(k_sub, max_pos, time_split):
    import mech as M
    ds = toy_dataset()
    spec = Spec(grid=4, od_grid=2, k_sub=k_sub, max_pos=max_pos, time_split=time_split, strata=STRATA)
    dom = PublicDomain(ds.pois, ds.bbox, spec)
    recs = {st: encode(dom, ds, st) for st in STRATA}
    for j in (0, 17, len(ds.lengths) - 1):
        ds2 = drop_trail(ds, j)
        dom2 = PublicDomain(ds2.pois, ds2.bbox, spec)
        hit = [st for st in STRATA if st[0] <= ds.lengths[j] <= st[1]]
        assert len(hit) == 1
        for st in STRATA:
            attrs = dom.attributes(st)
            assert dom2.attributes(st) == attrs
            rec, rec2 = recs[st], encode(dom2, ds2, st)
            row = np.flatnonzero(stratum_view(ds, st)[4] == j)
            for a in attrs:
                assert np.array_equal(rec2[a], np.delete(rec[a], row))
            cliques = [cl for cl, _ in M.build_workload(attrs, "seq")] + [(a,) for a in attrs]
            for cl in cliques:
                diff = marginal(rec, attrs, cl) - marginal(rec2, attrs, cl)
                expect = 1.0 if st == hit[0] else 0.0
                assert np.abs(diff).sum() == expect and np.sqrt((diff ** 2).sum()) == expect
        for d_ in (ds, ds2):
            w = np.repeat(1.0 / d_.lengths, d_.lengths)
            d_.h_pop = np.bincount(d_.checkin_venue, weights=w, minlength=len(ds.venue_cat))
            d_.h_cat = np.bincount(ds.venue_cat[d_.checkin_venue], weights=w, minlength=len(ds.categories))
        assert np.abs(ds.h_pop - ds2.h_pop).sum() <= 1 + 1e-9
        assert np.abs(ds.h_cat - ds2.h_cat).sum() <= 1 + 1e-9


def test_user_level_bounding():
    ds = toy_dataset(n_trails=400, n_users=50)
    kept = set(common.one_trail_per_user(ds.trail_ids, ds.user_ids, 0))
    assert len(kept) == len(set(ds.user_ids))
    for u in np.unique(ds.user_ids)[:10]:
        m = ds.user_ids != u
        kept2 = set(common.one_trail_per_user(ds.trail_ids[m], ds.user_ids[m], 0))
        assert kept2 <= kept and len(kept - kept2) <= 1


def toy_mbi(n=3000, seed=0):
    import mech as M
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 4, n)
    b = (a + rng.integers(0, 2, n)) % 5
    c = rng.integers(0, 3, n)
    attrs = {"a": 4, "b": 5, "c": 3}
    return M.make_dataset({"a": a, "b": b, "c": c}, attrs), attrs


def test_aim_ledger_matches_budget():
    import mech as M
    data, attrs = toy_mbi()
    rho = 0.05
    W = [(("a", "b"), 1.0), (("b", "c"), 1.0)]
    sample, total, ledger, info = M.run_aim(data, W, rho, 1, rounds=None, max_model_size=1.0, max_iters=100)
    assert ledger.rho <= rho * (1 + 1e-9)
    assert ledger.rho >= rho * (1 - 1e-6)
    assert len(ledger.gauss) == 3 + len(ledger.em)
    assert abs(total - 3000) < 300
    s1, s2 = sample(777, 2), sample(777, 2)
    assert len(s1.to_dict()["a"]) == 777
    assert all(np.array_equal(s1.to_dict()[a], s2.to_dict()[a]) for a in attrs)
    with pytest.raises(ValueError):
        M.run_aim(data, W, rho, 1, rounds=2)


def test_aim_presence_init_accounting():
    import mech as M
    ds = toy_dataset()
    dom = PublicDomain(ds.pois, ds.bbox, Spec(grid=4, od_grid=2, k_sub=2, max_pos=4, time_split=True))
    attrs = dom.attributes()
    data = M.make_dataset(encode(dom, ds), attrs)
    init = M.presence_init(attrs)
    assert len(init) == len(attrs) and set(attrs) <= {a for cl in init for a in cl}
    assert ("L", "r3") in init and ("r3", "d3") in init and ("d3", "b3") in init
    rho = 0.1
    W = M.build_workload(attrs, "seq")
    sample, total, ledger, info = M.run_aim(data, W, rho, 5, rounds=2 * len(attrs), max_model_size=2.0,
                                            max_iters=50, initial_cliques=init)
    assert rho * (1 - 1e-6) <= ledger.rho <= rho * (1 + 1e-9)
    assert sorted(s for s, _ in ledger.gauss[:len(attrs)]) == [ledger.gauss[0][0]] * len(attrs)
    with pytest.raises(ValueError):
        M.run_aim(data, W, rho, 5, rounds=2 * len(attrs), initial_cliques=init[:-1])


def test_fast_marginals_match_official():
    import itertools
    import mech as M
    from mbi import LinearMeasurement, estimation
    rng = np.random.default_rng(3)
    n = 4000
    cols = {"a": rng.integers(0, 3, n)}
    cols["b"] = (cols["a"] + rng.integers(0, 2, n)) % 4
    cols["c"] = (cols["b"] + rng.integers(0, 2, n)) % 3
    cols["d"] = (cols["c"] * 2 + rng.integers(0, 3, n)) % 5
    cols["e"] = (cols["d"] + cols["a"] + rng.integers(0, 2, n)) % 2
    cols["f"] = rng.integers(0, 4, n)
    attrs = {"a": 3, "b": 4, "c": 3, "d": 5, "e": 2, "f": 4}
    data = M.make_dataset(cols, attrs)
    cls = [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", "a"), ("f",)]
    ms = [LinearMeasurement(np.asarray(data.project(cl).datavector()) + rng.normal(0, 5, data.domain.size(cl)),
                            cl, 5.0) for cl in cls]
    model = estimation.MirrorDescent().estimate(data.domain, ms, iters=300)
    fm = M.FastMarginals(model)
    for cl in list(itertools.combinations(attrs, 2)) + [(a,) for a in attrs] + [("d", "a"), ("f", "c")]:
        ref = np.asarray(model.project(cl).datavector())
        got = fm.marginal(cl)
        assert got is not None and np.allclose(got, ref, rtol=1e-7, atol=1e-7 * n), cl


def test_mst_ledger_matches_budget():
    import mech as M
    data, attrs = toy_mbi()
    rho = 0.05
    sample, total, ledger, info = M.run_mst(data, rho, 3)
    assert abs(ledger.rho - rho) <= 1e-9 * rho
    assert len(sample(500, 4).to_dict()["a"]) == 500


@pytest.mark.parametrize("max_pos,strata", [(20, ((2, 20),)), (6, ((2, 20),)), (10, STRATA)])
def test_decode_toy(max_pos, strata):
    ds = toy_dataset()
    dom = PublicDomain(ds.pois, ds.bbox, Spec(grid=4, od_grid=2, k_sub=3, max_pos=max_pos, time_split=True,
                                              strata=strata))
    rng = np.random.default_rng(0)
    for st in strata:
        rec = encode(dom, ds, st)
        attrs = dom.attributes(st)
        fake = {a: rng.integers(0, attrs[a], 500) for a in attrs}
        for recs in (rec, fake):
            vidx, L, tmin, stt = Decoder(dom, None).decode(recs, np.random.default_rng(1), 1, st)
            assert len(L) == len(next(iter(recs.values()))) and L.min() >= st[0] and L.max() <= st[1]
            assert len(vidx) == L.sum() == len(tmin) and vidx.min() >= 0 and vidx.max() < dom.n_venues
            off = np.concatenate([[0], np.cumsum(L)])
            g = np.diff(tmin)
            inner = np.ones(len(tmin) - 1, bool)
            inner[off[1:-1] - 1] = False
            assert (g[inner] >= 1).all() and (g[inner] <= 480).all()
            assert (tmin[off[:-1]] < 7 * 24 * 60).all()
        vidx, L, tmin, stt = Decoder(dom, None).decode(rec, np.random.default_rng(2), 1, st)
        off = np.concatenate([[0], np.cumsum(L)])
        rec2 = dom.encode(vidx, L, (tmin[off[:-1]] // 60) % 168,
                          np.concatenate([np.diff(tmin[off[i]:off[i + 1]]) for i in range(len(L))]), st)
        for a in [x for x in ("L", "dow", "hod") if x in attrs] + [f"g{k}" for k in dom.spec.positions(st)]:
            assert np.array_equal(rec2[a], rec[a]), a
        relaxed_hops = sum(v for k, v in stt.items() if k.startswith(("root+", "any+", "nearest")))
        relaxed_start = stt.get("start_root", 0) + stt.get("start_any", 0) + stt.get("start_nearest", 0)
        cls_mis = int((rec2["r1"] != rec["r1"]).sum() + ((rec2["r1"] == rec["r1"]) & (rec2["s1"] != rec["s1"])).sum())
        assert cls_mis <= relaxed_start
        hop_mis = sum(int(((rec2[f"r{k}"] != rec[f"r{k}"]) | (rec2[f"s{k}"] != rec[f"s{k}"])).sum())
                      for k in dom.spec.positions(st))
        assert hop_mis <= relaxed_hops


def test_official_workload_matches_aim_main():
    import itertools
    import aim as official
    import mech as M
    params = official.default_params()
    assert params["degree"] == 2 and params["max_cells"] == M.OFFICIAL_MAX_CELLS and params["num_marginals"] is None
    assert params["max_model_size"] == 80 and params["max_iters"] == 1000
    attrs = {"L": 19, "wh": 168, "sc": 256, "ec": 64, "r1": 11, "s1": 17, "d2": 14}
    data = M.make_dataset({a: np.zeros(3, np.int64) for a in attrs}, attrs)
    ref = [cl for cl in itertools.combinations(data.domain, 2) if data.domain.size(cl) <= params["max_cells"]]
    W = M.build_workload(attrs, "official2")
    assert [cl for cl, _ in W] == ref and all(w == 1.0 for _, w in W)
    excl = {("wh", "sc"), ("wh", "ec"), ("sc", "ec")}
    assert excl.isdisjoint({cl for cl, _ in W}) and len(W) == 21 - 3


def test_v2_config_and_sensitivity():
    import json
    import mech as M
    import run_aimseq as R
    cfg = json.loads((HERE / "frozen_v2.json").read_text())["defaults"]
    assert cfg["workload"] == "official2" and cfg["rounds"] == 0 and cfg["max_model_size"] == 80.0
    assert cfg["max_iters"] == 1000 and cfg["init"] == "oneway" and cfg["strata"] == "2-20" and cfg["tilt"] == 0
    assert cfg["prior_frac"] == 0.1 and cfg["mech"] == "aim"
    assert R.frozen_defaults(4.0, HERE / "frozen_v2.json") == cfg
    ds = toy_dataset()
    st = R.parse_strata(cfg["strata"])[0]
    spec = Spec(grid=cfg["grid"], od_grid=cfg["od_grid"], k_sub=cfg["k_sub"], max_pos=cfg["max_pos"],
                time_split=bool(cfg["time_split"]), strata=(st,))
    dom = PublicDomain(ds.pois, ds.bbox, spec)
    attrs = dom.attributes(st)
    assert len(attrs) == 51 and attrs["wh"] == 168 and attrs["sc"] == 256 and attrs["ec"] == 64
    W = M.build_workload(attrs, cfg["workload"])
    rec = encode(dom, ds, st)
    for j in (0, 33, len(ds.lengths) - 1):
        ds2 = drop_trail(ds, j)
        dom2 = PublicDomain(ds2.pois, ds2.bbox, spec)
        assert dom2.attributes(st) == attrs
        rec2 = encode(dom2, ds2, st)
        for a in attrs:
            assert np.array_equal(rec2[a], np.delete(rec[a], j))
        for cl in [cl for cl, _ in W] + [(a,) for a in attrs]:
            diff = marginal(rec, attrs, cl) - marginal(rec2, attrs, cl)
            assert np.abs(diff).sum() == 1.0 and np.sqrt((diff ** 2).sum()) == 1.0


def test_aim_official2_ledger():
    import mech as M
    ds = toy_dataset()
    dom = PublicDomain(ds.pois, ds.bbox, Spec(grid=4, od_grid=2, k_sub=2, max_pos=3, time_split=False))
    attrs = dom.attributes()
    data = M.make_dataset(encode(dom, ds), attrs)
    rho = 0.2
    W = M.build_workload(attrs, "official2")
    sample, total, ledger, info = M.run_aim(data, W, rho, 7, rounds=None, max_model_size=1.0, max_iters=50)
    assert info["rounds_param"] == 16 * len(attrs)
    assert rho * (1 - 1e-6) <= ledger.rho <= rho * (1 + 1e-9)
    assert len(sample(123, 1).to_dict()["L"]) == 123
