from __future__ import annotations
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

import copy
import hashlib
import json
import math
import random
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG.parent))
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(HERE))
import common
import pategail as pg
import pg_accounting as acc
import pg_official
import synth

import torch

torch.set_num_threads(2)
DELTA = 1e-6
EPS_GRID = (1.0, 2.0, 4.0, 8.0)


def test_frozen_constants_match_spec():
    h = pg.H
    want = {"K": 8221, "rounds": 50, "beta": 0.01, "delta_w": 0.01, "train_iter": 100, "max_closest_rank": 200,
            "epr_alpha": 0.6, "time_scale": 48, "cell_deg": 0.01, "disc_iter": 10, "batch_size": 50,
            "ppo_updates": 3, "policy_iter": 1, "value_iter": 1, "clip_eps": 0.2, "entropy_weight": 0.001,
            "gamma": 0.99, "lam": 0.95, "lr": 1e-3, "weight_decay_policy_value": 1e-5, "clip_grad": 10.0,
            "venue_floor": 0.6, "venue_fp": 100.0, "start_cell_fp": 100.0, "n_out": 50000, "seed": 0,
            "traj_length_special_case": 20, "slot_minutes": 30}
    for k, v in want.items():
        assert h[k] == v, (k, h[k], v)
    assert h["teacher"] | {} == {**h["teacher"], "loc_emb": 8, "tim_emb": 8, "pre_pos_emb": 8, "stay_emb": 4,
                                 "act_emb": 64, "hidden": 64, "mlp_in": 92}
    assert (h["policy"]["loc_emb"], h["policy"]["tim_emb"], h["policy"]["pre_pos_emb"], h["policy"]["stay_emb"],
            h["policy"]["hidden"], h["policy"]["heads"]) == (32, 8, 8, 8, 64, 8)
    p = pg.CFG["privacy"]
    assert (p["delta"], p["aux_frac"], p["n_aux_hists"], p["pilot_share"]) == (1e-6, 0.1, 6, 0.1)
    assert pg.CFG["owner_decisions"]["D-K"]["K"] == 8221 and pg.CFG["owner_decisions"]["D-S"]["run_pg_uniform"]
    assert pg.CFG["official"]["commit"] == pg_official.COMMIT
    assert pg.K_FROZEN == 8221 and pg.ROUNDS == 50 and pg.LEN_MAX == 20


def test_official_sources_pinned_and_dims_as_run():
    for rel in pg_official.SHA256:
        pg_official.source(rel)
    net = pg_official.net()
    d = net.Discriminator(50, 48, 32, 8, 8, 8, 4, 64)
    assert d.loc_embedding.embedding_dim == 8 and d.tim_embedding.embedding_dim == 8
    assert d.pre_pos_count_embedding.embedding_dim == 8 and d.stay_time_embedding.embedding_dim == 4
    assert d.act_embedding.embedding_dim == 64 and d.mlp_layer[0].in_features == 92
    n_par = sum(p.numel() for p in d.parameters())
    assert n_par == 15553 + 8 * 50


def _cdp_rho_reimpl(eps, delta):
    def dlt(rho):
        a = np.exp(np.linspace(np.log(1.0001), np.log(1e5), 200001))
        logd = (a - 1) * (a * rho - eps) + a * np.log1p(-1 / a) - np.log(a - 1)
        return float(np.exp(logd.min()))
    lo, hi = 0.0, eps + 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if dlt(mid) <= delta else (lo, mid)
    return lo


def test_cdp_rho_official_matches_aimseq_and_reimplementation():
    sys.path.insert(0, str(PKG.parent / "aimseq"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("aimseq_accounting", PKG.parent / "aimseq" / "accounting.py")
    am = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(am)
    for eps in EPS_GRID:
        r = acc.cdp_rho(eps, DELTA)
        assert r == am.cdp_rho(eps, DELTA)
        assert math.isclose(r, _cdp_rho_reimpl(eps, DELTA), rel_tol=1e-6)
        assert acc.cdp_delta(r, eps) <= DELTA


def test_calibration_table_spec_4_4():
    want = {1: (0.024356, 0.01667, 0.02352, 4.70e-4, (0.0591, 0.0642), (0.125, 0.136)),
            2: (0.088153, 0.03333, 0.08482, 1.70e-3, (0.0311, 0.0338), (0.0660, 0.0717)),
            4: (0.311065, 0.06667, 0.29773, 5.96e-3, (0.0166, 0.0181), (0.0353, 0.0383)),
            8: (1.052358, 0.13333, 0.99902, 2.00e-2, (0.00907, 0.00985), (0.0192, 0.0209))}
    K = 8221
    for eps, (rt, eh, rp, rr, bm, bp) in want.items():
        b = acc.calibrate(float(eps), K, 50)
        assert math.isclose(b["rho_total"], rt, rel_tol=2e-5)
        assert math.isclose(b["eps_h"], eh, rel_tol=1e-3) and math.isclose(b["rho_aux"], eps ** 2 / 1200)
        assert math.isclose(b["rho_pate"], rp, rel_tol=2e-4)
        rho_r = b["rho_pate"] / 50
        assert math.isclose(rho_r, rr, rel_tol=6e-3)
        main = [(1 / K) / math.sqrt(0.9 * rho_r / n) for n in (100, 118)]
        pil = [(1 / K) / acc.pilot_eps(rho_r, n, 0.1) for n in (100, 118)]
        for got, w in zip(main + pil, bm + bp):
            assert math.isclose(got, w, rel_tol=6e-3), (eps, got, w)


def _simulate_ledger(eps, arm, rng, K=8221, rounds=50):
    b = acc.calibrate(eps, K, rounds)
    rho_r = b["rho_pate"] / rounds
    led = acc.Ledger(b["rho_total"], b["rho_aux"])
    events = [(b["eps_h"], 6)]
    per_round = []
    for r in range(1, rounds + 1):
        n_r = int(rng.integers(100, 119))
        rec = {"round": r, "n_r": n_r}
        if arm == "main":
            e_p = acc.pilot_eps(rho_r, n_r, 0.1)
            led.charge("M7", r, np.full(n_r, e_p))
            events.append((e_p, n_r))
            pilot = rng.uniform(-0.5, 1.5, n_r) * (rng.random() < 0.5) + rng.normal(0.5, 0.3, n_r)
            eq = acc.main_eps(acc.weights(pilot, 0.01), rho_r, 0.1)
            rec["eps_p"] = e_p
        else:
            eq = acc.uniform_eps(rho_r, n_r)
            rec["eps_p"] = None
        led.charge("M8", r, eq)
        led.charge("M9", r, eq)
        events += [(e, 2) for e in eq]
        rec["sum_eps_q_sq"] = math.fsum((eq * eq).tolist())
        per_round.append(rec)
    return b, led, events, per_round


def test_accounting_composed_eps_le_target():
    rng = np.random.default_rng(0)
    for eps in EPS_GRID:
        for arm in ("main", "uniform"):
            b, led, events, per_round = _simulate_ledger(eps, arm, rng)
            assert led.rho <= b["rho_total"] * (1 + 1e-12)
            assert math.isclose(led.rho, b["rho_total"], rel_tol=1e-9)
            assert math.isclose(acc.rho_from_sidecar({"per_round": per_round}, b["rho_aux"]), led.rho, rel_tol=1e-12)
            assert acc.cdp_eps(led.rho, DELTA) <= eps + 1e-9
            e_dpa = acc.dpacc_eps_from_zcdp(led.rho, DELTA)
            assert e_dpa <= eps + 1e-9, (eps, arm, e_dpa)
            e_rdp = acc.exact_laplace_rdp_eps(events, DELTA)
            assert e_rdp <= eps, (eps, arm, e_rdp)
            if arm == "uniform":
                ev = {}
                for e, c in events:
                    ev[e] = ev.get(e, 0) + c
                e_pld = acc.pld_eps(list(ev.items()), DELTA)
                assert e_pld <= eps, (eps, e_pld)


def test_ledger_refuses_overspend():
    b = acc.calibrate(1.0, 8221, 50)
    led = acc.Ledger(b["rho_total"], b["rho_aux"])
    led.charge("M8", 1, np.full(100, math.sqrt(2 * b["rho_pate"] / 100)))
    assert math.isclose(led.rho, b["rho_total"], rel_tol=1e-12)
    try:
        led.charge("M9", 1, np.array([1e-4]))
    except acc.BudgetExceeded:
        return
    raise AssertionError("ledger accepted a release beyond rho_total")


def test_weights_and_allocation():
    rho_r = 4.7e-4
    p = np.array([-3.0, 0.0, 0.3, 0.7, 1.0, 9.0])
    w = acc.weights(p, 0.01)
    assert np.allclose(w, [1.01, 1.01, 0.71, 0.31, 0.01, 0.01])
    e = acc.main_eps(w, rho_r, 0.1)
    assert math.isclose(float(np.sum(e * e)), 0.9 * rho_r, rel_tol=1e-12)
    assert np.all(np.diff(e[[1, 2, 3, 4]]) < 0)
    u = acc.uniform_eps(rho_r, 110)
    assert math.isclose(float(np.sum(u * u)), rho_r, rel_tol=1e-12)
    assert math.isclose(110 * acc.pilot_eps(rho_r, 110, 0.1) ** 2 / 2, 0.1 * rho_r, rel_tol=1e-12)


def _mean_var(x):
    return float(np.mean(x)), float(np.var(x))


def test_sensitivity_mean_and_variance():
    rng = np.random.default_rng(1)
    for K in (2, 3, 7, 83, 1000, 8221):
        bound_m, bound_v = 1.0 / K, (K - 1) / K ** 2
        vecs = [rng.random(K), np.zeros(K), np.ones(K), (rng.random(K) < 0.5).astype(float), np.full(K, 0.5)]
        for x in vecs:
            m0, v0 = _mean_var(x)
            for j in rng.integers(0, K, 5):
                for new in (0.0, 1.0, rng.random()):
                    y = x.copy()
                    y[j] = new
                    m1, v1 = _mean_var(y)
                    assert abs(m1 - m0) <= bound_m + 1e-12
                    assert abs(v1 - v0) <= bound_v + 1e-12
        y = np.zeros(K)
        y[0] = 1.0
        assert math.isclose(_mean_var(y)[1] - _mean_var(np.zeros(K))[1], bound_v, rel_tol=1e-9)


def _raw_hists(ds, geo_cells_first, n_cells):
    h_start = np.bincount(ds.start_wh, minlength=common.N_WEEK_HOURS).astype(float)
    b = common.gap_bin(np.concatenate(ds.gaps))
    w = np.repeat(1.0 / (ds.lengths - 1), ds.lengths - 1)
    h_gap = np.bincount(b, weights=w, minlength=common.N_GAP_BINS)
    h_len = np.bincount(np.clip(ds.lengths, 2, 20) - 2, minlength=19).astype(float)
    wv = np.repeat(1.0 / ds.lengths, ds.lengths)
    pop = np.bincount(ds.checkin_venue, weights=wv, minlength=len(ds.venue_cat))
    cat = np.bincount(ds.venue_cat[ds.checkin_venue], weights=wv, minlength=len(ds.categories))
    start = np.bincount(geo_cells_first, minlength=n_cells).astype(float)
    return [h_start, h_gap, h_len, pop, cat, start]


class _RecordingRng:

    def __init__(self):
        self.calls = []

    def laplace(self, loc, scale, size):
        self.calls.append((float(scale), size if isinstance(size, int) else tuple(np.atleast_1d(size))))
        return np.zeros(size)


def test_sensitivity_aux_histograms_and_noise_scales():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        pois, train = synth.make_dataset(root, "sens", seed=3, n_trails=120)
        old = common.DATA
        common.DATA = root
        try:
            ds = common.load_dataset("sens")
            priv, geo = pg.load_private("sens", None, ds)
            base = _raw_hists(ds, priv.cells[priv.offsets[:-1]], geo.n)
            tids = train["trail_id"].unique()
            for drop in tids[[0, 7, 50, -1]]:
                sub = train[train["trail_id"] != drop]
                synth.write_dataset(root, "sens_m", pois, sub)
                ds2 = common.load_dataset("sens_m")
                priv2, geo2 = pg.load_private("sens_m", None, ds2)
                assert np.array_equal(geo2.cell_ij, geo.cell_ij)
                other = _raw_hists(ds2, priv2.cells[priv2.offsets[:-1]], geo2.n)
                for a, b in zip(base, other):
                    assert np.abs(a - b).sum() <= 1.0 + 1e-9
            eh = 1.0 / 60
            rec = _RecordingRng()
            common.AuxModel.fit_central(ds, eh, eh, eh, rec)
            common.VenuePrior.fit_central(ds, eh, eh, rec, 0.6, 100.0)
            pg.fit_start_cells(priv.cells[priv.offsets[:-1]], geo.n, eh, rec, 100.0)
            sizes = [c[1] for c in rec.calls]
            assert all(math.isclose(c[0], 1 / eh) for c in rec.calls)
            assert sizes == [(168,), (11,), (19,), len(ds.venue_cat), len(ds.categories), geo.n], sizes
        finally:
            common.DATA = old


def _toy_teacher_setup(K, n_loc, rng):
    n_pairs = 200
    pairs = {f: rng.integers(0, 4 if f == "act" else (n_loc if f == "pos" else 20), n_pairs) for f in pg.PAIR_FIELDS}
    return pairs


def test_partition_coupling_one_trail_changes_one_teacher():
    K, n_loc, rounds, B = 7, 30, 3, pg.BATCH
    rng = np.random.default_rng(5)
    n_trails = 25
    tids = np.array([f"trail{i}" for i in range(n_trails)])
    L = rng.integers(2, 6, n_trails)
    trail_pairs = [{f: rng.integers(0, 4 if f == "act" else (n_loc if f == "pos" else 20), l - 1)
                    for f in pg.PAIR_FIELDS} for l in L]
    bufs = [{f: rng.integers(0, 4 if f == "act" else (n_loc if f == "pos" else 20), 110) for f in pg.PAIR_FIELDS}
            for _ in range(rounds)]
    x = 11

    def run(keep):
        ids = tids[keep]
        teach = pg.teacher_of(ids, K)
        tp = np.concatenate([np.full(L[i] - 1, teach[j]) for j, i in enumerate(np.flatnonzero(keep))])
        pairs = {f: np.concatenate([trail_pairs[i][f] for i in np.flatnonzero(keep)]) for f in pg.PAIR_FIELDS}
        order = np.argsort(tp, kind="stable")
        pairs = {f: v[order] for f, v in pairs.items()}
        counts = np.bincount(tp, minlength=K)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        nonempty = counts > 0
        ss = pg.streams(0)
        t_rngs = [np.random.Generator(np.random.PCG64(s)) for s in ss["teachers"].spawn(K)]
        gen = torch.Generator().manual_seed(int(ss["teacher_init"].generate_state(1)[0]))
        ens = pg.TeacherEnsemble(K, n_loc, gen, torch.device("cpu"))
        means = []
        for r in range(rounds):
            buf = bufs[r]
            Dk = ens.predict(buf)
            Dk[~nonempty] = 0.5
            means.append(Dk.mean(0))
            real_idx = np.zeros((K, B), np.int64)
            gen_idx = np.tile(np.arange(B), (K, 1))
            for k in np.flatnonzero(nonempty):
                real_idx[k] = starts[k] + t_rngs[k].integers(0, counts[k], size=B)
                gen_idx[k] = t_rngs[k].choice(110, size=B, replace=False)
            real = tuple(torch.as_tensor(pairs[f][real_idx]) for f in pg.PAIR_FIELDS)
            genb = tuple(torch.as_tensor(buf[f][gen_idx]) for f in pg.PAIR_FIELDS)
            ens.train(real, genb, torch.as_tensor(nonempty))
        state = {n: p.detach().clone() for n, p in ens.p.items()}
        opt = {i: {k: v.clone() for k, v in s.items() if torch.is_tensor(v)} for i, s in
               enumerate(ens.opt.state.values())}
        return state, opt, means, pg.teacher_of(tids[x:x + 1], K)[0]

    full = np.ones(n_trails, bool)
    minus = full.copy()
    minus[x] = False
    s1, o1, m1, tx = run(full)
    s2, o2, m2, _ = run(minus)
    for k in range(K):
        same = all(torch.equal(s1[n][k], s2[n][k]) for n in s1) and \
            all(torch.equal(o1[i][f][k], o2[i][f][k]) for i in o1 for f in o1[i] if o1[i][f].dim() > 0)
        assert same == (k != tx), (k, tx, same)
    for a, b in zip(m1, m2):
        assert np.max(np.abs(a - b)) <= 1.0 / K + 1e-12


def test_ensemble_equals_official_discriminators():
    net = pg_official.net()
    K, n_loc, B = 5, 40, 50
    torch.manual_seed(3)
    mods = [net.Discriminator(n_loc, 48, 32, 8, 8, 8, 4, 64) for _ in range(K)]
    opts = [torch.optim.Adam(m.parameters(), lr=0.001) for m in mods]
    ens = pg.TeacherEnsemble(K, n_loc, torch.Generator().manual_seed(1), torch.device("cpu"))
    ens.load_official(mods)
    g = torch.Generator().manual_seed(4)

    def batch():
        return (torch.randint(0, n_loc, (K, B), generator=g), torch.randint(0, 48, (K, B), generator=g),
                torch.randint(0, 4, (K, B), generator=g), torch.randint(0, 20, (K, B), generator=g),
                torch.randint(0, 20, (K, B), generator=g))
    real, gen = batch(), batch()
    with torch.no_grad():
        mods[2].mlp_layer[6].weight.mul_(200.0)
        ens.p["W3"][2].mul_(200.0)
    bce = torch.nn.BCELoss()
    norms = []
    for _ in range(10):
        for k, (m, o) in enumerate(zip(mods, opts)):
            er = m.forward(*(x[k] for x in real))
            cr = m.forward(*(x[k] for x in gen))
            loss = (bce(er, torch.ones(B, 1)) + bce(cr, torch.zeros(B, 1))) / 2
            o.zero_grad()
            loss.backward()
            norms.append(float(torch.nn.utils.clip_grad_norm_(m.parameters(), 10)))
            o.step()
    assert max(norms) > 10 and min(norms) < 10
    ens.train(real, gen, torch.ones(K, dtype=torch.bool), n_iter=10)
    ref = pg.TeacherEnsemble(K, n_loc, torch.Generator().manual_seed(1), torch.device("cpu"))
    ref.load_official(mods)
    for n in ens.p:
        assert torch.allclose(ens.p[n], ref.p[n], rtol=1e-5, atol=1e-6), n
    out_e = ens.forward(*real)
    out_o = torch.stack([m.forward(*(x[k] for x in real)).squeeze(-1) for k, m in enumerate(mods)])
    assert torch.allclose(out_e, out_o, rtol=1e-5, atol=1e-6)


def test_masked_teacher_untouched():
    K, n_loc = 4, 20
    ens = pg.TeacherEnsemble(K, n_loc, torch.Generator().manual_seed(0), torch.device("cpu"))
    before = {n: p.detach().clone() for n, p in ens.p.items()}
    b = tuple(torch.randint(0, 4 if i == 2 else 20, (K, 50)) for i in range(5))
    ens.train(b, b, torch.tensor([True, False, True, False]))
    for n in ens.p:
        assert torch.equal(ens.p[n][1], before[n][1]) and torch.equal(ens.p[n][3], before[n][3])
    assert all(not torch.equal(ens.p[n][0], before[n][0]) for n in ("loc", "W0", "W3", "b3"))


class _FixedUniform:

    def __init__(self):
        self.u = 0.5

    def uniform(self, a, b):
        return self.u


def test_env_and_phi_match_official():
    rng = np.random.default_rng(2)
    n_cells, n_rank = 30, 12
    lat = rng.uniform(35.6, 35.7, n_cells)
    lon = rng.uniform(139.6, 139.7, n_cells)
    rank = pg.rank_list(lat, lon, n_cells, threads=1)
    kcdf = pg.explore_cdf(n_rank, 0.6)
    geo = pg.Geo(np.zeros((n_cells, 2), int), lat, lon, np.zeros(0, int), np.zeros((n_cells, 4)), rank[:, :n_rank],
                 kcdf)
    m = pg_official.env_module()
    fixed = _FixedUniform()
    m.random = fixed
    e = object.__new__(m.timegeo_env)
    e.rank_list, e.traj_length, e.count = rank, 20, 0
    e.k_prob = [(x ** (-0.6)) / sum([i ** -0.6 for i in range(1, n_rank + 1)]) for x in range(1, n_rank + 1)]
    e.track_data = np.zeros((3, 5), int)
    n_special = 0
    for trial in range(400):
        c0 = int(rng.integers(n_cells))
        L = int(rng.integers(2, 21)) if trial % 5 else 20
        acts = rng.integers(0, 4, L - 1)
        if trial % 5 == 0:
            acts[:] = 0
            acts[17:] = 2
        us = rng.random(L - 1)
        env = pg.TrailEnv(np.array([c0]), geo)
        e.current_pos, e.t, e.home_point = c0, 0, c0
        hist = [c0]
        ppc, stay = env.phi(np.array([0]))
        assert (ppc[0], stay[0]) == (1, 1)
        for t in range(L - 1):
            a, u = int(acts[t]), float(us[t])
            H1 = list(env.hist[0, :env.h]) + [int(env.cur[0])]
            nxt = env.transition(np.array([0]), np.array([a]), np.array([u]))
            special = a == 2 and len(H1) == 19 and len(set(H1)) == 1
            fixed.u = u
            pos_o, t_o, done, hist, home_o, pre_o, st_o = e.step(a, hist, c0)
            if a == 3 or special:
                if special:
                    n_special += 1
                    assert nxt[0] == rank[c0, 1] and pos_o == c0
                else:
                    his = [h for h in H1 if h != c0]
                    assert (not his and nxt[0] == H1[-1]) or (his and nxt[0] in his)
                e.current_pos = int(nxt[0])
                pre_o = len(set(e.history_pos))
                st_o = 0
                for p in reversed(e.history_pos):
                    if p == e.current_pos:
                        st_o += 1
                    else:
                        break
            else:
                assert pos_o == nxt[0], (trial, t, a, pos_o, nxt[0])
            assert done == (t + 1 >= 19) and t_o == t + 1
            env.advance(np.array([0]), nxt)
            ppc2, stay2 = env.phi(np.array([0]))
            assert (ppc2[0], stay2[0]) == (pre_o, st_o), (trial, t)
            assert list(env.hist[0, :env.h]) == list(hist)
    assert n_special >= 20


def test_return_action_probabilities():
    geo = pg.Geo(np.zeros((10, 2), int), np.zeros(10), np.zeros(10), np.zeros(0, int), np.zeros((10, 4)),
                 np.tile(np.arange(10), (10, 1)), pg.explore_cdf(10, 0.6))
    env = pg.TrailEnv(np.array([0]), geo)
    for c in (4, 4, 7, 0, 4):
        env.advance(np.array([0]), np.array([c]))
    N = 4000
    u = (np.arange(N) + 0.5) / N
    idx = np.zeros(N, np.int64)
    env2 = pg.TrailEnv(np.zeros(N, np.int64), geo)
    env2.hist[:, :env.h] = env.hist[0, :env.h]
    env2.h = env.h
    env2.cur[:] = env.cur[0]
    nxt = env2.transition(np.arange(N), np.full(N, 3), u)
    assert (nxt == 4).sum() == 3000 and (nxt == 7).sum() == 1000
    del idx


def test_labels_match_official_sample_real_data():
    rng = np.random.default_rng(7)
    traj = rng.integers(0, 6, (40, 12))
    picks = rng.integers(0, 40 * 11, 300)
    stub_np = type("NP", (), {"random": type("R", (), {"choice": staticmethod(lambda n, size: picks)})})
    ns = {"np": stub_np, "random": random, "torch": torch}
    f = pg_official.method("models/mp_gail.py", "gail", "sample_real_data", ns)

    class Self:
        file = {0: traj}
        batch_size = 300
    (pt, actions, ppc_o, stay_o) = f(Self(), 0)
    geo = pg.Geo(np.zeros((6, 2), int), np.zeros(6), np.zeros(6), np.zeros(0, int), np.zeros((6, 4)),
                 np.tile(np.arange(6), (6, 1)), pg.explore_cdf(6, 0.6))
    offsets = np.arange(0, 40 * 12 + 1, 12)
    pairs = pg.real_pairs(traj.ravel(), np.tile(np.arange(12), 40), offsets, geo)
    ours = pairs["act"].reshape(40, 11)
    for x, a in zip(picks, actions):
        i, t = x // 11, x % 11
        assert ours[i, t] == a
        assert pairs["pos"].reshape(40, 11)[i, t] == traj[i, t]


def test_phi_replay_bit_identical():
    rng = np.random.default_rng(11)
    n_cells = 40
    lat = rng.uniform(35.6, 35.7, n_cells)
    lon = rng.uniform(139.6, 139.7, n_cells)
    rank = pg.rank_list(lat, lon, 20, threads=1)
    geo = pg.Geo(np.zeros((n_cells, 2), int), lat, lon, np.zeros(0, int), np.zeros((n_cells, 4)), rank,
                 pg.explore_cdf(20, 0.6))

    class RandomPolicy:
        pass
    n = 500
    L = rng.integers(2, 21, n)
    offsets = np.concatenate([[0], np.cumsum(L)])
    slots = rng.integers(0, 48, offsets[-1])
    c0 = rng.integers(0, n_cells, n)
    env = pg.TrailEnv(c0, geo)
    cells = np.empty(offsets[-1], np.int64)
    cells[offsets[:-1]] = c0
    gen = {f: [] for f in ("trail", "t", "ppc", "stay", "act", "nxt", "cur")}
    for t in range(19):
        idx = np.flatnonzero(L - 1 > t)
        if not len(idx):
            break
        ppc, stay = env.phi(idx)
        a = rng.integers(0, 4, len(idx))
        nxt = env.transition(idx, a, rng.random(len(idx)))
        for k, v in (("trail", idx), ("t", np.full(len(idx), t)), ("ppc", ppc), ("stay", stay), ("act", a),
                     ("nxt", nxt), ("cur", env.cur[idx].copy())):
            gen[k].append(v)
        env.advance(idx, nxt)
        cells[offsets[idx] + t + 1] = nxt
    gen = {k: np.concatenate(v) for k, v in gen.items()}
    o = np.lexsort((gen["t"], gen["trail"]))
    gen = {k: v[o] for k, v in gen.items()}
    real = pg.real_pairs(cells, slots, offsets, geo)
    assert np.array_equal(real["ppc"], gen["ppc"]) and np.array_equal(real["stay"], gen["stay"])
    assert np.array_equal(real["pos"], gen["cur"])
    home = c0[gen["trail"]]
    unamb = ((gen["act"] == 0) | ((gen["act"] == 2) & (gen["cur"] != home))
             | ((gen["act"] == 1) & (gen["nxt"] != gen["cur"]) & (gen["nxt"] != home)
                & ~np.array([gen["nxt"][i] in cells[offsets[gen["trail"][i]]:offsets[gen["trail"][i]] + gen["t"][i] + 1]
                             for i in range(len(o))]))
             | ((gen["act"] == 3) & (gen["nxt"] != gen["cur"])))
    assert unamb.sum() > 1000
    assert np.array_equal(real["act"][unamb], gen["act"][unamb])


def test_gae_matches_official_replay_buffer():
    rb = pg_official.replay_buffer().replay_buffer(1000, 0.99, 0.95)
    rng = np.random.default_rng(4)
    L = [5, 3, 9, 2]
    rewards, dones, values = [], [], []
    for l in L:
        for t in range(l - 1):
            r, d, v = float(rng.normal()), float(t == l - 2), 0.25 + 1e-8 * float(rng.normal())
            rb.store(0, 0, [0], 0, 0, 0, 0, r, d, v)
            rewards.append(r)
            dones.append(d)
            values.append(v)
    rb.process()
    ret, adv = pg.gae(np.array(rewards), np.array(dones), np.array(values))
    assert np.array_equal(ret, np.array([m[10] for m in rb.memory]))
    assert np.array_equal(adv, np.array([m[11] for m in rb.memory]))


def test_ppo_dp4_and_official_first_update():
    dev = torch.device("cpu")
    n_loc = 25
    rng = np.random.default_rng(0)
    b = {"pos": rng.integers(0, n_loc, 50), "slot": rng.integers(0, 48, 50), "ppc": rng.integers(1, 10, 50),
         "stay": rng.integers(0, 10, 50), "act": rng.integers(0, 4, 50), "ret": rng.random(50),
         "adv": rng.normal(size=50) * 5}
    pol, val, popt, vopt = pg.make_policy_value(n_loc, 0, dev)
    b["bprob"] = pg.policy_probs(pol, b["pos"], b["slot"], b["ppc"], b["stay"], dev)[np.arange(50), b["act"]]
    pol2, val2 = copy.deepcopy(pol), copy.deepcopy(val)
    popt2 = torch.optim.Adam(pol2.parameters(), lr=1e-3, weight_decay=1e-5)
    vopt2 = torch.optim.Adam(val2.parameters(), lr=1e-3, weight_decay=1e-5)
    info = pg.ppo_update(pol, val, popt, vopt, b, dev, entropy_weight=0.0)
    assert abs(info["ratio_min"] - 1) < 1e-5 and abs(info["ratio_max"] - 1) < 1e-5
    ns = {"torch": torch, "np": np}
    f = pg_official.method("models/mp_gail.py", "gail", "ppo_train", ns)
    z = np.zeros((50, 1), np.int64)

    class Buf:
        def sample(self, n):
            return (b["pos"], b["slot"], z, z[:, 0], b["ppc"], b["stay"], b["act"], b["ret"], b["adv"])

    class Self:
        buffer = Buf()
        batch_size = 50
        value_net, policy_net, value_optimizer, policy_optimizer = val2, pol2, vopt2, popt2
        value_iter = policy_iter = 1
        epsilon, entropy_weight, clip_grad = 0.2, 0.001, 10
    saved = torch.Tensor.cuda
    try:
        torch.Tensor.cuda = lambda self, *a, **k: self
        f(Self())
    finally:
        torch.Tensor.cuda = saved
    for p1, p2 in zip(pol.parameters(), pol2.parameters()):
        assert torch.allclose(p1, p2, rtol=1e-5, atol=1e-7)
    for p1, p2 in zip(val.parameters(), val2.parameters()):
        assert torch.allclose(p1, p2, rtol=1e-5, atol=1e-7)
    for _ in range(25):
        info = pg.ppo_update(pol, val, popt, vopt, b, dev)
    assert info["clipped"] > 0
    u = torch.full((1, 4), 0.25)
    assert math.isclose(float(torch.distributions.Categorical(u).entropy()), math.log(4), rel_tol=1e-6)


def test_vincenty_reference_line():
    def dms(d, m, s, sign=1):
        return sign * (d + m / 60 + s / 3600)
    a = (dms(37, 57, 3.72030, -1), dms(144, 25, 29.52440))
    b = (dms(37, 39, 10.15610, -1), dms(143, 55, 35.38390))
    d = float(pg.vincenty(a[0], a[1], b[0], b[1]))
    assert abs(d - 54972.271) < 0.01, d
    assert float(pg.vincenty(b[0], b[1], a[0], a[1])) == d or abs(float(pg.vincenty(b[0], b[1], a[0], a[1])) - d) < 1e-6
    assert float(pg.vincenty(35.0, 139.0, 35.0, 139.0)) == 0.0
    assert 1100 < float(pg.vincenty(35.0, 139.0, 35.01, 139.0)) < 1112


def test_rank_list_and_cells():
    rng = np.random.default_rng(9)
    pois = synth.make_catalog(rng, 800)
    geo = pg.build_geo(pois, threads=2)
    li = np.round(pois["lat"].to_numpy() / 0.01).astype(int)
    lj = np.round(pois["lon"].to_numpy() / 0.01).astype(int)
    assert np.array_equal(geo.cell_ij[geo.venue_cell], np.c_[li, lj])
    assert len(np.unique(np.c_[li, lj], axis=0)) == geo.n
    assert geo.rank.shape == (geo.n, min(200, geo.n)) and np.array_equal(geo.rank[:, 0], np.arange(geo.n))
    d = pg.vincenty(geo.lat[3], geo.lon[3], geo.lat, geo.lon)
    assert np.all(np.diff(d[geo.rank[3]]) >= 0)
    assert math.isclose(geo.kcdf[-1], 1.0, rel_tol=1e-12)
    inside = ((pois["lon"].to_numpy() >= geo.rects[geo.venue_cell, 0]) & (pois["lon"].to_numpy() <= geo.rects[geo.venue_cell, 1])
              & (pois["lat"].to_numpy() >= geo.rects[geo.venue_cell, 2]) & (pois["lat"].to_numpy() <= geo.rects[geo.venue_cell, 3]))
    assert inside.mean() > 0.999


def test_sidecar_allow_list():
    ok = {"method": "PateGAIL++", "eps": 1.0, "budget": {"K": 8221}, "accounting": {"rho_spent": 0.1}}
    assert pg.sidecar_problems(ok, "istanbul") == []
    assert pg.sidecar_problems(ok | {"n_empty_teachers": 3}, "istanbul")
    assert pg.sidecar_problems(ok | {"budget": {"K": 1, "n_train": 5}}, "istanbul")
    assert pg.sidecar_problems(ok | {"accounting": {"teacher_sizes": [1]}}, "istanbul")
    assert pg.sidecar_problems(ok | {"private_diagnostics": {}}, "istanbul")
    assert pg.sidecar_problems(ok | {"private_diagnostics": {}}, "jakarta") == []


def _run_synth(root, name, eps, arm, out, K=40, rounds=3, n_out=50_000):
    old = common.DATA
    common.DATA = root
    try:
        return pg.run(name, eps, 0, arm, out, "cpu", 2, n_out=n_out, K=K, rounds=rounds)
    finally:
        common.DATA = old


def test_release_schema_validity_and_determinism():
    import subprocess
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        pois, _ = synth.make_dataset(root, "synthcity", seed=21, n_trails=300)
        o1, o2 = root / "a" / "synthcity_eps1_s0.parquet", root / "b" / "synthcity_eps1_s0.parquet"
        s1 = _run_synth(root, "synthcity", 1.0, "main", o1)
        _run_synth(root, "synthcity", 1.0, "main", o2)
        assert hashlib.sha256(o1.read_bytes()).hexdigest() == hashlib.sha256(o2.read_bytes()).hexdigest()
        side = json.loads(o1.with_suffix(".json").read_text())
        assert pg.sidecar_problems(side, "synthcity") == [] and "private_diagnostics" not in side
        code = ("import sys, json; sys.path.insert(0, sys.argv[1]); import sanity_check as sc, pandas as pd; "
                "p = pd.read_parquet(sys.argv[3]); p['venue_id'] = p['venue_id'].astype(str); "
                "r, df = sc.check(__import__('pathlib').Path(sys.argv[2]), 'synthcity', p, json.loads(sys.argv[4])); "
                "print(json.dumps({'ok': r['ok'], 'problems': r['problems'], 'n': r['n_trails'], "
                "'cols': list(df.columns)}))")
        r = subprocess.run([f"{_TP_PY_GRID}", "-I", "-B", "-c", code, str(PKG.parent), str(o1),
                            str(root / "synthcity" / "pois.parquet"), json.dumps(synth.BBOX)],
                           capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", "OMP_NUM_THREADS": "2"})
        res = json.loads(r.stdout.strip().splitlines()[-1])
        assert res["ok"], res["problems"]
        assert res["cols"] == common.OUT_COLS and res["n"] == 50_000
        assert side["n_trails"] == 50_000 and side["eps"] == 1.0 and side["delta"] == 1e-6
        a = side["accounting"]
        assert math.isclose(acc.rho_from_sidecar(a, side["budget"]["rho_aux"]), a["rho_spent"], rel_tol=1e-12)
        assert a["rho_spent"] <= side["budget"]["rho_total"] * (1 + 1e-12) and a["eps_spent"] <= 1.0 + 1e-9
        assert acc.dpacc_eps_from_zcdp(a["rho_spent"]) <= 1.0 + 1e-9
        assert s1["empty_cell_fallbacks"] == side["empty_cell_fallbacks"]


def test_uniform_arm_and_noise_free_reference():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        synth.make_dataset(root, "jakarta", seed=22, n_trails=250)
        su = _run_synth(root, "jakarta", 4.0, "uniform", root / "u" / "jakarta_eps4_s0.parquet", n_out=2000)
        assert su["method"] == "PG-uniform" and all(r["eps_p"] is None for r in su["accounting"]["per_round"])
        assert math.isclose(su["accounting"]["eps_q_min"], su["accounting"]["eps_q_max"], rel_tol=0.2)
        assert "private_diagnostics" in su and su["private_diagnostics"]["n_train_trails"] == 250
        sn = _run_synth(root, "jakarta", math.inf, "main", root / "n" / "jakarta_epsinf_s0.parquet", n_out=2000)
        assert sn["method"].endswith("(noise-free reference)")
        try:
            _run_synth(root, "jakarta2", math.inf, "main", root / "n" / "x.parquet", n_out=100)
        except SystemExit:
            pass
        else:
            raise AssertionError("eps = inf accepted outside Jakarta")


def test_forbidden_datasets_refused():
    for d in ("gw_austin", "gw_nyc"):
        for f in (lambda: pg.load_private(d), lambda: pg.run(d, 1.0, 0, "main", Path("/nonexistent/x.parquet"))):
            try:
                f()
            except SystemExit:
                continue
            raise AssertionError(f"{d} accepted")




def main():
    tests = [(n, f) for n, f in sorted(globals().items(), key=lambda kv: kv[1].__code__.co_firstlineno
                                        if callable(kv[1]) and hasattr(kv[1], "__code__") else 0)
             if n.startswith("test_") and callable(f)]
    only = sys.argv[1:]
    failed = 0
    for n, f in tests:
        if only and not any(o in n for o in only):
            continue
        try:
            f()
            print(f"PASS {n}", flush=True)
        except Exception:
            failed += 1
            print(f"FAIL {n}\n{traceback.format_exc()}", flush=True)
    print(f"{len(tests) - failed if not only else '-'} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
