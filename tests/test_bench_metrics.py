import copy
import math
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from trajpe import bench_metrics as bm
from trajpe.trails import TrailSet

LN = math.log
VENUES = np.array([[0.5, 0.5], [1.5, 0.5], [3.5, 0.5], [0.5, 1.5], [5.5, 5.5]])
CAT = SimpleNamespace(xy=VENUES, width_km=6.0, height_km=6.0, root=np.array([0, 1, 0, 2, 1]), n_roots=3)
R_TRAILS = [([0, 1, 2], [0, 30, 90]), ([0, 3], [600, 660]), ([4, 4], [1440, 1500]), ([1, 0], [2000, 2010])]
S_TRAILS = [([0, 2], [0, 60]), ([1, 1, 0], [100, 200, 300]), ([3, 0], [700, 720])]


def _bench(**overrides):
    P = copy.deepcopy(bm.load_protocol())
    for m in P["metrics"]:
        m["params"].update(overrides.get(m["name"], {}))
    return bm.Benchmark(CAT, P)


def _preps(B, a=R_TRAILS, b=S_TRAILS):
    return B.prepare(TrailSet.from_lists(a)), B.prepare(TrailSet.from_lists(b))


def close(a, b, tol=1e-6):
    assert abs(a - b) <= tol, (a, b)


def test_divergences():
    close(bm.jsd_skip0([1, 0], [0, 1]), LN(2))
    close(bm.jsd_eps([1, 0], [0, 1]), LN(2), 1e-7)
    close(bm.jsd_nats([2, 0], [0, 5]), LN(2))
    close(bm.jsd_nats([1, 2, 3], [2, 4, 6]), 0.0)
    assert bm.jsd_nats([0, 0], [0, 0]) == 0.0 and bm.jsd_nats([0, 0], [1, 0]) == LN(2)
    close(float(bm.bernoulli_jsd(1.0, 0.0)), LN(2))
    close(float(bm.bernoulli_jsd(0.3, 0.3)), 0.0)
    close(float(bm.bernoulli_jsd(0.5, 0.0)), 0.5 * (0.5 * LN(2) + 0.5 * LN(2 / 3)) + 0.5 * LN(4 / 3))


def test_grid_trajectories():
    c, o = bm.grid_trajectories(np.array([0, 3]), np.array([0, 1]), np.array([0, 2]), 6, "ldp")
    assert c.tolist() == [0, 7, 13, 19] and o.tolist() == [0, 4]
    ix, iy, off = np.array([2, 2, 2, 3]), np.array([2, 2, 2, 3]), np.array([0, 3, 4])
    ca, oa = bm.grid_trajectories(ix[:3], iy[:3], np.array([0, 3]), 6, "ada")
    cl, ol = bm.grid_trajectories(ix[:3], iy[:3], np.array([0, 3]), 6, "ldp")
    assert ca.tolist() == [14, 14] and cl.tolist() == [14]
    cd, od = bm.grid_trajectories(np.array([0, 0, 1, 1]), np.array([0, 0, 0, 0]), np.array([0, 4]), 6, "dedup")
    assert cd.tolist() == [0, 6] and od.tolist() == [0, 2]


def test_ngrams_and_topk():
    k, tr, _, _ = bm.ngrams(np.array([1, 2, 3, 5, 5]), np.array([0, 3, 5]), [2, 3], 10)
    assert sorted(k.tolist()) == [23, 34, 66, 234] and sorted(tr.tolist()) == [0, 0, 0, 1]
    k2, _, _, _ = bm.ngrams(np.array([5, 5, 6]), np.array([0, 3]), [2, 3], 10, no_repeat_above2=True)
    assert sorted(k2.tolist()) == [66, 67]
    u, c = bm.top_keys(np.array([7, 3, 9, 5]), np.array([2, 2, 5, 1]), 3)
    assert u.tolist() == [9, 3, 7]
    assert bm.lookup(np.array([3, 4, 9]), np.array([3, 9]), np.array([10, 20])).tolist() == [10, 0, 20]


def test_kendall_tie_conventions():
    a, s = [3, 1, 2], [3, 2, 2]
    close(bm.kendall_ldp(a, s), 1 / 3)
    close(bm.kendall_ada(a, s), 1.0)


def test_hotspot_ndcg():
    v = bm.hotspot_ndcg([5, 4, 3, 0], [4, 5, 0, 3], 2)
    close(v, (0.5 + 1 / math.log2(3)) / (1 + 0.5 / math.log2(3)))
    close(bm.hotspot_ndcg([5, 4, 3, 0], [5, 4, 3, 0], 3), 1.0)


def test_bucket_rules():
    v = np.array([0.0, 0.5, 1.0])
    assert bm.bucket_counts_ldp(v, 0.5, 2).tolist() == [2, 2]
    assert bm.bucket_counts_ada(v, 0.5, 2).tolist() == [1, 1]
    assert bm.llmob_hist([0.2, 1.0, 11.0, 30.0], 0, 10, 10).tolist() == [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2]
    assert bm.movesim_hist([0.1, 0.6], 0.0, 1.0, 2).tolist() == [1.0]


def test_queries_primitives():
    B = _bench()
    R, _ = _preps(B)
    n = bm.trails_in_circles(R, np.array([[2.5, 0.5], [2.5, 0.5]]), np.array([0.3, 1.0]))
    assert n.tolist() == [1, 2]
    n = bm.trails_in_circles(R, np.array([[5.5, 5.0], [5.5, 5.0]]), np.array([0.4, 0.6]))
    assert n.tolist() == [0, 1]
    assert bm.points_in_boxes(R.xy, np.array([[0.0, 1.0, 0.0, 1.0]])).tolist() == [3]


def test_length_and_diameter_errors():
    B = _bench(ada_length_error={"n_buckets": 4}, ldp_length_error={"n_buckets": 4},
               ada_diameter_error={"n_buckets": 4}, ldp_diameter_error={"n_buckets": 4})
    R, S = _preps(B)
    assert R.travel.tolist() == [3, 1, 0, 1] and S.travel.tolist() == [3, 1, 1]
    assert np.allclose(R.diameter, [3, 1, 0, 1]) and np.allclose(S.diameter, [3, 1, 1])
    out = B.compare(R, S)
    ada = 0.5 * (0.25 * LN(2) + 0.5 * LN(6 / 7)) + 0.5 * (2 / 3) * LN(8 / 7)
    close(out["ada_length_error"], ada)
    close(out["ada_diameter_error"], ada)
    ldp = 0.5 * (0.25 * LN(2) + 0.75 * LN(6 / 7)) + 0.5 * LN(8 / 7)
    close(out["ldp_length_error"], ldp, 1e-7)
    close(out["ldp_diameter_error"], ldp, 1e-7)


def test_trip_density_kendall():
    B = _bench()
    R, S = _preps(B)
    out = B.compare(R, S)
    close(out["ada_trip_error"], 0.25 * LN(6 / 7) + 0.25 * LN(2) + LN(8 / 7) / 3 + LN(2) / 6)
    close(out["ldp_trip_error"], out["ada_trip_error"], 1e-7)
    close(out["ldp_density_error"], 0.5 * ((8 / 9) * LN(16 / 17) + LN(2) / 9) + 0.5 * LN(18 / 17), 1e-7)
    close(out["ldp_hotspot_error"], 0.0)
    close(bm.kendall_ldp(R.coverage(6, "ldp"), S.coverage(6, "ldp")), 129 / 630)
    close(bm.kendall_ada(R.coverage(6, "ada"), S.coverage(6, "ada")), 189 / 630)
    close(out["ldp_kendall_tau"], 129 / 630)
    close(bm.METRICS["retra_kendall_tau"](B, R, S, {"grid": 6}), 96 / 630)


def test_patterns():
    B = _bench(ada_fp_f1={"top_k": 2}, ada_fp_support_are={"top_k": 2}, ldp_pattern_f1={"top_k": 2},
               ldp_pattern_error={"top_k": 2})
    R, S = _preps(B)
    out = B.compare(R, S)
    close(out["ada_fp_f1"], 0.5)
    close(out["ldp_pattern_f1"], 0.5)
    close(out["ada_fp_support_are"], 2 / 3)
    close(out["ldp_pattern_error"], 2 / 3)


def test_pattern_counts_by_hand():
    P = copy.deepcopy(bm.load_protocol())
    B = bm.Benchmark(CAT, P)
    pr = B.prepare(TrailSet.from_lists([([0, 1, 0, 1], [0, 1, 2, 3]), ([0, 1], [5, 6])]))
    u, c = bm._hrnet_patterns(pr, {"grid": 6, "min_size": 2, "max_size": 3})
    d = dict(zip(u.tolist(), c.tolist()))
    k01 = 1 * 37 + 7
    assert d[k01] == 2
    u2, c2 = bm._ldp_patterns(pr, {"grid": 6, "min_size": 2, "max_size": 3})
    assert dict(zip(u2.tolist(), c2.tolist()))[k01] == 3


def test_retrasyn_time_metrics():
    B = _bench()
    R, S = _preps(B)
    out = B.compare(R, S)
    one_sided = 0.5 * LN((1 + 1e-8) / (0.5 + 1e-8))
    close(out["retra_density_error"], 14 * one_sided / 1008, 1e-9)
    B._fixed["retra_windows"] = np.array([0])
    close(bm.METRICS["retra_pattern_f1"](B, R, S, B.specs["retra_pattern_f1"]["params"]), 0.0)
    close(bm.METRICS["retra_pattern_f1"](B, R, R, B.specs["retra_pattern_f1"]["params"]), 3 / 100)
    S1 = B.prepare(TrailSet.from_lists(S_TRAILS[:1]))
    w = bm.METRICS["retra_hotspot_ndcg"](B, R, S1, B.specs["retra_hotspot_ndcg"]["params"])
    disc = 1 / np.log2(np.arange(2, 12))
    r = np.array([1, 1 / 3, 1 / 4, 1 / 5, 1 / 6, 1 / 7, 1 / 8, 1 / 2, 1 / 9, 1 / 10])
    close(w, float(np.sum(r * disc) / np.sum(1 / np.arange(1, 11) * disc)))
    close(bm.METRICS["retra_hotspot_ndcg"](B, R, R, B.specs["retra_hotspot_ndcg"]["params"]), 1.0)


def test_hrnet_metrics():
    B = _bench(hrnet_destination={"grid": 6, "n_start": 2}, hrnet_transition={"grid": 6, "n_start": 2},
               hrnet_waypoint={"grid": 6, "n_start": 2})
    R, S = _preps(B)
    out = B.compare(R, S)
    close(out["hrnet_destination"], 0.5 * (0.5 * (0.5 * LN(2 / 3) + 0.5 * LN(2)) + 0.5 * LN(4 / 3)))
    close(out["hrnet_transition"], LN(2))
    wp = 0.5 * (0.5 * LN(2) + 0.5 * LN(2 / 3)) + 0.5 * LN(4 / 3)
    close(out["hrnet_waypoint"], 0.5 * (3 * wp / 36))


def test_movesim_and_llmob():
    B = _bench()
    R, S = _preps(B)
    out = B.compare(R, S)
    kl_p = 0.25 * LN(6 / 7) + 0.25 * LN(2) + 0.5 * LN(6 / 5)
    kl_q = (LN(8 / 7) + LN(4 / 5) + LN(2)) / 3
    close(out["movesim_duration"], 0.5 * (kl_p + kl_q))
    close(out["llmob_si"], 0.5 * (0.4 * LN(0.4 / 0.325) + 0.6 * LN(0.6 / 0.675))
          + 0.5 * (0.25 * LN(0.25 / 0.325) + 0.75 * LN(0.75 / 0.675)))
    close(out["llmob_sd"], 0.5 * (0.2 * LN(0.2 / 0.225) + 0.6 * LN(0.6 / 0.55) + 0.2 * LN(2))
          + 0.5 * (0.25 * LN(0.25 / 0.225) + 0.5 * LN(0.5 / 0.55) + 0.25 * LN(2)))
    close(out["movesim_dailyloc"], 0.5 * (0.5 * LN(2) + 0.5 * LN(2 / 3)) + 0.5 * LN(4 / 3))
    assert np.allclose(bm.rank_profile(R, 3), np.array([5, 3, 1]) / 9)
    assert np.allclose(bm.rank_profile(S, 3), np.array([4, 3, 0]) / 7)
    close(out["movesim_irank_rank"], bm.jsd_nats(np.r_[5, 3, 1, np.zeros(97)], np.r_[4, 3, 0, np.zeros(97)]))


def test_identity_release_is_perfect():
    B = _bench()
    R, _ = _preps(B)
    R2, _ = _preps(B)
    out = B.compare(R, R2)
    info = bm.metric_info()
    for name, v in out.items():
        if info[name]["direction"] == "lower":
            assert abs(v) < 1e-9, (name, v)
    assert out["ada_fp_f1"] == 1.0 and out["ldp_hotspot_error"] == 0.0
    cov = R.coverage(6, "ldp")
    n = len(cov)
    untied = sum(cov[i] != cov[j] for i in range(n) for j in range(i + 1, n))
    close(out["ldp_kendall_tau"], untied / (n * (n - 1) / 2))
    close(out["ada_kendall_tau"], bm.kendall_ada(R.coverage(15, "ada"), R.coverage(15, "ada")))


def test_size_rescaling_of_counts():
    B = _bench()
    R, S = _preps(B)
    S2 = B.prepare(TrailSet.from_lists(S_TRAILS * 2))
    a, b = B.compare(R, S), B.compare(R, S2)
    for name in ("ada_query_avre", "ldp_query_error", "retra_query_error", "priv_density_are", "hrnet_traj_density",
                 "ada_fp_support_are", "ldp_pattern_error", "priv_pattern_are", "hrnet_traj_pattern"):
        close(a[name], b[name], 1e-9)


def test_queries_fixed_and_seeded():
    B1, B2 = _bench(), _bench()
    c1, r1 = B1.circles(B1.specs["ada_query_avre"]["params"], True)
    c2, r2 = B2.circles(B2.specs["ada_query_avre"]["params"], True)
    assert np.array_equal(c1, c2) and np.allclose(r1, 2.0)
    assert (c1 >= 0).all() and (c1 <= 6).all()
    boxes, t0 = B1.st_squares(B1.specs["retra_query_error"]["params"])
    e = math.sqrt(36 / 9)
    assert np.allclose(boxes[:, 1] - boxes[:, 0], e) and (boxes[:, 0] >= 0).all() and (boxes[:, 1] <= 6).all()
    assert t0.min() >= 0 and t0.max() <= 1008 - 20


def test_v3_split():
    import json
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "configs" / "v3_large_eps.json").read_text())
    split = bm.v3_split(cfg)
    assert split["ldp_trip_error"][0] == "measured" and split["movesim_grank"][0] == "measured"
    assert split["llmob_dard"][0] == "measured" and split["llmob_sd"][0] == "measured"
    assert split["ldp_density_error"][0] == "near" and split["retra_density_error"][0] == "near"
    assert split["movesim_duration"][0] == "near" and split["llmob_stvd"][0] == "near"
    for m in ("ada_fp_f1", "ldp_diameter_error", "priv_length_jsd", "hrnet_transition", "retra_transition_error",
              "movesim_irank", "hrnet_waypoint"):
        assert split[m][0] == "unmeasured", (m, split[m])
    assert bm.v3_split(cfg, metric_names=["nextpoi_markov_mrr_last"])["nextpoi_markov_mrr_last"][0] == "unmeasured"
    cfg2 = dict(cfg, work_families="gap,hour_root")
    assert bm.v3_split(cfg2)["ldp_trip_error"][0] != "measured"
    P = copy.deepcopy(bm.load_protocol())
    del P["v3_split"]["families"]["ctrans16"]
    assert bm.v3_split(cfg, P)["ldp_trip_error"][0] == "measured"
    try:
        bm.v3_split(dict(cfg, work_families=cfg["work_families"] + ",ctrans16"), P)
        raise AssertionError("a used family without a descriptor must be refused")
    except KeyError:
        pass
    try:
        bm.v3_split(dict(cfg, work_families="gap,no_such_family"))
        raise AssertionError("an unknown family must be refused")
    except ValueError:
        pass


def test_every_protocol_metric_runs_and_is_finite():
    B = _bench()
    R, S = _preps(B)
    timings = {}
    out = B.compare(R, S, timings)
    assert set(out) == {m["name"] for m in bm.load_protocol()["metrics"]} == set(timings)
    bad = {k: v for k, v in out.items() if not np.isfinite(v)}
    assert not bad, bad


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"all {len(tests)} benchmark-metric tests passed")
