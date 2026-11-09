from __future__ import annotations

import copy
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))
import ctcl_testlib as TL

C = TL.C
sys.path.insert(0, str(TL.CTCL))
import calibrate as CB
import check as CK

TAB9 = {"PubMed": (75316, {4: (3.01, 3.004, 3.03, 3.023), 2: (5.49, 5.470, 5.63, 5.605), 1: (10.3, 10.256, 11.33, 11.283)}),
        "ChatbotArena": (180000, {4: (1.47, 1.465, 1.48, 1.473), 2: (2.5, 2.492, 2.57, 2.553),
                                  1: (4.58, 4.559, 5.08, 5.051)}),
        "MSC": (17940, {4: (11.38, 11.376, 11.45, 11.446), 2: (21.01, 21.010, 21.47, 21.465),
                        1: (39.41, 39.407, 42.70, 42.704)}),
        "Yelp": (1939290, {4: (0.63, 0.629, 0.63, 0.630), 2: (0.77, 0.765, 0.77, 0.768), 1: (0.91, 0.905, 0.94, 0.922)}),
        "OpenReview": (8396, {4: (23.3, 23.305, 23.44, 23.438), 2: (42.87, 42.872, 43.72, 43.723),
                              1: (80.05, 80.053, 86.05, 86.049)})}


def code_rule(args):
    N, eps, hist = args
    import dp_accounting
    delta, q, T = 1.0 / (N * math.log(N)), 4096 / N, 2000

    def comp(s):
        ev = dp_accounting.SelfComposedDpEvent(dp_accounting.PoissonSampledDpEvent(q, dp_accounting.GaussianDpEvent(s)), T)
        if hist:
            ev = dp_accounting.ComposedDpEvent([dp_accounting.GaussianDpEvent(10.0), ev])
        return dp_accounting.pld.PLDAccountant().compose(ev).get_epsilon(delta)
    lo, hi = 0.0, 100.0
    for _ in range(30):
        mid = (lo + hi) / 2
        if comp(mid) > eps:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def test_tab9():
    jobs = [(name, N, eps, hist) for name, (N, rows) in TAB9.items() for eps in rows for hist in (False, True)]
    with ProcessPoolExecutor(max_workers=12) as ex:
        sig = list(ex.map(code_rule, [(N, eps, hist) for _, N, eps, hist in jobs]))
    worst = {}
    for (name, N, eps, hist), s in zip(jobs, sig):
        pv, rv, pc, rc = TAB9[name][1][eps]
        printed, repro = (pc, rc) if hist else (pv, rv)
        TL.assert_close(s, repro, 6e-4 * max(1.0, repro), f"{name} eps {eps} {'CTCL' if hist else 'vanilla'} repro")
        rel = abs(printed / s - 1)
        worst[(name, hist)] = max(worst.get((name, hist), 0.0), rel)
        if name == "Yelp" and hist and eps == 1:
            assert 0.018 < rel < 0.021, rel
        elif name == "Yelp":
            assert rel < 0.0072, (name, eps, hist, rel)
        else:
            assert rel < 0.007, (name, eps, hist, rel)
    import dp_accounting
    N = 75316
    delta = 1.0 / (N * math.log(N))
    e_h = dp_accounting.pld.PLDAccountant().compose(dp_accounting.GaussianDpEvent(10.0)).get_epsilon(delta)
    TL.assert_close(e_h, 0.393, 6e-4, "PubMed histogram eps (Tab. 8)")
    print("Tab. 9 reproduced (worst relative deviation per row):",
          {f"{k[0]}{'/CTCL' if k[1] else '/vanilla'}": round(v, 4) for k, v in worst.items()})


def _calib(args):
    eps, variant, config = args
    return CB.calibrate(eps, variant, config)


def prv(sigma, q, T, gauss):
    return CK.prv_subprocess(sigma, q, T, gauss)


def test_spec_table():
    cases = [(e, v, "code") for v in ("plain", "vp") for e in (1, 2, 4, 8)] + [(1, "plain", "paper"), (4, "plain", "paper")]
    with ProcessPoolExecutor(max_workers=10) as ex:
        res = list(ex.map(_calib, cases))
    for (e, v, cfg), r in zip(cases, res):
        ref = C.SIGMA_REF[(cfg, v)][e]
        TL.assert_close(r["sigma_sgd"], ref, C.SIGMA_REF_TOL, f"sigma {cfg}/{v} eps {e}")
        assert r["eps_composed_pld"] <= e - C.PRV_MARGIN + 1e-12, r
        assert r["sample_rate"] == C.CONFIGS[cfg]["batch"] / C.N_REF and r["steps"] == C.CONFIGS[cfg]["steps"]
        if v == "vp":
            TL.assert_close(r["sigma_prior"], C.SIGMA_PRIOR_REF[e], 1e-4, f"sigma_prior eps {e}")
            TL.assert_close(r["sigma_eq"], C.SIGMA_EQ_REF[e], 1e-4, f"sigma_eq eps {e}")
            TL.assert_close(r["sigma_eq"], (1 / 100 + 2 / r["sigma_prior"] ** 2) ** -0.5, 1e-12, "sigma_eq formula")
        else:
            assert r["sigma_prior"] is None and r["n_prior"] == 0 and r["sigma_eq"] == 10.0
        gauss = [(10.0, 1)] + ([(r["sigma_prior"], 2)] if v == "vp" else [])
        p = prv(r["sigma_sgd"], r["sample_rate"], r["steps"], gauss)
        assert math.isfinite(p["eps_prv"]) and p["eps_prv"] <= e, (cfg, v, e, p)
        print(f"{cfg:5s} {v:5s} eps {e}: sigma {r['sigma_sgd']:.5f} (ref {ref}), PLD {r['eps_composed_pld']:.5f}, "
              f"PRV {p['eps_prv']:.4f}, RDP {p['eps_rdp']:.3f}")
    TL.assert_close(res[0]["eps_topic_hist_only_pld"], 0.3969, 1e-4, "histogram alone at delta 1e-6")
    return res


def test_sigma_keys(res):
    vp1 = next(r for r in res if r["variant"] == "vp" and r["eps_total"] == 1)
    pc = CB.prior_calib(vp1)
    assert pc["sigma_hist"] == vp1["sigma_prior"] and pc["n_hist"] == 2 and pc["sigma_hist"] != vp1["sigma_eq"]
    CB.check_prior_calib(pc, vp1)
    bad = dict(pc, sigma_hist=vp1["sigma_eq"])
    try:
        CB.check_prior_calib(bad, vp1)
        raise AssertionError("a prior_calib carrying sigma_eq was accepted")
    except AssertionError as e:
        assert "sigma" in str(e)
    assert set(vp1) >= {"sigma_topic_hist", "n_topic_hist", "sigma_prior", "n_prior", "sigma_eq", "sigma_sgd"}
    assert "sigma_hist" not in vp1, "calib.json must not use the ambiguous key sigma_hist"
    try:
        CB.prior_calib(next(r for r in res if r["variant"] == "plain"))
        raise AssertionError("prior_calib for the plain variant")
    except ValueError:
        pass
    print("sigma keys: prior_calib.sigma_hist = sigma_prior, swapped file rejected")


def reference_sidecar(cal: dict, method: str) -> dict:
    variant, config, param = C.METHODS[method]
    gauss = [(cal["sigma_topic_hist"], 1)] + ([(cal["sigma_prior"], 2)] if variant == "vp" else [])
    p = prv(cal["sigma_sgd"], cal["sample_rate"], cal["steps"], gauss)
    return {"method": method, "variant": variant, "config": config, "param": param, "eps_target": cal["eps_total"],
            "eps": cal["eps_total"], "delta": cal["delta"], "noise_multiplier": cal["sigma_sgd"],
            "sample_rate": cal["sample_rate"], "steps": cal["steps"], "b_nom": cal["b_nom"], "clip": cal["clip"],
            "sigma_topic_hist": cal["sigma_topic_hist"], "n_topic_hist": cal["n_topic_hist"], "K_topics": C.K_TOPICS,
            "sigma_prior": cal["sigma_prior"], "n_prior": cal["n_prior"], "sigma_eq": cal["sigma_eq"],
            "vp_calibration": {"sigma_hist": cal["sigma_eq"], "n_hist": 1}, "eps_composed_pld": cal["eps_composed_pld"],
            "eps_prv": p["eps_prv"], "eps_rdp": p["eps_rdp"], "complete": True, "steps_run": cal["steps"]}


def test_check(res):
    good = []
    for method, v, e in (("ctcl", "plain", 1), ("ctcl", "plain", 4), ("ctclvp", "vp", 1), ("ctclvp", "vp", 8)):
        cal = next(r for r in res if r["variant"] == v and r["eps_total"] == e and r["config"] == "code")
        s = reference_sidecar(cal, method)
        r = CK.check_sidecar(s)
        assert r["ok"], (method, e, r["problems"])
        TL.assert_close(r["eps_pld_recomposed"], cal["eps_composed_pld"], 1e-12, "recomposition")
        good.append(s)
    cal = next(r for r in res if r["config"] == "paper" and r["eps_total"] == 4)
    assert CK.check_sidecar(reference_sidecar(cal, "ctclpaper"))["ok"]
    s0 = good[0]
    no_prv = lambda *a: {"eps_prv": 0.5, "eps_rdp": 1.0}
    mut = {"sigma 0": {"noise_multiplier": 0.0}, "sigma NaN": {"noise_multiplier": float("nan")},
           "sigma inf": {"noise_multiplier": float("inf")}, "q differs": {"sample_rate": 256 / C.N_REF},
           "T differs": {"steps": 1000}, "clip differs": {"clip": 2.0}, "B differs": {"b_nom": 256},
           "delta differs": {"delta": 1e-5}, "sigma_h differs": {"sigma_topic_hist": 20.0},
           "K differs": {"K_topics": 1300}, "n_h differs": {"n_topic_hist": 2}, "incomplete": {"complete": False},
           "steps_run": {"steps_run": 1000}, "metric": {"composite_vs_train": -2.0},
           "prior in plain": {"sigma_prior": 5.0, "n_prior": 2}, "sigma_eq": {"sigma_eq": 9.0},
           "vp_calibration": {"vp_calibration": {"sigma_hist": 8.0, "n_hist": 1}},
           "sigma below ref": {"noise_multiplier": s0["noise_multiplier"] - 0.01},
           "calibration eps mismatch": {"eps_composed_pld": s0["eps_composed_pld"] - 0.01}}
    for what, upd in mut.items():
        s = dict(copy.deepcopy(s0), **upd)
        r = CK.check_sidecar(s, prv=no_prv if what != "calibration eps mismatch" else prv)
        assert not r["ok"], f"check.py accepted: {what}"
    for k in ("noise_multiplier", "sample_rate", "steps", "sigma_topic_hist", "eps_target", "n_prior", "K_topics"):
        s = copy.deepcopy(s0)
        s.pop(k)
        assert not CK.check_sidecar(s, prv=no_prv)["ok"], f"missing {k} accepted"
    vp = good[2]
    for what, upd in {"sigma_prior differs": {"sigma_prior": vp["sigma_prior"] * 0.9}, "n_prior 1": {"n_prior": 1},
                      "prior missing": {"sigma_prior": None}}.items():
        assert not CK.check_sidecar(dict(copy.deepcopy(vp), **upd), prv=no_prv)["ok"], what
    s = dict(copy.deepcopy(s0), eps_target=0.5, eps=0.5)
    assert not CK.check_sidecar(s)["ok"]
    print(f"check.py: {len(good) + 1} reference sidecars accepted, {len(mut) + 7 + 3 + 1} broken ones rejected")


def test_noise_topics():
    from scipy.stats import norm
    e = sum(k * (norm.cdf((k + 0.5) / 10) - norm.cdf((k - 0.5) / 10)) for k in range(1, 200))
    TL.assert_close(e, 3.988, 5e-4, "E[max(0, round(Z))]")
    TL.assert_close(1440 * e, 5742, 2, "pseudo-counts over 1,440 empty bins")


if __name__ == "__main__":
    t0 = time.time()
    test_noise_topics()
    test_tab9()
    res = test_spec_table()
    test_sigma_keys(res)
    test_check(res)
    print(f"test_accounting: all passed ({time.time() - t0:.0f}s)")
