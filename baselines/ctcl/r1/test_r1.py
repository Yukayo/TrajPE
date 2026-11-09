from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

R1 = Path(__file__).resolve().parent
CTCL = R1.parent
PROJECT = CTCL.parents[1]
sys.path.insert(0, str(CTCL / "tests"))
sys.path.insert(0, str(CTCL))
sys.path.insert(0, str(R1))
import ctcl_testlib as TL
import sample_ctcl as SC
import sample_ctcl_r1 as S1
import topics as TP

C = TL.C
FILES = ["baselines/ctcl/r1/sample_ctcl_r1.py", "baselines/ctcl/r1/pipeline_ctcl_r1.py",
         "baselines/ctcl/r1/test_r1.py", "scripts/queue_ctcl_r1.py"]


def quiet(*a, **k):
    pass


def small_quotas(seed=0):
    rng = np.random.default_rng(seed)
    h = np.zeros(1440)
    occ = rng.choice(1440, 300, replace=False)
    h[occ] = rng.gamma(0.6, 12.0, 300)
    return TP.quotas(h + rng.normal(0, 10, 1440))


def both(n, w, rates, pois, seed):
    cats = set(pois["category"])
    out = []
    for mod in (SC, S1):
        try:
            out.append(mod.run_rounds(n, w, TL.FakeGenerator(rates, pois, seed=seed), cats, log_fn=quiet))
        except SC.BrokenModel as e:
            out.append(e)
    return out


def strip_r1(st: dict) -> dict:
    out = {k: v for k, v in st.items() if not k.startswith("r1_")}
    for k in ("rounds", "extra_rounds"):
        out[k] = [{f: x for f, x in r.items() if f != "sec"} for r in out[k]]
    return out


def test_equivalence():
    pois = TL.public_catalog("bandung")
    cases = []
    w, n = small_quotas(0)
    rng = np.random.default_rng(11)
    cases.append(("heterogeneous", n, w,
                  {k: float(rng.choice([0.9, 0.3, 0.08, 0.01, 0.0], p=[0.6, 0.25, 0.1, 0.03, 0.02]))
                   for k in range(1440)}))
    w2, n2 = small_quotas(3)
    cases.append(("high rates", n2, w2, {k: 0.85 for k in range(1440)}))
    rng = np.random.default_rng(5)
    cases.append(("mid rates", n2, w2, {k: float(rng.uniform(0.05, 0.6)) for k in range(1440)}))
    for name, n_, w_, rates in cases:
        (t1, s1), (t2, s2) = both(n_, w_, rates, pois, seed=1)
        assert t1 == t2, f"{name}: r1 texts differ from v1 on a passing execution"
        assert strip_r1(s1) == strip_r1(s2), f"{name}: statistics differ"
        assert s2["r1_global_fill"]["applied"] is False
        assert s2["r1_gate"].get("passed", True)
        print(f"equivalence ({name}): v1 passes, r1 identical ({len(t1)} texts, {len(s1['rounds'])} rounds, "
              f"{len(s1['extra_rounds'])} extra rounds, deficit {s1['reapportionment']['deficit']})")


def test_rule_i():
    pois = TL.public_catalog("bandung")
    w, n = small_quotas(0)
    rates5 = {k: [0.9, 0.3, 0.08, 0.01, 0.0][k % 5] for k in range(1440)}
    v1, r1 = both(n, w, rates5, pois, seed=1)
    assert isinstance(v1, SC.BrokenModel) and "model is broken" in v1.why, "v1 should exit 3 by the cumulative gate"
    first = v1.stats["rounds"][0]["valid_rate"]
    assert first >= 0.02 and v1.stats["valid_rate"] < 0.02
    if isinstance(r1, SC.BrokenModel):
        assert "rule (i)" not in r1.why, f"r1 stopped by rule (i) although the first-round rate is {first}"
        res = f"r1 ends at its safety net: {r1.why}"
    else:
        texts, st = r1
        assert len(texts) == int(n.sum()) and st["r1_gate"]["passed"]
        assert abs(st["r1_gate"]["first_round_valid_rate"] - first) < 1e-4
        res = (f"r1 completes: {len(texts)} texts, {len(st['rounds'])} rounds, cumulative rate {st['valid_rate']:.4f}, "
               f"global fill {st['r1_global_fill']['from_global_surplus']}")
    print(f"rule (i) drift case: v1 exit 3 at cumulative {v1.stats['valid_rate']:.4f} (first round {first:.4f}); {res}")
    gen = TL.FakeGenerator({k: 0.01 for k in range(1440)}, pois, seed=2)
    try:
        S1.run_rounds(n, w, gen, set(pois["category"]), log_fn=quiet)
        raise AssertionError("a uniformly broken model did not exit 3 under r1")
    except SC.BrokenModel as e:
        assert "rule (i)" in e.why and len(e.stats["rounds"]) == 1 and e.stats["n_generated"] >= 20000
        print(f"rule (i) broken model (rate 0.01): exit 3 after round 1 ({e.stats['n_generated']} samples)")


class LuckyGenerator:

    def __init__(self, rates, lucky, pois, seed=0):
        self.base = TL.FakeGenerator(rates, pois, seed=seed)
        self.lucky, self.seen = set(lucky), {}
        self.rng = np.random.default_rng([seed, 7])
        self.pois = pois.iloc[:5000]

    def __call__(self, reqs):
        out = self.base([(k, c) for k, c in reqs if k not in self.lucky])
        for k, c in reqs:
            if k in self.lucky:
                for _ in range(c):
                    first = self.seen.get(k, 0) == 0
                    self.seen[k] = self.seen.get(k, 0) + 1
                    out.append((k, TL.render_trail(self.rng, self.pois, 3) if first else "Day: Someday\n99:99 | ?", True))
        order = {k: i for i, (k, _) in enumerate(reqs)}
        out.sort(key=lambda x: order[x[0]])
        return out


def test_rule_ii():
    pois = TL.public_catalog("bandung")
    cats = set(pois["category"])
    n = np.zeros(1440, np.int64)
    n[0], n[1], n[2], n[3] = 3000, 3000, 1, 50
    w = n.copy()
    w[2] = 3000
    rates = {0: 0.9, 1: 0.9, 3: 0.0}
    gens = [LuckyGenerator(rates, [2], pois, seed=4) for _ in range(2)]
    try:
        SC.run_rounds(n, w, gens[0], cats, log_fn=quiet)
        raise AssertionError("v1 should reach its safety net")
    except SC.BrokenModel as e:
        assert "safety net" in e.why
        v1why = e.why
    texts, st = S1.run_rounds(n, w, gens[1], cats, log_fn=quiet)
    assert len(texts) == int(n.sum())
    g = st["r1_global_fill"]
    assert g["applied"] and g["from_global_surplus"] == g["missing_after_extra_rounds"] > 0
    got = np.bincount([k for _, k in texts], minlength=1440)
    assert got[2] == 1 and got[3] == 0 and got[0] + got[1] == 6050
    assert abs(int(got[0]) - int(got[1])) <= 1 + g["missing_after_extra_rounds"]
    print(f"rule (ii) lucky-fill case: v1 '{v1why}'; r1 fills {g['from_global_surplus']} texts from the global "
          f"surplus ({g['global_surplus_available']} available, {g['topics_giving']} topics) -> {len(texts)} texts")
    n[0] = n[1] = 12000
    n[3] = 1000
    w = n.copy()
    w[2] = 10 ** 7
    try:
        S1.run_rounds(n, w, LuckyGenerator({0: 0.9, 1: 0.9, 3: 0.0}, [2], pois, seed=4), cats, log_fn=quiet)
        raise AssertionError("r1 should exit 3 when the global surplus is too small")
    except SC.BrokenModel as e:
        assert "rule (ii)" in e.why
        print(f"rule (ii) insufficient surplus: exit 3 ({e.why})")


def test_global_fill():
    w = np.array([10, 10, 30, 50, 0], float)
    assert S1.global_fill(10, w, {0: 100, 1: 100, 2: 100, 3: 100}) == {0: 1, 1: 1, 2: 3, 3: 5}
    t = S1.global_fill(10, w, {0: 100, 1: 100, 2: 100, 3: 2})
    assert t[3] == 2 and sum(t.values()) == 10 and t[2] >= t[0], t
    assert S1.global_fill(10, w, {0: 3, 1: 3, 2: 3}) is None
    assert S1.global_fill(9, w, {0: 3, 1: 3, 2: 3}) == {0: 3, 1: 3, 2: 3}
    assert S1.global_fill(3, w, {4: 5}) == {4: 3}
    rounds = [{"round": 1, "requests": 57550, "valid_rate": 0.0437}, {"round": 2, "requests": 150332,
                                                                       "valid_rate": 0.03224}]
    v = S1.rule_i_from_rounds(rounds)
    assert v["passed"] and v["rounds_in_prefix"] == 1 and v["first_round_valid_rate"] == 0.0437
    v = S1.rule_i_from_rounds([{"round": 1, "requests": 57456, "valid_rate": 5e-05}])
    assert not v["passed"]
    v = S1.rule_i_from_rounds([{"round": 1, "requests": 15000, "valid_rate": 0.5},
                               {"round": 2, "requests": 6000, "valid_rate": 0.01}])
    assert v["rounds_in_prefix"] == 2 and not v["passed"]
    assert S1.rule_i_from_rounds([{"round": 1, "requests": 100, "valid_rate": 0.0}]) is None
    print("global_fill and rule_i_from_rounds: as specified")


def test_pipeline():
    import pandas as pd
    import pipeline_ctcl_r1 as P1
    tmp = Path(tempfile.mkdtemp(prefix="ctcl_r1_test_", dir=os.environ.get("CTCL_TEST_TMP") or None))
    calls = []

    def runner(stage, cmd, ok_codes=(0,)):
        calls.append((stage, [str(c) for c in cmd]))
        out = Path(cmd[cmd.index("--out") + 1])
        pd.DataFrame({"text": ["x"], "finished": [True], "topic": [0]}).to_parquet(out)
        out.with_suffix(".json").write_text(json.dumps({"generation": {"ctcl": {"rev": "r1"}}}))
        return 0

    def job(name):
        a = P1.PC.parse(["--method", "ctcl", "--dataset", "bandung", "--eps", "1", "--out",
                         str(tmp / name / "rel.parquet"), "--work", str(tmp / name / "w")])
        j = P1.JobR1(a, runner=runner, log=open(os.devnull, "w"))
        j.m2_status = lambda: "done"
        return j
    j = job("palembang_like")
    j.failed.write_text(json.dumps({"why": "valid rate 0.0001 < 0.02 after 57456 samples: model is broken",
                                    "rounds": [{"round": 1, "requests": 57456, "valid_rate": 5e-05}]}))
    assert j.do_sample() == P1.EXIT_BROKEN and not calls
    rec = json.loads(j.failed_r1.read_text())
    assert rec["from_v1_record"] and not rec["r1_gate"]["passed"] and j.failed.exists()
    assert j.do_sample() == P1.EXIT_BROKEN and not calls
    j = job("melbourne_like")
    j.failed.write_text(json.dumps({"why": "valid rate 0.0199 < 0.02 after 1410603 samples: model is broken",
                                    "rounds": [{"round": 1, "requests": 57550, "valid_rate": 0.0437},
                                               {"round": 2, "requests": 150332, "valid_rate": 0.03224}]}))
    assert j.do_sample() == 0 and len(calls) == 1 and calls[0][1][3].endswith("r1/sample_ctcl_r1.py")
    assert not j.failed.exists() and j.failed_v1_kept.exists() and j.texts.exists()
    assert j.do_sample() == 0 and len(calls) == 1
    j = job("r1_failed")
    j.failed_r1.write_text("{}")
    assert j.do_sample() == P1.EXIT_BROKEN and len(calls) == 1
    j = job("finish_symlink")
    tgt = tmp / "offload" / "model"
    tgt.mkdir(parents=True)
    (tgt / "w.safetensors").write_text("x")
    j.wdir.symlink_to(tgt)
    j.rel.parent.mkdir(parents=True, exist_ok=True)
    j.rel.write_text("r")
    j.rel.with_suffix(".json").write_text("{}")
    j.finish()
    assert not tgt.exists() and not j.wdir.is_symlink() and not j.wdir.exists() and j.out.exists()
    print("pipeline_ctcl_r1: v1 record failing rule (i) -> failed_r1 without sampling; passing -> moved aside and "
          "re-sampled with the r1 sampler; failed_r1 final; symlinked weights deleted at their target")


def file_hashes() -> dict:
    return {f: hashlib.sha256((PROJECT / f).read_bytes()).hexdigest() for f in FILES if (PROJECT / f).exists()}


if __name__ == "__main__":
    t0 = time.time()
    before = file_hashes()
    rows = []
    for t in (test_global_fill, test_rule_ii, test_rule_i, test_equivalence, test_pipeline):
        t1 = time.time()
        t()
        rows.append({"test": t.__name__, "sec": round(time.time() - t1, 1)})
    after = file_hashes()
    assert before == after, "files changed while testing"
    res = {"finished": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"), "all_passed": True,
           "tests": rows, "code_sha256": after}
    (R1 / "RESULTS_r1.json").write_text(json.dumps(res, indent=1) + "\n")
    print(f"test_r1: all passed ({time.time() - t0:.0f}s); wrote {R1 / 'RESULTS_r1.json'}")
