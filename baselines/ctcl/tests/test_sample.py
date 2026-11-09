from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))
import ctcl_testlib as TL

C = TL.C
sys.path.insert(0, str(TL.CTCL))
import sample_ctcl as SC
import topics as TP

textfmt = C.textfmt


def quiet(*a, **k):
    pass


def quotas_like_small_dataset(seed=0):
    rng = np.random.default_rng(seed)
    h = np.zeros(1440)
    occ = rng.choice(1440, 300, replace=False)
    h[occ] = rng.gamma(0.6, 12.0, 300)
    return TP.quotas(h + rng.normal(0, 10, 1440))


def test_heterogeneous():
    pois = TL.public_catalog("bandung")
    cats = set(pois["category"])
    w, n = quotas_like_small_dataset(0)
    rng = np.random.default_rng(11)
    rates = {k: float(rng.choice([0.9, 0.3, 0.08, 0.01, 0.0], p=[0.6, 0.25, 0.1, 0.03, 0.02])) for k in range(1440)}
    res = []
    for _ in range(2):
        gen = TL.FakeGenerator(rates, pois, seed=1)
        texts, st = SC.run_rounds(n, w, gen, cats, log_fn=quiet)
        res.append((texts, st))
    (texts, st), (texts2, st2) = res
    assert texts == texts2 and st["reasons"] == st2["reasons"], "not deterministic under the seed"
    assert len(texts) == 50000
    assert all(textfmt.clean(t, cats, True)[0] == t for t, _ in texts[:2000])
    assert st["valid_rate"] >= 0.02 and set(st) >= {"n_generated", "n_ok", "valid_rate", "reasons", "rounds",
                                                     "extra_rounds", "reapportionment", "dropped_lines_in_kept"}
    got = np.bincount([k for _, k in texts], minlength=1440)
    rp = st["reapportionment"]
    U = [k for k in range(1440) if n[k] > 0 and got[k] < n[k]]
    F = [k for k in range(1440) if n[k] > 0 and got[k] >= n[k]]
    D = int(sum(n[k] - got[k] for k in U))
    assert D == rp["deficit"] > 0 and len(U) == rp["unfilled_topics"]
    assert all(got[k] == 0 for k in U if rates[k] == 0.0)
    assert all(rates[k] in (0.0, 0.01, 0.08) for k in U), "a topic with a high valid rate stayed unfilled"
    from trajpe.ablations import apportion
    want = apportion(np.array([float(w[k]) for k in F]), D)
    assert np.array_equal(got[F] - n[F], want), "the deficit is not re-apportioned in proportion to w"
    assert rp["from_surplus"] + rp["from_extra_rounds"] == D and rp["from_surplus"] > 0
    t1 = time.time()
    rates5 = {k: [0.9, 0.3, 0.08, 0.01, 0.0][k % 5] for k in range(1440)}
    try:
        SC.run_rounds(n, w, TL.FakeGenerator(rates5, pois, seed=1), cats, log_fn=quiet)
        raise AssertionError("expected exit 3")
    except SC.BrokenModel as e:
        assert e.stats["valid_rate"] < 0.02 and e.stats["n_generated"] >= 20000
        assert all(r["valid_rate"] >= 0.02 for r in e.stats["rounds"][:-1]), "exit 3 came late"
        print(f"equal shares incl. 40 % near-dead topics: exit 3 after {e.stats['n_generated']} samples, "
              f"{len(e.stats['rounds'])} rounds, overall valid rate {e.stats['valid_rate']:.4f} "
              f"({time.time() - t1:.0f}s)")
    print(f"heterogeneous rates: 50,000 texts, {st['n_generated']} generated, overall valid rate "
          f"{st['valid_rate']:.3f}, {len(st['rounds'])} rounds; deficit {rp['deficit']} over {rp['unfilled_topics']} "
          f"unfilled topics re-apportioned ({rp['from_surplus']} from surplus, {rp['from_extra_rounds']} from "
          f"{len(st['extra_rounds'])} extra rounds); deterministic")


def test_broken():
    pois = TL.public_catalog("bandung")
    w, n = quotas_like_small_dataset(1)
    gen = TL.FakeGenerator({k: 0.01 for k in range(1440)}, pois, seed=2)
    try:
        SC.run_rounds(n, w, gen, set(pois["category"]), log_fn=quiet)
        raise AssertionError("a uniformly broken model did not exit 3")
    except SC.BrokenModel as e:
        assert e.stats["n_generated"] >= 20000 and e.stats["valid_rate"] < 0.02 and "broken" in e.why
        print(f"rate 0.01: exit 3 after {e.stats['n_generated']} samples (valid rate {e.stats['valid_rate']:.4f})")
    gen = TL.FakeGenerator({k: (0.9 if k == 7 else 0.0) for k in range(1440)}, pois, seed=3)
    n1 = np.zeros(1440, np.int64)
    n1[7], n1[8] = 30, 20
    texts, st = SC.run_rounds(n1, n1.copy(), gen, set(pois["category"]), log_fn=quiet)
    assert len(texts) == 50 and all(k == 7 for _, k in texts) and st["reapportionment"]["deficit"] == 20


def test_sizing():
    m, v = {0: 1}, 0.8
    gen, ok, seq = {0: 0}, {0: 0}, []
    for _ in range(6):
        r = SC.sizing(m, gen, ok, v)[0]
        seq.append(r)
        gen[0] += r
    assert seq[:5] == [2, 4, 10, 21, 21], seq
    big = {k: 1000 for k in range(400)}
    req = SC.sizing(big, {k: 0 for k in big}, {k: 0 for k in big}, 0.9)
    assert sum(req.values()) <= C.ROUND_CAP + len(big) and sum(req.values()) >= C.ROUND_CAP
    print("sizing: spec example 2, 4, 10, 21, 21; round cap 150,000 enforced")


def test_length_cap():
    from transformers import AutoTokenizer
    os.environ.setdefault("HF_HOME", C.HF_HOME)
    tok = AutoTokenizer.from_pretrained(C.MODEL, revision=C.MODEL_COMMIT)
    end = tok.convert_tokens_to_ids("<|im_end|>")
    rng = np.random.default_rng(0)
    worst, n_checked = 0, 0
    for d in C.TABLE_DATASETS:
        pois = TL.public_catalog(d)
        w = pois["category"].value_counts()
        cats = set(pois["category"])
        for i in range(60):
            text = TL.render_trail(rng, pois, 20)
            ids = tok(text, add_special_tokens=False)["input_ids"] + [end]
            worst = max(worst, len(ids))
            assert len(ids) <= C.MAX_RESP_TOKENS, (d, len(ids))
            assert ids[:C.MAX_RESP_TOKENS] == ids
            cl, why, nd = textfmt.clean(text, cats, True)
            assert why == "ok" and nd == 0 and len(cl.split("\n")) == 21
            n_checked += 1
            if i == 0:
                long = TL.render_trail(rng, pois, 30)
                cut_ids = tok(long, add_special_tokens=False)["input_ids"][:C.MAX_RESP_TOKENS]
                cut = tok.decode(cut_ids)
                complete = cut.count("\n")
                assert complete - 1 >= 20, (d, complete)
                cl, why, _ = textfmt.clean(cut, cats, finished=False)
                assert why == "ok" and len(cl.split("\n")) - 1 == complete - 1
                parsed, why = textfmt.parse(cl, cats, True)
                assert why == "ok" and len(parsed[1]) == 20
                short = "\n".join(long.split("\n")[:16]) + "\n12:3"
                cl, why, _ = textfmt.clean(short, cats, finished=False)
                assert cl is None and why == "unfinished"
                lines = text.split("\n")
                lines[7] = "25:99 | Not A Category | 1.0,2.0"
                cl, why, nd = textfmt.clean("\n".join(lines), cats, True)
                assert why == "ok" and nd == 1 and len(cl.split("\n")) == 20
    assert w is not None
    print(f"length cap: {n_checked} synthetic 20-check-in texts over 19 catalogs, longest {worst} <= 640 Qwen tokens; "
          f"cut / unfinished / malformed-line rules as specified")


if __name__ == "__main__":
    t0 = time.time()
    test_sizing()
    test_heterogeneous()
    test_broken()
    test_length_cap()
    print(f"test_sample: all passed ({time.time() - t0:.0f}s)")
