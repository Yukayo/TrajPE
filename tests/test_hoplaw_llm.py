import os
import shutil
import sys
import tempfile
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="trajpe_hoplaw_")
os.environ["TRAJPE_WORK"] = _TMP
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from trajpe import workload as W
from trajpe.api_llm import LLMAPI
from trajpe.api_sim import SimulatorRandom
from test_v31 import _SimCat


class _FakeEngine:
    model = "fake"
    n_prompt_tokens = n_output_tokens = 0

    def chat(self, convs, **kw):
        return ['{"checkins": []}'] * len(convs)


def _cat():
    c = _SimCat(np.random.default_rng(5))
    c.name = "jakarta"
    c.lat = c.xy[:, 1] / 111.0
    c.lon = c.xy[:, 0] / 111.0
    return c


def _law():
    probs = np.zeros(len(W.HOP_EDGES) - 1)
    probs[len(probs) // 2] = 1.0
    return W.HOP_EDGES, probs, 28.3


def _run(cat, n, seed, law=None, sim_frac=0.5):
    shutil.rmtree(Path(_TMP) / "runs", ignore_errors=True)
    api = LLMAPI(cat, sim_frac=sim_frac, engine=_FakeEngine())
    if law is not None:
        api.set_hop_law(*law)
    rng = np.random.default_rng(seed)
    ts = api.random(n, rng)
    assert api.fallback.hop_law is None
    return ts.to_lists(), int(np.random.default_rng(seed).integers(0, 2**31 - 1)), api


def _same(a, b):
    return len(a) == len(b) and all(np.array_equal(p, q) and np.array_equal(s, t) for (p, s), (q, t) in zip(a, b))


def test_llmapi_has_set_hop_law():
    assert hasattr(LLMAPI(_cat(), engine=_FakeEngine()), "set_hop_law")


def test_sim_share_uses_dp_hop_law():
    cat, n = _cat(), 120
    n_sim = int(round(n * 0.5))
    with_law, sub_seed, _ = _run(cat, n, 3, _law())
    without, sub_seed2, _ = _run(cat, n, 3)
    assert sub_seed == sub_seed2
    ref = SimulatorRandom(cat)
    ref.set_hop_law(*_law())
    exp_law = ref(n_sim, np.random.default_rng(sub_seed + 1)).to_lists()
    exp_frozen = SimulatorRandom(cat)(n_sim, np.random.default_rng(sub_seed + 1)).to_lists()
    assert _same(with_law[n - n_sim:], exp_law)
    assert _same(without[n - n_sim:], exp_frozen)
    assert not _same(exp_law, exp_frozen)
    assert _same(with_law[:n - n_sim], without[:n - n_sim])


def test_llm_only_unaffected():
    cat = _cat()
    a, _, _ = _run(cat, 40, 7, _law(), sim_frac=0.0)
    b, _, _ = _run(cat, 40, 7, None, sim_frac=0.0)
    assert _same(a, b)


if __name__ == "__main__":
    try:
        test_llmapi_has_set_hop_law()
        test_sim_share_uses_dp_hop_law()
        test_llm_only_unaffected()
    finally:
        shutil.rmtree(_TMP, ignore_errors=True)
    print("all LLMAPI hop-law tests passed")
