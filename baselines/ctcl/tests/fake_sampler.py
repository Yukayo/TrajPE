from __future__ import annotations

import os
import sys
import time
from pathlib import Path

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))
import ctcl_testlib as TL

sys.path.insert(0, str(TL.CTCL))
import sample_ctcl as SC


def main():
    a = SC.parse(sys.argv[1:])
    t0 = time.time()
    if SC.failed_marker(a).exists():
        print("failed marker exists", flush=True)
        sys.exit(3)
    hist, _, n, w, cats = SC.load_inputs(a)
    rate = float(os.environ.get("CTCL_FAKE_RATE", "0.85"))
    gen = TL.FakeGenerator({k: rate for k in range(len(n))}, TL.public_catalog(a.dataset), seed=a.seed)
    sys.exit(SC.synthesize(a, hist, n, w, cats, gen, {"model": a.model_dir or a.adapter, "sampler": "fake (test)"},
                           t0, lambda: {"generated_tokens": 0, "max_mem_gb": None}))


if __name__ == "__main__":
    main()
