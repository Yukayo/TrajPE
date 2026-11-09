from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))
import ctcl_testlib as TL

C = TL.C
sys.path.insert(0, str(TL.CTCL))
import calibrate as CB
import pipeline_ctcl as PL

TINY = TL.SCRATCH / "tiny_model"
ROOT = TL.SCRATCH / "release"


def e2e(method: str, eps: int) -> tuple:
    name = f"{method}_eps{eps}"
    root = ROOT / name
    shutil.rmtree(root, ignore_errors=True)
    TL.fake_dataset(root / "data", "bandung", 500, seed=11)
    os.environ.update(TRAJPE_DATA=str(root / "data"), CTCL_CALIB_CACHE=str(ROOT / "_calib"), OMP_NUM_THREADS="8",
                      MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", CUDA_VISIBLE_DEVICES="")
    out = root / "out" / f"bandung_eps{eps}_s0.parquet"
    argv = ["--method", method, "--dataset", "bandung", "--eps", str(eps), "--out", str(out), "--work",
            str(root / "work"), "--test_data_ok", "--extra_train", f"--model {TINY} --device cpu"]
    (root / "work").mkdir(parents=True)
    logf = open(root / "work" / "test.log", "a")
    real = PL.Runner(logf)

    def runner(stage, cmd, ok_codes=(0,)):
        cmd = [str(c) for c in cmd]
        if stage == "sample":
            cmd = [C.TRAJPE_PY, "-I", "-B", str(TESTS / "fake_sampler.py")] + cmd[cmd.index("--dataset"):]
        return real(stage, cmd, ok_codes)
    t0 = time.time()
    rc = PL.Job(PL.parse(argv), runner=runner, log=logf)()
    assert rc == 0, (rc, (root / "work" / "test.log").read_text()[-3000:])
    return root, out, time.time() - t0


def check_release(root: Path, out: Path, method: str, eps: int):
    work = root / "work"
    side = json.loads(out.with_suffix(".json").read_text())
    gate = json.loads((work / "gate.json").read_text())
    assert gate["ok"] and gate["validity_ok"] and gate["accounting_ok"] and gate["pld_check_ok"], gate
    assert not (work / "model").exists() and not (work / "release" / out.name).exists()
    df = pd.read_parquet(out)
    assert list(df.columns) == ["trail_id", "user_id", "venue_id", "lat", "lon", "category", "t_local", "pos"]
    assert df["trail_id"].nunique() == 50000 == side["n_trails"]
    lens = df.groupby("trail_id").size()
    assert lens.min() >= 2 and lens.max() <= 20
    assert side["method"] == method and side["eps"] == eps == side["eps_target"] and side["delta"] == 1e-6
    for k in ("private_stats", "log", "composite_vs_train", "composite_vs_test", "response_tokens"):
        assert k not in side, k
    acc = side["ctcl"]["accounting"]
    names = [m["name"] for m in acc["mechanisms"]]
    assert acc["eps_composed_pld"] <= eps and acc["eps_target"] == eps and acc["delta"] == 1e-6
    assert all("sensitivity_l2" in m and "eps_alone_pld" in m for m in acc["mechanisms"])
    assert names[:2] == ["M2 DP-Adam (full fine-tuning)", "M1 DP topic histogram"]
    assert side["noise_multiplier"] == json.loads((work / "calib.json").read_text())["sigma_sgd"]
    assert side["ctcl"]["sampling"]["reapportionment"] is not None and side["sampling"]["valid_rate"] > 0
    assert set(side["ctcl"]["code_sha256"]) >= set(C.CODE_FILES)
    r = subprocess.run([C.TRAJPE_PY, "-I", "-B", str(TL.CTCL / "check.py"), str(out.with_suffix(".json"))],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-1500:]
    if method == "ctclvp":
        assert len(names) == 3 and names[2].startswith("M3")
        pc = json.loads((work / "prior_calib.json").read_text())
        assert pc["sigma_hist"] == CB.sigma_prior_for(float(eps)) and pc["n_hist"] == 2
        g = side["grounding"]["dp_prior"]
        assert g["sigma_hist"] == pc["sigma_hist"]
        assert g["grounding_v2"] == {"ground_scale_km": 0.05, "pop_radius_km": C.pop_radius_km(eps)}
        assert side["vp_calibration"]["sigma_hist"] == side["sigma_eq"] != pc["sigma_hist"]
    else:
        assert side["grounding"]["dp_prior"] is None and side["vp_calibration"]["sigma_hist"] == 10.0
    return side


def test_ground_determinism(root: Path):
    texts = pd.read_parquet(root / "work" / "texts.parquet").iloc[:2000]
    f = root / "det_texts.parquet"
    texts.to_parquet(f, index=False)
    outs = []
    for i in range(2):
        o = root / f"det_{i}" / "bandung_eps4_s0.parquet"
        r = subprocess.run([C.TRAJPE_PY, "-I", "-B", str(C.DPLORA / "ground.py"), "--dataset", "bandung", "--texts",
                            str(f), "--out", str(o), "--seed", "0", "--jitter", "0", "--n_out", "2000", "--method",
                            "ctcl"], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-1500:]
        outs.append(pd.read_parquet(o))
    pd.testing.assert_frame_equal(outs[0], outs[1])


if __name__ == "__main__":
    t0 = time.time()
    assert (TINY / "config.json").exists(), "run test_train.py first (it saves the tiny model)"
    for method, eps in (("ctcl", 4), ("ctclvp", 1)):
        root, out, sec = e2e(method, eps)
        side = check_release(root, out, method, eps)
        print(f"{method} eps {eps}: release valid ({sec:.0f}s end to end on CPU); composed PLD eps "
              f"{side['ctcl']['accounting']['eps_composed_pld']:.5f}, PRV {side['eps_prv']:.4f}; weights deleted")
        rc = PL.main(["--method", method, "--dataset", "bandung", "--eps", str(eps), "--out", str(out), "--work",
                      str(root / "work"), "--test_data_ok"])
        assert rc == 0
    test_ground_determinism(ROOT / "ctcl_eps4")
    print(f"test_release: all passed ({time.time() - t0:.0f}s)")
