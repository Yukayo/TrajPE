from __future__ import annotations

import collections
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

ROOT = TL.SCRATCH / "pipeline"


def arg(cmd, flag):
    return Path(cmd[cmd.index(flag) + 1])


class Stub:

    def __init__(self, fail=None, sample_exit3=False, m2_crash_first=False, m2_oom_first=False, swap_prior=False):
        self.calls = collections.Counter()
        self.fail, self.sample_exit3 = fail, sample_exit3
        self.m2_crash_first, self.m2_oom_first, self.swap_prior = m2_crash_first, m2_oom_first, swap_prior
        self.prior_calib_seen = None
        self.gate_seq = [True]

    def __call__(self, stage, cmd, ok_codes=(0,)):
        cmd = [str(c) for c in cmd]
        self.calls[stage] += 1
        getattr(self, stage)(cmd)
        if self.fail == stage and self.calls[stage] == 1:
            raise subprocess.CalledProcessError(1, cmd)
        if stage == "sample" and self.sample_exit3:
            return 3
        return 0

    def calib(self, cmd):
        out = arg(cmd, "--out")
        if out.exists():
            return
        eps, variant = float(cmd[cmd.index("--eps") + 1]), cmd[cmd.index("--variant") + 1]
        sp = CB.sigma_prior_for(eps) if variant == "vp" else None
        cal = {"variant": variant, "config": "code", "eps_total": eps, "delta": 1e-6, "pld_target": eps - 0.015,
               "sample_rate": 128 / C.N_REF, "steps": 1166, "b_nom": 128, "clip": 1.0,
               "sigma_sgd": C.SIGMA_REF[("code", variant)][int(eps)], "sigma_topic_hist": 10.0, "n_topic_hist": 1,
               "sigma_prior": sp, "n_prior": 2 if sp else 0, "prior_mu2_share": 0.1 if sp else None,
               "sigma_eq": CB.sigma_eq(sp), "eps_composed_pld": eps - 0.016, "eps_sgd_only_pld": eps - 0.05,
               "eps_topic_hist_only_pld": 0.3969, "eps_prior_only_pld": 0.1 if sp else None, "accountant": "stub"}
        C.write_json_atomic(out, cal)
        if "--prior_out" in cmd:
            pc = CB.prior_calib(cal)
            if self.swap_prior:
                pc["sigma_hist"] = cal["sigma_eq"]
            C.write_json_atomic(arg(cmd, "--prior_out"), pc)

    def m1(self, cmd):
        out = arg(cmd, "--out")
        sha = C.write_json_atomic(out, {"dataset": cmd[cmd.index("--dataset") + 1], "noise_stream": [0, 20261026],
                                        "quota_sha256": "q", "n_topics_with_quota": 5, "table_sha256": "t",
                                        "quota": [10000] * 5})
        out.with_name(out.name + ".sha256").write_text(sha + "\n")

    def m2(self, cmd):
        w = arg(cmd, "--out_dir")
        (w / "model").mkdir(parents=True, exist_ok=True)
        (w / "model" / "model.safetensors").write_bytes(os.urandom(64))
        if self.m2_crash_first and self.calls["m2"] == 1:
            raise subprocess.CalledProcessError(1, cmd)
        oom = 1 if (self.m2_oom_first and self.calls["m2"] == 1) else 0
        C.write_json_atomic(w / "train.json", {
            "complete": True, "steps_run": 1166, "steps": 1166, "eps_target": float(cmd[cmd.index("--eps") + 1]),
            "private_stats": {"oom_retries": oom},
            "weights": {"dir": "model", "files": {"model.safetensors": C.sha256_file(w / "model" / "model.safetensors")}}})

    def m3(self, cmd):
        self.prior_calib_seen = json.loads(arg(cmd, "--calib").read_text())
        arg(cmd, "--out").write_bytes(b"prior")

    def sample(self, cmd):
        out = arg(cmd, "--out")
        assert arg(cmd, "--model_dir").exists(), "sampling without weights"
        if self.sample_exit3:
            C.write_json_atomic(out.with_name(out.stem + ".failed.json"), {"why": "broken", "valid_rate": 0.01})
            return
        pd.DataFrame({"text": ["x"], "finished": [True], "topic": [0]}).to_parquet(out)
        C.write_json_atomic(out.with_suffix(".json"), {"generation": {"ctcl": {"rounds": []}}, "valid_rate": 0.8})

    def ground(self, cmd):
        out = arg(cmd, "--out")
        pd.DataFrame({"trail_id": ["a"]}).to_parquet(out)
        side = json.loads(arg(cmd, "--train_json").read_text())
        side.pop("private_stats", None)
        side.update({"method": cmd[cmd.index("--method") + 1], "eps": side.get("eps_target")})
        C.write_json_atomic(out.with_suffix(".json"), side)


class GateJob(PL.Job):
    def do_gate(self):
        seq = self.run.gate_seq
        ok = seq.pop(0) if len(seq) > 1 else seq[0]
        res = {"ok": ok}
        C.write_json_atomic(self.w / "gate.json", res)
        return res


def job_args(name, method="ctcl", dataset="bandung", eps=1):
    d = ROOT / name
    return ["--method", method, "--dataset", dataset, "--eps", str(eps), "--out", str(d / f"{dataset}_eps{eps}_s0.parquet"),
            "--work", str(d / "work")]


def run(argv, stub):
    import io
    return GateJob(PL.parse(argv), runner=stub, log=io.StringIO())()


def test_failure_after_each_stage():
    for method in ("ctcl", "ctclvp"):
        for stage in ("calib", "m1", "m2", "m3", "sample", "ground", "gate"):
            if stage == "m3" and method == "ctcl":
                continue
            name = f"{method}_{stage}"
            shutil.rmtree(ROOT / name, ignore_errors=True)
            argv = job_args(name, method)
            stub = Stub(fail=stage if stage != "gate" else None)
            stub.gate_seq = [False, True] if stage == "gate" else [True]
            rcs = []
            for attempt in range(3):
                try:
                    rc = run(argv, stub)
                except subprocess.CalledProcessError:
                    rc = "crash"
                rcs.append(rc)
                w = ROOT / name / "work"
                if rc != 0 and stage in ("sample", "ground", "gate"):
                    assert (w / "model" / "model.safetensors").exists(), f"{name}: weights lost after a {stage} failure"
                if rc == 0:
                    break
            assert rcs[-1] == 0 and rcs[0] != 0, (name, rcs)
            assert stub.calls["m1"] == 1 and stub.calls["m2"] == 1, (name, stub.calls)
            assert stub.calls["m3"] == (1 if method == "ctclvp" else 0), (name, stub.calls)
            out = ROOT / name / "bandung_eps1_s0.parquet"
            assert out.exists() and out.with_suffix(".json").exists()
            assert json.loads(out.with_suffix(".json").read_text())["ctcl"]["method_key"] == method
            assert not (ROOT / name / "work" / "model").exists(), "weights kept after a passed gate"
            assert run(argv, stub) == 0 and stub.calls["m2"] == 1
            if method == "ctclvp":
                pc = stub.prior_calib_seen
                assert pc["sigma_hist"] == CB.sigma_prior_for(1.0) and pc["n_hist"] == 2
    print("failure after each stage: M1 / M2 / M3 ran exactly once, weights kept until the gate passed")


def test_exit3_marker():
    name = "exit3"
    shutil.rmtree(ROOT / name, ignore_errors=True)
    stub = Stub(sample_exit3=True)
    assert run(job_args(name), stub) == 3
    assert (ROOT / name / "work" / "texts.failed.json").exists() and stub.calls["sample"] == 1
    for _ in range(2):
        assert run(job_args(name), stub) == 3
    assert stub.calls["sample"] == 1 and stub.calls["m2"] == 1
    assert (ROOT / name / "work" / "model").exists(), "weights of a failed job are kept for the owner"
    print("exit 3: texts.failed.json stops the queue's retries without re-sampling")


def test_unusable_training():
    for kind in ("crash", "oom"):
        name = f"m2_{kind}"
        shutil.rmtree(ROOT / name, ignore_errors=True)
        stub = Stub(m2_crash_first=kind == "crash", m2_oom_first=kind == "oom")
        try:
            rc = run(job_args(name), stub)
        except subprocess.CalledProcessError:
            rc = "crash"
        if kind == "oom":
            assert rc == PL.EXIT_STATE, rc
        rc = run(job_args(name), stub)
        assert rc == 0 and stub.calls["m2"] == 2 and stub.calls["m1"] == 1
        aside = [p.name for p in (ROOT / name / "work").iterdir() if ".unusable_" in p.name]
        assert aside and all(not p.startswith("texts") for p in aside), aside
    name = "weights_gone"
    shutil.rmtree(ROOT / name, ignore_errors=True)
    stub = Stub(fail="sample")
    try:
        run(job_args(name), stub)
    except subprocess.CalledProcessError:
        pass
    shutil.rmtree(ROOT / name / "work" / "model")
    assert run(job_args(name), stub) == PL.EXIT_STATE and stub.calls["m2"] == 1
    name = "hist_tampered"
    shutil.rmtree(ROOT / name, ignore_errors=True)
    stub = Stub(fail="m2")
    try:
        run(job_args(name), stub)
    except subprocess.CalledProcessError:
        pass
    h = ROOT / name / "work" / "hist.json"
    h.write_text(h.read_text().replace("10000", "10001"))
    assert run(job_args(name), stub) == PL.EXIT_STATE and stub.calls["m1"] == 1
    print("unusable training renamed aside and restarted; vanished weights / tampered hist.json stop the job")


def test_prior_keys_and_guards():
    name = "swapped"
    shutil.rmtree(ROOT / name, ignore_errors=True)
    stub = Stub(swap_prior=True)
    try:
        run(job_args(name, "ctclvp"), stub)
        raise AssertionError("a prior_calib with sigma_eq was used")
    except (AssertionError, SystemExit) as e:
        assert "sigma" in str(e) or "differs" in str(e), e
    assert stub.calls["m3"] == 0
    bad = [job_args("g1", dataset="bandung") + ["--log_private"], job_args("g2", dataset="gw_austin"),
           job_args("g3", dataset="gw_nyc")]
    for argv in bad:
        try:
            PL.parse(argv)
            raise AssertionError(f"accepted {argv}")
        except SystemExit:
            pass
    PL.parse(job_args("g5", dataset="jakarta") + ["--log_private"])
    os.environ["TRAJPE_DATA"] = "/tmp/x"
    try:
        PL.parse(job_args("g6"))
        raise AssertionError("TRAJPE_DATA accepted")
    except SystemExit:
        pass
    finally:
        os.environ.pop("TRAJPE_DATA")
    print("vp.py prior gets sigma_hist = sigma_prior (swapped file refused before M3); argument guards hold")


if __name__ == "__main__":
    t0 = time.time()
    shutil.rmtree(ROOT, ignore_errors=True)
    test_failure_after_each_stage()
    test_exit3_marker()
    test_unusable_training()
    test_prior_keys_and_guards()
    print(f"test_pipeline: all passed ({time.time() - t0:.0f}s)")
