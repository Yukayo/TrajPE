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

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ctcl_conf as C

EXIT_BROKEN, EXIT_GATE, EXIT_STATE = 3, 4, 5
MIN_FREE_GB = 4.0


class StateError(Exception):
    pass


class Runner:

    def __init__(self, log):
        self.log = log

    def __call__(self, stage: str, cmd: list, ok_codes=(0,)) -> int:
        cmd = [str(c) for c in cmd]
        self.log.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [{stage}] $ {' '.join(cmd)}\n")
        self.log.flush()
        env = dict(os.environ)
        env["PATH"] = str(Path(cmd[0]).parent) + os.pathsep + env.get("PATH", "")
        r = subprocess.run(cmd, stdout=self.log, stderr=subprocess.STDOUT, env=env)
        if r.returncode not in ok_codes:
            raise subprocess.CalledProcessError(r.returncode, cmd)
        return r.returncode


def stamp() -> str:
    return time.strftime("%m%d_%H%M%S")


def read_json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


class Job:
    def __init__(self, a, runner=None, log=None):
        self.a = a
        self.variant, self.config, self.param = C.METHODS[a.method]
        self.vp = self.variant == "vp"
        self.out = Path(a.out) if a.out else C.release_path(a.method, a.dataset, a.eps, a.seed)
        self.work = Path(a.work) if a.work else self.out.parent / "_work" / self.out.stem
        self.work.mkdir(parents=True, exist_ok=True)
        self.logf = log or open(self.work / "pipeline.log", "a")
        self.run = runner or Runner(self.logf)
        self.state_f = self.work / "state.json"
        self.state = read_json(self.state_f) or {"stages": {}}
        self.w = self.work
        self.calib, self.prior_calib = self.w / "calib.json", self.w / "prior_calib.json"
        self.hist, self.tj = self.w / "hist.json", self.w / "train.json"
        self.wdir = self.w / ("adapter" if self.param == "lora" else "model")
        self.prior, self.texts = self.w / "prior.npz", self.w / "texts.parquet"
        self.failed = self.w / "texts.failed.json"
        self.rel = self.w / "release" / self.out.name
        self.table = Path(a.table) if a.table else C.table_path(a.dataset)

    def note(self, stage: str, **kw):
        self.state["stages"].setdefault(stage, {}).update(kw, time=time.strftime("%Y-%m-%d %H:%M:%S"))
        C.write_json_atomic(self.state_f, self.state)

    def say(self, msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.logf.write(line + "\n")
        self.logf.flush()

    def wait_for_disk(self, need_gb: float = MIN_FREE_GB, wait_min: int = 30):
        for i in range(wait_min // 5 + 1):
            free = shutil.disk_usage(self.w).free / 1e9
            if free >= need_gb:
                return
            self.say(f"only {free:.1f} GB free on the work disk (< {need_gb}); waiting")
            time.sleep(300)
        raise StateError(f"less than {need_gb} GB free on the work disk")

    def timed(self, stage: str, cmd: list, ok_codes=(0,)) -> int:
        t0 = time.time()
        rc = self.run(stage, cmd, ok_codes)
        self.note(stage, sec=round(time.time() - t0, 1), rc=rc)
        return rc

    def do_calib(self):
        cmd = [C.TRAJPE_PY, "-I", "-B", HERE / "calibrate.py", "--eps", C.eps_tag(self.a.eps), "--variant",
               self.variant, "--config", self.config, "--out", self.calib]
        if self.vp:
            cmd += ["--prior_out", self.prior_calib]
        self.timed("calib", cmd)

    def do_m1(self):
        sha_f = self.hist.with_name(self.hist.name + ".sha256")
        if self.hist.exists():
            want = (sha_f.read_text().split()[0] if sha_f.exists() else None) or \
                self.state["stages"].get("m1", {}).get("sha256")
            if want is None or C.sha256_file(self.hist) != want:
                raise StateError(f"{self.hist} exists but its SHA-256 does not match the record")
            self.say("M1: hist.json exists (reused, never redrawn)")
            return
        self.timed("m1", [C.TRAJPE_PY, "-I", "-B", HERE / "topics.py", "hist", "--dataset", self.a.dataset,
                          "--seed", self.a.seed, "--table", self.table, "--out", self.hist])
        self.note("m1", sha256=C.sha256_file(self.hist))

    def m2_status(self) -> str:
        tj = read_json(self.tj)
        if tj is None:
            return "unusable" if self.wdir.exists() else "absent"
        if not tj.get("complete") or (tj.get("private_stats") or {}).get("oom_retries", 0):
            return "unusable"
        if self.state["stages"].get("weights_deleted"):
            return "consumed"
        files = (tj.get("weights") or {}).get("files") or {}
        if not files or not self.wdir.exists():
            raise StateError(f"{self.tj} is complete but its weights are missing (not deleted by the gate)")
        for name, h in files.items():
            if C.sha256_file(self.wdir / name) != h:
                raise StateError(f"{self.wdir / name}: SHA-256 differs from train.json")
        return "done"

    def do_m2(self):
        st = self.m2_status()
        if st in ("done", "consumed"):
            self.say(f"M2: train.json complete ({st}); training skipped")
            return
        if st == "unusable":
            s = stamp()
            for p in (self.tj, self.wdir):
                if p.exists():
                    p.rename(p.with_name(f"{p.name}.unusable_{s}"))
            for p in (self.texts, self.texts.with_suffix(".json"), self.failed):
                if p.exists():
                    p.rename(p.with_name(f"{p.name}.unusable_{s}"))
            self.say(f"M2: an unusable earlier run was renamed aside (*.unusable_{s}); restarting from step 0")
        self.wait_for_disk()
        cmd = [C.DPLORA_PY, "-I", "-B", HERE / "train_ctcl.py", "--dataset", self.a.dataset, "--eps",
               C.eps_tag(self.a.eps), "--variant", self.variant, "--config", self.config, "--param", self.param,
               "--seed", self.a.seed, "--calib", self.calib, "--table", self.table, "--out_dir", self.w,
               "--mem_cap_gb", self.a.mem_cap_gb]
        if self.a.log_private:
            cmd.append("--log_private")
        cmd += self.a.extra_train.split()
        self.timed("m2", cmd)
        if self.m2_status() != "done":
            raise StateError("training finished without a complete train.json and weights")

    def do_m3(self):
        if not self.vp:
            return
        if self.prior.exists():
            self.say("M3: prior.npz exists (reused)")
            return
        sys.path.insert(0, str(HERE))
        import calibrate as CB
        CB.check_prior_calib(json.loads(self.prior_calib.read_text()), json.loads(self.calib.read_text()))
        self.timed("m3", [C.TRAJPE_PY, "-I", "-B", C.DPLORA / "vp.py", "prior", "--dataset", self.a.dataset,
                          "--calib", self.prior_calib, "--seed", self.a.seed, "--out", self.prior])

    def do_sample(self) -> int:
        if self.failed.exists():
            self.say(f"{self.failed.name} exists: the sampling failed for good (R13); exit 3 without re-sampling")
            return EXIT_BROKEN
        if self.texts.exists() and self.texts.with_suffix(".json").exists():
            self.say("sampling: texts.parquet exists (reused)")
            return 0
        if self.m2_status() == "consumed":
            raise StateError("weights were deleted but the texts are missing")
        src = ["--adapter", self.wdir] if self.param == "lora" else ["--model_dir", self.wdir]
        rc = self.timed("sample", [C.TRAJPE_PY, "-I", "-B", HERE / "sample_ctcl.py", "--dataset", self.a.dataset]
                        + src + ["--hist", self.hist, "--table", self.table, "--out", self.texts, "--seed", self.a.seed,
                                 "--n_out", self.a.n_out], ok_codes=(0, EXIT_BROKEN))
        return EXIT_BROKEN if rc == EXIT_BROKEN else 0

    def do_ground(self):
        rj = self.rel.with_suffix(".json")
        if self.rel.exists() and rj.exists() and (read_json(rj) or {}).get("ctcl"):
            self.say("grounding: release in the work dir exists (reused)")
            return
        self.rel.parent.mkdir(parents=True, exist_ok=True)
        for p in (self.rel, rj):
            p.unlink(missing_ok=True)
        cmd = [C.TRAJPE_PY, "-I", "-B", C.DPLORA / "ground.py", "--dataset", self.a.dataset, "--texts", self.texts,
               "--train_json", self.tj, "--out", self.rel, "--seed", self.a.seed, "--jitter", 0, "--n_out",
               self.a.n_out, "--method", self.a.method]
        if self.vp:
            cmd += ["--prior", self.prior, "--ground_scale_km", C.VP_GROUND_SCALE_KM, "--pop_radius_km",
                    C.pop_radius_km(self.a.eps)]
        self.timed("ground", cmd)
        self.augment(rj)

    def augment(self, rj: Path):
        side = json.loads(rj.read_text())
        cal, hist = json.loads(self.calib.read_text()), json.loads(self.hist.read_text())
        samp = read_json(self.texts.with_suffix(".json")) or {}
        mech = [{"name": "M2 DP-Adam (full fine-tuning)" if self.param == "full" else "M2 DP-Adam (LoRA fallback)",
                 "event": "PoissonSampled(q, Gaussian(sigma_sgd)) x T", "sigma_sgd": cal["sigma_sgd"],
                 "sample_rate": cal["sample_rate"], "steps": cal["steps"], "clip": cal["clip"], "b_nom": cal["b_nom"],
                 "sensitivity_l2": cal["clip"], "eps_alone_pld": cal["eps_sgd_only_pld"]},
                {"name": "M1 DP topic histogram", "event": "Gaussian(sigma_topic_hist) x 1",
                 "sigma_topic_hist": cal["sigma_topic_hist"], "n_topic_hist": cal["n_topic_hist"],
                 "K": C.K_TOPICS, "sensitivity_l2": 1.0, "eps_alone_pld": cal["eps_topic_hist_only_pld"]}]
        if self.vp:
            mech.append({"name": "M3 DP venue prior (dploravp vp.py prior)", "event": "Gaussian(sigma_prior) x 2",
                         "sigma_prior": cal["sigma_prior"], "n_prior": cal["n_prior"], "sensitivity_l2": 1.0,
                         "prior_mu2_share": cal["prior_mu2_share"], "eps_alone_pld": cal["eps_prior_only_pld"]})
        code = C.code_hashes()
        for f in ("ground.py", "vp.py", "textfmt.py", "pld_check.py"):
            code[f"dplora/{f}"] = C.sha256_file(C.DPLORA / f)
        code["trajpe/ablations.py"] = C.sha256_file(C.PROJECT / "trajpe" / "ablations.py")
        side["ctcl"] = {
            "method_key": self.a.method, "label": C.LABELS[self.a.method], "variant": self.variant,
            "config": self.config, "param": self.param,
            "spec": "docs/baselines/CTCL_SPEC.md (pre-registration; frozen/ctcl_v1.sha256)",
            "accounting": {"neighbouring": "add/remove one trail", "eps_target": cal["eps_total"],
                           "delta": cal["delta"], "eps_composed_pld": cal["eps_composed_pld"],
                           "pld_target": cal["pld_target"], "accountant": cal["accountant"],
                           "eps_prv_composed": side.get("eps_prv"), "eps_rdp_composed": side.get("eps_rdp"),
                           "sigma_eq": cal["sigma_eq"], "mechanisms": mech,
                           "guarantee": "(eps_composed_pld <= eps, delta = 1e-6)-DP at the trail level, assuming the "
                                        "mechanisms' random bits are secret (published seeds are for reproducibility)"},
            "m1": {"noise_stream": hist["noise_stream"], "quota_sha256": hist["quota_sha256"],
                   "n_topics_with_quota": hist["n_topics_with_quota"], "hist_sha256": C.sha256_file(self.hist)},
            "topic_table": {"file": str(self.table), "sha256": hist["table_sha256"]},
            "sampling": (samp.get("generation") or {}).get("ctcl"),
            "code_sha256": code, "data_dir": str(C.DATA),
            "runtimes_sec": {k: v.get("sec") for k, v in self.state["stages"].items() if isinstance(v, dict)},
            "runtimes_note": "a non-DP side channel shared by all DP-SGD baselines (spec §4.3)"}
        C.write_json_atomic(rj, side)

    def do_gate(self) -> dict:
        rj = self.rel.with_suffix(".json")
        res = {"release": str(self.rel), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        r = subprocess.run([C.BASE_PY, "-I", "-B", C.PROJECT / "scripts" / "queue_baselines.py", "check",
                            str(self.rel)], capture_output=True, text=True, cwd="/tmp")
        rows = [json.loads(ln) for ln in r.stdout.splitlines() if ln.startswith("{")]
        res["validity"] = rows[-1] if rows else {"ok": False, "problems": [r.stderr[-400:]]}
        res["validity_ok"] = r.returncode == 0 and bool(rows) and bool(rows[-1].get("ok"))
        r = subprocess.run([C.TRAJPE_PY, "-I", "-B", HERE / "check.py", str(rj)], capture_output=True, text=True,
                           cwd="/tmp")
        rows = [json.loads(ln) for ln in r.stdout.splitlines() if ln.startswith("{")]
        res["accounting"] = rows[-1] if rows else {"ok": False, "problems": [r.stderr[-400:]]}
        res["accounting_ok"] = r.returncode == 0 and bool(rows) and bool(rows[-1].get("ok"))
        r = subprocess.run([C.TRAJPE_PY, "-I", "-B", C.DPLORA / "pld_check.py", str(rj)], capture_output=True,
                           text=True, cwd="/tmp")
        res["pld_check"] = (r.stdout.strip().splitlines() or [""])[-1][:300]
        res["pld_check_ok"] = r.returncode == 0
        res["ok"] = res["validity_ok"] and res["accounting_ok"] and res["pld_check_ok"]
        C.write_json_atomic(self.w / "gate.json", res)
        return res

    def finish(self):
        self.note("weights_deleted", deleted=True)
        if self.wdir.exists():
            shutil.rmtree(self.wdir)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        os.replace(self.rel, self.out)
        os.replace(self.rel.with_suffix(".json"), self.out.with_suffix(".json"))
        self.note("released", out=str(self.out))

    def __call__(self) -> int:
        a = self.a
        if self.out.exists() and self.out.with_suffix(".json").exists():
            self.say(f"{self.out} exists: nothing to do")
            return 0
        self.say(f"CTCL {a.method} {a.dataset} eps={a.eps:g} seed={a.seed}: work {self.w}")
        try:
            self.do_calib()
            self.do_m1()
            self.do_m2()
            self.do_m3()
            if self.do_sample() == EXIT_BROKEN:
                return EXIT_BROKEN
            self.do_ground()
            g = self.do_gate()
            if not g["ok"]:
                self.say(f"release gate FAILED (weights kept): {json.dumps(g)[:1500]}")
                return EXIT_GATE
            self.finish()
        except StateError as e:
            self.say(f"inconsistent work dir: {e}")
            return EXIT_STATE
        self.say(f"released {self.out} (no metric computed)")
        return 0


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=list(C.METHODS), required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--eps", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_out", type=int, default=C.N_OUT)
    ap.add_argument("--out", default="", help=f"default {_TP_WORK}/runs/baselines_v2/ctcl/<subdir>/<d>_eps<e>_s<seed>")
    ap.add_argument("--work", default="", help="default <out dir>/_work/<out stem>")
    ap.add_argument("--table", default="")
    ap.add_argument("--mem_cap_gb", type=float, default=0.0)
    ap.add_argument("--log_private", action="store_true", help="tuning city (jakarta) only")
    ap.add_argument("--extra_train", default="", help="test only (e.g. --max_steps); a debug run fails the gate")
    ap.add_argument("--test_data_ok", action="store_true", help="test only: allow TRAJPE_DATA != processed")
    a = ap.parse_args(argv)
    C.check_dataset(a.dataset)
    if a.log_private and a.dataset != C.TUNING_CITY:
        ap.error("--log_private is allowed on the tuning city (jakarta) only (R14)")
    if os.environ.get("TRAJPE_DATA", "processed") != "processed" and not a.test_data_ok:
        ap.error("TRAJPE_DATA is set: releases are made from data/processed only")
    return a


def main(argv=None, runner=None) -> int:
    return Job(parse(argv), runner=runner)()


if __name__ == "__main__":
    sys.exit(main())
