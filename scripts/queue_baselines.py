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
import hashlib
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(_TP_ROOT).resolve()
B = PROJECT / "baselines"
DATA = PROJECT / "data" / os.environ.get("TRAJPE_DATA", "processed")
R = Path(f"{_TP_WORK}/runs/baselines_v2")
STATE = Path(f"{_TP_WORK}/gpuq")
QPY = f"{_TP_PY}"
BPY = f"{_TP_PY_GRID}"
DPLORA_PY = f"{_TP_PY_DPLORA}"
OLD = ["istanbul", "tokyo", "sao_paulo", "kuwait_city", "nyc"]
EPS = [1, 4]
SEED = 0
FORBIDDEN = {"gw_austin", "gw_nyc", "jakarta"}
CPU_METHODS = ["ldptrace"]
METHODS = CPU_METHODS + ["dplora_plain", "dplora_vp", "naivepe", "dplora7b_plain", "dplora7b_vp"]
PRIO = 5


def registry() -> dict:
    return json.loads((PROJECT / "configs" / "datasets_v3.json").read_text())


def new_datasets() -> list[str]:
    ds = registry()["datasets"]
    return [k for k, v in ds.items() if v.get("status") == "new"
            and v["role"] in ("evaluation", "evaluation_devcity", "evaluation_lowdata")]


def table_of(d: str) -> str:
    v = registry()["datasets"][d]
    return v.get("table") or v.get("expected_table") or "?"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def verify_hashes(datasets: list[str], require_frozen: bool = False) -> None:
    """Check prepared inputs; optionally enforce the original experiment's private freeze manifest."""
    if not require_frozen:
        missing = [str(DATA / d / f) for d in dict.fromkeys(datasets)
                   for f in ("train.parquet", "pois.parquet", "meta.json") if not (DATA / d / f).is_file()]
        if missing:
            raise SystemExit("prepared inputs missing (run README data preparation first):\n" + "\n".join(missing))
        return
    for name in ("datasets_v3_preregistration.sha256", "data_v3_hashes.sha256"):
        if not (PROJECT / "frozen" / name).is_file():
            raise SystemExit(f"--require-frozen needs frozen/{name}; private experiment records are not distributed")
    for ln in (PROJECT / "frozen" / "datasets_v3_preregistration.sha256").read_text().splitlines():
        if ln.strip() and not ln.startswith("#") and ln.split()[1] == "configs/datasets_v3.json":
            if sha(PROJECT / "configs" / "datasets_v3.json") != ln.split()[0]:
                raise SystemExit("configs/datasets_v3.json differs from its pre-registration hash")
    want = {}
    for ln in (PROJECT / "frozen" / "data_v3_hashes.sha256").read_text().splitlines():
        if ln.strip() and not ln.startswith("#"):
            h, p = ln.split(maxsplit=1)
            want[p.strip()] = h
    for d in datasets:
        if d in OLD:
            continue
        files = [p for p in want if p.startswith(f"data/processed/{d}/")]
        if not files:
            raise SystemExit(f"{d}: no hashes in frozen/data_v3_hashes.sha256")
        for p in files:
            if sha(PROJECT / p) != want[p]:
                raise SystemExit(f"{p}: SHA-256 differs from frozen/data_v3_hashes.sha256")


def command(argv: list, **settings) -> str:
    """A portable shell command, also when the checkout and work paths contain spaces."""
    env = dict(TRAJPE_ROOT=str(PROJECT), TRAJPE_WORK=_TP_WORK, TRAJPE_DATA=str(DATA),
               TRAJPE_PYTHON=QPY, TRAJPE_PYTHON_DPLORA=DPLORA_PY, TRAJPE_PYTHON_GRID=BPY,
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
               TRAJPE_GRID_NOMETRICS="1")
    env.update(settings)
    return shlex.join(["env"] + [f"{k}={v}" for k, v in env.items()] + [str(x) for x in argv])


def cpu_job(m: str, d: str, e: int, user: bool) -> dict:
    stem = f"{d}_{'user_' if user else ''}eps{e}_s{SEED}"
    out = R / m / f"{stem}.parquet"
    cmd = command([BPY, "-I", B / f"run_{m}.py", "--dataset", d, "--eps", e, "--seed", SEED,
                   "--out", out] + (["--user_level"] if user else []))
    return {"name": f"bl_{m}_{stem}", "cmd": cmd, "mem": 0, "where": "cpu", "prio": PRIO, "out": out}


def dplora_job(d: str, e: int, variant: str, model: str) -> dict:
    spec = importlib.util.spec_from_file_location("trajpe_dplora_jobs", B / "dplora" / "jobs.py")
    dj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dj)
    (j,) = dj.v2_jobs([str(e)], [d], [SEED], [variant], model)
    fz = json.loads(dj.FROZEN_V2.read_text())
    cap = fz["models"][model]["cap_gb"]
    out = Path(j["args"][j["args"].index("--out") + 1])
    cmd = command([DPLORA_PY, "-I", "-B", B / "dplora" / "pipeline.py"] + j["args"]
                  + ["--stage", "gpu", "--mem_cap_gb", cap], PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    cmd += " && " + command([QPY, "-I", "-B", PROJECT / "scripts" / "queue_baselines.py", "dplora_cpu"] + j["args"])
    return {"name": j["name"], "cmd": cmd, "mem": cap + 4, "where": "0,1", "prio": 7 if model != "0.5b" else PRIO,
            "out": out}


def naivepe_job(d: str, e: int) -> dict:
    tag = f"F_naivepe_eps{e}_s{SEED}"
    out = Path(f"{_TP_WORK}/runs") / d / tag / "synth.parquet"
    cmd = f"[ -f {shlex.quote(str(out))} ] || " + command(
        [QPY, "-I", PROJECT / "scripts" / "run_pe.py", "--dataset", d, "--api", "llm", "--sim_frac", 0,
         "--eps", e, "--seed", SEED, "--n_syn", 10000, "--T", 3, "--threshold_sigmas", 1.0,
         "--marginals", "off", "--venue_aware", "off", "--transitions", "off", "--embed", "text",
         "--llm_variation", "llm", "--tag", tag], TRAJPE_VLLM_UTIL="0.40")
    return {"name": f"{d}__{tag}", "cmd": cmd, "mem": 40, "where": "0,1", "prio": PRIO, "out": out}


def grid(methods: list[str], datasets: list[str]) -> list[dict]:
    js = []
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise SystemExit(f"methods not included in the public queue: {sorted(unknown)}")
    known = registry()["datasets"]
    for d in datasets:
        if d in FORBIDDEN or d not in known:
            raise SystemExit(f"{d} is not part of this grid")
        for e in EPS:
            for m in methods:
                new = []
                if m in CPU_METHODS:
                    new = [cpu_job(m, d, e, False), cpu_job(m, d, e, True)]
                elif m in ("dplora_plain", "dplora_vp"):
                    new = [dplora_job(d, e, m.split("_")[1], "0.5b")]
                elif m in ("dplora7b_plain", "dplora7b_vp"):
                    if d in OLD:
                        new = [dplora_job(d, e, m.split("_")[1], "7b")]
                elif m == "naivepe":
                    new = [naivepe_job(d, e)]
                for j in new:
                    j.update(method=m, dataset=d, eps=e)
                js.extend(new)
    return js


def queue_names() -> dict:
    out = {}
    path = STATE / "jobs.jsonl"
    if not path.exists():
        return out
    for ln in path.read_text().splitlines():
        if ln.strip():
            j = json.loads(ln)
            out[j["name"]] = j
    return out


def status(j: dict, names: dict) -> str:
    n = j["name"]
    for s in ("done", "running", "failed"):
        if (STATE / s / n).exists():
            if s == "failed":
                txt = (STATE / s / n).read_text(errors="replace")[:40]
                return "cancelled" if txt.startswith("cancelled") else "failed"
            return s
    if n in names:
        return "pending"
    out = Path(j["out"])
    side = out.with_suffix(".json") if out.name != "synth.parquet" else out.parent / "result.json"
    if out.exists() and side.exists():
        return "exists"
    return "missing"


def check_file(f: Path) -> dict:
    sys.path.insert(0, str(B))
    import common
    import sanity_check as sc
    naive = f.name == "synth.parquet"
    d = f.parent.parent.name if naive else f.stem.split("_eps")[0].removesuffix("_user")
    meta = json.loads((common.DATA / d / "meta.json").read_text())
    pois = __import__("pandas").read_parquet(common.DATA / d / "pois.parquet")
    pois["venue_id"] = pois["venue_id"].astype(str)
    if naive:
        res = {"file": str(f), "problems": []}
        df = __import__("pandas").read_parquet(f)
        lens = df.groupby("trail_id").size()
        res["n_trails"] = int(len(lens))
        if list(df.columns) != common.OUT_COLS:
            res["problems"].append("columns")
        if len(lens) != common.N_OUT:
            res["problems"].append(f"n_trails {len(lens)} != {common.N_OUT}")
        if lens.min() < 2 or lens.max() > 20:
            res["problems"].append("length outside [2, 20]")
        res["frac_known_venue"] = float(df["venue_id"].astype(str).isin(set(pois["venue_id"])).mean())
        side = json.loads((f.parent / "result.json").read_text())
        res["privacy"] = {k: side["privacy"].get(k) for k in ("eps", "delta", "eps_check", "unit")}
        if "vs_train" in side or "vs_test" in side:
            res["problems"].append("result.json holds metrics (TRAJPE_GRID_NOMETRICS not active)")
        if abs(side["privacy"]["eps_check"] - side["privacy"]["eps"]) > 1e-6:
            res["problems"].append("eps_check != eps")
    else:
        r, _ = sc.check(f, d, pois, meta["bbox"])
        res = {k: r[k] for k in ("file", "method", "eps", "n_trails", "user_level", "frac_known_venue", "frac_in_bbox",
                                 "problems") if k in r}
        side = json.loads(f.with_suffix(".json").read_text())
        for k in ("budget", "accounting", "eps_prv", "eps_rdp", "eps_target", "vp_calibration"):
            if k in side:
                res[k] = side[k] if k != "accounting" else {kk: side[k].get(kk) for kk in ("basic_composition_eps",)}
        if "composite_vs_train" in side or "composite_vs_test" in side:
            res["problems"].append("sidecar holds a composite (eval ran)")
        b = side.get("budget")
        if b and "eps_total" in b and abs(b["eps_total"] - side.get("eps", b["eps_total"])) > 1e-9:
            res["problems"].append("budget eps_total != eps")
        if b and side.get("method") in ("PrivTrace", "AdaTrace", "LDPTrace"):
            spent = b["eps_gen"] + sum(v for k, v in b.items() if k.startswith("eps_") and k not in ("eps_total", "eps_gen"))
            if spent > b["eps_total"] + 1e-9:
                res["problems"].append(f"budget parts sum to {spent} > eps_total")
    res["ok"] = not res["problems"]
    return res


def dplora_cpu(argv: list[str]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="")
    ap.add_argument("--jitter", type=int, default=0)
    ap.add_argument("--n_out", type=int, default=50_000)
    ap.add_argument("--variant", choices=["plain", "vp"], default="plain")
    ap.add_argument("--ground_k", type=int, default=10)
    ap.add_argument("--ground_scale_km", type=float, default=0.5)
    ap.add_argument("--pop_radius_km", type=float, default=1.0)
    a, _ = ap.parse_known_args(argv)
    ground_v2 = []
    for n in ("ground_k", "ground_scale_km", "pop_radius_km"):
        v = getattr(a, n)
        if v != ap.get_default(n):
            ground_v2 += [f"--{n}", v]
    out = Path(a.out)
    work = Path(a.work) if a.work else out.parent / "_work" / out.stem
    tj, texts, calib, prior = work / "train.json", work / "texts.parquet", work / "calib.json", work / "prior.npz"
    here = B / "dplora"
    vp = a.variant == "vp"
    with open(work / "pipeline.log", "a") as log:
        def run(cmd):
            print(f"[{time.strftime('%H:%M:%S')}] $ {' '.join(map(str, cmd))}", flush=True)
            env = dict(os.environ)
            env["PATH"] = str(Path(str(cmd[0])).parent) + os.pathsep + env.get("PATH", "")
            subprocess.run([str(c) for c in cmd], check=True, stdout=log, stderr=subprocess.STDOUT, env=env)
        if vp and not prior.exists():
            run([QPY, "-I", "-B", here / "vp.py", "prior", "--dataset", a.dataset, "--calib", calib,
                 "--seed", a.seed, "--out", prior])
        if not out.exists():
            run([QPY, "-I", "-B", here / "ground.py", "--dataset", a.dataset, "--texts", texts,
                 "--train_json", tj, "--out", out, "--seed", a.seed, "--jitter", a.jitter,
                 "--n_out", a.n_out] + (["--prior", prior, "--method", "dploravp"] + ground_v2 if vp else []))
        print("eval_synth.py / score.py skipped (evaluation dataset; metrics only under the benchmark protocol)")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "dplora_cpu":
        return dplora_cpu(sys.argv[2:])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["plan", "queue", "check"])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--eps", nargs="+", type=int, default=EPS, choices=EPS)
    ap.add_argument("--levels", nargs="+", default=["trail", "user"], choices=["trail", "user"],
                    help="privacy levels of the CPU baseline")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--require-frozen", action="store_true",
                    help="require the original experiment's external frozen/data_v3_hashes.sha256 record")
    ap.add_argument("--json_out", default="")
    a = ap.parse_args()
    EPS[:] = a.eps
    if a.cmd == "check":
        rows = [check_file(Path(f)) for f in a.files]
        for r in rows:
            print(json.dumps(r, default=str))
        if a.json_out:
            Path(a.json_out).write_text(json.dumps(rows, indent=1, default=str))
        sys.exit(0 if all(r["ok"] for r in rows) else 1)
    datasets = a.datasets or OLD + new_datasets()
    js = [j for j in grid(a.methods, datasets)
          if j["method"] not in CPU_METHODS or ("user" if "_user_" in j["name"] else "trail") in a.levels]
    names = queue_names()
    if a.cmd == "plan":
        rows = []
        for j in js:
            rows.append({k: str(v) for k, v in j.items() if k != "cmd"} | {"status": status(j, names)})
            print(f"{status(j, names):9s} {j['name']}")
        if a.json_out:
            Path(a.json_out).write_text(json.dumps(rows, indent=1))
        return
    if not a.dry or a.require_frozen:
        verify_hashes(datasets, require_frozen=a.require_frozen)
    for j in js:
        st = status(j, names)
        if st != "missing":
            print(f"skip ({st}) {j['name']}")
            continue
        add = [QPY, "-I", str(PROJECT / "scripts" / "gpuq.py"), "add", "--state", str(STATE), "--name", j["name"],
               "--cmd", j["cmd"], "--mem", f"{j['mem']:g}", "--where", j["where"], "--prio", str(j["prio"])]
        if a.dry:
            print("[dry]", j["name"], j["mem"], j["where"], j["prio"], j["cmd"])
        else:
            subprocess.run(add, check=True)


if __name__ == "__main__":
    main()
