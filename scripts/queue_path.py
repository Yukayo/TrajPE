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
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
import queue_baselines as QB

STATE = QB.STATE
QPY = f"{_TP_PY}"
BPY = f"{_TP_PY_GRID}"
PATH_DIR = PROJECT / "baselines" / "path" / "v2"
RUNS = Path(f"{_TP_WORK}/runs/baselines_v2")
SPEC = PROJECT / "docs" / "baselines" / "PATH_SPEC.md"
MARK = "<!-- END OF PRE-REGISTRATION -->"
SEED = 0
DATASETS = ["istanbul", "tokyo", "sao_paulo", "kuwait_city", "nyc", "tsmc_nyc", "petaling_jaya", "moscow", "bandung",
            "gw_stockholm", "gw_san_francisco", "sydney", "melbourne", "tangerang", "palembang", "shanghai",
            "beijing", "ms_tokyo"]
DATASETS_7B = ["istanbul", "tokyo", "sao_paulo", "kuwait_city"]
NEVER = {"gw_austin", "gw_nyc"}
SMOKE, SMOKE_DIR = "jakarta", "_smoke_v2"
ROWS = {"0.5b": ("path", "path"), "7b": ("path_7b", "path7b")}
SMOKE_PREFIX = {"0.5b": "pathv2", "7b": "path7bv2"}
PRIO = {"smoke": -1, "grid": 3, "grid28": 6, "7b": 5}
SMOKE_MEM = {"0.5b": 30.0, "7b": 60.0}
MEM_MAX = 90.0
OWN = ["baselines/path/README.md", "baselines/path/v2/pathfmt.py", "baselines/path/v2/calibrate.py",
       "baselines/path/v2/train_dp.py", "baselines/path/v2/sample_pool.py", "baselines/path/v2/release.py",
       "baselines/path/v2/pld_check.py", "baselines/path/v2/pipeline.py",
       "baselines/path/tests/v2/test_accounting.py", "baselines/path/tests/v2/test_dpsgd.py",
       "baselines/path/tests/v2/test_release.py", "baselines/path/tests/v2/run_tests.sh", "scripts/queue_path.py"]
DEPS = ["baselines/dplora/textfmt.py", "baselines/dplora/accounting.py", "baselines/dplora/hybrid.py",
        "trajpe/dp.py", "trajpe/api_llm.py", "trajpe/api_sim.py", "trajpe/catalog.py", "trajpe/trails.py",
        "trajpe/embed.py", "trajpe/data.py", "trajpe/paths.py", "trajpe/public_catalog.py",
        "baselines/common.py", "baselines/sanity_check.py", "scripts/queue_baselines.py", "scripts/gpuq.py"]


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def spec_prefix() -> bytes:
    t = SPEC.read_text()
    marks = list(re.finditer(rf"^{re.escape(MARK)}[ \t]*$", t, flags=re.M))
    if len(marks) != 1:
        raise SystemExit(f"{SPEC} needs exactly one pre-registration marker line, found {len(marks)}")
    return t[:marks[0].start()].encode()


def out_path(d: str, e: int, mk: str) -> Path:
    rdir = RUNS / ROWS[mk][0]
    return (rdir / SMOKE_DIR if d == SMOKE else rdir) / f"{d}_eps{e:g}_s{SEED}.parquet"


def work_dir(out: Path) -> Path:
    return out.parent / "_work" / out.stem


def job(d: str, e: int, mk: str, part: str, mem: float) -> dict:
    if d in NEVER:
        raise SystemExit(f"{d} is never read or run")
    pre = SMOKE_PREFIX[mk] if d == SMOKE else ROWS[mk][1]
    stem = f"{d}_eps{e}_s{SEED}"
    m = float(mem)
    cmd = (f"cd /tmp && {QPY} -I -B {PATH_DIR / 'pipeline.py'} --dataset {d} --eps {e} --model_key {mk} "
           f"--seed {SEED} --mem_cap_gb {m - 4:g}")
    return {"name": f"{pre}_{stem}", "cmd": cmd, "mem": m, "where": "0,1", "prio": PRIO[part],
            "out": out_path(d, e, mk), "dataset": d, "eps": e, "model_key": mk, "part": part}


def plan(mem: dict | None = None) -> list[dict]:
    mem = mem or dict(SMOKE_MEM)
    js = [job(SMOKE, 1, "0.5b", "smoke", SMOKE_MEM["0.5b"]), job(SMOKE, 4, "0.5b", "smoke", SMOKE_MEM["0.5b"]),
          job(SMOKE, 4, "7b", "smoke", SMOKE_MEM["7b"])]
    js += [job(d, e, "0.5b", "grid", mem["0.5b"]) for d in DATASETS for e in (1, 4)]
    js += [job(d, e, "7b", "7b", mem["7b"]) for d in DATASETS_7B for e in (1, 4)]
    js += [job(d, e, "0.5b", "grid28", mem["0.5b"]) for d in DATASETS for e in (2, 8)]
    return js


def digest(js: list[dict]) -> str:
    canon = [{k: str(j[k]) for k in ("name", "dataset", "eps", "model_key", "prio", "out", "where")} for j in js]
    return sha(json.dumps(canon, sort_keys=True).encode())


def status(j: dict, names: dict) -> str:
    s = QB.status(j, names)
    if s in ("done", "exists") and not Path(j["out"]).exists():
        return "failed" if (STATE / "failed" / j["name"]).exists() else s
    work = work_dir(Path(j["out"]))
    if (work / "FAILED").exists() and s != "running":
        return "genuine_failure"
    return s


def newest_freeze() -> Path | None:
    fs = sorted((PROJECT / "frozen").glob("path_v2*.sha256"),
                key=lambda p: (len(p.stem), p.stem))
    return fs[-1] if fs else None


def freeze_mismatches(js: list[dict], f: Path | None = None) -> list:
    f = f or newest_freeze()
    if f is None or not f.exists():
        return ["no frozen/path_v2*.sha256"]
    want, bad = None, []
    for ln in f.read_text().splitlines():
        if ln.startswith("# plan digest:"):
            want = ln.split(":", 1)[1].split()[0]
        if not ln.strip() or ln.startswith("#"):
            continue
        h, p = ln.split(maxsplit=1)
        p = p.strip()
        fp = PROJECT / p.split("#")[0]
        if not fp.exists():
            bad.append(f"{p} (missing)")
            continue
        got = sha(spec_prefix()) if p.endswith("#pre-registration") else sha(fp.read_bytes())
        if got != h:
            bad.append(p)
    if want != digest(js):
        bad.append(f"plan digest {digest(js)} (frozen: {want})")
    return bad


def cmd_freeze(out: Path, note: str, reason: str, supersedes: str, extra: list[str]) -> None:
    if out.exists():
        raise SystemExit(f"{out} exists: a freeze file is never overwritten")
    js = plan()
    lines = [f"# PATH baseline freeze ({out.name}), written {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
             f"# status: {note}"]
    if reason:
        lines.append(f"# reason: {reason}")
    if supersedes:
        lines.append(f"# supersedes: {supersedes}")
    lines += [f"# {x}" for x in extra]
    lines += [f"# full SPEC file sha256 at freeze (information only; the binding entry is #pre-registration): "
              f"{sha(SPEC.read_bytes())}",
              f"# plan digest: {digest(js)}  ({len(js)} jobs; scripts/queue_path.py digest)",
              f"{sha(spec_prefix())}  docs/baselines/PATH_SPEC.md#pre-registration"]
    lines += [f"{sha((PROJECT / p).read_bytes())}  {p}" for p in OWN]
    lines.append("# read-only dependencies imported by the jobs")
    lines += [f"{sha((PROJECT / p).read_bytes())}  {p}" for p in DEPS]
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out} ({len(OWN) + len(DEPS) + 1} entries)")


def cmd_validity(files: list[str]) -> None:
    rows = [QB.check_file(Path(f)) for f in files]
    for r in rows:
        print(json.dumps(r, default=str))
    sys.exit(0 if rows and all(r["ok"] for r in rows) else 3)


def run_json(cmd: list) -> tuple[int, list, str]:
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, cwd="/tmp")
    rows = []
    for ln in r.stdout.splitlines():
        if ln.startswith("{"):
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
    return r.returncode, rows, r.stdout + r.stderr


def check_one(f: Path) -> dict:
    if f.stem.split("_eps")[0] in NEVER:
        raise SystemExit(f"{f}: never checked")
    _, v, txt = run_json([BPY, "-I", "-B", Path(__file__).resolve(), "validity", f])
    _, p, txt2 = run_json([QPY, "-I", "-B", PATH_DIR / "pld_check.py", f.with_suffix(".json")])
    r = {"file": str(f), "problems": []}
    if not v:
        r["problems"].append("validity produced no row: " + txt[-300:])
    else:
        r["problems"] += v[0]["problems"]
        r.update({k: v[0].get(k) for k in ("n_trails", "frac_known_venue", "frac_in_bbox")})
        sc = json.loads(Path(f).with_suffix(".json").read_text())
        r["len_mean"] = round(float(sc.get("n_checkins", 0)) / max(1, int(sc.get("n_trails", 1))), 3)
        r["version"] = sc.get("version")
    if not p:
        r["problems"].append("pld_check produced no row: " + txt2[-300:])
    else:
        r["problems"] += p[0]["problems"]
        r.update({k: p[0].get(k) for k in ("eps", "eps_joint_pld_recomputed", "eps_joint_prv_recorded",
                                            "eps_joint_rdp_dp_accounting", "sigma_sgd", "sigma_sel")})
    r["ok"] = not r["problems"]
    return r


def cmd_check(js: list[dict], files: list[str], json_out: str) -> None:
    names = QB.queue_names()
    todo = [Path(f) for f in files] if files else [Path(j["out"]) for j in js if Path(j["out"]).exists()]
    res = []
    for f in todo:
        r = check_one(f)
        res.append(r)
        print(json.dumps(r, default=str))
    st = {}
    for j in js:
        s = status(j, names)
        st[s] = st.get(s, 0) + 1
    print(f"checked {len(res)} releases: {sum(r['ok'] for r in res)} valid; queue status of the {len(js)} planned "
          f"jobs: {st}; files changed since the freeze: {freeze_mismatches(plan()) or 'none'}")
    if json_out:
        Path(json_out).write_text(json.dumps(res, indent=1, default=str))
    if any(not r["ok"] for r in res):
        sys.exit(3)


def smoke_info(j: dict) -> dict:
    out = Path(j["out"])
    w = work_dir(out)
    r = {"name": j["name"], "model_key": j["model_key"], "eps": j["eps"], "release": out.exists()}
    tj = w / "train.json"
    if tj.exists():
        t = json.loads(tj.read_text())
        ps = t.get("private_stats") or {}
        r.update({"train_complete": t.get("complete"), "oom_retries": ps.get("oom_retries"),
                  "train_loop_h": round(ps.get("train_loop_sec", 0) / 3600, 3),
                  "train_max_alloc_gb": ps.get("max_mem_allocated_gb"),
                  "train_max_reserved_gb": ps.get("max_mem_reserved_gb"), "eps_prv": t.get("eps_prv")})
    pj = w / "pool.json"
    if pj.exists():
        p = json.loads(pj.read_text())
        r.update({"pool_valid_rate": p.get("valid_rate"), "pool_generated": p.get("n_generated"),
                  "pool_tokens": p.get("generated_tokens"), "sample_h": round(p.get("sample_total_sec", 0) / 3600, 3),
                  "sample_gpu_mem_gb": p.get("gpu_mem_gb_after_round_max")})
    rp = w / "release_private.json"
    if rp.exists():
        q = json.loads(rp.read_text())
        r.update({"release_h": round(q.get("release_total_sec", 0) / 3600, 3),
                  "release_max_reserved_gb": q.get("gpu_max_mem_reserved_gb")})
    if (w / "FAILED").exists():
        r["genuine_failure"] = (w / "FAILED").read_text().strip()
    return r


def declared_mem(info: dict) -> float | None:
    peaks = [info.get(k) for k in ("train_max_reserved_gb", "sample_gpu_mem_gb", "release_max_reserved_gb")]
    peaks = [float(x) for x in peaks if x is not None]
    if not peaks or info.get("train_max_reserved_gb") is None:
        return None
    return float(min(MEM_MAX, math.ceil(max(peaks) + 4 + 2)))


def cmd_smoke_report(js: list[dict]) -> None:
    names = QB.queue_names()
    for j in [j for j in js if j["part"] == "smoke"]:
        info = smoke_info(j) | {"status": status(j, names)}
        info["declared_mem_rule"] = declared_mem(info)
        print(json.dumps(info))


def gate_7b(js: list[dict]) -> None:
    sm = [j for j in js if j["part"] == "smoke" and j["model_key"] == "7b"][0]
    info = smoke_info(sm)
    if info.get("train_complete") is not True or info.get("oom_retries") != 0:
        raise SystemExit(f"7B smoke has not shown that the row fits memory: {info}")


def cmd_queue(js: list[dict], part: str, dry: bool) -> None:
    bad = freeze_mismatches(plan())
    if bad and not dry:
        raise SystemExit(f"differs from {newest_freeze()} (frozen before anything is queued): {bad}")
    sel = [j for j in js if j["part"] == part]
    if part == "7b" and not dry:
        gate_7b(plan())
    QB.verify_hashes([j["dataset"] for j in sel if j["dataset"] != SMOKE])
    names = QB.queue_names()
    n = 0
    for j in sel:
        st = status(j, names)
        if st != "missing":
            print(f"skip ({st}) {j['name']}")
            continue
        if j["mem"] > MEM_MAX:
            raise SystemExit(f"{j['name']}: {j['mem']} GiB exceeds an H100 slot")
        add = [QPY, "-I", str(PROJECT / "scripts" / "gpuq.py"), "add", "--state", str(STATE), "--name", j["name"],
               "--cmd", j["cmd"], "--mem", f"{j['mem']:g}", "--where", j["where"], "--prio", str(j["prio"])]
        if dry:
            print("[dry]", j["name"], j["mem"], j["where"], j["prio"], j["cmd"])
        else:
            r = subprocess.run(add, check=True, capture_output=True, text=True)
            if not r.stdout.startswith("added"):
                raise SystemExit(f"gpuq add {j['name']}: {r.stdout.strip()}")
        n += 1
    print(f"{'would queue' if dry else 'queued'} {n} jobs ({part})")


def cmd_cancel(js: list[dict], rows: list[str], reason: str) -> None:
    names = QB.queue_names()
    n = 0
    for j in js:
        if j["model_key"] in rows and status(j, names) == "pending":
            (STATE / "failed" / j["name"]).write_text(f"cancelled {reason} {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            n += 1
    print(f"cancelled {n} pending PATH jobs of {rows}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "validity":
        return cmd_validity(sys.argv[2:])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["plan", "digest", "queue", "check", "smoke_report", "cancel", "freeze"])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--part", choices=list(PRIO))
    ap.add_argument("--rows", nargs="+", default=list(ROWS), choices=list(ROWS))
    ap.add_argument("--mem_05b", type=float, default=0.0, help="GiB per 0.5B grid job, declared from the smoke run")
    ap.add_argument("--mem_7b", type=float, default=0.0, help="GiB per 7B grid job, declared from the 7B smoke run")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--json_out", default="")
    ap.add_argument("--reason", default="")
    ap.add_argument("--supersedes", default="")
    ap.add_argument("--extra", action="append", default=[], help="freeze: extra header line (repeatable)")
    ap.add_argument("--out", default="")
    ap.add_argument("--note", default="no run yet")
    a = ap.parse_args()
    mem = {"0.5b": a.mem_05b or SMOKE_MEM["0.5b"], "7b": a.mem_7b or SMOKE_MEM["7b"]}
    js = [j for j in plan(mem) if j["model_key"] in a.rows]
    if a.cmd == "plan":
        names = QB.queue_names()
        rows = []
        for j in js:
            s = status(j, names)
            rows.append({k: str(v) for k, v in j.items() if k != "cmd"} | {"status": s})
            print(f"{s:16s} {j['prio']:>2} {j['mem']:>4g} {j['name']}")
        if a.json_out:
            Path(a.json_out).write_text(json.dumps(rows, indent=1))
    elif a.cmd == "digest":
        print(digest(plan()))
    elif a.cmd == "queue":
        if not a.part:
            raise SystemExit("queue needs --part")
        if a.part in ("grid", "grid28") and not a.mem_05b:
            raise SystemExit("declare --mem_05b from the 0.5B smoke runs (smoke_report: declared_mem_rule)")
        if a.part == "7b" and not a.mem_7b:
            raise SystemExit("declare --mem_7b from the 7B smoke run (smoke_report: declared_mem_rule)")
        cmd_queue(js, a.part, a.dry)
    elif a.cmd == "check":
        cmd_check(js, a.files, a.json_out)
    elif a.cmd == "smoke_report":
        cmd_smoke_report(js)
    elif a.cmd == "cancel":
        if not a.reason:
            raise SystemExit("cancel needs --reason")
        cmd_cancel(js, a.rows, a.reason)
    elif a.cmd == "freeze":
        if not a.out:
            raise SystemExit("freeze needs --out")
        cmd_freeze(PROJECT / a.out if not a.out.startswith("/") else Path(a.out), a.note, a.reason, a.supersedes,
                   a.extra)


if __name__ == "__main__":
    main()
