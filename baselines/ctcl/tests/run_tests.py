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
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

TESTS = Path(__file__).resolve().parent
CTCL = TESTS.parent
PROJECT = CTCL.parents[1]
TRAJPE_PY = f"{_TP_PY}"
DPLORA_PY = f"{_TP_PY_DPLORA}"
ORDER = [("test_accounting.py", TRAJPE_PY), ("test_sample.py", TRAJPE_PY),
         ("test_pipeline.py", TRAJPE_PY), ("test_train.py", DPLORA_PY), ("test_release.py", TRAJPE_PY)]


def code_hashes() -> dict:
    files = sorted(CTCL.glob("*.py")) + sorted(TESTS.glob("*.py"))
    return {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmp", default="")
    ap.add_argument("--gpu_json", default="")
    ap.add_argument("--only", nargs="+", choices=[name for name, _ in ORDER], default=None,
                    help="run selected public tests; RESULTS.json records whether the full suite ran")
    a = ap.parse_args()
    selected = [(name, py) for name, py in ORDER if a.only is None or name in a.only]
    missing = [name for name, _ in selected if not (TESTS / name).is_file()]
    if missing:
        ap.error("missing required test files: " + ", ".join(missing))
    before = code_hashes()
    env = dict(os.environ, OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8")
    if a.tmp:
        env["CTCL_TEST_TMP"] = a.tmp
    rows = []
    for name, py in selected:
        t0 = time.time()
        r = subprocess.run([py, "-I", "-B", str(TESTS / name)], capture_output=True, text=True, env=env, cwd="/tmp")
        lines = [ln for ln in (r.stdout + r.stderr).splitlines() if ln.strip() and "warn" not in ln.lower()
                 and "it/s]" not in ln]
        rows.append({"test": name, "env": py, "rc": r.returncode, "sec": round(time.time() - t0, 1),
                     "tail": lines[-12:]})
        print(f"{'PASS' if r.returncode == 0 else 'FAIL'} {name} ({rows[-1]['sec']:.0f}s)", flush=True)
        if r.returncode != 0:
            print("\n".join(lines[-30:]), flush=True)
    after = code_hashes()
    res = {"finished": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
           "all_passed": bool(rows) and all(r["rc"] == 0 for r in rows) and before == after,
           "full_suite": len(selected) == len(ORDER),
           "tests": rows, "code_sha256": after, "code_unchanged_during_run": before == after}
    if a.gpu_json and Path(a.gpu_json).exists():
        res["gpu_check"] = json.loads(Path(a.gpu_json).read_text())
    out = TESTS / "RESULTS.json"
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(res, indent=1))
    tmp.replace(out)
    print(f"all_passed={res['all_passed']} full_suite={res['full_suite']} -> {out}")
    sys.exit(0 if res["all_passed"] else 1)


if __name__ == "__main__":
    main()
