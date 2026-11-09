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
import itertools
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = f"{_TP_PY_AIMSEQ}"
DEV = {"jakarta", "nyc"}


def cores_list(spec: str) -> list:
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--eps", nargs="+", type=float, default=[1.0, 4.0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--user_eps", nargs="*", type=float, default=[], help="user-level cells (seed 0)")
    ap.add_argument("--cores", default="68-79")
    ap.add_argument("--cores_per_slot", type=int, default=2)
    ap.add_argument("--out_root", type=Path, default=Path(f"{_TP_WORK}/runs/baselines_v2/aimseq"))
    ap.add_argument("--allow_eval", action="store_true", help="allow evaluation datasets (after the design freeze)")
    ap.add_argument("--extra", default="", help="extra run_aimseq.py flags, e.g. '--mech mst'")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    bad = [d for d in args.datasets if d not in DEV]
    if bad and not args.allow_eval:
        sys.exit(f"refusing evaluation datasets {bad} without --allow_eval (design freeze first)")
    jobs = []
    for d, e, s in itertools.product(args.datasets, args.eps, args.seeds):
        jobs.append((d, e, s, False))
    for d, e in itertools.product(args.datasets, args.user_eps):
        jobs.append((d, e, 0, True))
    cores = cores_list(args.cores)
    slots = [cores[i:i + args.cores_per_slot] for i in range(0, len(cores) - args.cores_per_slot + 1,
                                                               args.cores_per_slot)]
    args.out_root.mkdir(parents=True, exist_ok=True)
    todo = []
    for d, e, s, ul in jobs:
        out = args.out_root / f"{d}_{'user_' if ul else ''}eps{e:g}_s{s}.parquet"
        if not out.exists():
            todo.append((d, e, s, ul, out))
    print(f"{len(todo)} of {len(jobs)} runs to do on {len(slots)} slots x {args.cores_per_slot} cores", flush=True)
    if args.dry_run:
        for d, e, s, ul, out in todo:
            print(out.name, "user_level" if ul else "")
        return
    running: dict = {}
    while todo or running:
        for slot in range(len(slots)):
            if slot in running and running[slot][0].poll() is not None:
                p, out = running.pop(slot)
                print(f"[{time.strftime('%T')}] {out.name}: exit {p.returncode}", flush=True)
            if slot not in running and todo:
                d, e, s, ul, out = todo.pop(0)
                c = ",".join(map(str, slots[slot]))
                n = str(len(slots[slot]))
                cmd = ["taskset", "-c", c, PY, "-I", "-B", str(HERE / "run_aimseq.py"), "--dataset", d, "--eps", f"{e:g}",
                       "--seed", str(s), "--out", str(out)] + (["--user_level"] if ul else []) + args.extra.split()
                print(f"[{time.strftime('%T')}] start {out.name} on cores {c}", flush=True)
                env = {**os.environ, "OMP_NUM_THREADS": n, "OPENBLAS_NUM_THREADS": n, "MKL_NUM_THREADS": n}
                log = open(out.with_suffix(".log"), "w")
                running[slot] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), out)
        time.sleep(5)


if __name__ == "__main__":
    main()
