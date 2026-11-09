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
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trajpe.paths import RESULTS, RUNS

ROOT = Path(__file__).resolve().parents[1]
CITIES = ["istanbul", "tokyo", "sao_paulo", "kuwait_city", "nyc"]
BASELINES = ["privtrace", "adatrace", "ldptrace", "privtraceaux", "privtracediv", "adatracediv", "dplora", "dploravp"]


def releases(pattern: str):
    for city in CITIES:
        for d in sorted((RUNS / city).glob("*")):
            if re.search(pattern, d.name) and (d / "synth.parquet").exists():
                yield city, d.name, d / "synth.parquet"
        for m in BASELINES:
            for f in sorted((RUNS / "baselines_v2" / m).glob(f"{city}_*eps*_s*.parquet")):
                yield city, f"{m}_{f.stem[len(city) + 1:]}", f


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default=r"^F_")
    ap.add_argument("--only", default="", help="regex on '<city>/<name>' to restrict")
    ap.add_argument("--model", default="transformer")
    ap.add_argument("--workers", type=int, default=3, help="concurrent evaluations (all on the visible GPU)")
    ap.add_argument("--train_seed", type=int, default=0, help="downstream model seed; outputs get _t<seed> if > 0")
    ap.add_argument("--extra", default="", help="extra train_eval.py arguments, e.g. '--early-stop holdout'")
    ap.add_argument("--suffix", default="", help="output name suffix for --extra variants, e.g. _holdout")
    args = ap.parse_args()
    todo = []
    for city, name, path in releases(args.pattern):
        suffix = (f"_t{args.train_seed}" if args.train_seed else "") + args.suffix
        out = RESULTS / "downstream" / city / f"{name}_{args.model}{suffix}.json"
        if out.exists() or (args.only and not re.search(args.only, f"{city}/{name}")):
            continue
        todo.append((0 if name.endswith("_s0") else 1, city, name, path, out))
    todo.sort(key=lambda x: x[0])
    print(f"{len(todo)} releases to evaluate", flush=True)

    def run(item):
        _, city, name, path, out = item
        out.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run([f"{_TP_PY_GRID}", "-I", str(ROOT / "downstream" / "train_eval.py"),
                            "--dataset", city, "--train", str(path), "--out", str(out), "--model", args.model,
                            "--seed", str(args.train_seed)] + args.extra.split(), capture_output=True, text=True)
        tail = "" if r.returncode == 0 else " | " + (r.stderr.strip().splitlines() or [""])[-1][:200]
        print(("ok  " if r.returncode == 0 else "FAIL") + f" {city}/{name}{tail}", flush=True)

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(run, todo))
