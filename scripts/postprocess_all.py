import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trajpe.paths import RUNS

ROOT = Path(__file__).resolve().parents[1]
CITIES = ["istanbul", "tokyo", "sao_paulo", "kuwait_city", "nyc"]
SKIP_METHODS = {"naivepe"}


def todo():
    for city in CITIES:
        for d in sorted((RUNS / city).glob("F_*")):
            method = d.name.split("_")[1]
            if method in SKIP_METHODS or method.endswith("div"):
                continue
            if not ((d / "synth.npz").exists() and (d / "result.json").exists()):
                continue
            out = "_".join(["F", method + "div"] + d.name.split("_")[2:])
            if not (RUNS / city / out / "result.json").exists():
                yield city, d.name, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--radius_km", type=float, required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    jobs = list(todo())
    print(f"{len(jobs)} releases to post-process", flush=True)

    def run(job):
        city, tag, out = job
        r = subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / "postprocess_release.py"), "--dataset", city,
                            "--tag", tag, "--out_tag", out, "--p", str(args.p), "--radius_km", str(args.radius_km)],
                           capture_output=True, text=True)
        tail = "" if r.returncode == 0 else " | " + (r.stderr.strip().splitlines() or [""])[-1][:200]
        print(("ok  " if r.returncode == 0 else "FAIL") + f" {city}/{out}{tail}", flush=True)

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(run, jobs))
