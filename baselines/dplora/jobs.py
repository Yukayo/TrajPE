"""Build DP-LoRA v2 job arguments from the bundled frozen configuration.

Print jobs with ``python jobs.py v2 --eps 1 4 --cities istanbul --print``.
Without --print, append them to --jobs (default: TRAJPE_WORK/runs/baselines_v2/
dplora_v2/_queue/jobs.jsonl). This command prepares jobs; it does not run training.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = Path(os.environ.get("TRAJPE_ROOT", str(HERE.parents[1])))
WORK = Path(os.environ.get("TRAJPE_WORK", str(PROJECT / "work")))
FROZEN_V2 = HERE / "frozen_v2.json"
ROOT_V2 = WORK / "runs" / "baselines_v2" / "dplora_v2"
JOBS_V2 = ROOT_V2 / "_queue" / "jobs.jsonl"
CITIES = ["istanbul", "tokyo", "sao_paulo", "kuwait_city", "nyc"]

# Public workload estimates affect job ordering only, never model hyperparameters.
TOKENS_M = {
    "istanbul": 15.2, "tokyo": 13.1, "nyc": 7.7, "sao_paulo": 7.2,
    "kuwait_city": 6.4, "jakarta": 9.5, "tsmc_nyc": 4.73,
    "petaling_jaya": 5.22, "moscow": 2.75, "bandung": 3.54, "sydney": 0.86,
    "melbourne": 0.66, "tangerang": 0.77, "palembang": 0.28, "shanghai": 0.3,
    "beijing": 0.04, "ms_tokyo": 0.43, "gw_stockholm": 4.94,
    "gw_san_francisco": 2.59,
}


def eps_str(e: float) -> str:
    return "inf" if not math.isfinite(e) else f"{e:g}"


def job(name, city, e, b, ep, lr, r, c, out, log_private=False, variant="plain", seed=0):
    args = ["--dataset", city, "--eps", eps_str(e), "--seed", seed,
            "--batch", b, "--epochs", ep, "--lr", lr, "--rank", r,
            "--clip", c, "--out", str(out)]
    if log_private:
        args.append("--log_private")
    if variant != "plain":
        args += ["--variant", variant]
    return {"name": name, "args": [str(a) for a in args]}


def v2_jobs(eps_list, cities, seeds, variants, model_key):
    """Return the frozen v2 pipeline jobs, with the original names and arguments."""
    fz = json.loads(FROZEN_V2.read_text())
    mdl = fz["models"][model_key]
    js = []
    for variant in variants:
        root = ROOT_V2 / (variant if model_key == fz["default_model"] else f"{variant}_{model_key}")
        for es in eps_list:
            e = float(es)
            if variant == "vp" and not math.isfinite(e):
                continue
            hp = fz["eps"][eps_str(e)]
            for city in cities:
                for sd in seeds:
                    j = job(f"v2_{variant}_{model_key}_{city}_eps{eps_str(e)}_s{sd}",
                            city, e, hp["batch"], hp["epochs"], hp["lr"], hp["rank"],
                            hp.get("clip", 1.0), root / f"{city}_eps{eps_str(e)}_s{sd}.parquet",
                            False, variant, seed=sd)
                    flags = {**fz["format"], **hp.get("train_v2", {}), **hp["sample_v2"],
                             **mdl.get("pipeline", {}),
                             **(hp.get("vp_ground_v2", {}) if variant == "vp" else {})}
                    for k, v in flags.items():
                        if v is True:
                            j["args"].append(f"--{k}")
                        elif v not in (False, None):
                            j["args"] += [f"--{k}", str(v)]
                    j["cost"] = (TOKENS_M.get(city, 1.0) * hp["epochs"]
                                 * mdl.get("cost_factor", 1.0)
                                 * (1.0 if math.isfinite(e) else 1.4))
                    js.append(j)
    js.sort(key=lambda j: -j["cost"])
    return [{k: v for k, v in j.items() if k != "cost"} for j in js]


def append(jobs, path=JOBS_V2):
    """Append new job names, retaining the original JSONL job schema."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = set(path.parent.glob("jobs*.jsonl"))
    if path.exists():
        existing.add(path)
    have = {json.loads(line)["name"] for f in existing
            for line in f.read_text().splitlines() if line.strip()}
    with path.open("a") as f:
        for j in jobs:
            if j["name"] in have:
                print("exists", j["name"])
                continue
            f.write(json.dumps(j) + "\n")
            have.add(j["name"])
            print("added", j["name"])


def main(argv=None):
    fz = json.loads(FROZEN_V2.read_text())
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v2 = sub.add_parser("v2", help="prepare jobs using frozen_v2.json")
    v2.add_argument("--eps", nargs="+", default=["1", "4", "inf"], choices=list(fz["eps"]))
    v2.add_argument("--cities", nargs="+", default=CITIES)
    v2.add_argument("--seeds", nargs="+", type=int, default=[0])
    v2.add_argument("--variants", nargs="+", choices=["plain", "vp"], default=["plain", "vp"])
    v2.add_argument("--model", default=fz["default_model"], choices=list(fz["models"]))
    output = v2.add_mutually_exclusive_group()
    output.add_argument("--print", dest="print_jobs", action="store_true", help="print JSONL without writing files")
    output.add_argument("--jobs", type=Path, default=JOBS_V2, help="destination JSONL file")
    args = ap.parse_args(argv)
    js = v2_jobs(args.eps, args.cities, args.seeds, args.variants, args.model)
    if args.print_jobs:
        for j in js:
            print(json.dumps(j))
    else:
        append(js, args.jobs)


if __name__ == "__main__":
    main()
