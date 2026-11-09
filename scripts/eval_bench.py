from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from trajpe import bench_metrics as bm
from trajpe.catalog import Catalog
from trajpe.data import load_split
from trajpe.paths import PROCESSED, RESULTS
from trajpe.trails import from_frame

FREEZE_FILE = ROOT / "frozen" / "benchmark_protocol_v1.2.1.sha256"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_freeze() -> dict:
    if not FREEZE_FILE.exists():
        return {"frozen": False, "mismatch": ["<no freeze file>"], "files": {}}
    files, bad = {}, []
    for line in FREEZE_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, rel = line.split(None, 1)
        rel = rel.lstrip("*")
        p = ROOT / rel
        cur = sha256(p) if p.exists() else "<missing>"
        files[rel] = cur
        if cur != digest:
            bad.append(rel)
    return {"frozen": not bad, "mismatch": bad, "files": files}


def load_downstream(paths) -> dict:
    out = {}
    for p in paths or []:
        d = json.loads(Path(p).read_text())
        if "bench_metrics" not in d:
            raise ValueError(f"{p}: no 'bench_metrics' block (run downstream/train_eval.py with --bench-metrics)")
        key = d["model"] if d["model"] not in out else f"{d['model']}_s{d['seed']}"
        out[key] = {**d["bench_metrics"], "model": d["model"], "train_path": d["train_path"], "seed": d["seed"],
                    "source": str(p), "protocol": d.get("protocol")}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--synth", help="release parquet (standard check-in schema)")
    ap.add_argument("--name", help="output name (default: <method>_eps<eps>_s<seed>)")
    ap.add_argument("--method")
    ap.add_argument("--eps", type=float)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--downstream", nargs="*", default=[], help="train_eval.py --bench-metrics JSON outputs")
    ap.add_argument("--jobs", type=Path, help="JSON list of releases to score in one process (shares the real data)")
    ap.add_argument("--reference", action="store_true", help="score the real test split against train")
    ap.add_argument("--refs", default="train,test", help="real reference splits (first = primary)")
    ap.add_argument("--metrics", help="comma-separated subset (default: every protocol metric)")
    ap.add_argument("--out-dir", type=Path, default=None, help="default results/bench/<dataset>")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args(argv)

    P = bm.load_protocol()
    dev = set(P["datasets"]["tuning"]) | set(P["datasets"]["development"])
    freeze = verify_freeze()
    if args.dataset not in dev and not freeze["frozen"]:
        print(f"warning: protocol freeze record not verified for '{args.dataset}' ({freeze['mismatch']})", file=sys.stderr)

    jobs = []
    if args.jobs:
        jobs = json.loads(args.jobs.read_text())
    elif args.reference:
        jobs = [{"reference": True, "name": "real_test", "method": "real_test", "eps": None, "seed": None}]
    else:
        if not (args.synth and args.method is not None and args.eps is not None and args.seed is not None):
            ap.error("give --synth, --method, --eps and --seed (or --jobs / --reference)")
        jobs = [{"synth": args.synth, "name": args.name, "method": args.method, "eps": args.eps, "seed": args.seed,
                 "downstream": args.downstream}]
    out_dir = args.out_dir or (RESULTS / "bench" / args.dataset)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    catalog = Catalog(args.dataset)
    refs = [r for r in args.refs.split(",") if r]
    real = {r: from_frame(load_split(args.dataset, r), catalog)[0] for r in set(refs) | {"test"}}
    names = args.metrics.split(",") if args.metrics else None
    bench = bm.Benchmark(catalog, P, names)
    preps = {r: bench.prepare(real[r]) for r in refs}
    t_load = time.time() - t0
    print(f"[{datetime.now():%H:%M:%S}] {args.dataset}: catalog {len(catalog)} venues, "
          + ", ".join(f"{r} {len(real[r])} trails" for r in refs) + f" ({t_load:.1f}s)", flush=True)

    for job in jobs:
        is_ref = bool(job.get("reference"))
        name = job.get("name") or f"{job['method']}_eps{job['eps']:g}_s{job['seed']}"
        path = out_dir / f"{name}.json"
        if args.skip_existing and path.exists():
            print(f"skip {path}")
            continue
        t1 = time.time()
        if is_ref:
            ts = real["test"]
            n_rows = n_dropped = None
            rrefs = [r for r in refs if r != "test"]
        else:
            df = pd.read_parquet(job["synth"])
            ts = from_frame(df, catalog)[0]
            n_rows, n_dropped = int(len(df)), int(len(df) - len(ts.poi))
            rrefs = refs
        rel = bench.prepare(ts)
        t_prep = time.time() - t1
        res = {"protocol": P["version"], "dataset": args.dataset, "name": name, "method": job.get("method"),
               "eps": job.get("eps"), "seed": job.get("seed"), "synth": job.get("synth"), "reference": is_ref,
               "n_trails": int(rel.n), "n_checkins": int(rel.n_checkins), "release_rows": n_rows,
               "rows_dropped_outside_catalog_or_short": n_dropped,
               "primary_reference": rrefs[0] if rrefs else None}
        timings, rt = {}, {"prep": round(t_prep, 2)}
        for r in rrefs:
            t2 = time.time()
            res[f"vs_{r}"] = bench.compare(preps[r], rel, timings)
            rt[f"vs_{r}"] = round(time.time() - t2, 2)
        if job.get("downstream"):
            res["next_poi"] = load_downstream(job["downstream"])
        rt["total"] = round(time.time() - t1, 2)
        res["runtime_s"] = rt
        res["metric_runtime_s"] = {k: round(v, 3) for k, v in sorted(timings.items(), key=lambda kv: -kv[1])}
        res["freeze"] = {"frozen": freeze["frozen"], "mismatch": freeze["mismatch"]}
        res["created"] = datetime.now().isoformat(timespec="seconds")
        res["data_dir"] = str(PROCESSED / args.dataset)
        res["argv"] = sys.argv if argv is None else argv
        path.write_text(json.dumps(res, indent=1, default=float))
        head = res.get(f"vs_{rrefs[0]}", {}) if rrefs else {}
        print(f"[{datetime.now():%H:%M:%S}] {name}: {rel.n} trails, {rt}  "
              + "  ".join(f"{k}={v:.4f}" for k, v in list(head.items())[:6]) + f" -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
