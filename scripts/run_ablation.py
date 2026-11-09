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
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SNAP = Path(os.environ.get("TRAJPE_ABL_SNAP", _TP_ROOT))
FREEZE = PROJECT / "frozen" / "ablations_v4.sha256"
CODE = ("trajpe/ablations.py", "scripts/run_ablation.py")
NOMETRIC = "not computed (TRAJPE_GRID_NOMETRICS=1)"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def verify_code() -> dict:
    if not FREEZE.exists():
        return {}
    want = {}
    for ln in FREEZE.read_text().splitlines():
        if ln.strip() and not ln.startswith("#"):
            h, p = ln.split(maxsplit=1)
            want[p.strip()] = h
    got = {}
    for p in CODE:
        got[p] = sha(PROJECT / p)
        if want.get(p) != got[p]:
            raise SystemExit(f"{p} differs from {FREEZE.name}")
    man = SNAP / "code.sha256"
    got["snapshot_manifest"] = sha(man)
    for ln in man.read_text().splitlines():
        h, p = ln.split(maxsplit=1)
        p = p.strip()
        if p.startswith(("trajpe/", "baselines/dplora/hybrid.py", "baselines/dplora/textfmt.py", "scripts/run_hybrid.py",
                         "scripts/run_pe.py", "configs/v4_")) and sha(SNAP / p) != h:
            raise SystemExit(f"snapshot file {p} differs from its manifest")
    return got


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--eps", type=float, required=True, help="TOTAL epsilon of the v4 generator's calibration")
    ap.add_argument("--work", required=True, help="v4 generator work dir (calib.json, train.json, texts.parquet)")
    ap.add_argument("--abl", required=True, help="comma list of ab0, ab1, ab2, ab3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag_prefix", default="abl")
    a = ap.parse_args()
    if a.dataset in ("gw_austin", "gw_nyc"):
        raise SystemExit(f"refusing {a.dataset} (sealed test / smoke only)")
    which = [x for x in a.abl.split(",") if x]
    code = verify_code()
    sys.path[:0] = [str(SNAP), str(SNAP / "scripts"), str(SNAP / "baselines" / "dplora")]
    spec = importlib.util.spec_from_file_location("trajpe_ablations", PROJECT / "trajpe" / "ablations.py")
    AB = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(AB)
    if any(x not in AB.ABLATIONS for x in which):
        raise SystemExit(f"unknown ablation in {which}")
    import numpy as np
    import hybrid as HY
    import run_hybrid as RH
    import trajpe
    if Path(trajpe.__file__).resolve().parent != (SNAP / "trajpe").resolve():
        raise SystemExit(f"trajpe resolved to {trajpe.__file__}, not the snapshot")
    from trajpe.catalog import Catalog
    from trajpe.data import load_split
    from trajpe.paths import RUNS
    from trajpe.trails import from_frame, to_frame

    nometrics = os.environ.get("TRAJPE_GRID_NOMETRICS") == "1"
    cfgf = SNAP / "configs" / ("v4_large_eps.json" if a.eps >= 4 else "v4_small_eps.json")
    cfg = json.loads(cfgf.read_text())
    if cfg.get("_accepted") is not True or not cfg.get("_frozen"):
        raise SystemExit(f"{cfgf} is not the frozen v4 configuration")
    work = Path(a.work)
    cal, tj, texts = RH.load_generator(work, a.eps)
    catalog = Catalog(a.dataset)
    recs = RH.parse_texts(texts, catalog.categories, int(cfg["pool_n"]))
    priv, users = from_frame(load_split(a.dataset, "train"), catalog)
    mu_pe = float(cal["mu_pe_run"])
    t0 = time.time()
    logs = []

    def log(x):
        print(json.dumps(x, default=float), flush=True)
        logs.append(x)

    res = {}
    if {"ab0", "ab1", "ab2"} & set(which):
        pool, _ = AB.pool_from_texts(catalog, recs, cfg["ground_k"], cfg["ground_scale_km"], cfg["pop_radius_km"], a.seed)
        log({"pool": len(pool), "grounding": "public (nearest venue of the category)", "sec": round(time.time() - t0, 1)})
        res.update(AB.run_ab012(priv, catalog, pool, which, mu_pe, a.seed, log=log))
    if "ab3" in which:
        S, mechs, info = AB.run_ab3(priv, users, catalog, recs, cfg, mu_pe, a.seed, log=log)
        res["ab3"] = (S, mechs, info)
    runtime = round(time.time() - t0, 1)
    for ab in which:
        S, mechs, info = res[ab]
        acc = AB.accounting(cal, tj, mechs, HY)
        tag = f"{a.tag_prefix}_{ab}_e{a.eps:g}_s{a.seed}"
        out = RUNS / a.dataset / tag
        out.mkdir(parents=True, exist_ok=True)
        S.save(out / "synth.npz")
        to_frame(S, catalog).to_parquet(out / "synth.parquet", index=False)
        side = {"dataset": a.dataset, "tag": tag, "ablation": ab, "doc": "docs/ABLATIONS_v4.md",
                "generator": {"work": str(work), "texts": str(texts), "calib": cal},
                "privacy": {"eps": acc["eps_release"], "eps_budget": acc["eps_budget"], "delta": acc["delta"],
                            "unit": "trail", "mu_pe_spent": acc["mu_pe_spent"], "n_out": AB.N_OUT},
                "accounting": acc, "info": info, "config": str(cfgf), "seed": a.seed, "code": code,
                "snapshot": str(SNAP), "runtime_sec": runtime}
        if nometrics:
            side["metrics"] = NOMETRIC
        (out / "result.json").write_text(json.dumps(side, indent=1, default=float))
        print(json.dumps({"tag": tag, "eps_release": acc["eps_release"], "eps_budget": acc["eps_budget"],
                          "independent": acc["independent_check"]["eps_pld_per_mechanism"]}), flush=True)
    print("ABL_DONE", a.dataset, a.eps, ",".join(which), flush=True)


if __name__ == "__main__":
    main()
