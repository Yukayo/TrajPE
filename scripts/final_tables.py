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
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

import bench_tables as BT
from trajpe import bench_metrics as bm

ALPHA = 0.05
FAMILIES_FILE = ROOT / "configs" / "benchmark_protocol_v1.2.1_families.json"
V3_CONFIGS = (ROOT / "configs" / "v4_large_eps.json", ROOT / "configs" / "v4_small_eps.json")
FREEZE = ROOT / "frozen" / "final_analysis_plan.sha256"
NEXTPOI_FAMILIES = ("nextpoi_transformer", "nextpoi_markov")
NEAR_ZERO = 1e-3
P14, P28, PALL = (1, 4), (2, 8), (1, 2, 4, 8)

ANALYSES = [
    dict(key="A1_primary", pool="primary", comps="trail", eps=P14, governs="primary SOTA (a), (b); strict (d), (e)"),
    dict(key="A3_primary_7b", pool="primary_7b", comps="trail7b", eps=P14, governs="primary SOTA (c); strict (d), (e)"),
    dict(key="X1_primary_eps28", pool="primary", comps="trail", eps=P28,
         governs="secondary eps {2, 8} verdict (a), (b); strict (d), (e)"),
    dict(key="A1_primary_failed_as_loss", pool="primary", comps="trail", eps=P14, impute_failed=True),
    dict(key="A3_primary_7b_failed_as_loss", pool="primary_7b", comps="trail7b", eps=P14, impute_failed=True),
    dict(key="S1_secondary", pool="secondary", comps="trail", eps=P14),
    dict(key="S1_secondary_7b", pool="secondary_7b", comps="trail7b", eps=P14),
    dict(key="S2_lowdata", pool="lowdata", comps="trail", eps=P14),
    dict(key="S2_lowdata_failed_as_loss", pool="lowdata", comps="trail", eps=P14, impute_failed=True),
    dict(key="S2_lowdata_no_beijing", pool="lowdata", comps="trail", eps=P14, drop=("beijing",)),
    dict(key="S3_all18", pool="all18", comps="trail", eps=P14),
    dict(key="S3_all18_failed_as_loss", pool="all18", comps="trail", eps=P14, impute_failed=True),
    dict(key="S3_all18_no_beijing", pool="all18", comps="trail", eps=P14, drop=("beijing",)),
    dict(key="S4_unseen", pool="unseen", comps="trail", eps=P14),
    dict(key="S5_primary_ref_test", pool="primary", comps="trail", eps=P14, ref="test"),
    dict(key="S5_primary_7b_ref_test", pool="primary_7b", comps="trail7b", eps=P14, ref="test"),
    dict(key="S6_primary_eps1", pool="primary", comps="trail", eps=(1,)),
    dict(key="S6_primary_eps4", pool="primary", comps="trail", eps=(4,)),
    dict(key="S7_primary_7b_all10", pool="primary_7b", comps="trail+trail7b", eps=P14),
    dict(key="S8_primary_all_eps", pool="primary", comps="trail", eps=PALL),
    dict(key="X1_primary_eps28_failed_as_loss", pool="primary", comps="trail", eps=P28, impute_failed=True),
    dict(key="X_secondary_eps28", pool="secondary", comps="trail", eps=P28),
    dict(key="X_lowdata_eps28", pool="lowdata", comps="trail", eps=P28),
    dict(key="X_lowdata_eps28_failed_as_loss", pool="lowdata", comps="trail", eps=P28, impute_failed=True),
    dict(key="X_lowdata_eps28_no_beijing", pool="lowdata", comps="trail", eps=P28, drop=("beijing",)),
    dict(key="X_all18_eps28", pool="all18", comps="trail", eps=P28),
    dict(key="X_all18_eps28_failed_as_loss", pool="all18", comps="trail", eps=P28, impute_failed=True),
    dict(key="X_all18_eps28_no_beijing", pool="all18", comps="trail", eps=P28, drop=("beijing",)),
    dict(key="X_unseen_eps28", pool="unseen", comps="trail", eps=P28),
    dict(key="X_primary_ref_test_eps28", pool="primary", comps="trail", eps=P28, ref="test"),
    dict(key="X_primary_eps2", pool="primary", comps="trail", eps=(2,)),
    dict(key="X_primary_eps8", pool="primary", comps="trail", eps=(8,)),
    dict(key="U_primary_user", pool="primary", comps="user", eps=P14),
    dict(key="U_lowdata_user", pool="lowdata", comps="user", eps=P14),
    dict(key="U_primary_user_eps28", pool="primary", comps="user", eps=P28),
    dict(key="U_lowdata_user_eps28", pool="lowdata", comps="user", eps=P28),
    dict(key="B_primary_ablation", pool="primary", comps="ablation", eps=P14, governs="ablation claims (plan section 9)"),
    dict(key="B_secondary_ablation", pool="secondary", comps="ablation", eps=P14),
    dict(key="B_lowdata_ablation", pool="lowdata", comps="ablation", eps=P14),
    dict(key="B_primary_ablation_eps28", pool="primary", comps="ablation", eps=P28),
    dict(key="J_jakarta", pool="jakarta", comps="trail+trail7b", eps=P14),
    dict(key="J_jakarta_ablation", pool="jakarta", comps="ablation", eps=P14),
    dict(key="P_primary_prior", pool="primary", comps="prior", eps=P14),
    dict(key="P_primary_prior_eps28", pool="primary", comps="prior", eps=P28),
]
SPEC = {s["key"]: s for s in ANALYSES}
VERDICTS = {
    "primary": ("A1_primary", "A3_primary_7b", P14),
    "secondary_eps28": ("X1_primary_eps28", None, P28),
}
ABLATION_CLAIMS = {
    "abl_ab3": "the PE vote rounds add value over workload I-projection of the same DP pool",
    "abl_ab2": "v4 beats PATH-style noisy top-k selection on the same generator and total eps",
    "abl_ab1": "v4 beats Yu et al.'s one-shot DP resampling on the same generator and total eps",
    "abl_ab0": "v4's use of mu_PE adds over the DP generator alone",
    "trajpe_e2": "the DP-SGD generator adds over the training-free E2 (same PE, full budget)",
}


PREREG_MARK = "<!-- END OF PRE-REGISTRATION -->"


def frozen_hash(rel: str) -> str:
    if rel.endswith("#pre-registration"):
        f = ROOT / rel.split("#", 1)[0]
        if not f.exists() or PREREG_MARK not in f.read_text():
            return "<missing or no marker>"
        return hashlib.sha256(f.read_text().split(PREREG_MARK, 1)[0].encode()).hexdigest()
    f = Path(rel) if rel.startswith("/") else ROOT / rel
    return hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else "<missing>"


REPORT_ONLY = "#report-only"


def verify_freeze(path: Path, report_only: bool = False) -> list[str]:
    if not path.exists():
        return [f"<missing {path}>"]
    bad = []
    for ln in path.read_text().splitlines():
        if not ln.strip() or ln.startswith("#"):
            continue
        h, rel = ln.split(None, 1)
        rel = rel.strip()
        ro = rel.endswith(REPORT_ONLY)
        if ro != report_only:
            continue
        if frozen_hash(rel[:-len(REPORT_ONLY)] if ro else rel) != h:
            bad.append(rel)
    return bad


def read_inventory(paths) -> list[dict]:
    rows = []
    for p in paths:
        with open(p, newline="") as f:
            rows += list(csv.DictReader(f))
    for r in rows:
        r["eps"] = int(float(r["eps"])) if r.get("eps") not in ("", None) else None
    return rows


def pools(datasets_cfg: dict, inv: list[dict]) -> dict:
    role = {d: v["role"] for d, v in datasets_cfg.items()}
    status = {d: v.get("status", "") for d, v in datasets_cfg.items()}
    present = {r["dataset"] for r in inv}
    has7b = {r["dataset"] for r in inv if r["group"] == "trail7b"}
    primary = [d for d in datasets_cfg if role[d] == "evaluation" and d in present]
    secondary = primary + [d for d in datasets_cfg if role[d] in ("evaluation_devcity", "development") and d in present]
    lowdata = [d for d in datasets_cfg if role[d] == "evaluation_lowdata" and d in present]
    return {"primary": primary, "secondary": secondary, "lowdata": lowdata, "all18": secondary + lowdata,
            "unseen": [d for d in primary if status[d] == "new"],
            "primary_7b": [d for d in primary if d in has7b], "secondary_7b": [d for d in secondary if d in has7b],
            "jakarta": [d for d in datasets_cfg if role[d] == "tuning" and d in present]}


def comparator_sets(inv: list[dict], ours: str) -> dict:
    g = defaultdict(list)
    for r in inv:
        m = r["method"]
        if m == ours or r["group"].startswith("reference"):
            continue
        key = "prior" if r["group"] == "ours" else r["group"]
        if m not in g[key]:
            g[key].append(m)
    out = dict(g)
    out["trail+trail7b"] = out.get("trail", []) + out.get("trail7b", [])
    return out


class Data:

    def __init__(self, inv, bench_root: Path, ours: str):
        self.inv, self.ours, self.bench_root = inv, ours, bench_root
        self.P = bm.load_protocol()
        self.F = json.loads(FAMILIES_FILE.read_text())
        BT.add_descriptors(self.P, self.F)
        self.info = bm.metric_info(self.P)
        _, self.fam_of, self.suites = BT.load_families(FAMILIES_FILE, self.info)
        self.paths, self.missing, self.problems = {}, [], []
        for r in inv:
            if r["status"] != "ok" or r["group"] == "reference_nextpoi":
                continue
            p = bench_root / r["dataset"] / f"{r['bench_name']}.json"
            if p.exists():
                self.paths[(r["dataset"], r["method"], r["eps"])] = p
            else:
                self.missing.append({"dataset": r["dataset"], "method": r["method"], "eps": r["eps"],
                                     "reason": "no bench JSON"})
        self.V, self.ref = {}, {}
        for refname in ("train", "test"):
            rows = BT.load(sorted(map(str, self.paths.values())), refname)
            V, R = defaultdict(dict), {}
            for row in rows:
                for m, v in row["metrics"].items():
                    if m not in self.info or v is None or not np.isfinite(v):
                        continue
                    if row["reference"]:
                        R[(row["dataset"], m)] = float(v)
                    else:
                        V[(row["dataset"], int(row["eps"]), m)][row["method"]] = float(v)
            self.V[refname], self.ref[refname] = V, R
        for (d, m, e), p in self.paths.items():
            if m == "real_test":
                continue
            js = json.loads(p.read_text())
            want = next(r for r in inv if r["dataset"] == d and r["method"] == m and r["eps"] == e)
            if js.get("synth") != want["release"] or js.get("method") != m or js.get("seed") != 0 or \
                    float(js.get("eps")) != e:
                self.problems.append(f"{p}: bench JSON does not match the inventory row or is not frozen")
            for model in ("transformer", "markov"):
                if model not in (js.get("next_poi") or {}):
                    self.missing.append({"dataset": d, "method": m, "eps": e,
                                         "reason": f"no next-POI {model} block (family nextpoi_{model} excluded)"})
        all_names = list(self.info)
        splits = [bm.v3_split(json.loads(c.read_text()), self.P, all_names) for c in V3_CONFIGS]
        if any({k: v[0] for k, v in s.items()} != {k: v[0] for k, v in splits[0].items()} for s in splits[1:]):
            sys.exit("v3 split differs between configs/v4_large_eps.json and v4_small_eps.json")
        self.split = splits[0]
        present = set()
        for V in self.V.values():
            for c in V:
                present.add(c[2])
        self.all_metrics = [m for m in self.info if m in present]
        self.order = {m: i for i, m in enumerate(self.all_metrics)}

    def cellkeys(self, ref, datasets, eps):
        ks = [c for c in self.V[ref] if c[0] in datasets and c[1] in eps]
        return sorted(ks, key=lambda c: (c[0], c[1], self.order[c[2]]))


def fam_cells(D: Data, V, keys, ours, b, keep, imputed=()):
    acc = defaultdict(list)
    for c in keys:
        m = c[2]
        if D.info[m]["role"] != "primary" or not keep(m):
            continue
        vo, vb = V[c].get(ours), V[c].get(b)
        if vo is None:
            continue
        if vb is None:
            if (c[0], c[1]) in imputed:
                acc[(c[0], c[1], D.fam_of[m])].append((m, 1.0))
            continue
        acc[(c[0], c[1], D.fam_of[m])].append((m, BT.natural_diff(vo, vb, D.info[m]["direction"])))
    return {k: (float(np.mean([x[1] for x in v])), [x[0] for x in v]) for k, v in acc.items()}


def metric_cells(D: Data, V, keys, ours, b, keep, imputed=()):
    out = []
    for c in keys:
        m = c[2]
        if D.info[m]["role"] != "primary" or not keep(m):
            continue
        vo, vb = V[c].get(ours), V[c].get(b)
        if vo is None:
            continue
        if vb is None:
            if (c[0], c[1]) in imputed:
                out.append((c, 1.0, "win"))
            continue
        d = D.info[m]["direction"]
        lo, lb = BT.loss(vo, d), BT.loss(vb, d)
        tie = abs(lo - lb) <= 1e-12 * max(1.0, abs(lo), abs(lb))
        out.append((c, BT.norm_diff(lb, lo), "tie" if tie else "win" if lo < lb else "loss"))
    return out


def v11_tests(ds):
    ds = np.asarray(ds, float)
    nz = ds[ds != 0]
    from scipy import stats
    pw = float(stats.wilcoxon(nz).pvalue) if len(nz) >= 2 else float("nan")
    ps = float(stats.binomtest(int((nz > 0).sum()), len(nz), 0.5).pvalue) if len(nz) else float("nan")
    return pw, ps, nz


def v11_verdict(p_holm, nz):
    med = float(np.median(nz)) if len(nz) else 0.0
    if np.isfinite(p_holm) and p_holm < ALPHA and med != 0:
        return "win" if med > 0 else "loss"
    w, l_ = int((nz > 0).sum()), int((nz < 0).sum())
    return "n.s. (" + ("more wins" if w > l_ else "more losses" if l_ > w else "even") + ")"


def family_summary(D, V, keys, ours, comps, keep, imputed):
    Ds, fcs, pw, ps = {}, {}, {}, {}
    for b in comps:
        fcs[b] = fam_cells(D, V, keys, ours, b, keep, imputed.get(b, ()))
        Ds[b] = [fcs[b][k][0] for k in sorted(fcs[b])]
        pw[b], ps[b] = BT._tests(Ds[b])
    hw, hs = BT.holm(pw), BT.holm(ps)
    res = {}
    for b in comps:
        w, t, l_ = BT._wtl(Ds[b])
        res[b] = {"cells": len(Ds[b]), "wins": w, "ties": t, "losses": l_,
                  "median_D": float(np.median(Ds[b])) if Ds[b] else None, "wilcoxon_p": pw[b], "sign_p": ps[b],
                  "holm_wilcoxon_p": hw[b], "holm_sign_p": hs[b], "verdict": BT._verdict(hw[b], Ds[b], ALPHA)}
    return res, fcs


def metric_summary(D, V, keys, ours, comps, keep, imputed):
    res, pw, ps, nzs, mcs = {}, {}, {}, {}, {}
    for b in comps:
        mcs[b] = metric_cells(D, V, keys, ours, b, keep, imputed.get(b, ()))
        pw[b], ps[b], nzs[b] = v11_tests([x[1] for x in mcs[b]])
    hw, hs = BT.holm(pw), BT.holm(ps)
    for b in comps:
        oc = [x[2] for x in mcs[b]]
        ds = [x[1] for x in mcs[b]]
        res[b] = {"cells": len(ds), "wins": oc.count("win"), "ties": oc.count("tie"), "losses": oc.count("loss"),
                  "median_d": float(np.median(ds)) if ds else None, "wilcoxon_p": pw[b], "sign_p": ps[b],
                  "holm_wilcoxon_p": hw[b], "holm_sign_p": hs[b], "verdict": v11_verdict(hw[b], nzs[b])}
    return res, mcs


def per_paper(D, V, keys, ours, comps, imputed):
    out = {}
    for b in comps:
        pw, ps, Ds, PC = {}, {}, {}, {}
        for p, ms in D.suites.items():
            S = set(ms)
            fc = fam_cells(D, V, keys, ours, b, lambda m, S=S: m in S, imputed.get(b, ()))
            Ds[p] = [fc[k][0] for k in sorted(fc)]
            pw[p], ps[p] = BT._tests(Ds[p])
            byc = defaultdict(list)
            for k, v in fc.items():
                byc[(k[0], k[1])].append(v[0])
            PC[p] = [float(np.mean(v)) for v in byc.values()]
        hw, hs = BT.holm(pw), BT.holm(ps)
        for p in D.suites:
            w, t, l_ = BT._wtl(Ds[p])
            pc = BT._wtl(PC[p])
            out.setdefault(b, {})[p] = {
                "verdict": BT._verdict(hw[p], Ds[p], ALPHA), "family_cells": len(Ds[p]), "wins": w, "ties": t,
                "losses": l_, "median_D": float(np.median(Ds[p])) if Ds[p] else None, "wilcoxon_p": pw[p],
                "sign_p": ps[p], "holm_wilcoxon_p": hw[p], "holm_sign_p": hs[p], "paper_cells": len(PC[p]),
                "paper_cell_wins": pc[0], "paper_cell_ties": pc[1], "paper_cell_losses": pc[2]}
    return out


def analyze(D: Data, spec: dict, sets: dict, inv: list[dict], pools_: dict) -> dict:
    ours, ref = D.ours, spec.get("ref", "train")
    datasets = [d for d in pools_[spec["pool"]] if d not in spec.get("drop", ())]
    eps = tuple(spec["eps"])
    have = {r["method"] for r in inv if r["dataset"] in datasets and r["eps"] in eps}
    comps = [b for b in sets.get(spec["comps"], []) if b in have]
    V = D.V[ref]
    keys = D.cellkeys(ref, datasets, eps)
    imputed = defaultdict(set)
    if spec.get("impute_failed"):
        for r in inv:
            if r["status"] == "failed-baseline" and r["method"] in comps and r["dataset"] in datasets \
                    and r["eps"] in eps:
                imputed[r["method"]].add((r["dataset"], r["eps"]))
    split = D.split
    res = {"spec": {k: (list(v) if isinstance(v, tuple) else v) for k, v in spec.items()}, "datasets": datasets,
           "eps": list(eps), "comparators": comps, "imputed": {b: sorted(v) for b, v in imputed.items()}}
    if not comps or not keys:
        res["empty"] = True
        return res
    res["family"], fcs = family_summary(D, V, keys, ours, comps, lambda m: True, imputed)
    res["metric"], mcs = metric_summary(D, V, keys, ours, comps, lambda m: True, imputed)
    res["per_paper"] = per_paper(D, V, keys, ours, comps, imputed)
    res["nextpoi"] = {}
    for f in NEXTPOI_FAMILIES:
        res["nextpoi"][f], _ = family_summary(D, V, keys, ours, comps, lambda m, f=f: D.fam_of.get(m) == f, imputed)
    res["split"] = {}
    for key, keep in (("not_measured", lambda m: split[m][0] != "measured"),
                      ("unmeasured", lambda m: split[m][0] == "unmeasured"),
                      ("measured", lambda m: split[m][0] == "measured")):
        kk = (lambda m, keep=keep: m in split and keep(m))
        res["split"][key] = {"family": family_summary(D, V, keys, ours, comps, kk, imputed)[0],
                             "metric": metric_summary(D, V, keys, ours, comps, kk, imputed)[0]}
    res["per_family"] = {}
    for f in D.F["families"]:
        for b in comps:
            Dv = [v[0] for k, v in fcs[b].items() if k[2] == f]
            w, t, l_ = BT._wtl(Dv)
            res["per_family"].setdefault(f, {})[b] = {"wins": w, "ties": t, "losses": l_,
                                                      "mean_D": float(np.mean(Dv)) if Dv else None}
    res["_fcs"], res["_mcs"], res["_keys"] = fcs, mcs, keys
    exp = {(d, e) for d in datasets for e in eps}
    res["coverage"] = {b: sorted(exp - {(k[0], k[1]) for k in fcs[b]}) for b in comps}
    res["coverage_ours"] = sorted(exp - {(c[0], c[1]) for c in keys if V[c].get(ours) is not None})
    return res


def protocol_run(D: Data, spec: dict, res: dict, out_dir: Path) -> dict:
    if spec.get("impute_failed") or res.get("empty"):
        return {"ok": None, "note": "not representable by bench_tables.py (imputed cells) or empty"}
    if not any((d, D.ours, e) in D.paths for d in res["datasets"] for e in res["eps"]):
        return {"ok": None, "note": "the method under test has no bench input in this analysis"}
    files = [str(D.paths[(d, m, e)]) for (d, m, e) in D.paths
             if d in res["datasets"] and ((m in res["comparators"] + [D.ours] and e in res["eps"]) or m == "real_test")]
    od = out_dir / spec["key"]
    argv = ["--inputs", *files, "--ours", D.ours, "--baselines", *res["comparators"], "--ref",
            spec.get("ref", "train"), "--v3-config", str(V3_CONFIGS[0]), "--families", str(FAMILIES_FILE),
            "--seeds", "0", "--out-dir", str(od)]
    BT.main(argv)
    sf = json.loads((od / "summary_families.json").read_text())["summary"]
    s1 = json.loads((od / "summary.json").read_text())["summary"]
    diffs = []

    def same(a, b):
        if a is None or b is None:
            return a is b
        if isinstance(a, str) or isinstance(b, str):
            return a == b
        return (math.isnan(a) and math.isnan(b)) or abs(a - b) <= 1e-12 * max(1.0, abs(a))

    for b in res["comparators"]:
        for k in ("wins", "ties", "losses", "median_D", "wilcoxon_p", "sign_p", "holm_wilcoxon_p", "verdict"):
            if not same(res["family"][b][k], sf["all"]["baselines"][b][k]):
                diffs.append(f"family {b} {k}: {res['family'][b][k]} vs {sf['all']['baselines'][b][k]}")
        for k in ("cells", "wins", "ties", "losses", "median_d", "wilcoxon_p", "sign_p", "holm_wilcoxon_p"):
            if not same(res["metric"][b][k], s1["all"]["baselines"][b][k]):
                diffs.append(f"metric {b} {k}: {res['metric'][b][k]} vs {s1['all']['baselines'][b][k]}")
        for p, v in res["per_paper"][b].items():
            for k in ("verdict", "wins", "ties", "losses", "holm_wilcoxon_p"):
                if not same(v[k], sf["per_paper"][p][b][k]):
                    diffs.append(f"per-paper {b} {p} {k}: {v[k]} vs {sf['per_paper'][p][b][k]}")
        for key in ("not_measured", "unmeasured", "measured"):
            for k in ("wins", "losses", "holm_wilcoxon_p", "verdict"):
                if not same(res["split"][key]["family"][b][k], sf[key]["baselines"][b][k]):
                    diffs.append(f"split {key} {b} {k}")
    ranks = sf["all"].get("ranks")
    return {"ok": not diffs, "diffs": diffs, "out_dir": str(od), "ranks": ranks}


def incomplete_inputs(D: Data, inv: list[dict], P: dict, sets: dict, eps: tuple, with7b: bool) -> list[str]:
    ours = D.ours
    pool_of = {ours: set(P["primary"]) | (set(P["primary_7b"]) if with7b else set())}
    pool_of.update({b: set(P["primary"]) for b in sets.get("trail", [])})
    if with7b:
        pool_of.update({b: set(P["primary_7b"]) for b in sets.get("trail7b", [])})
    out = []
    for r in inv:
        m = r["method"]
        if m not in pool_of or r["eps"] not in eps or r["dataset"] not in pool_of[m]:
            continue
        if r["status"] in ("missing", "invalid"):
            out.append(f"{r['dataset']} {r['bench_name']}: inventory status {r['status']}")
    for x in D.missing:
        m = x["method"]
        if m in pool_of and x["eps"] in eps and x["dataset"] in pool_of[m] and (
                x["reason"] == "no bench JSON" or (m == ours and x["reason"].startswith("no next-POI"))):
            out.append(f"{x['dataset']} {m} eps {x['eps']}: {x['reason']}")
    return out


def rule_verdict(R: dict, sets: dict, ours: str, main_key: str, b7_key: str | None, incomplete: list) -> dict:
    a1 = R[main_key]
    a3 = R[b7_key] if b7_key else {}
    trail, b7 = sets.get("trail", []), (sets.get("trail7b", []) if b7_key else [])

    def fam_ok(res, comps):
        return {b: res.get("family", {}).get(b, {}).get("verdict") == "win" for b in comps}

    def np_ok(res, comps):
        return {f"{b}/{f}": b in res.get("nextpoi", {}).get(f, {}) and not str(
            res["nextpoi"][f][b]["verdict"]).startswith("loss") for b in comps for f in NEXTPOI_FAMILIES}

    def met_ok(res, comps):
        return {b: res.get("metric", {}).get(b, {}).get("verdict") == "win" for b in comps}

    def pp(res, comps, good):
        out = {}
        for b in comps:
            per = res.get("per_paper", {}).get(b)
            if not per:
                out[f"{b}/(none)"] = False
                continue
            out.update({f"{b}/{p}": good(v["verdict"]) for p, v in per.items()})
        return out

    win, not_loss = (lambda v: v == "win"), (lambda v: v != "loss")
    comp = {"a_family_primary": fam_ok(a1, trail), "b_nextpoi_primary": np_ok(a1, trail),
            "d_metric_level": met_ok(a1, trail), "e_per_paper_win": pp(a1, trail, win),
            "e_report_per_paper_no_loss": pp(a1, trail, not_loss)}
    if b7_key:
        comp["c1_family_7b"] = fam_ok(a3, b7)
        comp["c2_nextpoi_7b"] = np_ok(a3, b7)
        comp["d_metric_level"].update({f"7b:{k}": v for k, v in met_ok(a3, b7).items()})
        comp["e_per_paper_win"].update({f"7b:{k}": v for k, v in pp(a3, b7, win).items()})
        comp["e_report_per_paper_no_loss"].update({f"7b:{k}": v for k, v in pp(a3, b7, not_loss).items()})
    need = ["a_family_primary", "b_nextpoi_primary"] + (["c1_family_7b", "c2_nextpoi_7b"] if b7_key else [])
    sota = all(bool(comp[k]) and all(comp[k].values()) for k in need)
    strict = sota and all(bool(comp[k]) and all(comp[k].values()) for k in ("d_metric_level", "e_per_paper_win"))
    failing = {k: sorted(x for x, ok in v.items() if not ok) for k, v in comp.items()}
    label = ("incomplete" if incomplete else "strict SOTA" if strict else
             "SOTA, not strict" if sota else "not SOTA")
    return {"ours": ours, "label": label, "sota": bool(sota) and not incomplete,
            "strict_sota": bool(strict) and not incomplete, "analyses": [main_key] + ([b7_key] if b7_key else []),
            "components": comp, "failing": failing, "incomplete": incomplete}


def fmt(x, nd=4):
    return BT.fmt(x, nd) if isinstance(x, (int, float)) and x is not None else ("-" if x is None else str(x))


def write_csv(path: Path, rows: list[dict], fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for r in rows:
            fields += [k for k in r if k not in fields]
        fields = fields or ["empty"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def loss_rows(D: Data, key: str, res: dict, ref: str):
    V, R = D.V[ref], D.ref[ref]
    fam_rows, met_rows = [], []
    for b, fc in res.get("_fcs", {}).items():
        for (d, e, f), (Dv, ms) in sorted(fc.items()):
            if Dv >= -BT.TIE:
                continue
            det = []
            for m in ms:
                vo, vb = V[(d, e, m)].get(D.ours), V[(d, e, m)].get(b)
                dirn = D.info[m]["direction"]
                rv = R.get((d, m))
                dd = 1.0 if vb is None else BT.natural_diff(vo, vb, dirn)
                near = vb is not None and max(abs(vo), abs(vb)) < NEAR_ZERO
                within = (rv is not None and vb is not None and
                          ((vo <= rv and vb <= rv) if dirn == "lower" else (vo >= rv and vb >= rv)))
                met_rows.append({"analysis": key, "comparator": b, "dataset": d, "eps": e, "family": f, "metric": m,
                                 "direction": dirn, "v_ours": vo, "v_comparator": vb, "d": dd, "real_test_ref": rv,
                                 "v3_split": D.split[m][0], "near_zero": near, "both_within_real_test": within})
                det.append(f"{m}={fmt(vo)}/{fmt(vb)}")
            fam_rows.append({"analysis": key, "comparator": b, "dataset": d, "eps": e, "family": f, "D": Dv,
                             "n_metrics": len(ms), "v3_split": ",".join(sorted({D.split[m][0] for m in ms})),
                             "metrics_ours_vs_comparator": " ".join(det)})
    return fam_rows, met_rows


def metric_loss_rows(D: Data, key: str, res: dict, ref: str):
    V = D.V[ref]
    rows = []
    for b, mc in res.get("_mcs", {}).items():
        for (c, dv, oc) in mc:
            if oc != "loss":
                continue
            rows.append({"analysis": key, "comparator": b, "dataset": c[0], "eps": c[1], "metric": c[2],
                         "family": D.fam_of.get(c[2], ""), "direction": D.info[c[2]]["direction"],
                         "v_ours": V[c].get(D.ours), "v_comparator": V[c].get(b), "d_v11": dv,
                         "source": D.info[c[2]]["source_paper"], "v3_split": D.split[c[2]][0]})
    return rows


def baseline_wins_by_cell(D: Data, key: str, res: dict):
    by = defaultdict(list)
    for b, fc in res.get("_fcs", {}).items():
        for k, (Dv, _) in fc.items():
            if Dv < -BT.TIE:
                by[k].append((Dv, b))
    rows = []
    for (d, e, f), lst in sorted(by.items()):
        lst.sort()
        rows.append({"analysis": key, "dataset": d, "eps": e, "family": f, "n_comparators_beating_ours": len(lst),
                     "comparators_beating_ours": " ".join(f"{b}({Dv:.3f})" for Dv, b in lst),
                     "strongest": lst[0][1]})
    return rows


def loss_summary_by_family(D: Data, key: str, res: dict):
    rows = []
    for f, per in res.get("per_family", {}).items():
        tot_l = sum(v["losses"] for v in per.values())
        fc = res["_fcs"]
        lost = [(v[0], b, k) for b in fc for k, v in fc[b].items() if k[2] == f and v[0] < -BT.TIE]
        worst = sorted(lost)[:3]
        rows.append({"analysis": key, "family": f, "total_losses": tot_l,
                     **{f"{b}": f"{v['wins']}-{v['ties']}-{v['losses']} ({fmt(v['mean_D'], 3)})" for b, v in per.items()},
                     "worst_cells": "; ".join(f"{b} {k[0]} eps{k[1]} D={Dv:.3f}" for Dv, b, k in worst)})
    return rows


def per_dataset_tables(D: Data, inv, sets: dict, out_dir: Path, ds_ref: dict):
    order = [D.ours] + sets.get("trail", []) + sets.get("trail7b", []) + sets.get("ablation", []) + \
        sets.get("user", []) + sets.get("prior", [])
    main_cols = set([D.ours] + sets.get("trail", []) + sets.get("trail7b", []))
    V, R = D.V["train"], D.ref["train"]
    foot = {(r["dataset"], r["method"], r["eps"]): r["footnote"] for r in inv if r.get("footnote")}
    for d in sorted({c[0] for c in V}):
        lines = [f"# {d}: bench-v1.2.1 metric values (reference = real train)", "",
                 "Bold = best among the method under test and the trail-level baselines (lower is better unless "
                 "'higher'). 'real test' = the real test split scored against train (sampling reference); "
                 "'real train 50k' = next-POI trained on 50,000 real train trails (context). Values of failed or n/a "
                 "releases are '-'. eps 1 and 4 are the primary analysis, eps 2 and 8 the secondary one.", ""]
        for e in PALL:
            cols = [m for m in order if any(V[c].get(m) is not None for c in V if c[0] == d and c[1] == e)]
            if not cols:
                continue
            lines += [f"## {d}, eps = {e} ({'primary' if e in P14 else 'secondary'})", "",
                      "| metric | dir | role | v3 | " + " | ".join(cols) + " | real test | real train 50k |",
                      "|---|---|---|---|" + "---|" * (len(cols) + 2)]
            for m in D.all_metrics:
                c = (d, e, m)
                if c not in V:
                    continue
                dirn = D.info[m]["direction"]
                vals = {x: V[c].get(x) for x in cols}
                best = [BT.loss(v, dirn) for x, v in vals.items() if v is not None and x in main_cols]
                bl = min(best) if best else None
                cells = []
                for x in cols:
                    v = vals[x]
                    if v is None:
                        cells.append("-")
                        continue
                    s = fmt(v)
                    if x in main_cols and bl is not None and abs(BT.loss(v, dirn) - bl) <= 1e-12 * max(1.0, abs(bl)):
                        s = f"**{s}**"
                    cells.append(s)
                rt = ds_ref.get((d, m))
                lines.append(f"| {m} | {dirn} | {D.info[m]['role']} | {D.split[m][0][0].upper()} | " + " | ".join(cells)
                             + f" | {fmt(R.get((d, m)))} | {fmt(rt)} |")
            fn = [f"{m}: {foot[(d, m, e)]}" for m in cols if (d, m, e) in foot]
            lines += [""] + ([f"Footnote: {x}" for x in fn] + [""] if fn else [])
        (out_dir / "tables").mkdir(parents=True, exist_ok=True)
        (out_dir / "tables" / f"{d}.md").write_text("\n".join(lines))


def downstream_reference(inv, ds_root: Path) -> dict:
    out = {}
    for r in inv:
        if r["group"] != "reference_nextpoi":
            continue
        for model in ("transformer", "markov"):
            p = ds_root / r["dataset"] / f"real_train50k_{model}.json"
            if p.exists():
                b = json.loads(p.read_text()).get("bench_metrics", {})
                for scope in ("last", "all"):
                    for k, v in (b.get(scope) or {}).items():
                        out[(r["dataset"], f"nextpoi_{model}_{k}_{scope}")] = v
    return out


def ablation_extras(D: Data, inv, res: dict, comp_root: Path) -> list[dict]:
    from trajpe.metrics import composite_score
    rows = []
    V = D.V["train"]
    meths = [D.ours] + res.get("comparators", [])
    for d in res.get("datasets", []):
        for e in res.get("eps", []):
            row = {"analysis": res["spec"]["key"], "dataset": d, "eps": e}
            for m in meths:
                p = comp_root / d / f"{m}_eps{e}_s0.eval.json"
                row[f"composite:{m}"] = composite_score(json.loads(p.read_text())["vs_train"]) if p.exists() else None
                for k in ("nextpoi_markov_mrr_last", "nextpoi_markov_mrr_all", "nextpoi_transformer_mrr_last",
                          "nextpoi_transformer_mrr_all"):
                    row[f"{k}:{m}"] = V.get((d, e, k), {}).get(m)
            rows.append(row)
    return rows


def wtl_lower(rows, ours, b, key):
    w = t = l_ = 0
    for r in rows:
        vo, vb = r.get(f"{key}:{ours}"), r.get(f"{key}:{b}")
        if vo is None or vb is None:
            continue
        if abs(vo - vb) <= 1e-12:
            t += 1
        elif (vo < vb) == key.startswith("composite"):
            w += 1
        else:
            l_ += 1
    return f"{w}-{t}-{l_}"


def headline_md(res: dict, title: str) -> list[str]:
    if res.get("empty"):
        return [f"### {title}", "", "(no cells)", ""]
    L = [f"### {title}", "", f"Datasets: {', '.join(res['datasets'])}; eps {res['eps']}. "
         "Equal prominence (protocol section 11): family headline (bench-v1.2.1, primary): W-T-L of family cells, "
         "median D, Holm-Wilcoxon p, verdict; metric level (bench-v1.1): W-T-L of primary metric cells, median d, "
         "Holm-Wilcoxon p, verdict; per paper: suites won / n.s. / lost (lost suites named); next-POI: family verdict "
         "per model (W-T-L over dataset x eps).", "",
         "| comparator | family W-T-L | median D | Holm p | verdict | metric W-T-L | median d | Holm p | verdict | "
         "per paper (win/n.s./loss) | next-POI transformer | next-POI Markov |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for b in res["comparators"]:
        f, m = res["family"][b], res["metric"][b]
        pp = res["per_paper"][b]
        nw = sum(v["verdict"] == "win" for v in pp.values())
        nl = [p for p, v in pp.items() if v["verdict"] == "loss"]
        nn = len(pp) - nw - len(nl)
        npv = [res["nextpoi"][x][b]["verdict"] + f" ({res['nextpoi'][x][b]['wins']}-{res['nextpoi'][x][b]['ties']}-"
               f"{res['nextpoi'][x][b]['losses']})" if b in res["nextpoi"][x] else "-" for x in NEXTPOI_FAMILIES]
        L.append(f"| {b} | {f['wins']}-{f['ties']}-{f['losses']} | {fmt(f['median_D'], 3)} | {fmt(f['holm_wilcoxon_p'])}"
                 f" | {f['verdict']} | {m['wins']}-{m['ties']}-{m['losses']} | {fmt(m['median_d'], 3)} | "
                 f"{fmt(m['holm_wilcoxon_p'])} | {m['verdict']} | {nw}/{nn}/{len(nl)}{' (' + ', '.join(nl) + ')' if nl else ''}"
                 f" | {npv[0]} | {npv[1]} |")
    dis = [b for b in res["comparators"] if res["family"][b]["verdict"].split()[0] != res["metric"][b]["verdict"].split()[0]]
    L += ["", "Family vs metric-level disagreement (reporting rule): " + (", ".join(
        f"{b} (family {res['family'][b]['verdict']}, metric {res['metric'][b]['verdict']})" for b in dis) or "none"), ""]
    return L


COMP_NAMES = {
    "a_family_primary": "(a) family headline, primary pool, every trail-level baseline: significant win",
    "b_nextpoi_primary": "(b) next-POI, primary pool: no significant loss on either model's family",
    "c1_family_7b": "(c) 7B subset family headline: significant win over both DP-LoRA 7B variants",
    "c2_nextpoi_7b": "(c) 7B subset next-POI: no significant loss",
    "d_metric_level": "strict (d) bench-v1.1 metric-level headline: significant win against every comparator",
    "e_per_paper_win": "strict (e) every per-paper suite: significant win against every comparator",
    "e_report_per_paper_no_loss": "(report only) no per-paper suite significantly lost",
}


def verdict_md(v: dict, title: str) -> list[str]:
    L = [f"## {title}: **{v['label']}**", "", f"Analyses: {', '.join(v['analyses'])}.", ""]
    for k, txt in COMP_NAMES.items():
        if k not in v["components"]:
            continue
        bad = v["failing"][k]
        L.append(f"- {txt}: {'PASS' if not bad and v['components'][k] else 'FAIL'}"
                 + (f" (fails: {', '.join(bad)})" if bad else ""))
    if v["incomplete"]:
        L += ["", "**Incomplete inputs** (the verdict cannot be claimed): " + "; ".join(map(str, v["incomplete"][:20]))]
    return L + [""]


def where_loses_md(R: dict, keys) -> list[str]:
    L = []
    for key in keys:
        r = R[key]
        if r.get("empty"):
            continue
        L += [f"### {key}", "", "Per family: comparator W-T-L over dataset x eps (mean D); D < 0 = v4 loses.", "",
              "| family | " + " | ".join(r["comparators"]) + " | lost cells |",
              "|---|" + "---|" * (len(r["comparators"]) + 1)]
        for f, per in r["per_family"].items():
            tl = sum(v["losses"] for v in per.values())
            L.append(f"| {f} | " + " | ".join(f"{v['wins']}-{v['ties']}-{v['losses']} ({fmt(v['mean_D'], 2)})"
                                             for v in per.values()) + f" | {tl} |")
        L.append("")
    return L


def split_md(R: dict, key: str, title: str) -> list[str]:
    r = R[key]
    if r.get("empty"):
        return []
    L = [f"### Metrics v4 does not measure directly ({title}; v3 split with configs/v4_*_eps.json)", "",
         "| comparator | near+unmeasured family W-T-L / verdict | unmeasured W-T-L / verdict | measured W-T-L / verdict |",
         "|---|---|---|---|"]
    for b in r["comparators"]:
        s = r["split"]
        L.append(f"| {b} | " + " | ".join(
            f"{s[k]['family'][b]['wins']}-{s[k]['family'][b]['ties']}-{s[k]['family'][b]['losses']} / "
            f"{s[k]['family'][b]['verdict']}" for k in ("not_measured", "unmeasured", "measured")) + " |")
    return L + [""]


def ablation_md(D, R, key, abl_rows) -> list[str]:
    b = R[key]
    L = headline_md(b, f"{key}: v4 vs " + ", ".join(b.get("comparators", [])))
    if b.get("empty"):
        return L
    if key == "B_primary_ablation":
        L += ["Claims (made iff the family verdict is 'win'; Holm over the ablation rows of this analysis):", ""]
        for c, txt in ABLATION_CLAIMS.items():
            if c in b["family"]:
                v = b["family"][c]["verdict"]
                L.append(f"- {txt}: **{'claimed' if v == 'win' else 'not claimed'}** ({c}: {v})")
        L.append("")
    rows = [r for r in abl_rows if r["analysis"] == key]
    L += ["Composite (lower is better) and MRR alongside (reported only): v4 W-T-L against each row over "
          "dataset x eps.", "", "| row | composite | Markov MRR last | Markov MRR all | transformer MRR last | "
          "transformer MRR all |", "|---|---|---|---|---|---|"]
    for c in b["comparators"]:
        L.append(f"| {c} | " + " | ".join(wtl_lower(rows, D.ours, c, k) for k in (
            "composite", "nextpoi_markov_mrr_last", "nextpoi_markov_mrr_all", "nextpoi_transformer_mrr_last",
            "nextpoi_transformer_mrr_all")) + " |")
    return L + [""]


def results_md(D, R, verdicts, sets, inv, abl_rows, prot, args) -> str:
    L = ["# TrajPE final results (pre-registered analysis, docs/FINAL_ANALYSIS_PLAN.md)", "",
         f"Generated by scripts/final_tables.py from {len(D.paths)} bench files. Method under test: `{D.ours}`. "
         "Seed 0, delta 1e-6, reference = real train (primary). PRIMARY analysis: eps {1, 4}; SECONDARY: eps {2, 8}, "
         "the same tables and tests, reported separately. Everything below is mechanical output of the frozen plan; "
         "nothing was chosen after seeing it.", ""]
    L += verdict_md(verdicts["primary"], "Primary verdict (eps 1, 4)")
    L += headline_md(R["A1_primary"], "Primary: v4 vs every trail-level baseline (primary pool, eps 1, 4)")
    L += headline_md(R["A3_primary_7b"], "Primary, 7B subset: v4 vs DP-LoRA v2 7B plain / vp (eps 1, 4)")
    L += split_md(R, "A1_primary", "primary") + split_md(R, "A3_primary_7b", "7B subset")
    L += ["## Where v4 loses, primary (full lists: results/final/losses.csv, losses_metrics.csv, metric_losses.csv, "
          "baseline_wins_by_cell.csv)", ""] + where_loses_md(R, ("A1_primary", "A3_primary_7b"))
    L += verdict_md(verdicts["secondary_eps28"], "Secondary verdict (eps 2, 8; reported separately)")
    L += headline_md(R["X1_primary_eps28"], "Secondary: v4 vs every trail-level baseline (primary pool, eps 2, 8)")
    L += split_md(R, "X1_primary_eps28", "eps 2, 8")
    L += ["## Where v4 loses, eps 2, 8", ""] + where_loses_md(R, ("X1_primary_eps28",))
    L += ["## Ablations (same generator, same total eps; AB0-AB3 at eps 1, 4 only)", ""]
    for key in ("B_primary_ablation", "B_primary_ablation_eps28"):
        L += ablation_md(D, R, key, abl_rows)
    L += ["## User-level baselines (context only)", "",
          "v4's guarantee is trail-level only; the user-level baselines protect a whole user (one trail per user), a "
          "stronger guarantee. No user-level SOTA claim is made.", ""]
    L += headline_md(R["U_primary_user"], "primary pool, eps 1, 4") + headline_md(R["U_primary_user_eps28"],
                                                                                  "primary pool, eps 2, 8")
    L += ["## Secondary and sensitivity analyses (reported; they do not change any verdict)", "",
          "Entry: family verdict / metric-level verdict (family W-T-L). Full tables: results/final/sensitivity.md, "
          "subset_7b.md, user_level.md, ablation.md.", ""]
    for key, r in R.items():
        if key in ("A1_primary", "A3_primary_7b", "X1_primary_eps28") or r.get("empty"):
            continue
        L.append(f"- **{key}** ({', '.join(r['datasets'])}; eps {r['eps']}): " + "; ".join(
            f"{c}: {r['family'][c]['verdict']} / {r['metric'][c]['verdict']} ({r['family'][c]['wins']}-"
            f"{r['family'][c]['ties']}-{r['family'][c]['losses']})" for c in r["comparators"]))
    L += ["", "## Missing cells, failed releases and footnotes", ""]
    for r in inv:
        if r["status"] != "ok":
            L.append(f"- {r['dataset']} {r['bench_name']}: {r['status']} ({r.get('notes', '')})")
        elif r.get("footnote"):
            L.append(f"- {r['dataset']} {r['bench_name']}: footnote: {r['footnote']}")
    for m in D.missing:
        L.append(f"- {m['dataset']} {m['method']} eps {m['eps']}: {m['reason']}")
    L += ["", "## Protocol cross-check", "",
          "bench_tables.py run on the same inputs for every representable analysis (results/final/protocol/<analysis>/); "
          "headline numbers equal: " + ", ".join(f"{k} {'ok' if v.get('ok') else v.get('ok')}" for k, v in prot.items()),
          "", "## Commands", "", "```", *args.commands, "```", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inventory", type=Path, nargs="+", default=[ROOT / "results" / "final" / "inventory.csv"])
    ap.add_argument("--bench-root", type=Path, default=ROOT / "results" / "final" / "bench")
    ap.add_argument("--downstream-root", type=Path, default=ROOT / "results" / "final" / "downstream")
    ap.add_argument("--composite-root", type=Path, default=ROOT / "results" / "final" / "composite")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results" / "final")
    ap.add_argument("--summary", type=Path, default=ROOT / "results" / "FINAL_RESULTS.md")
    ap.add_argument("--ours", default="trajpe_v4")
    ap.add_argument("--freeze", type=Path, nargs="+", default=[FREEZE], help="plan freeze + amendment freezes")
    ap.add_argument("--no-freeze-check", action="store_true", help="tests only")
    ap.add_argument("--no-protocol-check", action="store_true", help="tests only: skip the bench_tables.main runs")
    ap.add_argument("--allow-incomplete", action="store_true")
    args = ap.parse_args(argv)
    args.commands = [f"{_TP_PY} -I -B scripts/final_tables.py " + " ".join(argv or sys.argv[1:])]
    if not args.no_freeze_check:
        bad = [x for f in args.freeze for x in verify_freeze(f)]
        if bad:
            sys.exit(f"the analysis plan is not frozen or changed: {bad}")
        import eval_bench as EB
        fr = EB.verify_freeze()
        if not fr["frozen"]:
            sys.exit(f"benchmark protocol bench-v1.2.1 changed: {fr['mismatch']}")
    inv = read_inventory(args.inventory)
    D = Data(inv, args.bench_root, args.ours)
    if D.problems:
        sys.exit("bench inputs inconsistent with the inventory:\n" + "\n".join(D.problems[:20]))
    cfg = json.loads((ROOT / "configs" / "datasets_v3.json").read_text())["datasets"]
    P = pools(cfg, inv)
    sets = comparator_sets(inv, args.ours)
    R = {s["key"]: analyze(D, s, sets, inv, P) for s in ANALYSES}
    inc = {k: incomplete_inputs(D, inv, P, sets, eps, b7 is not None) for k, (_, b7, eps) in VERDICTS.items()}
    if inc["primary"] and not args.allow_incomplete:
        sys.exit("incomplete primary inputs (use --allow-incomplete for partial tables): " + str(inc["primary"][:10]))
    prot = {}
    if not args.no_protocol_check:
        for s in ANALYSES:
            prot[s["key"]] = protocol_run(D, s, R[s["key"]], args.out_dir / "protocol")
            if prot[s["key"]]["ok"] is False:
                sys.exit(f"{s['key']}: this script disagrees with bench_tables.py: {prot[s['key']]['diffs'][:10]}")
    verdicts = {k: rule_verdict(R, sets, args.ours, main_key, b7, inc[k])
                for k, (main_key, b7, _) in VERDICTS.items()}
    od = args.out_dir
    od.mkdir(parents=True, exist_ok=True)
    rows = []
    for refname in ("train", "test"):
        for (d, e, m), per in sorted(D.V[refname].items()):
            for meth, v in per.items():
                rows.append({"ref": refname, "dataset": d, "eps": e, "metric": m, "method": meth, "value": v,
                             "direction": D.info[m]["direction"], "role": D.info[m]["role"],
                             "family": D.fam_of.get(m, ""), "source": D.info[m]["source_paper"],
                             "v3_split": D.split[m][0]})
        for (d, m), v in sorted(D.ref[refname].items()):
            rows.append({"ref": refname, "dataset": d, "eps": "", "metric": m, "method": "real_test", "value": v,
                         "direction": D.info[m]["direction"], "role": D.info[m]["role"], "family": D.fam_of.get(m, ""),
                         "source": D.info[m]["source_paper"], "v3_split": D.split[m][0]})
    write_csv(od / "cells.csv", rows)
    fam, met, head, pp, npt, spl, lf, lm, ml, bw, ls = [], [], [], [], [], [], [], [], [], [], []
    for key, r in R.items():
        if r.get("empty"):
            continue
        ref = SPEC[key].get("ref", "train")
        for b, fc in r["_fcs"].items():
            for (d, e, f), (Dv, ms) in sorted(fc.items()):
                fam.append({"analysis": key, "comparator": b, "dataset": d, "eps": e, "family": f, "D": Dv,
                            "outcome": "win" if Dv > BT.TIE else "loss" if Dv < -BT.TIE else "tie",
                            "n_metrics": len(ms), "metrics": " ".join(ms),
                            "imputed": (d, e) in set(map(tuple, r["imputed"].get(b, [])))})
        for b, mc in r["_mcs"].items():
            for (c, dv, oc) in mc:
                met.append({"analysis": key, "comparator": b, "dataset": c[0], "eps": c[1], "metric": c[2], "d_v11": dv,
                            "outcome": oc})
        for b in r["comparators"]:
            head.append({"analysis": key, "comparator": b, **{f"family_{k}": v for k, v in r["family"][b].items()},
                         **{f"metric_{k}": v for k, v in r["metric"][b].items()},
                         "missing_cells": " ".join(f"{d}/eps{e}" for d, e in r["coverage"][b])})
            for p, v in r["per_paper"][b].items():
                pp.append({"analysis": key, "comparator": b, "paper": p, **v})
            for f in NEXTPOI_FAMILIES:
                if b in r["nextpoi"][f]:
                    npt.append({"analysis": key, "comparator": b, "family": f, **r["nextpoi"][f][b]})
            for sk, sv in r["split"].items():
                spl.append({"analysis": key, "comparator": b, "subset": sk,
                            **{f"family_{k}": v for k, v in sv["family"][b].items()},
                            **{f"metric_{k}": v for k, v in sv["metric"][b].items()}})
        a, bb = loss_rows(D, key, r, ref)
        lf += a
        lm += bb
        ml += metric_loss_rows(D, key, r, ref)
        bw += baseline_wins_by_cell(D, key, r)
        ls += loss_summary_by_family(D, key, r)
    write_csv(od / "family_cells.csv", fam)
    write_csv(od / "metric_cells.csv", met)
    write_csv(od / "headline.csv", head)
    write_csv(od / "per_paper.csv", pp)
    write_csv(od / "nextpoi_tests.csv", npt)
    write_csv(od / "split_summaries.csv", spl)
    write_csv(od / "losses.csv", lf)
    write_csv(od / "losses_metrics.csv", lm)
    write_csv(od / "metric_losses.csv", ml)
    write_csv(od / "baseline_wins_by_cell.csv", bw)
    fields = sorted({k for r in ls for k in r}, key=lambda k: (k not in ("analysis", "family", "total_losses"), k))
    write_csv(od / "loss_summary_by_family.csv", ls, fields)
    write_csv(od / "missing.csv", D.missing + [{"dataset": r["dataset"], "method": r["method"], "eps": r["eps"],
                                                "reason": f"inventory status {r['status']}: {r.get('notes', '')}"}
                                               for r in inv if r["status"] != "ok"], ["dataset", "method", "eps", "reason"])
    abl = []
    for key in ("B_primary_ablation", "B_secondary_ablation", "B_lowdata_ablation", "B_primary_ablation_eps28"):
        if not R[key].get("empty"):
            abl += ablation_extras(D, inv, R[key], args.composite_root)
    write_csv(od / "ablation_extras.csv", abl)
    per_dataset_tables(D, inv, sets, od, downstream_reference(inv, args.downstream_root))
    for fname, keys in (("user_level.md", ("U_primary_user", "U_lowdata_user", "U_primary_user_eps28",
                                           "U_lowdata_user_eps28")),
                        ("subset_7b.md", ("A3_primary_7b", "A3_primary_7b_failed_as_loss", "S1_secondary_7b",
                                          "S5_primary_7b_ref_test", "S7_primary_7b_all10")),
                        ("ablation.md", ("B_primary_ablation", "B_secondary_ablation", "B_lowdata_ablation",
                                         "B_primary_ablation_eps28", "J_jakarta_ablation")),
                        ("sensitivity.md", tuple(k for k in R if k.startswith(("A1_primary_", "S", "X", "J_jakarta",
                                                                               "P_")) and k != "J_jakarta_ablation"))):
        md = []
        for key in keys:
            md += headline_md(R[key], key)
        (od / fname).write_text("\n".join(md))
    (od / "sota_verdict.json").write_text(json.dumps({"label": verdicts["primary"]["label"], **verdicts}, indent=1,
                                                     default=str))
    clean = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in R.items()}
    import scipy
    (od / "summary.json").write_text(json.dumps(
        {"ours": args.ours, "verdict": verdicts["primary"], "verdict_secondary_eps28": verdicts["secondary_eps28"],
         "analyses": clean, "pools": P, "comparator_sets": sets, "protocol_check": prot, "missing": D.missing,
         "scipy": scipy.__version__, "inputs": sorted(map(str, D.paths.values()))}, indent=1, default=str))
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(results_md(D, R, verdicts, sets, inv, abl, prot, args))
    print(f"verdict (primary, eps 1/4): {verdicts['primary']['label']}; secondary (eps 2/8): "
          f"{verdicts['secondary_eps28']['label']}; wrote {od} and {args.summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
