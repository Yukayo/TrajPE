from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import sys
import warnings
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy import stats

from trajpe import bench_metrics as bm

REF_METHOD = "real_test"


def flatten(res: dict, ref: str) -> dict:
    out = dict(res.get(f"vs_{ref}", {}))
    acc = defaultdict(list)
    for key, block in (res.get("next_poi") or {}).items():
        model = block.get("model", key.split("_s")[0])
        for scope in ("last", "all"):
            for m, v in (block.get(scope) or {}).items():
                if m.startswith(("acc@", "ndcg@", "mrr")):
                    acc[f"nextpoi_{model}_{m}_{scope}"].append(v)
    out.update({k: float(np.mean(v)) for k, v in acc.items()})
    return out


def loss(v: float, direction: str) -> float:
    return v if direction == "lower" else 1.0 - v


def norm_diff(lb: float, lo: float) -> float:
    s = lb + lo
    return 0.0 if s == 0 else (lb - lo) / s


def holm(pvals: dict) -> dict:
    items = sorted((p, k) for k, p in pvals.items() if np.isfinite(p))
    out, m, run = {k: float("nan") for k in pvals}, len(items), 0.0
    for i, (p, k) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def fmt(x, nd=4):
    if x is None or not np.isfinite(x):
        return "-"
    a = abs(x)
    if a != 0 and (a < 1e-3 or a >= 1e4):
        return f"{x:.2e}"
    return f"{x:.{nd}f}"


def load(paths, ref):
    rows = []
    for p in paths:
        r = json.loads(Path(p).read_text())
        rows.append({"dataset": r["dataset"], "method": r.get("method") or REF_METHOD, "eps": r.get("eps"),
                     "seed": r.get("seed"), "reference": r.get("reference", False), "metrics": flatten(r, ref),
                     "path": str(p), "runtime_s": r.get("runtime_s", {}).get("total")})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True, help="bench JSON files or globs")
    ap.add_argument("--ours", required=True, help="method name of TrajPE in the bench JSONs")
    ap.add_argument("--baselines", nargs="*", default=None, help="default: every other method")
    ap.add_argument("--ref", default="train", choices=["train", "test"], help="reference split block (vs_<ref>)")
    ap.add_argument("--v3-config", type=Path, default=ROOT / "configs" / "v3_large_eps.json")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--families", type=Path, default=None,
                    help="bench-v1.2: statistic-family map (configs/benchmark_protocol_v1.2_families.json); also "
                         "writes tables_families.md, family_cells.csv, summary_families.json (v1.1 outputs unchanged)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="bench-v1.2.1: use only the releases with these seeds (headline: --seeds 0); default: all")
    args = ap.parse_args(argv)

    P = bm.load_protocol()
    if args.families is not None:
        add_descriptors(P, json.loads(args.families.read_text()))
    info = bm.metric_info(P)
    paths = sorted({q for p in args.inputs for q in (glob.glob(p) or [p])})
    rows = load(paths, args.ref)
    if args.seeds is not None:
        rows = [r for r in rows if r["reference"] or r["seed"] in set(args.seeds)]
    if not rows:
        sys.exit("no inputs")
    for r in rows:
        for m in r["metrics"]:
            if m.startswith("nextpoi_") and m not in info:
                info[m] = {"direction": "higher", "role": "unregistered", "family": "downstream",
                           "source_paper": "Massive-STEPS"}
    all_metrics = [m for m in info if any(m in r["metrics"] for r in rows)]
    split = bm.v3_split(json.loads(args.v3_config.read_text()), P, all_metrics)
    methods = sorted({r["method"] for r in rows if not r["reference"]})
    if args.ours not in methods:
        sys.exit(f"--ours {args.ours} not among methods {methods}")
    baselines = args.baselines if args.baselines is not None else [m for m in methods if m != args.ours]
    order = [args.ours] + baselines

    values = defaultdict(lambda: defaultdict(list))
    refvals = {}
    for r in rows:
        for m, v in r["metrics"].items():
            if m not in info or v is None or not np.isfinite(v):
                continue
            if r["reference"]:
                refvals[(r["dataset"], m)] = v
            elif r["method"] in order:
                values[(r["dataset"], r["eps"], m)][r["method"]].append(v)
    cells = sorted(values, key=lambda c: (c[0], c[1], all_metrics.index(c[2])))

    def summary(subset_fn, title):
        lines = [f"### {title}", ""]
        sel = [c for c in cells if info[c[2]]["role"] == "primary" and subset_fn(c[2])]
        res = {"n_cells": len(sel), "baselines": {}}
        lines.append(f"Cells: {len(sel)} (dataset x eps x primary metric). d > 0 means {args.ours} is better.")
        lines.append("")
        lines.append("Welch tests need >= 2 seeds on both sides; other cells are 'untestable' and counted by mean only.")
        lines.append("")
        lines.append("| baseline | cells | wins | ties | losses | sig. wins | sig. ties | sig. losses | untestable | "
                     "median d | Wilcoxon p | sign p | Holm Wilcoxon | Holm sign |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        pw, ps = {}, {}
        for b in baselines:
            w = t = l_ = sw = st = sl = nt = 0
            ds = []
            for c in sel:
                vo, vb = values[c].get(args.ours), values[c].get(b)
                if not vo or not vb:
                    continue
                d = info[c[2]]["direction"]
                mo, mb = float(np.mean(vo)), float(np.mean(vb))
                lo, lb = loss(mo, d), loss(mb, d)
                if abs(lo - lb) <= 1e-12 * max(1.0, abs(lo), abs(lb)):
                    t += 1
                elif lo < lb:
                    w += 1
                else:
                    l_ += 1
                if len(vo) < 2 or len(vb) < 2:
                    nt += 1
                else:
                    with np.errstate(all="ignore"), warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        p = float(stats.ttest_ind(vo, vb, equal_var=False).pvalue)
                    if np.isfinite(p) and p < args.alpha:
                        sw, sl = (sw + 1, sl) if lo < lb else (sw, sl + 1)
                    else:
                        st += 1
                ds.append(norm_diff(lb, lo))
            ds = np.asarray(ds)
            nz = ds[ds != 0]
            pw[b] = float(stats.wilcoxon(nz).pvalue) if len(nz) >= 2 else float("nan")
            ps[b] = float(stats.binomtest(int((nz > 0).sum()), len(nz), 0.5).pvalue) if len(nz) else float("nan")
            res["baselines"][b] = {"cells": len(ds), "wins": w, "ties": t, "losses": l_, "sig_wins": sw,
                                   "sig_ties": st, "sig_losses": sl, "untestable": nt,
                                   "median_d": float(np.median(ds)) if len(ds) else None,
                                   "wilcoxon_p": pw[b], "sign_p": ps[b]}
        hw, hs = holm(pw), holm(ps)
        for b in baselines:
            s = res["baselines"][b]
            s["holm_wilcoxon_p"], s["holm_sign_p"] = hw[b], hs[b]
            lines.append(f"| {b} | {s['cells']} | {s['wins']} | {s['ties']} | {s['losses']} | {s['sig_wins']} | "
                         f"{s['sig_ties']} | {s['sig_losses']} | {s['untestable']} | {fmt(s['median_d'], 3)} | "
                         f"{fmt(s['wilcoxon_p'])} | {fmt(s['sign_p'])} | {fmt(hw[b])} | {fmt(hs[b])} |")
        full = [c for c in sel if all(values[c].get(m) for m in order)]
        lines.append("")
        if len(order) >= 2 and full:
            L = np.array([[loss(float(np.mean(values[c][m])), info[c[2]]["direction"]) for m in order] for c in full])
            ranks = np.vstack([stats.rankdata(row) for row in L])
            avg = ranks.mean(0)
            k, N = len(order), len(full)
            fr = stats.friedmanchisquare(*L.T) if k >= 3 else None
            q = stats.studentized_range.ppf(1 - args.alpha, k, np.inf) / math.sqrt(2) if k >= 2 else float("nan")
            cd = q * math.sqrt(k * (k + 1) / (6.0 * N))
            res["ranks"] = {"methods": order, "avg_rank": avg.tolist(), "n_cells": N,
                            "friedman_chi2": float(fr.statistic) if fr else None,
                            "friedman_p": float(fr.pvalue) if fr else None, "nemenyi_cd": cd}
            lines.append(f"Average rank over {N} complete cells (1 = best; Nemenyi CD at alpha {args.alpha} = {cd:.3f}"
                         + (f"; Friedman chi2 = {fr.statistic:.2f}, p = {fmt(fr.pvalue)}" if fr else "") + "):")
            lines.append("")
            lines.append("| " + " | ".join(order) + " |")
            lines.append("|" + "---|" * len(order))
            lines.append("| " + " | ".join(f"{a:.2f}" for a in avg) + " |")
            lines.append("")
            if fr is not None:
                worse = [m for m, a in zip(order, avg) if a - avg[0] > cd]
                better = [m for m, a in zip(order, avg) if avg[0] - a > cd]
                lines.append(f"Nemenyi: {args.ours} significantly better than {worse or 'none'}; "
                             f"significantly worse than {better or 'none'}.")
                lines.append("")
        return lines, res

    out = [f"# Benchmark protocol {P['version']} tables", "",
           f"Reference: real {args.ref} split. TrajPE = `{args.ours}`; baselines: {', '.join(baselines)}. "
           f"Cells show mean +- sd over seeds (n); best mean per row in bold (lower is better unless marked "
           f"higher); 'real test' is the sampling reference at the test split's size (not a lower bound). "
           f"v3 split from `{args.v3_config.name}`: M = measured, N = near, U = unmeasured.", ""]
    csv_rows = []
    for ds_eps in sorted({(c[0], c[1]) for c in cells}, key=lambda x: (x[0], x[1])):
        ds, eps = ds_eps
        out += [f"## {ds}, eps = {eps:g}", "", "| metric | dir | v3 | " + " | ".join(order) + " | real test |",
                "|---|---|---|" + "---|" * (len(order) + 1)]
        for c in [c for c in cells if (c[0], c[1]) == ds_eps]:
            m = c[2]
            d = info[m]["direction"]
            means = {meth: float(np.mean(values[c][meth])) for meth in order if values[c].get(meth)}
            bl = min(loss(v, d) for v in means.values()) if means else None
            cols = []
            for meth in order:
                v = values[c].get(meth)
                if not v:
                    cols.append("-")
                    continue
                s = (f"{fmt(np.mean(v))} +- {fmt(np.std(v, ddof=1))} ({len(v)})" if len(v) > 1
                     else f"{fmt(v[0])} (1)")
                best = abs(loss(means[meth], d) - bl) <= 1e-12 * max(1.0, abs(bl))
                cols.append(f"**{s}**" if best else s)
                csv_rows.append({"dataset": ds, "eps": eps, "metric": m, "method": meth, "n": len(v),
                                 "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                                 "values": json.dumps(v), "direction": d, "role": info[m]["role"],
                                 "source": info[m]["source_paper"], "v3_split": split[m][0]})
            tag = {"primary": "", "variant": " (variant)", "unregistered": " (unregistered model)"}.get(
                info[m]["role"], f" (= {info[m].get('duplicate_of')})")
            out.append(f"| {m}{tag} | {'higher' if d == 'higher' else 'lower'} | {split[m][0][0].upper()} | "
                       + " | ".join(cols) + f" | {fmt(refvals.get((ds, m)))} |")
        out.append("")

    summ = {}
    out += ["## Summary over all primary metrics", ""]
    lines, summ["all"] = summary(lambda m: True, "All primary metrics")
    out += lines
    out += ["## Metrics TrajPE v3 does not measure directly", "",
            "Split by the pre-registered rule (configs/benchmark_protocol_v1.json, v3_split): a metric is *measured* "
            "when a DP histogram that v3 releases covers its unit and variables; *near* when it does after relaxing "
            "the unit (grid points, coverage, stays) or as a joint of several measured variables; otherwise "
            "*unmeasured*.", ""]
    lines, summ["not_measured"] = summary(lambda m: split[m][0] != "measured", "Near + unmeasured")
    out += lines
    lines, summ["unmeasured"] = summary(lambda m: split[m][0] == "unmeasured", "Unmeasured only")
    out += lines
    lines, summ["measured"] = summary(lambda m: split[m][0] == "measured", "Measured (for comparison)")
    out += lines
    out += ["## v3 split of the metrics", "", "| metric | role | split | reason |", "|---|---|---|---|"]
    for m in all_metrics:
        out.append(f"| {m} | {info[m]['role']} | {split[m][0]} | {split[m][1]} |")
    out.append("")
    rt = [r["runtime_s"] for r in rows if r["runtime_s"] is not None]
    out += [f"Inputs: {len(rows)} files; eval_bench runtime per file: median {np.median(rt):.1f} s, max {max(rt):.1f} s."
            if rt else f"Inputs: {len(rows)} files.", ""]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "tables.md").write_text("\n".join(out))
    with open(args.out_dir / "cells.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0]) if csv_rows else ["dataset"])
        w.writeheader()
        w.writerows(csv_rows)
    (args.out_dir / "summary.json").write_text(json.dumps(
        {"protocol": P["version"], "reference": args.ref, "ours": args.ours, "baselines": baselines,
         "v3_config": str(args.v3_config), "split": {m: list(split[m]) for m in all_metrics}, "summary": summ,
         "inputs": paths}, indent=1, default=float))
    print(f"wrote {args.out_dir / 'tables.md'} ({len(cells)} cells, {len(rows)} inputs)")
    if args.families is not None:
        family_report(args, P, info, values, cells, split, order, baselines, paths, summ)
    return 0


TIE = 1e-12


def add_descriptors(P: dict, F: dict) -> None:
    from trajpe.workload import FAMILIES
    fam = P["v3_split"]["families"]
    for f, d in (F.get("v3_split_descriptors") or {}).items():
        if f == "rule":
            continue
        if f in fam:
            sys.exit(f"--families: v3_split_descriptors: {f} is already described by the protocol")
        if f not in FAMILIES:
            sys.exit(f"--families: v3_split_descriptors: {f} is not a workload family")
        if set(d) - {"unit", "vars", "restricted", "source"} or not isinstance(d.get("vars"), list) or "unit" not in d:
            sys.exit(f"--families: v3_split_descriptors: malformed descriptor for {f}")
        fam[f] = {k: d[k] for k in ("unit", "vars", "restricted") if k in d}


def natural_diff(vo: float, vb: float, direction: str) -> float:
    s = abs(vo) + abs(vb)
    if s == 0:
        return 0.0
    return (vo - vb) / s if direction == "higher" else (vb - vo) / s


def load_families(path: Path, info: dict):
    F = json.loads(Path(path).read_text())
    fam_of = {}
    for f, spec in F["families"].items():
        for m in spec["metrics"]:
            if m not in info:
                sys.exit(f"--families: family {f}: unknown metric {m}")
            if info[m]["role"] != "primary":
                sys.exit(f"--families: family {f}: {m} is {info[m]['role']}, not primary")
            if m in fam_of:
                sys.exit(f"--families: {m} in both {fam_of[m]} and {f}")
            fam_of[m] = f
    missing = [m for m, i in info.items() if i["role"] == "primary" and m not in fam_of]
    if missing:
        sys.exit(f"--families: primary metrics without a family: {missing}")
    suites = {}
    for p, ms in F["paper_suites"].items():
        if p == "rule":
            continue
        suites[p] = [x for m in ms for x in (F["families"][m[7:]]["metrics"] if m.startswith("family:") else [m])]
    derived = defaultdict(set)
    for m, i in info.items():
        if i["role"] == "primary":
            derived[i["source_paper"]].add(m)
    if {p: set(v) for p, v in suites.items()} != dict(derived):
        sys.exit("--families: paper_suites differ from the source_paper field of the protocol")
    return F, fam_of, suites


def _tests(D):
    D = np.asarray(D, float)
    nz = D[np.abs(D) > TIE]
    pw = float(stats.wilcoxon(nz).pvalue) if len(nz) >= 2 else float("nan")
    ps = float(stats.binomtest(int((nz > 0).sum()), len(nz), 0.5).pvalue) if len(nz) else float("nan")
    return pw, ps


def _wtl(D):
    D = np.asarray(D, float)
    return int((D > TIE).sum()), int((np.abs(D) <= TIE).sum()), int((D < -TIE).sum())


def _verdict(p_holm, D, alpha):
    D = np.asarray(D, float)
    nz = D[np.abs(D) > TIE]
    med = float(np.median(nz)) if len(nz) else 0.0
    if np.isfinite(p_holm) and p_holm < alpha and med != 0:
        return "win" if med > 0 else "loss"
    w, _, l_ = _wtl(D)
    return "n.s. (" + ("more wins" if w > l_ else "more losses" if l_ > w else "even") + ")"


def family_report(args, P, info, values, cells, split, order, baselines, paths, summ_v11=None):
    F, fam_of, suites = load_families(args.families, info)
    fams = list(F["families"])
    ours, alpha = args.ours, args.alpha
    rule = F.get("metric_difference_rule", "v1.1_d")
    if rule not in ("v1.1_d", "natural_symmetric"):
        sys.exit(f"--families: unknown metric_difference_rule {rule}")
    version = F.get("version", "bench-v1.2")

    def mdiff(vo, vb, d):
        if rule == "natural_symmetric":
            return natural_diff(vo, vb, d)
        return norm_diff(loss(vb, d), loss(vo, d))

    def fam_cells(b, keep):
        acc = defaultdict(list)
        for c in cells:
            m = c[2]
            if info[m]["role"] != "primary" or not keep(m):
                continue
            vo, vb = values[c].get(ours), values[c].get(b)
            if not vo or not vb:
                continue
            d = info[m]["direction"]
            acc[(c[0], c[1], fam_of[m])].append((m, mdiff(float(np.mean(vo)), float(np.mean(vb)), d)))
        return {k: (float(np.mean([x[1] for x in v])), [x[0] for x in v]) for k, v in acc.items()}

    csv_rows = []

    def summary(keep, title, key):
        lines = [f"### {title}", "", f"Family cells (dataset x eps x family); D = mean normalized difference d over "
                 f"the family's metrics, D > 0 means {ours} is better. Verdict: Holm-Wilcoxon p < {alpha} with the "
                 f"sign of the median of the non-tied D.", "",
                 "| baseline | family cells | wins | ties | losses | median D | Wilcoxon p | sign p | Holm Wilcoxon | "
                 "Holm sign | verdict |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        res, pw, ps, Ds = {"baselines": {}}, {}, {}, {}
        for b in baselines:
            fc = fam_cells(b, keep)
            Ds[b] = [fc[k][0] for k in sorted(fc)]
            pw[b], ps[b] = _tests(Ds[b])
            for k in sorted(fc):
                csv_rows.append({"summary": key, "baseline": b, "dataset": k[0], "eps": k[1], "family": k[2],
                                 "D": fc[k][0], "n_metrics": len(fc[k][1]), "metrics": " ".join(fc[k][1])})
            res["baselines"][b] = {"family_cells": len(Ds[b])}
        hw, hs = holm(pw), holm(ps)
        for b in baselines:
            w, t, l_ = _wtl(Ds[b])
            med = float(np.median(Ds[b])) if Ds[b] else None
            v = _verdict(hw[b], Ds[b], alpha)
            res["baselines"][b].update({"wins": w, "ties": t, "losses": l_, "median_D": med, "wilcoxon_p": pw[b],
                                        "sign_p": ps[b], "holm_wilcoxon_p": hw[b], "holm_sign_p": hs[b], "verdict": v})
            lines.append(f"| {b} | {len(Ds[b])} | {w} | {t} | {l_} | {fmt(med, 3) if med is not None else '-'} | "
                         f"{fmt(pw[b])} | {fmt(ps[b])} | {fmt(hw[b])} | {fmt(hs[b])} | {v} |")
        lines.append("")
        acc = defaultdict(list)
        for c in cells:
            m = c[2]
            if info[m]["role"] != "primary" or not keep(m) or not all(values[c].get(x) for x in order):
                continue
            L = [loss(float(np.mean(values[c][x])), info[m]["direction"]) for x in order]
            acc[(c[0], c[1], fam_of[m])].append(stats.rankdata(L))
        if len(order) >= 2 and acc:
            R = np.vstack([stats.rankdata(np.mean(acc[k], axis=0)) for k in sorted(acc)])
            avg, k_, N = R.mean(0), len(order), len(acc)
            fr = stats.friedmanchisquare(*R.T) if k_ >= 3 else None
            cd = stats.studentized_range.ppf(1 - alpha, k_, np.inf) / math.sqrt(2) * math.sqrt(k_ * (k_ + 1) / (6.0 * N))
            res["ranks"] = {"methods": order, "avg_rank": avg.tolist(), "n_family_cells": N,
                            "friedman_chi2": float(fr.statistic) if fr else None,
                            "friedman_p": float(fr.pvalue) if fr else None, "nemenyi_cd": cd}
            lines += [f"Average family rank over {N} complete family cells (1 = best; Nemenyi CD = {cd:.3f}"
                      + (f"; Friedman chi2 = {fr.statistic:.2f}, p = {fmt(fr.pvalue)}" if fr else "") + "):", "",
                      "| " + " | ".join(order) + " |", "|" + "---|" * len(order),
                      "| " + " | ".join(f"{a:.2f}" for a in avg) + " |", ""]
        return lines, res

    out = [f"# Benchmark protocol {version} family tables ({args.families.name})", "",
           f"Metric protocol {P['version']} (unchanged); TrajPE = `{ours}`; baselines: {', '.join(baselines)}. "
           f"Each family counts once per dataset x eps. Reference: real {args.ref} split.", ""]
    if rule == "natural_symmetric":
        out += [f"Metric difference d = (v_ours - v_base) / (|v_ours| + |v_base|) for higher-is-better and "
                f"(v_base - v_ours) / (|v_base| + |v_ours|) for lower-is-better metrics (natural scale, {version}). "
                + (f"Releases: seeds {args.seeds} only." if args.seeds is not None else
                   "Releases: all seeds (seed means)."), ""]
        if summ_v11 is not None:
            g = summ_v11["all"]
            out += ["## Metric-level headline (bench-v1.1 tables, reported with equal prominence)", "",
                    f"{g['n_cells']} cells (dataset x eps x primary metric), v1.1 d on the loss scale; per-cell "
                    "Welch tests need >= 2 seeds on both sides (untestable with one release per method).", "",
                    "| baseline | cells | wins | ties | losses | untestable | median d | Wilcoxon p | sign p | "
                    "Holm Wilcoxon | Holm sign |", "|---|---|---|---|---|---|---|---|---|---|---|"]
            for b in baselines:
                s = g["baselines"][b]
                out.append(f"| {b} | {s['cells']} | {s['wins']} | {s['ties']} | {s['losses']} | {s['untestable']} | "
                           f"{fmt(s['median_d'], 3)} | {fmt(s['wilcoxon_p'])} | {fmt(s['sign_p'])} | "
                           f"{fmt(s['holm_wilcoxon_p'])} | {fmt(s['holm_sign_p'])} |")
            out.append("")
    out += ["## Families", "", "| family | kind | primary metrics |", "|---|---|---|"]
    for f in fams:
        out.append(f"| {f} | {F['families'][f]['kind']} | {', '.join(F['families'][f]['metrics'])} |")
    out.append("")
    summ = {}
    for key, keep, title in [
            ("all", lambda m: True, "Headline: all families"),
            ("not_measured", lambda m: split[m][0] != "measured", "Near + unmeasured metrics (v3 split), by family"),
            ("unmeasured", lambda m: split[m][0] == "unmeasured", "Unmeasured metrics only, by family"),
            ("measured", lambda m: split[m][0] == "measured", "Measured metrics (for comparison), by family")]:
        lines, summ[key] = summary(lambda m, keep=keep: m in split and keep(m), title, key)
        out += lines

    out += ["## Per family (all datasets x eps)", "", "Entry: wins-ties-losses over dataset x eps (mean D).", "",
            "| family | " + " | ".join(baselines) + " |", "|---|" + "---|" * len(baselines)]
    per_b = {b: fam_cells(b, lambda m: True) for b in baselines}
    summ["per_family"] = {}
    for f in fams:
        row = []
        for b in baselines:
            D = [v[0] for k, v in per_b[b].items() if k[2] == f]
            w, t, l_ = _wtl(D)
            row.append(f"{w}-{t}-{l_} ({fmt(float(np.mean(D)), 3)})" if D else "-")
            summ["per_family"].setdefault(f, {})[b] = {"wins": w, "ties": t, "losses": l_,
                                                       "mean_D": float(np.mean(D)) if D else None}
        out.append(f"| {f} | " + " | ".join(row) + " |")
    out.append("")

    out += ["## Per source paper (each paper's own primary metrics, grouped into families)", "",
            "Entry: verdict; family cells W-T-L; median D; Holm-Wilcoxon p (Holm over the papers within each "
            "baseline); paper cells (dataset x eps, mean of the suite's family D) W-T-L.", "",
            "| paper | " + " | ".join(baselines) + " |", "|---|" + "---|" * len(baselines)]
    summ["per_paper"] = {}
    cellmap = {}
    for b in baselines:
        pw, ps, Ds, PC = {}, {}, {}, {}
        for p, ms in suites.items():
            S = set(ms)
            fc = fam_cells(b, lambda m, S=S: m in S)
            Ds[p] = [fc[k][0] for k in sorted(fc)]
            pw[p], ps[p] = _tests(Ds[p])
            byc = defaultdict(list)
            for k, v in fc.items():
                byc[(k[0], k[1])].append(v[0])
            PC[p] = [float(np.mean(v)) for v in byc.values()]
        hw, hs = holm(pw), holm(ps)
        for p in suites:
            med = float(np.median(Ds[p])) if Ds[p] else None
            v = _verdict(hw[p], Ds[p], alpha)
            w, t, l_ = _wtl(Ds[p])
            pc = _wtl(PC[p])
            summ["per_paper"].setdefault(p, {})[b] = {
                "verdict": v, "family_cells": len(Ds[p]), "wins": w, "ties": t, "losses": l_, "median_D": med,
                "wilcoxon_p": pw[p], "sign_p": ps[p], "holm_wilcoxon_p": hw[p], "holm_sign_p": hs[p],
                "paper_cells": len(PC[p]), "paper_cell_wins": pc[0], "paper_cell_ties": pc[1],
                "paper_cell_losses": pc[2]}
            cellmap[(p, b)] = (f"{v}; {w}-{t}-{l_}; {fmt(med, 3) if med is not None else '-'}; {fmt(hw[p])}; "
                               f"{pc[0]}-{pc[1]}-{pc[2]}") if Ds[p] else "-"
    for p in suites:
        out.append(f"| {p} | " + " | ".join(cellmap[(p, b)] for b in baselines) + " |")
    out.append("")

    (args.out_dir / "tables_families.md").write_text("\n".join(out))
    with open(args.out_dir / "family_cells.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["summary", "baseline", "dataset", "eps", "family", "D", "n_metrics",
                                          "metrics"])
        w.writeheader()
        w.writerows(csv_rows)
    (args.out_dir / "summary_families.json").write_text(json.dumps(
        {"protocol": version, "metric_protocol": P["version"], "families_config": str(args.families),
         "reference": args.ref, "ours": ours, "baselines": baselines, "family_of": fam_of, "summary": summ,
         "inputs": paths, **({"metric_difference_rule": rule, "seeds": args.seeds,
                              "metric_level_v1_1": (summ_v11 or {}).get("all")} if rule != "v1.1_d" else {})},
        indent=1, default=float))
    print(f"wrote {args.out_dir / 'tables_families.md'} ({len(fams)} families)")


if __name__ == "__main__":
    sys.exit(main())
