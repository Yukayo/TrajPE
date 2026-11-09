import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from trajpe.metrics import composite_score
from trajpe.paths import RESULTS, ROOT

TAB = ROOT / "paper" / "latex" / "tables"
CITIES = ["istanbul", "tokyo", "sao_paulo", "kuwait_city", "nyc"]
CITY_TEX = {"istanbul": "IST", "tokyo": "TKY", "sao_paulo": "SAO", "kuwait_city": "KWI", "nyc": "NYC$^\\dagger$"}
MAIN = ["TrajPE", "TrajPE-sim", "Naive PE", "DP-LoRA", "DP-LoRA + DP venue prior", "PrivTrace", "AdaTrace", "LDPTrace (LDP)"]
ABL = [("nomarg", "no marginals"), ("prompt", "+ prompt (MAPLE-style)"), ("prompt_ground", "+ grounding"),
       ("all_stages", "+ variation"), ("venue_aware", "+ venue-aware voting"), ("full", "+ DP transitions")]


def bold_best(vals: dict, lower=True):
    ok = {k: v for k, v in vals.items() if v == v}
    if not ok:
        return set()
    best = min(ok.values()) if lower else max(ok.values())
    return {k for k, v in ok.items() if v == best}


def tab_main(df):
    lines = ["\\begin{tabular}{l" + "r" * (2 * len(CITIES)) + "}", "\\toprule",
             " & " + " & ".join(f"\\multicolumn{{{len(CITIES)}}}{{c}}{{$\\varepsilon={e}$}}" for e in (1, 4)) + " \\\\",
             "Method & " + " & ".join(CITY_TEX[c] for _ in (1, 4) for c in CITIES) + " \\\\", "\\midrule"]
    g = df[df.unit == "trail"]
    cells = {}
    for e in (1, 4):
        for c in CITIES:
            s = g[(g.eps == e) & (g.city == c)].groupby("method")["composite"]
            mean, std = s.mean(), s.std(ddof=0)
            best = bold_best({m: v for m, v in mean.to_dict().items() if m in MAIN})
            for m in MAIN:
                if m in mean:
                    t = f"{mean[m]:.2f}" + (f"\\scriptsize{{$\\pm${std[m]:.2f}}}" if s.count()[m] > 1 else "")
                    cells[(m, e, c)] = f"\\textbf{{{t}}}" if m in best else t
    for m in MAIN:
        if any((m, e, c) in cells for e in (1, 4) for c in CITIES):
            lines.append(m + " & " + " & ".join(cells.get((m, e, c), "--") for e in (1, 4) for c in CITIES) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "tab_main.tex").write_text("\n".join(lines) + "\n")


def tab_downstream(df):
    g = df[df.unit == "trail"]
    if "ds_mrr" not in g:
        return
    tf_col = "ds_mrr_tavg" if "ds_mrr_tavg" in g else "ds_mrr"
    cols = [(tf_col, "TF")] + ([("ds_mk_mrr", "MK")] if "ds_mk_mrr" in g else [])
    lines = ["\\begin{tabular}{ll" + "r" * (len(CITIES) * len(cols)) + "}", "\\toprule",
             " & & " + " & ".join(f"\\multicolumn{{{len(cols)}}}{{c}}{{{CITY_TEX[c]}}}" for c in CITIES) + " \\\\",
             "$\\varepsilon$ & Method & " + " & ".join(n for _ in CITIES for _, n in cols) + " \\\\", "\\midrule"]
    for e in (1, 4):
        sub = g[g.eps == e]
        means = {col: sub.groupby(["method", "city"])[col].mean() for col, _ in cols}
        best = {(col, c): bold_best({m: means[col].get((m, c), float("nan")) for m in MAIN}, lower=False)
                for col, _ in cols for c in CITIES}
        for m in MAIN:
            vals = []
            for c in CITIES:
                for col, _ in cols:
                    v = means[col].get((m, c), float("nan"))
                    vals.append("--" if v != v else ("\\textbf{%.3f}" if m in best[(col, c)] else "%.3f") % v)
            if any(v != "--" for v in vals):
                lines.append(f"{e} & {m} & " + " & ".join(vals) + " \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    (TAB / "tab_downstream.tex").write_text("\n".join(lines) + "\n")


def tab_ablation():
    lines = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
             " & \\multicolumn{3}{c}{NYC$^\\dagger$} & \\multicolumn{3}{c}{Istanbul} \\\\",
             "Variant & comp. & cat. & MRR & comp. & cat. & MRR \\\\", "\\midrule"]
    for key, name in ABL:
        row = [name]
        for city in ("nyc", "istanbul"):
            f = RESULTS / "pe" / city / f"A_sim_eps4_{key}.json"
            if not f.exists():
                row += ["--"] * 3
                continue
            r = json.loads(f.read_text())["vs_train"]
            d = RESULTS / "downstream" / city / f"A_sim_eps4_{key}_transformer.json"
            mrr = f"{json.loads(d.read_text())['metrics']['mrr']:.3f}" if d.exists() else "--"
            row += [f"{composite_score(r):.3f}", f"{r['cat_jsd']:.3f}", mrr]
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "tab_ablation.tex").write_text("\n".join(lines) + "\n")


def tab_openloop():
    rows = [("real", "real data"), ("llm_raw", "LLM prior (no DP input)"), ("open_loop", "open loop (DP prompts)"),
            ("release", "closed loop (\\sys{})")]
    lines = ["\\begin{tabular}{llrrrrr}", "\\toprule",
             "City & Trails & gaps $\\equiv 0$ (15') & gaps $<$10' & food & transit & cat.\\ JSD \\\\", "\\midrule"]
    for city, name in (("nyc", "NYC$^\\dagger$"), ("tokyo", "Tokyo")):
        f = RESULTS / "stereotypes" / f"{city}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        if not all(k in d and "short_gap" in d[k] for k, _ in rows):
            continue
        roots = d["roots"]
        food, transit = roots.index("Dining and Drinking"), roots.index("Travel and Transportation")
        for i, (k, label) in enumerate(rows):
            v = d[k]
            cat = "--" if k == "real" else f"{v['cat_jsd_to_real']:.3f}"
            lines.append(f"{name if i == 0 else ''} & {label} & {100 * v['gap_mult15']:.0f}\\% & "
                         f"{100 * v['short_gap']:.0f}\\% & {100 * v['root_share'][food]:.0f}\\% & "
                         f"{100 * v['root_share'][transit]:.0f}\\% & {cat} \\\\")
        lines.append("\\midrule")
    if len(lines) == 4:
        return
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    (TAB / "tab_openloop.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    TAB.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RESULTS / "tables" / "long.csv")
    tab_main(df)
    tab_downstream(df)
    tab_ablation()
    tab_openloop()
    print("tables written to", TAB)
