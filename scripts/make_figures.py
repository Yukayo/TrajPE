import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from trajpe.paths import RESULTS, ROOT, RUNS

FIG = ROOT / "paper" / "latex" / "figures"
EVAL = ["istanbul", "tokyo", "sao_paulo", "kuwait_city"]
TITLES = {"istanbul": "Istanbul", "tokyo": "Tokyo", "sao_paulo": "São Paulo", "kuwait_city": "Kuwait City",
          "nyc": "New York"}
STYLE = {"TrajPE": ("C0", "o", "-"), "TrajPE-sim": ("C0", "s", "--"), "PrivTrace": ("C1", "^", "-"),
         "AdaTrace": ("C2", "v", "-"), "LDPTrace (LDP)": ("C7", "x", ":"), "Naive PE": ("C4", "d", "-"),
         "DP-LoRA": ("C3", "P", "-")}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 7, "figure.dpi": 150})


def fig_eps():
    df = pd.read_csv(RESULTS / "tables" / "long.csv")
    df = df[df.unit == "trail"]
    fig, axes = plt.subplots(1, len(EVAL), figsize=(7.0, 1.9), sharey=False)
    for ax, city in zip(axes, EVAL):
        g = df[df.city == city]
        for m, (c, mk, ls) in STYLE.items():
            s = g[g.method == m].groupby("eps")["composite"].agg(["mean", "std"]).sort_index()
            if len(s):
                ax.errorbar(s.index, s["mean"], yerr=s["std"].fillna(0), color=c, marker=mk, ls=ls, ms=3, lw=1,
                            capsize=1.5, label=m)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
        ax.set_title(TITLES[city])
        ax.set_xlabel(r"$\varepsilon$")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("composite (lower is better)")
    h, lab = axes[0].get_legend_handles_labels()
    if h:
        fig.legend(h, lab, loc="upper center", ncol=len(lab), bbox_to_anchor=(0.5, 1.08), frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_eps.pdf", bbox_inches="tight")


def fig_epochs(epochs=(1, 2, 4, 8, 16), cities=EVAL, folder="epoch_curve_v21"):
    curves = {"TrajPE": "hybriddiv4", "TrajPE w/o div.": "hybrid4", "PrivTrace": "privtrace4", "AdaTrace": "adatrace4"}
    style = dict(STYLE, **{"TrajPE w/o div.": ("C0", "o", ":")})
    fig, axes = plt.subplots(1, len(cities), figsize=(7.0, 1.75))
    for ax, city in zip(axes, cities):
        for label, key in curves.items():
            xs, ys = [], []
            for e in epochs:
                f = RUNS / "downstream" / folder / f"{city}_{key}_e{e}.json"
                if f.exists():
                    xs.append(e)
                    ys.append(json.loads(f.read_text())["metrics"]["mrr"])
            if xs:
                c, mk, ls = style.get(label, ("k", ".", "-"))
                ax.plot(xs, ys, color=c, marker=mk, ls=ls, ms=3, lw=1, label=label)
        ax.set_xscale("log", base=2)
        ax.set_xticks(list(epochs), [str(e) for e in epochs])
        ax.set_xlabel("training epochs")
        ax.set_title(TITLES[city])
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("next-POI MRR")
    h, lab = axes[0].get_legend_handles_labels()
    if h:
        fig.legend(h, lab, loc="upper center", ncol=len(lab), bbox_to_anchor=(0.5, 1.1), frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_epochs.pdf", bbox_inches="tight")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    fig_eps()
    fig_epochs()
    print("figures written to", FIG)
