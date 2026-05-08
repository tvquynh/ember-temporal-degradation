"""Generate Figure showing distribution of `appeared` timestamps per dataset.

Addresses reviewer concern about cross-era validity: we show that EMBER 2017
and EMBER 2018 release samples are predominantly from their respective release
years (~92% each), supporting the use of dataset release as a proxy for era.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIG_DIR = HERE / "manuscript_pass_a" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "text.usetex": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})


def main() -> None:
    df = pd.read_csv(RESULTS / "temporal_dist.csv")
    df["dataset"] = df["dataset"].str.replace("_", " ")
    pivot = df.pivot_table(index="year", columns="dataset", values="n",
                           aggfunc="sum", fill_value=0).sort_index()
    # Restrict to 2009-2020 for readability (everything before 2009 < 0.5%).
    pivot = pivot.loc[(pivot.index >= 2009) & (pivot.index <= 2020)]
    years = pivot.index.tolist()
    cols = ["EMBER 2017", "EMBER 2018", "BODMAS"]

    # Stacked bar chart with monochrome shading.
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    width = 0.27
    x = np.arange(len(years))
    fills = ["#D0D0D0", "#7A7A7A", "#2E2E2E"]
    edges = ["black", "black", "black"]
    for i, c in enumerate(cols):
        if c in pivot.columns:
            ax.bar(x + (i - 1) * width, pivot[c].values, width=width,
                   color=fills[i], edgecolor=edges[i], linewidth=0.5,
                   label=c)
    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=0)
    ax.set_xlabel("appeared year")
    ax.set_ylabel("samples (labelled)")
    ax.set_yscale("log")
    ax.set_title("Temporal coverage by appeared timestamp")
    ax.legend(loc="upper left", ncol=3, framealpha=0.95)
    fig.tight_layout()
    out = FIG_DIR / "fig_temporal_coverage.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    # Print summary stats for paper text.
    for c in cols:
        if c in pivot.columns:
            total = pivot[c].sum()
            year_match = {"EMBER 2017": 2017, "EMBER 2018": 2018,
                          "BODMAS": 2020}.get(c)
            if year_match and year_match in pivot.index:
                in_year = pivot.loc[year_match, c]
                print(f"  {c}: total={total:,}, "
                      f"appeared={year_match} -> {in_year:,} ({100*in_year/total:.1f}%)")


if __name__ == "__main__":
    main()
