"""Regenerate ALL figures for the IEEE Access resubmission.

Outputs land in `manuscript_pass_a/figures/`. Includes:
  - fig_exp2b_monthly.pdf      (regenerated from Table 5; fixes R3 x-axis typo)
  - fig_exp3_monthly.pdf       (regenerated from Table 6; fixes R3 x-axis typo)
  - fig_exp4_retraining.pdf    (regenerated from Table 7; fixes R3 x-axis typo)
  - fig_exp7_aut_overview.pdf  (NEW, R1 B3)
  - fig_exp8_drift_detector.pdf (NEW, R1 B4+B6: 2-panel)
  - fig_exp9_uncertainty_vs_random.pdf (NEW, R1 B7)

All figures use a consistent style (no seaborn; matplotlib only) and are saved
as vector PDFs sized for IEEE Access two-column layout.
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

# IEEE single-column ~ 3.5 in wide; two-column figure ~ 7.16 in wide.
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "lines.linewidth": 1.2,
})

# ---------------------------------------------------------------------------
# Data sources duplicated from manuscript Tables 5/6/7 for figure regeneration.
# ---------------------------------------------------------------------------

TABLE5_MONTHS = [
    "2019-08", "2019-09", "2019-10", "2019-11", "2019-12",
    "2020-01", "2020-02", "2020-03", "2020-04", "2020-05",
    "2020-06", "2020-07", "2020-08", "2020-09",
]
TABLE5_TRAINED_2017 = [
    95.59, 98.71, 98.68, 98.70, 99.05,
    98.26, 97.68, 98.31, 98.52, 99.08,
    98.53, 99.14, 99.55, 99.23,
]
TABLE5_TRAINED_2018 = [
    96.42, 98.65, 98.68, 98.35, 98.38,
    98.62, 98.06, 98.05, 98.96, 98.93,
    98.65, 98.85, 99.36, 99.31,
]

TABLE6_MONTHS = [
    "2019-09", "2019-10", "2019-11", "2019-12",
    "2020-01", "2020-02", "2020-03", "2020-04",
    "2020-05", "2020-06", "2020-07", "2020-08", "2020-09",
]
TABLE6_F1 = [
    97.70, 97.46, 97.90, 98.53,
    96.30, 95.44, 97.83, 99.15,
    97.80, 98.03, 98.18, 98.12, 97.47,
]

TABLE7_MONTHS = TABLE6_MONTHS
TABLE7_BASE = TABLE6_F1
TABLE7_INCREMENTAL = [
    97.68, 97.78, 98.31, 98.76,
    96.67, 95.58, 98.46, 99.16,
    98.51, 99.05, 99.01, 99.30, 99.27,
]
TABLE7_PVAL = [
    0.1523, 0.0020, 0.0020, 0.0020,
    0.0020, 0.0840, 0.0020, 0.7695,
    0.0020, 0.0020, 0.0020, 0.0020, 0.0020,
]

# ---------------------------------------------------------------------------
# Helper: format month axis cleanly with rotated tick labels.
# ---------------------------------------------------------------------------

def _set_month_axis(ax, months: list[str]) -> None:
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(months, rotation=45, ha="right")
    ax.set_xlabel("Month (YYYY-MM)")


def fig_exp2b_monthly() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    x = range(len(TABLE5_MONTHS))
    ax.plot(x, TABLE5_TRAINED_2017, marker="o", label="Train: EMBER 2017")
    ax.plot(x, TABLE5_TRAINED_2018, marker="s", label="Train: EMBER 2018")
    ax.set_ylim(94, 100)
    ax.set_ylabel("F1 (%)")
    _set_month_axis(ax, TABLE5_MONTHS)
    ax.set_title("Exp 2b: EMBER -> BODMAS monthly F1 (LightGBM)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = FIG_DIR / "fig_exp2b_monthly.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_exp3_monthly() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    x = range(len(TABLE6_MONTHS))
    ax.plot(x, TABLE6_F1, marker="o", color="C2")
    ax.set_ylim(94, 100)
    ax.set_ylabel("F1 (%)")
    _set_month_axis(ax, TABLE6_MONTHS)
    ax.set_title("Exp 3: BODMAS internal monthly drift (LightGBM, train 2019-08)")
    # Annotate dip area (Jan-Feb 2020 FPR spike). One label, one arrow,
    # placed clearly below the dip so the two months are covered together.
    i_jan = TABLE6_MONTHS.index("2020-01")
    i_feb = TABLE6_MONTHS.index("2020-02")
    midpoint_x = (i_jan + i_feb) / 2.0
    midpoint_y = (TABLE6_F1[i_jan] + TABLE6_F1[i_feb]) / 2.0
    ax.annotate(
        "FPR spike\n(Jan-Feb 2020)",
        xy=(midpoint_x, midpoint_y),
        xytext=(midpoint_x + 2.2, midpoint_y - 1.7),
        arrowprops=dict(arrowstyle="->", lw=0.6, color="#555"),
        fontsize=7.5,
        ha="left",
    )
    fig.tight_layout()
    out = FIG_DIR / "fig_exp3_monthly.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_exp4_retraining() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    x = np.arange(len(TABLE7_MONTHS))
    ax.plot(x, TABLE7_BASE, marker="o", color="#7F7F7F", label="Static baseline")
    ax.plot(x, TABLE7_INCREMENTAL, marker="s", color="C0", label="Incremental 1%/month")
    # Mark statistically significant gains (p < 0.01).
    sig_mask = [p < 0.01 for p in TABLE7_PVAL]
    sig_x = [x[i] for i in range(len(x)) if sig_mask[i]]
    sig_y = [TABLE7_INCREMENTAL[i] for i in range(len(x)) if sig_mask[i]]
    ax.scatter(sig_x, sig_y, marker="*", s=80, color="C3", zorder=10, label="p < 0.01")
    ax.set_ylim(94, 100)
    ax.set_ylabel("F1 (%)")
    _set_month_axis(ax, TABLE7_MONTHS)
    ax.set_title("Exp 4: Incremental retraining vs. static baseline (LightGBM)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = FIG_DIR / "fig_exp4_retraining.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_exp7_aut_overview() -> None:
    """Bar chart summarizing AUT values for the original Exps 2b/3/4."""
    df = pd.read_csv(RESULTS / "exp7_aut.csv")
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    labels = [f"{r['experiment']}: {r['series']}" for _, r in df.iterrows()]
    x = np.arange(len(df))
    bars = ax.bar(x, df["aut_pct"].to_numpy(), color=["C0", "C0", "C2", "#7F7F7F", "C0"])
    for bar, val in zip(bars, df["aut_pct"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.05, f"{val:.2f}",
                ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(96, 99.5)
    ax.set_ylabel("AUT (%)")
    ax.set_title("Exp 7: Cumulative AUT for Experiments 2b, 3, and 4")
    fig.tight_layout()
    out = FIG_DIR / "fig_exp7_aut_overview.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_exp8_drift_detector() -> None:
    """Two-panel: (left) AUT per (clf, condition) with CI; (right) mean retrains."""
    df = pd.read_csv(RESULTS / "summary_per_condition.csv")
    classifiers = ["lightgbm", "rf", "mlp"]
    pretty_clf = {"lightgbm": "LightGBM", "rf": "Random Forest", "mlp": "MLP"}
    conditions = ["static", "fixed_1pct", "adwin", "ddm"]
    pretty_cond = {"static": "Static", "fixed_1pct": "Fixed 1%/mo",
                   "adwin": "ADWIN", "ddm": "DDM"}
    colors = {"static": "#7F7F7F", "fixed_1pct": "C0", "adwin": "C1", "ddm": "C3"}

    fig, (ax_aut, ax_re) = plt.subplots(1, 2, figsize=(7.16, 3.4))

    width = 0.18
    base_x = np.arange(len(classifiers))

    for j, cond in enumerate(conditions):
        sub = df[df.condition == cond].set_index("classifier").reindex(classifiers)
        means = sub["mean_aut"].to_numpy() * 100  # to percent
        ci_low = sub["ci_low"].to_numpy() * 100
        ci_high = sub["ci_high"].to_numpy() * 100
        err_lo = means - ci_low
        err_hi = ci_high - means
        offset = (j - (len(conditions) - 1) / 2) * width
        ax_aut.bar(base_x + offset, means, width=width,
                   label=pretty_cond[cond], color=colors[cond],
                   yerr=[err_lo, err_hi], capsize=2,
                   edgecolor="black", linewidth=0.3)
    ax_aut.set_xticks(base_x)
    ax_aut.set_xticklabels([pretty_clf[c] for c in classifiers])
    ax_aut.set_ylabel("AUT (%)")
    ax_aut.set_ylim(88, 100)
    ax_aut.set_title("(a) Cumulative AUT, 95% bootstrap CI (10 seeds)")
    ax_aut.legend(loc="lower right", ncol=2, framealpha=0.9)

    # Retrain count panel.
    for j, cond in enumerate(conditions):
        sub = df[df.condition == cond].set_index("classifier").reindex(classifiers)
        retrains = sub["mean_retrains"].to_numpy()
        offset = (j - (len(conditions) - 1) / 2) * width
        ax_re.bar(base_x + offset, retrains, width=width,
                  label=pretty_cond[cond], color=colors[cond],
                  edgecolor="black", linewidth=0.3)
    ax_re.set_xticks(base_x)
    ax_re.set_xticklabels([pretty_clf[c] for c in classifiers])
    ax_re.set_ylabel("Mean retrains over 13 months")
    ax_re.set_title("(b) Retrain frequency per condition")
    ax_re.set_ylim(0, 14)
    ax_re.axhline(13, ls=":", lw=0.7, color="black")
    ax_re.text(0.02, 13.2, "max = 13 (1 per month)", fontsize=7)

    fig.tight_layout()
    out = FIG_DIR / "fig_exp8_drift_detector.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_exp9_uncertainty_vs_random() -> None:
    """Per-month F1 trend with shaded 10-seed CI for random vs uncertainty."""
    rows = []
    for f in sorted((RESULTS / "exp9_active_learning").glob("seed_*.csv")):
        if "_smoke" in f.stem:
            continue
        rows.append(pd.read_csv(f))
    df = pd.concat(rows, ignore_index=True)
    pivot_rand = df[df.condition == "random_1pct"].pivot_table(
        index="seed", columns="month", values="f1"
    ).sort_index(axis=1)
    pivot_unc = df[df.condition == "uncertainty_1pct"].pivot_table(
        index="seed", columns="month", values="f1"
    ).sort_index(axis=1)
    months = pivot_rand.columns.tolist()
    rand_mean = pivot_rand.mean(axis=0).to_numpy() * 100
    rand_lo = pivot_rand.quantile(0.025, axis=0).to_numpy() * 100
    rand_hi = pivot_rand.quantile(0.975, axis=0).to_numpy() * 100
    unc_mean = pivot_unc.mean(axis=0).to_numpy() * 100
    unc_lo = pivot_unc.quantile(0.025, axis=0).to_numpy() * 100
    unc_hi = pivot_unc.quantile(0.975, axis=0).to_numpy() * 100

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    x = range(len(months))
    ax.fill_between(x, rand_lo, rand_hi, alpha=0.18, color="C0")
    ax.plot(x, rand_mean, marker="o", color="C0", label="Random 1%")
    ax.fill_between(x, unc_lo, unc_hi, alpha=0.18, color="C3")
    ax.plot(x, unc_mean, marker="s", color="C3", label="Uncertainty 1%")
    ax.set_ylim(94, 100)
    ax.set_ylabel("F1 (%)")
    _set_month_axis(ax, months)
    ax.set_title("Exp 9: Uncertainty vs. random sampling, LightGBM (10 seeds)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = FIG_DIR / "fig_exp9_uncertainty_vs_random.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    fig_exp2b_monthly()
    fig_exp3_monthly()
    fig_exp4_retraining()
    fig_exp7_aut_overview()
    fig_exp8_drift_detector()
    fig_exp9_uncertainty_vs_random()
    print("\nAll figures written to:", FIG_DIR)
