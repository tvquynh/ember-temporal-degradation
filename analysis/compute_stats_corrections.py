"""Compute Holm-Bonferroni corrected p-values and Cohen's d_z paired effect
sizes for the per-month Wilcoxon tests in Exp 4 (incremental retraining)
and Exp 8 / 9 (drift-triggered, active learning).

Inputs are the per-seed CSVs already present in `results/`.

Outputs:
  results/stats/exp4_holm.csv         (per-month corrected p, Cohen's d_z)
  results/stats/exp8_paired_effect.csv
  results/stats/exp9_paired_effect.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = RESULTS / "stats"
OUT.mkdir(parents=True, exist_ok=True)


def holm_correct(pvals: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down adjustment. Returns adjusted p-values."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty_like(pvals)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, pvals[idx] * (m - rank))
        adj[idx] = min(running_max, 1.0)
    return adj


def cohens_dz(differences: np.ndarray) -> float:
    """Paired-samples Cohen's d_z = mean(diff) / std(diff, ddof=1)."""
    sd = np.std(differences, ddof=1)
    if sd == 0:
        return float("inf") if np.mean(differences) > 0 else 0.0
    return float(np.mean(differences) / sd)


# ---------------------------------------------------------------------------
# Exp 4 — paper Table 8: per-month F1 incremental vs base.
# Hard-coded from the manuscript Table 8 because original Exp 4 results live
# in a frozen artefact; we re-state the numbers here for transparency.
# ---------------------------------------------------------------------------

TABLE8_MONTHS = [
    "2019-09", "2019-10", "2019-11", "2019-12",
    "2020-01", "2020-02", "2020-03", "2020-04",
    "2020-05", "2020-06", "2020-07", "2020-08", "2020-09",
]
TABLE8_BASE  = [97.70, 97.46, 97.90, 98.53, 96.30, 95.44, 97.83,
                99.15, 97.80, 98.03, 98.18, 98.12, 97.47]
TABLE8_INCR  = [97.68, 97.78, 98.31, 98.76, 96.67, 95.58, 98.46,
                99.16, 98.51, 99.05, 99.01, 99.30, 99.27]
TABLE8_PVAL  = [0.1523, 0.0020, 0.0020, 0.0020, 0.0020, 0.0840,
                0.0020, 0.7695, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020]


def exp4_table() -> pd.DataFrame:
    p = np.array(TABLE8_PVAL)
    holm = holm_correct(p)
    diffs = np.array(TABLE8_INCR) - np.array(TABLE8_BASE)
    df = pd.DataFrame({
        "month":      TABLE8_MONTHS,
        "delta_pp":   diffs,
        "p_raw":      p,
        "p_holm":     holm,
        "sig_holm":   holm < 0.01,
    })
    df.to_csv(OUT / "exp4_holm.csv", index=False)
    print("\n=== Exp 4 — Holm-Bonferroni adjusted p-values ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nAcross-month effect size: Cohen's d_z = {cohens_dz(diffs):.3f}")
    return df


# ---------------------------------------------------------------------------
# Exp 8 — paired AUT differences across seeds.
# Use the per-seed CSVs to compute AUT per (classifier, condition, seed),
# then paired effect sizes for {fixed - static, adwin - static, ddm - static,
# adwin - fixed, ddm - fixed} on LightGBM.
# ---------------------------------------------------------------------------

def per_seed_aut(df_one_seed: pd.DataFrame, classifier: str, condition: str) -> float:
    """Trapezoidal AUT in percent over the monthly F1 series."""
    sub = df_one_seed[(df_one_seed.classifier == classifier) &
                      (df_one_seed.condition == condition)].copy()
    sub = sub.sort_values("month").reset_index(drop=True)
    if len(sub) < 2:
        return float("nan")
    f1 = sub["f1"].values
    return 100.0 * np.trapezoid(f1, dx=1.0) / (len(f1) - 1)


def exp8_paired() -> pd.DataFrame:
    files = sorted((RESULTS / "exp8_drift_detector").glob("seed_*.csv"))
    files = [f for f in files if "_smoke" not in f.stem]
    rows = []
    for f in files:
        d = pd.read_csv(f)
        seed = int(d["seed"].iloc[0])
        clf = d["classifier"].iloc[0]
        for cond in ["static", "fixed_1pct", "adwin", "ddm"]:
            rows.append({"seed": seed, "classifier": clf,
                         "condition": cond,
                         "aut_pct": per_seed_aut(d, clf, cond)})
    per = pd.DataFrame(rows)

    pairings = [
        ("fixed_1pct", "static"),
        ("adwin",      "static"),
        ("ddm",        "static"),
        ("adwin",      "fixed_1pct"),
        ("ddm",        "fixed_1pct"),
    ]
    out = []
    for clf in ["lightgbm", "rf", "mlp"]:
        for a, b in pairings:
            xs = per[(per.classifier == clf) & (per.condition == a)
                     ].sort_values("seed")["aut_pct"].values
            ys = per[(per.classifier == clf) & (per.condition == b)
                     ].sort_values("seed")["aut_pct"].values
            d = xs - ys
            try:
                wstat, p = stats.wilcoxon(xs, ys, zero_method="zsplit")
            except ValueError:
                wstat, p = float("nan"), float("nan")
            out.append({
                "classifier": clf, "comparison": f"{a} - {b}",
                "n_seeds": len(d),
                "mean_diff_pp": float(np.mean(d)),
                "cohens_dz":    cohens_dz(d),
                "wilcoxon_W":   float(wstat),
                "p_two_sided":  float(p),
            })
    df = pd.DataFrame(out)
    df.to_csv(OUT / "exp8_paired_effect.csv", index=False)
    print("\n=== Exp 8 — paired Cohen's d_z + Wilcoxon ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return df


def exp9_paired() -> pd.DataFrame:
    files = sorted((RESULTS / "exp9_active_learning").glob("seed_*.csv"))
    files = [f for f in files if "_smoke" not in f.stem]
    rows = []
    for f in files:
        d = pd.read_csv(f)
        seed = int(d["seed"].iloc[0])
        for cond in ["random_1pct", "uncertainty_1pct"]:
            sub = d[d.condition == cond].sort_values("month")
            f1 = sub["f1"].values
            aut = 100.0 * np.trapezoid(f1, dx=1.0) / (len(f1) - 1) if len(f1) > 1 else float("nan")
            rows.append({"seed": seed, "condition": cond, "aut_pct": aut})
    per = pd.DataFrame(rows)
    xs = per[per.condition == "uncertainty_1pct"].sort_values("seed")["aut_pct"].values
    ys = per[per.condition == "random_1pct"].sort_values("seed")["aut_pct"].values
    d = xs - ys
    try:
        wstat, p = stats.wilcoxon(xs, ys, zero_method="zsplit")
    except ValueError:
        wstat, p = float("nan"), float("nan")
    out = pd.DataFrame([{
        "comparison": "uncertainty_1pct - random_1pct",
        "n_seeds": len(d),
        "mean_diff_pp": float(np.mean(d)),
        "cohens_dz":    cohens_dz(d),
        "wilcoxon_W":   float(wstat),
        "p_two_sided":  float(p),
    }])
    out.to_csv(OUT / "exp9_paired_effect.csv", index=False)
    print("\n=== Exp 9 — paired Cohen's d_z + Wilcoxon ===")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return out


if __name__ == "__main__":
    exp4_table()
    exp8_paired()
    exp9_paired()
    print("\nAll stats written to", OUT)
