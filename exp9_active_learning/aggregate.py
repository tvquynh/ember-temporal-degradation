"""Aggregate Exp 9 active-learning per-seed CSVs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.metrics import aut_with_bootstrap  # noqa: E402
from common.seeds import SEEDS  # noqa: E402

CONDITIONS = ("random_1pct", "uncertainty_1pct")


def load_all(in_dir: Path) -> pd.DataFrame:
    frames = []
    for f in sorted(in_dir.glob("seed_*.csv")):
        if "_smoke" in f.stem:
            continue
        frames.append(pd.read_csv(f))
    if not frames:
        raise FileNotFoundError(f"No non-smoke seed CSVs in {in_dir}")
    return pd.concat(frames, ignore_index=True)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cond in CONDITIONS:
        sub = df[df.condition == cond]
        if sub.empty:
            continue
        pivot = sub.pivot_table(index="seed", columns="month", values="f1").sort_index()
        curves = pivot.to_numpy()
        aut_stats = aut_with_bootstrap(curves)
        labels_used = sub.sort_values("month").groupby("seed").tail(1)["cumulative_labels_used"]
        rows.append({
            "condition": cond,
            "n_seeds": len(curves),
            "n_months": curves.shape[1],
            "mean_aut": aut_stats["mean"],
            "std_aut": aut_stats["std"],
            "ci_low": aut_stats["ci_low"],
            "ci_high": aut_stats["ci_high"],
            "mean_labels_used": float(labels_used.mean()),
            "mean_f1_overall": float(sub["f1"].mean()),
        })
    return pd.DataFrame(rows)


def validate(df: pd.DataFrame) -> None:
    actual = df.groupby("condition")["seed"].nunique().min()
    if actual < len(SEEDS):
        print(f"[validate] WARN: some conditions only have {actual} seeds (expected {len(SEEDS)})")
    nan_count = int(df[["f1", "mcc", "fpr", "fnr"]].isna().sum().sum())
    if nan_count:
        print(f"[validate] WARN: {nan_count} NaN values across f1/mcc/fpr/fnr")


def paired_significance(df: pd.DataFrame) -> dict[str, float]:
    """Wilcoxon signed-rank test: uncertainty vs random per-seed AUT."""
    from scipy.stats import wilcoxon

    rand_pivot = df[df.condition == "random_1pct"].pivot_table(
        index="seed", columns="month", values="f1",
    ).sort_index()
    unc_pivot = df[df.condition == "uncertainty_1pct"].pivot_table(
        index="seed", columns="month", values="f1",
    ).sort_index()
    common_seeds = sorted(set(rand_pivot.index) & set(unc_pivot.index))
    if len(common_seeds) < 5:
        return {"n_seeds": len(common_seeds), "p_value": float("nan")}
    from common.metrics import aut as compute_aut
    rand_aut = np.array([compute_aut(rand_pivot.loc[s].to_numpy()) for s in common_seeds])
    unc_aut = np.array([compute_aut(unc_pivot.loc[s].to_numpy()) for s in common_seeds])
    stat, p = wilcoxon(unc_aut, rand_aut)
    return {
        "n_seeds": len(common_seeds),
        "mean_diff_aut": float((unc_aut - rand_aut).mean()),
        "wilcoxon_stat": float(stat),
        "p_value": float(p),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, default=Path("results") / "exp9_active_learning")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    df = load_all(args.in_dir)
    if args.validate:
        validate(df)
    summary = aggregate(df)
    summary.to_csv(args.in_dir / "summary.csv", index=False)
    sig = paired_significance(df)
    (args.in_dir / "wilcoxon_uncertainty_vs_random.txt").write_text(
        "\n".join(f"{k}: {v}" for k, v in sig.items())
    )
    print(summary.to_string(index=False))
    print("\nWilcoxon (uncertainty vs random AUT):")
    for k, v in sig.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
