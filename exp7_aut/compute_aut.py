"""Exp 7 — AUT post-hoc computation from manuscript Tables 5/6/7.

Reviewer 1 concern B3: paper lacks a cumulative metric. We compute AUT (Pendlebury
2019, eq. 2) over the monthly F1 series already reported, without re-running.

Sources (paper main.tex, lines 273-289 / 304-321 / 336-352):
  Table 5 = Exp 2b: EMBER -> BODMAS monthly, LGBM, 14 months (Aug 2019 - Sep 2020).
  Table 6 = Exp 3: BODMAS internal drift, LGBM, 13 months (Sep 2019 - Sep 2020).
  Table 7 = Exp 4: incremental retraining vs static, LGBM, 13 months.

AUT uses trapezoidal integration normalized by window length (see common/metrics.py).
Point estimates only; full bootstrap CI would require per-seed series, which the
new Exp 8/9 produce but the original Exp 2b/3/4 do not store at per-seed granularity.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.metrics import aut  # noqa: E402

# --- Table 5 (Exp 2b: EMBER -> BODMAS monthly F1, LGBM) -------------------------
# Months in chronological order; values from manuscript Table 5 (lines 280-287).
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

# --- Table 6 (Exp 3: BODMAS internal monthly drift, LGBM) -----------------------
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

# --- Table 7 (Exp 4: incremental retraining vs static, LGBM) --------------------
TABLE7_MONTHS = TABLE6_MONTHS  # same 13 months
TABLE7_BASE = [
    97.70, 97.46, 97.90, 98.53,
    96.30, 95.44, 97.83, 99.15,
    97.80, 98.03, 98.18, 98.12, 97.47,
]
TABLE7_INCREMENTAL = [
    97.68, 97.78, 98.31, 98.76,
    96.67, 95.58, 98.46, 99.16,
    98.51, 99.05, 99.01, 99.30, 99.27,
]


def main() -> None:
    rows = []

    rows.append({
        "experiment": "Exp 2b",
        "series": "EMBER 2017 -> BODMAS",
        "n_months": len(TABLE5_TRAINED_2017),
        "first_month": TABLE5_MONTHS[0],
        "last_month": TABLE5_MONTHS[-1],
        "mean_f1": round(sum(TABLE5_TRAINED_2017) / len(TABLE5_TRAINED_2017), 4),
        "aut_pct": round(aut(TABLE5_TRAINED_2017), 4),
    })
    rows.append({
        "experiment": "Exp 2b",
        "series": "EMBER 2018 -> BODMAS",
        "n_months": len(TABLE5_TRAINED_2018),
        "first_month": TABLE5_MONTHS[0],
        "last_month": TABLE5_MONTHS[-1],
        "mean_f1": round(sum(TABLE5_TRAINED_2018) / len(TABLE5_TRAINED_2018), 4),
        "aut_pct": round(aut(TABLE5_TRAINED_2018), 4),
    })
    rows.append({
        "experiment": "Exp 3",
        "series": "BODMAS internal (static)",
        "n_months": len(TABLE6_F1),
        "first_month": TABLE6_MONTHS[0],
        "last_month": TABLE6_MONTHS[-1],
        "mean_f1": round(sum(TABLE6_F1) / len(TABLE6_F1), 4),
        "aut_pct": round(aut(TABLE6_F1), 4),
    })
    rows.append({
        "experiment": "Exp 4",
        "series": "Static baseline",
        "n_months": len(TABLE7_BASE),
        "first_month": TABLE7_MONTHS[0],
        "last_month": TABLE7_MONTHS[-1],
        "mean_f1": round(sum(TABLE7_BASE) / len(TABLE7_BASE), 4),
        "aut_pct": round(aut(TABLE7_BASE), 4),
    })
    rows.append({
        "experiment": "Exp 4",
        "series": "Incremental 1%/month",
        "n_months": len(TABLE7_INCREMENTAL),
        "first_month": TABLE7_MONTHS[0],
        "last_month": TABLE7_MONTHS[-1],
        "mean_f1": round(sum(TABLE7_INCREMENTAL) / len(TABLE7_INCREMENTAL), 4),
        "aut_pct": round(aut(TABLE7_INCREMENTAL), 4),
    })

    # Cross-series differences (the manuscript-relevant numbers).
    incr_minus_base = aut(TABLE7_INCREMENTAL) - aut(TABLE7_BASE)
    print(f"Incremental AUT advantage over Static (Exp 4): {incr_minus_base:+.4f} pp")

    df = pd.DataFrame(rows)
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "exp7_aut.csv", index=False)
    (out_dir / "exp7_aut_summary.json").write_text(
        json.dumps({
            "rows": rows,
            "incremental_minus_base_aut_pp": round(incr_minus_base, 4),
        }, indent=2)
    )
    print(df.to_string(index=False))
    print(f"\nWrote {out_dir / 'exp7_aut.csv'}")


if __name__ == "__main__":
    main()
