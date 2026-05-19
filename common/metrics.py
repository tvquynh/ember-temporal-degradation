"""Metrics: F1, AUC, MCC, FPR, FNR, TPR@1%FPR, AUT (Area Under Time).

AUT is the Pendlebury 2019 TESSERACT cumulative metric requested by Reviewer 1.
We follow the original definition: AUT = trapezoidal integration of monthly F1
over the deployment window, normalized by window length so values are comparable
across windows of different length.

  AUT(F1, T) = (1 / (T - 1)) * sum_{t=0}^{T-2} (F1[t] + F1[t+1]) / 2

This matches the formulation in:
  F. Pendlebury et al., "TESSERACT: Eliminating Experimental Bias in Malware
  Classification across Space and Time," USENIX Security 2019, eq. (2).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


def basic_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None) -> dict[str, float]:
    """Compute the standard suite. y_proba is the malware-class probability."""
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    out = {"f1": f1, "mcc": mcc, "fpr": fpr, "fnr": fnr}
    if y_proba is not None and len(np.unique(y_true)) == 2:
        out["auc"] = float(roc_auc_score(y_true, y_proba))
        out["tpr_at_1fpr"] = tpr_at_fpr(y_true, y_proba, target_fpr=0.01)
    return out


def tpr_at_fpr(y_true: np.ndarray, y_proba: np.ndarray, target_fpr: float = 0.01) -> float:
    """TPR at a fixed FPR threshold (operational deployment metric)."""
    pos = y_proba[y_true == 1]
    neg = y_proba[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    threshold = np.quantile(neg, 1.0 - target_fpr)
    return float((pos >= threshold).mean())


def aut(values: Sequence[float]) -> float:
    """Area Under Time, trapezoidal integration normalized to window length.

    >>> aut([1.0, 1.0, 1.0])  # constant 1.0 over 3 timestamps
    1.0
    >>> abs(aut([1.0, 0.5, 0.0]) - 0.5) < 1e-9
    True
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"aut expects 1-D sequence; got shape {arr.shape}")
    n = arr.size
    if n < 2:
        raise ValueError(f"aut requires >= 2 points; got {n}")
    # Trapezoidal: each pair contributes (a + b) / 2; sum is sum of trapezoid heights.
    # Normalize by number of trapezoids (n - 1) so AUT(constant c) == c.
    return float((arr[:-1] + arr[1:]).sum() / (2.0 * (n - 1)))


def aut_with_bootstrap(
    per_seed_curves: np.ndarray,
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Compute AUT mean and bootstrap CI from multi-seed F1 curves.

    Args:
      per_seed_curves: shape (n_seeds, n_months) of per-seed monthly F1.
      n_bootstrap: number of bootstrap resamples over seeds.
      ci: confidence level (e.g., 0.95).
      rng: numpy random Generator (defaults to fresh).

    Returns dict with keys: mean, std, ci_low, ci_high.
    """
    arr = np.asarray(per_seed_curves, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"per_seed_curves must be 2-D (n_seeds, n_months); got {arr.shape}")
    n_seeds = arr.shape[0]
    per_seed_aut = np.array([aut(arr[i]) for i in range(n_seeds)])
    rng = rng if rng is not None else np.random.default_rng(0)
    boot_means = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n_seeds, n_seeds)
        boot_means[b] = per_seed_aut[idx].mean()
    alpha = (1.0 - ci) / 2.0
    return {
        "mean": float(per_seed_aut.mean()),
        "std": float(per_seed_aut.std(ddof=1)) if n_seeds > 1 else 0.0,
        "ci_low": float(np.quantile(boot_means, alpha)),
        "ci_high": float(np.quantile(boot_means, 1.0 - alpha)),
    }
