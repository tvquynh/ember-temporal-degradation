"""Re-run Exp 1 (in-era + cross-era) and Exp 6 (feature-group ablation) with
**strict appeared-year split** to address reviewer concern MI-1 about
cross-era validity.

Era definition (strict):
  era 2017 = EMBER 2017 samples with appeared.year == 2017
  era 2018 = EMBER 2018 samples with appeared.year == 2018

This eliminates the ~6-8% tail of earlier-year samples present in each
EMBER release that confounds the release-name-based split used in the
original Exp 1.

Outputs:
  results/strict_exp1_inera.csv      (in-era F1/AUC/FNR per seed × clf × era)
  results/strict_exp1_crossera.csv   (cross-era 2017<->2018 per seed × clf)
  results/strict_exp6_ablation.csv   (per-group F1 drop, in-era vs cross-era, LightGBM)

Usage on the compute server:
  python run_strict.py
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier

SEEDS = [42, 123, 456, 789, 1011, 2026, 3141, 4242, 5555, 6789]
DATA_ROOT = Path("/home/apps")
RESULTS_DIR = Path("/home/apps/scripts/p_ieee_access_temporal_resubmit/results/strict")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EMBER_FEATURE_COUNT = 2381


# ---------------------------------------------------------------------------
# Data loading (strict appeared-year filter)
# ---------------------------------------------------------------------------

def load_ember(year: int) -> tuple[np.ndarray, np.ndarray]:
    """Load EMBER 20YY samples whose `appeared` falls in `year`. Combines train + test."""
    train = pd.read_parquet(DATA_ROOT / f"ember{year}{'_2' if year == 2017 else ''}/ember{year}{'_2' if year == 2017 else ''}_train.parquet")
    test  = pd.read_parquet(DATA_ROOT / f"ember{year}{'_2' if year == 2017 else ''}/ember{year}{'_2' if year == 2017 else ''}_test.parquet")
    df = pd.concat([train, test], ignore_index=True)
    df = df[df["label"] != -1].copy()
    df["appeared"] = pd.to_datetime(df["appeared"])
    df = df[df["appeared"].dt.year == year]
    feat_cols = [c for c in df.columns if c not in ("label", "appeared", "sha256", "avclass")]
    feat_cols = feat_cols[:EMBER_FEATURE_COUNT]
    X = df[feat_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.int8)
    print(f"  era {year}: n={len(y):,}, features={X.shape[1]}, "
          f"malware={(y == 1).sum():,}")
    return X, y


# ---------------------------------------------------------------------------
# Classifier factory (matches paper Section 3.2)
# ---------------------------------------------------------------------------

def make_classifier(name: str, seed: int):
    if name == "lightgbm":
        return LGBMClassifier(
            boosting_type="gbdt", n_estimators=500, num_leaves=64,
            random_state=seed, n_jobs=-1, is_unbalance=True,
            deterministic=True, force_row_wise=True, verbose=-1,
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=200, max_depth=30, class_weight="balanced",
            random_state=seed, n_jobs=-1,
        )
    if name == "mlp":
        return ImbPipeline([
            ("oversample", RandomOverSampler(random_state=seed)),
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(256, 128),
                                  random_state=seed, max_iter=200)),
        ])
    raise ValueError(name)


def fit_predict(name: str, seed: int, Xtr, ytr, Xte, yte) -> dict:
    clf = make_classifier(name, seed)
    t0 = time.time()
    clf.fit(Xtr, ytr)
    train_s = time.time() - t0
    proba = clf.predict_proba(Xte)[:, 1]
    pred  = (proba >= 0.5).astype(np.int8)
    return {
        "f1":   f1_score(yte, pred, zero_division=0),
        "auc":  roc_auc_score(yte, proba) if len(np.unique(yte)) == 2 else float("nan"),
        "fnr":  float(((pred == 0) & (yte == 1)).sum() / max(1, (yte == 1).sum())),
        "fpr":  float(((pred == 1) & (yte == 0)).sum() / max(1, (yte == 0).sum())),
        "n_train": len(ytr),
        "n_test":  len(yte),
        "train_s": train_s,
    }


# ---------------------------------------------------------------------------
# Exp 1 — in-era + cross-era binary classification
# ---------------------------------------------------------------------------

def run_exp1(X17, y17, X18, y18) -> None:
    inera_records = []
    cross_records = []

    for clf_name in ["lightgbm", "rf", "mlp"]:
        for seed in SEEDS:
            rng = np.random.default_rng(seed)

            # In-era 2017 (80/20 split)
            idx = rng.permutation(len(y17))
            split = int(0.8 * len(idx))
            tr_idx, te_idx = idx[:split], idx[split:]
            r = fit_predict(clf_name, seed, X17[tr_idx], y17[tr_idx],
                            X17[te_idx], y17[te_idx])
            r.update(seed=seed, classifier=clf_name, era="2017",
                     condition="in-era")
            inera_records.append(r)
            print(f"  in-era 2017 {clf_name} seed={seed}: F1={r['f1']:.4f}")

            # In-era 2018 (80/20 split)
            idx = rng.permutation(len(y18))
            split = int(0.8 * len(idx))
            tr_idx, te_idx = idx[:split], idx[split:]
            r = fit_predict(clf_name, seed, X18[tr_idx], y18[tr_idx],
                            X18[te_idx], y18[te_idx])
            r.update(seed=seed, classifier=clf_name, era="2018",
                     condition="in-era")
            inera_records.append(r)
            print(f"  in-era 2018 {clf_name} seed={seed}: F1={r['f1']:.4f}")

            # Cross-era 2017->2018
            r = fit_predict(clf_name, seed, X17, y17, X18, y18)
            r.update(seed=seed, classifier=clf_name, direction="2017->2018",
                     condition="cross-era")
            cross_records.append(r)
            print(f"  17->18 {clf_name} seed={seed}: F1={r['f1']:.4f}")

            # Cross-era 2018->2017
            r = fit_predict(clf_name, seed, X18, y18, X17, y17)
            r.update(seed=seed, classifier=clf_name, direction="2018->2017",
                     condition="cross-era")
            cross_records.append(r)
            print(f"  18->17 {clf_name} seed={seed}: F1={r['f1']:.4f}")

    pd.DataFrame(inera_records).to_csv(RESULTS_DIR / "strict_exp1_inera.csv", index=False)
    pd.DataFrame(cross_records).to_csv(RESULTS_DIR / "strict_exp1_crossera.csv", index=False)


# ---------------------------------------------------------------------------
# Exp 6 — feature-group ablation (LightGBM only)
# ---------------------------------------------------------------------------

# EMBER v2 feature group ranges (zero-based, inclusive start, exclusive stop).
# Source: ember.PEFeatureExtractor v2 layout, total 2381 features.
EMBER_GROUPS = {
    "ByteHistogram":          (0,    256),
    "ByteEntropyHistogram":   (256,  512),
    "StringExtractor":        (512,  616),
    "GeneralFileInfo":        (616,  626),
    "HeaderFileInfo":         (626,  688),
    "SectionInfo":            (688,  943),
    "ImportsInfo":            (943, 2223),
    "ExportsInfo":            (2223, 2351),
    "DataDirectories":        (2351, 2381),
}


def run_exp6(X17, y17, X18, y18) -> None:
    """Zero out each feature group then measure F1 drop on in-era 2017 (held-out
    20%) and cross-era 2017->2018."""
    records = []

    # Build the in-era 2017 train/test once per seed for fair comparison.
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(y17))
        split = int(0.8 * len(idx))
        tr_idx, te_idx = idx[:split], idx[split:]
        Xtr_inera, ytr_inera = X17[tr_idx], y17[tr_idx]
        Xte_inera, yte_inera = X17[te_idx], y17[te_idx]

        # Baseline (no zeroing)
        baseline_in   = fit_predict("lightgbm", seed,
                                    Xtr_inera, ytr_inera,
                                    Xte_inera, yte_inera)
        baseline_cross = fit_predict("lightgbm", seed,
                                    X17, y17, X18, y18)
        f1_base_in    = baseline_in["f1"]
        f1_base_cross = baseline_cross["f1"]

        for grp_name, (lo, hi) in EMBER_GROUPS.items():
            # In-era ablation: zero columns lo:hi in BOTH train and test of in-era
            Xtr_z = Xtr_inera.copy(); Xtr_z[:, lo:hi] = 0
            Xte_z = Xte_inera.copy(); Xte_z[:, lo:hi] = 0
            r_in = fit_predict("lightgbm", seed, Xtr_z, ytr_inera, Xte_z, yte_inera)
            # Cross-era ablation: zero columns in train (2017) and test (2018)
            X17_z = X17.copy(); X17_z[:, lo:hi] = 0
            X18_z = X18.copy(); X18_z[:, lo:hi] = 0
            r_cross = fit_predict("lightgbm", seed, X17_z, y17, X18_z, y18)
            records.append({
                "seed": seed,
                "group": grp_name,
                "f1_in_baseline":  f1_base_in,
                "f1_in_zeroed":    r_in["f1"],
                "delta_in_pp":     100 * (f1_base_in - r_in["f1"]),
                "f1_cross_baseline": f1_base_cross,
                "f1_cross_zeroed":   r_cross["f1"],
                "delta_cross_pp":    100 * (f1_base_cross - r_cross["f1"]),
            })
            print(f"  seed={seed} {grp_name}: in-era ΔF1={records[-1]['delta_in_pp']:.2f}pp, "
                  f"cross-era ΔF1={records[-1]['delta_cross_pp']:.2f}pp")

    pd.DataFrame(records).to_csv(RESULTS_DIR / "strict_exp6_ablation.csv", index=False)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Loading strict-year data ===")
    X17, y17 = load_ember(2017)
    X18, y18 = load_ember(2018)
    print()
    print("=== Exp 1 — in-era + cross-era ===")
    run_exp1(X17, y17, X18, y18)
    print()
    print("=== Exp 6 — feature-group ablation ===")
    run_exp6(X17, y17, X18, y18)
    print()
    print("All outputs in", RESULTS_DIR)
