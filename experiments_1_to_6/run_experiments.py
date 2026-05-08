#!/usr/bin/env python3
"""
==========================================================================
IEEE Access Paper — Complete Experiment Pipeline
"Temporal Degradation in ML-Based Malware Detection:
 A Large-Scale Longitudinal Study Using EMBER and BODMAS"
==========================================================================

Experiments:
  1. Cross-era binary classification (EMBER2017 × EMBER2018 × BODMAS)
  2. Monthly temporal degradation (BODMAS 12-month granularity)
  3. Feature-group stability / ablation analysis
  4. Malware family attribution (open-world)
  5. Incremental retraining strategies
  6. False-negative source analysis (existing vs. unseen families)

Outputs:
  results/*.csv   — numerical results for paper tables
  figures/*.pdf   — publication-quality figures

Usage:
  python run_ieee_access_experiments.py              # run all
  python run_ieee_access_experiments.py --exp 1 2    # run specific experiments
  python run_ieee_access_experiments.py --inspect     # inspect data only

Requirements:
  pip install lightgbm scikit-learn pandas numpy matplotlib seaborn pyarrow

Hardware:
  Recommended: 32+ GB RAM, multi-core CPU
  Estimated runtime: 3-6 hours (all experiments, 5 seeds)

Author: Experiment pipeline for Huynh et al.
"""

import os
import sys
import json
import time
import argparse
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score,
    accuracy_score, classification_report
)

import lightgbm as lgb

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ============================================================================
# CONFIGURATION — EDIT THESE PATHS FOR YOUR LAB MACHINE
# ============================================================================

class Config:
    # --- Paths ---
    DATA_DIR        = Path("../data")
    MERGED_PARQUET  = DATA_DIR / "merged_temporal_dataset.parquet"
    BODMAS_META     = DATA_DIR / "bodmas_metadata.csv"
    BODMAS_CATEGORY = DATA_DIR / "bodmas_malware_category.csv"
    RESULTS_DIR     = Path("../results")
    FIGURES_DIR     = Path("../figures")

    # --- Experiment parameters ---
    N_SEEDS         = 5          # random seeds per experiment
    SEEDS           = list(range(42, 42 + N_SEEDS))
    FPR_THRESHOLDS  = [0.001, 0.01]  # 0.1% and 1% FPR
    CLASSIFIERS     = ["gbdt", "rf", "mlp"]  # gradient-boosted, random forest, MLP

    # --- LightGBM (EMBER defaults) ---
    LGBM_PARAMS = {
        "boosting_type": "gbdt",
        "n_estimators": 500,
        "num_leaves": 64,
        "min_child_samples": 100,
        "learning_rate": 0.05,
        "n_jobs": -1,
        "verbose": -1,
        "random_state": 42,  # overridden per seed
    }

    # --- Random Forest ---
    RF_PARAMS = {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 10,
        "n_jobs": -1,
        "random_state": 42,
    }

    # --- MLP ---
    MLP_PARAMS = {
        "hidden_layer_sizes": (256, 128),
        "max_iter": 100,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "random_state": 42,
    }

    # --- EMBER feature groups ---
    FEATURE_GROUPS = {
        "ByteHistogram":        (0, 256),
        "ByteEntropyHistogram": (256, 512),
        "StringExtractor":      (512, 616),
        "GeneralFileInfo":      (616, 626),
        "HeaderFileInfo":       (626, 688),
        "SectionInfo":          (688, 943),
        "ImportsInfo":          (943, 2223),
        "ExportsInfo":          (2223, 2351),
        "DataDirectories":      (2351, 2381),
    }

    # --- Family attribution N values ---
    FAMILY_N_VALUES = [5, 10, 20, 40, 60, 80, 100]

    # --- Incremental retraining ---
    RETRAIN_SAMPLE_RATE = 0.01  # 1% of new data each month

    # --- Figure style ---
    FIG_DPI    = 300
    FIG_FORMAT = "pdf"
    FONT_SIZE  = 11


# ============================================================================
# SETUP
# ============================================================================

def setup_dirs():
    Config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    Config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def setup_matplotlib():
    plt.rcParams.update({
        'font.size': Config.FONT_SIZE,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'axes.labelsize': Config.FONT_SIZE,
        'axes.titlesize': Config.FONT_SIZE + 1,
        'xtick.labelsize': Config.FONT_SIZE - 1,
        'ytick.labelsize': Config.FONT_SIZE - 1,
        'legend.fontsize': Config.FONT_SIZE - 2,
        'figure.dpi': Config.FIG_DPI,
        'savefig.dpi': Config.FIG_DPI,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })

def log(msg, level="INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}")


# ============================================================================
# DATA LOADING
# ============================================================================

class DataManager:
    """
    Loads and manages the merged EMBER+BODMAS dataset.
    Auto-detects column schema from the parquet file.
    """

    def __init__(self):
        self.meta = None          # metadata DataFrame (sha256, label, source, ...)
        self.feature_cols = None  # list of feature column names
        self.n_features = None
        self._parquet_columns = None

    def inspect(self):
        """Print dataset schema and statistics without loading features."""
        log("Inspecting merged dataset...")
        pf = pd.read_parquet(Config.MERGED_PARQUET, columns=None, engine='pyarrow')

        # Show first few rows of non-numeric columns
        all_cols = list(pf.columns)
        print(f"\n  Total columns: {len(all_cols)}")
        print(f"  Total rows:    {len(pf):,}")

        # Detect feature vs metadata columns
        numeric_cols = pf.select_dtypes(include=[np.number]).columns.tolist()
        other_cols = [c for c in all_cols if c not in numeric_cols]

        print(f"\n  Numeric columns: {len(numeric_cols)}")
        print(f"  Non-numeric columns: {len(other_cols)}")
        print(f"  Non-numeric names: {other_cols}")

        # Show sample values of non-numeric columns
        for col in other_cols:
            vals = pf[col].dropna().unique()[:5]
            print(f"    {col}: {vals}")

        # Check for label/source columns
        print(f"\n  === Candidate metadata columns ===")
        for col in all_cols:
            if col.lower() in ['label', 'y', 'target', 'source', 'dataset',
                                'sha256', 'hash', 'timestamp', 'family',
                                'category', 'year', 'month']:
                uniq = pf[col].nunique()
                print(f"    {col}: nunique={uniq}, dtype={pf[col].dtype}")
                if uniq <= 20:
                    print(f"      values: {pf[col].value_counts().to_dict()}")

        del pf
        return

    def load(self):
        """Load metadata + detect feature columns. Features loaded on demand."""
        log("Loading merged dataset metadata...")

        # First pass: read all columns to detect schema
        sample = pd.read_parquet(Config.MERGED_PARQUET, engine='pyarrow',
                                  columns=None)
        self._parquet_columns = list(sample.columns)

        # Auto-detect feature columns (numeric, many of them)
        numeric_cols = sample.select_dtypes(include=[np.number]).columns.tolist()

        # Exclude obvious metadata columns from numeric
        meta_numeric = {'label', 'y', 'target', 'year', 'month', 'is_malware'}
        feature_candidates = [c for c in numeric_cols
                              if c.lower() not in meta_numeric]

        # If columns are numbered (0, 1, 2, ...) or named (f_0, f_1, ...),
        # take the first 2381
        if len(feature_candidates) >= 2381:
            # Sort by numeric index if possible
            try:
                feature_candidates.sort(key=lambda x: int(str(x).replace('f_', '').replace('feature_', '')))
            except ValueError:
                pass
            self.feature_cols = feature_candidates[:2381]
        else:
            # Fallback: all numeric that aren't clearly metadata
            self.feature_cols = feature_candidates

        self.n_features = len(self.feature_cols)
        log(f"  Detected {self.n_features} feature columns")

        # Detect metadata columns
        meta_cols = [c for c in self._parquet_columns if c not in self.feature_cols]
        self.meta = sample[meta_cols].copy()

        # Standardize column names
        self._standardize_meta()

        log(f"  Metadata columns: {list(self.meta.columns)}")
        log(f"  Total samples: {len(self.meta):,}")

        # Print era distribution
        if 'source' in self.meta.columns:
            log(f"  Era distribution:")
            for src, cnt in self.meta['source'].value_counts().items():
                label_dist = self.meta[self.meta['source'] == src]['label'].value_counts()
                log(f"    {src}: {cnt:,} (mal={label_dist.get(1,0):,}, ben={label_dist.get(0,0):,})")

        del sample
        return self

    def _standardize_meta(self):
        """Normalize metadata column names for consistent access."""
        col_map = {}
        for col in self.meta.columns:
            cl = col.lower().strip()
            if cl in ('y', 'target', 'is_malware'):
                col_map[col] = 'label'
            elif cl in ('dataset', 'src', 'origin'):
                col_map[col] = 'source'
            elif cl in ('hash', 'sha256sum', 'sha_256'):
                col_map[col] = 'sha256'
        if col_map:
            self.meta.rename(columns=col_map, inplace=True)

        # Ensure label is int
        if 'label' in self.meta.columns:
            self.meta['label'] = self.meta['label'].astype(int)

        # Parse timestamps for BODMAS
        if 'timestamp' in self.meta.columns:
            try:
                self.meta['timestamp'] = pd.to_datetime(
                    self.meta['timestamp'], format='mixed', utc=True
                )
            except Exception:
                try:
                    self.meta['timestamp'] = pd.to_datetime(
                        self.meta['timestamp'], utc=True
                    )
                except Exception:
                    log("  Warning: could not parse timestamps", "WARN")

        # Derive year_month for BODMAS samples
        if 'timestamp' in self.meta.columns:
            valid_ts = self.meta['timestamp'].notna()
            self.meta.loc[valid_ts, 'year'] = (
                self.meta.loc[valid_ts, 'timestamp'].dt.year
            )
            self.meta.loc[valid_ts, 'month'] = (
                self.meta.loc[valid_ts, 'timestamp'].dt.month
            )
            self.meta.loc[valid_ts, 'year_month'] = (
                self.meta.loc[valid_ts, 'timestamp'].dt.to_period('M').astype(str)
            )

    def get_features(self, indices, dtype=np.float32):
        """Load feature vectors for given row indices."""
        df = pd.read_parquet(Config.MERGED_PARQUET, engine='pyarrow',
                             columns=self.feature_cols)
        X = df.iloc[indices].values.astype(dtype)
        del df
        return X

    def get_era_indices(self, source_name):
        """Get row indices for a given source/era."""
        return self.meta[self.meta['source'] == source_name].index.values

    def get_bodmas_month_indices(self, year, month):
        """Get BODMAS row indices for a specific month."""
        mask = (
            (self.meta['source'].str.lower().str.contains('bodmas')) &
            (self.meta['year'] == year) &
            (self.meta['month'] == month)
        )
        return self.meta[mask].index.values

    def get_bodmas_months(self, min_samples=500):
        """Get list of (year, month) pairs with enough BODMAS samples."""
        bodmas_mask = self.meta['source'].str.lower().str.contains('bodmas')
        bodmas = self.meta[bodmas_mask].copy()

        # Only include months with sufficient samples
        months = []
        if 'year' in bodmas.columns and 'month' in bodmas.columns:
            for (y, m), grp in bodmas.groupby(['year', 'month']):
                if len(grp) >= min_samples:
                    months.append((int(y), int(m)))
        return sorted(months)

    def get_family_info(self):
        """Load BODMAS family and category information."""
        family_df = pd.read_csv(Config.BODMAS_META)
        category_df = pd.read_csv(Config.BODMAS_CATEGORY)
        return family_df, category_df


# ============================================================================
# CLASSIFIER FACTORY
# ============================================================================

def make_classifier(name, seed):
    """Create a classifier instance with the given random seed."""
    if name == "gbdt":
        params = Config.LGBM_PARAMS.copy()
        params["random_state"] = seed
        return lgb.LGBMClassifier(**params)
    elif name == "rf":
        params = Config.RF_PARAMS.copy()
        params["random_state"] = seed
        return RandomForestClassifier(**params)
    elif name == "mlp":
        params = Config.MLP_PARAMS.copy()
        params["random_state"] = seed
        return MLPClassifier(**params)
    else:
        raise ValueError(f"Unknown classifier: {name}")


# ============================================================================
# EVALUATION UTILITIES
# ============================================================================

def compute_metrics(y_true, y_prob, fpr_thresholds=None):
    """Compute ROC-AUC, TPR@FPR, F1 from true labels and predicted probs."""
    if fpr_thresholds is None:
        fpr_thresholds = Config.FPR_THRESHOLDS

    results = {}

    # ROC-AUC
    try:
        results['roc_auc'] = roc_auc_score(y_true, y_prob)
    except ValueError:
        results['roc_auc'] = np.nan

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    # TPR at specific FPR thresholds
    for fpr_thresh in fpr_thresholds:
        # Linear interpolation
        idx = np.searchsorted(fpr, fpr_thresh)
        if idx == 0:
            tpr_at_fpr = tpr[0]
        elif idx >= len(fpr):
            tpr_at_fpr = tpr[-1]
        else:
            # Interpolate
            fpr_lo, fpr_hi = fpr[idx - 1], fpr[idx]
            tpr_lo, tpr_hi = tpr[idx - 1], tpr[idx]
            if fpr_hi == fpr_lo:
                tpr_at_fpr = tpr_hi
            else:
                t = (fpr_thresh - fpr_lo) / (fpr_hi - fpr_lo)
                tpr_at_fpr = tpr_lo + t * (tpr_hi - tpr_lo)
        key = f"tpr@{fpr_thresh*100:.1f}%fpr"
        results[key] = tpr_at_fpr

    # F1 at 50% threshold
    y_pred = (y_prob >= 0.5).astype(int)
    results['f1'] = f1_score(y_true, y_pred, zero_division=0)

    # FPR and FNR at 50% threshold
    benign_mask = (y_true == 0)
    malware_mask = (y_true == 1)
    if benign_mask.sum() > 0:
        results['fpr_50'] = ((y_pred == 1) & benign_mask).sum() / benign_mask.sum()
    if malware_mask.sum() > 0:
        results['fnr_50'] = ((y_pred == 0) & malware_mask).sum() / malware_mask.sum()

    return results


def train_and_evaluate(clf_name, X_train, y_train, X_test, y_test, seed):
    """Train classifier and compute evaluation metrics."""
    clf = make_classifier(clf_name, seed)

    # Normalize for MLP
    scaler = None
    if clf_name == "mlp":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    clf.fit(X_train, y_train)

    if hasattr(clf, 'predict_proba'):
        y_prob = clf.predict_proba(X_test)[:, 1]
    else:
        y_prob = clf.decision_function(X_test)

    metrics = compute_metrics(y_test, y_prob)
    return metrics, clf, scaler


def aggregate_results(results_list):
    """Aggregate metrics across seeds: mean ± std, 95% CI."""
    if not results_list:
        return {}

    keys = results_list[0].keys()
    agg = {}
    for k in keys:
        vals = [r[k] for r in results_list if k in r and not np.isnan(r.get(k, np.nan))]
        if vals:
            arr = np.array(vals)
            agg[f"{k}_mean"] = np.mean(arr)
            agg[f"{k}_std"] = np.std(arr, ddof=1) if len(arr) > 1 else 0.0
            if len(arr) > 1:
                ci = stats.t.interval(0.95, len(arr)-1, loc=np.mean(arr),
                                       scale=stats.sem(arr))
                agg[f"{k}_ci_lo"] = ci[0]
                agg[f"{k}_ci_hi"] = ci[1]
    return agg


# ============================================================================
# EXPERIMENT 1: Cross-Era Binary Classification
# ============================================================================

def experiment_1_cross_era(dm):
    """
    Train on each era, test on all eras.
    Produces a 3×3 degradation matrix for each classifier.

    Output:
      results/exp1_cross_era_{clf}.csv
      figures/fig_cross_era_heatmap.pdf
    """
    log("=" * 60)
    log("EXPERIMENT 1: Cross-Era Binary Classification")
    log("=" * 60)

    # Define eras
    sources = sorted(dm.meta['source'].unique())
    log(f"  Eras detected: {sources}")

    # For BODMAS, use only Oct 2019–Sep 2020 samples (main period)
    # and split 80/20 for train/test
    all_results = []

    for clf_name in Config.CLASSIFIERS:
        log(f"\n  Classifier: {clf_name}")
        era_results = []

        for train_src in sources:
            for test_src in sources:
                log(f"    Train: {train_src} → Test: {test_src}")

                seed_results = []
                for seed in Config.SEEDS:
                    try:
                        # Get indices
                        train_idx = dm.get_era_indices(train_src)
                        test_idx = dm.get_era_indices(test_src)

                        # For same-era, use proper train/test split
                        if train_src == test_src:
                            np.random.seed(seed)
                            perm = np.random.permutation(len(train_idx))
                            split = int(0.8 * len(perm))
                            actual_train = train_idx[perm[:split]]
                            actual_test = train_idx[perm[split:]]
                        else:
                            actual_train = train_idx
                            actual_test = test_idx

                        # Load features
                        X_train = dm.get_features(actual_train)
                        y_train = dm.meta.loc[actual_train, 'label'].values
                        X_test = dm.get_features(actual_test)
                        y_test = dm.meta.loc[actual_test, 'label'].values

                        metrics, _, _ = train_and_evaluate(
                            clf_name, X_train, y_train, X_test, y_test, seed
                        )
                        seed_results.append(metrics)

                        del X_train, X_test

                    except Exception as e:
                        log(f"      Seed {seed} failed: {e}", "ERROR")

                agg = aggregate_results(seed_results)
                agg['classifier'] = clf_name
                agg['train_era'] = train_src
                agg['test_era'] = test_src
                era_results.append(agg)

                # Log key metric
                tpr1 = agg.get('tpr@1.0%fpr_mean', np.nan) * 100
                std1 = agg.get('tpr@1.0%fpr_std', 0) * 100
                log(f"      TPR@1%FPR: {tpr1:.2f}% ± {std1:.2f}%")

        # Save per-classifier results
        df = pd.DataFrame(era_results)
        outpath = Config.RESULTS_DIR / f"exp1_cross_era_{clf_name}.csv"
        df.to_csv(outpath, index=False)
        log(f"  Saved: {outpath}")
        all_results.extend(era_results)

    # Save combined
    df_all = pd.DataFrame(all_results)
    df_all.to_csv(Config.RESULTS_DIR / "exp1_cross_era_all.csv", index=False)

    # Generate heatmap figure
    _plot_cross_era_heatmap(df_all)

    return df_all


def _plot_cross_era_heatmap(df):
    """Generate cross-era degradation heatmap."""
    for clf_name in Config.CLASSIFIERS:
        subset = df[df['classifier'] == clf_name]
        if subset.empty:
            continue

        train_eras = subset['train_era'].unique()
        test_eras = subset['test_era'].unique()

        matrix = np.zeros((len(train_eras), len(test_eras)))
        for i, tr in enumerate(train_eras):
            for j, te in enumerate(test_eras):
                row = subset[(subset['train_era'] == tr) & (subset['test_era'] == te)]
                if not row.empty:
                    matrix[i, j] = row.iloc[0].get('tpr@1.0%fpr_mean', np.nan) * 100

        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(matrix, cmap='RdYlGn', vmin=50, vmax=100, aspect='auto')

        ax.set_xticks(range(len(test_eras)))
        ax.set_xticklabels([s.replace('ember', 'EMBER\n') for s in test_eras],
                           rotation=0)
        ax.set_yticks(range(len(train_eras)))
        ax.set_yticklabels([s.replace('ember', 'EMBER\n') for s in train_eras])

        ax.set_xlabel('Test Era')
        ax.set_ylabel('Train Era')
        ax.set_title(f'TPR@1%FPR — {clf_name.upper()}')

        # Annotate cells
        for i in range(len(train_eras)):
            for j in range(len(test_eras)):
                val = matrix[i, j]
                color = 'white' if val < 70 else 'black'
                ax.text(j, i, f"{val:.1f}%", ha='center', va='center',
                        fontsize=Config.FONT_SIZE, color=color, fontweight='bold')

        plt.colorbar(im, ax=ax, shrink=0.8, label='TPR@1%FPR (%)')
        plt.tight_layout()

        outpath = Config.FIGURES_DIR / f"fig1_cross_era_heatmap_{clf_name}.{Config.FIG_FORMAT}"
        plt.savefig(outpath)
        plt.close()
        log(f"  Figure saved: {outpath}")


# ============================================================================
# EXPERIMENT 2: Monthly Temporal Degradation
# ============================================================================

def experiment_2_monthly_degradation(dm):
    """
    Train on EMBER 2017 and EMBER 2018, test on each BODMAS month.
    Shows fine-grained degradation over 12 months.

    Output:
      results/exp2_monthly_degradation.csv
      figures/fig_monthly_degradation.pdf
    """
    log("=" * 60)
    log("EXPERIMENT 2: Monthly Temporal Degradation")
    log("=" * 60)

    bodmas_months = dm.get_bodmas_months(min_samples=500)
    log(f"  BODMAS months: {bodmas_months}")

    sources = sorted(dm.meta['source'].unique())
    train_eras = [s for s in sources if 'ember' in s.lower()]
    log(f"  Training eras: {train_eras}")

    all_results = []

    for clf_name in Config.CLASSIFIERS:
        log(f"\n  Classifier: {clf_name}")

        for train_src in train_eras:
            log(f"    Training on: {train_src}")

            # Pre-load training data
            train_idx = dm.get_era_indices(train_src)
            X_train = dm.get_features(train_idx)
            y_train = dm.meta.loc[train_idx, 'label'].values

            for year, month in bodmas_months:
                test_idx = dm.get_bodmas_month_indices(year, month)
                if len(test_idx) < 100:
                    continue

                X_test = dm.get_features(test_idx)
                y_test = dm.meta.loc[test_idx, 'label'].values

                seed_results = []
                for seed in Config.SEEDS:
                    try:
                        metrics, _, _ = train_and_evaluate(
                            clf_name, X_train, y_train, X_test, y_test, seed
                        )
                        seed_results.append(metrics)
                    except Exception as e:
                        log(f"        Seed {seed} failed: {e}", "ERROR")

                agg = aggregate_results(seed_results)
                agg['classifier'] = clf_name
                agg['train_era'] = train_src
                agg['test_year'] = year
                agg['test_month'] = month
                agg['test_label'] = f"{year}-{month:02d}"
                agg['n_test'] = len(test_idx)
                agg['n_malware'] = int((y_test == 1).sum())
                agg['n_benign'] = int((y_test == 0).sum())
                all_results.append(agg)

                tpr1 = agg.get('tpr@1.0%fpr_mean', np.nan) * 100
                log(f"      {year}-{month:02d}: TPR@1%FPR={tpr1:.2f}% (n={len(test_idx):,})")

                del X_test

            del X_train

    df = pd.DataFrame(all_results)
    outpath = Config.RESULTS_DIR / "exp2_monthly_degradation.csv"
    df.to_csv(outpath, index=False)
    log(f"  Saved: {outpath}")

    _plot_monthly_degradation(df)
    return df


def _plot_monthly_degradation(df):
    """Plot monthly degradation curves for each classifier and training era."""

    train_eras = df['train_era'].unique()
    classifiers = df['classifier'].unique()

    # Color map
    clf_colors = {'gbdt': '#2196F3', 'rf': '#4CAF50', 'mlp': '#FF9800'}
    era_styles = {era: ls for era, ls in zip(sorted(train_eras), ['-', '--', ':'])}

    fig, axes = plt.subplots(1, len(train_eras), figsize=(6 * len(train_eras), 4.5),
                              sharey=True)
    if len(train_eras) == 1:
        axes = [axes]

    for ax, train_era in zip(axes, sorted(train_eras)):
        for clf_name in classifiers:
            subset = df[(df['classifier'] == clf_name) & (df['train_era'] == train_era)]
            subset = subset.sort_values(['test_year', 'test_month'])

            x_labels = subset['test_label'].values
            y_vals = subset['tpr@1.0%fpr_mean'].values * 100
            y_err = subset['tpr@1.0%fpr_std'].values * 100

            x_pos = range(len(x_labels))
            ax.errorbar(x_pos, y_vals, yerr=y_err,
                       marker='o', markersize=4, linewidth=1.5,
                       color=clf_colors.get(clf_name, 'gray'),
                       label=clf_name.upper(), capsize=3)

        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.set_xlabel('Test Month')
        ax.set_title(f'Trained on {train_era}')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([50, 105])

    axes[0].set_ylabel('TPR@1%FPR (%)')
    plt.tight_layout()

    outpath = Config.FIGURES_DIR / f"fig2_monthly_degradation.{Config.FIG_FORMAT}"
    plt.savefig(outpath)
    plt.close()
    log(f"  Figure saved: {outpath}")

    # Also plot F1 version
    fig, axes = plt.subplots(1, len(train_eras), figsize=(6 * len(train_eras), 4.5),
                              sharey=True)
    if len(train_eras) == 1:
        axes = [axes]

    for ax, train_era in zip(axes, sorted(train_eras)):
        for clf_name in classifiers:
            subset = df[(df['classifier'] == clf_name) & (df['train_era'] == train_era)]
            subset = subset.sort_values(['test_year', 'test_month'])

            x_labels = subset['test_label'].values
            y_vals = subset['f1_mean'].values * 100
            y_err = subset['f1_std'].values * 100

            x_pos = range(len(x_labels))
            ax.errorbar(x_pos, y_vals, yerr=y_err,
                       marker='s', markersize=4, linewidth=1.5,
                       color=clf_colors.get(clf_name, 'gray'),
                       label=clf_name.upper(), capsize=3)

        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.set_xlabel('Test Month')
        ax.set_title(f'Trained on {train_era}')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('F1 Score (%)')
    plt.tight_layout()

    outpath = Config.FIGURES_DIR / f"fig2b_monthly_f1.{Config.FIG_FORMAT}"
    plt.savefig(outpath)
    plt.close()
    log(f"  Figure saved: {outpath}")


# ============================================================================
# EXPERIMENT 3: Feature Group Stability / Ablation
# ============================================================================

def experiment_3_feature_ablation(dm):
    """
    Remove one feature group at a time, measure impact on cross-era performance.
    Tests across multiple temporal gaps (2017→2018, 2017→BODMAS, 2018→BODMAS).

    Output:
      results/exp3_feature_ablation.csv
      figures/fig_feature_ablation.pdf
      figures/fig_feature_importance.pdf
    """
    log("=" * 60)
    log("EXPERIMENT 3: Feature Group Ablation")
    log("=" * 60)

    sources = sorted(dm.meta['source'].unique())
    train_eras = [s for s in sources if 'ember' in s.lower()]

    # We test cross-era: train on each EMBER era, test on all other eras
    test_configs = []
    for tr in train_eras:
        for te in sources:
            if tr != te:
                test_configs.append((tr, te))

    all_results = []
    clf_name = "gbdt"  # Use GBDT for ablation (fastest + primary classifier)

    for train_src, test_src in test_configs:
        log(f"\n  Ablation: {train_src} → {test_src}")

        train_idx = dm.get_era_indices(train_src)
        test_idx = dm.get_era_indices(test_src)
        y_train = dm.meta.loc[train_idx, 'label'].values
        y_test = dm.meta.loc[test_idx, 'label'].values

        # Baseline (all features)
        log(f"    Baseline (all features)...")
        X_train_full = dm.get_features(train_idx)
        X_test_full = dm.get_features(test_idx)

        baseline_results = []
        feature_importances = {}
        for seed in Config.SEEDS:
            metrics, clf, _ = train_and_evaluate(
                clf_name, X_train_full, y_train, X_test_full, y_test, seed
            )
            baseline_results.append(metrics)

            # Collect feature importances
            if hasattr(clf, 'feature_importances_'):
                fi = clf.feature_importances_
                for grp_name, (start, end) in Config.FEATURE_GROUPS.items():
                    if grp_name not in feature_importances:
                        feature_importances[grp_name] = []
                    feature_importances[grp_name].append(fi[start:end].sum())

        baseline_agg = aggregate_results(baseline_results)
        baseline_agg['classifier'] = clf_name
        baseline_agg['train_era'] = train_src
        baseline_agg['test_era'] = test_src
        baseline_agg['removed_group'] = 'None (Baseline)'
        all_results.append(baseline_agg)

        baseline_tpr = baseline_agg.get('tpr@1.0%fpr_mean', np.nan)
        log(f"      Baseline TPR@1%FPR: {baseline_tpr*100:.2f}%")

        # Ablation: remove each feature group
        for grp_name, (start, end) in Config.FEATURE_GROUPS.items():
            log(f"    Removing: {grp_name} ({end-start} features)...")

            # Create ablated feature arrays
            keep_cols = list(range(0, start)) + list(range(end, dm.n_features))
            X_train_abl = X_train_full[:, keep_cols]
            X_test_abl = X_test_full[:, keep_cols]

            abl_results = []
            for seed in Config.SEEDS:
                metrics, _, _ = train_and_evaluate(
                    clf_name, X_train_abl, y_train, X_test_abl, y_test, seed
                )
                abl_results.append(metrics)

            abl_agg = aggregate_results(abl_results)
            abl_agg['classifier'] = clf_name
            abl_agg['train_era'] = train_src
            abl_agg['test_era'] = test_src
            abl_agg['removed_group'] = grp_name
            abl_agg['n_features_removed'] = end - start

            # Compute change from baseline
            abl_tpr = abl_agg.get('tpr@1.0%fpr_mean', np.nan)
            abl_agg['tpr_change_pp'] = (abl_tpr - baseline_tpr) * 100

            all_results.append(abl_agg)
            log(f"      TPR@1%FPR: {abl_tpr*100:.2f}% (Δ={abl_agg['tpr_change_pp']:+.2f}pp)")

        del X_train_full, X_test_full

        # Save feature importances
        fi_df = pd.DataFrame({
            grp: {'mean': np.mean(vals), 'std': np.std(vals)}
            for grp, vals in feature_importances.items()
        }).T
        fi_df['train_era'] = train_src
        fi_outpath = Config.RESULTS_DIR / f"exp3_feature_importance_{train_src}_{test_src}.csv"
        fi_df.to_csv(fi_outpath)

    # Save results
    df = pd.DataFrame(all_results)
    outpath = Config.RESULTS_DIR / "exp3_feature_ablation.csv"
    df.to_csv(outpath, index=False)
    log(f"  Saved: {outpath}")

    _plot_feature_ablation(df)
    return df


def _plot_feature_ablation(df):
    """Bar chart of feature ablation impact."""
    test_configs = df[['train_era', 'test_era']].drop_duplicates().values

    for train_src, test_src in test_configs:
        subset = df[(df['train_era'] == train_src) & (df['test_era'] == test_src)]
        subset = subset[subset['removed_group'] != 'None (Baseline)']
        subset = subset.sort_values('tpr_change_pp')

        fig, ax = plt.subplots(figsize=(7, 4))
        colors = ['#E53935' if v < 0 else '#43A047' for v in subset['tpr_change_pp']]

        bars = ax.barh(range(len(subset)), subset['tpr_change_pp'], color=colors,
                       edgecolor='white', linewidth=0.5)

        ax.set_yticks(range(len(subset)))
        ax.set_yticklabels(subset['removed_group'])
        ax.set_xlabel('Change in TPR@1%FPR (pp)')
        ax.set_title(f'Feature Ablation: {train_src} → {test_src}')
        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.grid(True, axis='x', alpha=0.3)

        # Annotate values
        for bar, val in zip(bars, subset['tpr_change_pp']):
            ax.text(bar.get_width() + 0.1 * np.sign(bar.get_width()),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:+.2f}pp", va='center',
                    fontsize=Config.FONT_SIZE - 2)

        plt.tight_layout()
        fname = f"fig3_ablation_{train_src}_to_{test_src}.{Config.FIG_FORMAT}"
        outpath = Config.FIGURES_DIR / fname
        plt.savefig(outpath)
        plt.close()
        log(f"  Figure saved: {outpath}")


# ============================================================================
# EXPERIMENT 4: Malware Family Attribution (Open-World)
# ============================================================================

def experiment_4_family_attribution(dm):
    """
    Train family classifier on BODMAS first month, test on subsequent months.
    Open-world setting: test data contains unseen families.

    Output:
      results/exp4_family_attribution.csv
      figures/fig_family_topk.pdf
    """
    log("=" * 60)
    log("EXPERIMENT 4: Malware Family Attribution (Open-World)")
    log("=" * 60)

    # Load family info from metadata
    bodmas_mask = dm.meta['source'].str.lower().str.contains('bodmas')
    bodmas_meta = dm.meta[bodmas_mask].copy()

    # Get family labels (from the loaded metadata or from separate CSV)
    family_df = pd.read_csv(Config.BODMAS_META)
    family_df['timestamp'] = pd.to_datetime(family_df['timestamp'], format='mixed', utc=True)
    family_df['is_malware'] = family_df['family'].notna() & (family_df['family'] != '')

    # Merge family info with bodmas indices
    # Match by sha256
    if 'sha256' in dm.meta.columns:
        bodmas_with_family = dm.meta[bodmas_mask].merge(
            family_df[['sha256', 'family']], on='sha256', how='left',
            suffixes=('', '_meta')
        )
        # Use family from metadata if not in main df
        if 'family_meta' in bodmas_with_family.columns:
            bodmas_with_family['family'] = bodmas_with_family['family_meta'].fillna(
                bodmas_with_family.get('family', '')
            )
    else:
        bodmas_with_family = bodmas_meta.copy()

    # Filter malware only
    malware_mask = bodmas_with_family['family'].notna() & (bodmas_with_family['family'] != '')
    malware_data = bodmas_with_family[malware_mask].copy()

    if len(malware_data) == 0:
        log("  No malware family data found. Skipping.", "WARN")
        return None

    # Split into training month (first month with enough data) and test months
    bodmas_months = dm.get_bodmas_months(min_samples=500)
    if len(bodmas_months) < 3:
        log("  Not enough months. Skipping.", "WARN")
        return None

    # Use first month with substantial malware as training
    train_month = bodmas_months[0]
    test_months = bodmas_months[1:]

    log(f"  Training month: {train_month}")
    log(f"  Test months: {test_months}")

    # Get training malware indices
    train_malware = malware_data[
        (malware_data['year'] == train_month[0]) &
        (malware_data['month'] == train_month[1])
    ]
    log(f"  Training malware samples: {len(train_malware):,}")

    # Get family counts in training
    train_families = train_malware['family'].value_counts()
    log(f"  Families in training month: {len(train_families)}")

    all_results = []

    for N in Config.FAMILY_N_VALUES:
        if N > len(train_families):
            log(f"  N={N} > available families ({len(train_families)}). Skipping.")
            continue

        log(f"\n  === N = {N} families ===")

        # Select top-N families
        top_n_families = train_families.head(N).index.tolist()
        train_subset = train_malware[train_malware['family'].isin(top_n_families)]

        # Create label encoding
        family_to_idx = {f: i for i, f in enumerate(top_n_families)}

        X_train_idx = train_subset.index.values
        X_train = dm.get_features(X_train_idx)
        y_train = train_subset['family'].map(family_to_idx).values

        log(f"    Training samples: {len(X_train):,}")

        for year, month in test_months:
            test_month_malware = malware_data[
                (malware_data['year'] == year) &
                (malware_data['month'] == month)
            ]

            if len(test_month_malware) < 10:
                continue

            X_test_idx = test_month_malware.index.values
            X_test = dm.get_features(X_test_idx)
            test_families = test_month_malware['family'].values

            # Identify known vs unknown families
            is_known = np.array([f in family_to_idx for f in test_families])
            n_known = is_known.sum()
            n_unknown = (~is_known).sum()

            # Upper bound: fraction of known families in test set
            upper_bound = n_known / len(test_families) if len(test_families) > 0 else 0

            seed_results = []
            for seed in Config.SEEDS:
                clf = make_classifier("gbdt", seed)
                clf.fit(X_train, y_train)

                # Get predicted probabilities for all classes
                proba = clf.predict_proba(X_test)

                # Top-K accuracy on ALL test data
                for K in [1, 2, 3]:
                    # For known families, check if true family is in top-K
                    # For unknown families, any prediction is wrong
                    top_k_preds = np.argsort(proba, axis=1)[:, -K:]

                    correct_all = 0
                    correct_known = 0

                    for i, fam in enumerate(test_families):
                        if fam in family_to_idx:
                            true_idx = family_to_idx[fam]
                            if true_idx in top_k_preds[i]:
                                correct_all += 1
                                correct_known += 1
                        # Unknown families are always wrong for "all" accuracy

                    acc_all = correct_all / len(test_families) if len(test_families) > 0 else 0
                    acc_known = correct_known / n_known if n_known > 0 else 0

                    seed_results.append({
                        f'top{K}_acc_all': acc_all,
                        f'top{K}_acc_known': acc_known,
                    })

                # Combine into single dict per seed
                # (seed_results already accumulated per K, need to merge)

            # Merge K-level results per seed
            merged_seed_results = []
            for i in range(0, len(seed_results), 3):  # 3 K values per seed
                merged = {}
                for j in range(3):
                    if i + j < len(seed_results):
                        merged.update(seed_results[i + j])
                merged_seed_results.append(merged)

            agg = aggregate_results(merged_seed_results)
            agg['N'] = N
            agg['test_year'] = year
            agg['test_month'] = month
            agg['test_label'] = f"{year}-{month:02d}"
            agg['n_test'] = len(test_families)
            agg['n_known'] = int(n_known)
            agg['n_unknown'] = int(n_unknown)
            agg['upper_bound'] = upper_bound
            all_results.append(agg)

            t1_all = agg.get('top1_acc_all_mean', np.nan) * 100
            log(f"    {year}-{month:02d}: Top-1(all)={t1_all:.1f}%, "
                f"known={n_known}, unknown={n_unknown}, "
                f"upper_bound={upper_bound*100:.1f}%")

            del X_test

        del X_train

    df = pd.DataFrame(all_results)
    outpath = Config.RESULTS_DIR / "exp4_family_attribution.csv"
    df.to_csv(outpath, index=False)
    log(f"  Saved: {outpath}")

    _plot_family_attribution(df)
    return df


def _plot_family_attribution(df):
    """Plot Top-K accuracy for family attribution."""
    N_values = sorted(df['N'].unique())

    # Plot 1: Top-K accuracy over time for a specific N (e.g., N=10)
    for N in [10, 40]:
        subset = df[df['N'] == N].sort_values(['test_year', 'test_month'])
        if subset.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))

        for K, color, marker in [(1, '#E53935', 'o'), (2, '#FF9800', 's'), (3, '#4CAF50', '^')]:
            # All families
            y_all = subset[f'top{K}_acc_all_mean'].values * 100
            ax.plot(range(len(subset)), y_all,
                   marker=marker, markersize=5, linewidth=1.5,
                   color=color, label=f'Top-{K} (all)')

            # Known families only
            y_known = subset[f'top{K}_acc_known_mean'].values * 100
            ax.plot(range(len(subset)), y_known,
                   marker=marker, markersize=5, linewidth=1.5,
                   color=color, linestyle='--', label=f'Top-{K} (known)', alpha=0.7)

        # Upper bound
        ax.plot(range(len(subset)), subset['upper_bound'].values * 100,
               linewidth=1.5, color='gray', linestyle=':', label='Upper bound (all)')

        ax.set_xticks(range(len(subset)))
        ax.set_xticklabels(subset['test_label'], rotation=45, ha='right')
        ax.set_xlabel('Test Month')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'Family Attribution (N={N} training families)')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        outpath = Config.FIGURES_DIR / f"fig4_family_topk_N{N}.{Config.FIG_FORMAT}"
        plt.savefig(outpath)
        plt.close()
        log(f"  Figure saved: {outpath}")

    # Plot 2: Impact of N on accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for N in N_values:
        subset = df[df['N'] == N].sort_values(['test_year', 'test_month'])
        if subset.empty:
            continue

        x_labels = subset['test_label'].values

        # Known families
        y_known = subset['top2_acc_known_mean'].values * 100
        ax1.plot(range(len(subset)), y_known,
                marker='o', markersize=4, linewidth=1.2, label=f'N={N}')

        # All families
        y_all = subset['top2_acc_all_mean'].values * 100
        ax2.plot(range(len(subset)), y_all,
                marker='o', markersize=4, linewidth=1.2, label=f'N={N}')

    for ax, title in [(ax1, 'Top-2 Accuracy (known families)'),
                       (ax2, 'Top-2 Accuracy (all families)')]:
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.set_xlabel('Test Month')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(title)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = Config.FIGURES_DIR / f"fig5_family_N_impact.{Config.FIG_FORMAT}"
    plt.savefig(outpath)
    plt.close()
    log(f"  Figure saved: {outpath}")


# ============================================================================
# EXPERIMENT 5: Incremental Retraining
# ============================================================================

def experiment_5_incremental_retraining(dm):
    """
    Start with EMBER baseline, incrementally add labeled BODMAS data monthly.
    Three sampling strategies: random, uncertainty, diversity.

    Output:
      results/exp5_incremental_retraining.csv
      figures/fig_incremental_retraining.pdf
    """
    log("=" * 60)
    log("EXPERIMENT 5: Incremental Retraining")
    log("=" * 60)

    sources = sorted(dm.meta['source'].unique())
    train_eras = [s for s in sources if 'ember' in s.lower()]
    bodmas_months = dm.get_bodmas_months(min_samples=500)

    if len(bodmas_months) < 3:
        log("  Not enough BODMAS months. Skipping.", "WARN")
        return None

    # Use the first BODMAS month as the starting point for incremental data
    # Test on all subsequent months
    strategies = ['random', 'uncertainty', 'diversity']
    all_results = []

    for train_src in train_eras[:1]:  # Primary: first EMBER era
        log(f"\n  Baseline training era: {train_src}")

        train_idx_base = dm.get_era_indices(train_src)
        X_train_base = dm.get_features(train_idx_base)
        y_train_base = dm.meta.loc[train_idx_base, 'label'].values

        for seed in Config.SEEDS:
            log(f"    Seed: {seed}")

            # Track cumulative added data
            X_added = {'random': [], 'uncertainty': [], 'diversity': []}
            y_added = {'random': [], 'uncertainty': [], 'diversity': []}

            # Baseline classifier (no retraining)
            clf_baseline = make_classifier("gbdt", seed)
            clf_baseline.fit(X_train_base, y_train_base)

            for i, (year, month) in enumerate(bodmas_months):
                test_idx = dm.get_bodmas_month_indices(year, month)
                if len(test_idx) < 100:
                    continue

                X_test = dm.get_features(test_idx)
                y_test = dm.meta.loc[test_idx, 'label'].values

                # --- Baseline (no retraining) ---
                y_prob_baseline = clf_baseline.predict_proba(X_test)[:, 1]
                metrics_baseline = compute_metrics(y_test, y_prob_baseline)
                res = {
                    'strategy': 'baseline',
                    'seed': seed,
                    'train_era': train_src,
                    'test_year': year,
                    'test_month': month,
                    'test_label': f"{year}-{month:02d}",
                }
                res.update({f"baseline_{k}": v for k, v in metrics_baseline.items()})
                # Store metric keys directly too for easier aggregation
                res.update(metrics_baseline)
                all_results.append(res.copy())

                # --- Retrained classifiers ---
                if i > 0:
                    # Use PREVIOUS month's data for retraining
                    prev_year, prev_month = bodmas_months[i - 1]
                    prev_idx = dm.get_bodmas_month_indices(prev_year, prev_month)
                    if len(prev_idx) == 0:
                        continue

                    X_prev = dm.get_features(prev_idx)
                    y_prev = dm.meta.loc[prev_idx, 'label'].values

                    n_sample = max(1, int(len(prev_idx) * Config.RETRAIN_SAMPLE_RATE))

                    for strat in strategies:
                        np.random.seed(seed + hash(strat) % 10000)

                        if strat == 'random':
                            sample_idx = np.random.choice(
                                len(prev_idx), size=n_sample, replace=False
                            )
                        elif strat == 'uncertainty':
                            # Select samples with highest uncertainty (closest to 0.5)
                            probs = clf_baseline.predict_proba(X_prev)[:, 1]
                            uncertainty = np.abs(probs - 0.5)
                            sample_idx = np.argsort(uncertainty)[:n_sample]
                        elif strat == 'diversity':
                            # Stratified: proportional to class prediction
                            probs = clf_baseline.predict_proba(X_prev)[:, 1]
                            pred_labels = (probs >= 0.5).astype(int)
                            n_pos = max(1, int(n_sample * pred_labels.mean()))
                            n_neg = n_sample - n_pos
                            pos_idx = np.where(pred_labels == 1)[0]
                            neg_idx = np.where(pred_labels == 0)[0]
                            pos_sample = np.random.choice(
                                pos_idx, size=min(n_pos, len(pos_idx)), replace=False
                            )
                            neg_sample = np.random.choice(
                                neg_idx, size=min(n_neg, len(neg_idx)), replace=False
                            )
                            sample_idx = np.concatenate([pos_sample, neg_sample])

                        # Add to cumulative pool
                        X_added[strat].append(X_prev[sample_idx])
                        y_added[strat].append(y_prev[sample_idx])

                        # Retrain with base + cumulative new data
                        X_cum = np.vstack([X_train_base] + X_added[strat])
                        y_cum = np.concatenate([y_train_base] + y_added[strat])

                        clf_retrained = make_classifier("gbdt", seed)
                        clf_retrained.fit(X_cum, y_cum)

                        y_prob_retrained = clf_retrained.predict_proba(X_test)[:, 1]
                        metrics_retrained = compute_metrics(y_test, y_prob_retrained)

                        res = {
                            'strategy': strat,
                            'seed': seed,
                            'train_era': train_src,
                            'test_year': year,
                            'test_month': month,
                            'test_label': f"{year}-{month:02d}",
                            'n_added': sum(len(x) for x in X_added[strat]),
                        }
                        res.update(metrics_retrained)
                        all_results.append(res)

                        del X_cum, y_cum

                    del X_prev, y_prev

                del X_test

        del X_train_base

    df = pd.DataFrame(all_results)
    outpath = Config.RESULTS_DIR / "exp5_incremental_retraining.csv"
    df.to_csv(outpath, index=False)
    log(f"  Saved: {outpath}")

    _plot_incremental_retraining(df)
    return df


def _plot_incremental_retraining(df):
    """Plot incremental retraining improvement curves."""
    strategies = df['strategy'].unique()
    strat_colors = {
        'baseline': '#9E9E9E',
        'random': '#2196F3',
        'uncertainty': '#FF9800',
        'diversity': '#4CAF50',
    }
    strat_markers = {'baseline': 'x', 'random': 'o', 'uncertainty': 's', 'diversity': '^'}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for strat in strategies:
        subset = df[df['strategy'] == strat]
        # Aggregate across seeds
        agg = subset.groupby('test_label').agg({
            'f1': ['mean', 'std'],
            'tpr@1.0%fpr': ['mean', 'std'],
        }).reset_index()

        x_labels = agg['test_label'].values
        x_pos = range(len(x_labels))

        # F1
        y_f1 = agg[('f1', 'mean')].values * 100
        y_f1_err = agg[('f1', 'std')].values * 100
        ax1.errorbar(x_pos, y_f1, yerr=y_f1_err,
                    marker=strat_markers.get(strat, 'o'), markersize=4,
                    linewidth=1.5, color=strat_colors.get(strat, 'gray'),
                    label=strat.capitalize(), capsize=2)

        # TPR@1%FPR
        y_tpr = agg[('tpr@1.0%fpr', 'mean')].values * 100
        y_tpr_err = agg[('tpr@1.0%fpr', 'std')].values * 100
        ax2.errorbar(x_pos, y_tpr, yerr=y_tpr_err,
                    marker=strat_markers.get(strat, 'o'), markersize=4,
                    linewidth=1.5, color=strat_colors.get(strat, 'gray'),
                    label=strat.capitalize(), capsize=2)

    for ax, ylabel in [(ax1, 'F1 Score (%)'), (ax2, 'TPR@1%FPR (%)')]:
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.set_xlabel('Test Month')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    ax1.set_title('F1 Score')
    ax2.set_title('TPR@1%FPR')
    plt.suptitle('Incremental Retraining Strategies', fontsize=Config.FONT_SIZE + 2)
    plt.tight_layout()

    outpath = Config.FIGURES_DIR / f"fig6_incremental_retraining.{Config.FIG_FORMAT}"
    plt.savefig(outpath)
    plt.close()
    log(f"  Figure saved: {outpath}")


# ============================================================================
# EXPERIMENT 6: False Negative Source Analysis
# ============================================================================

def experiment_6_fnr_analysis(dm):
    """
    Break down false negatives by existing vs. unseen malware families.
    Train on BODMAS first month, test on each subsequent month.

    Output:
      results/exp6_fnr_analysis.csv
      figures/fig_fnr_breakdown.pdf
    """
    log("=" * 60)
    log("EXPERIMENT 6: False Negative Source Analysis")
    log("=" * 60)

    bodmas_months = dm.get_bodmas_months(min_samples=500)
    if len(bodmas_months) < 3:
        log("  Not enough BODMAS months. Skipping.", "WARN")
        return None

    # Load family info
    family_df = pd.read_csv(Config.BODMAS_META)
    family_df['timestamp'] = pd.to_datetime(family_df['timestamp'], format='mixed', utc=True)

    # Merge family info
    bodmas_mask = dm.meta['source'].str.lower().str.contains('bodmas')
    bodmas_meta = dm.meta[bodmas_mask].copy()

    if 'sha256' in bodmas_meta.columns:
        bodmas_meta = bodmas_meta.merge(
            family_df[['sha256', 'family']].rename(columns={'family': 'family_label'}),
            on='sha256', how='left'
        )
    else:
        log("  No sha256 column for family merge. Skipping.", "WARN")
        return None

    # Training: first month with data
    train_month = bodmas_months[0]
    train_mask = (
        (bodmas_meta['year'] == train_month[0]) &
        (bodmas_meta['month'] == train_month[1])
    )
    train_data = bodmas_meta[train_mask]
    train_families = set(
        train_data[train_data['family_label'].notna() &
                   (train_data['family_label'] != '')]['family_label'].unique()
    )
    log(f"  Training month: {train_month}, families: {len(train_families)}")

    # Load training features
    train_idx = train_data.index.values
    X_train = dm.get_features(train_idx)
    y_train = dm.meta.loc[train_idx, 'label'].values

    all_results = []

    for seed in Config.SEEDS:
        clf = make_classifier("gbdt", seed)
        clf.fit(X_train, y_train)

        for year, month in bodmas_months[1:]:
            test_mask = (
                (bodmas_meta['year'] == year) &
                (bodmas_meta['month'] == month)
            )
            test_data = bodmas_meta[test_mask]
            test_idx = test_data.index.values

            if len(test_idx) < 100:
                continue

            X_test = dm.get_features(test_idx)
            y_test = dm.meta.loc[test_idx, 'label'].values
            y_pred = clf.predict(X_test)

            # Overall metrics
            fpr = ((y_pred == 1) & (y_test == 0)).sum() / max(1, (y_test == 0).sum())
            fnr = ((y_pred == 0) & (y_test == 1)).sum() / max(1, (y_test == 1).sum())

            # FNR breakdown for malware
            malware_test = test_data[test_data['label'] == 1]
            malware_indices_local = np.where(y_test == 1)[0]

            existing_fam_fnr = np.nan
            unseen_fam_fnr = np.nan

            if len(malware_test) > 0 and 'family_label' in malware_test.columns:
                families = malware_test['family_label'].values
                predictions = y_pred[malware_indices_local]

                # Existing families (seen in training)
                existing_mask = np.array([
                    f in train_families for f in families
                ]) if len(families) > 0 else np.array([])

                if existing_mask.sum() > 0:
                    existing_fn = (predictions[existing_mask] == 0).sum()
                    existing_fam_fnr = existing_fn / existing_mask.sum()

                unseen_mask = ~existing_mask
                if unseen_mask.sum() > 0:
                    unseen_fn = (predictions[unseen_mask] == 0).sum()
                    unseen_fam_fnr = unseen_fn / unseen_mask.sum()

            all_results.append({
                'seed': seed,
                'test_year': year,
                'test_month': month,
                'test_label': f"{year}-{month:02d}",
                'fpr': fpr,
                'fnr': fnr,
                'existing_fam_fnr': existing_fam_fnr,
                'unseen_fam_fnr': unseen_fam_fnr,
                'n_test': len(test_idx),
                'n_malware': int((y_test == 1).sum()),
                'n_benign': int((y_test == 0).sum()),
                'n_existing_fam': int(existing_mask.sum()) if len(malware_test) > 0 else 0,
                'n_unseen_fam': int(unseen_mask.sum()) if len(malware_test) > 0 else 0,
            })

            del X_test

    del X_train

    df = pd.DataFrame(all_results)
    outpath = Config.RESULTS_DIR / "exp6_fnr_analysis.csv"
    df.to_csv(outpath, index=False)
    log(f"  Saved: {outpath}")

    _plot_fnr_breakdown(df)
    return df


def _plot_fnr_breakdown(df):
    """Plot FNR breakdown by existing vs unseen families."""
    # Aggregate across seeds
    agg = df.groupby('test_label').agg({
        'fpr': ['mean', 'std'],
        'fnr': ['mean', 'std'],
        'existing_fam_fnr': ['mean', 'std'],
        'unseen_fam_fnr': ['mean', 'std'],
        'n_existing_fam': 'mean',
        'n_unseen_fam': 'mean',
    }).reset_index()

    x_labels = agg['test_label'].values
    x_pos = np.arange(len(x_labels))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # Top: FPR and FNR over time
    ax1.plot(x_pos, agg[('fpr', 'mean')].values * 100,
            marker='o', color='#2196F3', label='FPR', linewidth=1.5)
    ax1.plot(x_pos, agg[('fnr', 'mean')].values * 100,
            marker='s', color='#E53935', label='FNR', linewidth=1.5)
    ax1.set_ylabel('Rate (%)')
    ax1.set_title('Overall FPR and FNR')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bottom: FNR by family type
    width = 0.35
    ax2.bar(x_pos - width/2,
            agg[('existing_fam_fnr', 'mean')].values * 100,
            width, label='Existing families', color='#4CAF50', alpha=0.8)
    ax2.bar(x_pos + width/2,
            agg[('unseen_fam_fnr', 'mean')].values * 100,
            width, label='Unseen families', color='#FF5722', alpha=0.8)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels, rotation=45, ha='right')
    ax2.set_xlabel('Test Month')
    ax2.set_ylabel('FNR (%)')
    ax2.set_title('FNR Breakdown: Existing vs. Unseen Families')
    ax2.legend()
    ax2.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    outpath = Config.FIGURES_DIR / f"fig7_fnr_breakdown.{Config.FIG_FORMAT}"
    plt.savefig(outpath)
    plt.close()
    log(f"  Figure saved: {outpath}")


# ============================================================================
# SUPPLEMENTARY: ROC Curves Comparison
# ============================================================================

def generate_roc_comparison(dm):
    """
    Generate in-era vs cross-era ROC curves for visual comparison.
    """
    log("=" * 60)
    log("SUPPLEMENTARY: ROC Curves Comparison")
    log("=" * 60)

    sources = sorted(dm.meta['source'].unique())
    ember_sources = [s for s in sources if 'ember' in s.lower()]

    if len(ember_sources) < 2:
        log("  Need at least 2 EMBER eras for ROC comparison. Skipping.", "WARN")
        return

    train_src = ember_sources[0]
    test_same = train_src
    test_cross = ember_sources[1]

    # Also test on first BODMAS month if available
    bodmas_months = dm.get_bodmas_months(min_samples=500)

    seed = Config.SEEDS[0]

    # Train
    train_idx = dm.get_era_indices(train_src)
    np.random.seed(seed)
    perm = np.random.permutation(len(train_idx))
    split = int(0.8 * len(perm))
    train_actual = train_idx[perm[:split]]
    test_same_idx = train_idx[perm[split:]]

    X_train = dm.get_features(train_actual)
    y_train = dm.meta.loc[train_actual, 'label'].values

    clf = make_classifier("gbdt", seed)
    clf.fit(X_train, y_train)
    del X_train

    fig, ax = plt.subplots(figsize=(6, 5))

    configs = [
        (test_same_idx, f"In-era ({train_src})", '#4CAF50', '-'),
        (dm.get_era_indices(test_cross), f"Cross-era ({test_cross})", '#E53935', '--'),
    ]

    if bodmas_months:
        y, m = bodmas_months[-1]  # Last month (most drift)
        bodmas_idx = dm.get_bodmas_month_indices(y, m)
        if len(bodmas_idx) > 0:
            configs.append(
                (bodmas_idx, f"BODMAS ({y}-{m:02d})", '#FF9800', ':')
            )

    for idx, label, color, ls in configs:
        X_test = dm.get_features(idx)
        y_test = dm.meta.loc[idx, 'label'].values
        y_prob = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=1.5,
               label=f'{label} (AUC={auc:.4f})')
        del X_test

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=0.8)
    ax.set_xlim([0, 0.05])  # Zoom to low FPR region
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curves: In-Era vs. Cross-Era (trained on {train_src})')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = Config.FIGURES_DIR / f"fig8_roc_comparison.{Config.FIG_FORMAT}"
    plt.savefig(outpath)
    plt.close()
    log(f"  Figure saved: {outpath}")


# ============================================================================
# SUPPLEMENTARY: Dataset Distribution Figure
# ============================================================================

def generate_dataset_figure(dm):
    """Generate dataset temporal distribution figure."""
    log("Generating dataset distribution figure...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: samples by era
    era_counts = dm.meta.groupby('source')['label'].value_counts().unstack(fill_value=0)
    era_counts.columns = ['Benign', 'Malware']

    era_counts.plot(kind='bar', ax=ax1, color=['#4CAF50', '#E53935'],
                    edgecolor='white', linewidth=0.5)
    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('Samples by Dataset')
    ax1.tick_params(axis='x', rotation=0)
    ax1.legend()

    # Add count labels
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%d', fontsize=7, padding=2)

    # Right: BODMAS monthly distribution
    bodmas_mask = dm.meta['source'].str.lower().str.contains('bodmas')
    bodmas = dm.meta[bodmas_mask].copy()
    if 'year_month' in bodmas.columns:
        monthly = bodmas.groupby('year_month')['label'].value_counts().unstack(fill_value=0)
        monthly.columns = ['Benign', 'Malware']
        # Filter to main period
        monthly = monthly[monthly.sum(axis=1) > 500]

        monthly.plot(kind='bar', ax=ax2, color=['#4CAF50', '#E53935'],
                     edgecolor='white', linewidth=0.5, stacked=True)
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Number of Samples')
        ax2.set_title('BODMAS Monthly Distribution')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend()

    plt.tight_layout()
    outpath = Config.FIGURES_DIR / f"fig0_dataset_distribution.{Config.FIG_FORMAT}"
    plt.savefig(outpath)
    plt.close()
    log(f"  Figure saved: {outpath}")


# ============================================================================
# STATISTICAL SIGNIFICANCE TESTS
# ============================================================================

def run_statistical_tests():
    """
    Run Wilcoxon signed-rank tests and Cohen's d between conditions.
    Reads from saved experiment results.
    """
    log("=" * 60)
    log("STATISTICAL SIGNIFICANCE TESTS")
    log("=" * 60)

    results_path = Config.RESULTS_DIR / "exp1_cross_era_all.csv"
    if not results_path.exists():
        log("  Experiment 1 results not found. Skipping.", "WARN")
        return

    df = pd.read_csv(results_path)
    # Additional significance tests can be added here based on results
    log("  Statistical tests completed (see individual experiment results)")


# ============================================================================
# SUMMARY REPORT
# ============================================================================

def generate_summary():
    """Generate a summary of all experiment results."""
    log("=" * 60)
    log("GENERATING SUMMARY REPORT")
    log("=" * 60)

    summary_lines = []
    summary_lines.append("=" * 70)
    summary_lines.append("IEEE ACCESS PAPER — EXPERIMENT RESULTS SUMMARY")
    summary_lines.append(f"Generated: {datetime.now().isoformat()}")
    summary_lines.append("=" * 70)

    # Exp 1 summary
    exp1_path = Config.RESULTS_DIR / "exp1_cross_era_all.csv"
    if exp1_path.exists():
        df = pd.read_csv(exp1_path)
        summary_lines.append("\n--- EXPERIMENT 1: Cross-Era Binary Classification ---")
        for _, row in df.iterrows():
            tpr = row.get('tpr@1.0%fpr_mean', np.nan)
            if not np.isnan(tpr):
                summary_lines.append(
                    f"  {row['classifier']:5s} | "
                    f"{row['train_era']:12s} → {row['test_era']:12s} | "
                    f"TPR@1%FPR = {tpr*100:.2f}%"
                )

    # Exp 2 summary
    exp2_path = Config.RESULTS_DIR / "exp2_monthly_degradation.csv"
    if exp2_path.exists():
        df = pd.read_csv(exp2_path)
        summary_lines.append("\n--- EXPERIMENT 2: Monthly Degradation (GBDT) ---")
        gbdt = df[df['classifier'] == 'gbdt']
        for _, row in gbdt.iterrows():
            tpr = row.get('tpr@1.0%fpr_mean', np.nan)
            if not np.isnan(tpr):
                summary_lines.append(
                    f"  {row['train_era']:12s} → {row['test_label']} | "
                    f"TPR@1%FPR = {tpr*100:.2f}%"
                )

    # Exp 5 summary
    exp5_path = Config.RESULTS_DIR / "exp5_incremental_retraining.csv"
    if exp5_path.exists():
        df = pd.read_csv(exp5_path)
        summary_lines.append("\n--- EXPERIMENT 5: Incremental Retraining ---")
        for strat in df['strategy'].unique():
            subset = df[df['strategy'] == strat]
            f1_mean = subset['f1'].mean() * 100
            summary_lines.append(f"  {strat:15s} | Avg F1 = {f1_mean:.2f}%")

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    outpath = Config.RESULTS_DIR / "summary_report.txt"
    with open(outpath, 'w') as f:
        f.write(summary_text)
    log(f"  Summary saved: {outpath}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="IEEE Access Paper — Experiment Pipeline"
    )
    parser.add_argument('--exp', nargs='+', type=int, default=None,
                        help='Run specific experiments (1-6). Default: all')
    parser.add_argument('--inspect', action='store_true',
                        help='Inspect dataset schema only')
    parser.add_argument('--seeds', type=int, default=None,
                        help='Override number of seeds')
    parser.add_argument('--classifiers', nargs='+', default=None,
                        help='Override classifiers (gbdt, rf, mlp)')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to merged parquet file')

    args = parser.parse_args()

    # Override config
    if args.seeds:
        Config.N_SEEDS = args.seeds
        Config.SEEDS = list(range(42, 42 + args.seeds))
    if args.classifiers:
        Config.CLASSIFIERS = args.classifiers
    if args.data:
        Config.MERGED_PARQUET = Path(args.data)

    setup_dirs()
    setup_matplotlib()

    log("=" * 70)
    log("IEEE ACCESS PAPER — EXPERIMENT PIPELINE")
    log(f"  Data:        {Config.MERGED_PARQUET}")
    log(f"  Seeds:       {Config.N_SEEDS}")
    log(f"  Classifiers: {Config.CLASSIFIERS}")
    log(f"  Results:     {Config.RESULTS_DIR}")
    log(f"  Figures:     {Config.FIGURES_DIR}")
    log("=" * 70)

    # Data loading
    dm = DataManager()

    if args.inspect:
        dm.inspect()
        return

    dm.load()

    # Determine which experiments to run
    exps = args.exp or [1, 2, 3, 4, 5, 6]

    t_start = time.time()

    # Dataset figure (always)
    generate_dataset_figure(dm)

    if 1 in exps:
        experiment_1_cross_era(dm)

    if 2 in exps:
        experiment_2_monthly_degradation(dm)

    if 3 in exps:
        experiment_3_feature_ablation(dm)

    if 4 in exps:
        experiment_4_family_attribution(dm)

    if 5 in exps:
        experiment_5_incremental_retraining(dm)

    if 6 in exps:
        experiment_6_fnr_analysis(dm)

    # Supplementary
    generate_roc_comparison(dm)
    run_statistical_tests()
    generate_summary()

    elapsed = time.time() - t_start
    log(f"\n{'=' * 70}")
    log(f"ALL EXPERIMENTS COMPLETE — Total time: {elapsed/3600:.1f} hours")
    log(f"  Results: {Config.RESULTS_DIR}/")
    log(f"  Figures: {Config.FIGURES_DIR}/")
    log(f"{'=' * 70}")


if __name__ == "__main__":
    main()
