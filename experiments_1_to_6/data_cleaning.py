#!/usr/bin/env python3
"""
=============================================================================
Data Cleaning Script for EMBER + BODMAS
=============================================================================
Based on exploration results. Steps:
  1. Load all 5 parquet files
  2. Remove unlabeled samples (label=-1)
  3. Cross-dataset hash deduplication (report + remove)
  4. Align label dtype to int32
  5. Verify feature compatibility
  6. Save cleaned datasets

Usage:
  python data_cleaning.py \
      --data_dir <data-root> \
      --output_dir <data-root>/cleaned \
      > cleaning_report.txt 2>&1
=============================================================================
"""

import argparse
import os
import sys
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd

SEP = "=" * 80
FEATURE_COLS = [f"feature_{i:04d}" for i in range(2381)]


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ============================================================================
# STEP 1: LOAD
# ============================================================================
def load_all(data_dir):
    """Load all parquet files, return dict of DataFrames."""
    files = {
        'ember2017v2_train': 'ember_2017_2/ember2017_2_train.parquet',
        'ember2017v2_test':  'ember_2017_2/ember2017_2_test.parquet',
        'ember2018_train':   'ember_2018/ember2018_train.parquet',
        'ember2018_test':    'ember_2018/ember2018_test.parquet',
        'bodmas':            'bodmas/bodmas.parquet',
    }
    datasets = {}
    for name, rel_path in files.items():
        path = os.path.join(data_dir, rel_path)
        log(f"Loading {name} from {path}...")
        df = pd.read_parquet(path)
        log(f"  {name}: {df.shape[0]:,} rows × {df.shape[1]} cols")
        datasets[name] = df
    return datasets


# ============================================================================
# STEP 2: REMOVE UNLABELED (label=-1)
# ============================================================================
def remove_unlabeled(datasets):
    """Remove samples with label=-1."""
    print(f"\n{SEP}")
    print("  STEP 2: REMOVE UNLABELED SAMPLES (label=-1)")
    print(SEP)

    for name, df in datasets.items():
        before = len(df)
        unlabeled = (df['label'] == -1).sum()
        if unlabeled > 0:
            datasets[name] = df[df['label'] != -1].reset_index(drop=True)
            after = len(datasets[name])
            log(f"  {name}: {before:,} → {after:,} "
                f"(removed {unlabeled:,} unlabeled, "
                f"{unlabeled/before*100:.1f}%)")
        else:
            log(f"  {name}: {before:,} → no unlabeled found ✓")

    return datasets


# ============================================================================
# STEP 3: CROSS-DATASET HASH DEDUPLICATION
# ============================================================================
def cross_dataset_hash_analysis(datasets):
    """Analyze and remove cross-dataset hash duplicates."""
    print(f"\n{SEP}")
    print("  STEP 3: CROSS-DATASET HASH DEDUPLICATION")
    print(SEP)

    # 3a. Collect all hashes
    hash_to_datasets = {}
    for name, df in datasets.items():
        for h in df['sha256']:
            hash_to_datasets.setdefault(h, []).append(name)

    # 3b. Find duplicates
    multi = {h: ds for h, ds in hash_to_datasets.items() if len(ds) > 1}
    log(f"\n  Total unique hashes across all datasets: "
        f"{len(hash_to_datasets):,}")
    log(f"  Hashes appearing in 2+ datasets: {len(multi):,}")

    if not multi:
        log("  ✓ No cross-dataset duplicates found!")
        return datasets

    # 3c. Breakdown by dataset pair
    combo_counts = Counter(tuple(sorted(set(ds))) for ds in multi.values())
    log(f"\n  Overlap breakdown:")
    for combo, cnt in combo_counts.most_common():
        log(f"    {' ∩ '.join(combo)}: {cnt:,}")

    # 3d. Detailed pairwise analysis
    names = list(datasets.keys())
    print(f"\n  Pairwise overlap matrix:")
    print(f"  {'':>25}", end="")
    for n in names:
        print(f"  {n[:12]:>12}", end="")
    print()

    hash_sets = {n: set(df['sha256']) for n, df in datasets.items()}
    for n1 in names:
        print(f"  {n1:>25}", end="")
        for n2 in names:
            if n1 == n2:
                print(f"  {len(hash_sets[n1]):>12,}", end="")
            else:
                overlap = len(hash_sets[n1] & hash_sets[n2])
                print(f"  {overlap:>12,}", end="")
        print()

    # 3e. Dedup strategy:
    #   - Between EMBER train and test of same year: keep in test, remove from train
    #   - Between EMBER 2017 and 2018: keep in 2017 (older), remove from 2018
    #   - Between EMBER and BODMAS: keep in EMBER, remove from BODMAS
    #   - Within EMBER (train/test same year): keep in test
    print(f"\n  Dedup strategy:")
    print(f"    Priority: test > train; older > newer; EMBER > BODMAS")
    print(f"    Keep in higher-priority dataset, remove from lower-priority")

    # Priority order (highest = keep)
    priority = [
        'ember2017v2_test',   # highest - test sets are sacred
        'ember2018_test',
        'ember2017v2_train',  # older train
        'ember2018_train',    # newer train
        'bodmas',             # lowest - we remove from here
    ]

    removed_counts = {n: 0 for n in names}
    hashes_to_remove = {n: set() for n in names}

    for h, ds_list in multi.items():
        # Find which datasets contain this hash
        containing = set(ds_list)
        # Keep in highest-priority dataset, remove from all others
        for p in priority:
            if p in containing:
                # Keep in p, remove from all others
                for other in containing:
                    if other != p:
                        hashes_to_remove[other].add(h)
                break

    # Apply removal
    print(f"\n  Samples to remove per dataset:")
    for name in names:
        n_remove = len(hashes_to_remove[name])
        if n_remove > 0:
            before = len(datasets[name])
            mask = ~datasets[name]['sha256'].isin(hashes_to_remove[name])
            datasets[name] = datasets[name][mask].reset_index(drop=True)
            after = len(datasets[name])
            log(f"    {name}: removed {n_remove:,} duplicates "
                f"({before:,} → {after:,})")
        else:
            log(f"    {name}: 0 duplicates to remove ✓")

    # Verify no more cross-dataset duplicates
    all_hashes_after = []
    for name, df in datasets.items():
        all_hashes_after.extend(df['sha256'].tolist())
    total = len(all_hashes_after)
    unique = len(set(all_hashes_after))
    remaining_dups = total - unique
    log(f"\n  After dedup: {total:,} total, {unique:,} unique, "
        f"{remaining_dups:,} remaining dups")
    if remaining_dups > 0:
        log(f"  ⚠ WARNING: {remaining_dups} duplicates remain!")
    else:
        log(f"  ✓ All cross-dataset duplicates removed!")

    return datasets


# ============================================================================
# STEP 4: UNIFY COLUMN NAMES & ALIGN DTYPES
# ============================================================================
def unify_columns_and_dtypes(datasets):
    """
    Unify column names across EMBER and BODMAS:
      EMBER 'appeared'  → 'timestamp'  (parse "2017-11" → datetime)
      EMBER 'avclass'   → 'family'
      EMBER (missing)   → 'category' = None
      BODMAS already has: timestamp, family, category
    Also align label to int8.
    """
    print(f"\n{SEP}")
    print("  STEP 4: UNIFY COLUMN NAMES & ALIGN DTYPES")
    print(SEP)

    for name, df in datasets.items():
        is_ember = 'ember' in name

        if is_ember:
            # --- appeared → timestamp ---
            if 'appeared' in df.columns:
                # EMBER appeared format: "2017-11", "2006-12", etc.
                # Parse to datetime (first day of that month)
                df['timestamp'] = pd.to_datetime(
                    df['appeared'] + '-01', format='%Y-%m-%d',
                    errors='coerce', utc=True)
                n_parsed = df['timestamp'].notna().sum()
                n_failed = df['timestamp'].isna().sum()
                log(f"  {name}: 'appeared' → 'timestamp' "
                    f"(parsed: {n_parsed:,}, failed: {n_failed:,})")
                # Show appeared date range
                valid_ts = df['timestamp'].dropna()
                if len(valid_ts) > 0:
                    log(f"    Range: {valid_ts.min()} → {valid_ts.max()}")
                df.drop(columns=['appeared'], inplace=True)
            else:
                log(f"  {name}: 'appeared' not found, skipping")

            # --- avclass → family ---
            if 'avclass' in df.columns:
                df.rename(columns={'avclass': 'family'}, inplace=True)
                # Clean: 'na', '', empty → None
                df['family'] = df['family'].replace(
                    {'na': None, '': None, 'nan': None})
                n_with_family = df['family'].notna().sum()
                log(f"  {name}: 'avclass' → 'family' "
                    f"(non-null: {n_with_family:,})")
            else:
                df['family'] = None
                log(f"  {name}: 'family' column created (all None)")

            # --- category: EMBER doesn't have this ---
            df['category'] = None
            log(f"  {name}: 'category' column created (all None)")

        else:
            # BODMAS - already has timestamp, family, category
            # Just verify and clean family
            if 'family' in df.columns:
                df['family'] = df['family'].replace(
                    {'': None, 'nan': None, 'None': None})
                # Also handle actual None/NaN
                n_with_family = df['family'].notna().sum()
                log(f"  {name}: 'family' cleaned "
                    f"(non-null: {n_with_family:,})")

            if 'timestamp' in df.columns:
                # Ensure UTC timezone
                if df['timestamp'].dt.tz is None:
                    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
                ts = df['timestamp'].dropna()
                log(f"  {name}: timestamp range: {ts.min()} → {ts.max()}")

            if 'category' in df.columns:
                n_with_cat = df['category'].notna().sum()
                cats = df['category'].dropna().unique()
                log(f"  {name}: categories ({n_with_cat:,} non-null): "
                    f"{sorted(cats)}")

        # --- Label to int8 ---
        old_dtype = df['label'].dtype
        df['label'] = df['label'].astype(np.int8)
        log(f"  {name}: label {old_dtype} → int8")

        # --- Verify features are float32 ---
        feature_dtypes = df[FEATURE_COLS].dtypes.unique()
        log(f"  {name}: feature dtypes = {feature_dtypes}")

        datasets[name] = df

    # Show unified schema
    print(f"\n  Unified column schema:")
    print(f"    sha256     : str (PE file hash)")
    print(f"    label      : int8 (0=benign, 1=malware)")
    print(f"    timestamp  : datetime64[ns, UTC] (first seen)")
    print(f"    family     : str or None (malware family name)")
    print(f"    category   : str or None (trojan/worm/... BODMAS only)")
    print(f"    feature_*  : float32 × 2381 (EMBER-format PE features)")

    return datasets


# ============================================================================
# STEP 5: VERIFY FEATURE COMPATIBILITY
# ============================================================================
def verify_features(datasets):
    """Verify all datasets have compatible features."""
    print(f"\n{SEP}")
    print("  STEP 5: VERIFY FEATURE COMPATIBILITY")
    print(SEP)

    all_ok = True
    for name, df in datasets.items():
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            log(f"  ⚠ {name}: missing {len(missing)} features!")
            all_ok = False
        else:
            # Check for NaN/Inf
            feat_data = df[FEATURE_COLS].values
            has_nan = np.isnan(feat_data).any()
            has_inf = np.isinf(feat_data).any()
            log(f"  {name}: 2381 features ✓ | "
                f"NaN={has_nan} | Inf={has_inf}")
            if has_nan or has_inf:
                all_ok = False
                nan_count = np.isnan(feat_data).sum()
                inf_count = np.isinf(feat_data).sum()
                log(f"    NaN count: {nan_count:,}, Inf count: {inf_count:,}")

    if all_ok:
        log(f"\n  ✓ All features compatible and clean!")
    return all_ok


# ============================================================================
# STEP 6: FINAL SUMMARY & SAVE
# ============================================================================
def save_cleaned(datasets, output_dir):
    """Save cleaned datasets and print summary."""
    print(f"\n{SEP}")
    print("  STEP 6: SAVE CLEANED DATASETS")
    print(SEP)

    os.makedirs(output_dir, exist_ok=True)

    # Unified column order for all datasets
    meta_cols = ['sha256', 'label', 'timestamp', 'family', 'category']
    keep_cols = meta_cols + FEATURE_COLS

    total_samples = 0
    for name, df in datasets.items():
        cols = [c for c in keep_cols if c in df.columns]

        out_path = os.path.join(output_dir, f"{name}_cleaned.parquet")
        df[cols].to_parquet(out_path, index=False, compression='snappy')
        size_mb = os.path.getsize(out_path) / 1e6
        total_samples += len(df)
        log(f"  Saved {name}: {len(df):,} rows, {len(cols)} cols, "
            f"{size_mb:.1f} MB → {out_path}")

    # Final summary table
    print(f"\n{SEP}")
    print("  FINAL DATASET SUMMARY")
    print(SEP)
    print(f"\n  {'Dataset':<25} {'Rows':>10} {'Benign':>10} "
          f"{'Malware':>10} {'Mal%':>6}")
    print(f"  {'-'*63}")
    for name, df in datasets.items():
        n_ben = (df['label'] == 0).sum()
        n_mal = (df['label'] == 1).sum()
        pct = n_mal / len(df) * 100
        print(f"  {name:<25} {len(df):>10,} {n_ben:>10,} "
              f"{n_mal:>10,} {pct:>5.1f}%")
    print(f"  {'-'*63}")
    print(f"  {'TOTAL':<25} {total_samples:>10,}")

    # Timeline overview
    print(f"\n  Timeline:")
    print(f"    EMBER 2017v2: samples from 2017 (train+test)")
    print(f"    EMBER 2018:   samples from 2018 (train+test)")
    print(f"    BODMAS:       Aug 2019 – Sep 2020 (timestamped)")

    # Family stats for all datasets
    print(f"\n  Family info per dataset:")
    for name, df in datasets.items():
        if 'family' in df.columns:
            mal = df[df['label'] == 1]
            families = mal['family'].dropna()
            families = families[families.str.strip() != '']
            if len(families) > 0:
                n_unique = families.nunique()
                top3 = families.value_counts().head(3).to_dict()
                coverage = len(families) / len(mal) * 100
                print(f"    {name}: {n_unique} families, "
                      f"{coverage:.0f}% malware labeled, top3={top3}")
            else:
                print(f"    {name}: no family labels")
        else:
            print(f"    {name}: no family column")

    print(f"\n  Output directory: {output_dir}")
    print(f"\n✅ Cleaning complete!")


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Clean EMBER + BODMAS datasets")
    parser.add_argument('--data_dir', type=str,
                       default='<data-root>')
    parser.add_argument('--output_dir', type=str,
                       default='<data-root>/cleaned')
    args = parser.parse_args()

    print(SEP)
    print("  DATA CLEANING: EMBER + BODMAS")
    print(f"  Input:  {args.data_dir}")
    print(f"  Output: {args.output_dir}")
    print(SEP)

    # Step 1: Load
    log("STEP 1: Loading datasets...")
    datasets = load_all(args.data_dir)

    # Step 2: Remove unlabeled
    datasets = remove_unlabeled(datasets)

    # Step 3: Cross-dataset hash dedup
    datasets = cross_dataset_hash_analysis(datasets)

    # Step 4: Unify columns & align dtypes
    datasets = unify_columns_and_dtypes(datasets)

    # Step 5: Verify features
    verify_features(datasets)

    # Step 6: Save
    save_cleaned(datasets, args.output_dir)


if __name__ == '__main__':
    main()
