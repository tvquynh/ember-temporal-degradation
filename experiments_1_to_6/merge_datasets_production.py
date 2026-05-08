#!/usr/bin/env python3
"""
WORKING Merge Script - All dtype issues fixed
"""

import polars as pl
from pathlib import Path
import sys
import json
from datetime import datetime

def load_ember_dataset(data_dir, year):
    """Load EMBER."""
    
    print(f"\nLOADING EMBER {year}")
    
    if year == 2017:
        train_path = f"{data_dir}/ember_2017_2/ember2017_2_train.parquet"
        test_path = f"{data_dir}/ember_2017_2/ember2017_2_test.parquet"
    else:
        train_path = f"{data_dir}/ember_2018/ember2018_train.parquet"
        test_path = f"{data_dir}/ember_2018/ember2018_test.parquet"
    
    combined = pl.concat([
        pl.read_parquet(train_path),
        pl.read_parquet(test_path)
    ]).filter(pl.col("label") != -1)
    
    print(f"  Samples: {len(combined):,}")
    return combined


def load_bodmas_dataset(data_dir):
    """Load BODMAS."""
    
    print(f"\nLOADING BODMAS")
    bodmas = pl.read_parquet(f"{data_dir}/bodmas/bodmas.parquet")
    print(f"  Samples: {len(bodmas):,}")
    return bodmas


def align_schemas(ember2017, ember2018, bodmas):
    """Align schemas - all dtypes standardized."""
    
    print(f"\nALIGNING SCHEMAS")
    
    feature_cols = [f"feature_{i:04d}" for i in range(2381)]
    
    # Standard datetime dtype (microseconds, no timezone)
    DATETIME_DTYPE = pl.Datetime('us')
    
    # COLUMN ORDER
    COLUMN_ORDER = [
        "sha256", "label", "year", "source", "dataset_id",
        "temporal_order", "timestamp", "family", "category",
        "appeared_str", "avclass"
    ] + feature_cols
    
    # ===================================================================
    # EMBER 2017
    # ===================================================================
    ember2017_aligned = ember2017.select([
        "sha256",
        pl.col("label").cast(pl.Int32),
        *feature_cols
    ]).with_columns([
        pl.lit("2017").alias("year"),
        pl.lit("EMBER").alias("source"),
        pl.lit("EMBER_2017").alias("dataset_id"),
        pl.datetime(2017, 6, 15, 0, 0, 0).cast(DATETIME_DTYPE).alias("temporal_order"),
        pl.lit(None).cast(DATETIME_DTYPE).alias("timestamp"),
        pl.lit(None).cast(pl.Utf8).alias("family"),
        pl.lit(None).cast(pl.Utf8).alias("category"),
        pl.lit(None).cast(pl.Utf8).alias("appeared_str"),
        pl.lit(None).cast(pl.Utf8).alias("avclass"),
    ]).select(COLUMN_ORDER)
    
    # ===================================================================
    # EMBER 2018
    # ===================================================================
    ember2018_aligned = ember2018.select([
        "sha256",
        pl.col("label").cast(pl.Int32),
        *feature_cols
    ]).with_columns([
        pl.lit("2018").alias("year"),
        pl.lit("EMBER").alias("source"),
        pl.lit("EMBER_2018").alias("dataset_id"),
        pl.datetime(2018, 6, 15, 0, 0, 0).cast(DATETIME_DTYPE).alias("temporal_order"),
        pl.lit(None).cast(DATETIME_DTYPE).alias("timestamp"),
        pl.lit(None).cast(pl.Utf8).alias("family"),
        pl.lit(None).cast(pl.Utf8).alias("category"),
        pl.lit(None).cast(pl.Utf8).alias("appeared_str"),
        pl.lit(None).cast(pl.Utf8).alias("avclass"),
    ]).select(COLUMN_ORDER)
    
    # ===================================================================
    # BODMAS - cast timestamp to remove timezone and standardize
    # ===================================================================
    bodmas_aligned = bodmas.select([
        "sha256",
        pl.col("label").cast(pl.Int32),
        *feature_cols,
        pl.col("timestamp").dt.replace_time_zone(None).cast(DATETIME_DTYPE).alias("timestamp"),
        "family",
        "category"
    ]).with_columns([
        pl.col("timestamp").dt.year().cast(pl.Utf8).alias("year"),
        pl.lit("BODMAS").alias("source"),
        pl.lit("BODMAS").alias("dataset_id"),
        pl.col("timestamp").alias("temporal_order"),
        pl.lit(None).cast(pl.Utf8).alias("appeared_str"),
        pl.lit(None).cast(pl.Utf8).alias("avclass"),
    ]).select(COLUMN_ORDER)
    
    # Verify
    assert ember2017_aligned.columns == ember2018_aligned.columns == bodmas_aligned.columns
    print(f"  ✅ Schemas aligned: {len(COLUMN_ORDER)} columns")
    print(f"  EMBER 2017: {len(ember2017_aligned):,}")
    print(f"  EMBER 2018: {len(ember2018_aligned):,}")
    print(f"  BODMAS:     {len(bodmas_aligned):,}")
    
    return ember2017_aligned, ember2018_aligned, bodmas_aligned


def merge_and_validate(ember2017, ember2018, bodmas):
    """Merge and validate."""
    
    print(f"\nMERGING")
    merged = pl.concat([ember2017, ember2018, bodmas]).sort("temporal_order")
    print(f"  ✅ Total: {len(merged):,} samples")
    
    print(f"\nVALIDATION")
    
    # Counts
    counts = merged.group_by(["source", "year"]).len().sort(["source", "year"])
    for row in counts.iter_rows():
        print(f"  {row[0]} {row[1]}: {row[2]:,}")
    
    # Labels
    labels = merged.group_by("label").len()
    for row in labels.iter_rows():
        name = "benign" if row[0] == 0 else "malware"
        print(f"  {name}: {row[1]:,}")
    
    print(f"  ✅ Validation complete")
    
    return merged


def main():
    DATA_DIR = "<data-root>"
    OUTPUT_FILE = "../data/merged_temporal_dataset.parquet"
    
    print("="*70)
    print("IEEE ACCESS PAPER - DATASET MERGE")
    print("="*70)
    
    # Load
    ember2017 = load_ember_dataset(DATA_DIR, 2017)
    ember2018 = load_ember_dataset(DATA_DIR, 2018)
    bodmas = load_bodmas_dataset(DATA_DIR)
    
    # Align
    e17, e18, bod = align_schemas(ember2017, ember2018, bodmas)
    
    # Merge
    merged = merge_and_validate(e17, e18, bod)
    
    # Save
    print(f"\nSAVING: {OUTPUT_FILE}")
    merged.write_parquet(OUTPUT_FILE, compression="snappy")
    
    size_mb = Path(OUTPUT_FILE).stat().st_size / (1024 * 1024)
    print(f"✅ Saved: {size_mb:.1f} MB")
    
    # Summary
    summary = {
        "created_at": datetime.now().isoformat(),
        "total_samples": len(merged),
        "total_features": 2381
    }
    
    with open("../data/merge_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✅ COMPLETE: {len(merged):,} samples ready")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
