# Experiments 1–6 — original pipeline

This folder contains the data-cleaning, merge, and analysis scripts for the six experiments reported as **Experiment 1 through Experiment 6** in the paper:

| Paper experiment | Code function in `run_experiments.py`     |
|------------------|------------------------------------------ |
| Exp 1 — In-era baselines                          | `experiment_1_cross_era` (in-era diagonal) |
| Exp 2 — Cross-era degradation (EMBER↔EMBER + EMBER→BODMAS monthly) | `experiment_1_cross_era` (off-diagonal) + `experiment_2_monthly_degradation` |
| Exp 3 — BODMAS internal monthly drift             | `experiment_2_monthly_degradation` (BODMAS-only mode) |
| Exp 4 — Incremental retraining (1%/month)         | `experiment_5_incremental_retraining`     |
| Exp 5 — Family-level FNR (existing vs unseen)     | `experiment_4_family_attribution` + `experiment_6_fnr_analysis` |
| Exp 6 — Feature-group drift (zero-ablation)       | `experiment_3_feature_ablation`           |

(Note: the `experiment_N_*` function numbers in the code are an internal pipeline ordering and do not match the paper's Exp 1–6 numbering. The paper's Exp 7, 8, 9 — cumulative AUT, drift-triggered retraining, and active learning — live in their own folders at the repository root.)

## Files

| File                              | Purpose |
|-----------------------------------|---------|
| `data_cleaning.py`                | Drop unlabeled samples, hash-deduplicate across datasets, unify column schema, verify NaN/Inf |
| `merge_datasets_production.py`    | Combine the three cleaned datasets into one EMBER-v2-compatible Parquet corpus |
| `run_experiments.py`              | Run any subset of the six experiments and save tables/figures |

## Quick start

```bash
# 1. Clean & merge raw datasets
python data_cleaning.py --data_dir <data-root> --output_dir <data-root>/cleaned
python merge_datasets_production.py

# 2. Run all six experiments (10 seeds; several hours on a 60-core node)
python run_experiments.py

# 3. Run a specific experiment only
python run_experiments.py --exp 1     # cross-era
python run_experiments.py --exp 2     # monthly degradation
python run_experiments.py --exp 3     # feature ablation
python run_experiments.py --exp 4     # family attribution
python run_experiments.py --exp 5     # incremental retraining
python run_experiments.py --exp 6     # FNR analysis

# 4. Inspect data without running
python run_experiments.py --inspect
```

Outputs land in `results/` (CSV tables) and `figures/` (PDF) at the working directory.

## Reproducibility notes

- Random seeds: `[42, 123, 456, 789, 1011, 2026, 3141, 4242, 5555, 6789]` — same as Experiments 7–9 in the parent folders.
- LightGBM version 4.6 with `is_unbalance=True`; Random Forest 200 trees, depth 30, balanced class weights; MLP (256, 128) with StandardScaler + RandomOverSampler.
- Temporal integrity: training data always precedes test data chronologically.

See the parent `RUNBOOK.md` for the full operational guide.
