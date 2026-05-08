# RUNBOOK — IEEE Access Resubmission Experiments

## 1. Pre-flight checklist

- [ ] the compute server reachable: `ssh <your-server> "hostname"`
- [ ] Data synced: `ssh <your-server> "ls <server-data-path>{ember2017_2,ember2018,bodmas}/*.parquet"`
- [ ] Venv active: `source <server-venv>/bin/activate`
- [ ] river installed: `python -c "from river.drift import ADWIN; from river.drift.binary import DDM"`
- [ ] Code synced the compute server: `<server-code-path>/`
- [ ] Disk free >100GB on a 60-core server (`df -h /home`)

## 2. Smoke test (1 seed, 1 classifier, 1 month) — <5 minutes

```bash
ssh <your-server>
cd <server-code-path>
source <server-venv>/bin/activate
python exp8_drift_detector/run_seed.py --seed 42 --classifier lightgbm --smoke
python exp9_active_learning/run_seed.py --seed 42 --smoke
```

Smoke artifacts written to `results/smoke/`. Verify:
- F1 in [0.85, 1.00]
- No NaN/Inf
- All 4 conditions (Exp 8) or 2 conditions (Exp 9) produced

## 3. Full runs

### Exp 8 — drift detector (4 conditions × 3 clf × 13 months × 10 seeds)

```bash
bash exp8_drift_detector/run.sh
```

Pattern: 5 seeds parallel × 2 waves (60 cores / 12 = 5 parallel; n_jobs=12 per process).
Wall clock estimate: ~3 days.
Output: `results/exp8_drift_detector/seed_<S>_<clf>.csv`

### Exp 9 — active learning (2 conditions × LGBM only × 13 months × 10 seeds)

```bash
bash exp9_active_learning/run.sh
```

Wall clock estimate: ~1 day.
Output: `results/exp9_active_learning/seed_<S>.csv`

### Exp 7 — AUT post-hoc (no training, ~5 min)

```bash
python exp7_aut/compute_aut.py
```

Reads existing manuscript Tables 5/6/7 monthly F1 + applies trapezoidal integration. Output: `results/exp7_aut.csv`.

## 4. Monitoring

**the user Em  check khi the user Quick status:
```bash
ssh <your-server> "ls results/exp8_drift_detector/*.csv 2>/dev/null | wc -l"  # expect 30 = 10 seeds × 3 clf
ssh <your-server> "ls results/exp9_active_learning/*.csv 2>/dev/null | wc -l"  # expect 10 seeds
```

## 5. Verification

```bash
ssh <your-server> "cd <server-code-path> && python -m pytest tests/ -v"
ssh <your-server> "cd <server-code-path> && python exp8_drift_detector/aggregate.py --validate"
ssh <your-server> "cd <server-code-path> && python exp9_active_learning/aggregate.py --validate"
```

Validation checks:
- All 30 Exp 8 CSVs present, each with 13 month rows × 4 conditions
- All 10 Exp 9 CSVs present, each with 13 months × 2 conditions
- F1 values in [0, 1]; no NaN
- Per-seed AUT computed; bootstrap 95% CI < 1pp width

## 6. Analysis (red flags)

**Sanity checks** before integrating into manuscript:

| Red flag | Threshold | Action |
|---|---|---|
| AUT(static) < AUT(fixed 1%/mo) | always | OK — confirms paper's existing finding |
| AUT(ADWIN) > AUT(fixed 1%/mo) | expected if R1 right | If false → strengthens paper's defense of fixed schedule |
| AUT(uncertainty) > AUT(random) | expected per Pendlebury 2024 | If false → notable + explainable finding |
| Per-seed std > 1pp | unexpected | Investigate seed-specific issues |
| F1 < 0.85 any seed | unexpected | Likely data leakage or env mismatch |

## 7. Hand-off

When all 3 experiments done:
- Aggregate `results/aggregated_summary.json` produced
- Pass to manuscript integration (Tier A writing): tables for Section 4 + Section 5 discussion

## 8. Backup

Original manuscript zip preserved at `E:\IEEE_Access\Temporal_Degradation_in_Machine_Learning_Based_Malware_Detection__A_Multi_Dataset__Multi_Year_Empirical_Study.zip`. Do not modify.

Original paper code (if needed for reference): `<local-path>`.

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: river` | venv not activated | `source <server-venv>/bin/activate` |
| OOM kill | 5 parallel × full data | Reduce to 3 parallel, increase n_jobs to 20 |
| ADWIN never triggers | Stream too short | Lower `delta` parameter (default 0.002 → 0.01) |
| Per-seed F1 huge variance | Non-deterministic LGBM | Verify `deterministic=True` + `force_row_wise=True` set |
| AUT computation crash on Exp 4 row | Missing month value | Check `exp7_aut/compute_aut.py` input table format |
