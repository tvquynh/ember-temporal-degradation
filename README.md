# Temporal Degradation in ML-Based Malware Detection

Reproducibility artifact for the paper:

> **Temporal Degradation in Machine Learning-Based Malware Detection: A Multi-Dataset, Multi-Year Empirical Study**
> Trong-Thua Huynh, De-Thu Huynh, Minh Thang Nguyen, Van-Quynh Trinh (corresponding)
> **IEEE Access**, 2026. DOI: [10.1109/ACCESS.2026.3706553](https://doi.org/10.1109/ACCESS.2026.3706553)

This repository contains the experiment scripts, data-cleaning pipeline, statistical analysis, and figure-generation code used to produce all numerical results and figures in the paper.

[![DOI (article)](https://img.shields.io/badge/DOI-10.1109%2FACCESS.2026.3706553-blue)](https://doi.org/10.1109/ACCESS.2026.3706553)
[![DOI (software)](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20279488-blue)](https://doi.org/10.5281/zenodo.20279488)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Datasets

The paper combines three publicly available PE malware datasets, restricted to the EMBER v2 feature space (2,381 dimensions per sample):

| Dataset      | Source                                                       | Period      | Samples used |
|--------------|--------------------------------------------------------------|-------------|--------------|
| EMBER 2017v2 | https://github.com/elastic/ember                             | 2017        | 800,000      |
| EMBER 2018   | https://github.com/elastic/ember                             | 2018        | 750,000      |
| BODMAS       | https://whyisyoung.github.io/BODMAS/                         | 2019–2020   | 134,338      |
| **Total**    |                                                              | **2017–2020** | **1,684,338** |

## What's in this repository

| Folder / file              | Purpose                                                         |
|----------------------------|-----------------------------------------------------------------|
| `experiments_1_to_6/`      | Experiments 1--6 — data cleaning, merge, cross-era, monthly drift, family FNR, feature-group ablation, incremental retraining |
| `common/`                  | Shared modules: data loader, classifier factory, metrics, sampling utilities, project seeds |
| `exp7_aut/`                | Experiment 7 — cumulative AUT post-hoc analysis                 |
| `exp8_drift_detector/`     | Experiment 8 — drift-triggered retraining (ADWIN, DDM)          |
| `exp9_active_learning/`    | Experiment 9 — uncertainty sampling vs. random sampling         |
| `strict_temporal_rerun/`   | Strict appeared-year re-run of Experiments 1 and 6              |
| `analysis/`                | Statistical analysis (Holm correction, Cohen's d) and figure helpers |
| `tests/`                   | Unit tests (data loader / metrics / sampling)                   |
| `generate_figures.py`      | Re-generate the monthly / bar / cumulative-AUT figures          |
| `requirements.txt`         | Pinned Python dependencies                                      |
| `RUNBOOK.md`               | Operational guide (pre-flight, smoke, full run, verification)   |
| `CITATION.cff`             | Machine-readable citation (CFF 1.2.0)                           |
| `.zenodo.json`             | Zenodo deposition metadata (auto-synced on each GitHub release) |

## Reproducing the experiments

### Environment

- Python 3.10
- Linux x86_64 with sufficient RAM (≥ 64 GB recommended for the full BODMAS sweep)
- See `requirements.txt` for pinned package versions (LightGBM 4.6, scikit-learn 1.7, river 0.22, pandas, pyarrow, imbalanced-learn 0.14, ...)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Data layout

Place the three Parquet files (or raw EMBER JSONL → Parquet via `ember.create_vectorized_features`) at:

```
<data-root>/
├── ember2017_2/{ember2017_2_train.parquet, ember2017_2_test.parquet}
├── ember2018/{ember2018_train.parquet,    ember2018_test.parquet}
└── bodmas/bodmas.parquet
```

Set the data root in `common/data_loader.py` or via the `DATA_ROOT` environment variable.

### Quick start

```bash
# Smoke test (1 seed, ~5 minutes)
bash exp8_drift_detector/run.sh --smoke
bash exp9_active_learning/run.sh --smoke

# Full run (10 seeds, several hours on a 60-core node)
bash exp8_drift_detector/run.sh
bash exp9_active_learning/run.sh

# Post-hoc AUT analysis (no GPU; computes from manuscript Tables 5/6/7)
python exp7_aut/compute_aut.py

# Aggregate per-condition summaries
python exp8_drift_detector/aggregate.py --validate
python exp9_active_learning/aggregate.py --validate

# Regenerate all figures
python generate_figures.py
```

See `RUNBOOK.md` for detailed step-by-step instructions including monitoring, verification, and red-flag checklist.

### Random seeds

All experiments use a fixed list of ten seeds:

```python
SEEDS = [42, 123, 456, 789, 1011, 2026, 3141, 4242, 5555, 6789]
```

## Citation

If you use this code in your research, please cite **both** the paper and this software release.

**Paper (preferred citation):**

```bibtex
@article{huynh2026temporal,
  author  = {Huynh, Trong-Thua and Huynh, De-Thu and Nguyen, Minh Thang and Trinh, Van-Quynh},
  title   = {Temporal Degradation in Machine Learning-Based Malware Detection: A Multi-Dataset, Multi-Year Empirical Study},
  journal = {IEEE Access},
  year    = {2026},
  doi     = {10.1109/ACCESS.2026.3706553},
  url     = {https://doi.org/10.1109/ACCESS.2026.3706553}
}
```

**Software release:**

```bibtex
@software{huynh2026temporal_software,
  author  = {Huynh, Trong-Thua and Huynh, De-Thu and Nguyen, Minh Thang and Trinh, Van-Quynh},
  title   = {{ember-temporal-degradation}: Reproducibility artifact for ``Temporal Degradation in Machine Learning-Based Malware Detection''},
  year    = {2026},
  version = {v1.5.2},
  doi     = {10.5281/zenodo.20279488},
  url     = {https://github.com/tvquynh/ember-temporal-degradation}
}
```

See `CITATION.cff` for a machine-readable version of these citations.

## License

This code is released under the [MIT License](LICENSE). The underlying datasets retain their original licences (see the dataset URLs above).

## Contact

Corresponding author: **Van-Quynh Trinh** — `quynhtv@ptithcm.edu.vn`
