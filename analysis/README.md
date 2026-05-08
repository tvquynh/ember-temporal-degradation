# Statistical analysis scripts

These scripts compute the auxiliary statistics reported alongside the main results:

| Script | Purpose |
|---|---|
| `compute_stats_corrections.py` | Holm-Bonferroni adjusted p-values for the 13-month family of paired Wilcoxon tests in Exp~4; paired Cohen's d_z effect sizes for Exp~4, Exp~8, and Exp~9. |
| `generate_temporal_histogram.py` | Reproduce the Figure showing distribution of `appeared` year per dataset (EMBER 2017 / 2018, BODMAS). |

Inputs: per-seed CSVs in `../results/exp{8,9}_*/seed_*.csv`.

Outputs: `../results/stats/*.csv` (corrections + effect sizes), `../manuscript_pass_a/figures/fig_temporal_coverage.pdf`.
