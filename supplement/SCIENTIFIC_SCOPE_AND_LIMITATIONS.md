# DrugShift-UQ — Incremental strengthening, not a new primary experiment

**Purpose:** Improve empirical rigor while preserving all previously completed experiments and any earlier results. The add-on **does not** retrain the 22 original jobs, modify original source files, change split definitions, or replace the existing reports.

## What is added

1. **Matched-data conventional baselines:** train Ridge, Random Forest, and histogram gradient boosting on **all available training pairs**, using the *same per-split training-fitted features and exact model settings as the original baseline estimators*. Runs: 4 split protocols × 3 seeds × 3 models = **36 new baseline fits**. The earlier baselines sampled at most 18,000 training pairs; comparisons to those baselines therefore had a training-data imbalance.
2. **Repeat the original architecture ablation variants:** original ablations were seed 42 only for cold-drug and double-cold. The exact variants `no_gate`, `no_bilinear`, `no_cell`, `no_drug`, and `early_fusion` are added for seeds 123 and 2026 in both cold regimes: **20 new jobs**, each with the original two-member, up-to-24-epoch configuration. The original seed-42 ablations are reused, never retrained.
3. **Saved-prediction analyses, zero extra training:** paired entity-cluster bootstrap RMSE difference summaries, repeated-ablation variability, conditional interval coverage by held-out entity, conformal interval scoring and tradeoff plots.

## Nonnegotiable reproducibility checks

- Require 12 intact original full-job `COMPLETE.json` markers with correct file hashes.
- Require the original processed response, gene and drug feature files to match all SHA-256 hashes in the run manifest.
- Require the original nine code files to match the run manifest. If they have been altered, the add-on stops rather than silently comparing different preprocessing or splits.
- Before fitting each baseline, recompute and verify all row-index, cell-ID, drug-ID and train/validation/calibration/test split assignments against that original job's `split_manifest.csv`.
- Before accepting new baseline predictions, compare test row indices and targets with the original saved predictions.
- Fit scalers and select features using training entities only (as in the original pipeline).
- Store each new baseline's completed predictions and metrics in a hash-validated checkpoint folder. Rerun skips completed fits. Each new neural ablation reuses the original epoch-level checkpointing.

## Scientific interpretation: what this does **not** establish

- No new clinical/patient dataset; cancer cell lines are not patients.
- Three split seeds do not represent independent cohorts and cannot alone establish significance or generalizability.
- Cluster-bootstrap intervals are *descriptive*, conditional on each held-out test split, and use a **single clustering axis**: cell for cold-cell; drug otherwise. The double-cold case is not a fully two-way-cluster bootstrap.
- Using the same training data does not guarantee a **matched hyperparameter search or computational budget**. Comparisons should be described as equal-training-data rather than equal-compute.
- Adding ablation seeds does not guarantee the full model is best; an ablation can outperform the full model and should be reported.
- Conformal marginal guarantees require exchangeability; the cold-shift evaluations are empirical diagnostics, not guaranteed coverage under shift.
- A named cross-gated architecture is a testable proposal, **not proof of novelty or superior performance**. Check prior work and make claims based on results.
- New analyses are *exploratory* extensions executed after observing the first 22-job results. Do not claim they were pre-registered or confirmatory.

## Outputs

`results/STRENGTHENED/<original run ID>/` contains:

- `STRENGTHENED_REPORT.html` — report, honest caveats, figures
- `figures/` — 450-DPI PNG plus vector PDF
- `tables/` — per-seed CSV results, summary CSV, LaTeX tables
- `matched_baselines/` — one auditable, checkpointed job per split/seed/estimator
- `STRENGTHENING_MANIFEST.json` — input identity and hashes

The 20 ablation jobs are written to **previously unused** paths under `results/runs/<original run ID>/jobs/` so the original resumable trainer can be called without refactoring it; existing jobs are not overwritten.

## Why one click still needs computation

These 56 new fits/ablations cannot be inferred from the original 22-job results. The add-on submits **all required supplementary jobs in one uninterrupted/resumable batch**. It is not a shortcut that could invent the missing evidence.
