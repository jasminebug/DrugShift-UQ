# DrugShift-UQ preregistered-style experimental protocol

**Scope:** In-silico anticancer drug-sensitivity regression in cancer cell lines, not prediction of clinical patient benefit. All protocol decisions are recorded in `config.json` before running real experiments.

## Data provenance, licensing and target

- DrEvalPy repository: https://github.com/daisybio/drevalpy
- Pinned record: https://zenodo.org/records/21807175 (GDSC2.zip; MD5 49dc16929c6baa2b5d75c508429fa11f).
- Preprocessing provenance: https://github.com/daisybio/preprocess_drp_data
- The exact prediction endpoint is `LN_IC50_curvecurator`. The pipeline will **stop**, not switch to a different response, if this exact field is missing.
- Drug IDs are PubChem IDs, cell lines are identified by cell_line_name or cell_line_id, dependent on the official upstream schema.
- All comparisons are made against the *same target and test rows for a given split*. Different regimes have different test cohorts and should not be interpreted as a paired comparison of exactly the same observations.
- Check the source dataset license and citation conditions before sharing any raw files. Raw data are excluded from the output ZIP and Git.

## Leakage prevention

- Collapse repeated drug–cell measurements by median **before** random splitting. The original replicate count is logged.
- Four groups: model-fitting train (65%), early-stopping validation (10%), split-conformal calibration (10%), untouched final test (15%).
- Pair split: random unique pairs, while reporting how many test entities were not seen during training.
- Cold cell: cell-line entity partition shared by all rows involving that cell line.
- Cold drug: compound entity partition shared by all rows involving that compound.
- Double cold: partition both entity sets, use only the train×train, val×val, cal×cal, test×test quadrants. Excluded cross-quadrant pairs are **not used**; report excluded sample count and shifted test distribution.
- Verify no row or relevant entity leaks between partitions. Store `split_manifest.csv` for every job.
- Feature selection (top training-cell variance and top training-drug-bit variance), missing-value imputation and standardization are fit only with training entities. Although all possible features are globally provided by the benchmark, training fold statistics never use other entities.
- Test labels never drive training, early stopping, feature selection, calibration scale estimation, or hyperparameter search.

## Experimental model hypothesis

Cross-Gated Interaction (CGI) model:

1. Two independent multi-layer encoders map selected expression / drug fingerprint features into 64-dimension representations `c`, `d`.
2. Reciprocal learned gates: `c'=c*(1+sigmoid(Wc d))`, `d'=d*(1+sigmoid(Wd c))`.
3. Factorized bilinear interaction: `(Uc') ⊙ (Vd')` (32-dimensional).
4. Residual interaction vector: `[c', d', c'⊙d', |c'-d'|, (Uc')⊙(Vd')]`.
5. Feed-forward head gives log IC50. Independently seeded ensemble of model copies produces mean + across-model dispersion.
6. Calibration options: absolute residual conformal, ensemble-scaled conformal, and novelty × ensemble-scaled conformal.

It is a **new combination and implementation hypothesis**, not a verified world's-first architecture. Prior-art and performance comparisons must determine genuine novelty. Do not claim clinical efficacy or better drug discovery outcomes from in-vitro data alone.

## Shift-aware interval formula

- For each cell line/drug, find its nearest feature-space neighbor among training entities (Euclidean distance normalized by feature dimension). For seen entities, distance = 0.
- For drug–cell pair i: `D_i = d_cell(i)+d_drug(i)`.
- Ensemble SD floor `s_i = sqrt(Var[member_predictions_i] + floor^2)`; floor is `max(0.08*sd_train_response,0.05)`.
- Calibration shift reference `r = max(median(D_cal), 0.05)`.
- Proposed scale `S_i = s_i * (1 + D_i/r)`.
- Split conformal score `|y_i - mu_i| / S_i` on calibration only. Quantile is finite-sample `ceil((n_cal+1)*(1-alpha))` order statistic, capped for edge conditions.
- Output interval `mu_i ± q*S_i`. Baselines use absolute-residual conformal intervals.

**Coverage caveat**: exchangeable calibration and test scores are needed for the usual finite-sample marginal guarantee. In cold held-out entities, this condition may not hold. The evaluation measures empirical coverage without promising out-of-distribution guarantee. The distance is a heuristic, not a likelihood or calibrated OOD score.

## Main comparisons

- Global constant mean, per-cell mean and per-drug mean (training information only; fall back to global mean for unseen IDs).
- Ridge regression, Random Forest and histogram gradient boosting trained on the same seed and feature schema. To respect RAM, only `baseline_rows` sampled *training* pairs used for Ridge/RF/gradient boosting; disclose this compute asymmetry. They never use validation, calibration or test labels for training.
- Early fusion MLP controls a simpler architecture in the ablation test.
- Main model with 2 independent neural ensemble seeds, 3 outer split seeds (42, 123, 2026) by default.

## Ablations

Under cold drug and double-cold with the split seed 42:

- `no_gate`: remove reciprocal gating while retaining towers and other terms.
- `no_bilinear`: remove factorized bilinear feature.
- `no_cell`: zero cell feature vector, leaving no varying cell feature information.
- `no_drug`: zero drug feature vector, leaving no varying drug feature information.
- `early_fusion`: directly concatenate all two input modalities and apply a standard MLP.
- For UQ component ablation, compare `ensemble_abs`, `ensemble_scaled`, `drugshift_uq` intervals for the SAME point predictions.

## Reporting and interpretation

Primary model selection: out-of-sample RMSE and MAE; secondary: Pearson and Spearman (undefined for constant predictions and recorded as missing), interval coverage and width. Report mean and SD over split seeds. Three seeds support descriptive comparison only, not definitive inferential claims. A good report includes failure cases, limitations under cold drug shifts, missing feature and response exclusions, and non-clinical applicability. Never choose test results to tune architecture or select intervals after reporting.

Test metrics are created *only* by actual execution of the Python program. Synthetic fixture tests are NOT manuscript results.
