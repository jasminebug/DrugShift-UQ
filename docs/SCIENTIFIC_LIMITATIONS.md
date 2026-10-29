# Scientific scope and limitations

The numerical evidence derives from a single publicly available, curated **GDSC2** endpoint (`LN_IC50_curvecurator`) and **three split seeds**, not clinical response in patients. Methods were compared with matched partitions but not equal hyperparameter search/computational budgets.

- **Generalization:** The neural ensemble has lowest measured mean RMSE for random held-out pairs and unseen cells, but histogram gradient boosting and Random Forest win for unseen drugs and double-cold, respectively.
- **Double cold:** Only matched entity quadrants were retained, excluding over half the response pairs; direct quantitative comparison with other regimes is limited.
- **Ablations:** Reciprocal gate deletion does not consistently worsen performance; ablation effects vary by seed.
- **Conformal coverage:** Exchangeability is violated or uncertain under entity shifts. Prediction intervals are diagnostic, not mathematically guaranteed under shift. The feature-distance heuristic worsens average interval score versus absolute-residual conformal intervals in every regime.
- **Interpretability:** Feature distance is not an OOD probability; the measured double-cold distance-error association is weak.
- **Data features:** Repeated gene-expression identifiers were reconciled using a documented policy; drugs with incomplete fingerprints were excluded, changing the population evaluated.
- **Reproducibility:** Full independent reproducibility requires the raw Zenodo dataset, the exact code hash, environment dependencies, and verification of source-data licensing. Aggregate reporting does not mean availability of patient-level validation.

No clinical decision-making, treatment recommendation, or patient-level treatment prediction is supported.
