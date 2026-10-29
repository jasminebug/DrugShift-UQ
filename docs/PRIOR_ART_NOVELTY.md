# Closest research precedents and defensible positioning

This is a research idea, not a legal/prior-art uniqueness guarantee. Some important precedents already exist:

1. **Iversen, Renard, Baum (2026)**, *Leveraging uncertainty estimates for drug response prediction in cancer cell lines*, Bioinformatics Advances. DOI: https://doi.org/10.1093/bioadv/vbag246 . Neural ensembles, distribution shift and uncertainty are **already extensively studied**. This work compares seven uncertainty methods and includes cross-screen transfer. Credit it. Our proposed split-conformal *interval scaling* and cold-both experiment are a different, specifically testable setup, not the first uncertainty-aware DRP.
2. **Hernández-Hernández, Guo, Ballester (2026)**, *Conformal prediction of molecule-induced cancer cell growth inhibition challenged by strong distribution shifts*, Pattern Recognition. DOI: https://doi.org/10.1016/j.patcog.2025.112501 . Conformal prediction under drug screening shift is **not new**. Our conformal shift scale must be evaluated against vanilla conformal, and may prove inferior.
3. **MDCDR (2026)**, *Predicting cancer drug response via multimodal feature fusion and feature disentanglement*, available via PubMed Central https://pmc.ncbi.nlm.nih.gov/articles/PMC13262659/ . Uses bilinear attention with molecular and transcriptomic features. **Bilinear fusion is not novel in itself**.
4. **InFo-DRP (2026)**, *Integrating invariant learning and foundation representations for unseen drug response prediction*, DOI https://doi.org/10.1109/JBHI.2026.3738922 . Includes unseen drug / scaffold generalization and multi-branch multimodal representations. The cold-drug setting is well-established; do not claim to introduce it.
5. **DrEvalPy (2026)**: https://github.com/daisybio/drevalpy . Standardizes drug response evaluation and provides strong baselines, publication-ready visualization and cross-dataset testing. Its dataset supplies the pinned GDSC2 data. Data preprocessing and benchmark evaluation are not ours; cite their methods.

## What this package actually investigates

- A specific combination of **reciprocal residual cross-gating, factorized bilinear interaction, independent ensemble and training-nearest-entity scale**.
- Four held-out regimes, including an explicitly cross-quadrant-excluded double-cold protocol.
- Matched-point-prediction comparison of three calibrators (`absolute`, `ensemble scaled`, `shift scaled`), to isolate calibration from predictor accuracy.
- Five architecture/modality ablations and traditional ML baselines.

**Originality test:** only assert a methodological contribution after comprehensive related-work comparison *and* substantive experimentally supported gains or explanatory findings. The proposed name, cosmetic architecture fusion, and shift score do not independently establish novelty. A detailed DrEvalPy-compatible evaluation against stronger literature baselines would be a significant next validation step, and is not built into this starter. Avoid claiming SOTA.
