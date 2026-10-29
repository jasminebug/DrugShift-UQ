# How to publish this folder on GitHub

1. Sign in to GitHub and open **+ → New repository** (or <https://github.com/new>).
2. Name: **DrugShift-UQ**; visibility: **Public**, if comfortable sharing code and evidence.
3. Add this description: `Reproducible drug-response prediction under unseen-drug and unseen-cell-line shifts, with cross-gated ensembles, uncertainty calibration, matched baselines and ablation studies.`
4. Set **Add a README**, **Add .gitignore**, and **Choose a license** to **None**: these files are already in this folder. You can keep MIT or replace it before uploading.
5. Click **Create repository**.
6. Select **uploading an existing file** or **Add file → Upload files**. Upload the **contents of the `DrugShift-UQ` folder**, not the outer ZIP itself. Dragging the whole extracted `DrugShift-UQ` folder into the browser upload pane preserves subfolders in supported browsers; check the preview before committing.
7. Commit message: `Initial research release: audited DrugShift-UQ implementation and evidence`.
8. Confirm GitHub displays `README.md` at the top level plus `src/`, `supplement/`, `tests/`, `docs/`, and `evidence/`.
9. Copy the new repository URL and add it to the manuscript's code-availability statement after verification.

**Alternative:** use GitHub Desktop to add/publish the extracted folder if the browser has trouble uploading nested directories.

Do not upload `.venv`, `data/raw`, `data/processed`, `results`, `logs`, `*.pt`, individual predictions, or the downloaded GDSC2 ZIP. GitHub's web uploader has per-file and per-batch limits; no large trained artifacts are present in this release.

Suggested topics: `drug-response-prediction`, `pharmacogenomics`, `gdsc2`, `deep-learning`, `uncertainty-quantification`, `conformal-prediction`, `distribution-shift`, `reproducible-research`, `pytorch`, `ablation-study`.
