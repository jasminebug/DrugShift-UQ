# Windows 11: one-click operation, recovery, and troubleshooting

## First launch

1. Extract the **entire ZIP** to a writable folder, for example `C:\Users\User\Desktop\Development\DrugShift-UQ_OneClick`.
2. Ensure you have an internet connection and enough free disk space (reserve 8 GB; a few GB for extracted data and model checkpoints; amount varies).
3. Double-click **`START_RESEARCH.bat`**. Keep your laptop plugged in and allow Windows firewall access to the official dataset host if prompted.
4. The bootstrap checks for Python 3.11/3.12, installs a signed official Python installer in `.python` only if needed, creates `.venv`, installs necessary packages, checks dependency resolution, runs tests, downloads and validates GDSC2, converts the data and executes all jobs. It opens `results\FINAL_REPORT.html` upon **complete** success.
5. The first setup needs access to `python.org`, `pypi.org`, `download.pytorch.org`, and `zenodo.org`. The package cannot run offline on the first launch. No data or Python packages are embedded in the ZIP.

## Shutdown, internet loss, interrupted training

- Re-run **`START_RESEARCH.bat`** or **`RESUME_RESEARCH.bat`** when Windows starts. Both are restart-safe entry points.
- Saved GDSC2 `.part` bytes are resumed with HTTP Range **when supported by the server**; when the server does not support Range, only the current download restarts.
- Dataset extraction uses a transactional staging folder, with a verified stamp.
- Each training member writes a checkpoint **after every completed epoch**, storing weights, optimizer state, random states, the best validation state and the epoch number. Only the unfinished epoch needs repeating.
- Individual completed jobs are marked by `COMPLETE.json` after the metrics and predictions have been saved; rerunning skips them.
- Identical code/data/config produces the same `run_id`. If you change any of these, the pipeline creates a separate immutable run directory rather than silently mixing results.
- Python or hardware-level nondeterminism can still cause small numerical differences. Make research conclusions based on multiple seeds and saved manifests.

### Optional: automatic restart when signing into Windows

To have Windows automatically restart the interrupted pipeline after user login, **opt in** by opening `ENABLE_AUTO_RESUME_ON_LOGIN.bat` once. It creates a shortcut in your account's Startup folder. Use `DISABLE_AUTO_RESUME.bat` to remove that shortcut. It runs under your user account; it does not boot the machine or prevent sleep, and it does not install a system service.

## Files to inspect

- `logs\pipeline.log` — cumulative error and progress logs.
- `results\runs\<run_id>\jobs\...\checkpoints\member_*.pt` — auto-restored epoch state.
- `results\runs\<run_id>\jobs\...\predictions.csv` — untouched test targets and predictions.
- `results\runs\<run_id>\jobs\...\split_manifest.csv` — exact leakage audit for each row.
- `results\runs\<run_id>\jobs\...\metrics.json` — metric record, seed and split sizes.
- `results\runs\<run_id>\figures\` — high-resolution, publication-quality PNG and vector PDF files.
- `results\runs\<run_id>\tables\` — tables in CSV and LaTeX.
- `results\FINAL_REPORT.html` — modern offline results page (only after all jobs finish).

## Troubleshooting

- Python download fails: check internet/firewall and Python.org access; inspect `logs\pipeline.log` and the launcher window.
- `pip` fails: make sure pypi.org and download.pytorch.org are not blocked; another click retries.
- Dataset hash error: download was corrupted or upstream replaced the pinned archive; do **not** disable checksum verification.
- Target / feature schema error: pinned dataset may have changed; review upstream official schema. The importer intentionally does not guess a different endpoint.
- Out of RAM: lower `batch_size` in `config.json` (default 1024) and/or `baseline_rows` (default 18,000); this creates a **new run fingerprint**, preserving existing runs. More free disk helps but cannot replace adequate RAM.
- Slow CPU training: this is a large factorial experiment; it may run for many hours. `config.json` controls `epochs`, `members`, `seeds`, `ablations` and `regimes`. Reducing them impacts scientific strength and requires honest disclosure. Do not confuse fewer seed runs with comparable uncertainty.
- Windows blocks BAT/script: click More info > Run anyway only if you have verified the files and trust the source. Review any commands before running, as the BAT installs packages and obtains data.
- Not a clinical decision support program. Public data and model predictions should never be used to select patient treatment.
