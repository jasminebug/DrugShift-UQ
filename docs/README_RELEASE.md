# Release integrity

This source package is assembled from the Windows one-click research system plus two hotfixes that were used in the successful experiment run. The base research-code files are **byte-for-byte identical** to the SHA256 entries in `evidence/run_manifest.json` (run ID `9734e917b726`).

The supplementary script in `supplement/strengthen.py` is the implementation used for full-data baseline and repeated-ablation checks. Aggregate evidence is sampled from the finished experiment artifacts and is not generated synthetic performance.

For a strict numerical reproduction, retain the original `config.json`, dataset record and pinned checksum. Installing future library releases can affect numeric reproducibility; the recorded setup used Windows 11, Python 3.12.10, and a CPU PyTorch distribution.

One-click re-execution on a different machine is a new experiment run, not an assertion that every seed will produce bitwise-identical floating-point results. The code was exercised with synthetic tests in this environment; downloading GDSC2 and rerunning the full dataset is intentionally **not** part of release packaging.
