DRUGSHIFT-UQ - INCREMENTAL EXPERIMENT STRENGTHENING v1.0
========================================================

IMPORTANT: This is an ADD-ON to your existing successful DrugShift-UQ_OneClick project.
It is not a replacement for that project, dataset or ZIP.

1. Extract the entire DrugShift-UQ_Strengthening_v1 folder inside:
   C:\Users\User\Desktop\DRP\DrugShift-UQ_OneClick\
2. Double-click RUN_STRENGTHENING_ONCE.bat inside the extracted folder.
3. Leave the command window open. If Windows switches off, run the SAME BAT again.
4. Open:
   results\STRENGTHENED\9734e917b726\STRENGTHENED_REPORT.html

No downloads, pip installs or repeat of the original 22 jobs.
Original 22 jobs, reports, datasets and prior checkpoints are NOT deleted.

What is added:
- 12 main split/seed groups x three conventional baselines, now each fitted on the
  FULL training partition instead of only 18,000 pairs (36 new baseline fits).
- All five architecture ablations under cold-drug and double-cold conditions
  repeated with the two missing seeds (20 new ablation jobs; seed-42 reused).
- Three-seed paired metrics, entity-cluster descriptive bootstrap CIs, entity-level
  interval-coverage diagnostics and interval scoring.
- New 450-DPI PNG and vector PDF figures, CSV tables, HTML report and hash manifest.

Possible duration on Core i7-11800H CPU: tens of minutes to several hours;
full-data random forests and gradient boosting dominate. No guarantee of a fixed ETA.

Each baseline fit is checkpointed when its predictions finish, and each ablation
neural model saves epoch checkpoints. If interrupted mid-baseline-estimator fitting,
ONLY that one incomplete estimator fit must run again.

Scientific caveats:
- The new design corrects a baseline sample-budget imbalance, not a guaranteed
  accuracy/novelty limitation. Do NOT assume the new model will win.
- The original network weights, predictions and its 22 jobs are untouched.
- Coverage is measured on test splits; exchangeability is NOT guaranteed in cold splits.
- The large new comparison set is a robustness evaluation, not preregistered
  confirmatory testing. Report exploratory status honestly.
- Three split seeds from one dataset cannot prove clinical generalizability.

Failures: copy the bottom of the console output; do not delete any project folders.

Detailed science and limitations: SCIENTIFIC_SCOPE_AND_LIMITATIONS.md
Execution log: logs\strengthening.log (within the original project).
Results are under results\STRENGTHENED\<run id>\

If you have to resume, use this same BAT file, not any old installer hotfix.

Software validation performed on a SYNTHETIC test project only (results excluded
from this ZIP): 22 original-job mini-matrix, 36 supplementary baseline fits,
20 supplementary ablations, three new vector/PNG figures, HTML and LaTeX output.
A second complete run skipped all 36 baselines and queued 0 ablations.
Six dedicated add-on tests also passed. The real GDSC2 supplementary matrix
must still execute on your laptop; the new performance cannot be promised.
