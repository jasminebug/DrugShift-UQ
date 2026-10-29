# ICDPNS 2026 author checklist — verify again before submission

Conference website: https://icdpns.in/submission
CMT submission: https://cmt3.research.microsoft.com/ICDPNS2026

As verified on 29 October 2026:
- Deadline: **30 October 2026**.
- Original unpublished full manuscript, **8–12 pages**, in English, Springer format; **PDF** for review.
- Single-blind peer review. Do not submit the same work simultaneously elsewhere.
- Overall similarity threshold: 15% (with specified exclusions); normally <3% from any one source.
- The conference requires disclosure of AI assistance, including text, code and figures.
- Verify the precise Springer series, publisher approval and the potential publication charges with the organizers before relying on indexing claims.

### Before writing any results in the manuscript

- [ ] Full real-data run done with the pinned GDSC2 archive and verified checksum.
- [ ] Verified `COMPLETE.json` for every planned job and `results/FINAL_REPORT.html` exists.
- [ ] No synthetic-data smoke-test statistics used as paper results.
- [ ] Check feature IDs, response units and assay meaning; describe censoring and grouping.
- [ ] Include count of unique response pairs and dropped/duplicated records.
- [ ] Disclose Random Forest/Ridge limited training subsample and why.
- [ ] Include four partition protocols, leakage checks, calibration design, seeds, and training details.
- [ ] Distinguish model uncertainty proxy from actual clinical uncertainty.
- [ ] Report adverse and neutral outcomes honestly. Do not claim superior performance without evidence.
- [ ] Compare against relevant prior work and established models before declaring a contribution novel.
- [ ] Cite the DrEvalPy dataset benchmark, GDSC source, conformal prediction references, and research antecedents.
- [ ] Include truthful AI-tool disclosure required by conference, describing code, visualization and text assistance.

**Note:** Running experiments, generating figures or saving a draft does not submit a paper to CMT. An author must inspect the work and complete the submission in the portal.
