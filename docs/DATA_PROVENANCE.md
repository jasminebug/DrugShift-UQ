# Dataset provenance and responsible redistribution

**Dataset:** DrEvalPy *Dataset for drevalpy* v47, GDSC2 archive at <https://zenodo.org/records/21807175> (DOI <https://doi.org/10.5281/zenodo.21807175>).

- Pinned ZIP: `GDSC2.zip`
- Recorded archive MD5: `49dc16929c6baa2b5d75c508429fa11f`
- Target: `LN_IC50_curvecurator`
- Raw response rows: 90,666; aligned eligible response pairs: 75,201; cell lines: 941; drugs: 234.
- Gene expression: `GDSC2/gene_expression.csv` (4 repeated IDs reconciled via audited median aggregation).
- Fingerprints: `GDSC2/drug_fingerprints/pubchem_id_to_demorgan_512_map.csv`, 512-bit map stored transposed in the upstream archive.
- 53 response drugs without usable fingerprint bits excluded, not imputed using fabricated structures.

The repository contains **no raw or processed third-party dataset**, and does not relicense the upstream data. Follow the dataset provider's terms before using or redistributing it. The downloader checks the pinned MD5 and retries partial transfers. Reproducibility manifests include processed-data SHA256 values for verification after legitimate download.

Source preprocessing documentation: <https://github.com/daisybio/preprocess_drp_data>.
