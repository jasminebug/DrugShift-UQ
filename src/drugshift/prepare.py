"""Pinned public data acquisition and strict schema validation. No silent target substitution."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request
import zipfile
import pandas as pd
import numpy as np


def checksum(path, algorithm="md5"):
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_resume(url, dest, expected_md5, retries=5):
    dest = Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and checksum(dest) == expected_md5:
        print(f"[download] Verified existing file: {dest}", flush=True)
        return dest
    part = dest.with_name(dest.name + '.part')
    if dest.exists():
        dest.unlink()  # a corrupted fully named archive must not be trusted
    for attempt in range(retries):
        try:
            have = part.stat().st_size if part.exists() else 0
            headers = {"User-Agent": "DrugShift-UQ/1.0 academic-reproduction"}
            if have:
                headers["Range"] = f"bytes={have}-"
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                code = response.status
                content_range = response.headers.get('Content-Range', '')
                if have and (code != 206 or not content_range.startswith(f'bytes {have}-')):
                    # Server ignored Range; restart from zero rather than append duplicate bytes.
                    have = 0
                mode = "ab" if have else "wb"
                with open(part, mode) as fout:
                    n = have
                    while True:
                        block = response.read(4 * 1024 * 1024)
                        if not block:
                            break
                        fout.write(block); n += len(block)
                        if n // (50 * 1024 * 1024) != (n-len(block)) // (50 * 1024 * 1024):
                            print(f"[download] {n/1024**2:,.0f} MiB saved", flush=True)
            if checksum(part) != expected_md5:
                # If bytes are complete yet invalid, a new download is required.
                if code == 200 or (code == 206 and response.headers.get('Content-Range', '').endswith(f'/{part.stat().st_size}')):
                    part.unlink(missing_ok=True)
                raise RuntimeError('Downloaded archive checksum does not match pinned Zenodo MD5')
            os.replace(part, dest)
            return dest
        except Exception as exc:
            print(f"[download] Attempt {attempt+1}/{retries}: {exc}", flush=True)
            if attempt == retries-1:
                raise
            time.sleep(min(2**attempt, 15))
    raise RuntimeError('Download failed')


def unzip_transactional(zippath, out):
    out = Path(out); sentinel = out / '.verified_extract'
    archive_md5 = checksum(zippath)
    if sentinel.exists() and sentinel.read_text(encoding='utf8') == archive_md5:
        return out
    stage = out.with_name(out.name + '_extracting')
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zippath) as z:
        for item in z.infolist():
            target = (stage/item.filename).resolve()
            if not target.is_relative_to(stage.resolve()):
                raise ValueError('Unsafe path in ZIP archive')
            if item.file_size > 4 * 1024**3:
                raise ValueError('Unexpectedly large member in dataset archive')
        z.extractall(stage)
    (stage / '.verified_extract').write_text(archive_md5, encoding='utf8')
    if out.exists():
        shutil.rmtree(out)
    stage.rename(out)
    return out


def locate(directory, filename):
    matches = sorted(Path(directory).rglob(filename), key=lambda p: (len(p.parts), str(p)))
    if not matches:
        raise FileNotFoundError(f"Expected file '{filename}' not found within {directory}. Check data version.")
    return matches[0]


def _id_choice(df, possible, label):
    for name in possible:
        if name in df.columns:
            return name
    raise ValueError(f"Missing {label}: expected one of {possible}, found {df.columns.tolist()[:12]}")


def _fingerprint_source(extracted):
    """Use the *actual* DrEvalPy v47 nested fingerprint map, not fingerprints.csv.

    DrEvalPy ships its DeMorgan maps with one drug/PubChem-ID per *column*
    and one fingerprint bit per *row*.  This is different from the usual
    feature-matrix CSV orientation.  Legacy flat files remain supported only
    for backward compatibility with test fixtures.
    """
    official = sorted(Path(extracted).rglob('pubchem_id_to_demorgan_512_map.csv'))
    official = [p for p in official if p.parent.name == 'drug_fingerprints']
    if official:
        return official[0], 'drevalpy_demorgan_512_transposed'
    legacy = sorted(Path(extracted).rglob('fingerprints.csv'))
    if legacy:
        return legacy[0], 'flat_fingerprints_csv'
    raise FileNotFoundError(
        "No supported drug fingerprint table found. Expected "
        "GDSC2/drug_fingerprints/pubchem_id_to_demorgan_512_map.csv "
        "(DrEvalPy v47), or fingerprints.csv (legacy). "
        "Inspect the archive content before changing the dataset version."
    )


def _convert_drug_fingerprints(path, mode, response_drug_ids):
    if mode == 'drevalpy_demorgan_512_transposed':
        # Official DrEvalPy code also transposes the DeMorgan map.  No
        # PubChem-ID-hashed/random fallback is used: absent structures are
        # excluded from the feature-matched evaluation population.
        raw = pd.read_csv(path, low_memory=False)
        if raw.shape[0] != 512:
            raise ValueError(
                f"Expected 512 fingerprint-bit rows in {path}, found "
                f"{raw.shape[0]}. Stop instead of silently changing features."
            )
        df = raw.T
        df.columns = [f'fp_{i:03d}' for i in range(512)]
        df.index.name = 'drug_id'
        df = df.reset_index()
    else:
        df = pd.read_csv(path, low_memory=False,
                         dtype={k:str for k in ('pubchem_id','drug_id')})
        key = _id_choice(df, ['pubchem_id','drug_id'], 'drug identifier')
        df = df.rename(columns={key:'drug_id'})
    df['drug_id'] = df['drug_id'].astype('string').str.strip()
    # Excludes artefactual CSV index columns and orphan IDs, rather than
    # letting these entries leak into training/evaluation.
    df = df.loc[df['drug_id'].isin(response_drug_ids)].copy()
    if df['drug_id'].duplicated().any():
        raise ValueError(f'Duplicate PubChem drug identifiers in {path}')
    feature_names = [c for c in df.columns if c != 'drug_id']
    df[feature_names] = df[feature_names].apply(pd.to_numeric, errors='coerce')
    if not feature_names:
        raise ValueError(f'No fingerprint feature columns in {path}')
    usable = df[feature_names].notna().all(axis=1)
    skipped = int((~usable).sum())
    if skipped:
        print(f'[prepare] Excluding {skipped} drugs with missing fingerprint bits; '
              'no synthetic fallback fingerprints.', flush=True)
    df = df.loc[usable, ['drug_id'] + feature_names]
    if df.empty:
        raise ValueError(f'No valid response-matched fingerprint rows in {path}')
    # Strict 0/1 check for the pinned DeMorgan map; legacy CSVs can be
    # continuous molecular embeddings in custom fixtures.
    if mode == 'drevalpy_demorgan_512_transposed':
        arr = df[feature_names].to_numpy(dtype=float)
        if not ((arr == 0) | (arr == 1)).all():
            raise ValueError(f'DeMorgan fingerprints must be binary (0/1), check {path}')
    return df, len(feature_names), skipped



def _reconcile_cell_features(cell, response_cell_ids):
    """Reconcile repeated cell-line IDs with an auditable, identity-safe rule.

    Rows are grouped ONLY by the cell-line identifier that also appears in the
    response table.  Identical expression vectors are reduced to one row.
    Non-identical expression vectors with the same ID are median-collapsed
    only when their available stable secondary IDs (Cellosaurus) do not
    contradict one another.  Any collision with conflicting Cellosaurus IDs
    is excluded.  No globally fitted data transform is used here.

    Returns (unique_features, provenance_counts, duplicate_audit).
    """
    cell = cell.copy()
    cell['cell_line_id'] = cell['cell_line_id'].astype('string').str.strip()
    cell = cell.dropna(subset=['cell_line_id'])
    cell = cell.loc[(cell['cell_line_id'] != '') &
                    cell['cell_line_id'].isin(response_cell_ids)].copy()
    total_rows = len(cell)

    # Ignore metadata even if its values happen to be numeric. In particular,
    # Cellosaurus IDs are biological identifiers, not expression features.
    exclude = {'cell_line_id', 'cellosaurus_id', 'cell_line_name',
               'cosmic_id', 'pubchem_id', 'drug_id', 'index'}
    numeric_cols = [c for c in cell.columns
                    if c not in exclude and not str(c).lower().startswith('unnamed:')
                    and pd.api.types.is_numeric_dtype(cell[c])]
    if not numeric_cols:
        raise ValueError('No numeric gene-expression columns found; inspect gene_expression.csv')

    has_secondary = 'cellosaurus_id' in cell.columns
    rows = []
    audit = []
    stats = {
        'gene_expression_rows_after_response_filter': int(total_rows),
        'gene_expression_duplicate_id_groups': 0,
        'gene_expression_duplicate_rows_excess': 0,
        'gene_expression_exact_duplicate_groups': 0,
        'gene_expression_median_collapsed_groups': 0,
        'gene_expression_ambiguous_id_groups_excluded': 0,
        'gene_expression_ambiguous_rows_excluded': 0,
        'gene_expression_all_nan_groups_excluded': 0,
        'gene_expression_features': int(len(numeric_cols)),
        'gene_expression_processing_policy': (
            'exact duplicates collapsed; compatible repeated profiles ' 
            'median-aggregated by cell-line ID; conflicting Cellosaurus ' 
            'identities excluded; gene-expression normalization fitted ' 
            'only to training entities downstream'
        ),
    }

    for identifier, group in cell.groupby('cell_line_id', sort=True, dropna=False):
        n = len(group)
        secondary = set()
        if has_secondary:
            sec = group['cellosaurus_id'].astype('string').str.strip()
            secondary = {str(v) for v in sec.dropna().tolist() if v and v.lower() not in {'nan', 'none', '<na>'}}
        matrix = group[numeric_cols].to_numpy(dtype=np.float64)
        matrix[~np.isfinite(matrix)] = np.nan
        if n > 1:
            stats['gene_expression_duplicate_id_groups'] += 1
            stats['gene_expression_duplicate_rows_excess'] += n - 1
        if len(secondary) > 1:
            stats['gene_expression_ambiguous_id_groups_excluded'] += 1
            stats['gene_expression_ambiguous_rows_excluded'] += n
            decision = 'exclude_conflicting_cellosaurus_ids'
            med = None
        elif not np.isfinite(matrix).any():
            stats['gene_expression_all_nan_groups_excluded'] += 1
            decision = 'exclude_all_missing_expression'
            med = None
        else:
            if n == 1:
                decision = 'retain_single'
                med = matrix[0]
            elif np.allclose(matrix, matrix[:1], atol=1e-8, rtol=1e-7, equal_nan=True):
                stats['gene_expression_exact_duplicate_groups'] += 1
                decision = 'collapse_identical'
                med = matrix[0]
            else:
                # Same primary cell ID with no contradictory Cellosaurus ID.
                # Use an explicitly documented per-gene replicate median.
                # Avoid a random `drop_duplicates(keep=first)` selection.
                with np.errstate(all='ignore'):
                    med = np.nanmedian(matrix, axis=0)
                stats['gene_expression_median_collapsed_groups'] += 1
                decision = 'median_repeated_cell_line_measurements'
            rows.append([str(identifier), *med.tolist()])
        if n > 1 or decision.startswith('exclude'):
            audit.append({
                'cell_line_id': str(identifier),
                'source_rows': int(n),
                'cellosaurus_ids': ';'.join(sorted(secondary)),
                'decision': decision,
                'response_linkage': 'by cell_line_name only',
            })

    features = pd.DataFrame(rows, columns=['cell_line_id', *numeric_cols])
    if features.empty:
        raise ValueError(
            'No usable cell-line gene-expression profiles after duplicate ' 
            'reconciliation. Do not force merging biologically distinct IDs; ' 
            'inspect gene_expression.csv and mapping to GDSC2.csv.'
        )
    assert features['cell_line_id'].is_unique, 'Duplicate reconciliation produced nonunique IDs'
    audit_frame = pd.DataFrame(audit, columns=[
        'cell_line_id', 'source_rows', 'cellosaurus_ids', 'decision', 'response_linkage'
    ])
    stats['cell_line_id_features'] = len(numeric_cols)
    stats['cell_line_id_entities'] = len(features)
    return features, stats, audit_frame


def convert_gdsc2(extracted, processed, target="LN_IC50_curvecurator"):
    processed = Path(processed)
    processed.mkdir(parents=True, exist_ok=True)
    fp_path, fp_mode = _fingerprint_source(extracted)
    source_paths = [locate(extracted, 'GDSC2.csv'),
                    locate(extracted, 'gene_expression.csv'), fp_path]
    fingerprint = {str(p): (p.stat().st_size, int(p.stat().st_mtime_ns))
                   for p in source_paths}
    metadata_path = processed/'provenance.json'
    expected = {'source_files': fingerprint, 'target_column': target,
                'dataset': 'GDSC2', 'fingerprint_format': fp_mode,
                'converter_version': '2026-10-29-drugshift-duplicate-cellfix-v2'}
    output_files = [processed/x for x in ('responses.csv','cell_features.csv','drug_features.csv')]
    if metadata_path.exists() and (processed/'duplicate_cell_audit.csv').exists() and all(p.exists() and p.stat().st_size > 0 for p in output_files):
        try:
            if all(json.loads(metadata_path.read_text())[k] == v for k, v in expected.items()):
                print('[prepare] Reusing verified cached processed dataset', flush=True)
                return processed
        except (OSError, KeyError, ValueError):
            pass

    resp = pd.read_csv(source_paths[0], low_memory=False,
                       dtype={k:str for k in ('cell_line_name','cell_line_id','pubchem_id','drug_id')})
    if target not in resp:
        raise ValueError(f"Target {target!r} unavailable. Columns: {resp.columns.tolist()[:40]}. Do not substitute another endpoint without changing config.")
    cid = _id_choice(resp,['cell_line_name','cell_line_id'], 'cell line identifier')
    did = _id_choice(resp,['pubchem_id','drug_id'], 'drug identifier')
    resp = resp[[cid, did, target]].rename(columns={cid:'cell_line_id',did:'drug_id',target:'response'})
    resp['cell_line_id'] = resp['cell_line_id'].astype('string').str.strip()
    resp['drug_id'] = resp['drug_id'].astype('string').str.strip()
    resp['response'] = pd.to_numeric(resp['response'], errors='coerce')
    resp = resp.dropna(subset=['cell_line_id','drug_id','response'])
    if resp.empty: raise ValueError('Empty response table after endpoint/ID checks')
    response_stats = {'raw_rows':len(resp), 'raw_cell_lines':resp.cell_line_id.nunique(),
                      'raw_drugs':resp.drug_id.nunique()}

    cell = pd.read_csv(source_paths[1], low_memory=False,
                       dtype={k:str for k in ('cell_line_name','cell_line_id', 'cellosaurus_id')})
    cell_key = _id_choice(cell, ['cell_line_name','cell_line_id'], 'cell line identifier')
    cell = cell.rename(columns={cell_key:'cell_line_id'})
    cell, duplicate_stats, duplicate_audit = _reconcile_cell_features(
        cell, set(resp['cell_line_id'].astype(str))
    )
    response_stats.update(duplicate_stats)
    if duplicate_stats['gene_expression_duplicate_id_groups']:
        print('[prepare] Gene-expression duplicates reconciled: '
              f"{duplicate_stats['gene_expression_duplicate_id_groups']} repeated IDs; "
              f"{duplicate_stats['gene_expression_exact_duplicate_groups']} exact, "
              f"{duplicate_stats['gene_expression_median_collapsed_groups']} median-aggregated, "
              f"{duplicate_stats['gene_expression_ambiguous_id_groups_excluded']} excluded for "
              'conflicting Cellosaurus identity. See duplicate_cell_audit.csv.',flush=True)
    # An audit file is always written, including when there are no duplicates.
    audit_file = processed/'duplicate_cell_audit.csv'
    audit_temp = processed/'duplicate_cell_audit.csv.tmp'
    duplicate_audit.to_csv(audit_temp, index=False)
    os.replace(audit_temp, audit_file)

    drugs, feature_count, skipped = _convert_drug_fingerprints(
        source_paths[2], fp_mode, set(resp.drug_id.astype(str)))
    response_stats['drug_id_features'] = feature_count
    response_stats['drug_id_entities'] = len(drugs)
    response_stats['drugs_excluded_missing_fingerprint'] = skipped
    response_stats['fingerprint_source'] = str(source_paths[2])
    # Only pairs with real measured responses and *real* source features.
    good = resp.cell_line_id.isin(cell.cell_line_id) & resp.drug_id.isin(drugs.drug_id)
    resp = resp.loc[good].copy()
    response_stats['matched_response_rows'] = len(resp)
    if resp.empty:
        raise ValueError('No fully feature-matched response rows; verify the identifier mapping')

    for frame, filename in [(cell, 'cell_features.csv'),
                            (drugs, 'drug_features.csv'),
                            (resp, 'responses.csv')]:
        temp = processed/(filename+'.tmp')
        frame.to_csv(temp, index=False)
        os.replace(temp, processed/filename)
    temp = metadata_path.with_name(metadata_path.name+'.tmp')
    temp.write_text(json.dumps({**expected, **response_stats}, indent=2), encoding='utf8')
    os.replace(temp, metadata_path)
    print(f'[prepare] {response_stats}', flush=True)
    return processed

def prepare(root, conf):
    root = Path(root)
    archive = root/'data/raw/GDSC2.zip'
    url = f"https://zenodo.org/records/{conf['zenodo_record']}/files/{conf['zenodo_file']}?download=1"
    download_resume(url,archive,conf['zenodo_md5'])
    extracted = unzip_transactional(archive,root/'data/raw/unpacked')
    return convert_gdsc2(extracted, root/'data/processed', conf['target'])
