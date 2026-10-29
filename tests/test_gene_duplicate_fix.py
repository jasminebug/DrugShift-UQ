"""Regression tests for GDSC2 gene-expression duplicate IDs and metadata."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from drugshift.prepare import _reconcile_cell_features, convert_gdsc2


def test_mixed_duplicate_profiles_and_conflicting_stable_ids():
    raw = pd.DataFrame({
        'cell_line_id':['A','A','B','B','C','C','D','D','E'],
        'cellosaurus_id':['CVCL_0001','CVCL_0001','CVCL_0002','CVCL_0002',
                         'CVCL_0003','CVCL_0004',np.nan,np.nan,'CVCL_0005'],
        'GENE1':[1.,1.,2.,4.,8.,9.,7.,7.,5.],
        'GENE2':[5.,5.,0.,2.,1.,1.,3.,3.,7.],
        'cosmic_id':[101,101,102,102,103,104,105,105,106],
    })
    features, p, audit = _reconcile_cell_features(raw, set('ABCDE'))
    assert features.cell_line_id.tolist() == ['A','B','D','E']
    assert features.loc[features.cell_line_id=='B','GENE1'].iloc[0] == 3
    assert features.loc[features.cell_line_id=='B','GENE2'].iloc[0] == 1
    assert 'cosmic_id' not in features and 'cellosaurus_id' not in features
    assert p['gene_expression_duplicate_id_groups']==4
    assert p['gene_expression_exact_duplicate_groups']==2
    assert p['gene_expression_median_collapsed_groups']==1
    assert p['gene_expression_ambiguous_id_groups_excluded']==1
    assert audit.loc[audit.cell_line_id=='C','decision'].iloc[0]=='exclude_conflicting_cellosaurus_ids'


def test_duplicate_without_secondary_id_uses_documented_median():
    raw = pd.DataFrame({'cell_line_id':['X','X','Y'], 'G1':[0.,4.,2.], 'G2':[2.,6.,8.]})
    features, p, audit = _reconcile_cell_features(raw, {'X','Y'})
    assert features.loc[features.cell_line_id=='X','G1'].iloc[0]==2.
    assert p['gene_expression_median_collapsed_groups']==1
    assert audit.loc[audit.cell_line_id=='X','decision'].iloc[0]=='median_repeated_cell_line_measurements'


def test_conversion_dataset_duplicate_audit_and_rerun(tmp_path):
    path=tmp_path/'source'/'GDSC2';path.mkdir(parents=True)
    pd.DataFrame({'cell_line_name':['A','A','B','C','D','D'],
                  'pubchem_id':['111','222','111','111','111','222'],
                  'LN_IC50_curvecurator':[1.,2.,3.,4.,5.,6.]}).to_csv(path/'GDSC2.csv',index=False)
    pd.DataFrame({'cell_line_name':['A','A','B','C','C','D','D'],
                  'cellosaurus_id':['CVCL_A','CVCL_A','CVCL_B','CVCL_C1',
                                   'CVCL_C2','CVCL_D','CVCL_D'],
                  'G1':[1.,1.,2.,3.,4.,5.,7.],
                  'G2':[4.,4.,5.,6.,7.,8.,10.]}).to_csv(path/'gene_expression.csv',index=False)
    pd.DataFrame({'pubchem_id':['111','222'],'fp1':[0,1],'fp2':[1,0]}).to_csv(path/'fingerprints.csv',index=False)
    out=convert_gdsc2(tmp_path/'source',tmp_path/'processed')
    resp=pd.read_csv(out/'responses.csv')
    cells=pd.read_csv(out/'cell_features.csv')
    audit=pd.read_csv(out/'duplicate_cell_audit.csv')
    p=json.loads((out/'provenance.json').read_text(encoding='utf-8'))
    assert set(resp.cell_line_id)=={'A','B','D'}
    assert cells.cell_line_id.is_unique
    assert len(audit)==3
    assert p['gene_expression_ambiguous_id_groups_excluded']==1
    assert p['gene_expression_median_collapsed_groups']==1
    assert p['matched_response_rows']==5
    assert convert_gdsc2(tmp_path/'source',tmp_path/'processed')==out
