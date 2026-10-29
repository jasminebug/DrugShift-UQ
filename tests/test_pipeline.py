from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from drugshift.prepare import convert_gdsc2,checksum
from drugshift.data import read_dataset,FoldFeatures
from drugshift.splits import make_split,validate
from drugshift.metrics import finite_sample_q, interval
from drugshift.models import CrossGatedInteraction,fit_member,predict
import torch


def sample_tables(path,nc=45,nd=28,seed=42):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed)
    cells=pd.DataFrame(rng.normal(size=(nc,10)),columns=[f'g{i}' for i in range(10)])
    cells.insert(0,'cell_line_id',[f'C{i}' for i in range(nc)])
    drugs=pd.DataFrame(rng.integers(0,2,(nd,14)),columns=[f'fp{i}' for i in range(14)])
    drugs.insert(0,'drug_id',[f'D{i}' for i in range(nd)])
    pairs=pd.DataFrame([(a,b) for a in cells.cell_line_id for b in drugs.drug_id],columns=['cell_line_id','drug_id'])
    factors=(cells.set_index('cell_line_id').loc[pairs.cell_line_id,'g0'].to_numpy() +
             drugs.set_index('drug_id').loc[pairs.drug_id,'fp0'].to_numpy())
    pairs['response']=factors+rng.normal(0,.1,len(pairs))
    cells.to_csv(path/'cell_features.csv',index=False)
    drugs.to_csv(path/'drug_features.csv',index=False)
    pairs.to_csv(path/'responses.csv',index=False)
    return path

def test_four_splits_no_leakage(tmp_path):
    ds=read_dataset(sample_tables(tmp_path))
    for regime in ('pair','cold_cell','cold_drug','cold_both'):
        s=make_split(ds.rows,regime,42);validate(ds.rows,s,regime)
        assert len(s.train)>0 and len(s.test)>0
        if regime=='cold_both':assert len(s.excluded)>0

def test_preprocessing_only_training(tmp_path):
    ds=read_dataset(sample_tables(tmp_path))
    s=make_split(ds.rows,'cold_cell',42)
    f=FoldFeatures(genes=5,drug_bits=6).fit(ds,s.train)
    assert f.matrix(s.test).shape[1]==11
    assert np.all(np.isfinite(f.matrix(s.test)))

def test_finite_sample_quantile():
    residuals=np.arange(100)
    assert finite_sample_q(residuals,0.1)==90
    cal=np.zeros(100);actual=np.zeros(10);pred=np.zeros(10)
    stat,lo,hi=interval(cal,cal,actual,pred,.1)
    assert stat['coverage']==1 and np.isfinite(lo).all()

def test_network_architecture_variants():
    for variant in ('full','no_gate','no_bilinear','no_cell','no_drug','early_fusion'):
        m=CrossGatedInteraction(5,6,variant=variant)
        a=m(torch.randn(7,5),torch.randn(7,6))
        assert a.shape==(7,) and torch.isfinite(a).all()

def test_epoch_checkpoint_resume(tmp_path):
    ds=read_dataset(sample_tables(tmp_path/'raw'))
    s=make_split(ds.rows,'pair',42); f=FoldFeatures(5,6).fit(ds,s.train)
    y=ds.rows.response.to_numpy('float32')
    checkpoint=tmp_path/'ckpt.pt'
    mem1=fit_member(f,y,s.train,s.val,checkpoint,epochs=1,patience=10,batch_size=128,
                    seed=10,device='cpu',config_tag='one')
    assert mem1['trained_epochs']==1
    mem2=fit_member(f,y,s.train,s.val,checkpoint,epochs=2,patience=10,batch_size=128,
                    seed=10,device='cpu',config_tag='one')
    assert mem2['trained_epochs']==2
    p=predict(mem2,f,s.test)
    assert p.shape==(len(s.test),) and np.isfinite(p).all()
    with pytest.raises(RuntimeError):
        fit_member(f,y,s.train,s.val,checkpoint,epochs=2,seed=10,config_tag='changed')

def test_convert_strict_endpoint_and_schema(tmp_path):
    source=tmp_path/'source'/'GDSC2';source.mkdir(parents=True)
    pd.DataFrame({'cell_line_name':['C1','C2'],'pubchem_id':['123','123'],
                  'LN_IC50_curvecurator':[.5,1.2]}).to_csv(source/'GDSC2.csv',index=False)
    pd.DataFrame({'cell_line_name':['C1','C2'],'g1':[1.4,2.3]}).to_csv(source/'gene_expression.csv',index=False)
    pd.DataFrame({'pubchem_id':['123'],'f1':[1]}).to_csv(source/'fingerprints.csv',index=False)
    out=convert_gdsc2(tmp_path/'source',tmp_path/'out')
    assert list(pd.read_csv(out/'responses.csv').columns)==['cell_line_id','drug_id','response']
    with pytest.raises(ValueError):convert_gdsc2(tmp_path/'source',tmp_path/'out2',target='absent')


def test_real_drevalpy_512bit_layout_and_missing_filter(tmp_path):
    """Reproduces the Zenodo v47 nested, transposed fingerprint schema."""
    source = tmp_path/'source'/'GDSC2'
    fp_dir = source/'drug_fingerprints'
    fp_dir.mkdir(parents=True)
    pd.DataFrame({
        'cell_line_name':['A','B','A','B']*40,
        'pubchem_id':['111','111','222','222']*40,
        'LN_IC50_curvecurator':np.tile([0.3,0.8,1.5,2.1],40),
    }).to_csv(source/'GDSC2.csv', index=False)
    pd.DataFrame({'cell_line_name':['A','B'],'gene1':[1.2,2.3],'gene2':[3.4,4.5]}).to_csv(
        source/'gene_expression.csv', index=False)
    fp = pd.DataFrame({'111': np.arange(512)%2,
                       '222': (np.arange(512)+1)%2,
                       '333': [np.nan]*512})
    fp.to_csv(fp_dir/'pubchem_id_to_demorgan_512_map.csv', index=False)
    out = convert_gdsc2(source, tmp_path/'processed')
    drug = pd.read_csv(out/'drug_features.csv', dtype={'drug_id':str})
    assert drug.shape == (2,513)
    assert set(drug.drug_id)=={'111','222'}
    assert drug.loc[drug.drug_id=='111','fp_000'].iloc[0] == 0
    assert drug.loc[drug.drug_id=='222','fp_000'].iloc[0] == 1
    assert drug.loc[drug.drug_id=='111','fp_511'].iloc[0] == 1
    prov = __import__('json').loads((out/'provenance.json').read_text())
    assert prov['fingerprint_format']=='drevalpy_demorgan_512_transposed'
    assert prov['drug_id_features']==512
    assert prov['matched_response_rows']==160
    # reruns reuse processed data instead of redoing conversion
    assert convert_gdsc2(source, tmp_path/'processed')==out
