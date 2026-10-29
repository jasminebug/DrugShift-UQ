"""Lightweight deterministic tests; NO network calls or full training."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
SPEC=importlib.util.spec_from_file_location('strengthen',Path(__file__).with_name('strengthen.py'))
S=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(S)

def test_cluster_bootstrap_identical_methods():
    y=np.arange(36,dtype=float)
    pred=y+np.sin(y)
    ci=S.cluster_bootstrap_delta(y,pred,pred,np.repeat(np.arange(12),3),seed=8,reps=100)
    assert np.allclose(ci,0,atol=1e-12)

def test_cluster_bootstrap_not_equal():
    y=np.arange(36,dtype=float)
    a=y+2
    b=y+1
    ci=S.cluster_bootstrap_delta(y,a,b,np.repeat(np.arange(12),3),seed=8,reps=50)
    assert np.allclose(ci,[1,1])

def test_interval_score_manual():
    y=np.array([0.,5.,10.]); lo=np.array([1.,2.,4.]);hi=np.array([3.,4.,6.])
    # 2 + 20*max(lo-y,0) + 20*max(y-hi,0)
    expected=[22,22,82]
    assert np.allclose(S.interval_score(y,lo,hi,alpha=.1),expected)

def test_verified_idempotent_checkpoint(tmp_path):
    out=tmp_path/'job';out.mkdir()
    (out/'predictions.csv').write_text('a,b\n1,2\n')
    S.write_complete(out,'identity',['predictions.csv'])
    assert S.checkpoint_complete(out,'identity')
    assert not S.checkpoint_complete(out,'different-input')
    (out/'predictions.csv').write_text('a,b\n2,2\n')
    assert not S.checkpoint_complete(out,'identity')

def test_export_empty_folder_never_claims_done(tmp_path):
    assert not S.checkpoint_complete(tmp_path/'missing')

def test_atomic_csv(tmp_path):
    x=pd.DataFrame({'value':[1,2,3]})
    p=tmp_path/'tables'/'sample.csv';S.save_csv(x,p)
    assert pd.read_csv(p).equals(x)
    assert not (p.parent/'sample.csv.tmp').exists()
