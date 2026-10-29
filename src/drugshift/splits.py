"""Leakage-conscious 65/10/10/15 partition protocols."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
@dataclass
class Split:
    train:np.ndarray
    val:np.ndarray
    cal:np.ndarray
    test:np.ndarray
    excluded:np.ndarray

def partition(items,rng):
    ids=rng.permutation(np.unique(items)); n=len(ids)
    if n<12: raise ValueError('Requires >=12 unique entities for entity-disjoint evaluation')
    cuts=[int(n*.65),int(n*.75),int(n*.85)]
    return [set(v) for v in np.split(ids,cuts)]

def make_split(frame,regime,seed):
    rng=np.random.default_rng(seed); n=len(frame)
    if regime=='pair':
        # A small number of rare entities may appear only outside training; report this explicitly.
        arr=rng.permutation(n);cuts=[int(n*.65),int(n*.75),int(n*.85)]
        part=np.split(arr,cuts)
    else:
        cs=frame.cell_line_id.to_numpy();ds=frame.drug_id.to_numpy()
        if regime=='cold_cell':
            sets=partition(cs,rng);part=[np.flatnonzero(np.isin(cs,list(g))) for g in sets]
        elif regime=='cold_drug':
            sets=partition(ds,rng);part=[np.flatnonzero(np.isin(ds,list(g))) for g in sets]
        elif regime=='cold_both':
            csets=partition(cs,rng); dsets=partition(ds,rng)
            # Only train×train, val×val, cal×cal, test×test quadrants are used.
            # Other cross-quadrant pairs are excluded from model selection and test.
            part=[np.flatnonzero(np.isin(cs,list(c)) & np.isin(ds,list(d))) for c,d in zip(csets,dsets)]
        else: raise ValueError('Unknown regime '+str(regime))
    if min(map(len,part))<10:raise ValueError('Fewer than 10 pairs in at least one partition')
    excluded=np.setdiff1d(np.arange(n),np.concatenate(part))
    split=Split(*[np.asarray(p,dtype=np.int64) for p in part],excluded)
    validate(frame,split,regime)
    return split

def validate(frame,s,regime):
    parts=[s.train,s.val,s.cal,s.test]
    indices=np.concatenate(parts+[s.excluded]); assert len(np.unique(indices))==len(frame),'row overlap or missing rows'
    keys=({'cold_cell':['cell_line_id'],'cold_drug':['drug_id'],
           'cold_both':['cell_line_id','drug_id']}).get(regime,[])
    for k in keys:
        sets=[set(frame.iloc[p][k]) for p in parts]
        for i in range(4):
            for j in range(i+1,4):
                assert sets[i].isdisjoint(sets[j]),f'Entity leakage: {k} {i}-{j}'
    pairs=[set(zip(frame.iloc[p].cell_line_id,frame.iloc[p].drug_id)) for p in parts]
    for i in range(4):
        for j in range(i+1,4):assert pairs[i].isdisjoint(pairs[j]),'Duplicate drug-cell pair leakage'

def write_manifest(frame,s,path):
    import os
    labels=np.repeat('excluded',len(frame)).astype('U12')
    for name in ['train','val','cal','test']:labels[getattr(s,name)]=name
    tmp=str(path)+'.tmp';out=frame[['cell_line_id','drug_id']].copy()
    out.insert(0,'row_index',np.arange(len(frame)))
    out['partition']=labels;out.to_csv(tmp,index=False);os.replace(tmp,path)
