"""Feature preprocessing fitted exclusively on training entities."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

@dataclass
class Dataset:
    rows: pd.DataFrame
    cell_features: pd.DataFrame
    drug_features: pd.DataFrame


def read_dataset(path):
    path=Path(path)
    resp=pd.read_csv(path/'responses.csv',dtype={'cell_line_id':str,'drug_id':str})
    for key in ['cell_line_id','drug_id','response']:
        if key not in resp: raise ValueError(f'Missing {key}')
    resp['response']=pd.to_numeric(resp.response,errors='coerce')
    resp=resp.replace([np.inf,-np.inf],np.nan).dropna(subset=['response','cell_line_id','drug_id'])
    original_rows=len(resp)
    # Replicate response measurements are collapsed before splitting, stopping pair leakage.
    resp=resp.groupby(['cell_line_id','drug_id'],sort=True,as_index=False)['response'].median()
    feats=[]
    for name, key in [('cell_features.csv','cell_line_id'),('drug_features.csv','drug_id')]:
        df=pd.read_csv(path/name,dtype={key:str},low_memory=False)
        if key not in df: raise ValueError(f'No {key} in {name}')
        if df[key].duplicated().any(): raise ValueError(f'Duplicate entity IDs in {name}')
        df=df.set_index(key)
        df=df.select_dtypes(include=[np.number]).replace([np.inf,-np.inf],np.nan)
        df=df.loc[:,df.notna().any(axis=0)]
        if df.empty: raise ValueError(f'No numeric features in {name}')
        feats.append(df.astype(np.float32))
    c,d=feats
    good=resp.cell_line_id.isin(c.index)&resp.drug_id.isin(d.index)
    resp=resp.loc[good].reset_index(drop=True)
    if len(resp)<100: raise ValueError('Fewer than 100 aligned drug-response pairs; check ID columns.')
    print(f'[data] {len(resp):,} aligned unique pairs, {resp.cell_line_id.nunique():,} cells, {resp.drug_id.nunique():,} drugs. Original rows: {original_rows:,}.',flush=True)
    return Dataset(resp,c,d)


class FoldFeatures:
    def __init__(self,genes=256,drug_bits=512): self.genes=genes; self.drug_bits=drug_bits
    def fit(self,ds,train_idx):
        c=ds.cell_features; d=ds.drug_features; frame=ds.rows.iloc[train_idx]
        c_train=c.loc[frame.cell_line_id.unique()].to_numpy(np.float64)
        d_train=d.loc[frame.drug_id.unique()].to_numpy(np.float64)
        with np.errstate(all='ignore'):
            self.cm=np.nanmedian(c_train,axis=0); self.dm=np.nanmedian(d_train,axis=0)
        self.cm=np.nan_to_num(self.cm,nan=0);self.dm=np.nan_to_num(self.dm,nan=0)
        c_train=np.where(np.isfinite(c_train),c_train,self.cm)
        d_train=np.where(np.isfinite(d_train),d_train,self.dm)
        self.csel=np.argsort(-np.var(c_train,axis=0),kind='stable')[:self.genes]
        self.dsel=np.argsort(-np.var(d_train,axis=0),kind='stable')[:self.drug_bits]
        self.scaler=StandardScaler().fit(c_train[:,self.csel])
        c_all=c.to_numpy(np.float64); d_all=d.to_numpy(np.float64)
        c_all=np.where(np.isfinite(c_all),c_all,self.cm)
        d_all=np.where(np.isfinite(d_all),d_all,self.dm)
        self.cv=np.clip(self.scaler.transform(c_all[:,self.csel]),-10,10).astype('float32')
        self.dv=d_all[:,self.dsel].astype('float32')
        self.cids=c.index.get_indexer(ds.rows.cell_line_id)
        self.dids=d.index.get_indexer(ds.rows.drug_id)
        assert min(self.cids.min(),self.dids.min())>=0
        self.columns={'genes':[str(c.columns[k]) for k in self.csel], 'drug_fingerprint_columns':[str(d.columns[k]) for k in self.dsel]}
        return self
    def matrix(self,indices):
        return np.concatenate([self.cv[self.cids[indices]],self.dv[self.dids[indices]]],axis=1)
