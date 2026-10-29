"""TESTING ONLY. Not scientific results. Safe for offline integration tests."""
from pathlib import Path
import numpy as np
import pandas as pd

def generate(root,nc=45,nd=28):
    root=Path(root);p=root/'data/processed';p.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(42)
    cell=pd.DataFrame(rng.normal(size=(nc,24)),columns=[f'g{i}' for i in range(24)])
    cell.insert(0,'cell_line_id',[f'c{i}' for i in range(nc)])
    drug=pd.DataFrame(rng.integers(0,2,size=(nd,32)),columns=[f'bit{i}' for i in range(32)])
    drug.insert(0,'drug_id',[f'd{i}' for i in range(nd)])
    pairs=pd.DataFrame([(a,b) for a in cell.cell_line_id for b in drug.drug_id],columns=['cell_line_id','drug_id'])
    pairs['response']=(np.repeat(cell.g0.to_numpy(),nd)+np.tile(drug.bit0.to_numpy(),nc)
                       +rng.normal(0,.25,len(pairs)))
    cell.to_csv(p/'cell_features.csv',index=False)
    drug.to_csv(p/'drug_features.csv',index=False)
    pairs.to_csv(p/'responses.csv',index=False)
    print('SYNTHETIC DATA GENERATED: strictly for software testing; never for manuscript metrics')
if __name__=='__main__':
    import sys
    generate(sys.argv[1] if len(sys.argv)>1 else '.')
