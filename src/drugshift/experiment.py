"""Individual auditable, idempotent experiment jobs and checkpointed training."""
import hashlib
import json
import os
from pathlib import Path
import time
from .prepare import checksum
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from .data import read_dataset,FoldFeatures
from .splits import make_split,write_manifest
from .models import fit_member,predict
from .metrics import point,interval,entity_shift


def _clean_json(value):
    if isinstance(value,dict): return {str(k):_clean_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [_clean_json(v) for v in value]
    if isinstance(value,(np.integer,)): return int(value)
    if isinstance(value,(np.floating,float)):
        value=float(value)
        return value if np.isfinite(value) else None
    if isinstance(value,np.ndarray): return _clean_json(value.tolist())
    return value

def atomic_json(data,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(_clean_json(data),indent=2,allow_nan=False),encoding='utf8')
    os.replace(tmp,path)

def json_safe(v):
    if isinstance(v,np.integer):return int(v)
    if isinstance(v,np.floating):return float(v)
    if isinstance(v,np.ndarray):return v.tolist()
    raise TypeError(str(type(v)))

def summarize_unseen(data,s):
    train=data.rows.iloc[s.train];test=data.rows.iloc[s.test]
    return {
      'test_unseen_cell_fraction':float((~test.cell_line_id.isin(set(train.cell_line_id))).mean()),
      'test_unseen_drug_fraction':float((~test.drug_id.isin(set(train.drug_id))).mean()),
      'train_cell_entities':int(train.cell_line_id.nunique()),'train_drug_entities':int(train.drug_id.nunique()),
      'test_cell_entities':int(test.cell_line_id.nunique()),'test_drug_entities':int(test.drug_id.nunique())}

def compute_job(root,run_id,regime,variant,seed,cfg,expected_hash):
    start=time.time()
    out=Path(root)/'results'/'runs'/run_id/'jobs'/regime/f'seed_{seed}'/variant
    done=out/'COMPLETE.json'
    if done.exists():
        try:
            marker=json.loads(done.read_text(encoding='utf8'))
            if (marker.get('job_hash')==expected_hash and marker.get('checksums') and
                all((out/k).exists() and checksum(out/k,'sha256')==v for k,v in marker['checksums'].items())):
                print(f'[skip] already complete: {regime} seed={seed} {variant}',flush=True)
                return out
        except (ValueError,OSError):pass
    out.mkdir(parents=True,exist_ok=True)
    print(f'\n[JOB] {regime} seed={seed} {variant}',flush=True)
    ds=read_dataset(Path(root)/'data/processed')
    s=make_split(ds.rows,regime,seed)
    ft=FoldFeatures(cfg['genes'],cfg['drug_bits']).fit(ds,s.train)
    y=ds.rows.response.to_numpy('float32')
    # Save exact split and feature schema for audit (no target-guided feature selection).
    write_manifest(ds.rows,s,out/'split_manifest.csv')
    atomic_json(ft.columns,out/'selected_features.json')
    if variant=='no_cell':ft.cv[:]=0
    if variant=='no_drug':ft.dv[:]=0
    test_frame=ds.rows.iloc[s.test].copy().rename(columns={'response':'actual'})
    test_frame.insert(0,'row_index',s.test)
    test_frame=test_frame.reset_index(drop=True)
    all_results={}

    def report(key,cal_pred,test_pred,scalecal=None,scaletest=None):
        nonlocal test_frame
        met,lo,hi=interval(y[s.cal],cal_pred,y[s.test],test_pred,cfg['alpha'],scalecal,scaletest)
        all_results[key]={'point':point(y[s.test],test_pred),'interval':met}
        test_frame[key+'_pred']=np.asarray(test_pred)
        test_frame[key+'_lo']=lo;test_frame[key+'_hi']=hi
        return met

    if variant=='full':
        mean=float(y[s.train].mean())
        report('global_mean',np.full(len(s.cal),mean),np.full(len(s.test),mean))
        # Naive entity mean baselines use training entities only; unseen entities fall back to global mean.
        train_table=ds.rows.iloc[s.train]
        for name,group in [('cell_mean','cell_line_id'),('drug_mean','drug_id')]:
            means=train_table.groupby(group)['response'].mean()
            report(name,ds.rows.iloc[s.cal][group].map(means).fillna(mean).to_numpy(),
                   ds.rows.iloc[s.test][group].map(means).fillna(mean).to_numpy())
        rng=np.random.default_rng(seed)
        sub=rng.choice(s.train,size=min(len(s.train),cfg['baseline_rows']),replace=False)
        Xtr=ft.matrix(sub);Xcal=ft.matrix(s.cal);Xtest=ft.matrix(s.test)
        # Persist baseline outputs as soon as each fit finishes, because RF/HGB fits do not
        # have native epoch checkpoints. On power loss only an unfinished fit is repeated.
        for name,est in [
            ('ridge',Ridge(alpha=100.)),
            ('random_forest',RandomForestRegressor(n_estimators=cfg['rf_trees'],max_depth=18,
                   min_samples_leaf=4,max_features=0.5,n_jobs=cfg['workers'],random_state=seed)),
            ('hist_gradient_boosting',HistGradientBoostingRegressor(max_iter=90,max_depth=7,
                   learning_rate=.07,l2_regularization=1.,random_state=seed))]:
            pred_file=out/'checkpoints'/f'{name}_predictions.npz'
            if pred_file.exists():
                cached=np.load(pred_file)
                cal_out=cached['cal'];test_out=cached['test']
                if len(cal_out)!=len(s.cal) or len(test_out)!=len(s.test):
                    raise RuntimeError('Baseline cached outputs mismatch current split')
                print(f'[checkpoint] Reusing saved {name} calibration/test predictions',flush=True)
            else:
                est.fit(Xtr,y[sub]);cal_out=est.predict(Xcal);test_out=est.predict(Xtest)
                pred_file.parent.mkdir(parents=True,exist_ok=True)
                temp=pred_file.with_name(pred_file.name+'.tmp')
                with open(temp,'wb') as f: np.savez_compressed(f,cal=cal_out,test=test_out)
                os.replace(temp,pred_file)
            report(name,cal_out,test_out)
            print(f'[baseline] {name} complete',flush=True)
        del Xtr,Xcal,Xtest
    cal_draws=[];test_draws=[];curves=[]
    for m in range(cfg['members']):
        mem=fit_member(ft,y,s.train,s.val,
             checkpoint=out/'checkpoints'/f'member_{m}.pt',variant=variant,
             seed=seed+1009*m,epochs=cfg['epochs'],patience=cfg['patience'],
             batch_size=cfg['batch_size'],device=cfg['device'],
             config_tag=expected_hash+f'_member_{m}')
        cal_draws.append(predict(mem,ft,s.cal));test_draws.append(predict(mem,ft,s.test))
        for ep in mem['history']:curves.append({'member':m,**ep})
        del mem
        import gc;gc.collect()
        if cfg['device']=='cuda':
            import torch;torch.cuda.empty_cache()
    pd.DataFrame(curves).to_csv(out/'training_curves.csv',index=False)
    C=np.stack(cal_draws);T=np.stack(test_draws)
    cal=C.mean(0);test=T.mean(0)
    report('ensemble_abs',cal,test)
    if variant=='full':
        floor=max(0.08*float(np.std(y[s.train])),0.05)
        cal_scale=np.sqrt(np.var(C,axis=0)+floor**2)
        test_scale=np.sqrt(np.var(T,axis=0)+floor**2)
        report('ensemble_scaled',cal,test,cal_scale,test_scale)
        novelty=entity_shift(ft,s.train)
        cal_dist=novelty[s.cal];test_dist=novelty[s.test]
        # Calibration-only scale reference, never calculated from test labels.
        reference=max(float(np.median(cal_dist)),0.05)
        cal_modifier=1+cal_dist/reference;test_modifier=1+test_dist/reference
        report('drugshift_uq',cal,test,cal_scale*cal_modifier,test_scale*test_modifier)
        test_frame['feature_shift_score']=test_dist
        test_frame['ensemble_std']=T.std(axis=0)
        all_results['calibration_summary']={
            'reference_shift_median_cal':reference,'scale_floor':floor,
            'calibration_shift_mean':float(cal_dist.mean()),
            'test_shift_mean':float(test_dist.mean()),
            'note':'No marginal coverage guarantee under non-exchangeable dataset shift.'}
    else:
        # Enables comparable architecture ablation accuracy and absolute conformal intervals.
        test_frame['ablation_variant']=variant
    meta={
      'dataset':'GDSC2','target':cfg['target'], 'regime':regime, 'variant':variant,'seed':seed,
      'n_train':len(s.train),'n_validation':len(s.val),'n_calibration':len(s.cal),
      'n_test':len(s.test),'n_excluded_cross_quadrant':len(s.excluded),
      'n_cell_features':ft.cv.shape[1],'n_drug_features':ft.dv.shape[1],
      'members':cfg['members'],'epochs_max':cfg['epochs'],'alpha':cfg['alpha'],
      'unseen':summarize_unseen(ds,s),'metrics':all_results,'job_hash':expected_hash,
      'training_wall_seconds':time.time()-start,
      'caveats':['Cell-line assays are not patient treatment outcomes.',
                 'Cold-both excludes cross-quadrant drug-cell pairs.',
                 'Split-conformal theoretical coverage does not transfer automatically to shifted test distributions.']}
    # Commit outputs only after both data files and metrics are fully written.
    tmp=out/'predictions.csv.tmp';test_frame.to_csv(tmp,index=False);os.replace(tmp,out/'predictions.csv')
    atomic_json(meta,out/'metrics.json')
    atomic_json({'job_hash':expected_hash,'completed_at_utc':pd.Timestamp.utcnow().isoformat(),
                 'elapsed_seconds':time.time()-start,
                 'checksums':{name:checksum(out/name,'sha256') for name in
                     ('metrics.json','predictions.csv','split_manifest.csv')}},done)
    print(f'[complete] {regime} {variant} seed={seed} RMSE={all_results["ensemble_abs"]["point"]["RMSE"]:.4f}',flush=True)
    return out
