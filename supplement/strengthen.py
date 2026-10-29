"""DrugShift-UQ incremental, idempotent evidence strengthening.

This add-on never launches the 22-job main pipeline and never edits main source.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import sys
import time
import traceback
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

VERSION='1.0.0'
ESTIMATORS=('ridge','random_forest','hist_gradient_boosting')
ABLATIONS=('no_gate','no_bilinear','no_cell','no_drug','early_fusion')
REGIMES=('pair','cold_cell','cold_drug','cold_both')


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        while True:
            b=f.read(2*1024*1024)
            if not b:break
            h.update(b)
    return h.hexdigest()


def atomic_json(x,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(x,indent=2,allow_nan=False,default=lambda x:float(x) if isinstance(x,np.floating) else int(x) if isinstance(x,np.integer) else str(x)),encoding='utf-8')
    os.replace(temp,path)


def save_csv(df,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    df.to_csv(temp,index=False);os.replace(temp,path)


def checkpoint_complete(path,inputs=None):
    path=Path(path);mark=path/'COMPLETE.json'
    if not mark.exists():return False
    try:
        info=json.loads(mark.read_text())
        if inputs is not None and info.get('input_hash')!=inputs:return False
        return bool(info.get('checksums')) and all((path/f).is_file() and digest(path/f)==h for f,h in info['checksums'].items())
    except (ValueError,OSError,KeyError):return False


def write_complete(path,inputs,files):
    path=Path(path)
    atomic_json({'input_hash':inputs,'completed_utc':datetime.now(timezone.utc).isoformat(),
        'checksums':{f:digest(path/f) for f in files}},path/'COMPLETE.json')


def setup(project,need_data):
    project=Path(project).expanduser().resolve()
    sys.path.insert(0,str(project/'src'))
    source=project/'results/LATEST_RUN.txt'
    if not source.exists():raise RuntimeError('Missing results/LATEST_RUN.txt: original run must have finished.')
    run_id=source.read_text().strip()
    run=project/'results/runs'/run_id
    manifest=json.loads((run/'run_manifest.json').read_text())
    cfg=manifest['config']
    if run_id!=manifest['run_id']:raise RuntimeError('Latest run ID differs from run manifest')
    if sorted(cfg['regimes'])!=sorted(REGIMES):raise RuntimeError('Unexpected split protocols')
    if len(cfg['seeds'])!=3:raise RuntimeError('Expected the original three split seeds')
    for regime in cfg['regimes']:
        for seed in cfg['seeds']:
            p=run/'jobs'/regime/f'seed_{seed}'/'full'
            if not checkpoint_complete(p):raise RuntimeError(f'Original completed job missing or altered: {p}')
    if need_data:
        for file,sha in manifest['data_sha256'].items():
            p=project/'data/processed'/file
            if not p.is_file():raise RuntimeError(f'Missing processed data: {p}. Data was not included with results; retain original project folder.')
            print(f'[verify] SHA256 {file}',flush=True)
            if digest(p)!=sha:raise RuntimeError(f'Processed data changed since original experiments: {file}. STOP: cannot claim paired comparison.')
        for file,sha in manifest['code_sha256'].items():
            p=project/file
            if not p.is_file() or digest(p)!=sha:
                raise RuntimeError(f'Original research code differs from the recorded run: {file}. No files were changed. Restore code that generated this run.')
    out=project/'results'/'STRENGTHENED'/run_id
    out.mkdir(parents=True,exist_ok=True)
    return project,run,run_id,cfg,manifest,out


def assert_same_split(ds,s,old):
    sp=pd.read_csv(old/'split_manifest.csv',dtype={'cell_line_id':str,'drug_id':str})
    if len(sp)!=len(ds.rows) or not np.array_equal(sp.row_index.to_numpy(),np.arange(len(ds.rows))):
        raise RuntimeError('Split row indices no longer aligned with original run')
    if not np.array_equal(sp.cell_line_id.to_numpy(),ds.rows.cell_line_id.to_numpy()) or not np.array_equal(sp.drug_id.to_numpy(),ds.rows.drug_id.to_numpy()):
        raise RuntimeError('Split cell/drug identities changed')
    actual=np.full(len(ds.rows),'excluded',dtype='U12')
    for name in ('train','val','cal','test'):actual[getattr(s,name)]=name
    if not np.array_equal(actual,sp.partition.to_numpy()):raise RuntimeError('Regenerated split differs from original manifest')


def job_input_hash(data_hash,run_id,regime,seed,method,cfg):
    payload={'version':VERSION,'run_id':run_id,'data':data_hash,'regime':regime,
        'seed':seed,'method':method,'rf_trees':cfg['rf_trees'],'workers':cfg['workers'],
        'feature_settings':[cfg['genes'],cfg['drug_bits']],
        'sklearn':__import__('sklearn').__version__}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def compute_baseline(project,run,run_id,cfg,manifest,out,ds,regime,seed,method):
    from drugshift.data import FoldFeatures
    from drugshift.splits import make_split
    from drugshift.metrics import point,interval
    start=time.monotonic()
    old=run/'jobs'/regime/f'seed_{seed}'/'full'
    loc=out/'matched_baselines'/regime/f'seed_{seed}'/method
    ident=job_input_hash(manifest['data_sha256'],run_id,regime,seed,method,cfg)
    if checkpoint_complete(loc,ident):
        print(f'[skip] verified matched baseline {regime}/{seed}/{method}',flush=True);return
    loc.mkdir(parents=True,exist_ok=True)
    s=make_split(ds.rows,regime,seed)
    assert_same_split(ds,s,old)
    ft=FoldFeatures(cfg['genes'],cfg['drug_bits']).fit(ds,s.train)
    recorded=json.loads((old/'selected_features.json').read_text())
    if ft.columns!=recorded:raise RuntimeError('Selected training features no longer identical to original job')
    y=ds.rows.response.to_numpy(dtype='float32')
    ck=loc/'predictions.npz'
    if ck.exists():
        vals=np.load(ck);cp=vals['cal'];tp=vals['test']
        if len(cp)!=len(s.cal) or len(tp)!=len(s.test):raise RuntimeError('Cached prediction dimensions mismatch')
        print(f'[checkpoint] {regime}/{seed}/{method}: completed fit reused',flush=True)
    else:
        est={
          'ridge':Ridge(alpha=100.),
          'random_forest':RandomForestRegressor(n_estimators=cfg['rf_trees'],max_depth=18,min_samples_leaf=4,
                                             max_features=.5,n_jobs=cfg['workers'],random_state=seed),
          'hist_gradient_boosting':HistGradientBoostingRegressor(max_iter=90,max_depth=7,learning_rate=.07,
                                                                l2_regularization=1.,random_state=seed)
        }[method]
        # Same full training partitions, feature selection and parameters as original baseline.
        Xtr=ft.matrix(s.train)
        print(f'[fit] {regime} seed={seed} {method} on ALL {len(s.train):,} training pairs',flush=True)
        est.fit(Xtr,y[s.train]);del Xtr
        cp=est.predict(ft.matrix(s.cal));tp=est.predict(ft.matrix(s.test))
        tmp=ck.with_name(ck.name+'.tmp')
        with tmp.open('wb') as f:np.savez_compressed(f,cal=cp,test=tp)
        os.replace(tmp,ck)
    met,lo,hi=interval(y[s.cal],cp,y[s.test],tp,cfg['alpha'])
    reference=pd.read_csv(old/'predictions.csv',dtype={'cell_line_id':str,'drug_id':str})
    if not np.array_equal(reference.row_index.to_numpy(),s.test):
        raise RuntimeError('Original test predictions have a different row order')
    if not np.allclose(reference.actual.to_numpy(),y[s.test],rtol=0,atol=1e-6):
        raise RuntimeError('Original targets no longer aligned')
    save_csv(pd.DataFrame({'row_index':s.test,'cell_line_id':reference.cell_line_id,
                            'drug_id':reference.drug_id,'actual':y[s.test],
                            'prediction':tp,'lo':lo,'hi':hi}),loc/'predictions.csv')
    atomic_json({'method':method,'regime':regime,'seed':seed,'n_train':len(s.train),
                  'training_sample_fraction':1.0,'point':point(y[s.test],tp),
                  'interval':met,'elapsed_seconds':round(time.monotonic()-start,2),
                  'reuse_original_features':True,'equal_training_pairs_to_neural':True},loc/'metrics.json')
    write_complete(loc,ident,['metrics.json','predictions.csv','predictions.npz'])
    print(f'[complete] matched {regime}/{seed}/{method} RMSE={point(y[s.test],tp)["RMSE"]:.4f}',flush=True)


def extra_ablations(project,run,run_id,cfg):
    from drugshift.experiment import compute_job
    from drugshift.pipeline import stable_hash
    todo=[]
    for regime in cfg['ablation_regimes']:
        for seed in cfg['seeds']:
            for variant in cfg['ablations']:
                old=run/'jobs'/regime/f'seed_{seed}'/variant
                key=stable_hash({'run_id':run_id,'regime':regime,'variant':variant,'seed':seed})
                # Existing seed-42 results must be reused; never rerun them.
                if checkpoint_complete(old) and json.loads((old/'COMPLETE.json').read_text()).get('job_hash')==key:
                    continue
                todo.append((regime,variant,seed,key))
    print(f'[ablation] {len(todo)} additional jobs needed (prior jobs reused)',flush=True)
    for i,(regime,variant,seed,key) in enumerate(todo,1):
        print(f'[ablation progress] {i}/{len(todo)} {regime}/{variant}/{seed}',flush=True)
        compute_job(project,run_id,regime,variant,seed,cfg,key)


def extract_original(run,regime,seed):
    old=run/'jobs'/regime/f'seed_{seed}'/'full'
    p=pd.read_csv(old/'predictions.csv',dtype={'cell_line_id':str,'drug_id':str})
    needed=['actual','drugshift_uq_pred','ensemble_abs_pred','drugshift_uq_lo','drugshift_uq_hi',
            'ensemble_abs_lo','ensemble_abs_hi','row_index','cell_line_id','drug_id']
    if not all(k in p.columns for k in needed):raise RuntimeError(f'Missing original predictions: {old}')
    return p


def cluster_bootstrap_delta(y,p_a,p_b,groups,seed=0,reps=1200):
    """Paired cluster resampling of test entities: RMSE_A - RMSE_B.
    Descriptive CI conditional on one observed test split, not an independent dataset CI.
    """
    y=np.asarray(y,float);a=np.asarray(p_a,float);b=np.asarray(p_b,float)
    groups=np.asarray(groups).astype(str)
    uniq,idx=np.unique(groups,return_inverse=True)
    counts=np.bincount(idx).astype(float)
    sa=np.bincount(idx,weights=(a-y)**2)
    sb=np.bincount(idx,weights=(b-y)**2)
    if len(uniq)<4: return (np.nan,np.nan)
    rng=np.random.default_rng(seed)
    sims=[]
    for _ in range(reps):
        draw=rng.integers(0,len(uniq),size=len(uniq))
        n=counts[draw].sum()
        sims.append(float(np.sqrt(sa[draw].sum()/n)-np.sqrt(sb[draw].sum()/n)))
    return np.quantile(sims,[.025,.975]).tolist()


def interval_score(y,lo,hi,alpha=.1):
    y=np.asarray(y);lo=np.asarray(lo);hi=np.asarray(hi)
    return (hi-lo)+(2/alpha)*np.maximum(lo-y,0)+(2/alpha)*np.maximum(y-hi,0)


def analyze(run,out,cfg,baseline_required=True):
    from drugshift.metrics import point
    new=[];delta_rows=[];cover_rows=[];abl_rows=[];entity_rows=[]
    for regime in cfg['regimes']:
        for seed in cfg['seeds']:
            orig=extract_original(run,regime,seed)
            yy=orig.actual.to_numpy(float)
            base=orig.ensemble_abs_pred.to_numpy(float)
            key='cell_line_id' if regime=='cold_cell' else 'drug_id'
            for method in ESTIMATORS:
                p=out/'matched_baselines'/regime/f'seed_{seed}'/method
                if not checkpoint_complete(p):
                    if baseline_required:raise RuntimeError(f'Incomplete new baseline: {p}')
                    continue
                dat=pd.read_csv(p/'predictions.csv',dtype={'cell_line_id':str,'drug_id':str})
                if not np.array_equal(dat.row_index.to_numpy(),orig.row_index.to_numpy()) or not np.allclose(dat.actual,orig.actual,rtol=0,atol=1e-6):
                    raise RuntimeError(f'Matched baseline test rows/targets differ: {regime}/{seed}/{method}')
                pp=dat.prediction.to_numpy(float)
                r=point(yy,pp)
                orig_r=point(yy,base)
                ci=cluster_bootstrap_delta(yy,base,pp,orig[key],seed=seed,reps=1000)
                new.append({'regime':regime,'seed':seed,'method':method,'n_test':len(yy),
                            'n_train':json.loads((p/'metrics.json').read_text())['n_train'],
                            'RMSE':r['RMSE'],'MAE':r['MAE'],'Pearson':r['Pearson']})
                delta_rows.append({'regime':regime,'seed':seed,'method':method,'model_RMSE':orig_r['RMSE'],
                                   'matched_RMSE':r['RMSE'],'RMSE_model_minus_matched':orig_r['RMSE']-r['RMSE'],
                                   'cluster_axis':key,'CI_low':ci[0],'CI_high':ci[1],
                                   'note':'split-conditional, clustered descriptive CI; not across-dataset statistical inference'})
            for k in ('ensemble_abs','drugshift_uq','ensemble_scaled'):
                lo=orig[f'{k}_lo'].to_numpy(float);hi=orig[f'{k}_hi'].to_numpy(float)
                cover=((yy>=lo)&(yy<=hi)).astype(float)
                entity=orig.groupby(key,sort=True).indices
                cvals=[]
                for g,ix in entity.items():
                    if len(ix)>=10:
                        cvals.append(float(np.mean(cover[ix])))
                        entity_rows.append({'regime':regime,'seed':seed,'interval':k,'cluster_axis':key,
                                           'entity_id':g,'n':len(ix),'coverage':float(np.mean(cover[ix])),
                                           'mean_width':float(np.mean(hi[ix]-lo[ix]))})
                cover_rows.append({'regime':regime,'seed':seed,'interval':k,'coverage':float(np.mean(cover)),
                    'mean_width':float(np.mean(hi-lo)),'interval_score':float(np.mean(interval_score(yy,lo,hi,cfg['alpha']))),
                    'cluster_axis':key,'worst_decile_entity_coverage':float(np.quantile(cvals,.1)) if cvals else np.nan,
                    'n_entities_10plus':len(cvals)})
    for regime in cfg['ablation_regimes']:
        for seed in cfg['seeds']:
            old=run/'jobs'/regime/f'seed_{seed}'/'full'
            full=json.loads((old/'metrics.json').read_text())['metrics']['ensemble_abs']['point']['RMSE']
            for variant in cfg['ablations']:
                ab=run/'jobs'/regime/f'seed_{seed}'/variant
                if not checkpoint_complete(ab):raise RuntimeError(f'Ablation incomplete {ab}')
                m=json.loads((ab/'metrics.json').read_text())['metrics']['ensemble_abs']['point']['RMSE']
                abl_rows.append({'regime':regime,'seed':seed,'variant':variant,
                                 'full_RMSE':full,'ablated_RMSE':m,'ablation_minus_full':m-full})
    tables=out/'tables';tables.mkdir(exist_ok=True)
    frames={
        'matched_baselines':pd.DataFrame(new),
        'paired_model_vs_matched_baseline':pd.DataFrame(delta_rows),
        'uncertainty_diagnostics':pd.DataFrame(cover_rows),
        'entity_coverage':pd.DataFrame(entity_rows),
        'repeated_ablations':pd.DataFrame(abl_rows),
    }
    for name,df in frames.items():save_csv(df,tables/f'{name}.csv')
    # Summaries across the three observed split seeds; avoid false significance claims.
    summ=frames['matched_baselines'].groupby(['regime','method'],as_index=False).agg(RMSE_mean=('RMSE','mean'),RMSE_sd=('RMSE','std'),MAE_mean=('MAE','mean'))
    save_csv(summ,tables/'matched_summary.csv')
    ab=frames['repeated_ablations'].groupby(['regime','variant'],as_index=False).agg(mean_delta=('ablation_minus_full','mean'),sd_delta=('ablation_minus_full','std'),n=('seed','size'),ablation_wins=('ablation_minus_full',lambda x:int((x<0).sum())))
    save_csv(ab,tables/'ablation_summary.csv')
    cover=frames['uncertainty_diagnostics'].groupby(['regime','interval'],as_index=False).agg(coverage_mean=('coverage','mean'),width_mean=('mean_width','mean'),interval_score_mean=('interval_score','mean'),worst_decile_mean=('worst_decile_entity_coverage','mean'))
    save_csv(cover,tables/'uncertainty_summary.csv')
    # Directly reusable typeset tables for the conference manuscript.
    for name,frame in [('matched_summary',summ),('ablation_summary',ab),('uncertainty_summary',cover)]:
        (tables/(name+'.tex')).write_text(frame.to_latex(index=False,float_format='%.4f'),encoding='utf-8')
    return frames,summ,ab,cover


def figures(out,matched,ablations,cover):
    figdir=out/'figures';figdir.mkdir(exist_ok=True)
    def export(name):
        plt.tight_layout();plt.savefig(figdir/f'{name}.png',dpi=450,bbox_inches='tight')
        plt.savefig(figdir/f'{name}.pdf',bbox_inches='tight');plt.close()
    colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
    regs=list(REGIMES)
    fig,axs=plt.subplots(2,2,figsize=(11,7.2),sharey=False)
    for ax,reg in zip(axs.flat,regs):
        d=matched[matched.regime==reg]
        methods=list(ESTIMATORS)
        values=[float(d[d.method==m].RMSE_mean.iloc[0]) if len(d[d.method==m]) else np.nan for m in methods]
        ax.bar(['Ridge','Random forest','Histogram GB'],values,color=colors[:3]);ax.set_title(reg.replace('_',' ').title());ax.set_ylabel('Test RMSE (lower is better)')
        ax.tick_params(axis='x',labelrotation=12)
    fig.suptitle('Matched-data conventional baselines (full training partitions)',fontsize=15)
    export('matched_baselines_rmse')
    fig,axs=plt.subplots(1,2,figsize=(12,5),sharey=True)
    for ax,reg in zip(axs,('cold_drug','cold_both')):
        d=ablations[ablations.regime==reg]
        labels=list(ABLATIONS)
        vals=[float(d[d.variant==m].mean_delta.iloc[0]) for m in labels]
        bars=ax.bar(range(len(labels)),vals,color=[colors[2] if v>=0 else colors[3] for v in vals]);ax.set_xticks(range(len(labels)),labels,rotation=35,ha='right');ax.axhline(0,color='black',linewidth=.8)
        ax.set_title(reg.replace('_',' ').title());ax.set_ylabel('Ablated minus full RMSE (positive = full better)')
    fig.suptitle('Repeated architecture ablations across 3 seeds',fontsize=14)
    export('repeated_ablation_effects')
    fig,axs=plt.subplots(1,2,figsize=(11,4.3))
    pick=['ensemble_abs','ensemble_scaled','drugshift_uq']
    for k,axis in [('coverage_mean',axs[0]),('width_mean',axs[1])]:
        for j,method in enumerate(pick):
            vals=[float(cover[(cover.regime==reg)&(cover.interval==method)][k].iloc[0]) for reg in regs]
            axis.plot(range(4),vals,'o-',label=method.replace('_',' '),color=colors[j])
        axis.set_xticks(range(4),['Pair','Cold cell','Cold drug','Both unseen'])
        axis.tick_params(axis='x',labelrotation=20);axis.set_ylabel('Coverage' if k=='coverage_mean' else 'Mean interval width')
        if k=='coverage_mean':axis.axhline(.9,linestyle='--',color='black',alpha=.6)
    axs[1].legend(loc='best',fontsize=8);fig.suptitle('Conformal intervals: calibration vs efficiency',fontsize=14)
    export('uncertainty_tradeoffs')


def html_report(out,run_id,matched,ab,coverage,frames):
    fmt=lambda df:df.to_html(index=False,na_rep='—',float_format=lambda n:f'{n:.4f}',border=0,classes='dataframe')
    bits=[]
    bits.append('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DrugShift-UQ Strengthened Evidence</title>')
    bits.append('''<style>body{font-family:Arial,Segoe UI,sans-serif;background:#f4f7fd;color:#1a2c45;margin:0}header{background:#1c335a;color:#fff;padding:38px max(24px,calc((100vw - 1120px)/2))}main{max-width:1120px;margin:25px auto;padding:0 24px}section{background:white;border:1px solid #dae2ee;border-radius:12px;padding:22px;margin:18px 0;overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border-bottom:1px solid #e5eaf1;padding:9px;text-align:left}th{background:#edf2f9}img{max-width:100%;height:auto}p{line-height:1.65}.note{color:#555}a{color:#245a9f}</style>''')
    bits.append(f'<header><h1>DrugShift-UQ — Strengthened Evidence</h1><p>Matched-data baselines · repeated ablations · grouped coverage diagnostics · transparent limitations</p><small>Original immutable run: {html.escape(run_id)} | Incremental version {VERSION}</small></header><main>')
    bits.append('<section><h2>Scope and boundaries</h2><p>This is a supplementary analysis of the original completed run. Original neural networks and the 22 jobs were not retrained. New baseline fits use the entire original training split, and two additional seeds are trained for each of five ablations in each of two cold regimes. Baseline comparisons are restricted to the same splits and feature-selection protocol.</p>')
    bits.append('<p class="note"><b>Interpretation:</b> Three split seeds are not three independent cohorts. Bootstrap confidence intervals resample held-out entity groups within one split and are descriptive, not proof of population-level significance or clinical efficacy. Conformal coverage is empirical under covariate shift.</p></section>')
    for title,data in [('Matched baselines (3 seeds)',matched),('Repeated ablations (3 seeds)',ab),('Uncertainty diagnostics',coverage)]:
        bits.append(f'<section><h2>{title}</h2>{fmt(data)}</section>')
    for fn,title in [('matched_baselines_rmse','Matched baseline comparison'),('repeated_ablation_effects','Ablation effects'),('uncertainty_tradeoffs','Interval coverage versus width')]:
        bits.append(f'<section><h2>{title}</h2><img src="figures/{fn}.png" alt="{title}"><p><a href="figures/{fn}.pdf">Download vector PDF</a></p></section>')
    bits.append('<section><h2>Reproducibility artifacts</h2><p>CSV files in <code>tables/</code> include all per-seed predictions and statistics. Per-job checkpoints and hashes are in <code>matched_baselines/</code>; 20 supplementary ablation jobs are written to the existing run tree in unused variant/seed paths. A separate supplement manifest records hashes.</p>')
    bits.append('<p class="note">A stronger claim of methodological novelty still requires literature comparison. Do not call an improvement significant merely because a descriptive interval excludes zero. Negative results remain scientifically useful.</p></section></main></html>')
    (out/'STRENGTHENED_REPORT.html').write_text(''.join(bits),encoding='utf-8')


def execute(project,mode='all'):
    project,run,run_id,cfg,manifest,out=setup(project,need_data=mode=='all')
    print(f'[existing] verified 12 original completed main runs: {run_id}',flush=True)
    if mode=='all':
        from drugshift.data import read_dataset
        ds=read_dataset(project/'data/processed')
        print('[schedule] 12 split-seed groups × 3 full-data matched baselines + 20 supplementary ablation jobs',flush=True)
        for i,(reg,seed) in enumerate([(r,s) for s in cfg['seeds'] for r in cfg['regimes']],1):
            print(f'[matched progress] {i}/12 regime={reg} seed={seed}',flush=True)
            for m in ESTIMATORS:compute_baseline(project,run,run_id,cfg,manifest,out,ds,reg,seed,m)
        del ds
        extra_ablations(project,run,run_id,cfg)
    else:
        print('[audit-only] No new training requested; using only existing saved evidence',flush=True)
    frames,matched,ablations,coverage=analyze(run,out,cfg,baseline_required=True)
    figures(out,matched,ablations,coverage)
    html_report(out,run_id,matched,ablations,coverage,frames)
    data={'version':VERSION,'original_run_id':run_id,
        'original_run_manifest_sha256':digest(run/'run_manifest.json'),
        'original_job_metrics_sha256':{f'{r}_{s}':digest(run/'jobs'/r/f'seed_{s}'/'full'/'metrics.json') for r in cfg['regimes'] for s in cfg['seeds']},
        'addon_sha256':digest(Path(__file__).resolve()),
        'supplementary_baselines':int(len(frames['matched_baselines'])),
        'supplementary_ablations':int(len(frames['repeated_ablations'])),
        'completed_utc':datetime.now(timezone.utc).isoformat(),
        'warnings':['Evaluation is preclinical cell-line screening only.',
                    'Seed variation does not imply independent replication.',
                    'Cluster bootstrap uses one-axis groups, not simultaneous cell/drug clustering.',
                    'Original model remained unchanged; no post-hoc model or hyperparameter retuning.']}
    atomic_json(data,out/'STRENGTHENING_MANIFEST.json')
    print(f'[DONE] {out/"STRENGTHENED_REPORT.html"}',flush=True)
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project',required=True)
    p.add_argument('--mode',choices=['all','audit'],default='all')
    a=p.parse_args()
    logs=Path(a.project).expanduser().resolve()/'logs'
    logs.mkdir(parents=True,exist_ok=True)
    class Tee:
        def __init__(self,main,log):self.main=main;self.log=log
        def write(self,value):self.main.write(value);self.log.write(value);self.log.flush()
        def flush(self):self.main.flush();self.log.flush()
    original_out,original_err=sys.stdout,sys.stderr
    with (logs/'strengthening.log').open('a',encoding='utf-8',buffering=1) as f:
        sys.stdout=Tee(original_out,f);sys.stderr=Tee(original_err,f)
        try:
            print('\n===== DrugShift-UQ incremental run: '+datetime.now(timezone.utc).isoformat()+' =====',flush=True)
            execute(a.project,a.mode)
        except Exception:
            traceback.print_exc()
            raise
        finally:
            sys.stdout,sys.stderr=original_out,original_err

if __name__=='__main__':main()
