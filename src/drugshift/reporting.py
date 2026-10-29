"""Paper-ready figures (PNG 450 dpi + vector PDF), LaTeX/CSV tables, offline HTML report."""
from __future__ import annotations
from pathlib import Path
from html import escape
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

BG='#f7f9fc'; INK='#13243a'; BLUE='#3862b4'; PURPLE='#7257b2'; GREEN='#18978b'; AMBER='#d09136'
COLORS={'global_mean':'#94a3b8','cell_mean':'#a8b4c3','drug_mean':'#7d8fa8',
        'ridge':'#c38b43','random_forest':'#755ea2','hist_gradient_boosting':'#cb7799','ensemble_abs':'#3862b4',
        'ensemble_scaled':'#18978b','drugshift_uq':'#d16b53'}
LABELS={'global_mean':'Global mean','cell_mean':'Cell mean','drug_mean':'Drug mean',
        'ridge':'Ridge','random_forest':'Random forest','hist_gradient_boosting':'Histogram gradient boosting','ensemble_abs':'Cross-gated ensemble',
        'ensemble_scaled':'Ensemble-scaled UQ','drugshift_uq':'DrugShift-UQ',
        'no_gate':'No reciprocal gate','no_bilinear':'No bilinear term',
        'no_cell':'Without cell-line features','no_drug':'Without drug features',
        'early_fusion':'Early-fusion MLP'}
REGIMES={'pair':'Random held-out pairs','cold_cell':'Unseen cell lines',
         'cold_drug':'Unseen compounds','cold_both':'Both unseen'}

def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':13,
        'axes.titleweight':'bold','axes.labelcolor':INK,'text.color':INK,'axes.edgecolor':'#bbc7d7',
        'axes.spines.top':False,'axes.spines.right':False,
        'figure.facecolor':'white','axes.facecolor':'white','grid.color':'#e8edf5',
        'savefig.facecolor':'white','pdf.fonttype':42,'svg.fonttype':'none'})

def save(fig,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path.with_suffix('.png'),dpi=450,bbox_inches='tight',pad_inches=.24)
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight',pad_inches=.24)
    plt.close(fig)

def load_jobs(folder):
    records=[]
    for p in sorted((folder/'jobs').rglob('COMPLETE.json')):
        metrics=p.parent/'metrics.json'
        if not metrics.exists():continue
        payload=json.loads(metrics.read_text(encoding='utf8'))
        for method, m in payload['metrics'].items():
            if 'point' not in m:continue
            records.append({'regime':payload['regime'],'variant':payload['variant'],
                'seed':payload['seed'],'method':method,**m['point'],
                **{k:v for k,v in m['interval'].items() if k in ('coverage','mean_width','median_width')},
                'test_n':payload['n_test'],'train_n':payload['n_train'],
                'excluded_n':payload['n_excluded_cross_quadrant']})
    if not records:raise ValueError('No verified completed experiments')
    return pd.DataFrame(records)

def scatter_plot(run,figdir):
    source=run/'jobs'/'cold_both'/'seed_42'/'full'/'predictions.csv'
    if not source.exists():return
    df=pd.read_csv(source)
    if 'drugshift_uq_pred' not in df: return
    rng=np.random.default_rng(42)
    if len(df)>3000:df=df.iloc[rng.choice(len(df),3000,replace=False)]
    fig,ax=plt.subplots(figsize=(7,5.3))
    ax.scatter(df.actual,df.drugshift_uq_pred,s=13,alpha=.33,c=BLUE,edgecolors='none')
    lo=min(df.actual.min(),df.drugshift_uq_pred.min());hi=max(df.actual.max(),df.drugshift_uq_pred.max())
    ax.plot([lo,hi],[lo,hi],'--',c='#c16c5d',lw=1.4,label='Ideal prediction')
    ax.set(xlabel='Observed log(IC50)',ylabel='Predicted log(IC50)',title='Double-cold: observed versus predicted');ax.legend(frameon=False)
    save(fig,figdir/'observed_vs_predicted')
    if 'feature_shift_score' in df:
        fig,ax=plt.subplots(figsize=(7,5))
        ax.scatter(df.feature_shift_score,abs(df.actual-df.drugshift_uq_pred),s=12,alpha=.35,c=PURPLE,edgecolors='none')
        ax.set(xlabel='Feature distance to training entities',ylabel='Absolute prediction error',
               title='Distribution shift versus prediction error')
        save(fig,figdir/'shift_vs_error')

def train_curves(run,figdir):
    f=run/'jobs'/'cold_both'/'seed_42'/'full'/'training_curves.csv'
    if not f.exists():return
    df=pd.read_csv(f);fig,ax=plt.subplots(figsize=(7,4.7))
    for member,part in df.groupby('member'):
        ax.plot(part.epoch,part.validation_mse_scaled,alpha=.85,label=f'Member {member+1}')
    ax.set(xlabel='Epoch',ylabel='Validation MSE (standardized target)',
           title='Restart-safe training history');ax.legend(frameon=False)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True));save(fig,figdir/'learning_curves')

def main_benchmark(raw,figdir):
    df=raw[raw.variant=='full']
    regimes=list(REGIMES)
    methods=['ridge','random_forest','hist_gradient_boosting','ensemble_abs']
    fig,axes=plt.subplots(2,2,figsize=(13,8.7),sharey=False)
    for ax,regime in zip(axes.flat,regimes):
        sub=df[df.regime==regime]
        means=[sub[sub.method==m].RMSE.mean() for m in methods]
        errs=[sub[sub.method==m].RMSE.std(ddof=1) for m in methods]
        ax.bar(np.arange(len(methods)),means,color=[COLORS[m] for m in methods],width=.66)
        ax.errorbar(np.arange(len(methods)),means,yerr=np.nan_to_num(errs),fmt='none',ecolor='#273d5a',capsize=4)
        ax.set_xticks(np.arange(len(methods)),[LABELS[m] for m in methods],rotation=22,ha='right')
        ax.set_title(REGIMES[regime]);ax.set_ylabel('RMSE ↓')
        ax.grid(axis='y',alpha=.7);ax.set_axisbelow(True)
    fig.suptitle('Prediction generalization across four evaluation regimes',size=16,weight='bold')
    fig.tight_layout();save(fig,figdir/'benchmark_rmse')


def uq_figure(raw,figdir,alpha):
    df=raw[(raw.variant=='full')&raw.method.isin(['ensemble_abs','ensemble_scaled','drugshift_uq'])]
    fig,axes=plt.subplots(1,2,figsize=(12.5,4.8))
    names=['ensemble_abs','ensemble_scaled','drugshift_uq'];xs=np.arange(4);w=.25
    for i,method in enumerate(names):
        values=[df[(df.method==method)&(df.regime==rg)].coverage.mean() for rg in REGIMES]
        axes[0].bar(xs+(i-1)*w,values,width=w,color=COLORS[method],label=LABELS[method])
        values2=[df[(df.method==method)&(df.regime==rg)].mean_width.mean() for rg in REGIMES]
        axes[1].bar(xs+(i-1)*w,values2,width=w,color=COLORS[method],label=LABELS[method])
    axes[0].axhline(1-alpha,c=AMBER,ls='--',lw=1.5,label='Nominal coverage')
    for ax in axes:
        ax.set_xticks(xs,[REGIMES[r] for r in REGIMES],rotation=25,ha='right');ax.set_axisbelow(True);ax.grid(axis='y',alpha=.7)
    axes[0].set(ylabel='Test coverage ↑',title='Empirical prediction-interval coverage',ylim=(0,1.04))
    axes[1].set(ylabel='Mean interval width ↓',title='Prediction interval efficiency')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=4,bbox_to_anchor=(.5,1.06),frameon=False)
    fig.tight_layout();save(fig,figdir/'uncertainty_coverage_width')


def ablation_plot(raw,figdir):
    df=raw[(raw.method=='ensemble_abs')]
    regs=['cold_drug','cold_both'];variants=['full','no_gate','no_bilinear','no_cell','no_drug','early_fusion']
    fig,ax=plt.subplots(figsize=(10,5.2))
    x=np.arange(len(variants));w=.35
    for i,regime in enumerate(regs):
        vals=[df[(df.regime==regime)&(df.variant==v)&(df.seed==42)].RMSE.mean() for v in variants]
        ax.bar(x+(i-.5)*w,vals,width=w,color=[BLUE,PURPLE][i],label=REGIMES[regime])
    ax.set_xticks(x,[LABELS.get(v,'Full model') for v in variants],rotation=18,ha='right')
    ax.set(ylabel='RMSE ↓',title='Ablation study: contribution of architecture and modalities')
    ax.grid(axis='y',alpha=.6);ax.set_axisbelow(True);ax.legend(frameon=False)
    fig.tight_layout();save(fig,figdir/'ablation_rmse')


def latex_table(df,filename,caption,label,float_fmt='.3f'):
    # LaTeX source for authors; use escaping to avoid accidental command injection in names.
    txt=df.to_latex(index=False,float_format=lambda x:format(x,float_fmt),escape=True,
                    caption=caption,label=label)
    Path(filename).write_text(txt,encoding='utf8')

def tables(raw,run):
    dest=run/'tables';dest.mkdir(exist_ok=True)
    raw.sort_values(['regime','seed','variant','method']).to_csv(dest/'all_experiments.csv',index=False)
    desc=raw.groupby(['regime','variant','method']).agg(
        RMSE_mean=('RMSE','mean'),RMSE_sd=('RMSE','std'),MAE_mean=('MAE','mean'),
        Pearson_mean=('Pearson','mean'),Spearman_mean=('Spearman','mean'),
        coverage_mean=('coverage','mean'),mean_width_mean=('mean_width','mean'),
        seeds=('seed','nunique'),test_pairs=('test_n','mean')).reset_index()
    desc.to_csv(dest/'summary_by_regime.csv',index=False)
    main=desc[(desc.variant=='full')&(desc.method.isin(['global_mean','ridge','random_forest','hist_gradient_boosting','ensemble_abs']))]
    latex_table(main[['regime','method','RMSE_mean','RMSE_sd','MAE_mean','Pearson_mean','seeds']],
                dest/'table_1_generalization.tex','Drug-response prediction across held-out regimes','tab:generalization')
    uq=desc[(desc.variant=='full')&(desc.method.isin(['ensemble_abs','ensemble_scaled','drugshift_uq']))]
    latex_table(uq[['regime','method','coverage_mean','mean_width_mean','seeds']],
                dest/'table_2_uncertainty.tex','Empirical interval coverage and width; results are descriptive','tab:uncertainty')
    abl=raw[(raw.method=='ensemble_abs')&(raw.seed==42)&(raw.regime.isin(['cold_drug','cold_both']))]
    latex_table(abl[['regime','variant','RMSE','MAE','Pearson']],
                dest/'table_3_ablations.tex','Architecture and modality ablations','tab:ablations')
    return desc

CSS='''
:root{font-family:Inter,Segoe UI,Arial,sans-serif;color:#16273e;background:#f2f6fc}
*{box-sizing:border-box}body{margin:0}a{color:#395eb1;text-decoration:none}
header{background:linear-gradient(125deg,#142d56,#345ea5 60%,#806bb5);padding:55px max(20px,calc((100vw - 1160px)/2));color:#fff}
header h1{font-size:44px;letter-spacing:-1.3px;margin:6px 0}header p{color:#e4ebff;font-size:17px;max-width:850px}
.tags{display:flex;flex-wrap:wrap;gap:9px;margin-top:21px}.tag{border-radius:30px;border:1px solid #ffffff50;padding:7px 13px;background:#ffffff19;font-size:13px}
main{max-width:1160px;margin:30px auto;padding:0 18px 30px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:15px;margin-bottom:24px}
.card,.panel{background:#fff;border-radius:15px;border:1px solid #e0e9f4;box-shadow:0 8px 24px #19395c0b}
.card{padding:21px}.card .number{font-size:28px;font-weight:800;color:#204983;margin-bottom:5px}.label{font-size:12px;text-transform:uppercase;letter-spacing:.9px;color:#677b96}
.panel{padding:26px;margin:22px 0}.panel h2{margin:0 0 13px;font-size:22px;letter-spacing:-.5px}.panel p{line-height:1.65;color:#56677e}
.figures{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:18px}.figure{border:1px solid #e7edf6;border-radius:12px;padding:15px}
.figure img{width:100%;height:auto}.figure b{display:block;margin-bottom:9px;font-size:15px}.figure a{font-size:12px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;border-bottom:1px solid #e8edf5;padding:10px 8px}th{background:#f3f7fc;color:#2a476a}tr:nth-child(even){background:#fafcff}
.small{font-size:13px;color:#65758c}.foot{margin:35px 0;font-size:13px;color:#677789;line-height:1.7}code{background:#eff3f9;padding:2px 5px;border-radius:4px}
@media(max-width:680px){header h1{font-size:31px}.figures{grid-template-columns:1fr}.panel{padding:15px}}
'''

def html_frame(raw,summary,run_id,figprefix,tableprefix,alpha):
    full=raw[(raw.variant=='full')&(raw.method=='ensemble_abs')]
    d=full.groupby('regime').RMSE.agg(['mean','std'])
    cards=[('Experimental runs',str(raw[['regime','variant','seed']].drop_duplicates().shape[0])),
           ('Split protocols',str(full.regime.nunique())),
           ('Independent seeds',str(full.seed.nunique())),
           ('Nominal interval coverage',f'{100*(1-alpha):.0f}%')]
    cardhtml=''.join(f'<div class="card"><div class="number">{v}</div><div class="label">{escape(k)}</div></div>' for k,v in cards)
    images=[('Proposed research architecture','architecture'),('Generalization performance','benchmark_rmse'),('Calibration and interval width','uncertainty_coverage_width'),
            ('Architecture ablations','ablation_rmse'),('Observed versus predicted','observed_vs_predicted'),
            ('Shift versus error','shift_vs_error'),('Learning curves','learning_curves')]
    media=''.join(f'<div class="figure"><b>{escape(label)}</b><img loading="lazy" src="{figprefix}/{name}.png" alt="{escape(label)}">'
                  f'<a href="{figprefix}/{name}.pdf">Vector PDF</a> · <a href="{figprefix}/{name}.png">450 dpi PNG</a></div>'
                  for label,name in images)
    table=summary[(summary.variant=='full')&(summary.method.isin(['ridge','random_forest','hist_gradient_boosting','ensemble_abs','drugshift_uq']))].copy()
    for key in ['RMSE_mean','RMSE_sd','MAE_mean','coverage_mean','mean_width_mean']:
        table[key]=table[key].map(lambda v:'—' if not np.isfinite(v) else f'{v:.3f}')
    table['regime']=table.regime.map(REGIMES);table['method']=table.method.map(lambda x:LABELS.get(x,x))
    tablehtml=table[['regime','method','RMSE_mean','RMSE_sd','MAE_mean','coverage_mean','mean_width_mean','seeds']].to_html(index=False,escape=True)
    output=f'''<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DrugShift-UQ | Research Evidence Report</title><style>{CSS}</style></head><body>
<header><div class="small" style="color:#dbe6ff">REPRODUCIBLE RESEARCH / GDSC2</div><h1>DrugShift-UQ</h1>
<p>Uncertainty-aware drug response generalization across unfamiliar cancer cell lines and compounds. A research results report—not a clinical prediction tool.</p>
<div class="tags"><span class="tag">Leakage-audited splits</span><span class="tag">Checkpointed ensemble</span><span class="tag">Cross-modal architecture</span><span class="tag">Ablation studies</span><span class="tag">Offline report</span></div></header>
<main><div class="cards">{cardhtml}</div><section class="panel"><h2>Executive findings</h2>
<p>All values below were calculated from completed experiment files. RMSE is lower-is-better. Interval coverage refers to the <b>observed test fraction</b>, not a guaranteed population probability. The theoretical split-conformal guarantee requires exchangeability; it can fail under cold-drug and double-cold shifts.</p>
<p class="small">Run fingerprint: <code>{escape(run_id)}</code> · Dataset: pinned Zenodo record 21807175, GDSC2 · Endpoint: LN_IC50_curvecurator · Full model seeds: {full.seed.nunique()}.</p></section>
<section class="panel"><h2>Generalization & uncertainty</h2>{tablehtml}<p class="small">RMSE_sd reflects variation across run seeds where available. A low seed count does not establish statistical significance.</p></section>
<section class="panel"><h2>Publication-quality figures</h2><div class="figures">{media}</div></section>
<section class="panel"><h2>Download tables and evidence</h2><p><a href="{tableprefix}/all_experiments.csv">All experiment metrics (CSV)</a> · <a href="{tableprefix}/summary_by_regime.csv">Summary statistics (CSV)</a> · <a href="{tableprefix}/table_1_generalization.tex">Generalization table (LaTeX)</a> · <a href="{tableprefix}/table_2_uncertainty.tex">Uncertainty table (LaTeX)</a> · <a href="{tableprefix}/table_3_ablations.tex">Ablation table (LaTeX)</a></p></section>
<div class="foot"><b>Scientific limitations.</b> The architecture is a research hypothesis, not a proven first-in-literature invention. Cell-line sensitivity measurements do not establish clinical efficacy. Feature-space distance is not an OOD probability. The double-cold protocol excludes cross-quadrant pairs and is a particularly difficult, lower-sample evaluation. Split-conformal calibration is descriptive under distribution shifts. Cite the original dataset creators and describe the computational environment, seeds, missing-data handling and all exclusions.</div></main></body></html>'''
    return output

def generate_report(root,run_id):
    root=Path(root);run=root/'results/runs'/run_id;figdir=run/'figures';figdir.mkdir(parents=True,exist_ok=True)
    style();architecture_figure(figdir);raw=load_jobs(run);cfg=json.loads((run/'run_manifest.json').read_text())['config']
    if any(raw[(raw.variant=='full')&(raw.regime==r)].seed.nunique()!=len(cfg['seeds']) for r in cfg['regimes']):
        raise RuntimeError('Missing completed main-run seed; refusing to create a misleading final report')
    summary=tables(raw,run)
    main_benchmark(raw,figdir);uq_figure(raw,figdir,cfg['alpha']);ablation_plot(raw,figdir)
    scatter_plot(run,figdir);train_curves(run,figdir)
    (run/'REPORT.html').write_text(html_frame(raw,summary,run_id,'figures','tables',cfg['alpha']),encoding='utf8')
    root.joinpath('results/FINAL_REPORT.html').write_text(html_frame(raw,summary,run_id,
         f'runs/{run_id}/figures',f'runs/{run_id}/tables',cfg['alpha']),encoding='utf8')
    print('[report] Finished HTML dashboard, 450 dpi PNG/PDF figures and LaTeX tables.',flush=True)

# This function is invoked by generate_report after run metrics are available.
def architecture_figure(figdir):
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig,ax=plt.subplots(figsize=(12.5,6.1));ax.set_xlim(0,12);ax.set_ylim(0,6);ax.axis('off')
    fig.patch.set_facecolor('#ffffff')
    def node(x,y,w,h,label,sub,color):
        patch=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.13,rounding_size=0.15',
                            fc=color,ec='#d3deec',lw=1.2)
        ax.add_patch(patch);ax.text(x+w/2,y+h*.67,label,ha='center',va='center',size=11,
                color=INK,weight='bold');ax.text(x+w/2,y+h*.30,sub,ha='center',va='center',size=8.5,color='#526880')
    def link(x1,y1,x2,y2,color='#8093ab',style='-|>'):
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle=style,
                    mutation_scale=14,color=color,lw=1.4,connectionstyle='arc3,rad=0'))
    node(.2,4.55,4.25,.9,'Cancer cell-line expression','Fold-fitted selection + scaling','#e9f4ff')
    node(7.55,4.55,4.25,.9,'Drug molecular fingerprints','Train-only feature selection','#efeaff')
    node(.5,2.8,3.55,.95,'Cell encoder','64-dimensional learned embedding','#e9f4ff')
    node(7.95,2.8,3.55,.95,'Drug encoder','64-dimensional learned embedding','#efeaff')
    link(2.3,4.5,2.3,3.9);link(9.8,4.5,9.8,3.9)
    node(4.45,2.67,3.1,1.15,'Cross-gated interaction','Reciprocal gates + low-rank bilinear','#eaf7f4')
    link(4.1,3.22,4.33,3.22);link(7.85,3.22,7.65,3.22)
    node(4.45,1.16,3.1,.87,'Residual fusion head','Ensemble log(IC50) prediction','#f6f0e8')
    link(6.0,2.55,6.0,2.1)
    node(.22,.05,4.0,.82,'Held-out calibration','Absolute / ensemble-scaled intervals','#f2f6fc')
    node(7.76,.05,4.0,.82,'Shift-aware calibration','Training-entity feature distances','#f2f6fc')
    link(5.2,1.04,4.2,.55);link(6.8,1.04,7.75,.55)
    ax.text(6,5.83,'DrugShift-UQ: hypothesized cross-gated interaction architecture',
            ha='center',va='center',size=15,weight='bold',color=INK)
    save(fig,Path(figdir)/'architecture')
