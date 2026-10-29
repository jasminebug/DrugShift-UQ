"""Point accuracy and marginal split-conformal evaluation; no invalid OOD guarantee claims."""
import numpy as np
from scipy.stats import pearsonr,spearmanr
from sklearn.neighbors import NearestNeighbors

def point(y,p):
    y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float)
    if len(y)!=len(p) or len(y)==0:raise ValueError('Empty / mismatched predictions')
    e=y-p
    def corr(fn):
        if len(y)<3 or np.std(y)<1e-10 or np.std(p)<1e-10:return np.nan
        return float(fn(y,p).statistic)
    return {'RMSE':float(np.sqrt(np.mean(e**2))),'MAE':float(np.mean(np.abs(e))),
            'Pearson':corr(pearsonr),'Spearman':corr(spearmanr),'n':len(y)}

def finite_sample_q(scores,alpha):
    scores=np.sort(np.asarray(scores,dtype=float))
    if len(scores)<10:raise ValueError('Need >=10 calibration samples')
    if not np.all(np.isfinite(scores)):raise ValueError('Non-finite calibration residuals')
    # (n+1) quantile, conservative in exchangeable calibration/test setting.
    index=min(len(scores)-1,int(np.ceil((len(scores)+1)*(1-alpha)))-1)
    return float(scores[index])

def interval_eval(y,lo,hi):
    y=np.asarray(y);lo=np.asarray(lo);hi=np.asarray(hi)
    return {'coverage':float(np.mean((y>=lo)&(y<=hi))),
            'mean_width':float(np.mean(hi-lo)),
            'median_width':float(np.median(hi-lo)),
            'n':len(y)}

def interval(ycal,pcal,ytest,ptest,alpha,scalecal=None,scaletest=None):
    if scalecal is None:scalecal=np.ones_like(pcal)
    if scaletest is None:scaletest=np.ones_like(ptest)
    if min(scalecal.min(),scaletest.min())<=0:raise ValueError('Nonpositive uncertainty scale')
    q=finite_sample_q(np.abs(ycal-pcal)/scalecal,alpha)
    lo=ptest-q*scaletest;hi=ptest+q*scaletest
    return {'q':q, **interval_eval(ytest,lo,hi)},lo,hi

def entity_shift(ft,train):
    vals=[]
    for vectors,rows in [(ft.cv,ft.cids),(ft.dv,ft.dids)]:
        seen=np.unique(rows[train])
        nn=NearestNeighbors(n_neighbors=1,metric='euclidean',algorithm='auto')
        nn.fit(vectors[seen]);dist=nn.kneighbors(vectors,return_distance=True)[0][:,0]
        vals.append((dist/max(1,np.sqrt(vectors.shape[1])))[rows])
    return np.asarray(vals[0]+vals[1],dtype=float)
