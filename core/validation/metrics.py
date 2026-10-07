"""Honest forecast metrics in log-return units; undefined statistics stay null."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

QUANTILES = (.1,.25,.5,.75,.9)

def _vector(value, name, n=None):
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise ValueError(name+' must be a nonempty finite vector')
    if n is not None and len(result)!=n: raise ValueError(name+' length mismatch')
    return result

def _corr(a,b,rank=False):
    if len(a)<3 or np.std(a)==0 or np.std(b)==0: return None
    result = spearmanr(a,b).statistic if rank else np.corrcoef(a,b)[0,1]
    return float(result) if np.isfinite(result) else None

def forecast_metrics(actual, predicted, probability=None, quantiles=None):
    y = _vector(actual,'actual')
    p = _vector(predicted,'predicted',len(y))
    error = p-y
    truth, decision = y>0, p>0
    recalls = []
    result = {'n':len(y), 'mae':float(np.abs(error).mean()),
              'rmse':float(np.sqrt(np.mean(error**2))),
              'median_absolute_error':float(np.median(np.abs(error))),
              'bias':float(error.mean()), 'directional_accuracy':float(np.mean(np.sign(p)==np.sign(y))),
              'pearson_ic':_corr(y,p), 'spearman_ic':_corr(y,p,True),
              'information_coefficient':_corr(y,p,True), 'brier':None,'log_loss':None,
              'expected_calibration_error':None,'calibration':[],
              'coverage_50':None,'coverage_80':None,'width_50':None,'width_80':None,
              'coverage_90':None,'width_90':None}
    # q10/q90 define 80%, not 90%; unavailable 90% remains null.
    for positive,name in [(True,'up'),(False,'down')]:
        selected = decision==positive
        observed = truth==positive
        hits = np.sum(selected & observed)
        result['precision_'+name] = float(hits/selected.sum()) if selected.any() else None
        result['recall_'+name] = float(hits/observed.sum()) if observed.any() else None
        if observed.any(): recalls.append(hits/observed.sum())
    result['balanced_accuracy'] = float(np.mean(recalls)) if len(recalls)==2 else None
    if probability is not None:
        prob = _vector(probability,'probability',len(y))
        if np.any((prob<0)|(prob>1)): raise ValueError('Probabilities must be in [0,1]')
        result['brier'] = float(np.mean((prob-truth)**2))
        safe = np.clip(prob,1e-12,1-1e-12)
        result['log_loss'] = float(-np.mean(truth*np.log(safe)+(1-truth)*np.log(1-safe)))
        ece = 0.
        edges = np.array([0,.1,.2,.3,.4,.5,.55,.6,.65,.7,.75,.8,.9,1.])
        for i,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
            mask = (prob>=lo)&((prob<=hi) if i==len(edges)-2 else (prob<hi))
            count = int(mask.sum())
            avg = float(prob[mask].mean()) if count else None
            rate = float(truth[mask].mean()) if count else None
            result['calibration'].append({'lower':float(lo),'upper':float(hi),'n':count,
                                           'predicted':avg,'observed':rate})
            if count: ece += count/len(y)*abs(avg-rate)
        result['expected_calibration_error'] = float(ece)
    if quantiles is not None:
        q = np.asarray(quantiles,dtype=float)
        if q.shape!=(len(y),5) or not np.isfinite(q).all() or np.any(np.diff(q,axis=1)<0):
            raise ValueError('Quantiles must be finite ordered n x 5 values')
        for i,level in enumerate(QUANTILES):
            residual = y-q[:,i]
            result['pinball_q'+str(round(level*100))] = float(np.mean(np.maximum(level*residual,(level-1)*residual)))
        for coverage,left,right in [(50,1,3),(80,0,4)]:
            result['coverage_'+str(coverage)] = float(np.mean((y>=q[:,left])&(y<=q[:,right])))
            result['width_'+str(coverage)] = float(np.mean(q[:,right]-q[:,left]))
    return result

def grouped_metrics(rows: pd.DataFrame, dimensions):
    result = {}
    for dimension in dimensions:
        if dimension not in rows: continue
        groups = {}
        for key,group in rows.groupby(dimension,dropna=False,sort=True):
            probs = group['probability'].to_numpy() if 'probability' in group and group['probability'].notna().all() else None
            quantiles = group[['q10','q25','q50','q75','q90']].to_numpy() if all(c in group for c in ['q10','q25','q50','q75','q90']) else None
            groups[str(key)] = forecast_metrics(group['actual'],group['predicted'],probs,quantiles)
        result[dimension] = groups
    return result
