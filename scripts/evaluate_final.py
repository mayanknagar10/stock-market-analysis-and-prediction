"""One-shot frozen retrospective research comparison. Never tunes or promotes models."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.evaluate_v4 import read_snapshot
from core.forecasting.service import active_research,load_research_model
from core.forecasting.panel import build_panel
from core.forecasting.relative import relative_targets,predict_relative
from core.forecasting.trust import evidence_policy
from core.validation.alignment import aware_index,session_cutoffs
from core.validation.metrics import forecast_metrics,grouped_metrics
from core.data.snapshots import SnapshotStore

LOCK=ROOT/'reports/validation/FINAL_PROTOCOL_LOCK.json'
CONSUMED=ROOT/'reports/validation/FINAL_TEST_CONSUMED.json'
OUTPUT=ROOT/'reports/validation/FINAL_COMPARISON.json'

def final_rows(meta,boundary,capture_cutoff):
    dates=aware_index(meta.feature_time)
    ends=aware_index(meta.label_end)
    boundary=aware_index([boundary])[0]
    captured=aware_index([capture_cutoff])[0]
    return np.flatnonzero((dates>=boundary)&(ends<=captured))

def check_cutoffs(metadata,boundary):
    boundary=aware_index([boundary])[0]
    for name in ['training_cutoff','blending_cutoff','calibration_cutoff','validation_cutoff']:
        if name not in metadata or aware_index([metadata[name]])[0]>=boundary:
            raise ValueError('Candidate decision data reaches final boundary: '+name)
    if metadata.get('scope')!='research_only' or metadata.get('point_in_time_verified') is not False:
        raise ValueError('This protocol admits only explicitly unverified retrospective research candidates')

def digest(file): return hashlib.sha256(Path(file).read_bytes()).hexdigest()

def freeze(manifest_path,references_path):
    if CONSUMED.exists(): raise ValueError('Final period already consumed; do not reuse it for development')
    pointer=active_research()
    protocol=json.loads((ROOT/'config/v5_validation.json').read_text(encoding='utf-8'))
    boundary=pd.Timestamp(protocol['final_test_start'],tz='UTC')
    files={}
    for key,path in pointer['models'].items():
        directory=ROOT/path
        metadata=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))['metadata']
        check_cutoffs(metadata,boundary)
        family=metadata['family']; horizon=metadata['horizon']
        bundle=load_research_model(family,horizon,pointer)
        dummy=pd.DataFrame(np.zeros((1,len(bundle['regression'].feature_names))),columns=list(bundle['regression'].feature_names))
        bundle['regression'].predict(dummy)
        bundle['distribution'].predict(bundle['regression'].scaled(dummy))
        for file in directory.iterdir():
            if file.is_file(): files[str(file.relative_to(ROOT))]=digest(file)
    sources=[*sorted((ROOT/'core').rglob('*.py')),ROOT/'scripts/evaluate_final.py',ROOT/'config/context_sources.json',ROOT/'config/v5_validation.json']
    record={'model_version':pointer['model_version'],'models':pointer['models'],'final_test_start':protocol['final_test_start'],
        'capture_manifest':str(Path(manifest_path).resolve()),'capture_manifest_hash':digest(manifest_path),
        'references_manifest':str(Path(references_path).resolve()),'references_manifest_hash':digest(references_path),
        'artifact_hashes':files,'source_hashes':{str(p.relative_to(ROOT)):digest(p) for p in sources},
        'locked_at':datetime.now(timezone.utc).isoformat(),'scope':'research_only','point_in_time_verified':False,
        'decision_rule':{'required_mae_improvements':6,'required_rmse_improvements':6,'required_brier_below_coinflip':6,
            'required_80_coverage_within_10pp':6,'required_calibrated_heads':8},
        'v4_baseline':'Original V4 600-tree XGB/LGB configurations refit on globally pre-test data; one-day compounding baseline plus recursive matched-origin sample',
        'limitations':['Retrospective revised prices/reconstructed publication times; not certified PIT',
            'Current six-stock universe and sector map; survivorship and small-universe limits',
            'Overlapping labels/correlated assets; observations are not independent',
            'V4 refit is an architecture benchmark, not the missing historical production record']}
    LOCK.parent.mkdir(parents=True,exist_ok=True)
    with LOCK.open('x',encoding='utf-8') as stream: json.dump(record,stream,indent=2,sort_keys=True)
    return record

def fit_v4_refit(frames,boundary,source):
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.preprocessing import RobustScaler
    from core.models import UniversalPredictor,_build_training_row_set
    parts=[]; labels=[]
    for ticker,frame in frames.items():
        X,y=_build_training_row_set(frame)
        family=source['assets'][ticker]['family']
        times=session_cutoffs(frame.index,family)+pd.Timedelta('20min')
        ends=pd.Series(times,index=frame.index).shift(-1).reindex(X.index)
        origins=pd.Series(times,index=frame.index).reindex(X.index)
        mask=(origins<boundary)&(ends<boundary)
        parts.append(X.loc[mask]); labels.append(y.loc[mask])
    X=pd.concat(parts,ignore_index=True); y=pd.concat(labels,ignore_index=True)
    scaler=RobustScaler().fit(X)
    scaled=scaler.transform(X)
    common=dict(n_estimators=600,learning_rate=.03,max_depth=5,subsample=.8,colsample_bytree=.7,
                reg_alpha=.1,reg_lambda=1.5,n_jobs=2,random_state=42)
    first=xgb.XGBRegressor(**common,objective='reg:squarederror',verbosity=0).fit(scaled,y.to_numpy())
    second=lgb.LGBMRegressor(**common,num_leaves=31,objective='regression',verbose=-1).fit(scaled,y.to_numpy())
    predictor=UniversalPredictor()
    predictor.loaded=True; predictor.feature_names=list(X.columns)
    predictor.xgb_model=first.get_booster(); predictor.lgb_model=second.booster_
    predictor.scaler_center=np.asarray(scaler.center_,dtype=np.float32)
    predictor.scaler_scale=np.asarray(scaler.scale_,dtype=np.float32)
    return predictor,{'n_train':len(X),'families_pooled':True,'parameters':common,'training_label_cutoff_policy':'Strictly before final-test start'}

def evaluate(record):
    source,frames=read_snapshot(record['capture_manifest'])
    references_manifest=json.loads(Path(record['references_manifest']).read_text(encoding='utf-8'))
    store=SnapshotStore(ROOT/'data/v5/snapshots')
    references={k:store.load(v) for k,v in references_manifest['snapshots'].items()}
    settings=json.loads((ROOT/'config/context_sources.json').read_text(encoding='utf-8'))
    boundary=pd.Timestamp(record['final_test_start'],tz='UTC')
    captured=pd.Timestamp(source['fetched_at'])
    predictor,v4_metadata=fit_v4_refit(frames,boundary,source)
    from core.indicators import build_ml_features
    from core.models import _simulate_paths
    rows=[]; matched={}; relative_results={}; counts={}
    for family in ['india_equity','us_equity']:
        for horizon in [1,5,10,20]:
            bundle=load_research_model(family,horizon,{'model_version':record['model_version'],'models':record['models']})
            check_cutoffs(bundle['metadata'],boundary)
            X,y,meta=build_panel(frames,source,horizon,family,references,settings)
            eligible=final_rows(meta,boundary,captured)
            if not len(eligible): raise ValueError('No final samples for '+family+' '+str(horizon))
            Xtest=X.iloc[eligible][list(bundle['regression'].feature_names)]
            actual=y.iloc[eligible].to_numpy()
            scaled=bundle['regression'].scaled(Xtest)
            predicted=bundle['regression'].predict(Xtest)
            distribution=bundle['distribution'].predict(scaled)
            probability=bundle['calibrator'].predict(distribution['probability']) if bundle['calibrator'] is not None else distribution['probability']
            quantiles=distribution['quantiles'].copy()
            quantiles[:,0]-=bundle['conformal_margin']; quantiles[:,4]+=bundle['conformal_margin']
            ood=bundle['ood'].score(scaled)
            # Frozen original V4 feature schema; no contextual features or V5 preprocessing.
            v4_X=[]
            for _,sample in meta.iloc[eligible].iterrows():
                old=build_ml_features(frames[sample.ticker].loc[:pd.Timestamp(sample.origin_session)])
                row=old.iloc[[-1]].replace([np.inf,-np.inf],np.nan).ffill().fillna(0)
                v4_X.append(row[predictor.feature_names].to_numpy()[0])
            v4_one=predictor.predict_next_return_batch(np.asarray(v4_X,dtype=np.float32))
            v4_compounded=v4_one*horizon
            baseline_pre=np.flatnonzero((meta.feature_time<boundary)&(meta.label_end<boundary))
            drift=float(y.iloc[baseline_pre].mean())
            key=family+'_'+str(horizon)
            cohort=[]
            for position,index in enumerate(eligible):
                sample=meta.iloc[index]
                components=[float(v[position]) for v in predicted['components'].values()]
                trust=evidence_policy(float(probability[position]),float(predicted['central_log_return'][position]),quantiles[position],components,
                    True,float(ood[position]),bundle['metadata']['validation_n'],bundle['calibrator'] is not None,False,False,
                    learned_threshold=bundle['policy'].get('probability_edge_min'))
                base={'ticker':sample.ticker,'sector':sample.sector,'family':family,'horizon':horizon,
                      'feature_time':sample.feature_time.isoformat(),'label_end':sample.label_end.isoformat(),
                      'origin_session':sample.origin_session,'maturity_session':sample.maturity_session,
                      'year':int(sample.year),'market_regime':sample.market_regime,'actual':float(actual[position]),
                      'point_in_time_verified':False}
                for method,estimate in [('v5_direct',predicted['central_log_return'][position]),('zero_return',0.),
                    ('historical_drift',drift),('v4_refit_compounding',v4_compounded[position])]:
                    row={**base,'method':method,'predicted':float(estimate)}
                    if method=='v5_direct':
                        row.update(probability=float(probability[position]),probability_state='calibrated' if bundle['calibrator'] is not None else 'uncalibrated',
                            ood_ratio=float(ood[position]),confidence=trust['confidence'],abstain=trust['abstain'])
                        for col,val in zip(['q10','q25','q50','q75','q90'],quantiles[position]): row[col]=float(val)
                        if horizon==20: matched.setdefault(sample.ticker,[]).append(row)
                    rows.append(row); cohort.append(row)
            alpha=relative_targets(meta,y,references,settings,family)
            relative_results[key]={}
            for task,heads in bundle.get('auxiliary',{}).items():
                truth=alpha[task][eligible]; mask=np.isfinite(truth)
                if not mask.any(): continue
                p=np.mean([m.predict(scaled[mask]) for m in heads['regression'].values()],axis=0)
                raw=np.mean([m.predict(scaled[mask]) for m in heads['classifiers'].values()],axis=0)
                calibrated=heads['calibrator'].predict(raw) if heads['calibrator'] is not None else raw
                relative_results[key][task]=forecast_metrics(truth[mask],p,calibrated)
            counts[key]={'n':len(eligible),'n_ood':int((ood>1).sum()),'calibration_accepted':bundle['calibrator'] is not None}
            print(key,'final evaluation complete',flush=True)
    # Matched recursive V4 origins are chosen by a fixed rule, never performance.
    recursive=[]
    for ticker,candidates in matched.items():
        candidates=sorted(candidates,key=lambda r:r['feature_time'])
        selected=np.unique(np.linspace(0,len(candidates)-1,min(12,len(candidates)),dtype=int))
        frame=frames[ticker]
        for position in selected:
            sample=candidates[position]
            origin=pd.Timestamp(sample['origin_session'])
            prefix=frame.loc[:origin]
            sigma=float(np.log(prefix.Close).diff().dropna().std())
            seed=int(hashlib.sha256((ticker+sample['feature_time']+record['model_version']).encode()).hexdigest()[:8],16)
            paths=_simulate_paths(prefix,predictor,sigma,20,n_paths=30,seed=seed)
            price=float(prefix.Close.iloc[-1])
            for horizon in [1,5,10,20]:
                companion=next((r for r in rows if r['method']=='v5_direct' and r['ticker']==ticker and r['origin_session']==sample['origin_session'] and r['horizon']==horizon),None)
                if companion is None: continue
                row={**companion,'method':'v4_refit_recursive','predicted':float(np.log(np.median(paths[:,horizon-1])/price)),
                     'simulation_lower_80':float(np.log(np.percentile(paths[:,horizon-1],10)/price)),
                     'simulation_upper_80':float(np.log(np.percentile(paths[:,horizon-1],90)/price)),'seed':seed}
                for field in ['probability','q10','q25','q50','q75','q90']: row.pop(field,None)
                recursive.append(row)
        print(ticker,'matched recursive legacy diagnostic complete',flush=True)
    rows.extend(recursive)
    frame=pd.DataFrame(rows)
    report={'model_version':record['model_version'],'scope':'research_only','final_test_evaluated':True,
        'point_in_time_verified':False,'final_test_start':record['final_test_start'],'capture_cutoff':captured.isoformat(),
        'limitations':record['limitations'],'v4_refit':v4_metadata,'sample_counts':counts,'relative_metrics':relative_results,'results':{}}
    for (family,horizon,method),group in frame.groupby(['family','horizon','method']):
        probability=group.probability.to_numpy() if 'probability' in group and group.probability.notna().all() else None
        q=group[['q10','q25','q50','q75','q90']].to_numpy() if all(c in group for c in ['q10','q25','q50','q75','q90']) and group.q10.notna().all() else None
        metrics=forecast_metrics(group.actual,group.predicted,probability,q)
        if method=='v4_refit_recursive':
            metrics['simulation_coverage_80']=float(((group.actual>=group.simulation_lower_80)&(group.actual<=group.simulation_upper_80)).mean())
            metrics['simulation_width_80']=float((group.simulation_upper_80-group.simulation_lower_80).mean())
        report['results'][family+'_'+str(horizon)+'_'+method]={'aggregate':metrics,
            'breakdowns':grouped_metrics(group,['ticker','sector','year','market_regime'])}
    gates={'mae_improvements':0,'rmse_improvements':0,'brier_below_coinflip':0,'coverage_80_within_10pp':0,'calibrated_heads':0}
    for family in ['india_equity','us_equity']:
        for horizon in [1,5,10,20]:
            prefix=family+'_'+str(horizon)+'_'
            model=report['results'][prefix+'v5_direct']['aggregate']; naive=report['results'][prefix+'zero_return']['aggregate']
            gates['mae_improvements']+=int(model['mae']<naive['mae'])
            gates['rmse_improvements']+=int(model['rmse']<naive['rmse'])
            gates['brier_below_coinflip']+=int(model['brier']<.25)
            gates['coverage_80_within_10pp']+=int(.7<=model['coverage_80']<=.9)
            gates['calibrated_heads']+=int(counts[family+'_'+str(horizon)]['calibration_accepted'])
    report['frozen_research_gate_counts']=gates
    report['research_metric_gate_passed']=bool(gates['mae_improvements']>=6 and gates['rmse_improvements']>=6 and
        gates['brier_below_coinflip']>=6 and gates['coverage_80_within_10pp']>=6 and gates['calibrated_heads']==8)
    report['production_promotion']='BLOCKED: unverified historical vintages/session calendars; no production path in this release'
    report['abstention_rate']=float(frame.loc[frame.method=='v5_direct','abstain'].mean())
    return report,frame

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--references',type=Path,default=ROOT/'data/v5/REFERENCE_CAPTURE.json')
    parser.add_argument('--freeze-only',action='store_true')
    args=parser.parse_args()
    if not LOCK.exists(): record=freeze(args.manifest,args.references)
    else: record=json.loads(LOCK.read_text(encoding='utf-8'))
    if args.freeze_only: print('Protocol frozen for',record['model_version']); return
    if CONSUMED.exists() or OUTPUT.exists(): raise ValueError('Final period already consumed; refusal to rerun')
    if digest(args.manifest)!=record['capture_manifest_hash'] or digest(args.references)!=record['references_manifest_hash']:
        raise ValueError('Frozen input manifest changed')
    for relative,expected in {**record['artifact_hashes'],**record['source_hashes']}.items():
        if digest(ROOT/relative)!=expected: raise ValueError('Frozen source/artifact changed: '+relative)
    with CONSUMED.open('x',encoding='utf-8') as stream:
        json.dump({'model_version':record['model_version'],'consumed_at':datetime.now(timezone.utc).isoformat(),'state':'CONSUMED_ONCE'},stream,indent=2)
    report,rows=evaluate(record)
    with OUTPUT.open('x',encoding='utf-8') as stream: json.dump(report,stream,indent=2,allow_nan=False)
    with OUTPUT.with_suffix('.csv').open('x',encoding='utf-8',newline='') as stream: rows.to_csv(stream,index=False)
    print('Frozen comparison persisted:',OUTPUT,flush=True)

if __name__=='__main__': main()
