"""Explicit purged research training. Selection, blending and calibration precede final test."""
from datetime import datetime,timezone
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from core.forecasting.heads import DirectRegressor
from core.forecasting.distribution import DistributionHeads,LEVELS
from core.forecasting.trust import PlattCalibrator,OODDetector,learn_edge_threshold
from core.validation.splits import purged_expanding_splits,assert_disjoint_label_intervals
from core.validation.metrics import forecast_metrics,grouped_metrics

# Freeze the source identity once at job import; edits cannot change mid-run provenance.
CODE_VERSION=hashlib.sha256(b''.join(p.read_bytes() for p in sorted(Path(__file__).parents[1].rglob('*.py')))).hexdigest()

def select_feature_groups(X,y,meta,family,horizon,group_map):
    stock=group_map.get('stock') or [c for c in X if not any(c.startswith(p) for p in ['market_','sector_','nasdaq_'])]
    folds=list(purged_expanding_splits(meta,252,63))[-3:]
    if len(folds)<2: raise ValueError('Insufficient pre-calibration folds for feature selection')
    choices={'stock':stock}
    for group,columns in group_map.items():
        if group!='stock' and columns: choices[group]=stock+columns
    scores={}
    for group,columns in choices.items():
        errors=[]
        for train,test in folds:
            model=DirectRegressor(family,horizon,n_estimators=80).fit(X.iloc[train][columns],y.iloc[train],
                meta.iloc[train].feature_time,meta.iloc[train].label_end,meta.iloc[test].feature_time.min())
            predicted=model.predict(X.iloc[test][columns])['central_log_return']
            errors.append(float(np.mean(np.abs(predicted-y.iloc[test].to_numpy()))))
        scores[group]=errors
    kept=['stock']
    baseline=np.asarray(scores['stock'])
    for group,errors in scores.items():
        if group!='stock' and np.mean(errors)<np.mean(baseline) and np.sum(np.asarray(errors)<baseline)>=2:
            kept.append(group)
    columns=[]
    for group in kept:
        for column in group_map[group]:
            if column not in columns: columns.append(column)
    return columns,{'fold_mae':scores,'kept_groups':kept,'selection_rule':'MAE improves overall and in at least two pre-calibration folds'}

def train_research_bundle(X,y,meta,family,horizon,version,source_provenance,group_map,relative=None):
    calendar=pd.DatetimeIndex(meta.feature_time).unique().sort_values()
    if len(calendar)<550: raise ValueError('At least 550 development sessions required')
    validation_start=calendar[-63]
    calibration_start=calendar[-126]
    blending_start=calendar[-189]
    base=np.flatnonzero((meta.feature_time<blending_start)&(meta.label_end<blending_start))
    blending=np.flatnonzero((meta.feature_time>=blending_start)&(meta.feature_time<calibration_start)&(meta.label_end<calibration_start))
    calibration=np.flatnonzero((meta.feature_time>=calibration_start)&(meta.feature_time<validation_start)&(meta.label_end<validation_start))
    validation=np.flatnonzero(meta.feature_time>=validation_start)
    assert_disjoint_label_intervals(meta.iloc[base],meta.iloc[blending])
    assert_disjoint_label_intervals(meta.iloc[blending],meta.iloc[calibration])
    assert_disjoint_label_intervals(meta.iloc[calibration],meta.iloc[validation])
    columns,ablation=select_feature_groups(X.iloc[base].reset_index(drop=True),y.iloc[base].reset_index(drop=True),
        meta.iloc[base].reset_index(drop=True),family,horizon,group_map)
    X=X[columns]
    regression=DirectRegressor(family,horizon).fit(X.iloc[base],y.iloc[base],meta.iloc[base].feature_time,
        meta.iloc[base].label_end,blending_start)
    regression.learn_weights(X.iloc[blending],y.iloc[blending])
    distribution=DistributionHeads().fit(regression.scaled(X.iloc[base]),y.iloc[base])
    scaled_cal=regression.scaled(X.iloc[calibration])
    scaled_blend=regression.scaled(X.iloc[blending])
    blend_labels=(y.iloc[blending].to_numpy()>0).astype(int)
    labels=(y.iloc[calibration].to_numpy()>0).astype(int)
    class_errors=[]
    quantile_errors=[]
    for kind in ['xgb','lgb']:
        probability=distribution.classifiers[kind].predict_proba(scaled_blend)[:,1]
        class_errors.append(float(np.mean((probability-blend_labels)**2)))
        loss=[]
        for level in LEVELS:
            predicted=distribution.quantile_models[str(level)][kind].predict(scaled_blend)
            residual=y.iloc[blending].to_numpy()-predicted
            loss.append(float(np.mean(np.maximum(level*residual,(level-1)*residual))))
        quantile_errors.append(np.mean(loss))
    inverse=1/np.maximum(class_errors,1e-12)
    distribution.class_weights=inverse/inverse.sum()
    inverse=1/np.maximum(quantile_errors,1e-12)
    distribution.weights=inverse/inverse.sum()
    cal_distribution=distribution.predict(scaled_cal)
    calibrator=PlattCalibrator().fit(cal_distribution['probability'],labels,meta.iloc[calibration].feature_time,
                                   meta.iloc[blending].label_end.max())
    cal_q=cal_distribution['quantiles']
    nonconformity=np.maximum(cal_q[:,0]-y.iloc[calibration].to_numpy(),y.iloc[calibration].to_numpy()-cal_q[:,4])
    quantile=min(1.,np.ceil((len(calibration)+1)*.8)/len(calibration))
    margin=float(max(0.,np.quantile(nonconformity,quantile,method='higher')))
    validation_return=regression.predict(X.iloc[validation])['central_log_return']
    raw=distribution.predict(regression.scaled(X.iloc[validation]))
    quantiles=raw['quantiles'].copy()
    quantiles[:,0]-=margin
    quantiles[:,4]+=margin
    calibrated=calibrator.predict(raw['probability'])
    raw_metrics=forecast_metrics(y.iloc[validation],validation_return,raw['probability'],quantiles)
    calibrated_metrics=forecast_metrics(y.iloc[validation],validation_return,calibrated,quantiles)
    accepted=calibrated_metrics['brier']<=raw_metrics['brier'] and calibrated_metrics['log_loss']<=raw_metrics['log_loss']
    if not accepted: calibrator=None; calibrated=raw['probability']
    metrics=calibrated_metrics if accepted else raw_metrics
    policy=learn_edge_threshold(y.iloc[validation],calibrated,validation_return)
    rows=meta.iloc[validation].copy()
    rows['actual']=y.iloc[validation].to_numpy()
    rows['predicted']=validation_return
    rows['probability']=calibrated
    for i,name in enumerate(['q10','q25','q50','q75','q90']): rows[name]=quantiles[:,i]
    detector=OODDetector().fit(regression.scaled(X.iloc[base]))
    created=datetime.now(timezone.utc).isoformat()
    metadata={'model_version':version,'family':family,'horizon':horizon,'feature_version':'v5-research-features-2',
        'feature_schema_hash':regression.schema_hash,'feature_names':columns,'target_version':'v5-session-labels-3',
        'training_start':meta.iloc[base].feature_time.min().isoformat(),
        'training_cutoff':meta.iloc[base].label_end.max().isoformat(),
        'blending_start':meta.iloc[blending].feature_time.min().isoformat(),
        'blending_cutoff':meta.iloc[blending].label_end.max().isoformat(),
        'calibration_start':meta.iloc[calibration].feature_time.min().isoformat(),
        'calibration_cutoff':meta.iloc[calibration].label_end.max().isoformat(),
        'validation_start':validation_start.isoformat(),'validation_cutoff':meta.iloc[validation].label_end.max().isoformat(),
        'training_universe':sorted(meta.iloc[base].ticker.unique().tolist()),'created_at':created,
        'parameters':{'trees':120,'depth':3,'learning_rate':.04,'seed':42},
        'validation_metrics':metrics,'raw_probability_metrics':raw_metrics,'calibration_accepted':bool(accepted),
        'validation_n':len(validation),'point_in_time_verified':False,'final_test_passed':False,
        'ablation':ablation,'feature_groups':{g:[c for c in values if c in columns] for g,values in group_map.items()},
        'source_provenance':source_provenance,'scope':'research_only',
        'limitation':'Retrospective current-universe price vintage; source availability reconstructed; no certified PIT or production promotion'}
    metadata['code_version']=CODE_VERSION
    import xgboost,lightgbm,sklearn
    metadata['library_versions']={'xgboost':xgboost.__version__,'lightgbm':lightgbm.__version__,'sklearn':sklearn.__version__,'pandas':pd.__version__,'numpy':np.__version__}
    analogues={'features':regression.scaled(X.iloc[base]),'returns':y.iloc[base].to_numpy(),
        'label_ends':np.asarray([t.isoformat() for t in meta.iloc[base].label_end],dtype='U40'),
        'timestamps':np.asarray([t.isoformat() for t in meta.iloc[base].feature_time],dtype='U40'),
        'tickers':np.asarray(meta.iloc[base].ticker.to_list(),dtype='U32')}
    auxiliary={}
    if relative is not None:
        for task,target in relative.items(): analogues['excess_'+task]=np.asarray(target)[base]
        from core.forecasting.relative import train_relative_heads
        for task,target in relative.items():
            heads=train_relative_heads(regression.scaled(X),target,meta,base,calibration,validation)
            if heads is not None: auxiliary[task]=heads
    metadata['relative_evidence']={task:{k:v for k,v in heads.items() if k not in {'regression','classifiers','calibrator'}} for task,heads in auxiliary.items()}
    return {'auxiliary':auxiliary,'regression':regression,'distribution':distribution,'calibrator':calibrator,'ood':detector,
        'metadata':metadata,'conformal_margin':margin,'policy':policy,'analogues':analogues},rows
