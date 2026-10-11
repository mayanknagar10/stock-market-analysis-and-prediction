"""Auxiliary excess-return heads, fitted only to matching known benchmark/sector sessions."""
import numpy as np
import pandas as pd
from core.forecasting.trust import PlattCalibrator
from core.validation.metrics import forecast_metrics

def relative_targets(meta,stock_targets,references,settings,family):
    output={'market':np.full(len(meta),np.nan),'sector':np.full(len(meta),np.nan)}
    market='market_india' if family=='india_equity' else 'market_us'
    cache={}
    for key,data in references.items():
        frame=data.frame
        dates=frame.index.tz_convert(data.metadata.exchange_timezone).tz_localize(None).normalize()
        close=pd.Series(frame['Adj Close'].to_numpy(),index=dates)
        if close.index.is_unique: cache[key]=close
    zone='Asia/Kolkata' if family=='india_equity' else 'America/New_York'
    for i,(_,row) in enumerate(meta.iterrows()):
        origin=pd.Timestamp(row['origin_session']).normalize() if 'origin_session' in row else row.feature_time.tz_convert(zone).tz_localize(None).normalize()
        end=pd.Timestamp(row['maturity_session']).normalize() if 'maturity_session' in row else row.label_end.tz_convert(zone).tz_localize(None).normalize()
        for task,key in [('market',market),('sector',settings.get('sector_map',{}).get(row.ticker))]:
            if key in cache and origin in cache[key].index and end in cache[key].index:
                start=float(cache[key].loc[origin]); future=float(cache[key].loc[end])
                if np.isfinite([start,future]).all() and min(start,future)>0:
                    output[task][i]=float(stock_targets.iloc[i])-float(np.log(future/start))
    return output

def train_relative_heads(scaled,target,meta,base,calibration,validation):
    import xgboost as xgb
    import lightgbm as lgb
    available=np.isfinite(target)
    train=np.asarray(base)[available[base]]
    cal=np.asarray(calibration)[available[calibration]]
    test=np.asarray(validation)[available[validation]]
    if len(train)<100 or len(cal)<40 or len(test)<20: return None
    labels=(target[train]>0).astype(int)
    if len(np.unique(labels))!=2 or len(np.unique(target[cal]>0))!=2: return None
    common=dict(n_estimators=80,max_depth=3,learning_rate=.04,random_state=42,n_jobs=2)
    regression={'xgb':xgb.XGBRegressor(**common,objective='reg:squarederror',tree_method='hist',verbosity=0),
                'lgb':lgb.LGBMRegressor(**common,objective='regression',num_leaves=15,verbose=-1)}
    classifiers={'xgb':xgb.XGBClassifier(**common,objective='binary:logistic',tree_method='hist',verbosity=0),
                 'lgb':lgb.LGBMClassifier(**common,objective='binary',num_leaves=15,verbose=-1)}
    for model in regression.values(): model.fit(scaled[train],target[train])
    for model in classifiers.values(): model.fit(scaled[train],labels)
    raw_cal=np.mean([m.predict_proba(scaled[cal])[:,1] for m in classifiers.values()],axis=0)
    calibrator=PlattCalibrator().fit(raw_cal,(target[cal]>0).astype(int),meta.iloc[cal].feature_time,meta.iloc[train].label_end.max())
    predicted=np.mean([m.predict(scaled[test]) for m in regression.values()],axis=0)
    raw=np.mean([m.predict_proba(scaled[test])[:,1] for m in classifiers.values()],axis=0)
    calibrated=calibrator.predict(raw)
    raw_metrics=forecast_metrics(target[test],predicted,raw)
    metrics=forecast_metrics(target[test],predicted,calibrated)
    accepted=metrics['brier']<=raw_metrics['brier'] and metrics['log_loss']<=raw_metrics['log_loss']
    return {'regression':regression,'classifiers':classifiers,'calibrator':calibrator if accepted else None,
            'metrics':metrics if accepted else raw_metrics,'n_train':len(train),'n_calibration':len(cal),
            'calibration_accepted':bool(accepted),'task':'cumulative log excess return'}

def predict_relative(heads,scaled):
    if heads is None: return None
    alpha=float(np.mean([m.predict(scaled)[0] for m in heads['regression'].values()]))
    raw=float(np.mean([(m.predict_proba(scaled)[0,1] if hasattr(m,'predict_proba') else m.predict(scaled)[0])
                       for m in heads['classifiers'].values()]))
    probability=float(heads['calibrator'].predict([raw])[0]) if heads['calibrator'] is not None else None
    return {'central_excess_log_return':alpha,'raw_p_outperform':raw,'calibrated_p_outperform':probability,
            'validation_metrics':heads['metrics'],'scope':'research_only'}
