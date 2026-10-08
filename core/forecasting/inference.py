"""Pure V5 inference. No training, network calls or ledger writes on inference."""
import numpy as np
from core.forecasting.heads import price_from_log_return
from core.forecasting.trust import evidence_policy
from core.validation.alignment import aware_index

def infer_bundle(bundle,row,current_price,as_of,data_eligible=True,research_only=True):
    metadata=bundle['metadata']
    cutoff=aware_index([as_of],'forecast as_of')[0]
    for field in ['training_cutoff','blending_cutoff','calibration_cutoff','validation_cutoff','created_at']:
        if metadata.get(field) and aware_index([metadata[field]])[0]>=cutoff:
            raise ValueError('Model '+field+' reaches requested inference cutoff')
    if research_only is not True:
        raise ValueError('This release is research-only; production promotion remains blocked')
    if len(row)!=1: raise ValueError('A forecast requires one feature row')
    regression=bundle['regression']
    forecast=regression.predict(row)
    scaled=regression.scaled(row)
    distribution=bundle['distribution'].predict(scaled)
    raw_probability=float(distribution['probability'][0])
    probability=float(bundle['calibrator'].predict([raw_probability])[0]) if bundle['calibrator'] is not None else raw_probability
    central=float(forecast['central_log_return'][0])
    quantiles=distribution['quantiles'][0].copy()
    quantiles[0]-=bundle['conformal_margin']
    quantiles[4]+=bundle['conformal_margin']
    ood=float(bundle['ood'].score(scaled)[0])
    # Operational OOD expansion; no IID coverage guarantee is claimed.
    widening=max(1.,min(ood,10.))
    quantiles[0]=quantiles[2]+(quantiles[0]-quantiles[2])*widening
    quantiles[4]=quantiles[2]+(quantiles[4]-quantiles[2])*widening
    components={name:float(value[0]) for name,value in forecast['components'].items()}
    trust=evidence_policy(probability,central,quantiles,list(components.values()),data_eligible,ood,
        metadata.get('validation_n',0),bundle['calibrator'] is not None,
        False,metadata.get('final_test_passed',False),
        learned_threshold=bundle['policy'].get('probability_edge_min'))
    from core.forecasting.relative import predict_relative
    alpha={task:predict_relative(heads,scaled) for task,heads in bundle.get('auxiliary',{}).items()}
    return {'relative_forecasts':alpha,'horizon':regression.horizon,'central_log_return':central,'central_return':float(np.expm1(central)),
        'central_price':price_from_log_return(current_price,central),'quantiles':quantiles.tolist(),
        'quantile_prices':[price_from_log_return(current_price,q) for q in quantiles],
        'p_positive':probability,'p_negative':1-probability,'raw_p_positive':raw_probability,
        'p_positive_calibrated':probability if bundle['calibrator'] is not None else None,
        'probability_components':{name:float(v[0]) for name,v in distribution['probability_components'].items()},
        'component_returns':components,'ood_score':ood,'trust':trust,'scope':'research_only' if research_only else 'production',
        'model_version':metadata['model_version'],'quantile_crossing_corrected':bool(distribution['quantile_crossing_corrected'][0]),
        'price_assumption':'Total-return-equivalent implied price; future splits/dividends are not known',
        'ood_interval_expanded':widening>1}

def model_contributions(bundle,row):
    import xgboost as xgb
    from core.forecasting.registry import NativeModel
    regression=bundle['regression']
    scaled=regression.scaled(row)
    values=[]
    for name,model in regression.models.items():
        booster=model.booster if isinstance(model,NativeModel) else model.get_booster() if name=='xgb' else model.booster_
        contribution=booster.predict(xgb.DMatrix(scaled),pred_contribs=True) if name=='xgb' else booster.predict(scaled,pred_contrib=True)
        values.append(np.asarray(contribution)[0])
    combined=sum(weight*contribution for weight,contribution in zip(regression.weights,values))
    return {'features':list(regression.feature_names),'values':combined[:-1].tolist(),'base_value':float(combined[-1]),
        'label':'Model contribution/association in log-return units; not causal proof'}

def historical_analogues(bundle,row,as_of,count=20):
    pool=bundle.get('analogues')
    if pool is None: return {'n':0,'status':'UNAVAILABLE'}
    cutoff=aware_index([as_of])[0]
    available=aware_index(pool['label_ends'])<cutoff
    indices=np.flatnonzero(available)
    if not len(indices): return {'n':0,'status':'NO_MATURED_ANALOGUES'}
    scaled=bundle['regression'].scaled(row)[0]
    distance=np.sqrt(np.mean((pool['features'][indices]-scaled)**2,axis=1))
    selected=indices[np.argsort(distance)[:count]]
    outcomes=np.expm1(pool['returns'][selected])
    return {'n':len(selected),'win_rate':float(np.mean(outcomes>0)),'median_return':float(np.median(outcomes)),
        'best_return':float(np.max(outcomes)),'worst_return':float(np.min(outcomes)),
        'distribution':outcomes.tolist(),'timestamps':pool['timestamps'][selected].tolist(),
        'tickers':pool['tickers'][selected].tolist(),'median_excess_return':float(np.nanmedian(np.expm1(pool['returns'][selected])-np.expm1(pool['returns'][selected]-pool['excess_market'][selected]))) if 'excess_market' in pool and np.isfinite(pool['excess_market'][selected]).any() else None,
        'limitation':'Overlapping historical neighbours are not independent statistical evidence'}
