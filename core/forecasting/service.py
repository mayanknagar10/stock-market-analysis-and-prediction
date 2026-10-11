"""Research application service: explicit snapshots, no training or silent model fallback."""
from datetime import datetime,timezone
import json
from pathlib import Path
import uuid
import numpy as np
import pandas as pd
from core.data.providers import YahooMarketDataProvider,equity_family
from core.data.snapshots import SnapshotStore
from core.data.quality import validate_market_data
from core.forecasting.registry import load_bundle
from core.forecasting.context import research_feature_frame,market_regime
from core.forecasting.inference import infer_bundle,model_contributions,historical_analogues
from core.forecasting.scenarios import ScenarioEngine
from core.forecasting.ledger import PredictionLedger
from core.validation.alignment import session_cutoffs

ROOT=Path(__file__).resolve().parents[2]

def active_research():
    file=ROOT/'models/v5/ACTIVE_RESEARCH.json'
    if not file.exists(): raise ValueError('No V5 research checkpoint. Run the explicit train_v5.py job.')
    pointer=json.loads(file.read_text(encoding='utf-8'))
    quarantine=ROOT/'models/v5/QUARANTINE.json'
    if quarantine.exists() and any(v['model_version']==pointer['model_version'] for v in json.loads(quarantine.read_text(encoding='utf-8'))['versions']):
        raise ValueError('Active research checkpoint was quarantined; no equivalent fallback is used')
    return pointer

def load_research_model(family,horizon,pointer=None):
    pointer=pointer or active_research()
    key=family+'_'+str(horizon)
    if key not in pointer['models']: raise ValueError('Unsupported family/horizon checkpoint')
    path=(ROOT/pointer['models'][key]).resolve()
    if ROOT.resolve() not in path.parents: raise ValueError('Invalid registry path')
    bundle=load_bundle(path)
    if bundle['regression'].family!=family or bundle['regression'].horizon!=horizon:
        raise ValueError('Registry family/horizon mismatch')
    return bundle

def load_reference_snapshots():
    catalog=ROOT/'data/v5/REFERENCE_CAPTURE.json'
    store=SnapshotStore(ROOT/'data/v5/snapshots')
    if catalog.exists():
        manifest=json.loads(catalog.read_text(encoding='utf-8'))
        try:
            references={key:store.load(identifier) for key,identifier in manifest['snapshots'].items()}
            return references,manifest
        except FileNotFoundError:
            pass
    settings=json.loads((ROOT/'config/context_sources.json').read_text(encoding='utf-8'))
    provider=YahooMarketDataProvider()
    references={}; manifest={'scope':'research_only','snapshots':{},'failures':{}}
    for key,symbol in settings['sources'].items():
        try:
            data=provider.capture_history(symbol,reference=True)
            references[key]=data; manifest['snapshots'][key]=store.save(data)
        except Exception as error: manifest['failures'][key]=str(error)
    return references,manifest

def adjusted_frame(snapshot,as_of):
    raw=snapshot.frame
    family=snapshot.metadata.family
    local=raw.index.tz_convert(snapshot.metadata.exchange_timezone).tz_localize(None).normalize()
    completed=session_cutoffs(local,family)+pd.Timedelta('20min')
    known_cutoff=min(pd.Timestamp(as_of),pd.Timestamp(snapshot.metadata.fetched_at))
    mask=completed<=known_cutoff
    raw=raw.loc[mask]
    local=local[mask]
    if raw.empty: raise ValueError('No completed source session at requested cutoff')
    frame=raw[['Open','High','Low','Close','Volume']].copy()
    factor=raw['Adj Close']/raw.Close
    frame[['Open','High','Low','Close']]=frame[['Open','High','Low','Close']].mul(factor,axis=0)
    frame.index=local
    return raw,frame

def independent_research_calendar(family,start,as_of,exclude_ticker=None):
    from core.data.runtime import market_snapshot
    peers=['RELIANCE.NS','TCS.NS','HDFCBANK.NS'] if family=='india_equity' else ['AAPL','JPM','XOM']
    calendar=None
    for ticker in peers:
        if ticker==exclude_ticker: continue
        peer=market_snapshot(ticker)
        _,frame=adjusted_frame(peer,as_of)
        sessions=frame.index[frame.Volume>0]
        calendar=sessions if calendar is None else calendar.union(sessions)
    if calendar is None: raise ValueError('Independent research calendar unavailable')
    return calendar[calendar>=start].sort_values()

def research_forecast(snapshot,bundles,references,current_as_of=None,persist=True):
    as_of=current_as_of or datetime.now(timezone.utc).isoformat()
    family=equity_family(snapshot.metadata.symbol)
    if snapshot.metadata.interval!='1d': raise ValueError('Forecast requires daily completed-session data')
    raw,frame=adjusted_frame(snapshot,as_of)
    expected_dates=independent_research_calendar(family,frame.index[0],as_of,exclude_ticker=snapshot.metadata.symbol)
    from core.data.cleaning import research_bar_cleanup
    frame,source_quality=research_bar_cleanup(frame,expected_dates)
    retained_dates=raw.index.tz_convert(snapshot.metadata.exchange_timezone).tz_localize(None).normalize().isin(frame.index)
    raw=raw.loc[retained_dates]
    if source_quality['unresolved_session_dates']:
        recent=frame.index[-200:]
        if any(pd.Timestamp(d) in recent for d in source_quality['unresolved_session_dates']):
            raise ValueError('Unresolved zero-volume trading session affects required feature history')
    source_expected=pd.DatetimeIndex(expected_dates).tz_localize(snapshot.metadata.exchange_timezone).tz_convert('UTC').as_unit('ns')
    health=validate_market_data(raw,as_of,source_expected)
    if not health.eligible: raise ValueError('Critical price data failed: '+', '.join(health.critical_flags))
    settings=json.loads((ROOT/'config/context_sources.json').read_text(encoding='utf-8'))
    features,context_health=research_feature_frame(frame,snapshot.metadata.symbol,family,references,settings,research_only=True)
    regime=market_regime(features).iloc[-1]
    current_price=float(raw.Close.iloc[-1])
    store=SnapshotStore(ROOT/'data/v5/snapshots')
    snapshot_id=store.save(snapshot)
    reference_ids={key:store.save(value) for key,value in references.items()}
    outputs=[]
    for horizon,bundle in bundles.items():
        names=list(bundle['regression'].feature_names)
        if not set(names).issubset(features): raise ValueError('Missing trained contextual family; inference abstains')
        row=features.iloc[[-1]][names]
        if not np.isfinite(row.to_numpy()).all(): raise ValueError('Latest feature values incomplete; no forecast')
        stale_context=any(status=='STALE' for key,status in context_health['source_statuses'].items() if any(name.startswith('market_' if key.startswith('market_') else 'sector_' if key.startswith('sector_') else key+'_') for name in names))
        result=infer_bundle(bundle,row,current_price,as_of,data_eligible=bool(health.eligible and not stale_context),research_only=True)
        if stale_context: result['trust']['reasons'].append('CONTEXT_STALE')
        if context_health['sector_status'] in {'UNAVAILABLE','INSUFFICIENT_OR_INVALID_HISTORY','STALE'}:
            result['relative_forecasts'].pop('sector',None)
        metadata=bundle['metadata']
        record={'prediction_id':str(uuid.uuid4()),'ticker':snapshot.metadata.symbol,'market':family,
            'forecast_as_of':as_of,'generated_at':datetime.now(timezone.utc).isoformat(),'information_cutoff':as_of,
            'origin_session':frame.index[-1].date().isoformat(),'timezone':'UTC','horizon':int(horizon),
            'current_price':current_price,'central_log_return':result['central_log_return'],'central_price':result['central_price'],
            'quantiles':result['quantiles'],'p_positive':result['p_positive'],'p_positive_calibrated':result['p_positive_calibrated'],
            'confidence':result['trust']['confidence'],'abstain':result['trust']['abstain'],'model_agreement':result['trust']['agreement'],
            'regime':regime,'data_quality':{**health.to_dict(),'context':context_health,'source_cleanup':source_quality,'calendar_status':'UNVERIFIED_RESEARCH_PROXY'},
            'ood_score':result['ood_score'],'model_version':metadata['model_version'],'feature_version':metadata['feature_version'],
            'training_cutoff':metadata['training_cutoff'],'provider_versions':{snapshot.metadata.provider_id:snapshot.metadata.provider_version},
            'snapshot_ids':[snapshot_id]+list(reference_ids.values()),'reference_snapshot_ids':reference_ids,
            'feature_schema_hash':metadata['feature_schema_hash'],'code_version':metadata['code_version'],
            'model_code_version':metadata['code_version'],
            'scope':'research_only','point_in_time_verified':False,'feature_values':row.iloc[0].to_dict(),
            'drivers':model_contributions(bundle,row),'price_assumption':result['price_assumption'],'trust_reasons':result['trust']['reasons']}
        outputs.append({'forecast':result,'record':record,'metadata':metadata,
                        'analogues':historical_analogues(bundle,row,as_of),
                        'scenario_engine':ScenarioEngine(bundle,frame,snapshot.metadata.symbol,family,references,settings,current_price,as_of)})
    if persist:
        PredictionLedger(ROOT/'data/v5/predictions.sqlite').append_many([item['record'] for item in outputs])
    return {'ticker':snapshot.metadata.symbol,'as_of':as_of,'origin_session':frame.index[-1].date().isoformat(),
            'current_price':current_price,'health':health.to_dict(),'context_health':{**context_health,'source_cleanup':source_quality},'forecasts':outputs}
