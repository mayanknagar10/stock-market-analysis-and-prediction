"""Frozen V5 shadow inference into a separate prospective archive and ledger."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import uuid
import numpy as np
import pandas as pd
from core.data.providers import YahooMarketDataProvider,equity_family
from core.data.quality import validate_market_data
from core.data.cleaning import research_bar_cleanup
from core.forecasting.service import adjusted_frame,active_research,load_research_model
from core.forecasting.context import research_feature_frame,market_regime
from core.forecasting.inference import infer_bundle,model_contributions,historical_analogues
from core.forecasting.scenarios import ScenarioEngine
from core.forecasting.ledger import PredictionLedger
from core.validation.alignment import aware_index,session_cutoffs
from research.post_v5.archive import RunArchive
ROOT=Path(__file__).resolve().parents[2]
STUDY='prospective-india-20261008-v1'
DIRECTORY=ROOT/'data/research'/STUDY
FROZEN='research-20261008T190859'

def record_prediction(snapshot,result,metadata,features,horizon,asof,origin,run_id,snapshot_ids,regime,provider_versions,health,current_price=None):
    trust=result['trust']
    if trust['confidence']!='LOW' or trust['abstain'] is not True: raise ValueError('Frozen shadow must remain LOW and abstain')
    if metadata.get('model_version')!=FROZEN: raise ValueError('Only frozen V5 reference candidate admitted')
    if aware_index([snapshot.metadata.fetched_at])[0]>aware_index([asof])[0]: raise ValueError('Source retrieval after prediction cutoff')
    current_price=float(snapshot.frame.Close.iloc[-1]) if current_price is None else float(current_price)
    return {'prediction_id':str(uuid.uuid4()),'ticker':snapshot.metadata.symbol,'market':equity_family(snapshot.metadata.symbol),
        'forecast_as_of':asof,'generated_at':datetime.now(timezone.utc).isoformat(),'information_cutoff':asof,
        'origin_session':origin,'timezone':'UTC','horizon':int(horizon),'current_price':current_price,
        'central_log_return':result['central_log_return'],'central_price':result['central_price'],'quantiles':result['quantiles'],
        'p_positive':result['p_positive'],'p_positive_calibrated':result.get('p_positive_calibrated'),
        'raw_p_positive':result.get('raw_p_positive',result['p_positive'] if result.get('p_positive_calibrated') is None else None),
        'probability_components':dict(result.get('probability_components',{})),'component_returns':dict(result.get('component_returns',{})),
        'confidence':'LOW','abstain':True,'calibration_state':'accepted on frozen V5 development' if result.get('p_positive_calibrated') is not None else 'raw uncalibrated',
        'model_agreement':trust.get('agreement'),'regime':str(regime),'data_quality':health,
        'ood_score':result['ood_score'],'model_version':FROZEN,'feature_version':metadata['feature_version'],
        'training_cutoff':metadata['training_cutoff'],'blending_cutoff':metadata.get('blending_cutoff'),
        'calibration_cutoff':metadata.get('calibration_cutoff'),'validation_cutoff':metadata.get('validation_cutoff'),
        'provider_versions':dict(provider_versions),'snapshot_ids':list(snapshot_ids.values()),'input_snapshot_ids':dict(snapshot_ids),
        'feature_schema_hash':metadata['feature_schema_hash'],'code_version':metadata['code_version'],
        'scope':'research_only','point_in_time_verified':False,'feature_values':dict(features),'trust_reasons':list(trust.get('reasons',[])),
        'input_run_id':run_id,'study_id':STUDY,'evidence_type':'prospective_shadow_forecast',
        'input_archive_observed_at_prediction':True,'source_retrieved_at':snapshot.metadata.fetched_at,
        'source_observation_timestamp':snapshot.metadata.source_timestamp,'historical_training_vintages_verified':False,
        'calendar_status':'INDEPENDENT_CAPTURED_SESSION_PROXY_UNVERIFIED','price_assumption':result.get('price_assumption','total-return-equivalent price')}

def forecast_from_run(run_id,ticker,asof=None,persist=True,directory=None,bundles=None):
    directory=Path(directory or DIRECTORY);archive=RunArchive(directory);run=archive.load_run(run_id)
    asof=asof or datetime.now(timezone.utc).isoformat()
    if run['study_id']!=STUDY: raise ValueError('Input run belongs to a different registered study')
    if aware_index([run['cutoff']])[0]>aware_index([asof])[0] or aware_index([run['archived_at']])[0]>aware_index([asof])[0]: raise ValueError('Input run was not archived by requested prediction cutoff')
    sources={key:archive.load_snapshot(identifier) for key,identifier in run['snapshots'].items()}
    if ticker not in sources: raise ValueError('No captured equity source; failed member remains in study')
    snapshot=sources[ticker];family=equity_family(ticker)
    raw,frame=adjusted_frame(snapshot,asof)
    issue=pd.Timestamp(asof)
    local_issue=issue.tz_convert(snapshot.metadata.exchange_timezone)
    closing_day=local_issue.tz_localize(None).normalize()
    closing_time=session_cutoffs(pd.DatetimeIndex([closing_day]),family)[0]
    if closing_day.weekday()>=5 or issue<closing_time:
        closing_day=closing_day-pd.offsets.BDay(1)
        closing_time=session_cutoffs(pd.DatetimeIndex([closing_day]),family)[0]
    if pd.Timestamp(snapshot.metadata.fetched_at)<closing_time and frame.index[-1]<closing_day:
        raise ValueError('Critical price data: potential session already closed since capture; a fresh snapshot is required')
    for captured in sources.values():
        if captured.metadata.family!=family:continue
        observed=captured.frame
        dates=observed.index.tz_convert(captured.metadata.exchange_timezone).tz_localize(None).normalize()
        closes=session_cutoffs(dates,family)
        expired=(dates>frame.index[-1])&(observed.Volume.to_numpy()>0)&(closes<=pd.Timestamp(asof))
        if np.any(expired):raise ValueError('Critical price data: captured active session already closed; a fresh completed snapshot is required')
    calendar=None
    for key,peer in sources.items():
        if key==ticker or peer.metadata.family!=family:continue
        try: _,part=adjusted_frame(peer,asof)
        except ValueError:continue
        dates=part.index[part.Volume>0]
        calendar=dates if calendar is None else calendar.union(dates)
    if calendar is None:raise ValueError('Independent captured session calendar missing')
    calendar=calendar[calendar>=frame.index.min()].sort_values()
    frame,cleanup=research_bar_cleanup(frame,calendar)
    raw=raw.loc[raw.index.tz_convert(snapshot.metadata.exchange_timezone).tz_localize(None).normalize().isin(frame.index)]
    if any(pd.Timestamp(d) in frame.index[-200:] for d in cleanup['unresolved_session_dates']):raise ValueError('Unresolved price/volume session in required feature history')
    expected=calendar.tz_localize(snapshot.metadata.exchange_timezone).tz_convert('UTC').as_unit('ns')
    health=validate_market_data(raw,asof,expected)
    if not health.eligible:raise ValueError('Critical price data failed: '+','.join(health.critical_flags))
    references={key:value for key,value in sources.items() if value.metadata.family=='research_reference'}
    settings=json.loads((ROOT/'config/context_sources.json').read_text(encoding='utf-8'))
    features,context=research_feature_frame(frame,ticker,family,references,settings,research_only=True)
    pointer=active_research()
    if pointer['model_version']!=FROZEN:raise ValueError('V5 pointer changed; frozen reference study stops')
    bundles=bundles or {h:load_research_model(family,h,pointer) for h in [1,5,10,20]}
    current=float(raw.Close.iloc[-1]);origin=frame.index[-1].date().isoformat()
    regime=market_regime(features).iloc[-1]
    outputs=[];records=[]
    universe_file=ROOT/'config/research'/STUDY/'UNIVERSE.json'
    universe=json.loads(universe_file.read_text(encoding='utf-8'))
    member=next((m for m in universe['members'] if m['ticker']==ticker),None)
    for horizon,bundle in bundles.items():
        names=list(bundle['regression'].feature_names)
        if not set(names).issubset(features):raise ValueError('Missing trained feature family')
        row=features.iloc[[-1]][names]
        if not np.isfinite(row.to_numpy()).all():raise ValueError('Incomplete latest feature vector')
        stale=any(status=='STALE' for key,status in context['source_statuses'].items()
            if any(name.startswith('market_' if key.startswith('market_') else 'sector_' if key.startswith('sector_') else key+'_') for name in names))
        result=infer_bundle(bundle,row,current,asof,data_eligible=bool(health.eligible and not stale),research_only=True)
        if stale:result['trust']['reasons'].append('CONTEXT_STALE')
        if context['sector_status'] in {'UNAVAILABLE','INSUFFICIENT_OR_INVALID_HISTORY','STALE'}:result['relative_forecasts'].pop('sector',None)
        metadata=bundle['metadata']
        providers={value.metadata.provider_id:value.metadata.provider_version for value in sources.values()}
        record=record_prediction(snapshot,result,metadata,row.iloc[0].to_dict(),horizon,asof,origin,run_id,run['snapshots'],
            regime,providers,{**health.to_dict(),'context':context,'source_cleanup':cleanup},current_price=current)
        record['frozen_universe_member']=member is not None;record['industry_at_freeze']=member['industry'] if member else 'outside fixed study'
        record['universe_hash']=hashlib.sha256(universe_file.read_bytes()).hexdigest()
        record['benchmark_source_key']='market_india' if family=='india_equity' else 'market_us'
        record['sector_source_key']=settings['sector_map'].get(ticker)
        record['drivers']=model_contributions(bundle,row)
        records.append(record)
        outputs.append({'forecast':result,'record':record,'metadata':metadata,'analogues':historical_analogues(bundle,row,asof),
            'scenario_engine':ScenarioEngine(bundle,frame,ticker,family,references,settings,current,asof)})
    for record in records: archive.put('forecasts',record)
    if persist:PredictionLedger(directory/'shadow_predictions.sqlite').append_many(records)
    return {'ticker':ticker,'as_of':asof,'origin_session':origin,'current_price':current,'health':health.to_dict(),
        'context_health':{**context,'source_cleanup':cleanup},'forecasts':outputs,'input_run_id':run_id}

def capture_request(ticker):
    family=equity_family(ticker)
    peers=['RELIANCE.NS','TCS.NS','HDFCBANK.NS'] if family=='india_equity' else ['AAPL','JPM','XOM']
    settings=json.loads((ROOT/'config/context_sources.json').read_text(encoding='utf-8'))
    sources={key:(key,False) for key in set(peers+[ticker])}
    sources.update({key:(symbol,True) for key,symbol in settings['sources'].items()})
    snapshots={};failures={};archive=RunArchive(DIRECTORY)
    from concurrent.futures import ThreadPoolExecutor,as_completed
    def capture(value):return YahooMarketDataProvider().capture_history(value[0],reference=value[1])
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs={pool.submit(capture,value):key for key,value in sources.items()}
        for future in as_completed(jobs):
            key=jobs[future]
            try:
                snapshot=future.result();archive.save_snapshot(snapshot);snapshots[key]=snapshot
            except Exception as error:failures[key]=str(error)
    cutoff=datetime.now(timezone.utc).isoformat()
    universe_hash=hashlib.sha256((ROOT/'config/research'/STUDY/'UNIVERSE.json').read_bytes()).hexdigest()
    return archive.save_run(snapshots,failures,cutoff,STUDY,purpose='forecast_request',universe_hash=universe_hash)

def forecast_request(ticker,bundles=None):
    run_id=capture_request(ticker)
    try:return forecast_from_run(run_id,ticker,bundles=bundles)
    except Exception as error:
        RunArchive(DIRECTORY).put('attempts',{'run_id':run_id,'ticker':ticker,'attempted_at':datetime.now(timezone.utc).isoformat(),
            'status':'forecast_failed','error':str(error),'study_id':STUDY})
        raise
