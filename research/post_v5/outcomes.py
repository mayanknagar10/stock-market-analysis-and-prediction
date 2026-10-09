"""Explicit shadow maturation, exact session endpoints and retained sector members."""
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd
from core.forecasting.ledger import PredictionLedger,resolve_outcome
from core.forecasting.service import adjusted_frame
from core.validation.alignment import session_cutoffs
from research.post_v5.archive import RunArchive
from research.post_v5.shadow import ROOT,DIRECTORY,STUDY

def endpoint_return(values,origin,maturity):
    points=values.reindex(pd.to_datetime([origin,maturity]))
    if len(points)!=2 or not np.isfinite(points).all() or (points<=0).any():return None
    return float(np.log(points.iloc[1]/points.iloc[0]))

def peer_sector_return(ticker,industry,members,prices,origin,maturity):
    peers=[m['ticker'] for m in members if m['industry']==industry and m['ticker']!=ticker]
    values=[]
    for peer in peers:
        value=endpoint_return(prices[peer],origin,maturity) if peer in prices else None
        if value is not None:values.append(value)
    return {'log_return':float(np.log(np.mean(np.exp(values)))) if len(peers)>=2 and len(values)==len(peers) else None,
        'n_fixed_peers':len(peers),'n_observed_peers':len(values),
        'policy':'Equal-weight fixed-industry buy-and-hold peer basket, excluding stock; all fixed peers required; not official sector index'}

def prices_for(snapshot,asof=None):
    raw=snapshot.frame
    dates=raw.index.tz_convert(snapshot.metadata.exchange_timezone).tz_localize(None).normalize()
    if dates.has_duplicates:raise ValueError('Duplicate outcome sessions')
    family=snapshot.metadata.family
    if family=='research_reference':
        if snapshot.metadata.exchange_timezone=='Asia/Kolkata':family='india_equity'
        elif snapshot.metadata.exchange_timezone in {'America/New_York','US/Eastern'}:family='us_equity'
        else:return pd.Series(dtype=float,index=pd.DatetimeIndex([]))
    if family not in {'india_equity','us_equity'}:return pd.Series(dtype=float,index=pd.DatetimeIndex([]))
    cutoff=pd.Timestamp(snapshot.metadata.fetched_at)
    if asof is not None:cutoff=min(cutoff,pd.Timestamp(asof))
    completed=session_cutoffs(dates,family)+pd.Timedelta('20min')
    known=np.asarray(completed<=cutoff)
    values=(raw['Adj Close'] if 'Adj Close' in raw else raw.Close).to_numpy(dtype=float).copy()
    if snapshot.metadata.family in {'india_equity','us_equity'}:
        if not {'Open','High','Low','Close','Volume'}.issubset(raw):values[:]=np.nan
        else:
            columns=raw[['Open','High','Low','Close','Volume']]
            valid=np.isfinite(columns).all(axis=1)&(raw.Volume>0)&(raw.Close>0)&(raw.Low>0)&(raw.High>=raw[['Open','Close','Low']].max(axis=1))&(raw.Low<=raw[['Open','Close']].min(axis=1))
            values[~valid.to_numpy()]=np.nan
    return pd.Series(values[known],index=dates[known])

def revision_diagnostics(original,outcome,origin):
    fields=['Open','High','Low','Close','Adj Close','Volume','Dividends','Stock Splits']
    def row(snapshot):
        frame=snapshot.frame
        dates=frame.index.tz_convert(snapshot.metadata.exchange_timezone).tz_localize(None).normalize()
        positions=np.flatnonzero(dates==pd.Timestamp(origin))
        if len(positions)!=1:raise ValueError('Original/outcome origin snapshot missing or duplicated')
        return frame.iloc[int(positions[0])]
    first=row(original);later=row(outcome);changed=[];values={}
    for field in fields:
        left=float(first[field]) if field in first else None;right=float(later[field]) if field in later else None
        left=left if left is not None and np.isfinite(left) else None
        right=right if right is not None and np.isfinite(right) else None
        values[field]={'at_prediction':left,'at_outcome':right}
        equal=(left is None and right is None) or (left is not None and right is not None and np.isclose(left,right,rtol=1e-9,atol=1e-10))
        if not equal:changed.append(field)
    return {'origin_session':origin,'fields_changed':changed,'origin_values':values,
        'interpretation':'Changes can reflect vendor revisions or later corporate-action normalization; source vintages are retained, not silently substituted',
        'prediction_source_retrieved_at':original.metadata.fetched_at,'outcome_source_retrieved_at':outcome.metadata.fetched_at}

def resolve_run(run_id,directory=None):
    directory=Path(directory or DIRECTORY);archive=RunArchive(directory);run=archive.load_run(run_id)
    sources={key:archive.load_snapshot(identifier) for key,identifier in run['snapshots'].items()}
    universe=json.loads((ROOT/'config/research'/STUDY/'UNIVERSE.json').read_text(encoding='utf-8'))
    ledger=PredictionLedger(directory/'shadow_predictions.sqlite');records=ledger.rows()
    if run['study_id']!=STUDY:raise ValueError('Outcome run belongs to another study')
    prices={key:prices_for(value,run['cutoff']) for key,value in sources.items()}
    peer_calendars={}
    for key,peer in sources.items():
        if peer.metadata.family not in {'india_equity','us_equity'}:continue
        raw=peer.frame
        dates=raw.index.tz_convert(peer.metadata.exchange_timezone).tz_localize(None).normalize()
        completed=session_cutoffs(dates,peer.metadata.family)+pd.Timedelta('20min')
        received_cutoff=min(pd.Timestamp(run['cutoff']),pd.Timestamp(peer.metadata.fetched_at))
        known=(completed<=received_cutoff)&(raw.Volume.to_numpy()>0)
        peer_calendars[key]=(peer.metadata.family,dates[known],completed[known])
    summary={'run_id':run_id,'study_id':STUDY,'resolved_at':datetime.now(timezone.utc).isoformat(),'matured':0,'pending':0,'failures':{}}
    for record in records:
        if record['outcome'] is not None:continue
        ticker=record['ticker']
        try:
            if ticker not in sources:raise ValueError('Missing captured outcome source; member retained')
            snapshot=sources[ticker]
            asof=min(pd.Timestamp(run['cutoff']),pd.Timestamp(snapshot.metadata.fetched_at),pd.Timestamp.now(tz='UTC')).isoformat()
            calendar=None
            for key,(family,peer_dates,completed) in peer_calendars.items():
                if key==ticker or family!=record['market']:continue
                dates=peer_dates[completed<=pd.Timestamp(asof)]
                if not len(dates):continue
                calendar=dates if calendar is None else calendar.union(dates)
            if calendar is None:raise ValueError('Independent captured outcome calendar unavailable')
            market=prices.get(record['benchmark_source_key'])
            outcome=resolve_outcome(record,snapshot.frame,asof,benchmark=market,expected_sessions=calendar.sort_values())
            if outcome is None:summary['pending']+=1;continue
            if outcome['status']!='matured':raise ValueError('Incomplete or unusable adjusted outcome endpoints; keep pending for a later captured vintage')
            if outcome['status']=='matured':
                maturity=pd.Timestamp(outcome['maturity_timestamp']).tz_convert(snapshot.metadata.exchange_timezone).date().isoformat()
                stock_return=endpoint_return(prices[ticker],record['origin_session'],maturity)
                if stock_return is None:raise ValueError('Incomplete, non-finite or non-traded equity outcome endpoint; keep pending without shifting maturity')
                benchmark=endpoint_return(market,record['origin_session'],maturity) if market is not None else None
                if benchmark is None:raise ValueError('Benchmark maturity endpoints unavailable; keep outcome pending')
                sector_key=record.get('sector_source_key')
                sector=endpoint_return(prices[sector_key],record['origin_session'],maturity) if sector_key in prices else None
                sector_details={'policy':'Captured source '+str(sector_key)}
                if sector is None and record.get('frozen_universe_member'):
                    sector_details=peer_sector_return(ticker,record['industry_at_freeze'],universe['members'],prices,record['origin_session'],maturity)
                    sector=sector_details['log_return']
                original_run=archive.load_run(record['input_run_id'])
                original_snapshot=archive.load_snapshot(original_run['snapshots'][ticker])
                revisions=revision_diagnostics(original_snapshot,snapshot,record['origin_session'])
                actual=outcome['actual_log_return']
                outcome.update({'benchmark_log_return':benchmark,'benchmark_return':float(np.expm1(benchmark)),
                    'sector_log_return':sector,'sector_return':float(np.expm1(sector)) if sector is not None else None,
                    'alpha_market':actual-benchmark,'alpha_market_arithmetic':float(np.expm1(actual)-np.expm1(benchmark)),
                    'alpha_sector':actual-sector if sector is not None else None,
                    'alpha_sector_arithmetic':float(np.expm1(actual)-np.expm1(sector)) if sector is not None else None,
                    'return_direction_correct':bool((actual>0)==(record['central_log_return']>0)),
                    'sector_reference':sector_details,'benchmark_policy':'Captured Yahoo benchmark; price-index return where source lacks TRI',
                    'outcome_basis':'Vendor adjusted close ratio in outcome vintage; raw actions and revision diagnostics retained',
                    'origin_raw_price_at_prediction':record['current_price'],'origin_revision_diagnostics':revisions,
                    'outcome_source_retrieved_at':snapshot.metadata.fetched_at,
                    'benchmark_source_retrieved_at':sources[record['benchmark_source_key']].metadata.fetched_at,
                    'outcome_snapshot_id':run['snapshots'][ticker],'outcome_input_run_id':run_id,
                    'benchmark_snapshot_id':run['snapshots'].get(record['benchmark_source_key']),
                    'calendar_status':'CAPTURED_INDEPENDENT_SESSION_PROXY_UNVERIFIED'})
            archive.put('outcomes',{'prediction_id':record['prediction_id'],'outcome':outcome})
            ledger.attach_outcome(record['prediction_id'],outcome);summary['matured']+=1
        except Exception as error:
            summary['pending']+=1
            summary['failures'][record['prediction_id']]=str(error)
            archive.put('attempts',{'status':'resolution_failed_or_pending','prediction_id':record['prediction_id'],
                'run_id':run_id,'at':datetime.now(timezone.utc).isoformat(),'error':str(error),'study_id':STUDY})
    identifier=archive.put('attempts',{**summary,'status':'resolution_batch_complete'})
    path=ROOT/'reports/post_v5/outcomes'/f'{identifier}.json';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:json.dump(summary,stream,indent=2)
    return summary
