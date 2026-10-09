"""Consumed-period exclusion before transformations. Historical research, not PIT replay."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from core.indicators import build_ml_features
from core.forecasting.context import research_feature_frame,market_regime
from core.validation.alignment import session_cutoffs,aware_index
from research.post_v5.archive import RunArchive
ROOT=Path(__file__).resolve().parents[2]
STUDY='prospective-india-20261008-v1'
BOUNDARY=pd.Timestamp('2025-10-01',tz='UTC')
def clip_frame(frame):
    frame=frame.copy()
    dates=pd.DatetimeIndex(frame.index)
    compare=dates.tz_localize('UTC') if dates.tz is None else dates.tz_convert('UTC')
    return frame.loc[compare<BOUNDARY].copy()

def design_frame(frame,horizon,references=None,ticker='FIXTURE.NS',settings=None,calendar=None):
    if horizon not in {1,5,10,20}: raise ValueError('Unsupported research horizon')
    frame=clip_frame(frame)
    if frame.index.tz is not None: frame.index=frame.index.tz_localize(None).normalize()
    if calendar is not None:
        calendar=pd.DatetimeIndex(calendar)
        if calendar.tz is not None:calendar=calendar.tz_localize(None)
        calendar=calendar[(calendar>=frame.index.min())&(calendar<=frame.index.max())]
        frame=frame.reindex(calendar)
    times=session_cutoffs(frame.index,'india_equity')+pd.Timedelta('20min')
    if references is None:
        features=build_ml_features(frame).replace([np.inf,-np.inf],np.nan);features.index=times
        groups={'technical':list(features)};regime=pd.Series('unknown',index=times)
    else:
        bounded={}
        from core.data.contracts import DataSnapshot
        for key,value in references.items():
            bounded[key]=DataSnapshot(clip_frame(value.frame),value.metadata) if hasattr(value,'metadata') else clip_frame(value)
        features,health=research_feature_frame(frame,ticker,'india_equity',bounded,settings or {},research_only=True)
        groups=health['feature_groups'];regime=market_regime(features)
    columns=list(features)
    close=pd.Series(frame.Close.to_numpy(),index=times)
    actual=np.log(close.shift(-horizon)/close)
    ends=pd.Series(times,index=times).shift(-horizon)
    bad=pd.Series((frame.Volume<=0)|~np.isfinite(frame[['Open','High','Low','Close','Volume']]).all(axis=1),index=times)
    bad_past=bad.rolling(200,min_periods=1).sum()>0
    bad_forward=bad.rolling(horizon+1,min_periods=1).sum().shift(-horizon)>0
    technical=list(build_ml_features(frame).columns)
    valid=np.isfinite(features[technical]).all(axis=1)&np.isfinite(actual)&~bad_past&~bad_forward&(ends<BOUNDARY)
    result=features.loc[valid].copy()
    result['actual']=actual.loc[valid];result['feature_time']=times[valid]
    result['label_end']=ends.loc[valid];result['origin_session']=frame.index[valid].astype(str)
    result['maturity_session']=pd.Series(frame.index.astype(str),index=times).shift(-horizon).loc[valid]
    result['volatility_20']=np.log(close).diff().rolling(20).std().loc[valid]
    result['market_regime']=regime.loc[valid]
    result.attrs={'feature_columns':columns,'feature_groups':groups,'historical_point_in_time_certified':False}
    return result.reset_index(drop=True)

def fixed_folds(metadata,folds):
    origins=aware_index(metadata.feature_time);ends=aware_index(metadata.label_end)
    if len(origins) and (origins.max()>=BOUNDARY or ends.max()>=BOUNDARY): raise ValueError('Development row enters consumed V5 period')
    for start,stop in folds:
        start=pd.Timestamp(start,tz='UTC');stop=pd.Timestamp(stop,tz='UTC')+pd.Timedelta('1D')
        if stop>BOUNDARY: raise ValueError('Fold enters consumed V5 period')
        train=np.flatnonzero((origins<start)&(ends<start));test=np.flatnonzero((origins>=start)&(origins<stop))
        if len(train) and len(test):yield train,test

def load_development(run_id,directory=None):
    archive=RunArchive(directory or ROOT/'data/research'/STUDY)
    run=archive.load_run(run_id)
    frames={};references={}
    for key,identifier in run['snapshots'].items():
        snapshot=archive.load_snapshot(identifier)
        raw=clip_frame(snapshot.frame)
        if raw.empty:continue
        from core.data.contracts import DataSnapshot
        if key.endswith('.NS'):
            factor=raw['Adj Close']/raw.Close
            frame=raw[['Open','High','Low','Close','Volume']].copy()
            frame[['Open','High','Low','Close']]=frame[['Open','High','Low','Close']].mul(factor,axis=0)
            frame.index=raw.index.tz_convert(snapshot.metadata.exchange_timezone).tz_localize(None).normalize()
            frames[key]=frame
        else: references[key]=DataSnapshot(raw,snapshot.metadata)
    settings=json.loads((ROOT/'config/context_sources.json').read_text(encoding='utf-8'))
    universe=json.loads((ROOT/'config/research'/STUDY/'UNIVERSE.json').read_text(encoding='utf-8'))
    protocol=json.loads((ROOT/'config/research'/STUDY/'PROTOCOL.json').read_text(encoding='utf-8'))
    return frames,references,settings,universe,protocol
