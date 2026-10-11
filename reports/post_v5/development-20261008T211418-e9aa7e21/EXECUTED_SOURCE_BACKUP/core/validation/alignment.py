"""Information-time alignment. No observation-date backfilling and no naive timestamps."""
from __future__ import annotations
import pandas as pd

def aware_index(values,name='timestamps',allow_nat=False):
    try: result = pd.DatetimeIndex(values)
    except (TypeError,ValueError) as exc: raise ValueError(name+' need consistent timezone-aware timestamps') from exc
    if result.tz is None: raise ValueError(name+' require an explicit timezone')
    if not allow_nat and result.hasnans: raise ValueError(name+' contain missing timestamps')
    return result.tz_convert('UTC').as_unit('ns')

def session_cutoffs(sessions,family,phase='eod'):
    """Standard-hours cutoffs for supplied sessions; does NOT invent an exchange calendar.

    Sessions must come from a verified exchange calendar for production. Special
    session cutoffs must be supplied explicitly by the data adapter. This helper
    alone is not historical publication-time evidence.
    """
    choices = {'india_equity':('Asia/Kolkata','15:30','09:00'),
               'us_equity':('America/New_York','16:00','09:00')}
    if family not in choices or phase not in {'eod','preopen'}:
        raise ValueError('Unsupported family/phase')
    days = pd.DatetimeIndex(sessions)
    if not days.is_unique or not days.is_monotonic_increasing or days.hasnans:
        raise ValueError('Sessions must be unique ordered dates')
    zone,close,preopen = choices[family]
    clock = close if phase=='eod' else preopen
    dates = [str(day.date())+' '+clock for day in days]
    cutoffs=pd.DatetimeIndex(dates).tz_localize(zone,ambiguous='raise',nonexistent='raise').tz_convert('UTC').as_unit('ns')
    # A dated weekend special session has no certified standard-hours schedule.
    # Research uses a conservative next-UTC-day cutoff rather than an earlier invented close.
    if phase=='eod':
        values=cutoffs.asi8.copy()
        for i,day in enumerate(days):
            if day.dayofweek>=5:
                values[i]=(pd.Timestamp(str(day.date()),tz='UTC')+pd.Timedelta('1D')).value
        cutoffs=pd.DatetimeIndex(values,tz='UTC')
    return cutoffs

def align_available(prediction_times,source,max_age=None):
    """Latest state for one source/entity, then latest eligible vintage.

    All rows need verified revision timestamps from the caller's provider
    contract. max_age measures observation age. Snapshot admission separately
    verifies provenance; this function never infers publication/vintage times.
    """
    times = aware_index(prediction_times,'prediction_times')
    if not times.is_monotonic_increasing or not times.is_unique:
        raise ValueError('Prediction times must be unique and ordered')
    required = {'source_timestamp','available_at','vintage_at'}
    if not required.issubset(source.columns): raise ValueError('Missing source timestamp or revision vintage metadata')
    right = source.copy().reset_index(drop=True)
    right['available_at'] = aware_index(right['available_at'],'available_at')
    right['source_timestamp'] = aware_index(right['source_timestamp'],'source_timestamp')
    if (right.source_timestamp>right.available_at).any():
        raise ValueError('Source observations cannot be available before observed')
    right['vintage_at'] = aware_index(right['vintage_at'],'vintage_at')
    if (right.source_timestamp>right.vintage_at).any():
        raise ValueError('Revision vintage cannot precede observation time')
    reserved = {'forecast_as_of','is_stale','known_at'}
    if reserved.intersection(right.columns): raise ValueError('Reserved alignment columns in source')
    right['known_at']=right['available_at']
    newer=right.vintage_at>right.known_at
    right.loc[newer,'known_at']=right.loc[newer,'vintage_at']
    if right.known_at.duplicated().any(): raise ValueError('Ambiguous duplicate information timestamps')
    reserved = {'forecast_as_of','is_stale'}
    if reserved.intersection(right.columns): raise ValueError('Reserved alignment columns in source')
    # Events update the latest observed state. A late revision to an older
    # observation must not displace a more recent eligible observation.
    events=right.sort_values('known_at')
    retained=[]
    latest_source=None
    for position,row in events.iterrows():
        if latest_source is None or row.source_timestamp>=latest_source:
            latest_source=row.source_timestamp
            retained.append(position)
    events=events.loc[retained]
    left = pd.DataFrame({'forecast_as_of':times})
    joined = pd.merge_asof(left,events,left_on='forecast_as_of',
                           right_on='known_at',direction='backward',allow_exact_matches=True)
    joined.index = times
    joined['is_stale'] = joined.available_at.isna()
    if max_age is not None:
        age = pd.Timedelta(max_age)
        if age< pd.Timedelta(0): raise ValueError('max_age must be nonnegative')
        joined['is_stale'] |= (joined.forecast_as_of-joined.source_timestamp)>age
    return joined

def assert_feature_cutoff(prediction_times,available_at):
    times = aware_index(prediction_times,'prediction_times')
    if len(times)!=len(available_at) or not aware_index(available_at.index,'feature index').equals(times):
        raise ValueError('Feature availability rows are not aligned to forecast timestamps')
    for column in available_at:
        observed = aware_index(available_at[column],column)
        if (observed>times).any(): raise ValueError('future feature availability: '+str(column))
    return True
