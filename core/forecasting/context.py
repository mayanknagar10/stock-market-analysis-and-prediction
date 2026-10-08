"""Conservative session-lag research references. These are never certified PIT features."""
import numpy as np
import pandas as pd
from core.indicators import build_ml_features
from core.validation.alignment import session_cutoffs, aware_index

def research_feature_frame(frame,ticker,family,references,settings,research_only=False):
    if research_only is not True: raise ValueError('Reconstructed contextual features require explicit research-only mode')
    features=build_ml_features(frame)
    times=(session_cutoffs(frame.index,family)+pd.Timedelta('20min')).as_unit('ns')
    features.index=times
    groups={'stock':list(features.columns),'market':[],'sector':[],'global':[]}
    market='market_india' if family=='india_equity' else 'market_us'
    sector=settings.get('sector_map',{}).get(ticker)
    statuses={}
    for key,data in references.items():
        if key.startswith('market_') and key!=market: continue
        if key.startswith('sector_') and key!=sector: continue
        if hasattr(data,'metadata'):
            values=data.frame
            local=values.index.tz_convert(data.metadata.exchange_timezone).tz_localize(None).normalize()
        else:
            values=data.copy()
            local=values.index.tz_localize(None).normalize() if values.index.tz is not None else values.index.normalize()
        close=values['Adj Close'].copy() if 'Adj Close' in values else values.Close.copy()
        close.index=local
        close=close.loc[~close.index.duplicated(keep=False)].sort_index()
        if len(close)<252 or not np.isfinite(close.dropna()).all() or (close.dropna()<=0).any():
            statuses[key]='INSUFFICIENT_OR_INVALID_HISTORY'
            continue
        name='market' if key==market else 'sector' if key==sector else key
        group='market' if name=='market' else 'sector' if name=='sector' else 'global'
        source=pd.DataFrame(index=close.index)
        for horizon in [1,5,20]: source[name+'_ret_'+str(horizon)+'d']=np.log(close/close.shift(horizon))
        source[name+'_vol_20']=np.log(close).diff().rolling(20).std()
        source[name+'_trend_50']=close/close.rolling(50).mean()-1
        source[name+'_drawdown']=close/close.rolling(252,min_periods=50).max()-1
        # Delay every reference session to the following UTC day. This excludes
        # same-day US closes from India EOD/preopen; exact provider release and
        # revision times remain unknown and must never be marked verified.
        available=(pd.DatetimeIndex(source.index).tz_localize('UTC')+pd.Timedelta('1D')).as_unit('ns')
        source['reference_available_estimate']=available
        joined=pd.merge_asof(pd.DataFrame({'as_of':times}),source.reset_index(drop=True).sort_values('reference_available_estimate'),
            left_on='as_of',right_on='reference_available_estimate',direction='backward')
        columns=[c for c in source if c!='reference_available_estimate']
        for column in columns: features[column]=joined[column].to_numpy()
        groups[group].extend(columns)
        statuses[key]='STALE' if times[-1]-available[-1]>pd.Timedelta('4D') else 'RECONSTRUCTED_UNVERIFIED'
    if 'market_ret_5d' in features:
        features['stock_minus_market_5d']=features.ret_5d-features.market_ret_5d
        features['rolling_market_beta']=features.ret_1d.rolling(60).cov(features.market_ret_1d)/features.market_ret_1d.rolling(60).var()
        features['rolling_market_correlation']=features.ret_1d.rolling(60).corr(features.market_ret_1d)
        groups['market']+=['stock_minus_market_5d','rolling_market_beta','rolling_market_correlation']
    if 'sector_ret_5d' in features:
        features['stock_minus_sector_5d']=features.ret_5d-features.sector_ret_5d
        groups['sector'].append('stock_minus_sector_5d')
        if 'market_ret_5d' in features:
            features['sector_minus_market_5d']=features.sector_ret_5d-features.market_ret_5d
            groups['sector'].append('sector_minus_market_5d')
    return features.replace([np.inf,-np.inf],np.nan),{'point_in_time_verified':False,'feature_groups':groups,
        'sector_status':statuses.get(sector,'UNAVAILABLE'),'source_statuses':statuses,
        'availability_policy':'Following UTC day estimate; retrospective revisions unverified'}

def market_regime(features):
    if 'market_trend_50' not in features or 'market_vol_20' not in features:
        return pd.Series('unknown',index=features.index)
    trend=features.market_trend_50
    threshold=features.market_vol_20.rolling(252,min_periods=60).median()
    regime=pd.Series(np.where(trend>.02,'bullish',np.where(trend<-.02,'bearish','sideways')),index=features.index)
    return regime+'_'+pd.Series(np.where(features.market_vol_20>threshold,'high_vol','low_vol'),index=features.index)
