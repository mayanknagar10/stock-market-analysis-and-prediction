"""Research panel from frozen adjusted bars; timestamp estimates remain explicitly unverified."""
import numpy as np
import pandas as pd
from core.indicators import build_ml_features
from core.validation.alignment import session_cutoffs
from core.validation.targets import direct_return_targets

def build_panel(frames,manifest,horizon,family,references=None,settings=None):
    features=[]
    labels=[]
    metadata=[]
    group_map={}
    for ticker,frame in frames.items():
        info=manifest['assets'][ticker]
        if info['family']!=family: continue
        from core.data.cleaning import research_bar_cleanup
        independent=[]
        for peer,peer_frame in frames.items():
            if peer!=ticker and manifest['assets'][peer]['family']==family:
                independent.append(peer_frame.index[peer_frame.Volume>0])
        if not independent: raise ValueError('Independent session calendar required')
        calendar=independent[0]
        for index in independent[1:]: calendar=calendar.union(index)
        calendar=calendar[(calendar>=frame.index.min())&(calendar<=frame.index.max())].sort_values()
        frame,source_quality=research_bar_cleanup(frame,calendar)
        if not frame.index.equals(calendar): raise ValueError('Missing or extra session in stock versus independent calendar: '+ticker)
        if references is not None:
            from core.forecasting.context import research_feature_frame,market_regime
            stock,context_meta=research_feature_frame(frame,ticker,family,references,settings or {},research_only=True)
            for group,columns in context_meta['feature_groups'].items(): group_map.setdefault(group,set()).update(columns)
            context_regime=market_regime(stock)
        else:
            stock=build_ml_features(frame).replace([np.inf,-np.inf],np.nan)
            context_regime=None
        times=session_cutoffs(calendar,family)+pd.Timedelta('20min')
        close=pd.Series(frame.Close.to_numpy(),index=times)
        target=direct_return_targets(close,times,horizons=[horizon])
        stock.index=times
        source_bad=pd.Series(frame.Volume.to_numpy()<=0,index=times)
        bad_forward=source_bad.rolling(horizon+1,min_periods=1).sum().shift(-horizon)>0
        bad_features=source_bad.rolling(200,min_periods=1).sum()>0
        mask=stock.notna().all(axis=1)&target['target_'+str(horizon)+'d'].notna()&~bad_forward&~bad_features
        stock=stock.loc[mask]
        y=target.loc[mask,'target_'+str(horizon)+'d']
        trend=frame.Close/frame.Close.rolling(50).mean()-1
        volatility=np.log(frame.Close).diff().rolling(20).std()
        vol_threshold=volatility.rolling(252,min_periods=60).median()
        regime=np.where(trend>.02,'bullish',np.where(trend<-.02,'bearish','sideways'))
        regime=pd.Series(regime,index=times)+'_'+pd.Series(np.where(volatility>vol_threshold,'high_vol','low_vol'),index=times)
        meta=pd.DataFrame({'ticker':ticker,'sector':info.get('sector','unknown'),'family':family,
            'feature_time':stock.index,'label_end':target.loc[mask,'label_end_'+str(horizon)+'d'].to_numpy(),
            'origin_session':pd.Series([d.date().isoformat() for d in frame.index],index=times).loc[mask].to_numpy(),
            'maturity_session':pd.Series([d.date().isoformat() for d in frame.index],index=times).shift(-horizon).loc[mask].to_numpy(),
            'year':stock.index.year,'stock_regime_proxy':regime.loc[mask].to_numpy(),
            'point_in_time_verified':False,
            'market_regime':context_regime.loc[stock.index].to_numpy() if context_regime is not None else 'unknown'})
        features.append(stock.reset_index(drop=True))
        labels.append(y.reset_index(drop=True))
        metadata.append(meta)
    if not features: raise ValueError('No data for model family: '+family)
    common=set(features[0].columns)
    for part in features[1:]: common.intersection_update(part.columns)
    columns=[c for c in features[0] if c in common]
    X=pd.concat([part[columns] for part in features],ignore_index=True)
    X.attrs['feature_groups']={g:[c for c in columns if c in values] for g,values in group_map.items()}
    y=pd.concat(labels,ignore_index=True)
    meta=pd.concat(metadata,ignore_index=True)
    order=meta.sort_values(['feature_time','ticker']).index
    result=X.loc[order].reset_index(drop=True)
    result.attrs=X.attrs.copy()
    return result,y.loc[order].reset_index(drop=True),meta.loc[order].reset_index(drop=True)
