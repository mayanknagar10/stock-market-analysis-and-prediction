"""Descriptive source liquidity/quality; never filters frozen membership."""
import numpy as np
from core.forecasting.service import adjusted_frame

def liquidity_diagnostics(snapshot,window=60):
    raw,_=adjusted_frame(snapshot,snapshot.metadata.fetched_at)
    frame=raw.tail(window);volume=frame.Volume.to_numpy(dtype=float);close=frame.Close.to_numpy(dtype=float)
    positive=np.isfinite(volume)&(volume>0)
    usable=positive&np.isfinite(close)&(close>0)
    turnover=close[usable]*volume[usable]
    return {'n_observed_sessions':len(frame),'n_usable_price_volume_sessions':int(usable.sum()),
        'positive_volume_fraction':float(positive.mean()) if len(frame) else None,
        'median_positive_volume_turnover_proxy':float(np.median(turnover)) if len(turnover) else None,
        'window_requested':window,'provider':snapshot.metadata.provider_id,'retrieved_at':snapshot.metadata.fetched_at,
        'last_observation':frame.index[-1].isoformat() if len(frame) else None,
        'selection_effect':'none; frozen membership retained',
        'measurement':'Vendor Close times Volume on completed positive-volume finite-price observations; not official NSE turnover'}
