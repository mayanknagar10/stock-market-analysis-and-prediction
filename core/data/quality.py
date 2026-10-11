"""Structural price health; model eligibility additionally requires snapshot as-of checks."""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from core.validation.alignment import aware_index

@dataclass(frozen=True)
class DataHealth:
    eligible: bool
    price_data_status: str
    overall_quality_score: float
    latest_timestamp: str | None
    stale_flags: tuple[str,...]
    critical_flags: tuple[str,...]
    sector_data_status: str='NOT_REQUESTED'
    global_data_status: str='NOT_REQUESTED'
    macro_data_status: str='NOT_REQUESTED'
    news_status: str='NOT_REQUESTED'
    fundamental_status: str='NOT_REQUESTED'
    eligibility_basis: str='Structural checks only; require verified snapshot availability/vintage gate'

    def to_dict(self): return asdict(self)

def validate_market_data(frame,as_of,expected_sessions,max_adjusted_jump=.35):
    if not 0<max_adjusted_jump<1: raise ValueError('Invalid jump threshold')
    cutoff=aware_index([as_of],'as_of')[0]
    expected=aware_index(expected_sessions,'expected exchange sessions')
    if not len(expected) or not expected.is_unique or not expected.is_monotonic_increasing or expected.max()>cutoff:
        raise ValueError('Expected sessions must be completed, nonempty, unique and ordered')
    flags=[]
    stale=[]
    latest=None
    required={'Open','High','Low','Close','Adj Close','Volume','Dividends','Stock Splits'}
    if frame.empty: flags.append('no_price_data')
    try:
        dates=aware_index(frame.index,'price source timestamps')
        if len(dates): latest=dates.max().isoformat()
        if dates.has_duplicates: flags.append('duplicate_timestamps')
        if not dates.is_monotonic_increasing: flags.append('unordered_timestamps')
        if len(dates) and dates.max()>cutoff: flags.append('future_candle')
        if len(dates)==0 or dates.max()<expected.max(): stale.append('stale_price_data')
        if not dates.equals(expected): flags.append('missing_or_extra_sessions')
    except ValueError: flags.append('invalid_timezone')
    if not required.issubset(frame.columns): flags.append('missing_ohlcv_or_actions')
    elif not frame.empty:
        values=frame[list(required)].to_numpy(dtype=float)
        if not np.isfinite(values).all(): flags.append('missing_or_nonfinite_values')
        if (frame[['Open','High','Low','Close','Adj Close']]<=0).any().any(): flags.append('nonpositive_prices')
        if (frame.Volume<0).any() or (frame.Volume==0).all(): flags.append('invalid_volume')
        if (frame.High<frame[['Open','Low','Close']].max(axis=1)).any() or (frame.Low>frame[['Open','High','Close']].min(axis=1)).any():
            flags.append('impossible_ohlc')
        if (frame.Dividends<0).any() or (frame['Stock Splits']<0).any(): flags.append('invalid_actions')
        adjusted_jumps=frame['Adj Close'].pct_change(fill_method=None).abs()
        if (adjusted_jumps>max_adjusted_jump).any(): flags.append('abnormal_adjusted_jump')
        actions=(frame['Stock Splits']!=0)|(frame.Dividends!=0)
        raw_jumps=frame.Close.pct_change(fill_method=None).abs()
        if ((raw_jumps>max_adjusted_jump)&~actions).any(): flags.append('unresolved_corporate_action')
        factor=frame['Adj Close']/frame.Close
        if ((factor.pct_change(fill_method=None).abs()>.001)&~actions).any(): flags.append('unexplained_adjustment_change')
    flags=list(dict.fromkeys(flags+stale))
    # Transparent passed-check fraction, NOT forecast confidence/probability.
    score=max(0.,1.-len(flags)/12.)
    return DataHealth(not flags,'OK' if not flags else 'INVALID',score,latest,tuple(stale),tuple(flags))
