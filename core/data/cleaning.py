"""Explicit research quarantine of provider placeholder bars; originals remain in snapshots."""
import numpy as np
import pandas as pd

def research_bar_cleanup(frame,independent_sessions):
    dates=pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
    expected=pd.DatetimeIndex(independent_sessions).tz_localize(None).normalize()
    flat=(frame.Volume==0)&np.isclose(frame.High,frame.Low,rtol=0,atol=1e-10)&np.isclose(frame.Open,frame.Close,rtol=0,atol=1e-10)
    non_session=~dates.isin(expected)
    quarantine=flat.to_numpy()&non_session
    unresolved=flat.to_numpy()&~non_session
    return frame.iloc[np.flatnonzero(~quarantine)].copy(),{
        'quarantined_non_session_dates':[d.date().isoformat() for d in dates[quarantine]],
        'unresolved_session_dates':[d.date().isoformat() for d in dates[unresolved]],
        'point_in_time_verified':False,
        'policy':'Only flat zero-volume bars absent from independent observed trading evidence are excluded; raw snapshot unchanged'}
