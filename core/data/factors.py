"""Kenneth French provider; latest vintages are research-only."""
import pandas as pd
from typing import Optional
try:
    import pandas_datareader.data as web
    _PDR_AVAILABLE=True
except ImportError:
    _PDR_AVAILABLE=False
FF_FACTOR_SETS={
    '3-Factor (Mkt, SMB, HML)':'F-F_Research_Data_Factors',
    '5-Factor (+ RMW, CMA)':'F-F_Research_Data_5_Factors_2x3',
}

def fetch_fama_french_factors(factor_set: str = "5-Factor (+ RMW, CMA)",
                              start: Optional[str] = None) -> pd.DataFrame:
    """
    Monthly Fama-French factor returns, in decimal (0.0123 = 1.23%).
    Columns are a subset of: Mkt-RF, SMB, HML, RMW, CMA, RF.

    Returns empty DataFrame if pandas_datareader isn't installed or the
    download fails (e.g. no internet access) — callers should handle
    that gracefully, same pattern as every other external_data fetcher
    in this app.
    """
    if not _PDR_AVAILABLE:
        return pd.DataFrame()
    dataset_name = FF_FACTOR_SETS.get(factor_set, "F-F_Research_Data_5_Factors_2x3")
    try:
        raw = web.DataReader(dataset_name, "famafrench", start=start)
        df = raw[0].copy()
        df.index = df.index.to_timestamp()
        df = df / 100.0
        return df
    except Exception:
        return pd.DataFrame()


