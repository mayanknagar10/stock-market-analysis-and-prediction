"""
Premium data provider hook — inert until you configure a key.

This is the "better data" upgrade path from the roadmap (Polygon.io,
Tiingo). Both are free-tier but require signing up for an API key —
which you said you're not ready for yet. This module makes that upgrade
a 2-minute change later instead of a rewrite: as-is, every function here
returns None immediately (zero cost, zero behaviour change), because no
key is configured. The moment you add one, real requests start flowing
automatically — no other file needs to change.

How to activate later (pick one):

  Polygon.io (unlimited historical data free tier):
    1. Sign up at https://polygon.io/dashboard/signup
    2. Add to .streamlit/secrets.toml:
         POLYGON_API_KEY = "your_key_here"
    3. Done — fetch_ohlcv() in data_fetcher.py will automatically try
       Polygon first, before falling back to the existing free chain.

  Tiingo (500 req/hour free, clean EOD + fundamentals):
    1. Sign up at https://tiingo.com/account/profile
    2. Add to .streamlit/secrets.toml:
         TIINGO_API_KEY = "your_key_here"
    3. Same automatic activation as above.

Both keys can also be set as environment variables (POLYGON_API_KEY /
TIINGO_API_KEY) for local development without touching secrets.toml.
"""

import os
import requests
import pandas as pd
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

_TIMEOUT = 8


def _get_key(name: str) -> Optional[str]:
    """Checks st.secrets first (Streamlit Cloud convention), then env vars."""
    try:
        import streamlit as st
        if name in st.secrets:
            val = st.secrets[name]
            if val:
                return str(val)
    except Exception:
        pass
    return os.environ.get(name) or None


def has_premium_provider() -> bool:
    """True if ANY premium key is configured — used to decide whether
    it's worth even trying the premium path in fetch_ohlcv()."""
    return bool(_get_key("POLYGON_API_KEY") or _get_key("TIINGO_API_KEY"))


def fetch_polygon_ohlcv(ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
    """Returns None if no key configured or the request fails — caller
    falls back to the free chain in either case."""
    api_key = _get_key("POLYGON_API_KEY")
    if not api_key:
        return None
    try:
        from datetime import datetime, timedelta
        days_map = {"1mo": 30, "3mo": 90, "6mo": 182, "1y": 365,
                   "2y": 730, "5y": 1825, "max": 3650}
        days = days_map.get(period, 365)
        end = datetime.utcnow().date()
        start = end - timedelta(days=days)
        url = (f"https://api.polygon.io/v2/aggs/ticker/{ticker.upper()}/range/"
               f"1/day/{start.isoformat()}/{end.isoformat()}")
        r = requests.get(url, params={"apiKey": api_key, "adjusted": "true",
                                      "sort": "asc", "limit": 5000},
                         timeout=_TIMEOUT)
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return None
        df = pd.DataFrame(results)
        df["Date"] = pd.to_datetime(df["t"], unit="ms")
        df = df.set_index("Date").rename(columns={
            "o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
        return df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception:
        return None


def fetch_tiingo_ohlcv(ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
    api_key = _get_key("TIINGO_API_KEY")
    if not api_key:
        return None
    try:
        from datetime import datetime, timedelta
        days_map = {"1mo": 30, "3mo": 90, "6mo": 182, "1y": 365,
                   "2y": 730, "5y": 1825, "max": 3650}
        days = days_map.get(period, 365)
        start = (datetime.utcnow().date() - timedelta(days=days)).isoformat()
        url = f"https://api.tiingo.com/tiingo/daily/{ticker.upper()}/prices"
        r = requests.get(url, params={"startDate": start, "token": api_key},
                         timeout=_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        if not data:
            return None
        df = pd.DataFrame(data)
        df["Date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df = df.set_index("Date").rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume"})
        return df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception:
        return None


def fetch_premium_ohlcv(ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
    """Tries Polygon first, then Tiingo. Returns None if neither is
    configured or both fail — the caller (fetch_ohlcv) then proceeds to
    the existing free-tier chain exactly as before."""
    df = fetch_polygon_ohlcv(ticker, period)
    if df is not None and not df.empty:
        return df
    df = fetch_tiingo_ohlcv(ticker, period)
    if df is not None and not df.empty:
        return df
    return None
