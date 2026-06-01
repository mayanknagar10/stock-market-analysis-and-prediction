"""
Professional data fetching module.
Handles OHLCV, fundamentals, news, and multi-ticker fetching with robust caching.

Market support:
  US equities  :  AAPL, MSFT, TSLA …
  NSE (India)  :  RELIANCE.NS, TCS.NS, INFY.NS …
  BSE (India)  :  RELIANCE.BO, TCS.BO …
  US indices   :  ^GSPC, ^IXIC, ^DJI
  NSE indices  :  ^NSEI (Nifty 50), ^NSEBANK (Bank Nifty)
  BSE index    :  ^BSESN (Sensex)
  Crypto       :  BTC-USD, ETH-USD …
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
import streamlit as st
import warnings
warnings.filterwarnings("ignore")


PERIOD_MAP = {
    "1 Month":  ("1mo",  "1d"),
    "3 Months": ("3mo",  "1d"),
    "6 Months": ("6mo",  "1d"),
    "1 Year":   ("1y",   "1d"),
    "2 Years":  ("2y",   "1wk"),
    "5 Years":  ("5y",   "1wk"),
    "Max":      ("max",  "1mo"),
}

# ── Currency helpers ───────────────────────────────────────────────────────

CURRENCY_SYMBOLS = {
    "INR": "₹",
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
    "JPY": "¥",
    "CNY": "¥",
    "AUD": "A$",
    "CAD": "C$",
    "HKD": "HK$",
    "SGD": "S$",
}

def currency_symbol(currency: str) -> str:
    """Return the display symbol for a given ISO currency code."""
    return CURRENCY_SYMBOLS.get(currency.upper(), currency + " ")


def detect_market(ticker: str) -> str:
    """
    Infer the market from the ticker suffix.
    Returns 'NSE', 'BSE', 'US', or 'OTHER'.
    """
    t = ticker.upper().strip()
    if t.endswith(".NS"):
        return "NSE"
    if t.endswith(".BO"):
        return "BSE"
    if t.startswith("^NSEI") or t.startswith("^BSE"):
        return "NSE"
    return "US"


def normalise_ticker(raw: str) -> str:
    """
    Accept common shorthand and return a Yahoo-Finance-compatible symbol.
    e.g.  'RELIANCE'  →  'RELIANCE.NS'  is NOT done here — the user must
    type the suffix.  This function just strips whitespace and uppercases.
    """
    return raw.upper().strip()


# ── Data fetching ──────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def fetch_ohlcv(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch OHLCV data from Yahoo Finance.
    Works for US, NSE (.NS), BSE (.BO), indices, and crypto.
    Returns a cleaned DataFrame with tz-naive DatetimeIndex, or empty DF on failure.
    """
    try:
        t = normalise_ticker(ticker)
        stock = yf.Ticker(t)
        df = stock.history(period=period, interval=interval, auto_adjust=True)

        if df is None or df.empty:
            return pd.DataFrame()

        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.dropna(subset=["Close"], inplace=True)

        # NSE/BSE indices have Volume = 0 legitimately, so only filter for equities
        if not ticker.startswith("^"):
            df = df[df["Volume"] >= 0]

        for col in ["Open", "High", "Low", "Close"]:
            df = df[df[col] > 0]

        return df

    except Exception as e:
        st.error(f"⚠️  Could not fetch data for **{ticker}**: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fundamentals(ticker: str) -> Dict:
    """
    Fetch fundamental info.
    Correctly returns INR-denominated values for NSE/BSE tickers.
    """
    defaults = {
        "name": ticker, "sector": "—", "industry": "—",
        "market_cap": None, "pe_ttm": None, "pe_fwd": None,
        "eps": None, "dividend_yield": None, "beta": None,
        "week52_high": None, "week52_low": None,
        "avg_volume_10d": None, "avg_volume_3m": None,
        "description": "", "website": "", "employees": None,
        "currency": "INR" if detect_market(ticker) in ("NSE", "BSE") else "USD",
        "exchange": detect_market(ticker),
        "revenue_ttm": None, "gross_margin": None,
        "operating_margin": None, "roe": None, "debt_equity": None,
    }
    try:
        info = yf.Ticker(normalise_ticker(ticker)).info or {}
        currency = info.get("currency", defaults["currency"])
        return {
            "name":             info.get("longName", ticker),
            "sector":           info.get("sector", "—"),
            "industry":         info.get("industry", "—"),
            "market_cap":       info.get("marketCap"),
            "pe_ttm":           info.get("trailingPE"),
            "pe_fwd":           info.get("forwardPE"),
            "eps":              info.get("trailingEps"),
            "dividend_yield":   info.get("dividendYield"),
            "beta":             info.get("beta"),
            "week52_high":      info.get("fiftyTwoWeekHigh"),
            "week52_low":       info.get("fiftyTwoWeekLow"),
            "avg_volume_10d":   info.get("averageVolume10days"),
            "avg_volume_3m":    info.get("averageVolume"),
            "description":      info.get("longBusinessSummary", ""),
            "website":          info.get("website", ""),
            "employees":        info.get("fullTimeEmployees"),
            "currency":         currency,
            "exchange":         info.get("exchange", detect_market(ticker)),
            "revenue_ttm":      info.get("totalRevenue"),
            "gross_margin":     info.get("grossMargins"),
            "operating_margin": info.get("operatingMargins"),
            "roe":              info.get("returnOnEquity"),
            "debt_equity":      info.get("debtToEquity"),
        }
    except Exception:
        return defaults


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_news(ticker: str, max_items: int = 8) -> List[Dict]:
    """Fetch recent news headlines for a ticker."""
    try:
        items = yf.Ticker(normalise_ticker(ticker)).news or []
        return items[:max_items]
    except Exception:
        return []


@st.cache_data(ttl=300, show_spinner=False)
def fetch_benchmark(period: str = "1y", market: str = "US") -> pd.DataFrame:
    """
    Fetch a default benchmark based on the detected market.
      US  →  ^GSPC  (S&P 500)
      NSE →  ^NSEI  (Nifty 50)
      BSE →  ^BSESN (Sensex)
    """
    bench_map = {
        "US":  "^GSPC",
        "NSE": "^NSEI",
        "BSE": "^BSESN",
    }
    symbol = bench_map.get(market, "^GSPC")
    return fetch_ohlcv(symbol, period=period)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_multi(tickers: Tuple[str, ...], period: str = "1y") -> Dict[str, pd.DataFrame]:
    """Fetch OHLCV for multiple tickers. Accepts tuple for hashability."""
    return {t: fetch_ohlcv(t, period) for t in tickers}


def returns_series(df: pd.DataFrame, col: str = "Close") -> pd.Series:
    """Compute log returns from a price series."""
    return np.log(df[col] / df[col].shift(1)).dropna()


def validate_ticker(ticker: str) -> Tuple[bool, str]:
    """Quick validation: check if a ticker exists and has recent data."""
    try:
        df = fetch_ohlcv(ticker, period="5d")
        if df.empty:
            market = detect_market(ticker)
            hint = ""
            if market == "US" and "." not in ticker and not ticker.startswith("^"):
                hint = " For NSE stocks add .NS suffix, e.g. RELIANCE.NS"
            return False, f"No data found for '{ticker}'.{hint}"
        return True, ""
    except Exception as e:
        return False, str(e)
