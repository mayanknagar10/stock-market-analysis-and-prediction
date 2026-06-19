"""
Professional data fetching module.
Supports US, NSE (.NS), BSE (.BO), indices and crypto via Yahoo Finance.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
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

CURRENCY_SYMBOLS: Dict[str, str] = {
    "INR": "₹", "USD": "$", "EUR": "€", "GBP": "£",
    "JPY": "¥", "CNY": "¥", "AUD": "A$", "CAD": "C$",
    "HKD": "HK$", "SGD": "S$",
}


def currency_symbol(currency: str) -> str:
    """
    Return display symbol for a currency.
    - ISO code  → look up in table  (e.g. 'USD' → '$')
    - Already a symbol → return as-is (e.g. '$' → '$', '₹' → '₹')
    No trailing spaces are added.
    """
    s = str(currency).strip()
    # If it looks like an ISO code (3 alpha chars) → look up
    if len(s) == 3 and s.isalpha():
        return CURRENCY_SYMBOLS.get(s.upper(), s)
    # Otherwise treat as a symbol already (pass through unchanged)
    return s


def detect_market(ticker: str) -> str:
    """Infer market from ticker suffix. Returns 'NSE', 'BSE', or 'US'."""
    t = ticker.upper().strip()
    if t.endswith(".NS") or t.startswith("^NSEI") or t.startswith("^NSEBANK"):
        return "NSE"
    if t.endswith(".BO") or t.startswith("^BSESN"):
        return "BSE"
    return "US"


@st.cache_data(ttl=300, show_spinner=False)
def fetch_ohlcv(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV from Yahoo Finance. Returns empty DataFrame on failure."""
    try:
        df = yf.Ticker(ticker.upper().strip()).history(
            period=period, interval=interval, auto_adjust=True
        )
        if df is None or df.empty:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.dropna(subset=["Close"], inplace=True)
        for col in ["Open", "High", "Low", "Close"]:
            df = df[df[col] > 0]
        return df
    except Exception as e:
        st.error(f"⚠️ Could not fetch **{ticker}**: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fundamentals(ticker: str) -> Dict:
    defaults = dict(
        name=ticker, sector="—", industry="—",
        market_cap=None, pe_ttm=None, pe_fwd=None, eps=None,
        dividend_yield=None, beta=None, week52_high=None, week52_low=None,
        avg_volume_10d=None, avg_volume_3m=None, description="",
        website="", employees=None,
        currency="INR" if detect_market(ticker) in ("NSE","BSE") else "USD",
        exchange=detect_market(ticker),
        revenue_ttm=None, gross_margin=None, operating_margin=None,
        roe=None, debt_equity=None,
    )
    try:
        info = yf.Ticker(ticker.upper().strip()).info or {}
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
            "currency":         info.get("currency", defaults["currency"]),
            "exchange":         info.get("exchange", defaults["exchange"]),
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
    try:
        return (yf.Ticker(ticker.upper().strip()).news or [])[:max_items]
    except Exception:
        return []


@st.cache_data(ttl=300, show_spinner=False)
def fetch_benchmark(period: str = "1y", market: str = "US") -> pd.DataFrame:
    sym = {"US": "^GSPC", "NSE": "^NSEI", "BSE": "^BSESN"}.get(market, "^GSPC")
    return fetch_ohlcv(sym, period=period)


def validate_ticker(ticker: str) -> Tuple[bool, str]:
    try:
        df = fetch_ohlcv(ticker, period="5d")
        if df.empty:
            hint = " For NSE stocks add .NS (e.g. RELIANCE.NS)" \
                   if detect_market(ticker) == "US" and "." not in ticker else ""
            return False, f"No data found for '{ticker}'.{hint}"
        return True, ""
    except Exception as e:
        return False, str(e)


def compute_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change().dropna()
