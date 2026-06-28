"""
Professional data fetching module.
Supports US, NSE (.NS), BSE (.BO), indices and crypto via Yahoo Finance.

Fundamentals fetch uses three strategies in order:
  1. fast_info  — very reliable, real-time prices + 52W range + vol
  2. info       — full fundamentals; may fail/rate-limit for NSE
  3. OHLCV calc — always works; computes 52W range / avg vol from price data
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
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

# ── Company domain map for Clearbit logo lookup ─────────────────────────────
# Clearbit: https://logo.clearbit.com/{domain} — completely free, no API key
_NSE_DOMAINS = {
    "RELIANCE.NS": "ril.com",        "TCS.NS": "tcs.com",
    "HDFCBANK.NS": "hdfcbank.com",   "INFY.NS": "infosys.com",
    "ICICIBANK.NS": "icicibank.com", "HINDUNILVR.NS": "hul.co.in",
    "ITC.NS": "itcportal.com",       "SBIN.NS": "sbi.co.in",
    "BHARTIARTL.NS": "airtel.in",    "KOTAKBANK.NS": "kotak.com",
    "LT.NS": "larsentoubro.com",     "AXISBANK.NS": "axisbank.com",
    "ASIANPAINT.NS": "asianpaints.com", "MARUTI.NS": "marutisuzuki.com",
    "HCLTECH.NS": "hcltech.com",     "SUNPHARMA.NS": "sunpharma.com",
    "TITAN.NS": "titancompany.in",   "BAJFINANCE.NS": "bajajfinserv.in",
    "WIPRO.NS": "wipro.com",         "TATAMOTORS.NS": "tatamotors.com",
    "JSWSTEEL.NS": "jsw.in",         "NESTLEIND.NS": "nestle.in",
    "POWERGRID.NS": "powergridindia.com",
    "NTPC.NS": "ntpc.co.in",
    "ONGC.NS": "ongcindia.com",
    "TECHM.NS": "techmahindra.com",
}
_US_DOMAINS = {
    "AAPL": "apple.com",      "MSFT": "microsoft.com",  "GOOGL": "google.com",
    "AMZN": "amazon.com",     "NVDA": "nvidia.com",      "META": "meta.com",
    "TSLA": "tesla.com",      "JPM": "jpmorganchase.com","V": "visa.com",
    "XOM": "exxonmobil.com",  "UNH": "unitedhealthgroup.com",
    "JNJ": "jnj.com",         "WMT": "walmart.com",      "HD": "homedepot.com",
    "BAC": "bankofamerica.com","NFLX": "netflix.com",     "AMD": "amd.com",
    "INTC": "intel.com",      "BA": "boeing.com",        "GS": "goldmansachs.com",
    "AMGN": "amgen.com",      "COST": "costco.com",      "PEP": "pepsico.com",
    "KO": "coca-cola.com",    "MRK": "merck.com",        "ABBV": "abbvie.com",
    "CVX": "chevron.com",     "AVGO": "broadcom.com",    "CRM": "salesforce.com",
    "ORCL": "oracle.com",     "TMO": "thermofisher.com", "ADBE": "adobe.com",
    "PYPL": "paypal.com",     "UBER": "uber.com",        "SPOT": "spotify.com",
    "SHOP": "shopify.com",    "SQ": "squareup.com",      "SNAP": "snap.com",
    "TWTR": "twitter.com",    "DIS": "disney.com",       "SBUX": "starbucks.com",
}
_ALL_DOMAINS = {**_NSE_DOMAINS, **_US_DOMAINS}


def currency_symbol(currency: str) -> str:
    s = str(currency).strip()
    if len(s) == 3 and s.isalpha():
        return CURRENCY_SYMBOLS.get(s.upper(), s)
    return s


def detect_market(ticker: str) -> str:
    t = ticker.upper().strip()
    if t.endswith(".NS") or t.startswith("^NSEI") or t.startswith("^NSEBANK"):
        return "NSE"
    if t.endswith(".BO") or t.startswith("^BSESN"):
        return "BSE"
    return "US"


def get_logo_url(ticker: str, website: str = "") -> str:
    """
    Returns a Clearbit logo URL (free, no API key).
    Tries: known domain map → yfinance website field → best-guess domain.
    Always returns a string (empty if all fail).
    """
    t = ticker.upper().strip()
    # 1. Known domain map
    domain = _ALL_DOMAINS.get(t, "")
    # 2. From yfinance website field
    if not domain and website:
        d = website.lower().replace("https://", "").replace("http://", "")
        d = d.split("/")[0].lstrip("www.")
        if "." in d:
            domain = d
    if domain:
        return f"https://logo.clearbit.com/{domain}"
    return ""


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
        df = df[df["Close"] > 0]
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fundamentals(ticker: str) -> Dict:
    """
    Fetch fundamentals using three strategies in order of reliability:

    Strategy 1 — fast_info (very reliable, minimal rate-limiting):
        Gives market_cap, 52W range, currency, avg_volume from a lightweight
        cached endpoint that rarely fails even for NSE stocks.

    Strategy 2 — info (full fundamentals, may fail for NSE):
        P/E, EPS, beta, margins, ROE, description etc. Wrapped in its own
        try/except so a failure here doesn't lose Strategy 1 data.

    Strategy 3 — OHLCV fallback (always works):
        Computes 52W high/low and avg volume from price data already in
        cache if both of the above failed to populate those fields.
    """
    t_sym = ticker.upper().strip()
    mkt   = detect_market(t_sym)
    curr  = "INR" if mkt in ("NSE", "BSE") else "USD"

    result: Dict = {
        "name": t_sym, "sector": "—", "industry": "—",
        "market_cap": None, "pe_ttm": None, "pe_fwd": None, "eps": None,
        "dividend_yield": None, "beta": None,
        "week52_high": None, "week52_low": None,
        "avg_volume_10d": None, "avg_volume_3m": None,
        "description": "", "website": "", "employees": None,
        "currency": curr, "exchange": mkt,
        "revenue_ttm": None, "gross_margin": None,
        "operating_margin": None, "roe": None, "debt_equity": None,
        "logo_url": "",
    }

    t_obj = yf.Ticker(t_sym)

    # ── Strategy 1: fast_info ─────────────────────────────────────────────
    try:
        fi = t_obj.fast_info
        if fi:
            result["market_cap"]    = _safe(getattr(fi, "market_cap", None))
            result["week52_high"]   = _safe(getattr(fi, "fifty_two_week_high", None))
            result["week52_low"]    = _safe(getattr(fi, "fifty_two_week_low", None))
            result["currency"]      = str(getattr(fi, "currency", curr) or curr)
            result["avg_volume_3m"] = _safe(getattr(fi, "three_month_average_volume", None))
            result["avg_volume_10d"]= _safe(getattr(fi, "ten_day_average_volume", None))
    except Exception:
        pass

    # ── Strategy 2: info ─────────────────────────────────────────────────
    try:
        info: Dict = t_obj.info or {}
        # yfinance sometimes returns a 1-key dict {'trailingPegRatio': None}
        # when the endpoint is down — ignore it
        if info and len(info) > 5:
            result["name"]        = info.get("longName") or info.get("shortName") or t_sym
            result["sector"]      = info.get("sector")   or "—"
            result["industry"]    = info.get("industry") or "—"
            result["pe_ttm"]      = _safe(info.get("trailingPE"))
            result["pe_fwd"]      = _safe(info.get("forwardPE"))
            result["eps"]         = _safe(info.get("trailingEps"))
            result["dividend_yield"] = _safe(info.get("dividendYield"))
            result["beta"]        = _safe(info.get("beta"))
            result["description"] = info.get("longBusinessSummary") or ""
            result["website"]     = info.get("website") or ""
            result["employees"]   = info.get("fullTimeEmployees")
            result["exchange"]    = info.get("exchange") or mkt
            result["revenue_ttm"]      = _safe(info.get("totalRevenue"))
            result["gross_margin"]     = _safe(info.get("grossMargins"))
            result["operating_margin"] = _safe(info.get("operatingMargins"))
            result["roe"]         = _safe(info.get("returnOnEquity"))
            result["debt_equity"] = _safe(info.get("debtToEquity"))
            # Only override if fast_info didn't populate these
            if result["market_cap"] is None:
                result["market_cap"]  = _safe(info.get("marketCap"))
            if result["week52_high"] is None:
                result["week52_high"] = _safe(info.get("fiftyTwoWeekHigh"))
            if result["week52_low"] is None:
                result["week52_low"]  = _safe(info.get("fiftyTwoWeekLow"))
            if result["avg_volume_3m"] is None:
                result["avg_volume_3m"] = _safe(info.get("averageVolume"))
            if result["avg_volume_10d"] is None:
                result["avg_volume_10d"] = _safe(info.get("averageVolume10days"))
    except Exception:
        pass

    # ── Strategy 3: OHLCV fallback ────────────────────────────────────────
    # 52W range and avg volume are always calculable from price data
    if result["week52_high"] is None or result["week52_low"] is None:
        try:
            df1y = fetch_ohlcv(t_sym, period="1y")
            if not df1y.empty:
                result["week52_high"] = float(df1y["High"].max())
                result["week52_low"]  = float(df1y["Low"].min())
                if result["avg_volume_3m"] is None:
                    result["avg_volume_3m"] = float(df1y["Volume"].tail(63).mean())
                if result["avg_volume_10d"] is None:
                    result["avg_volume_10d"] = float(df1y["Volume"].tail(10).mean())
        except Exception:
            pass

    # ── Logo ─────────────────────────────────────────────────────────────
    result["logo_url"] = get_logo_url(t_sym, result.get("website", ""))

    return result


def _safe(v):
    """Return v if it's a usable number, else None."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else f
    except (TypeError, ValueError):
        return None


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
            mkt = detect_market(ticker)
            hint = (" For NSE stocks add .NS (e.g. RELIANCE.NS)"
                    if mkt == "US" and "." not in ticker else "")
            return False, f"No data found for '{ticker}'.{hint}"
        return True, ""
    except Exception as e:
        return False, str(e)


def compute_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change().dropna()
