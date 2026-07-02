"""
External data sources that require NO account, NO login, and NO API key.

Every function here calls a plain public endpoint with `requests`. If any
of them ever start requiring auth, they fail safe (return empty data) —
the rest of the app keeps working normally.

Sources used:
  - CoinGecko       — crypto prices, market cap, historical charts
  - Frankfurter.app — live + historical FX rates (ECB data, no key)
  - World Bank API  — GDP, inflation, interest rates by country
  - SEC EDGAR       — US company filings (10-K, 10-Q, 8-K) — no key
  - Stooq           — historical OHLCV fallback when Yahoo Finance is down
"""

import requests
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

_TIMEOUT = 8
_HEADERS = {"User-Agent": "StockProAnalytics/1.0 (research tool)"}


# ─────────────────────────────────────────────────────────────────
# COINGECKO — crypto, no key required
# ─────────────────────────────────────────────────────────────────

_COINGECKO_BASE = "https://api.coingecko.com/api/v3"

# Common ticker → CoinGecko id map (CoinGecko uses slugs, not tickers)
_COINGECKO_IDS = {
    "BTC": "bitcoin", "ETH": "ethereum", "BNB": "binancecoin",
    "SOL": "solana", "XRP": "ripple", "ADA": "cardano",
    "DOGE": "dogecoin", "AVAX": "avalanche-2", "DOT": "polkadot",
    "MATIC": "matic-network", "LINK": "chainlink", "LTC": "litecoin",
    "TRX": "tron", "SHIB": "shiba-inu", "UNI": "uniswap",
    "ATOM": "cosmos", "XLM": "stellar", "NEAR": "near",
}


def is_crypto_ticker(ticker: str) -> bool:
    t = ticker.upper().strip().replace("-USD", "").replace("-USDT", "")
    return t in _COINGECKO_IDS


@st.cache_data(ttl=120, show_spinner=False)
def fetch_crypto_price(ticker: str) -> Dict:
    """Live price + 24h change + market cap for a crypto ticker. No key needed."""
    t = ticker.upper().strip().replace("-USD", "").replace("-USDT", "")
    coin_id = _COINGECKO_IDS.get(t)
    if not coin_id:
        return {}
    try:
        r = requests.get(
            f"{_COINGECKO_BASE}/simple/price",
            params={"ids": coin_id, "vs_currencies": "usd,inr",
                   "include_market_cap": "true", "include_24hr_change": "true",
                   "include_24hr_vol": "true"},
            headers=_HEADERS, timeout=_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json().get(coin_id, {})
        if not data:
            return {}
        return {
            "price_usd": data.get("usd"),
            "price_inr": data.get("inr"),
            "market_cap_usd": data.get("usd_market_cap"),
            "change_24h_pct": data.get("usd_24h_change"),
            "volume_24h_usd": data.get("usd_24h_vol"),
        }
    except Exception:
        return {}


@st.cache_data(ttl=300, show_spinner=False)
def fetch_crypto_history(ticker: str, days: int = 365) -> pd.DataFrame:
    """Historical daily OHLC-equivalent (CoinGecko gives close-only by default)."""
    t = ticker.upper().strip().replace("-USD", "").replace("-USDT", "")
    coin_id = _COINGECKO_IDS.get(t)
    if not coin_id:
        return pd.DataFrame()
    try:
        r = requests.get(
            f"{_COINGECKO_BASE}/coins/{coin_id}/market_chart",
            params={"vs_currency": "usd", "days": days, "interval": "daily"},
            headers=_HEADERS, timeout=_TIMEOUT,
        )
        r.raise_for_status()
        prices = r.json().get("prices", [])
        if not prices:
            return pd.DataFrame()
        df = pd.DataFrame(prices, columns=["ts", "Close"])
        df["Date"] = pd.to_datetime(df["ts"], unit="ms").dt.normalize()
        df = df.set_index("Date")[["Close"]]
        # CoinGecko free tier doesn't give O/H/L on the daily endpoint —
        # approximate from neighbouring closes so downstream chart code
        # (which expects OHLCV) doesn't break.
        df["Open"]  = df["Close"].shift(1).fillna(df["Close"])
        df["High"]  = df[["Open", "Close"]].max(axis=1) * 1.002
        df["Low"]   = df[["Open", "Close"]].min(axis=1) * 0.998
        df["Volume"] = 0.0
        return df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=120, show_spinner=False)
def fetch_crypto_top_movers(limit: int = 20) -> pd.DataFrame:
    """Top coins by market cap with 24h change — for a crypto screener/overview."""
    try:
        r = requests.get(
            f"{_COINGECKO_BASE}/coins/markets",
            params={"vs_currency": "usd", "order": "market_cap_desc",
                   "per_page": limit, "page": 1, "price_change_percentage": "24h"},
            headers=_HEADERS, timeout=_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        if not data:
            return pd.DataFrame()
        rows = [{
            "Symbol": c["symbol"].upper(), "Name": c["name"],
            "Price USD": c["current_price"],
            "24h %": c.get("price_change_percentage_24h"),
            "Market Cap": c["market_cap"], "Volume 24h": c["total_volume"],
        } for c in data]
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────
# FRANKFURTER — live + historical FX rates, no key, ECB-sourced
# ─────────────────────────────────────────────────────────────────

_FRANKFURTER_BASE = "https://api.frankfurter.app"


@st.cache_data(ttl=600, show_spinner=False)
def fetch_fx_rate(base: str = "USD", quote: str = "INR") -> Optional[float]:
    """Live exchange rate, e.g. fetch_fx_rate('USD','INR') -> 83.45."""
    try:
        r = requests.get(f"{_FRANKFURTER_BASE}/latest",
                         params={"from": base.upper(), "to": quote.upper()},
                         timeout=_TIMEOUT)
        r.raise_for_status()
        return float(r.json()["rates"][quote.upper()])
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def fetch_fx_history(base: str = "USD", quote: str = "INR", days: int = 90) -> pd.Series:
    """Historical daily FX series for the last `days` days."""
    try:
        end = datetime.utcnow().date()
        start = end - timedelta(days=days)
        r = requests.get(
            f"{_FRANKFURTER_BASE}/{start.isoformat()}..{end.isoformat()}",
            params={"from": base.upper(), "to": quote.upper()},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        rates = r.json().get("rates", {})
        if not rates:
            return pd.Series(dtype=float)
        s = pd.Series({pd.Timestamp(d): v[quote.upper()] for d, v in rates.items()})
        return s.sort_index()
    except Exception:
        return pd.Series(dtype=float)


# ─────────────────────────────────────────────────────────────────
# WORLD BANK — macro indicators (GDP, inflation, rates), no key
# ─────────────────────────────────────────────────────────────────

_WORLDBANK_BASE = "https://api.worldbank.org/v2"

WORLDBANK_INDICATORS = {
    "GDP Growth (%)":        "NY.GDP.MKTP.KD.ZG",
    "Inflation, CPI (%)":    "FP.CPI.TOTL.ZG",
    "Real Interest Rate (%)":"FR.INR.RINR",
    "Unemployment (%)":      "SL.UEM.TOTL.ZS",
    "GDP (current US$)":     "NY.GDP.MKTP.CD",
}

# ISO-3 country codes commonly relevant to NSE/US split
WORLDBANK_COUNTRIES = {"India": "IND", "United States": "USA",
                       "China": "CHN", "United Kingdom": "GBR",
                       "Japan": "JPN", "Germany": "DEU"}


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_macro_indicator(country_code: str, indicator_code: str,
                          start_year: int = 2015) -> pd.Series:
    """One macro indicator's history for one country. No key needed."""
    try:
        end_year = datetime.utcnow().year
        r = requests.get(
            f"{_WORLDBANK_BASE}/country/{country_code}/indicator/{indicator_code}",
            params={"format": "json", "date": f"{start_year}:{end_year}", "per_page": 100},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        payload = r.json()
        if len(payload) < 2 or not payload[1]:
            return pd.Series(dtype=float)
        rows = {int(d["date"]): d["value"] for d in payload[1] if d["value"] is not None}
        return pd.Series(rows).sort_index()
    except Exception:
        return pd.Series(dtype=float)


# ─────────────────────────────────────────────────────────────────
# SEC EDGAR — US company filings, completely free, no key
# ─────────────────────────────────────────────────────────────────

_SEC_BASE = "https://data.sec.gov"
_SEC_HEADERS = {"User-Agent": "StockProAnalytics research@example.com"}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_sec_cik(ticker: str) -> Optional[str]:
    """Look up a US ticker's 10-digit CIK number (needed for filing lookups)."""
    try:
        r = requests.get("https://www.sec.gov/files/company_tickers.json",
                         headers=_SEC_HEADERS, timeout=_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        t = ticker.upper().strip()
        for entry in data.values():
            if entry.get("ticker", "").upper() == t:
                return str(entry["cik_str"]).zfill(10)
        return None
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_sec_filings(ticker: str, form_types: Optional[List[str]] = None,
                      limit: int = 10) -> pd.DataFrame:
    """Recent SEC filings (10-K, 10-Q, 8-K, etc.) for a US ticker. No key needed."""
    cik = fetch_sec_cik(ticker)
    if not cik:
        return pd.DataFrame()
    try:
        r = requests.get(f"{_SEC_BASE}/submissions/CIK{cik}.json",
                         headers=_SEC_HEADERS, timeout=_TIMEOUT)
        r.raise_for_status()
        recent = r.json().get("filings", {}).get("recent", {})
        if not recent:
            return pd.DataFrame()
        df = pd.DataFrame({
            "Form": recent.get("form", []),
            "Date": recent.get("filingDate", []),
            "Description": recent.get("primaryDocDescription", []),
            "Document": recent.get("primaryDocument", []),
            "AccessionNo": recent.get("accessionNumber", []),
        })
        if form_types:
            df = df[df["Form"].isin(form_types)]
        df["URL"] = df.apply(
            lambda row: (
                f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                f"{row['AccessionNo'].replace('-', '')}/{row['Document']}"
            ), axis=1)
        return df.head(limit).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────
# STOOQ — historical OHLCV fallback when Yahoo Finance is rate-limited
# ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def fetch_stooq_ohlcv(ticker: str) -> pd.DataFrame:
    """
    Fallback OHLCV source — Stooq's CSV export needs no key/login.
    Ticker format: US stocks plain (aapl.us), NSE not well supported by
    Stooq so this is primarily a US-market fallback for yfinance outages.
    """
    t = ticker.lower().strip()
    if t.endswith(".ns") or t.endswith(".bo"):
        stooq_sym = t  # Stooq has very limited NSE coverage; best-effort
    elif "." not in t:
        stooq_sym = f"{t}.us"
    else:
        stooq_sym = t
    try:
        url = f"https://stooq.com/q/d/l/?s={stooq_sym}&i=d"
        r = requests.get(url, timeout=_TIMEOUT)
        r.raise_for_status()
        if "Date,Open" not in r.text[:20] and "<!DOCTYPE" in r.text[:20]:
            return pd.DataFrame()
        from io import StringIO
        df = pd.read_csv(StringIO(r.text))
        if df.empty or "Date" not in df.columns:
            return pd.DataFrame()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]]
        return df.dropna()
    except Exception:
        return pd.DataFrame()
