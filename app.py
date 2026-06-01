"""
StockPro Analytics — Professional Stock Market Analysis Platform
Entry point: Company Overview Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="StockPro Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Professional Stock Market Analysis Platform"},
)

from core.data_fetcher import (
    fetch_ohlcv, fetch_fundamentals, fetch_news,
    validate_ticker, PERIOD_MAP, detect_market, currency_symbol
)
from core.indicators import generate_signals, ema, sma, bollinger_bands
from utils.helpers import (
    inject_css, kpi_card, kpi_row, signal_badge, section_header,
    ticker_bar, signals_table, news_item,
    fmt_price, fmt_pct, fmt_large, fmt_ratio, fmt_pct_plain
)
from utils.charts import candlestick_chart

inject_css()


# ─── SIDEBAR ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:16px;
                font-weight:600;color:#3FB950;padding:8px 0 16px;">
    📈 StockPro
    <span style="font-size:10px;color:#8B949E;font-weight:400;
                 display:block;letter-spacing:.1em;margin-top:2px;">
    ANALYTICS TERMINAL</span></div>""", unsafe_allow_html=True)

    ticker = st.text_input(
        "Ticker Symbol", value="AAPL",
        placeholder="AAPL · RELIANCE.NS · TCS.NS",
        help=(
            "US stocks: AAPL, MSFT, TSLA\n"
            "NSE India: RELIANCE.NS  TCS.NS  INFY.NS\n"
            "BSE India: RELIANCE.BO  HDFCBANK.BO\n"
            "Indices:   ^NSEI  ^GSPC  ^BSESN"
        )
    ).upper().strip()

    period_label = st.selectbox(
        "Time Period", list(PERIOD_MAP.keys()), index=3
    )
    period, interval = PERIOD_MAP[period_label]

    st.divider()

    # ── Quick market status ──────────────────────────────────────────────
    st.markdown("""<div style="font-family:'IBM Plex Mono',monospace;font-size:10px;
    color:#8B949E;text-transform:uppercase;letter-spacing:.08em;
    margin-bottom:8px;">Market Indices</div>""", unsafe_allow_html=True)

    _indices = [
        ("Nifty 50",  "^NSEI"),
        ("Sensex",    "^BSESN"),
        ("S&P 500",   "^GSPC"),
        ("Nasdaq",    "^IXIC"),
    ]
    for _name, _sym in _indices:
        try:
            _idf = fetch_ohlcv(_sym, "5d", "1d")
            if not _idf.empty and len(_idf) >= 2:
                _last  = float(_idf["Close"].iloc[-1])
                _prev  = float(_idf["Close"].iloc[-2])
                _chg   = (_last - _prev) / _prev * 100
                _arrow = "▲" if _chg >= 0 else "▼"
                _col   = "#3FB950" if _chg >= 0 else "#F85149"
                _fmt   = f"{_last:,.0f}" if _last > 1000 else f"{_last:,.2f}"
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'padding:4px 0;border-bottom:1px solid #21262D;">'
                    f'<span style="font-family:\'IBM Plex Mono\',monospace;'
                    f'font-size:10px;color:#8B949E">{_name}</span>'
                    f'<span style="font-family:\'IBM Plex Mono\',monospace;'
                    f'font-size:10px;color:{_col}">{_arrow} {_chg:+.2f}%</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        except Exception:
            pass

    st.divider()

    # ── Ticker format help ───────────────────────────────────────────────
    with st.expander("📖 Ticker formats"):
        st.markdown("""
| Market | Format | Example |
|--------|--------|---------|
| US | Plain | `AAPL` |
| NSE | +`.NS` | `TCS.NS` |
| BSE | +`.BO` | `RELIANCE.BO` |
| Nifty | Index | `^NSEI` |
| Sensex | Index | `^BSESN` |
| S&P 500 | Index | `^GSPC` |
| Crypto | +`-USD` | `BTC-USD` |
        """)

    st.divider()
    st.caption("Data via Yahoo Finance · Not financial advice")


# ─── VALIDATION ───────────────────────────────────────────────────────────
if not ticker:
    st.info("Enter a ticker symbol in the sidebar to get started.")
    st.stop()

with st.spinner(f"Loading {ticker}…"):
    valid, err = validate_ticker(ticker)
    if not valid:
        st.error(f"**{ticker}** — {err}")
        st.stop()
    df   = fetch_ohlcv(ticker, period, interval)
    info = fetch_fundamentals(ticker)
    news = fetch_news(ticker)

if df.empty:
    st.error("No data returned. Try a different ticker or time range.")
    st.stop()


# ─── HEADER ───────────────────────────────────────────────────────────────
last   = df["Close"].iloc[-1]
prev   = df["Close"].iloc[-2] if len(df) > 1 else last
chg    = last - prev
chg_p  = (chg / prev) * 100 if prev else 0
curr   = info.get("currency", "USD")
curr_s = currency_symbol(curr)

ticker_bar(ticker, info.get("name", ticker), last, chg, chg_p, curr_s)


# ─── KPI ROW ──────────────────────────────────────────────────────────────
section_header("Key Metrics")
returns = df["Close"].pct_change().dropna()
ytd_start = df.loc[df.index.year == df.index[-1].year, "Close"]
ytd_ret   = (last / ytd_start.iloc[0] - 1) if len(ytd_start) > 0 else 0

kpi_row([
    kpi_card("Market Cap",     fmt_large(info.get("market_cap")), curr),
    kpi_card("P/E (TTM)",      f'{info["pe_ttm"]:.1f}' if info.get("pe_ttm") else "—"),
    kpi_card("P/E (Forward)",  f'{info["pe_fwd"]:.1f}' if info.get("pe_fwd") else "—"),
    kpi_card("EPS (TTM)",      fmt_price(info.get("eps"), currency=curr_s)),
    kpi_card("Beta",           f'{info["beta"]:.2f}' if info.get("beta") else "—"),
    kpi_card("Div. Yield",     fmt_pct_plain(info.get("dividend_yield", 0))),
    kpi_card("YTD Return",     fmt_pct(ytd_ret),
             colour="pos" if ytd_ret >= 0 else "neg"),
    kpi_card("52W High",       fmt_price(info.get("week52_high"), currency=curr_s)),
    kpi_card("52W Low",        fmt_price(info.get("week52_low"),  currency=curr_s)),
    kpi_card("Avg Vol (10D)",  fmt_large(info.get("avg_volume_10d"))),
    kpi_card("Ann. Volatility",f'{returns.std() * np.sqrt(252) * 100:.1f}%'),
    kpi_card("Employees",      fmt_large(info.get("employees")) if info.get("employees") else "—"),
])


# ─── PRICE CHART ──────────────────────────────────────────────────────────
section_header("Price Chart")

# Overlay controls
col_c, col_e = st.columns([6, 1])
with col_e:
    with st.expander("Overlays", expanded=False):
        show_sma20  = st.checkbox("SMA 20",  value=True)
        show_sma50  = st.checkbox("SMA 50",  value=True)
        show_sma200 = st.checkbox("SMA 200", value=False)
        show_ema20  = st.checkbox("EMA 20",  value=False)
        show_bb     = st.checkbox("Bollinger Bands", value=False)

overlays = {}
c = df["Close"]
if show_sma20:  overlays["SMA 20"]     = sma(c, 20)
if show_sma50:  overlays["SMA 50"]     = sma(c, 50)
if show_sma200: overlays["SMA 200"]    = sma(c, 200)
if show_ema20:  overlays["EMA 20"]     = ema(c, 20)
if show_bb:
    bb = bollinger_bands(c)
    overlays["BB Upper"] = bb["BB_Upper"]
    overlays["BB Mid"]   = bb["BB_Mid"]
    overlays["BB Lower"] = bb["BB_Lower"]

with col_c:
    fig = candlestick_chart(df, ticker, overlays=overlays, volume=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ─── SIGNALS + FUNDAMENTALS ──────────────────────────────────────────────
col_sig, col_fun = st.columns([1, 1])

with col_sig:
    section_header("Signal Summary")
    sig_data = generate_signals(df)
    composite = sig_data["composite"]
    buy_n  = sig_data["buy_count"]
    sell_n = sig_data["sell_count"]

    st.markdown(
        f'<p style="margin-bottom:12px;">Composite signal: '
        f'{signal_badge(composite)}&nbsp;&nbsp;'
        f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;'
        f'color:#8B949E;">{buy_n} BUY · {sell_n} SELL of 8 indicators</span></p>',
        unsafe_allow_html=True,
    )
    signals_table(sig_data["indicators"])

with col_fun:
    section_header("Fundamentals")
    sector   = info.get("sector",   "—")
    industry = info.get("industry", "—")
    exchange = info.get("exchange", "—")

    st.markdown(f"""
    <table style="width:100%;border-collapse:collapse;">
    {"".join(f'''
      <tr>
        <td style="padding:6px 12px;border-bottom:1px solid #30363D;
                   font-family:'IBM Plex Mono',monospace;font-size:10px;
                   color:#8B949E;text-transform:uppercase;letter-spacing:.06em;
                   width:38%">{k}</td>
        <td style="padding:6px 12px;border-bottom:1px solid #30363D;
                   font-family:'IBM Plex Mono',monospace;font-size:12px;
                   color:#C9D1D9;">{v}</td>
      </tr>''' for k, v in [
        ("Sector",           sector),
        ("Industry",         industry),
        ("Exchange",         exchange),
        ("Revenue (TTM)",    fmt_large(info.get("revenue_ttm"))),
        ("Gross Margin",     fmt_pct_plain(info.get("gross_margin",0))),
        ("Oper. Margin",     fmt_pct_plain(info.get("operating_margin",0))),
        ("ROE",              fmt_pct_plain(info.get("roe",0))),
        ("Debt / Equity",    f'{info["debt_equity"]:.1f}' if info.get("debt_equity") else "—"),
    ])}
    </table>""", unsafe_allow_html=True)


# ─── COMPANY DESCRIPTION ──────────────────────────────────────────────────
desc = info.get("description", "")
if desc:
    section_header("About")
    with st.expander("Company Description", expanded=False):
        st.markdown(
            f'<p style="font-size:13px;color:#C9D1D9;line-height:1.65;">'
            f'{desc[:800]}{"…" if len(desc) > 800 else ""}</p>',
            unsafe_allow_html=True,
        )
        if info.get("website"):
            st.markdown(
                f'[🌐 Visit website]({info["website"]})',
                unsafe_allow_html=False,
            )


# ─── NEWS ─────────────────────────────────────────────────────────────────
if news:
    section_header("Recent News")
    for item in news[:6]:
        title     = item.get("title", "")
        publisher = item.get("publisher", "")
        ts        = item.get("providerPublishTime", 0)
        link      = item.get("link", "#")
        if title:
            st.markdown(f"""
            <div class="news-card">
              <a href="{link}" target="_blank" style="text-decoration:none;">
                <div class="news-title">{title}</div>
              </a>
              <div class="news-meta">{publisher} · {
                  __import__('datetime').datetime.utcfromtimestamp(ts).strftime('%d %b %Y')
                  if ts else ''
              }</div>
            </div>""", unsafe_allow_html=True)
