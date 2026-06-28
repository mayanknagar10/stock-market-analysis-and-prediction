"""Overview Dashboard — price chart, KPIs, signals, fundamentals, news."""
import streamlit as st
import pandas as np_pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.data_fetcher  import (fetch_ohlcv, fetch_fundamentals, fetch_news,
                                 validate_ticker, PERIOD_MAP, detect_market,
                                 currency_symbol)
from core.indicators    import generate_signals, ema, sma, bollinger_bands
from utils.helpers      import (inject_css, kpi_card, kpi_row, signal_badge,
                                 section_header, signals_table, esc,
                                 fmt_price, fmt_pct, fmt_large, fmt_pct_plain,
                                 top_bar, footer_bar, sidebar_brand)
from utils.charts import candlestick_chart
import pandas as pd
import plotly.graph_objects as go

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    ticker = st.text_input(
        "Ticker Symbol", value="TCS.NS",
        placeholder="AAPL · RELIANCE.NS · TCS.NS",
        help="US: AAPL  |  NSE: RELIANCE.NS  |  BSE: RELIANCE.BO  |  Index: ^NSEI"
    ).upper().strip()
    period_label = st.selectbox("Time Period", list(PERIOD_MAP.keys()), index=3)
    period, interval = PERIOD_MAP[period_label]

    st.divider()
    with st.expander("📖 Ticker formats"):
        st.markdown("""| Market | Format | Example |
|---|---|---|
| US | Plain | `AAPL` |
| NSE | +`.NS` | `TCS.NS` |
| BSE | +`.BO` | `RELIANCE.BO` |
| Nifty 50 | `^NSEI` | Index |
| S&P 500 | `^GSPC` | Index |
| Crypto | +`-USD` | `BTC-USD` |""")

# ── Load ───────────────────────────────────────────────────────────────────────
if not ticker:
    st.info("Enter a ticker symbol in the sidebar.")
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
    st.error("No data returned. Try a different ticker or time period.")
    st.stop()

# ── Header values ─────────────────────────────────────────────────────────────
last  = float(df["Close"].iloc[-1])
prev  = float(df["Close"].iloc[-2]) if len(df) > 1 else last
chg   = last - prev
chg_p = (chg / prev * 100) if prev else 0.0
curr  = info.get("currency", "USD")
curr_s = currency_symbol(curr)
mkt   = detect_market(ticker)
name  = info.get("name", ticker)
logo  = info.get("logo_url", "")

# ── Top bar ───────────────────────────────────────────────────────────────────
top_bar(ticker, name, last, chg, chg_p, curr_s, mkt, logo)

# ── KPIs ──────────────────────────────────────────────────────────────────────
section_header("Key Metrics")
rets    = df["Close"].pct_change().dropna()
ytd_df  = df.loc[df.index.year == df.index[-1].year, "Close"]
ytd_ret = (last / float(ytd_df.iloc[0]) - 1) if len(ytd_df) > 0 else 0.0
ann_vol = float(rets.std() * np.sqrt(252) * 100)

def _v(val, fmt="num"):
    if val is None:
        return "—"
    if fmt == "pe":
        try: return f"{float(val):.1f}x"
        except: return "—"
    if fmt == "pct":
        try: return f"{float(val)*100:.2f}%"
        except: return "—"
    if fmt == "large":
        return fmt_large(val)
    if fmt == "price":
        try: return fmt_price(float(val), currency=curr_s)
        except: return "—"
    return str(val)

kpi_row([
    kpi_card("Market Cap",    _v(info.get("market_cap"), "large"),    curr),
    kpi_card("P/E (TTM)",     _v(info.get("pe_ttm"),     "pe")),
    kpi_card("P/E (Fwd)",     _v(info.get("pe_fwd"),     "pe")),
    kpi_card("EPS (TTM)",     _v(info.get("eps"),         "price")),
    kpi_card("Beta",
             f'{float(info["beta"]):.2f}' if info.get("beta") else "—"),
    kpi_card("Div. Yield",    _v(info.get("dividend_yield"), "pct")),
    kpi_card("YTD Return",    fmt_pct(ytd_ret), "",
             "pos" if ytd_ret >= 0 else "neg"),
    kpi_card("52W High",      _v(info.get("week52_high"), "price")),
    kpi_card("52W Low",       _v(info.get("week52_low"),  "price")),
    kpi_card("Avg Vol (10D)", _v(info.get("avg_volume_10d"), "large")),
    kpi_card("Avg Vol (3M)",  _v(info.get("avg_volume_3m"),  "large")),
    kpi_card("Ann. Vol",      f"{ann_vol:.1f}%"),
])

# ── Price chart ────────────────────────────────────────────────────────────────
section_header("Price Chart")
col_c, col_e = st.columns([6, 1])
with col_e:
    with st.expander("Overlays"):
        show_s20  = st.checkbox("SMA 20",          value=True)
        show_s50  = st.checkbox("SMA 50",          value=True)
        show_s200 = st.checkbox("SMA 200",         value=False)
        show_e20  = st.checkbox("EMA 20",          value=False)
        show_bb   = st.checkbox("Bollinger Bands", value=False)

c = df["Close"]
overlays = {}
if show_s20:  overlays["SMA 20"]    = sma(c, 20)
if show_s50:  overlays["SMA 50"]    = sma(c, 50)
if show_s200: overlays["SMA 200"]   = sma(c, 200)
if show_e20:  overlays["EMA 20"]    = ema(c, 20)
if show_bb:
    bb = bollinger_bands(c)
    overlays["BB Upper"] = bb["BB_Upper"]
    overlays["BB Mid"]   = bb["BB_Mid"]
    overlays["BB Lower"] = bb["BB_Lower"]

with col_c:
    st.plotly_chart(
        candlestick_chart(df, ticker, overlays=overlays, volume=True),
        use_container_width=True, config={"displayModeBar": False})

# ── Signals + Fundamentals ─────────────────────────────────────────────────────
col_sig, col_fun = st.columns(2)

with col_sig:
    section_header("Signal Summary")
    sig  = generate_signals(df)
    comp = sig["composite"]
    st.markdown(
        f'<p style="margin-bottom:12px">'
        f'Composite: {signal_badge(comp)}&nbsp;&nbsp;'
        f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;'
        f'color:#8B949E">'
        f'{sig["buy_count"]} BUY · {sig["sell_count"]} SELL of 8 indicators'
        f'</span></p>',
        unsafe_allow_html=True)
    signals_table(sig["indicators"])

with col_fun:
    section_header("Fundamentals")
    td_k = ("padding:6px 12px;border-bottom:1px solid #30363D;"
            "font-family:'IBM Plex Mono',monospace;font-size:10px;"
            "color:#8B949E;text-transform:uppercase;letter-spacing:.06em;width:40%")
    td_v = ("padding:6px 12px;border-bottom:1px solid #30363D;"
            "font-family:'IBM Plex Mono',monospace;font-size:12px;color:#C9D1D9")

    def _frow(label, val):
        return (f'<tr><td style="{td_k}">{label}</td>'
                f'<td style="{td_v}">{esc(str(val))}</td></tr>')

    rows = (
        _frow("Sector",       info.get("sector",   "—") or "—") +
        _frow("Industry",     info.get("industry", "—") or "—") +
        _frow("Exchange",     info.get("exchange", "—") or "—") +
        _frow("Revenue TTM",  fmt_large(info.get("revenue_ttm"))) +
        _frow("Gross Margin", fmt_pct_plain(info.get("gross_margin",  0) or 0)) +
        _frow("Oper. Margin", fmt_pct_plain(info.get("operating_margin", 0) or 0)) +
        _frow("ROE",          fmt_pct_plain(info.get("roe", 0) or 0)) +
        _frow("Debt/Equity",  f'{float(info["debt_equity"]):.1f}'
                              if info.get("debt_equity") else "—")
    )
    st.markdown(
        f'<table style="width:100%;border-collapse:collapse">'
        f'{rows}</table>',
        unsafe_allow_html=True)

# ── Company description ────────────────────────────────────────────────────────
desc = info.get("description", "")
if desc:
    section_header("About")
    with st.expander(f"About {esc(name)}", expanded=False):
        st.write(desc[:800] + ("…" if len(desc) > 800 else ""))
        if info.get("website"):
            st.markdown(f'[🌐 Visit website]({info["website"]})')

# ── News ───────────────────────────────────────────────────────────────────────
if news:
    section_header("Recent News")
    from datetime import datetime
    for item in news[:6]:
        title_ = item.get("title", "")
        pub    = item.get("publisher", "")
        ts     = item.get("providerPublishTime", 0)
        link   = item.get("link", "#")
        if not title_:
            continue
        try:    dt_s = datetime.utcfromtimestamp(ts).strftime("%d %b %Y")
        except: dt_s = ""
        st.markdown(
            f'<div class="news-card">'
            f'<a href="{link}" target="_blank" style="text-decoration:none">'
            f'<div class="news-title">{esc(title_)}</div></a>'
            f'<div class="news-meta">{esc(pub)} · {dt_s}</div>'
            f'</div>',
            unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
footer_bar()
