"""StockPro Analytics — Professional Overview Dashboard."""
import streamlit as st, pandas as pd, numpy as np
import sys; sys.path.insert(0,".")

st.set_page_config(page_title="StockPro Analytics",page_icon="📈",
                   layout="wide",initial_sidebar_state="expanded",
                   menu_items={"About":"Professional Stock Market Analysis Platform"})

from core.data_fetcher  import (fetch_ohlcv,fetch_fundamentals,fetch_news,
                                 validate_ticker,PERIOD_MAP,detect_market,currency_symbol)
from core.indicators    import generate_signals,ema,sma,bollinger_bands
from utils.helpers      import (inject_css,kpi_card,kpi_row,signal_badge,section_header,
                                 ticker_bar,signals_table,esc,
                                 fmt_price,fmt_pct,fmt_large,fmt_pct_plain)
from utils.charts import candlestick_chart
inject_css()

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:16px;'
                'font-weight:600;color:#3FB950;padding:8px 0 16px;">📈 StockPro'
                '<span style="font-size:10px;color:#8B949E;font-weight:400;display:block;'
                'letter-spacing:.1em;margin-top:2px;">ANALYTICS TERMINAL</span></div>',
                unsafe_allow_html=True)

    ticker = st.text_input("Ticker Symbol", value="AAPL",
                           placeholder="AAPL · RELIANCE.NS · TCS.NS",
                           help="US: AAPL  |  NSE: RELIANCE.NS  |  BSE: RELIANCE.BO  |  Index: ^NSEI"
                          ).upper().strip()
    period_label = st.selectbox("Time Period", list(PERIOD_MAP.keys()), index=3)
    period, interval = PERIOD_MAP[period_label]

    st.divider()
    # Live index strip
    st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;'
                'color:#8B949E;text-transform:uppercase;letter-spacing:.08em;'
                'margin-bottom:8px;">Market Indices</div>', unsafe_allow_html=True)
    for idx_name, idx_sym in [("Nifty 50","^NSEI"),("Sensex","^BSESN"),
                                ("S&P 500","^GSPC"),("Nasdaq","^IXIC")]:
        try:
            idf = fetch_ohlcv(idx_sym,"5d","1d")
            if not idf.empty and len(idf)>=2:
                last_=float(idf["Close"].iloc[-1]); prev_=float(idf["Close"].iloc[-2])
                chg_=(last_-prev_)/prev_*100; arrow="▲" if chg_>=0 else "▼"
                col_="#3FB950" if chg_>=0 else "#F85149"
                st.markdown(f'<div style="display:flex;justify-content:space-between;'
                            f'padding:4px 0;border-bottom:1px solid #21262D">'
                            f'<span style="font-family:\'IBM Plex Mono\',monospace;'
                            f'font-size:10px;color:#8B949E">{idx_name}</span>'
                            f'<span style="font-family:\'IBM Plex Mono\',monospace;'
                            f'font-size:10px;color:{col_}">{arrow} {chg_:+.2f}%</span>'
                            f'</div>', unsafe_allow_html=True)
        except: pass

    st.divider()
    with st.expander("📖 Ticker formats"):
        st.markdown("""| Market | Format | Example |
|---|---|---|
| US | Plain | `AAPL` |
| NSE | +`.NS` | `TCS.NS` |
| BSE | +`.BO` | `RELIANCE.BO` |
| Nifty | Index | `^NSEI` |
| S&P | Index | `^GSPC` |
| Crypto | +`-USD` | `BTC-USD` |""")
    st.divider()
    st.caption("Data via Yahoo Finance · Not financial advice")

# ── Load ───────────────────────────────────────────────────────────────────
if not ticker:
    st.info("Enter a ticker symbol in the sidebar to get started.")
    st.stop()

with st.spinner(f"Loading {ticker}…"):
    valid, err = validate_ticker(ticker)
    if not valid: st.error(f"**{ticker}** — {err}"); st.stop()
    df   = fetch_ohlcv(ticker, period, interval)
    info = fetch_fundamentals(ticker)
    news = fetch_news(ticker)

if df.empty: st.error("No data returned."); st.stop()

# ── Header ─────────────────────────────────────────────────────────────────
last  = float(df["Close"].iloc[-1]); prev = float(df["Close"].iloc[-2]) if len(df)>1 else last
chg   = last-prev; chg_p = (chg/prev*100) if prev else 0
curr  = info.get("currency","USD"); curr_s = currency_symbol(curr)
mkt   = detect_market(ticker)
flag  = "🇮🇳" if mkt in ("NSE","BSE") else "🇺🇸"

ticker_bar(ticker, f"{flag} {info.get('name',ticker)}", last, chg, chg_p, curr_s)

# ── KPIs ───────────────────────────────────────────────────────────────────
section_header("Key Metrics")
rets = df["Close"].pct_change().dropna()
ytd  = df.loc[df.index.year==df.index[-1].year,"Close"]
ytd_ret = (last/float(ytd.iloc[0])-1) if len(ytd)>0 else 0
kpi_row([
    kpi_card("Market Cap",    fmt_large(info.get("market_cap")),            curr),
    kpi_card("P/E (TTM)",     f'{info["pe_ttm"]:.1f}' if info.get("pe_ttm") else "—"),
    kpi_card("P/E (Fwd)",     f'{info["pe_fwd"]:.1f}' if info.get("pe_fwd") else "—"),
    kpi_card("EPS (TTM)",     fmt_price(info.get("eps"),currency=curr_s)),
    kpi_card("Beta",          f'{info["beta"]:.2f}' if info.get("beta") else "—"),
    kpi_card("Div. Yield",    fmt_pct_plain(info.get("dividend_yield",0))),
    kpi_card("YTD Return",    fmt_pct(ytd_ret),"", "pos" if ytd_ret>=0 else "neg"),
    kpi_card("52W High",      fmt_price(info.get("week52_high"),currency=curr_s)),
    kpi_card("52W Low",       fmt_price(info.get("week52_low"), currency=curr_s)),
    kpi_card("Avg Vol (10D)", fmt_large(info.get("avg_volume_10d"))),
    kpi_card("Ann. Vol",      f'{rets.std()*np.sqrt(252)*100:.1f}%'),
    kpi_card("Employees",     fmt_large(info.get("employees")) if info.get("employees") else "—"),
])

# ── Price chart ────────────────────────────────────────────────────────────
section_header("Price Chart")
col_c, col_e = st.columns([6,1])
with col_e:
    with st.expander("Overlays"):
        show_s20=st.checkbox("SMA 20",  value=True)
        show_s50=st.checkbox("SMA 50",  value=True)
        show_s200=st.checkbox("SMA 200",value=False)
        show_e20=st.checkbox("EMA 20",  value=False)
        show_bb =st.checkbox("Bollinger Bands",value=False)

c=df["Close"]; overlays={}
if show_s20:  overlays["SMA 20"]=sma(c,20)
if show_s50:  overlays["SMA 50"]=sma(c,50)
if show_s200: overlays["SMA 200"]=sma(c,200)
if show_e20:  overlays["EMA 20"]=ema(c,20)
if show_bb:
    bb=bollinger_bands(c)
    overlays["BB Upper"]=bb["BB_Upper"]; overlays["BB Mid"]=bb["BB_Mid"]; overlays["BB Lower"]=bb["BB_Lower"]
with col_c:
    st.plotly_chart(candlestick_chart(df,ticker,overlays=overlays,volume=True),
                    use_container_width=True,config={"displayModeBar":False})

# ── Signals + Fundamentals ─────────────────────────────────────────────────
col_sig, col_fun = st.columns([1,1])
with col_sig:
    section_header("Signal Summary")
    sig = generate_signals(df)
    comp = sig["composite"]
    st.markdown(
        f'<p style="margin-bottom:12px">Composite: {signal_badge(comp)}&nbsp;&nbsp;'
        f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;color:#8B949E">'
        f'{sig["buy_count"]} BUY · {sig["sell_count"]} SELL of 8 indicators</span></p>',
        unsafe_allow_html=True)
    signals_table(sig["indicators"])

with col_fun:
    section_header("Fundamentals")
    td_k=("padding:6px 12px;border-bottom:1px solid #30363D;"
          "font-family:'IBM Plex Mono',monospace;font-size:10px;"
          "color:#8B949E;text-transform:uppercase;letter-spacing:.06em;width:40%")
    td_v=("padding:6px 12px;border-bottom:1px solid #30363D;"
          "font-family:'IBM Plex Mono',monospace;font-size:12px;color:#C9D1D9")
    rows=[
        ("Sector",       esc(info.get("sector","—"))),
        ("Industry",     esc(info.get("industry","—"))),
        ("Exchange",     esc(info.get("exchange","—"))),
        ("Revenue TTM",  esc(fmt_large(info.get("revenue_ttm")))),
        ("Gross Margin", esc(fmt_pct_plain(info.get("gross_margin",0)))),
        ("Oper. Margin", esc(fmt_pct_plain(info.get("operating_margin",0)))),
        ("ROE",          esc(fmt_pct_plain(info.get("roe",0)))),
        ("Debt/Equity",  esc(f'{info["debt_equity"]:.1f}' if info.get("debt_equity") else "—")),
    ]
    st.markdown(
        f'<table style="width:100%;border-collapse:collapse">'
        + "".join(f'<tr><td style="{td_k}">{k}</td><td style="{td_v}">{v}</td></tr>'
                  for k,v in rows)
        + "</table>", unsafe_allow_html=True)

# ── Description ────────────────────────────────────────────────────────────
desc=info.get("description","")
if desc:
    section_header("About")
    with st.expander("Company Description", expanded=False):
        st.write(desc[:800]+("…" if len(desc)>800 else ""))
        if info.get("website"): st.markdown(f'[🌐 {esc(info["website"])}]({info["website"]})')

# ── News ───────────────────────────────────────────────────────────────────
if news:
    section_header("Recent News")
    from datetime import datetime
    for item in news[:6]:
        title_=item.get("title",""); publisher=item.get("publisher","")
        ts=item.get("providerPublishTime",0); link=item.get("link","#")
        if not title_: continue
        try: dt_s=datetime.utcfromtimestamp(ts).strftime("%d %b %Y")
        except: dt_s=""
        st.markdown(
            f'<div class="news-card">'
            f'<a href="{link}" target="_blank" style="text-decoration:none">'
            f'<div class="news-title">{esc(title_)}</div></a>'
            f'<div class="news-meta">{esc(publisher)} · {dt_s}</div>'
            f'</div>', unsafe_allow_html=True)
