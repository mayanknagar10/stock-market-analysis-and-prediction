"""
Page 6 — Compare Stocks
Side-by-side technical, fundamental, and risk comparison of two tickers.
Pure Streamlit — no external AI APIs.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

st.set_page_config(
    page_title="Compare · StockPro", page_icon="⚖️",
    layout="wide", initial_sidebar_state="expanded"
)

from core.data_fetcher  import fetch_ohlcv, fetch_fundamentals, validate_ticker, PERIOD_MAP, detect_market, currency_symbol
from core.indicators    import (rsi, macd, ema, sma, bollinger_bands, atr,
                                historical_volatility, generate_signals)
from core.risk_metrics  import (full_risk_report, compute_returns,
                                annualised_return, annualised_volatility,
                                sharpe_ratio, sortino_ratio, drawdown_analysis,
                                var_historical)
from utils.helpers      import (inject_css, section_header, kpi_row, kpi_card,
                                fmt_price, fmt_pct, fmt_pct_plain, fmt_large, fmt_ratio)
import plotly.graph_objects as go
from plotly.subplots import make_subplots

inject_css()

LAYOUT = dict(
    plot_bgcolor="#0D1117", paper_bgcolor="#0D1117",
    font=dict(family="IBM Plex Mono, monospace", color="#C9D1D9", size=11),
    margin=dict(l=12, r=12, t=36, b=12),
    xaxis=dict(gridcolor="#21262D", zeroline=False),
    yaxis=dict(gridcolor="#21262D", zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)
C1 = "#3FB950"
C2 = "#58A6FF"

# ─── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style="font-family:'IBM Plex Mono',monospace;font-size:16px;
    font-weight:600;color:#3FB950;padding:8px 0 16px;">📈 StockPro
    <span style="font-size:10px;color:#8B949E;font-weight:400;display:block;
    letter-spacing:.1em;margin-top:2px;">ANALYTICS TERMINAL</span></div>""",
    unsafe_allow_html=True)

    st.markdown("**Stocks to Compare**")
    ticker1 = st.text_input("Stock A", value="RELIANCE.NS",
                            help="US: AAPL  |  NSE: RELIANCE.NS  |  BSE: RELIANCE.BO"
                            ).upper().strip()
    ticker2 = st.text_input("Stock B", value="TCS.NS",
                            help="US: MSFT  |  NSE: TCS.NS  |  BSE: TCS.BO"
                            ).upper().strip()

    period_label = st.selectbox("Time Period", list(PERIOD_MAP.keys()), index=3)
    period, interval = PERIOD_MAP[period_label]

    st.divider()
    st.markdown("**Display Options**")
    show_normalised = st.checkbox("Normalise prices (indexed to 100)", value=True)
    show_bb         = st.checkbox("Show Bollinger Bands",  value=False)
    show_ema        = st.checkbox("Show EMA 20/50",         value=True)

    st.divider()
    st.caption("Data via Yahoo Finance · Not financial advice")

# ─── HEADER ─────────────────────────────────────────────────────────────────
# Detect markets early for header (flags not yet set, use detect_market directly)
_h_flag1 = "🇮🇳" if detect_market(ticker1 or "AAPL") in ("NSE","BSE") else "🇺🇸"
_h_flag2 = "🇮🇳" if detect_market(ticker2 or "MSFT") in ("NSE","BSE") else "🇺🇸"
st.markdown(f"""<div style="font-family:'IBM Plex Mono',monospace;padding:10px 0 6px;
border-bottom:1px solid #30363D;margin-bottom:16px;">
<span style="font-size:20px;font-weight:600;color:{C1}">{_h_flag1} {ticker1 or 'A'}</span>
&nbsp;<span style="font-size:16px;color:#8B949E">vs</span>&nbsp;
<span style="font-size:20px;font-weight:600;color:{C2}">{_h_flag2} {ticker2 or 'B'}</span>&nbsp;&nbsp;
<span style="font-size:13px;color:#8B949E">Side-by-Side Comparison</span>
</div>""", unsafe_allow_html=True)

if not ticker1 or not ticker2:
    st.info("Enter two ticker symbols in the sidebar.")
    st.stop()
if ticker1 == ticker2:
    st.warning("Enter two different tickers.")
    st.stop()

# ─── LOAD DATA ───────────────────────────────────────────────────────────────
with st.spinner(f"Loading {ticker1} & {ticker2}…"):
    df1   = fetch_ohlcv(ticker1, period, interval)
    df2   = fetch_ohlcv(ticker2, period, interval)
    info1 = fetch_fundamentals(ticker1)
    info2 = fetch_fundamentals(ticker2)

for t, df in [(ticker1, df1), (ticker2, df2)]:
    if df.empty:
        st.error(f"No data for **{t}**. Check the ticker and try again.")
        st.stop()

# Align on common dates
shared = df1.index.intersection(df2.index)
df1 = df1.loc[shared]
df2 = df2.loc[shared]

ret1 = compute_returns(df1["Close"])
ret2 = compute_returns(df2["Close"])

# Currency symbols per ticker
_sym1 = currency_symbol("INR" if detect_market(ticker1) in ("NSE","BSE") else "USD")
_sym2 = currency_symbol("INR" if detect_market(ticker2) in ("NSE","BSE") else "USD")
_flag1 = "🇮🇳" if detect_market(ticker1) in ("NSE","BSE") else "🇺🇸"
_flag2 = "🇮🇳" if detect_market(ticker2) in ("NSE","BSE") else "🇺🇸"

# ─── PRICE CHART ─────────────────────────────────────────────────────────────
section_header("Price Performance")

fig_price = go.Figure()

if show_normalised:
    y1 = df1["Close"] / df1["Close"].iloc[0] * 100
    y2 = df2["Close"] / df2["Close"].iloc[0] * 100
    y_label = "Indexed to 100"
else:
    y1 = df1["Close"]
    y2 = df2["Close"]
    y_label = "Price"

fig_price.add_trace(go.Scatter(x=y1.index, y=y1, name=ticker1,
    line=dict(color=C1, width=2)))
fig_price.add_trace(go.Scatter(x=y2.index, y=y2, name=ticker2,
    line=dict(color=C2, width=2)))

if show_ema and not show_normalised:
    for t, df_t, c in [(ticker1, df1, C1), (ticker2, df2, C2)]:
        fig_price.add_trace(go.Scatter(
            x=df_t.index, y=ema(df_t["Close"], 20), name=f"{t} EMA20",
            line=dict(color=c, width=1, dash="dot"), opacity=0.6))
        fig_price.add_trace(go.Scatter(
            x=df_t.index, y=ema(df_t["Close"], 50), name=f"{t} EMA50",
            line=dict(color=c, width=1, dash="dash"), opacity=0.5))

if show_normalised:
    fig_price.add_hline(y=100, line_color="#8B949E", line_dash="dot", line_width=0.8)

fig_price.update_layout(
    **{**LAYOUT, "height": 400,
       "title": dict(text=f"{ticker1} vs {ticker2} — {'Normalised ' if show_normalised else ''}Price", font_size=12)},
    yaxis_title=y_label,
)
st.plotly_chart(fig_price, use_container_width=True,
                config={"displayModeBar": False})

# ─── RETURN DELTA CHART ──────────────────────────────────────────────────────
section_header("Return Spread")

cum1    = (1 + ret1).cumprod() - 1
cum2    = (1 + ret2).cumprod() - 1
spread  = (cum1 - cum2) * 100

fig_spread = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.03, row_heights=[0.55, 0.45])
fig_spread.add_trace(go.Scatter(x=cum1.index, y=cum1 * 100, name=ticker1,
    line=dict(color=C1, width=1.8)), row=1, col=1)
fig_spread.add_trace(go.Scatter(x=cum2.index, y=cum2 * 100, name=ticker2,
    line=dict(color=C2, width=1.8)), row=1, col=1)
fig_spread.add_hline(y=0, line_color="#8B949E", line_dash="dot",
                     line_width=0.8, row=1, col=1)

colours_spread = ["rgba(63,185,80,0.5)" if v >= 0 else "rgba(248,81,73,0.5)"
                  for v in spread]
fig_spread.add_trace(go.Bar(x=spread.index, y=spread,
    name=f"{ticker1} – {ticker2}", marker_color=colours_spread, opacity=0.8,
    showlegend=True), row=2, col=1)
fig_spread.add_hline(y=0, line_color="#8B949E", line_dash="dot",
                     line_width=0.8, row=2, col=1)

fig_spread.update_layout(
    **{**LAYOUT, "height": 440,
       "title": dict(text="Cumulative Return (%) & Daily Spread", font_size=12)},
)
fig_spread.update_yaxes(title_text="Cumulative Return (%)", row=1, col=1,
                        gridcolor="#21262D")
fig_spread.update_yaxes(title_text=f"Spread ({ticker1}–{ticker2}) %",
                        row=2, col=1, gridcolor="#21262D")
st.plotly_chart(fig_spread, use_container_width=True,
                config={"displayModeBar": False})

# ─── SIDE-BY-SIDE INDICATOR PANELS ──────────────────────────────────────────
section_header("Technical Indicators")
tabs = st.tabs(["  RSI  ", "  MACD  ", "  Volatility  ", "  Volume  "])

with tabs[0]:
    rsi1 = rsi(df1["Close"])
    rsi2 = rsi(df2["Close"])
    fig_rsi = go.Figure()
    fig_rsi.add_hrect(y0=70, y1=100, fillcolor="rgba(248,81,73,0.05)", line_width=0)
    fig_rsi.add_hrect(y0=0,  y1=30,  fillcolor="rgba(63,185,80,0.05)",  line_width=0)
    fig_rsi.add_trace(go.Scatter(x=rsi1.index, y=rsi1, name=f"{ticker1} RSI",
        line=dict(color=C1, width=1.8)))
    fig_rsi.add_trace(go.Scatter(x=rsi2.index, y=rsi2, name=f"{ticker2} RSI",
        line=dict(color=C2, width=1.8)))
    fig_rsi.add_hline(y=70, line_color="#F85149", line_dash="dot", line_width=1)
    fig_rsi.add_hline(y=30, line_color="#3FB950", line_dash="dot", line_width=1)
    fig_rsi.add_hline(y=50, line_color="#8B949E", line_dash="dot", line_width=0.6)
    fig_rsi.update_yaxes(range=[0, 100])
    fig_rsi.update_layout(**{**LAYOUT, "height": 300,
        "title": dict(text="RSI (14) Comparison", font_size=12)})
    st.plotly_chart(fig_rsi, use_container_width=True,
                    config={"displayModeBar": False})

with tabs[1]:
    m1 = macd(df1["Close"])
    m2 = macd(df2["Close"])
    fig_macd = make_subplots(rows=1, cols=2, subplot_titles=(
        f"{ticker1} MACD", f"{ticker2} MACD"))
    for col_idx, (m, t, colour) in enumerate([(m1, ticker1, C1), (m2, ticker2, C2)], 1):
        fig_macd.add_trace(go.Scatter(x=m.index, y=m["MACD"], name=f"{t} MACD",
            line=dict(color=colour, width=1.8)), row=1, col=col_idx)
        fig_macd.add_trace(go.Scatter(x=m.index, y=m["Signal"], name=f"{t} Signal",
            line=dict(color="#E3B341", width=1.5)), row=1, col=col_idx)
        hist_c = ["rgba(63,185,80,0.6)" if v >= 0 else "rgba(248,81,73,0.6)"
                  for v in m["Hist"]]
        fig_macd.add_trace(go.Bar(x=m.index, y=m["Hist"], name=f"{t} Hist",
            marker_color=hist_c, opacity=0.7, showlegend=False),
            row=1, col=col_idx)
    fig_macd.update_layout(**{**LAYOUT, "height": 300,
        "title": dict(text="MACD (12,26,9) Comparison", font_size=12)})
    for axis in ["xaxis", "xaxis2", "yaxis", "yaxis2"]:
        fig_macd.update_layout(**{axis: dict(gridcolor="#21262D")})
    st.plotly_chart(fig_macd, use_container_width=True,
                    config={"displayModeBar": False})

with tabs[2]:
    hv1 = historical_volatility(df1["Close"], 20) * 100
    hv2 = historical_volatility(df2["Close"], 20) * 100
    fig_hv = go.Figure()
    fig_hv.add_trace(go.Scatter(x=hv1.index, y=hv1, name=f"{ticker1} HV20",
        line=dict(color=C1, width=1.8),
        fill="tozeroy", fillcolor="rgba(63,185,80,0.05)"))
    fig_hv.add_trace(go.Scatter(x=hv2.index, y=hv2, name=f"{ticker2} HV20",
        line=dict(color=C2, width=1.8),
        fill="tozeroy", fillcolor="rgba(88,166,255,0.05)"))
    fig_hv.update_layout(**{**LAYOUT, "height": 300,
        "title": dict(text="Historical Volatility 20D (Ann. %)", font_size=12)},
        yaxis_title="Volatility (%)")
    st.plotly_chart(fig_hv, use_container_width=True,
                    config={"displayModeBar": False})

with tabs[3]:
    fig_vol = make_subplots(rows=1, cols=2,
                            subplot_titles=(f"{ticker1} Volume", f"{ticker2} Volume"))
    for col_idx, (df_t, t, colour) in enumerate(
            [(df1, ticker1, C1), (df2, ticker2, C2)], 1):
        vc = ["rgba(63,185,80,0.6)" if c >= o else "rgba(248,81,73,0.6)"
              for c, o in zip(df_t["Close"], df_t["Open"])]
        fig_vol.add_trace(go.Bar(x=df_t.index, y=df_t["Volume"],
            marker_color=vc, opacity=0.65, name=t, showlegend=True),
            row=1, col=col_idx)
        fig_vol.add_trace(go.Scatter(x=df_t.index,
            y=df_t["Volume"].rolling(20).mean(),
            name=f"{t} 20D MA", line=dict(color="#E3B341", width=1.5)),
            row=1, col=col_idx)
    fig_vol.update_layout(**{**LAYOUT, "height": 300,
        "title": dict(text="Volume Comparison", font_size=12)})
    for axis in ["xaxis", "xaxis2", "yaxis", "yaxis2"]:
        fig_vol.update_layout(**{axis: dict(gridcolor="#21262D")})
    st.plotly_chart(fig_vol, use_container_width=True,
                    config={"displayModeBar": False})

# ─── RISK METRICS COMPARISON ─────────────────────────────────────────────────
section_header("Risk Metrics — Head to Head")

r_report1 = full_risk_report(df1["Close"])
r_report2 = full_risk_report(df2["Close"])

metrics = [
    ("Ann. Return",     fmt_pct(r_report1["annualised_return"]),
                        fmt_pct(r_report2["annualised_return"]),
                        "annualised_return", True),
    ("Ann. Volatility", f"{r_report1['annualised_volatility']*100:.1f}%",
                        f"{r_report2['annualised_volatility']*100:.1f}%",
                        "annualised_volatility", False),
    ("Sharpe Ratio",    f"{r_report1['sharpe_ratio']:.3f}",
                        f"{r_report2['sharpe_ratio']:.3f}",
                        "sharpe_ratio", True),
    ("Sortino Ratio",   f"{r_report1['sortino_ratio']:.3f}",
                        f"{r_report2['sortino_ratio']:.3f}",
                        "sortino_ratio", True),
    ("Max Drawdown",    f"{r_report1['max_drawdown']*100:.1f}%",
                        f"{r_report2['max_drawdown']*100:.1f}%",
                        "max_drawdown", True),  # higher = better (less negative)
    ("VaR 95% (Daily)", f"{r_report1['var_95_historical']*100:.2f}%",
                        f"{r_report2['var_95_historical']*100:.2f}%",
                        "var_95_historical", False),
    ("CVaR 95%",        f"{r_report1['cvar_95']*100:.2f}%",
                        f"{r_report2['cvar_95']*100:.2f}%",
                        "cvar_95", False),
    ("Skewness",        f"{r_report1['skewness']:.3f}",
                        f"{r_report2['skewness']:.3f}",
                        "skewness", True),
    ("Kurtosis",        f"{r_report1['kurtosis']:.2f}",
                        f"{r_report2['kurtosis']:.2f}",
                        None, None),
    ("DD Duration",     f"{r_report1['max_drawdown_duration']}d",
                        f"{r_report2['max_drawdown_duration']}d",
                        "max_drawdown_duration", False),
]

rows_html = ""
for label, v1, v2, key, higher_better in metrics:
    win_1, win_2 = "#C9D1D9", "#C9D1D9"
    if key and higher_better is not None:
        val1_raw = r_report1.get(key, 0) or 0
        val2_raw = r_report2.get(key, 0) or 0
        if higher_better:
            if val1_raw > val2_raw:  win_1 = C1
            elif val2_raw > val1_raw: win_2 = C2
        else:
            if val1_raw < val2_raw:  win_1 = C1
            elif val2_raw < val1_raw: win_2 = C2
    rows_html += f"""<tr style="border-bottom:1px solid #21262D">
      <td style="padding:8px 14px;font-family:'IBM Plex Mono',monospace;font-size:11px;
                 color:#8B949E;text-transform:uppercase;letter-spacing:.06em;
                 text-align:center">{label}</td>
      <td style="padding:8px 14px;font-family:'IBM Plex Mono',monospace;font-size:14px;
                 font-weight:600;color:{win_1};text-align:center">{v1}</td>
      <td style="padding:8px 14px;font-family:'IBM Plex Mono',monospace;font-size:14px;
                 font-weight:600;color:{win_2};text-align:center">{v2}</td>
    </tr>"""

st.markdown(f"""
<table style="width:100%;border-collapse:collapse;background:#161B22;
              border:1px solid #30363D;border-radius:8px;overflow:hidden">
  <thead><tr style="background:#21262D">
    <th style="padding:10px 14px;font-family:'IBM Plex Mono',monospace;font-size:10px;
               color:#8B949E;text-transform:uppercase;text-align:center">Metric</th>
    <th style="padding:10px 14px;font-family:'IBM Plex Mono',monospace;font-size:13px;
               color:{C1};font-weight:600;text-align:center">{ticker1}</th>
    <th style="padding:10px 14px;font-family:'IBM Plex Mono',monospace;font-size:13px;
               color:{C2};font-weight:600;text-align:center">{ticker2}</th>
  </tr></thead>
  <tbody>{rows_html}</tbody>
</table>""", unsafe_allow_html=True)

# ─── FUNDAMENTALS COMPARISON ─────────────────────────────────────────────────
section_header("Fundamentals — Head to Head")

fund_metrics = [
    ("Market Cap",       fmt_large(info1.get("market_cap")),   fmt_large(info2.get("market_cap"))),
    ("P/E (TTM)",        f'{info1["pe_ttm"]:.1f}' if info1.get("pe_ttm") else "—",
                         f'{info2["pe_ttm"]:.1f}' if info2.get("pe_ttm") else "—"),
    ("P/E (Forward)",    f'{info1["pe_fwd"]:.1f}' if info1.get("pe_fwd") else "—",
                         f'{info2["pe_fwd"]:.1f}' if info2.get("pe_fwd") else "—"),
    ("EPS (TTM)",        fmt_price(info1.get("eps"), currency=_sym1),
                         fmt_price(info2.get("eps"), currency=_sym2)),
    ("Beta",             f'{info1["beta"]:.2f}' if info1.get("beta") else "—",
                         f'{info2["beta"]:.2f}' if info2.get("beta") else "—"),
    ("Div. Yield",       fmt_pct_plain(info1.get("dividend_yield", 0)),
                         fmt_pct_plain(info2.get("dividend_yield", 0))),
    ("Revenue (TTM)",    fmt_large(info1.get("revenue_ttm")),   fmt_large(info2.get("revenue_ttm"))),
    ("Gross Margin",     fmt_pct_plain(info1.get("gross_margin", 0)),
                         fmt_pct_plain(info2.get("gross_margin", 0))),
    ("Oper. Margin",     fmt_pct_plain(info1.get("operating_margin", 0)),
                         fmt_pct_plain(info2.get("operating_margin", 0))),
    ("ROE",              fmt_pct_plain(info1.get("roe", 0)),    fmt_pct_plain(info2.get("roe", 0))),
    ("Debt/Equity",      f'{info1["debt_equity"]:.1f}' if info1.get("debt_equity") else "—",
                         f'{info2["debt_equity"]:.1f}' if info2.get("debt_equity") else "—"),
    ("52W High",         fmt_price(info1.get("week52_high")),   fmt_price(info2.get("week52_high"))),
    ("52W Low",          fmt_price(info1.get("week52_low")),    fmt_price(info2.get("week52_low"))),
    ("Employees",        fmt_large(info1.get("employees")),     fmt_large(info2.get("employees"))),
    ("Sector",           info1.get("sector", "—"),              info2.get("sector", "—")),
    ("Industry",         info1.get("industry", "—"),            info2.get("industry", "—")),
]

rows_f = ""
for label, v1, v2 in fund_metrics:
    rows_f += f"""<tr style="border-bottom:1px solid #21262D">
      <td style="padding:7px 14px;font-family:'IBM Plex Mono',monospace;font-size:10px;
                 color:#8B949E;text-transform:uppercase;letter-spacing:.06em;
                 text-align:center">{label}</td>
      <td style="padding:7px 14px;font-family:'IBM Plex Mono',monospace;font-size:12px;
                 color:#C9D1D9;text-align:center">{v1}</td>
      <td style="padding:7px 14px;font-family:'IBM Plex Mono',monospace;font-size:12px;
                 color:#C9D1D9;text-align:center">{v2}</td>
    </tr>"""

st.markdown(f"""
<table style="width:100%;border-collapse:collapse;background:#161B22;
              border:1px solid #30363D;border-radius:8px;overflow:hidden">
  <thead><tr style="background:#21262D">
    <th style="padding:10px 14px;font-family:'IBM Plex Mono',monospace;font-size:10px;
               color:#8B949E;text-transform:uppercase;text-align:center">Metric</th>
    <th style="padding:10px 14px;font-family:'IBM Plex Mono',monospace;font-size:13px;
               color:{C1};font-weight:600;text-align:center">{ticker1}<br>
        <span style="font-size:10px;color:#8B949E;font-weight:400">{info1.get('name',ticker1)[:30]}</span></th>
    <th style="padding:10px 14px;font-family:'IBM Plex Mono',monospace;font-size:13px;
               color:{C2};font-weight:600;text-align:center">{ticker2}<br>
        <span style="font-size:10px;color:#8B949E;font-weight:400">{info2.get('name',ticker2)[:30]}</span></th>
  </tr></thead>
  <tbody>{rows_f}</tbody>
</table>""", unsafe_allow_html=True)

# ─── SIGNAL COMPARISON ───────────────────────────────────────────────────────
section_header("Technical Signal Summary")
col_s1, col_s2 = st.columns(2)

for col, t, df_t, colour in [
    (col_s1, ticker1, df1, C1),
    (col_s2, ticker2, df2, C2),
]:
    with col:
        sig_data  = generate_signals(df_t)
        composite = sig_data["composite"]
        sig_colours = {
            "STRONG BUY": C1, "BUY": C1,
            "NEUTRAL": "#8B949E",
            "SELL": "#F85149", "STRONG SELL": "#F85149",
        }
        sc = sig_colours.get(composite, "#8B949E")
        st.markdown(f"""
        <div style="background:#161B22;border:1px solid #30363D;border-radius:8px;
        padding:16px;margin-bottom:8px;">
          <div style="font-family:'IBM Plex Mono',monospace;font-size:14px;
                      font-weight:600;color:{colour};margin-bottom:10px">{t}</div>
          <div style="margin-bottom:12px">
            <span style="font-family:'IBM Plex Mono',monospace;font-size:11px;
                         font-weight:600;color:{sc};border:1px solid {sc};
                         border-radius:4px;padding:4px 12px">{composite}</span>
            <span style="font-family:'IBM Plex Mono',monospace;font-size:10px;
                         color:#8B949E;margin-left:10px">
              {sig_data["buy_count"]} BUY · {sig_data["sell_count"]} SELL
            </span>
          </div>
          {"".join(f'''<div style="display:flex;justify-content:space-between;
            padding:5px 0;border-bottom:1px solid #21262D">
            <span style="font-family:'IBM Plex Mono',monospace;font-size:10px;
                         color:#8B949E">{k}</span>
            <span style="font-family:'IBM Plex Mono',monospace;font-size:11px;font-weight:600;
                         color:{'#3FB950' if v['signal']=='BUY' else ('#F85149' if v['signal']=='SELL' else '#8B949E')}">{v['signal']}</span>
            <span style="font-family:'IBM Plex Mono',monospace;font-size:10px;
                         color:#C9D1D9">{v['value']}</span>
          </div>''' for k, v in sig_data["indicators"].items())}
        </div>""", unsafe_allow_html=True)

# ─── DRAWDOWN COMPARISON ─────────────────────────────────────────────────────
section_header("Drawdown Comparison")
from core.risk_metrics import drawdown_series

dd1 = drawdown_series(df1["Close"]) * 100
dd2 = drawdown_series(df2["Close"]) * 100

fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(x=dd1.index, y=dd1, name=ticker1,
    fill="tozeroy", fillcolor="rgba(63,185,80,0.12)",
    line=dict(color=C1, width=1.5)))
fig_dd.add_trace(go.Scatter(x=dd2.index, y=dd2, name=ticker2,
    fill="tozeroy", fillcolor="rgba(88,166,255,0.10)",
    line=dict(color=C2, width=1.5)))
fig_dd.update_layout(**{**LAYOUT, "height": 300,
    "title": dict(text="Drawdown Comparison (%)", font_size=12)},
    yaxis_title="Drawdown (%)")
st.plotly_chart(fig_dd, use_container_width=True,
                config={"displayModeBar": False})

# ─── CORRELATION ─────────────────────────────────────────────────────────────
section_header("Return Correlation")
col_corr1, col_corr2 = st.columns([2, 1])
with col_corr1:
    roll_corr = ret1.rolling(30).corr(ret2)
    colours_c = ["rgba(63,185,80,0.6)" if v >= 0 else "rgba(248,81,73,0.5)"
                 for v in roll_corr.fillna(0)]
    fig_rc = go.Figure()
    fig_rc.add_trace(go.Bar(x=roll_corr.index, y=roll_corr,
        marker_color=colours_c, opacity=0.8, name="30D Rolling Corr"))
    overall = float(ret1.corr(ret2))
    fig_rc.add_hline(y=overall, line_color="#E3B341", line_dash="dash",
                     line_width=1.5,
                     annotation_text=f" Overall r = {overall:.3f}",
                     annotation_font_color="#E3B341")
    fig_rc.add_hline(y=0, line_color="#8B949E", line_dash="dot", line_width=0.8)
    fig_rc.update_layout(**{**LAYOUT, "height": 280,
        "title": dict(text=f"30-Day Rolling Correlation: {ticker1} vs {ticker2}", font_size=12)},
        yaxis_range=[-1.1, 1.1])
    st.plotly_chart(fig_rc, use_container_width=True,
                    config={"displayModeBar": False})

with col_corr2:
    st.markdown(f"""
    <div style="background:#161B22;border:1px solid #30363D;border-radius:8px;
    padding:16px;margin-top:4px">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#8B949E;
                  text-transform:uppercase;margin-bottom:12px">Correlation Stats</div>
      {"".join(f'''<div style="display:flex;justify-content:space-between;
        padding:7px 0;border-bottom:1px solid #21262D">
        <span style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#8B949E">{k}</span>
        <span style="font-family:'IBM Plex Mono',monospace;font-size:13px;font-weight:600;color:#C9D1D9">{v}</span>
      </div>''' for k, v in [
          ("Pearson r",    f"{ret1.corr(ret2):.4f}"),
          ("Spearman r",   f"{ret1.rank().corr(ret2.rank()):.4f}"),
          ("30D Min",      f"{float(roll_corr.min()):.3f}"),
          ("30D Max",      f"{float(roll_corr.max()):.3f}"),
          ("30D Avg",      f"{float(roll_corr.mean()):.3f}"),
          ("30D Current",  f"{float(roll_corr.iloc[-1]):.3f}"),
      ])}
    </div>""", unsafe_allow_html=True)
