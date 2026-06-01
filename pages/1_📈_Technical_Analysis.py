"""
Page 1 — Technical Analysis
Full indicator suite with interactive multi-panel charts, signal scanner, and pivot levels.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

st.set_page_config(page_title="Technical Analysis · StockPro", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

from core.data_fetcher import fetch_ohlcv, validate_ticker, PERIOD_MAP, detect_market, currency_symbol
from core.indicators  import (
    ema, sma, rsi, macd, bollinger_bands, stochastic, atr,
    keltner_channels, donchian_channels, adx, williams_r,
    cci, obv, money_flow_index, chaikin_money_flow,
    historical_volatility, parabolic_sar, volume_ratio,
    generate_signals
)
from utils.helpers import (
    inject_css, section_header, signal_badge,
    signals_table, fmt_price, kpi_row, kpi_card
)
from utils.charts import multi_panel_chart, THEME
import plotly.graph_objects as go
from plotly.subplots import make_subplots

inject_css()

# ─── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style="font-family:'IBM Plex Mono',monospace;font-size:16px;
    font-weight:600;color:#3FB950;padding:8px 0 16px;">📈 StockPro
    <span style="font-size:10px;color:#8B949E;font-weight:400;display:block;
    letter-spacing:.1em;margin-top:2px;">ANALYTICS TERMINAL</span></div>""",
    unsafe_allow_html=True)

    ticker = st.text_input(
        "Ticker Symbol", value="AAPL",
        placeholder="AAPL · RELIANCE.NS · TCS.NS",
        help="US: AAPL  |  NSE: RELIANCE.NS  |  BSE: RELIANCE.BO  |  Index: ^NSEI"
    ).upper().strip()
    period_label = st.selectbox("Time Period", list(PERIOD_MAP.keys()), index=3)
    period, interval = PERIOD_MAP[period_label]

    st.divider()
    st.markdown("**Chart Settings**")
    show_rsi  = st.checkbox("RSI Panel",  value=True)
    show_macd = st.checkbox("MACD Panel", value=True)
    show_bb   = st.checkbox("Bollinger Bands", value=True)
    show_ema_cross = st.checkbox("EMA Cross (20/50)", value=True)
    show_sar  = st.checkbox("Parabolic SAR", value=False)
    show_vol_overlay = st.checkbox("Volume Profile", value=True)

    st.divider()
    st.markdown("**Indicator Parameters**")
    rsi_period  = st.slider("RSI Period",  7,  30, 14)
    macd_fast   = st.slider("MACD Fast",   5,  20, 12)
    macd_slow   = st.slider("MACD Slow",  15,  50, 26)
    bb_window   = st.slider("BB Window",  10,  50, 20)
    bb_std      = st.slider("BB Std Dev", 1.0, 3.0, 2.0, 0.5)
    st.divider()
    st.caption("Data via Yahoo Finance · Not financial advice")

# ─── LOAD DATA ──────────────────────────────────────────────────────────────
if not ticker:
    st.info("Enter a ticker symbol in the sidebar.")
    st.stop()

with st.spinner(f"Loading {ticker}…"):
    valid, err = validate_ticker(ticker)
    if not valid:
        st.error(f"**{ticker}** — {err}")
        st.stop()
    df = fetch_ohlcv(ticker, period, interval)

if df.empty:
    st.error("No price data returned. Try a different ticker or period.")
    st.stop()

c = df["Close"]

# ─── PAGE HEADER ────────────────────────────────────────────────────────────
st.markdown(f"""<div style="font-family:'IBM Plex Mono',monospace;padding:10px 0 6px;
border-bottom:1px solid #30363D;margin-bottom:16px;">
<span style="font-size:20px;font-weight:600;color:#C9D1D9">{ticker}</span>&nbsp;&nbsp;
<span style="font-size:13px;color:#8B949E">Technical Analysis</span>&nbsp;
<span style="font-size:11px;color:#E3B341;font-family:'IBM Plex Mono',monospace">
  {'🇮🇳 NSE' if detect_market(ticker)=='NSE' else ('🇮🇳 BSE' if detect_market(ticker)=='BSE' else '🇺🇸 US')}
</span>
<span style="float:right;font-size:12px;color:#3FB950">{len(df)} sessions · {
df.index[0].strftime('%d %b %Y')} → {df.index[-1].strftime('%d %b %Y')}</span>
</div>""", unsafe_allow_html=True)

# ─── KPI SNAPSHOT ───────────────────────────────────────────────────────────
section_header("Current Indicators")
rsi_now   = rsi(c, rsi_period).iloc[-1]
macd_df   = macd(c, macd_fast, macd_slow)
macd_hist = macd_df["Hist"].iloc[-1]
bb_df     = bollinger_bands(c, bb_window, bb_std)
bb_pct    = bb_df["BB_%B"].iloc[-1]
atr_now   = atr(df).iloc[-1]
hv_now    = historical_volatility(c).iloc[-1] * 100
stoch_df  = stochastic(df)
k_now     = stoch_df["%K"].iloc[-1]
mfi_now   = money_flow_index(df).iloc[-1]
cci_now   = cci_val = cci(df).iloc[-1]
wr_now    = williams_r(df).iloc[-1]
vr_now    = volume_ratio(df).iloc[-1]

def rsi_label(v):
    if v < 30: return "OVERSOLD"
    if v > 70: return "OVERBOUGHT"
    return "NEUTRAL"

kpi_row([
    kpi_card("RSI", f"{rsi_now:.1f}", rsi_label(rsi_now),
             "neg" if rsi_now > 70 else ("pos" if rsi_now < 30 else "")),
    kpi_card("MACD Hist", f"{macd_hist:+.3f}", "",
             "pos" if macd_hist >= 0 else "neg"),
    kpi_card("BB %B", f"{bb_pct:.2f}", "<0 oversold >1 overbought",
             "pos" if bb_pct < 0.2 else ("neg" if bb_pct > 0.8 else "")),
    kpi_card("ATR (14)", f"{atr_now:.2f}", "Avg True Range"),
    kpi_card("Hist. Vol", f"{hv_now:.1f}%", "20-day annualised"),
    kpi_card("Stoch %K", f"{k_now:.1f}", "Stochastic Oscillator",
             "pos" if k_now < 20 else ("neg" if k_now > 80 else "")),
    kpi_card("MFI (14)", f"{mfi_now:.1f}", "Money Flow Index",
             "pos" if mfi_now < 20 else ("neg" if mfi_now > 80 else "")),
    kpi_card("CCI (20)", f"{cci_val:.0f}", "Commodity Channel",
             "pos" if cci_val < -100 else ("neg" if cci_val > 100 else "")),
    kpi_card("Williams %R", f"{wr_now:.1f}", "<-80 oversold",
             "pos" if wr_now < -80 else ("neg" if wr_now > -20 else "")),
    kpi_card("Vol Ratio", f"{vr_now:.2f}x", "vs 14-day average",
             "pos" if vr_now > 1.5 else ""),
])

# ─── MAIN CHART ─────────────────────────────────────────────────────────────
section_header("Price & Indicators")
fig = multi_panel_chart(df, ticker, show_rsi=show_rsi, show_macd=show_macd)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ─── EXTRA INDICATOR CHARTS (tabbed) ────────────────────────────────────────
section_header("Indicator Deep Dive")
tabs = st.tabs([
    "  Volatility  ", "  Oscillators  ", "  Volume  ",
    "  Trend  ", "  Pivot Levels  "
])

# TAB 1 — VOLATILITY ─────────────────────────────────────────────────────────
with tabs[0]:
    layout = dict(
        plot_bgcolor="#0D1117", paper_bgcolor="#0D1117",
        font=dict(family="IBM Plex Mono, monospace", color="#C9D1D9", size=11),
        margin=dict(l=12, r=12, t=36, b=12), height=420,
        xaxis=dict(gridcolor="#21262D", showspikes=True),
        yaxis=dict(gridcolor="#21262D"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )

    col1, col2 = st.columns(2)
    with col1:
        # Bollinger Bands
        bb = bollinger_bands(c, bb_window, bb_std)
        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(
            x=list(bb.index)+list(bb.index[::-1]),
            y=list(bb["BB_Upper"])+list(bb["BB_Lower"][::-1]),
            fill="toself", fillcolor="rgba(88,166,255,0.07)",
            line=dict(color="rgba(0,0,0,0)"), name="BB Band"))
        fig_bb.add_trace(go.Scatter(x=c.index, y=c, name="Close",
            line=dict(color="#C9D1D9", width=1.5)))
        fig_bb.add_trace(go.Scatter(x=bb.index, y=bb["BB_Upper"], name="Upper",
            line=dict(color="#58A6FF", width=1, dash="dot")))
        fig_bb.add_trace(go.Scatter(x=bb.index, y=bb["BB_Mid"], name="Mid",
            line=dict(color="#E3B341", width=1.2)))
        fig_bb.add_trace(go.Scatter(x=bb.index, y=bb["BB_Lower"], name="Lower",
            line=dict(color="#58A6FF", width=1, dash="dot")))
        fig_bb.update_layout(**layout, title=dict(text=f"Bollinger Bands ({bb_window},{bb_std})", font_size=12))
        st.plotly_chart(fig_bb, use_container_width=True, config={"displayModeBar": False})

    with col2:
        # BB Width + ATR
        fig_bw = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               vertical_spacing=0.04, row_heights=[0.5, 0.5])
        fig_bw.add_trace(go.Scatter(x=bb.index, y=bb["BB_Width"]*100,
            name="BB Width %", line=dict(color="#BC8CFF", width=1.5)), row=1, col=1)
        atr_s = atr(df)
        fig_bw.add_trace(go.Scatter(x=atr_s.index, y=atr_s,
            name="ATR 14", line=dict(color="#FFA657", width=1.5)), row=2, col=1)
        fig_bw.update_layout(**{**layout, "title": dict(text="Bandwidth & ATR", font_size=12)})
        fig_bw.update_yaxes(gridcolor="#21262D", row=1, col=1)
        fig_bw.update_yaxes(gridcolor="#21262D", row=2, col=1)
        st.plotly_chart(fig_bw, use_container_width=True, config={"displayModeBar": False})

    # Keltner + Historical Volatility
    col3, col4 = st.columns(2)
    with col3:
        kc = keltner_channels(df)
        fig_kc = go.Figure()
        fig_kc.add_trace(go.Scatter(
            x=list(kc.index)+list(kc.index[::-1]),
            y=list(kc["KC_Upper"])+list(kc["KC_Lower"][::-1]),
            fill="toself", fillcolor="rgba(63,185,80,0.06)",
            line=dict(color="rgba(0,0,0,0)"), name="KC Band"))
        fig_kc.add_trace(go.Scatter(x=c.index, y=c, name="Close",
            line=dict(color="#C9D1D9", width=1.5)))
        fig_kc.add_trace(go.Scatter(x=kc.index, y=kc["KC_Upper"], name="KC Upper",
            line=dict(color="#3FB950", width=1, dash="dot")))
        fig_kc.add_trace(go.Scatter(x=kc.index, y=kc["KC_Lower"], name="KC Lower",
            line=dict(color="#3FB950", width=1, dash="dot")))
        fig_kc.update_layout(**layout, title=dict(text="Keltner Channels", font_size=12))
        st.plotly_chart(fig_kc, use_container_width=True, config={"displayModeBar": False})

    with col4:
        hv20 = historical_volatility(c, 20) * 100
        hv50 = historical_volatility(c, 50) * 100
        fig_hv = go.Figure()
        fig_hv.add_trace(go.Scatter(x=hv20.index, y=hv20, name="HV 20",
            line=dict(color="#58A6FF", width=1.5)))
        fig_hv.add_trace(go.Scatter(x=hv50.index, y=hv50, name="HV 50",
            line=dict(color="#E3B341", width=1.5)))
        fig_hv.update_layout(**layout, title=dict(text="Historical Volatility (Annualised %)", font_size=12))
        st.plotly_chart(fig_hv, use_container_width=True, config={"displayModeBar": False})

# TAB 2 — OSCILLATORS ────────────────────────────────────────────────────────
with tabs[1]:
    col1, col2 = st.columns(2)
    layout2 = dict(
        plot_bgcolor="#0D1117", paper_bgcolor="#0D1117",
        font=dict(family="IBM Plex Mono, monospace", color="#C9D1D9", size=11),
        margin=dict(l=12, r=12, t=36, b=12), height=300,
        xaxis=dict(gridcolor="#21262D"), yaxis=dict(gridcolor="#21262D"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )

    with col1:
        # RSI
        rsi_s = rsi(c, rsi_period)
        fig_rsi = go.Figure()
        fig_rsi.add_hrect(y0=70, y1=100, fillcolor="rgba(248,81,73,0.05)", line_width=0)
        fig_rsi.add_hrect(y0=0,  y1=30,  fillcolor="rgba(63,185,80,0.05)",  line_width=0)
        fig_rsi.add_trace(go.Scatter(x=rsi_s.index, y=rsi_s, name=f"RSI {rsi_period}",
            line=dict(color="#79C0FF", width=1.8)))
        fig_rsi.add_hline(y=70, line_color="#F85149", line_dash="dot", line_width=1)
        fig_rsi.add_hline(y=30, line_color="#3FB950", line_dash="dot", line_width=1)
        fig_rsi.add_hline(y=50, line_color="#8B949E", line_dash="dot", line_width=0.8)
        fig_rsi.update_yaxes(range=[0, 100])
        fig_rsi.update_layout(**layout2, title=dict(text=f"RSI ({rsi_period})", font_size=12))
        st.plotly_chart(fig_rsi, use_container_width=True, config={"displayModeBar": False})

    with col2:
        # Stochastic
        stch = stochastic(df)
        fig_st = go.Figure()
        fig_st.add_hrect(y0=80, y1=100, fillcolor="rgba(248,81,73,0.05)", line_width=0)
        fig_st.add_hrect(y0=0,  y1=20,  fillcolor="rgba(63,185,80,0.05)",  line_width=0)
        fig_st.add_trace(go.Scatter(x=stch.index, y=stch["%K"], name="%K",
            line=dict(color="#E3B341", width=1.8)))
        fig_st.add_trace(go.Scatter(x=stch.index, y=stch["%D"], name="%D",
            line=dict(color="#BC8CFF", width=1.5, dash="dot")))
        fig_st.add_hline(y=80, line_color="#F85149", line_dash="dot", line_width=1)
        fig_st.add_hline(y=20, line_color="#3FB950", line_dash="dot", line_width=1)
        fig_st.update_yaxes(range=[0, 100])
        fig_st.update_layout(**layout2, title=dict(text="Stochastic (14,3)", font_size=12))
        st.plotly_chart(fig_st, use_container_width=True, config={"displayModeBar": False})

    col3, col4 = st.columns(2)
    with col3:
        # Williams %R
        wr = williams_r(df)
        fig_wr = go.Figure()
        fig_wr.add_hrect(y0=-20, y1=0,    fillcolor="rgba(248,81,73,0.05)", line_width=0)
        fig_wr.add_hrect(y0=-100,y1=-80,  fillcolor="rgba(63,185,80,0.05)",  line_width=0)
        fig_wr.add_trace(go.Scatter(x=wr.index, y=wr, name="Williams %R",
            line=dict(color="#FFA657", width=1.8)))
        fig_wr.add_hline(y=-20,  line_color="#F85149", line_dash="dot", line_width=1)
        fig_wr.add_hline(y=-80,  line_color="#3FB950", line_dash="dot", line_width=1)
        fig_wr.update_yaxes(range=[-100, 0])
        fig_wr.update_layout(**layout2, title=dict(text="Williams %R (14)", font_size=12))
        st.plotly_chart(fig_wr, use_container_width=True, config={"displayModeBar": False})

    with col4:
        # CCI
        cci_s = cci(df)
        fig_cci = go.Figure()
        fig_cci.add_hrect(y0=100, y1=cci_s.max()*1.1, fillcolor="rgba(248,81,73,0.05)", line_width=0)
        fig_cci.add_hrect(y0=cci_s.min()*1.1, y1=-100,  fillcolor="rgba(63,185,80,0.05)", line_width=0)
        fig_cci.add_trace(go.Scatter(x=cci_s.index, y=cci_s, name="CCI 20",
            line=dict(color="#3FB950", width=1.8)))
        fig_cci.add_hline(y=100,  line_color="#F85149", line_dash="dot", line_width=1)
        fig_cci.add_hline(y=-100, line_color="#3FB950", line_dash="dot", line_width=1)
        fig_cci.add_hline(y=0,    line_color="#8B949E", line_dash="dot", line_width=0.8)
        fig_cci.update_layout(**layout2, title=dict(text="CCI (20)", font_size=12))
        st.plotly_chart(fig_cci, use_container_width=True, config={"displayModeBar": False})

# TAB 3 — VOLUME ─────────────────────────────────────────────────────────────
with tabs[2]:
    layout3 = dict(
        plot_bgcolor="#0D1117", paper_bgcolor="#0D1117",
        font=dict(family="IBM Plex Mono, monospace", color="#C9D1D9", size=11),
        margin=dict(l=12, r=12, t=36, b=12), height=300,
        xaxis=dict(gridcolor="#21262D"), yaxis=dict(gridcolor="#21262D"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    col1, col2 = st.columns(2)
    with col1:
        obv_s = obv(df)
        fig_obv = go.Figure()
        fig_obv.add_trace(go.Scatter(x=obv_s.index, y=obv_s, name="OBV",
            line=dict(color="#58A6FF", width=1.8),
            fill="tozeroy", fillcolor="rgba(88,166,255,0.06)"))
        fig_obv.update_layout(**layout3, title=dict(text="On-Balance Volume", font_size=12))
        st.plotly_chart(fig_obv, use_container_width=True, config={"displayModeBar": False})

    with col2:
        mfi_s = money_flow_index(df)
        fig_mfi = go.Figure()
        fig_mfi.add_hrect(y0=80, y1=100, fillcolor="rgba(248,81,73,0.05)", line_width=0)
        fig_mfi.add_hrect(y0=0,  y1=20,  fillcolor="rgba(63,185,80,0.05)",  line_width=0)
        fig_mfi.add_trace(go.Scatter(x=mfi_s.index, y=mfi_s, name="MFI 14",
            line=dict(color="#3FB950", width=1.8)))
        fig_mfi.add_hline(y=80, line_color="#F85149", line_dash="dot", line_width=1)
        fig_mfi.add_hline(y=20, line_color="#3FB950", line_dash="dot", line_width=1)
        fig_mfi.update_yaxes(range=[0, 100])
        fig_mfi.update_layout(**layout3, title=dict(text="Money Flow Index (14)", font_size=12))
        st.plotly_chart(fig_mfi, use_container_width=True, config={"displayModeBar": False})

    col3, col4 = st.columns(2)
    with col3:
        cmf_s = chaikin_money_flow(df)
        fig_cmf = go.Figure()
        colours_cmf = ["#3FB950" if v >= 0 else "#F85149" for v in cmf_s]
        fig_cmf.add_trace(go.Bar(x=cmf_s.index, y=cmf_s, name="CMF 20",
            marker_color=colours_cmf, opacity=0.75))
        fig_cmf.add_hline(y=0, line_color="#8B949E", line_width=1)
        fig_cmf.update_layout(**layout3, title=dict(text="Chaikin Money Flow (20)", font_size=12))
        st.plotly_chart(fig_cmf, use_container_width=True, config={"displayModeBar": False})

    with col4:
        vr_s = volume_ratio(df)
        fig_vr = go.Figure()
        colours_vr = ["#3FB950" if c >= df["Close"].shift(1).iloc[i] else "#F85149"
                      for i, c in enumerate(df["Close"])]
        fig_vr.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
            marker_color=colours_vr, opacity=0.6))
        fig_vr.add_trace(go.Scatter(x=vr_s.index,
            y=df["Volume"].rolling(14).mean(), name="14D MA Vol",
            line=dict(color="#E3B341", width=1.8)))
        fig_vr.update_layout(**layout3, title=dict(text="Volume with 14D Moving Average", font_size=12))
        st.plotly_chart(fig_vr, use_container_width=True, config={"displayModeBar": False})

# TAB 4 — TREND ──────────────────────────────────────────────────────────────
with tabs[3]:
    layout4 = dict(
        plot_bgcolor="#0D1117", paper_bgcolor="#0D1117",
        font=dict(family="IBM Plex Mono, monospace", color="#C9D1D9", size=11),
        margin=dict(l=12, r=12, t=36, b=12), height=320,
        xaxis=dict(gridcolor="#21262D"), yaxis=dict(gridcolor="#21262D"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    col1, col2 = st.columns(2)
    with col1:
        adx_df = adx(df)
        fig_adx = go.Figure()
        fig_adx.add_trace(go.Scatter(x=adx_df.index, y=adx_df["ADX"], name="ADX",
            line=dict(color="#E3B341", width=2)))
        fig_adx.add_trace(go.Scatter(x=adx_df.index, y=adx_df["DI+"], name="+DI",
            line=dict(color="#3FB950", width=1.5)))
        fig_adx.add_trace(go.Scatter(x=adx_df.index, y=adx_df["DI-"], name="-DI",
            line=dict(color="#F85149", width=1.5)))
        fig_adx.add_hline(y=25, line_color="#8B949E", line_dash="dot", line_width=1,
                          annotation_text="Trend Threshold (25)")
        fig_adx.update_layout(**layout4, title=dict(text="ADX — Trend Strength", font_size=12))
        st.plotly_chart(fig_adx, use_container_width=True, config={"displayModeBar": False})

    with col2:
        # Moving average ribbon
        fig_ma = go.Figure()
        fig_ma.add_trace(go.Scatter(x=c.index, y=c, name="Close",
            line=dict(color="#C9D1D9", width=1.5)))
        ma_colours = ["#3FB950","#58A6FF","#E3B341","#BC8CFF","#FFA657"]
        for i, w in enumerate([9, 20, 50, 100, 200]):
            if len(df) >= w:
                fig_ma.add_trace(go.Scatter(x=c.index, y=ema(c, w), name=f"EMA {w}",
                    line=dict(color=ma_colours[i], width=1.2, dash="dot")))
        fig_ma.update_layout(**layout4, title=dict(text="EMA Ribbon (9/20/50/100/200)", font_size=12))
        st.plotly_chart(fig_ma, use_container_width=True, config={"displayModeBar": False})

    # Parabolic SAR
    col3, col4 = st.columns(2)
    with col3:
        psar = parabolic_sar(df)
        fig_psar = go.Figure()
        fig_psar.add_trace(go.Candlestick(
            x=df.index[-120:], open=df["Open"][-120:], high=df["High"][-120:],
            low=df["Low"][-120:], close=df["Close"][-120:], name="OHLC",
            increasing_line_color="#3FB950", decreasing_line_color="#F85149",
            increasing_fillcolor="#3FB950", decreasing_fillcolor="#F85149"))
        fig_psar.add_trace(go.Scatter(x=psar.index[-120:], y=psar.values[-120:],
            mode="markers", name="PSAR",
            marker=dict(size=3, color="#E3B341", symbol="circle")))
        fig_psar.update_layout(**layout4, title=dict(text="Parabolic SAR (last 120 sessions)", font_size=12))
        fig_psar.update_xaxes(rangeslider_visible=False)
        st.plotly_chart(fig_psar, use_container_width=True, config={"displayModeBar": False})

    with col4:
        # Donchian channels
        dc = donchian_channels(df)
        fig_dc = go.Figure()
        fig_dc.add_trace(go.Scatter(
            x=list(dc.index)+list(dc.index[::-1]),
            y=list(dc["DC_Upper"])+list(dc["DC_Lower"][::-1]),
            fill="toself", fillcolor="rgba(188,140,255,0.06)",
            line=dict(color="rgba(0,0,0,0)"), name="Donchian Band"))
        fig_dc.add_trace(go.Scatter(x=c.index, y=c, name="Close",
            line=dict(color="#C9D1D9", width=1.5)))
        fig_dc.add_trace(go.Scatter(x=dc.index, y=dc["DC_Upper"], name="DC Upper",
            line=dict(color="#BC8CFF", width=1, dash="dot")))
        fig_dc.add_trace(go.Scatter(x=dc.index, y=dc["DC_Mid"], name="DC Mid",
            line=dict(color="#E3B341", width=1.2)))
        fig_dc.add_trace(go.Scatter(x=dc.index, y=dc["DC_Lower"], name="DC Lower",
            line=dict(color="#BC8CFF", width=1, dash="dot")))
        fig_dc.update_layout(**layout4, title=dict(text="Donchian Channels (20)", font_size=12))
        st.plotly_chart(fig_dc, use_container_width=True, config={"displayModeBar": False})

# TAB 5 — PIVOT LEVELS ───────────────────────────────────────────────────────
with tabs[4]:
    # Classic pivot points (daily)
    last_row = df.iloc[-2]  # use previous session
    H, L, C_p = last_row["High"], last_row["Low"], last_row["Close"]
    PP  = (H + L + C_p) / 3
    R1  = 2 * PP - L;      S1 = 2 * PP - H
    R2  = PP + (H - L);    S2 = PP - (H - L)
    R3  = H + 2 * (PP - L);S3 = L - 2 * (H - PP)
    # Fibonacci pivots
    rng = H - L
    fib_r1 = PP + 0.382 * rng
    fib_r2 = PP + 0.618 * rng
    fib_r3 = PP + 1.000 * rng
    fib_s1 = PP - 0.382 * rng
    fib_s2 = PP - 0.618 * rng
    fib_s3 = PP - 1.000 * rng

    fig_piv = go.Figure()
    fig_piv.add_trace(go.Scatter(x=df.index[-60:], y=df["Close"][-60:],
        name="Close", line=dict(color="#C9D1D9", width=1.8)))

    pivot_levels = [
        (PP,  "#E3B341", "PP",   "dot"),
        (R1,  "#3FB950", "R1",   "dash"),
        (R2,  "#3FB950", "R2",   "dash"),
        (R3,  "#3FB950", "R3",   "dash"),
        (S1,  "#F85149", "S1",   "dash"),
        (S2,  "#F85149", "S2",   "dash"),
        (S3,  "#F85149", "S3",   "dash"),
    ]
    for level, colour, name, dash in pivot_levels:
        fig_piv.add_hline(y=level, line_color=colour, line_dash=dash,
                          line_width=1.2,
                          annotation_text=f"  {name} {level:.2f}",
                          annotation_font_size=10,
                          annotation_font_color=colour)

    fig_piv.update_layout(
        plot_bgcolor="#0D1117", paper_bgcolor="#0D1117",
        font=dict(family="IBM Plex Mono, monospace", color="#C9D1D9", size=11),
        margin=dict(l=12, r=12, t=36, b=12), height=420,
        xaxis=dict(gridcolor="#21262D"), yaxis=dict(gridcolor="#21262D"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        title=dict(text="Classic Pivot Points (last 60 sessions)", font_size=12),
    )
    st.plotly_chart(fig_piv, use_container_width=True, config={"displayModeBar": False})

    # Pivot table
    col_p, col_f = st.columns(2)
    with col_p:
        section_header("Classic Pivot Levels")
        pivot_data = {
            "Resistance 3": R3, "Resistance 2": R2, "Resistance 1": R1,
            "Pivot Point": PP,
            "Support 1": S1, "Support 2": S2, "Support 3": S3,
        }
        rows = ""
        for lbl, val in pivot_data.items():
            colour = "#3FB950" if "Resist" in lbl else ("#E3B341" if "Pivot" in lbl else "#F85149")
            dist   = ((val - float(c.iloc[-1])) / float(c.iloc[-1])) * 100
            rows += f"""<tr>
              <td style="padding:6px 12px;border-bottom:1px solid #30363D;
                         font-family:'IBM Plex Mono',monospace;font-size:12px;
                         color:{colour}">{lbl}</td>
              <td style="padding:6px 12px;border-bottom:1px solid #30363D;
                         font-family:'IBM Plex Mono',monospace;font-size:12px;
                         color:#C9D1D9">{val:.3f}</td>
              <td style="padding:6px 12px;border-bottom:1px solid #30363D;
                         font-family:'IBM Plex Mono',monospace;font-size:12px;
                         color:{'#3FB950' if dist>0 else '#F85149'}">{dist:+.2f}%</td>
            </tr>"""
        st.markdown(f"""<table style="width:100%;border-collapse:collapse;
          background:#161B22;border:1px solid #30363D;border-radius:6px;overflow:hidden">
          <thead><tr style="background:#21262D">
            <th style="padding:8px 12px;text-align:left;font-family:'IBM Plex Mono',monospace;
               font-size:10px;color:#8B949E;text-transform:uppercase">Level</th>
            <th style="padding:8px 12px;text-align:left;font-family:'IBM Plex Mono',monospace;
               font-size:10px;color:#8B949E;text-transform:uppercase">Price</th>
            <th style="padding:8px 12px;text-align:left;font-family:'IBM Plex Mono',monospace;
               font-size:10px;color:#8B949E;text-transform:uppercase">Distance</th>
          </tr></thead><tbody>{rows}</tbody></table>""", unsafe_allow_html=True)

    with col_f:
        section_header("Fibonacci Pivot Levels")
        fib_data = {
            "Fib R3 (100%)": fib_r3, "Fib R2 (61.8%)": fib_r2,
            "Fib R1 (38.2%)": fib_r1, "Pivot Point": PP,
            "Fib S1 (38.2%)": fib_s1, "Fib S2 (61.8%)": fib_s2,
            "Fib S3 (100%)": fib_s3,
        }
        rows_f = ""
        for lbl, val in fib_data.items():
            colour = "#3FB950" if "R" in lbl else ("#E3B341" if "Pivot" in lbl else "#F85149")
            dist   = ((val - float(c.iloc[-1])) / float(c.iloc[-1])) * 100
            rows_f += f"""<tr>
              <td style="padding:6px 12px;border-bottom:1px solid #30363D;
                         font-family:'IBM Plex Mono',monospace;font-size:12px;
                         color:{colour}">{lbl}</td>
              <td style="padding:6px 12px;border-bottom:1px solid #30363D;
                         font-family:'IBM Plex Mono',monospace;font-size:12px;
                         color:#C9D1D9">{val:.3f}</td>
              <td style="padding:6px 12px;border-bottom:1px solid #30363D;
                         font-family:'IBM Plex Mono',monospace;font-size:12px;
                         color:{'#3FB950' if dist>0 else '#F85149'}">{dist:+.2f}%</td>
            </tr>"""
        st.markdown(f"""<table style="width:100%;border-collapse:collapse;
          background:#161B22;border:1px solid #30363D;border-radius:6px;overflow:hidden">
          <thead><tr style="background:#21262D">
            <th style="padding:8px 12px;text-align:left;font-family:'IBM Plex Mono',monospace;
               font-size:10px;color:#8B949E;text-transform:uppercase">Level</th>
            <th style="padding:8px 12px;text-align:left;font-family:'IBM Plex Mono',monospace;
               font-size:10px;color:#8B949E;text-transform:uppercase">Price</th>
            <th style="padding:8px 12px;text-align:left;font-family:'IBM Plex Mono',monospace;
               font-size:10px;color:#8B949E;text-transform:uppercase">Distance</th>
          </tr></thead><tbody>{rows_f}</tbody></table>""", unsafe_allow_html=True)

# ─── SIGNAL SUMMARY ─────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
section_header("Signal Summary — All Indicators")
sig_data   = generate_signals(df)
composite  = sig_data["composite"]
buy_n      = sig_data["buy_count"]
sell_n     = sig_data["sell_count"]

st.markdown(
    f'<p style="margin-bottom:12px;">Composite signal: '
    f'{signal_badge(composite)}&nbsp;&nbsp;'
    f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;'
    f'color:#8B949E;">{buy_n} BUY &nbsp;·&nbsp; {sell_n} SELL &nbsp;·&nbsp; '
    f'{8 - buy_n - sell_n} NEUTRAL &nbsp; of 8 indicators</span></p>',
    unsafe_allow_html=True,
)
signals_table(sig_data["indicators"])
