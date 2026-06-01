"""
Page 5 — Stock Screener
Screen US (S&P 500 sample) and Indian (Nifty 50) stocks by fundamental +
technical filters. Pure Streamlit — no external AI APIs.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

st.set_page_config(
    page_title="Screener · StockPro", page_icon="🔍",
    layout="wide", initial_sidebar_state="expanded"
)

from core.data_fetcher  import fetch_ohlcv, fetch_fundamentals, currency_symbol, PERIOD_MAP
from core.indicators    import rsi, macd, ema, atr, bollinger_bands, generate_signals, volume_ratio
from utils.helpers      import inject_css, section_header, kpi_row, kpi_card, fmt_pct, fmt_large
import plotly.graph_objects as go

inject_css()

# ─── UNIVERSES ───────────────────────────────────────────────────────────────

# Nifty 50 — Yahoo Finance NSE symbols (.NS suffix)
NIFTY50 = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS",
    "HINDUNILVR.NS","ITC.NS","SBIN.NS","BHARTIARTL.NS","KOTAKBANK.NS",
    "LT.NS","AXISBANK.NS","ASIANPAINT.NS","MARUTI.NS","HCLTECH.NS",
    "SUNPHARMA.NS","TITAN.NS","BAJFINANCE.NS","WIPRO.NS","ULTRACEMCO.NS",
    "TECHM.NS","POWERGRID.NS","NTPC.NS","ONGC.NS","JSWSTEEL.NS",
    "TATAMOTORS.NS","TATASTEEL.NS","BAJAJFINSV.NS","ADANIENT.NS","ADANIPORTS.NS",
    "COALINDIA.NS","DIVISLAB.NS","DRREDDY.NS","EICHERMOT.NS","BRITANNIA.NS",
    "CIPLA.NS","HEROMOTOCO.NS","APOLLOHOSP.NS","GRASIM.NS","BPCL.NS",
    "HINDALCO.NS","TATACONSUM.NS","NESTLEIND.NS","INDUSINDBK.NS","SBILIFE.NS",
    "HDFCLIFE.NS","M&M.NS","UPL.NS","SHRIRAMFIN.NS","BEL.NS",
]

NIFTY50_SECTORS = {
    "RELIANCE.NS":"Energy","TCS.NS":"IT","HDFCBANK.NS":"Banking","INFY.NS":"IT",
    "ICICIBANK.NS":"Banking","HINDUNILVR.NS":"FMCG","ITC.NS":"FMCG",
    "SBIN.NS":"Banking","BHARTIARTL.NS":"Telecom","KOTAKBANK.NS":"Banking",
    "LT.NS":"Industrials","AXISBANK.NS":"Banking","ASIANPAINT.NS":"Consumer",
    "MARUTI.NS":"Auto","HCLTECH.NS":"IT","SUNPHARMA.NS":"Pharma",
    "TITAN.NS":"Consumer","BAJFINANCE.NS":"NBFC","WIPRO.NS":"IT",
    "ULTRACEMCO.NS":"Cement","TECHM.NS":"IT","POWERGRID.NS":"Utilities",
    "NTPC.NS":"Utilities","ONGC.NS":"Energy","JSWSTEEL.NS":"Metals",
    "TATAMOTORS.NS":"Auto","TATASTEEL.NS":"Metals","BAJAJFINSV.NS":"NBFC",
    "ADANIENT.NS":"Conglomerate","ADANIPORTS.NS":"Logistics","COALINDIA.NS":"Mining",
    "DIVISLAB.NS":"Pharma","DRREDDY.NS":"Pharma","EICHERMOT.NS":"Auto",
    "BRITANNIA.NS":"FMCG","CIPLA.NS":"Pharma","HEROMOTOCO.NS":"Auto",
    "APOLLOHOSP.NS":"Healthcare","GRASIM.NS":"Cement","BPCL.NS":"Energy",
    "HINDALCO.NS":"Metals","TATACONSUM.NS":"FMCG","NESTLEIND.NS":"FMCG",
    "INDUSINDBK.NS":"Banking","SBILIFE.NS":"Insurance","HDFCLIFE.NS":"Insurance",
    "M&M.NS":"Auto","UPL.NS":"Chemicals","SHRIRAMFIN.NS":"NBFC","BEL.NS":"Defence",
}

# S&P 500 sample — US symbols
SP500_SAMPLE = [
    "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","BRK-B","LLY","JPM",
    "V","UNH","XOM","JNJ","MA","PG","HD","AVGO","COST","MRK",
    "ABBV","CVX","KO","PEP","WMT","BAC","NFLX","TMO","CSCO","ABT",
    "CRM","ORCL","ACN","MCD","NKE","ADBE","DHR","INTC","NEE","PM",
    "TXN","AMD","QCOM","UPS","HON","IBM","CAT","GS","MS","BA",
]
SP500_SECTORS = {
    "AAPL":"Technology","MSFT":"Technology","GOOGL":"Technology","AMZN":"Consumer",
    "NVDA":"Technology","META":"Communication","TSLA":"Consumer","BRK-B":"Financials",
    "LLY":"Healthcare","JPM":"Financials","V":"Financials","UNH":"Healthcare",
    "XOM":"Energy","JNJ":"Healthcare","MA":"Financials","PG":"Staples",
    "HD":"Consumer","AVGO":"Technology","COST":"Staples","MRK":"Healthcare",
    "ABBV":"Healthcare","CVX":"Energy","KO":"Staples","PEP":"Staples",
    "WMT":"Staples","BAC":"Financials","NFLX":"Communication","TMO":"Healthcare",
    "CSCO":"Technology","ABT":"Healthcare","CRM":"Technology","ORCL":"Technology",
    "ACN":"Technology","MCD":"Consumer","NKE":"Consumer","ADBE":"Technology",
    "DHR":"Healthcare","INTC":"Technology","NEE":"Utilities","PM":"Staples",
    "TXN":"Technology","AMD":"Technology","QCOM":"Technology","UPS":"Industrials",
    "HON":"Industrials","IBM":"Technology","CAT":"Industrials","GS":"Financials",
    "MS":"Financials","BA":"Industrials",
}

# ─── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style="font-family:'IBM Plex Mono',monospace;font-size:16px;
    font-weight:600;color:#3FB950;padding:8px 0 16px;">📈 StockPro
    <span style="font-size:10px;color:#8B949E;font-weight:400;display:block;
    letter-spacing:.1em;margin-top:2px;">ANALYTICS TERMINAL</span></div>""",
    unsafe_allow_html=True)

    # ── Universe picker
    st.markdown("**Universe**")
    market_choice = st.radio(
        "Market",
        ["🇮🇳  NSE — Nifty 50", "🇺🇸  US — S&P 500", "✏️  Custom watchlist"],
        index=0,
        label_visibility="collapsed",
    )

    if "Custom" in market_choice:
        custom_raw = st.text_area(
            "Tickers (comma-separated)",
            value="RELIANCE.NS, TCS.NS, INFY.NS, AAPL, MSFT",
            height=80,
            help="NSE: add .NS  |  BSE: add .BO  |  US: plain symbol",
        )
        universe     = [t.strip().upper() for t in custom_raw.split(",") if t.strip()][:30]
        sector_map   = {}
    elif "NSE" in market_choice:
        nse_sectors  = sorted(set(NIFTY50_SECTORS.values()))
        sector_filter= st.multiselect("Sectors", nse_sectors, default=[], placeholder="All sectors")
        universe     = [t for t, s in NIFTY50_SECTORS.items()
                        if not sector_filter or s in sector_filter]
        sector_map   = NIFTY50_SECTORS
    else:
        us_sectors   = sorted(set(SP500_SECTORS.values()))
        sector_filter= st.multiselect("Sectors", us_sectors, default=[], placeholder="All sectors")
        universe     = [t for t, s in SP500_SECTORS.items()
                        if not sector_filter or s in sector_filter]
        sector_map   = SP500_SECTORS

    st.divider()
    st.markdown("**Fundamental Filters**")
    pe_max    = st.slider("Max P/E (TTM)",         0.0, 200.0, 80.0, 5.0)
    pe_min    = st.slider("Min P/E (TTM)",         0.0,  50.0,  0.0, 1.0)
    beta_max  = st.slider("Max Beta",              0.0,   4.0,  3.0, 0.25)
    div_min   = st.slider("Min Div. Yield (%)",    0.0,  10.0,  0.0, 0.5)

    st.divider()
    st.markdown("**Technical Filters**")
    rsi_max     = st.slider("RSI ≤",  0, 100, 100,  5)
    rsi_min     = st.slider("RSI ≥",  0, 100,   0,  5)
    require_ma  = st.selectbox(
        "MA Trend",
        ["Any","Price > EMA20","Price > EMA50",
         "EMA20 > EMA50 (Bullish)","EMA20 < EMA50 (Bearish)"]
    )
    require_sig = st.selectbox(
        "Signal Filter",
        ["Any","BUY only","SELL only","STRONG BUY","NEUTRAL only"]
    )

    st.divider()
    max_tickers = st.slider(
        "Max tickers to scan", 5, len(universe), min(20, len(universe)), 5
    )
    run_scan = st.button("▶  Run Screener", type="primary", use_container_width=True)
    st.caption("⚠️ Scanning many tickers takes time.")

# ─── HEADER ─────────────────────────────────────────────────────────────────
mkt_label = ("NSE" if "NSE" in market_choice
             else ("US" if "US" in market_choice else "Custom"))
st.markdown(f"""<div style="font-family:'IBM Plex Mono',monospace;padding:10px 0 6px;
border-bottom:1px solid #30363D;margin-bottom:16px;">
<span style="font-size:20px;font-weight:600;color:#C9D1D9">Stock Screener</span>&nbsp;&nbsp;
<span style="font-size:13px;color:#8B949E">
  {mkt_label} · Filter by fundamentals &amp; technical signals
</span>
</div>""", unsafe_allow_html=True)

if not run_scan and "screener_results" not in st.session_state:
    # Landing state
    flag = "🇮🇳" if "NSE" in market_choice else ("🇺🇸" if "US" in market_choice else "📋")
    st.markdown(f"""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
    min-height:40vh;text-align:center;padding:40px;">
      <div style="font-size:48px;margin-bottom:16px;">🔍</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:20px;font-weight:600;
                  color:#C9D1D9;margin-bottom:8px;">Stock Screener</div>
      <div style="font-size:13px;color:#8B949E;max-width:540px;line-height:1.7;">
        {flag} Market selected: <b style="color:#3FB950">{mkt_label}</b>
        &nbsp;·&nbsp; {len(universe[:max_tickers])} tickers in scope.<br>
        Set filters in the sidebar then click
        <b style="color:#3FB950">Run Screener</b>.
      </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ─── SCAN ────────────────────────────────────────────────────────────────────
scan_universe = universe[:max_tickers]

if run_scan:
    results  = []
    prog_bar = st.progress(0, text=f"Scanning {len(scan_universe)} tickers…")

    for i, t in enumerate(scan_universe):
        prog_bar.progress((i + 1) / len(scan_universe),
                          text=f"[{i+1}/{len(scan_universe)}]  {t}")
        try:
            df = fetch_ohlcv(t, "6mo", "1d")
            if df.empty or len(df) < 30:
                continue

            info  = fetch_fundamentals(t)
            c     = df["Close"]
            last  = float(c.iloc[-1])
            curr  = info.get("currency", "INR" if t.endswith(".NS") or t.endswith(".BO") else "USD")
            sym   = currency_symbol(curr)

            # ── Fundamental filters ──────────────────────────────────────
            pe = info.get("pe_ttm")
            if pe is not None and not (pe_min <= pe <= pe_max):
                continue
            beta_v = info.get("beta")
            if beta_v is not None and beta_v > beta_max:
                continue
            div_y = (info.get("dividend_yield") or 0) * 100
            if div_y < div_min:
                continue

            # ── Technical filters ────────────────────────────────────────
            rsi_val = float(rsi(c).iloc[-1])
            if not (rsi_min <= rsi_val <= rsi_max):
                continue

            ema20 = float(ema(c, 20).iloc[-1])
            ema50 = float(ema(c, 50).iloc[-1])
            if require_ma == "Price > EMA20"          and last <= ema20: continue
            if require_ma == "Price > EMA50"          and last <= ema50: continue
            if require_ma == "EMA20 > EMA50 (Bullish)" and ema20 <= ema50: continue
            if require_ma == "EMA20 < EMA50 (Bearish)" and ema20 >= ema50: continue

            sig_data  = generate_signals(df)
            composite = sig_data["composite"]
            if require_sig == "BUY only"     and "BUY"  not in composite: continue
            if require_sig == "SELL only"    and "SELL" not in composite: continue
            if require_sig == "STRONG BUY"   and composite != "STRONG BUY": continue
            if require_sig == "NEUTRAL only" and composite != "NEUTRAL":  continue

            # ── Collect metrics ──────────────────────────────────────────
            hist     = float(macd(c)["Hist"].iloc[-1])
            bb_pct   = float(bollinger_bands(c)["BB_%B"].iloc[-1])
            atr_v    = float(atr(df).iloc[-1])
            vr       = float(volume_ratio(df).iloc[-1])
            ret_1d   = float((c.iloc[-1]  - c.iloc[-2])  / c.iloc[-2]  * 100)
            ret_1m   = float((c.iloc[-1]  - c.iloc[-21]) / c.iloc[-21] * 100) if len(c) >= 21 else None
            ret_3m   = float((c.iloc[-1]  - c.iloc[-63]) / c.iloc[-63] * 100) if len(c) >= 63 else None
            vol_ann  = float(c.pct_change().std() * np.sqrt(252) * 100)

            results.append({
                "Ticker":      t,
                "Sector":      sector_map.get(t, info.get("sector", "—")),
                "Price":       f"{sym}{last:,.2f}",
                "Price_raw":   last,
                "Currency":    curr,
                "1D %":        round(ret_1d, 2),
                "1M %":        round(ret_1m, 2) if ret_1m is not None else None,
                "3M %":        round(ret_3m, 2) if ret_3m is not None else None,
                "Ann. Vol %":  round(vol_ann, 1),
                "RSI":         round(rsi_val, 1),
                "MACD Hist":   round(hist, 4),
                "BB %B":       round(bb_pct, 2),
                "Vol Ratio":   round(vr, 2),
                "P/E":         round(pe, 1) if pe else None,
                "Beta":        round(beta_v, 2) if beta_v else None,
                "Div Yield %": round(div_y, 2),
                "Mkt Cap":     info.get("market_cap"),
                "Signal":      composite,
            })
        except Exception:
            continue

    prog_bar.empty()
    st.session_state["screener_results"] = results
    st.session_state["screener_market"]  = mkt_label

# ─── DISPLAY ────────────────────────────────────────────────────────────────
results    = st.session_state.get("screener_results", [])
scanned_mkt = st.session_state.get("screener_market", "")

if not results:
    st.warning("No stocks matched your filters. Try relaxing the criteria.")
    st.stop()

# ── Summary KPIs ────────────────────────────────────────────────────────────
section_header(f"Results — {len(results)} matched ({scanned_mkt})")
buy_count  = sum(1 for r in results if "BUY"  in r["Signal"])
sell_count = sum(1 for r in results if "SELL" in r["Signal"])
avg_rsi    = np.mean([r["RSI"]       for r in results])
avg_vol    = np.mean([r["Ann. Vol %"] for r in results])

kpi_row([
    kpi_card("Matched",       str(len(results)),    f"of {len(scan_universe)} scanned"),
    kpi_card("BUY Signals",   str(buy_count),       f"{buy_count/len(results)*100:.0f}%", "pos"),
    kpi_card("SELL Signals",  str(sell_count),      f"{sell_count/len(results)*100:.0f}%",
             "neg" if sell_count > buy_count else ""),
    kpi_card("Avg RSI",       f"{avg_rsi:.1f}",     "Oversold<30 / Overbought>70",
             "neg" if avg_rsi > 70 else ("pos" if avg_rsi < 30 else "")),
    kpi_card("Avg Ann. Vol",  f"{avg_vol:.1f}%",    ""),
])

# ── Colour helpers ───────────────────────────────────────────────────────────
SIGNAL_COLOURS = {
    "STRONG BUY": "#3FB950", "BUY": "#3FB950",
    "NEUTRAL":    "#8B949E",
    "SELL":       "#F85149", "STRONG SELL": "#F85149",
}

# ── Results table ────────────────────────────────────────────────────────────
rows_html = ""
for row in results:
    sig   = row["Signal"]
    sc    = SIGNAL_COLOURS.get(sig, "#8B949E")
    r1d_c = "#3FB950" if (row["1D %"] and row["1D %"]  >= 0) else "#F85149"
    r1m_c = "#3FB950" if (row["1M %"] and row["1M %"]  >= 0) else "#F85149"
    r3m_c = "#3FB950" if (row["3M %"] and row["3M %"]  >= 0) else "#F85149"
    rsi_c = "#F85149" if row["RSI"] > 70 else ("#3FB950" if row["RSI"] < 30 else "#C9D1D9")
    mhc   = "#3FB950" if row["MACD Hist"] >= 0 else "#F85149"
    pe_s  = f"{row['P/E']:.1f}"  if row["P/E"]  else "—"
    b_s   = f"{row['Beta']:.2f}" if row["Beta"] else "—"
    mc_s  = fmt_large(row["Mkt Cap"]) if row["Mkt Cap"] else "—"
    mkt_flag = "🇮🇳" if row["Ticker"].endswith(".NS") or row["Ticker"].endswith(".BO") else "🇺🇸"

    rows_html += f"""<tr style="border-bottom:1px solid #21262D">
      <td style="padding:7px 10px;font-family:'IBM Plex Mono',monospace;font-size:12px;
                 font-weight:600;color:#C9D1D9">{mkt_flag} {row['Ticker']}</td>
      <td style="padding:7px 10px;font-size:11px;color:#8B949E">{row['Sector']}</td>
      <td style="padding:7px 10px;font-family:'IBM Plex Mono',monospace;font-size:12px;
                 color:#C9D1D9">{row['Price']}</td>
      <td style="padding:7px 10px;font-family:'IBM Plex Mono',monospace;font-size:11px;
                 color:{r1d_c}">{f"{row['1D %']:+.2f}%" if row['1D %'] is not None else "—"}</td>
      <td style="padding:7px 10px;font-family:'IBM Plex Mono',monospace;font-size:11px;
                 color:{r1m_c}">{f"{row['1M %']:+.2f}%" if row['1M %'] is not None else "—"}</td>
      <td style="padding:7px 10px;font-family:'IBM Plex Mono',monospace;font-size:11px;
                 color:{r3m_c}">{f"{row['3M %']:+.2f}%" if row['3M %'] is not None else "—"}</td>
      <td style="padding:7px 10px;font-family:'IBM Plex Mono',monospace;font-size:11px;
                 color:{rsi_c}">{row['RSI']:.1f}</td>
      <td style="padding:7px 10px;font-family:'IBM Plex Mono',monospace;font-size:11px;
                 color:{mhc}">{row['MACD Hist']:+.4f}</td>
      <td style="padding:7px 10px;font-family:'IBM Plex Mono',monospace;font-size:11px;
                 color:#C9D1D9">{row['Ann. Vol %']:.1f}%</td>
      <td style="padding:7px 10px;font-size:11px;color:#8B949E">{pe_s}</td>
      <td style="padding:7px 10px;font-size:11px;color:#8B949E">{b_s}</td>
      <td style="padding:7px 10px;font-size:11px;color:#8B949E">{mc_s}</td>
      <td style="padding:7px 10px">
        <span style="font-family:'IBM Plex Mono',monospace;font-size:10px;font-weight:600;
                     color:{sc};border:1px solid {sc};border-radius:4px;
                     padding:2px 7px;letter-spacing:.06em">{sig}</span>
      </td>
    </tr>"""

ths = ("Ticker","Sector","Price","1D","1M","3M","RSI",
       "MACD Hist","Ann. Vol","P/E","Beta","Mkt Cap","Signal")
headers_html = "".join(
    f'<th style="padding:8px 10px;text-align:left;font-family:\'IBM Plex Mono\',monospace;'
    f'font-size:10px;color:#8B949E;text-transform:uppercase;letter-spacing:.08em;'
    f'white-space:nowrap">{h}</th>' for h in ths
)

st.markdown(f"""
<div style="overflow-x:auto;margin-bottom:20px">
<table style="width:100%;border-collapse:collapse;background:#161B22;
              border:1px solid #30363D;border-radius:8px;overflow:hidden;min-width:900px">
  <thead><tr style="background:#21262D">{headers_html}</tr></thead>
  <tbody>{rows_html}</tbody>
</table>
</div>""", unsafe_allow_html=True)

# ── Distribution charts ───────────────────────────────────────────────────────
section_header("Signal & Sector Distribution")
col1, col2, col3 = st.columns(3)

with col1:
    sig_counts = pd.Series([r["Signal"] for r in results]).value_counts()
    sig_colors = [SIGNAL_COLOURS.get(s, "#8B949E") for s in sig_counts.index]
    fig_sig = go.Figure(go.Bar(
        x=sig_counts.index, y=sig_counts.values,
        marker_color=sig_colors, opacity=0.85,
        text=sig_counts.values, textposition="outside",
        textfont=dict(family="IBM Plex Mono, monospace", size=10, color="#C9D1D9"),
    ))
    fig_sig.update_layout(
        plot_bgcolor="#0D1117", paper_bgcolor="#0D1117", height=280,
        font=dict(family="IBM Plex Mono, monospace", color="#C9D1D9", size=10),
        margin=dict(l=8, r=8, t=30, b=8), showlegend=False,
        xaxis=dict(gridcolor="#21262D"), yaxis=dict(gridcolor="#21262D"),
        title=dict(text="Signal Distribution", font_size=12),
    )
    st.plotly_chart(fig_sig, use_container_width=True, config={"displayModeBar": False})

with col2:
    sec_counts = pd.Series([r["Sector"] for r in results]).value_counts()
    palette = ["#3FB950","#58A6FF","#E3B341","#BC8CFF","#FFA657",
               "#79C0FF","#F85149","#3FB950","#58A6FF","#E3B341",
               "#C9D1D9","#8B949E","#3FB950","#FFA657"]
    fig_sec = go.Figure(go.Pie(
        labels=sec_counts.index, values=sec_counts.values,
        hole=0.5, marker=dict(colors=palette[:len(sec_counts)],
                              line=dict(color="#0D1117", width=2)),
        textfont=dict(family="IBM Plex Mono, monospace", size=9),
        textinfo="label+percent", showlegend=False,
    ))
    fig_sec.update_layout(
        plot_bgcolor="#0D1117", paper_bgcolor="#0D1117", height=280,
        font=dict(family="IBM Plex Mono, monospace", color="#C9D1D9"),
        margin=dict(l=8, r=8, t=30, b=8),
        title=dict(text="Sector Breakdown", font_size=12),
    )
    st.plotly_chart(fig_sec, use_container_width=True, config={"displayModeBar": False})

with col3:
    rsi_vals = [r["RSI"] for r in results]
    fig_rsi_h = go.Figure(go.Histogram(
        x=rsi_vals, nbinsx=20, marker_color="#58A6FF", opacity=0.75))
    fig_rsi_h.add_vline(x=70, line_color="#F85149", line_dash="dot",
                        annotation_text=" 70", annotation_font_color="#F85149")
    fig_rsi_h.add_vline(x=30, line_color="#3FB950", line_dash="dot",
                        annotation_text=" 30", annotation_font_color="#3FB950")
    fig_rsi_h.update_layout(
        plot_bgcolor="#0D1117", paper_bgcolor="#0D1117", height=280,
        font=dict(family="IBM Plex Mono, monospace", color="#C9D1D9", size=10),
        margin=dict(l=8, r=8, t=30, b=8), showlegend=False,
        xaxis=dict(gridcolor="#21262D", title_text="RSI"),
        yaxis=dict(gridcolor="#21262D"),
        title=dict(text="RSI Distribution", font_size=12),
    )
    st.plotly_chart(fig_rsi_h, use_container_width=True, config={"displayModeBar": False})

# ── Opportunity map ───────────────────────────────────────────────────────────
section_header("RSI vs Volatility — Opportunity Map")
fig_sc = go.Figure()
for sig_name, sc in SIGNAL_COLOURS.items():
    sub = [r for r in results if r["Signal"] == sig_name]
    if not sub: continue
    fig_sc.add_trace(go.Scatter(
        x=[r["Ann. Vol %"]  for r in sub],
        y=[r["RSI"]         for r in sub],
        mode="markers+text",
        text=[r["Ticker"].replace(".NS","").replace(".BO","") for r in sub],
        textposition="top center",
        name=sig_name,
        marker=dict(color=sc, size=9, opacity=0.85,
                    line=dict(color="#0D1117", width=1)),
        textfont=dict(family="IBM Plex Mono, monospace", size=9, color="#C9D1D9"),
    ))
fig_sc.add_hline(y=70, line_color="#F85149", line_dash="dot",
                 line_width=1, annotation_text=" Overbought")
fig_sc.add_hline(y=30, line_color="#3FB950", line_dash="dot",
                 line_width=1, annotation_text=" Oversold")
fig_sc.update_layout(
    plot_bgcolor="#0D1117", paper_bgcolor="#0D1117", height=420,
    font=dict(family="IBM Plex Mono, monospace", color="#C9D1D9", size=11),
    margin=dict(l=12, r=12, t=36, b=12),
    xaxis=dict(gridcolor="#21262D", title_text="Annualised Volatility (%)"),
    yaxis=dict(gridcolor="#21262D", title_text="RSI (14)", range=[0, 100]),
    legend=dict(bgcolor="rgba(0,0,0,0)", font_size=10),
    title=dict(text="RSI vs Volatility", font_size=12),
)
st.plotly_chart(fig_sc, use_container_width=True, config={"displayModeBar": False})

# ── Export ───────────────────────────────────────────────────────────────────
section_header("Export Results")
export_cols = ["Ticker","Sector","Price","1D %","1M %","3M %",
               "RSI","MACD Hist","Ann. Vol %","P/E","Beta","Div Yield %","Signal"]
csv = pd.DataFrame([{k: r[k] for k in export_cols} for r in results]).to_csv(index=False)
st.download_button(
    label="⬇  Download CSV",
    data=csv,
    file_name=f"screener_{scanned_mkt.lower()}_results.csv",
    mime="text/csv",
)
