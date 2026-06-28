"""Page 5 — Stock Screener: NSE Nifty 50 + US S&P 500, fundamental + technical filters."""
import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.data_fetcher import fetch_ohlcv, fetch_fundamentals, currency_symbol
from core.indicators   import rsi, macd, ema, bollinger_bands, generate_signals, volume_ratio
from utils.helpers     import inject_css, section_header, kpi_row, kpi_card, fmt_large, esc
import plotly.graph_objects as go
inject_css()

NIFTY50 = [
    ("RELIANCE.NS","Reliance","Energy"), ("TCS.NS","TCS","IT"),
    ("HDFCBANK.NS","HDFC Bank","Banking"), ("INFY.NS","Infosys","IT"),
    ("ICICIBANK.NS","ICICI Bank","Banking"), ("HINDUNILVR.NS","HUL","FMCG"),
    ("ITC.NS","ITC","FMCG"), ("SBIN.NS","SBI","Banking"),
    ("BHARTIARTL.NS","Airtel","Telecom"), ("KOTAKBANK.NS","Kotak Bank","Banking"),
    ("LT.NS","L&T","Industrials"), ("AXISBANK.NS","Axis Bank","Banking"),
    ("ASIANPAINT.NS","Asian Paints","Consumer"), ("MARUTI.NS","Maruti","Auto"),
    ("HCLTECH.NS","HCL Tech","IT"), ("SUNPHARMA.NS","Sun Pharma","Pharma"),
    ("TITAN.NS","Titan","Consumer"), ("BAJFINANCE.NS","Bajaj Fin","NBFC"),
    ("WIPRO.NS","Wipro","IT"), ("TATAMOTORS.NS","Tata Motors","Auto"),
]
NIFTY_SECTORS = sorted(set(s for _, _, s in NIFTY50))

SP500 = [
    ("AAPL","Apple","Technology"), ("MSFT","Microsoft","Technology"),
    ("NVDA","Nvidia","Technology"), ("GOOGL","Alphabet","Technology"),
    ("META","Meta","Communication"), ("AMZN","Amazon","Consumer"),
    ("TSLA","Tesla","Consumer"), ("JPM","JP Morgan","Financials"),
    ("V","Visa","Financials"), ("XOM","ExxonMobil","Energy"),
    ("UNH","UnitedHlth","Healthcare"), ("JNJ","J&J","Healthcare"),
    ("WMT","Walmart","Staples"), ("HD","Home Depot","Consumer"),
    ("BAC","BofA","Financials"), ("NFLX","Netflix","Communication"),
    ("AMD","AMD","Technology"), ("INTC","Intel","Technology"),
    ("BA","Boeing","Industrials"), ("GS","Goldman","Financials"),
]
US_SECTORS = sorted(set(s for _, _, s in SP500))

SIG_COLS = {"STRONG BUY": "#3FB950", "BUY": "#3FB950", "NEUTRAL": "#8B949E",
            "SELL": "#F85149", "STRONG SELL": "#F85149"}

with st.sidebar:
    sidebar_brand()
    st.divider()
    st.markdown(
        '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:16px;font-weight:600;'
        'color:#3FB950;padding:8px 0 16px;">📈 StockPro'
        '<span style="font-size:10px;color:#8B949E;font-weight:400;display:block;'
        'letter-spacing:.1em;margin-top:2px;">ANALYTICS TERMINAL</span></div>',
        unsafe_allow_html=True)
    market_choice = st.radio("Market", ["🇮🇳 NSE — Nifty 50", "🇺🇸 US — S&P 500", "✏️ Custom"],
                             index=0, label_visibility="collapsed")
    if "Custom" in market_choice:
        raw = st.text_area("Tickers", value="RELIANCE.NS, TCS.NS, AAPL", height=80)
        universe = [(t.strip().upper(), t.strip().upper(), "—")
                    for t in raw.split(",") if t.strip()][:30]
    elif "NSE" in market_choice:
        sf = st.multiselect("Sectors", NIFTY_SECTORS, default=[], placeholder="All")
        universe = [x for x in NIFTY50 if not sf or x[2] in sf]
    else:
        sf = st.multiselect("Sectors", US_SECTORS, default=[], placeholder="All")
        universe = [x for x in SP500 if not sf or x[2] in sf]

    st.divider()
    st.markdown("**Fundamental Filters**")
    pe_max   = st.slider("Max P/E", 0.0, 200.0, 80.0, 5.0)
    pe_min   = st.slider("Min P/E", 0.0, 50.0, 0.0, 1.0)
    beta_max = st.slider("Max Beta", 0.0, 4.0, 3.0, 0.25)
    div_min  = st.slider("Min Div Yield %", 0.0, 10.0, 0.0, 0.5)

    st.divider()
    st.markdown("**Technical Filters**")
    rsi_max = st.slider("RSI ≤", 0, 100, 100, 5)
    rsi_min = st.slider("RSI ≥", 0, 100, 0, 5)
    req_ma  = st.selectbox("MA Trend",
        ["Any", "Price > EMA20", "Price > EMA50",
         "EMA20 > EMA50 (Bullish)", "EMA20 < EMA50 (Bearish)"])
    req_sig = st.selectbox("Signal",
        ["Any", "BUY only", "SELL only", "STRONG BUY", "NEUTRAL only"])

    st.divider()
    max_t = st.slider("Max to scan", 5, len(universe), min(20, len(universe)), 5)
    run_scan = st.button("▶  Run Screener", type="primary", use_container_width=True)
    st.caption("More tickers = slower scan")

mkt_lbl = "NSE" if "NSE" in market_choice else ("US" if "US" in market_choice else "Custom")
st.markdown(
    f'<div style="font-family:\'IBM Plex Mono\',monospace;padding:10px 0 6px;'
    f'border-bottom:1px solid #30363D;margin-bottom:16px">'
    f'<span style="font-size:20px;font-weight:600;color:#C9D1D9">Stock Screener</span>'
    f'&nbsp;&nbsp;<span style="font-size:13px;color:#8B949E">'
    f'{mkt_lbl} · {len(universe[:max_t])} tickers in scope</span></div>',
    unsafe_allow_html=True)

if not run_scan and "screener_results" not in st.session_state:
    st.markdown(
        '<div style="display:flex;flex-direction:column;align-items:center;'
        'justify-content:center;min-height:40vh;text-align:center;padding:40px">'
        '<div style="font-size:48px;margin-bottom:16px">🔍</div>'
        '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:20px;font-weight:600;'
        'color:#C9D1D9;margin-bottom:8px">Stock Screener</div>'
        '<div style="font-size:13px;color:#8B949E;max-width:540px;line-height:1.7">'
        'Set filters in the sidebar then click <b style="color:#3FB950">Run Screener</b>.'
        '</div></div>', unsafe_allow_html=True)
    st.stop()

if run_scan:
    results = []
    scan = universe[:max_t]
    prog = st.progress(0, f"Scanning {len(scan)} tickers…")
    for i, (sym, name, sector) in enumerate(scan):
        prog.progress((i + 1) / len(scan), f"[{i+1}/{len(scan)}] {sym}")
        try:
            df = fetch_ohlcv(sym, "6mo", "1d")
            if df.empty or len(df) < 30:
                continue
            info = fetch_fundamentals(sym)
            c = df["Close"]
            last = float(c.iloc[-1])
            sym_curr = currency_symbol("INR" if sym.endswith((".NS",".BO")) else "USD")
            pe = info.get("pe_ttm")
            bv = info.get("beta")
            dy = (info.get("dividend_yield") or 0) * 100
            if pe is not None and not (pe_min <= pe <= pe_max):
                continue
            if bv is not None and bv > beta_max:
                continue
            if dy < div_min:
                continue
            rv = float(rsi(c).iloc[-1])
            if not (rsi_min <= rv <= rsi_max):
                continue
            e20 = float(ema(c, 20).iloc[-1])
            e50 = float(ema(c, 50).iloc[-1])
            if req_ma == "Price > EMA20" and last <= e20: continue
            if req_ma == "Price > EMA50" and last <= e50: continue
            if req_ma == "EMA20 > EMA50 (Bullish)" and e20 <= e50: continue
            if req_ma == "EMA20 < EMA50 (Bearish)" and e20 >= e50: continue
            sig = generate_signals(df)
            comp = sig["composite"]
            if req_sig == "BUY only" and "BUY" not in comp: continue
            if req_sig == "SELL only" and "SELL" not in comp: continue
            if req_sig == "STRONG BUY" and comp != "STRONG BUY": continue
            if req_sig == "NEUTRAL only" and comp != "NEUTRAL": continue

            mh  = float(macd(c)["Hist"].iloc[-1])
            vr  = float(volume_ratio(df).iloc[-1])
            r1d = float((c.iloc[-1] - c.iloc[-2]) / c.iloc[-2] * 100)
            r1m = float((c.iloc[-1] - c.iloc[-21]) / c.iloc[-21] * 100) if len(c) >= 21 else None
            vol = float(c.pct_change().std() * np.sqrt(252) * 100)
            flag = "🇮🇳" if sym.endswith((".NS",".BO")) else "🇺🇸"

            results.append({
                "Ticker": f"{flag} {sym}", "Name": name, "Sector": sector,
                "Price": f"{sym_curr}{last:,.2f}", "1D %": round(r1d, 2),
                "1M %": round(r1m, 2) if r1m is not None else None,
                "Ann. Vol %": round(vol, 1), "RSI": round(rv, 1),
                "MACD Hist": round(mh, 4), "Vol Ratio": round(vr, 2),
                "P/E": round(pe, 1) if pe else None,
                "Beta": round(bv, 2) if bv else None,
                "Div %": round(dy, 2), "Mkt Cap": info.get("market_cap"),
                "Signal": comp,
            })
        except Exception:
            continue
    prog.empty()
    st.session_state["screener_results"] = results
    st.session_state["screener_mkt"] = mkt_lbl

results = st.session_state.get("screener_results", [])
s_mkt = st.session_state.get("screener_mkt", "")
if not results:
    st.warning("No stocks matched. Try relaxing the filters.")
    st.stop()

section_header(f"Results — {len(results)} matched ({s_mkt})")
buy_n  = sum(1 for r in results if "BUY" in r["Signal"])
sell_n = sum(1 for r in results if "SELL" in r["Signal"])
avg_rsi = np.mean([r["RSI"] for r in results])
avg_vol = np.mean([r["Ann. Vol %"] for r in results])
kpi_row([
    kpi_card("Matched", str(len(results)), f"of {min(max_t,len(universe))} scanned"),
    kpi_card("BUY Signals", str(buy_n), f"{buy_n/len(results)*100:.0f}%", "pos"),
    kpi_card("SELL Signals", str(sell_n), f"{sell_n/len(results)*100:.0f}%",
             "neg" if sell_n > buy_n else ""),
    kpi_card("Avg RSI", f"{avg_rsi:.1f}", ""),
    kpi_card("Avg Ann. Vol", f"{avg_vol:.1f}%", ""),
])

td = "border-bottom:1px solid #21262D;font-family:'IBM Plex Mono',monospace"
rows_h = ""
for r in results:
    sc = SIG_COLS.get(r["Signal"], "#8B949E")
    rc = "#3FB950" if (r["1D %"] or 0) >= 0 else "#F85149"
    pe_str = f"{r['P/E']:.1f}" if r["P/E"] else "—"
    mc_str = fmt_large(r["Mkt Cap"]) if r["Mkt Cap"] else "—"
    rows_h += (
        f'<tr><td style="padding:7px 10px;{td};font-size:12px;font-weight:600;color:#C9D1D9">'
        f'{r["Ticker"]}</td>'
        f'<td style="padding:7px 10px;{td};font-size:11px;color:#8B949E">{esc(r["Sector"])}</td>'
        f'<td style="padding:7px 10px;{td};font-size:12px;color:#C9D1D9">{r["Price"]}</td>'
        f'<td style="padding:7px 10px;{td};font-size:11px;color:{rc}">{r["1D %"]:+.2f}%</td>'
        f'<td style="padding:7px 10px;{td};font-size:11px;color:#C9D1D9">{r["RSI"]:.1f}</td>'
        f'<td style="padding:7px 10px;{td};font-size:11px;color:#8B949E">{pe_str}</td>'
        f'<td style="padding:7px 10px;{td};font-size:11px;color:#8B949E">{mc_str}</td>'
        f'<td style="padding:7px 10px;{td}"><span style="font-family:\'IBM Plex Mono\',monospace;'
        f'font-size:10px;font-weight:600;color:{sc};border:1px solid {sc};border-radius:4px;'
        f'padding:2px 7px">{r["Signal"]}</span></td></tr>')

th_ = ("padding:8px 10px;text-align:left;font-family:'IBM Plex Mono',monospace;"
       "font-size:9px;color:#8B949E;text-transform:uppercase")
st.markdown(
    f'<div style="overflow-x:auto;margin-bottom:20px">'
    f'<table style="width:100%;border-collapse:collapse;background:#161B22;'
    f'border:1px solid #30363D;border-radius:8px;overflow:hidden;min-width:800px">'
    f'<thead><tr style="background:#21262D">'
    f'<th style="{th_}">Ticker</th><th style="{th_}">Sector</th>'
    f'<th style="{th_}">Price</th><th style="{th_}">1D</th>'
    f'<th style="{th_}">RSI</th><th style="{th_}">P/E</th>'
    f'<th style="{th_}">Mkt Cap</th><th style="{th_}">Signal</th>'
    f'</tr></thead><tbody>{rows_h}</tbody></table></div>',
    unsafe_allow_html=True)

section_header("Signal & Sector Distribution")
col1, col2, col3 = st.columns(3)
with col1:
    sc_ = pd.Series([r["Signal"] for r in results]).value_counts()
    fig = go.Figure(go.Bar(x=sc_.index, y=sc_.values,
        marker_color=[SIG_COLS.get(s, "#8B949E") for s in sc_.index], opacity=0.85,
        text=sc_.values, textposition="outside",
        textfont=dict(size=9, family="IBM Plex Mono, monospace", color="#C9D1D9")))
    fig.update_layout(plot_bgcolor="#0D1117", paper_bgcolor="#0D1117", height=260,
        font=dict(family="IBM Plex Mono, monospace", color="#C9D1D9", size=10),
        margin=dict(l=8, r=8, t=36, b=8), showlegend=False,
        xaxis=dict(gridcolor="#21262D"), yaxis=dict(gridcolor="#21262D"),
        title=dict(text="Signal Distribution", font_size=12))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
with col2:
    sec_ = pd.Series([r["Sector"] for r in results]).value_counts()
    palette = ["#3FB950","#58A6FF","#E3B341","#BC8CFF","#FFA657","#79C0FF","#F85149"]
    fig = go.Figure(go.Pie(labels=sec_.index, values=sec_.values, hole=0.5,
        marker=dict(colors=palette[:len(sec_)], line=dict(color="#0D1117", width=2)),
        textfont=dict(family="IBM Plex Mono, monospace", size=9),
        textinfo="label+percent", showlegend=False))
    fig.update_layout(plot_bgcolor="#0D1117", paper_bgcolor="#0D1117", height=260,
        font=dict(family="IBM Plex Mono, monospace", color="#C9D1D9"),
        margin=dict(l=8, r=8, t=36, b=8), title=dict(text="Sector Breakdown", font_size=12))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
with col3:
    fig = go.Figure(go.Histogram(x=[r["RSI"] for r in results], nbinsx=20,
        marker_color="#58A6FF", opacity=0.75))
    fig.add_vline(x=70, line_color="#F85149", line_dash="dot")
    fig.add_vline(x=30, line_color="#3FB950", line_dash="dot")
    fig.update_layout(plot_bgcolor="#0D1117", paper_bgcolor="#0D1117", height=260,
        font=dict(family="IBM Plex Mono, monospace", color="#C9D1D9", size=10),
        margin=dict(l=8, r=8, t=36, b=8), showlegend=False,
        xaxis=dict(gridcolor="#21262D", title_text="RSI"), yaxis=dict(gridcolor="#21262D"),
        title=dict(text="RSI Distribution", font_size=12))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

section_header("Export Results")
csv = pd.DataFrame([{k: r[k] for k in ["Ticker","Sector","Price","1D %","RSI","P/E","Signal"]}
                     for r in results]).to_csv(index=False)
st.download_button("⬇  Download CSV", csv, f"screener_{s_mkt.lower()}.csv", "text/csv")

footer_bar()
