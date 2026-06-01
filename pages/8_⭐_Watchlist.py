"""
Page 8 — Watchlist
Personal watchlist with add/remove tickers, price targets, stop-loss levels,
live P&L tracking, and signal alerts. State persists within the session.
Pure Streamlit — no external AI APIs.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

st.set_page_config(
    page_title="Watchlist · StockPro", page_icon="⭐",
    layout="wide", initial_sidebar_state="expanded"
)

from core.data_fetcher  import fetch_ohlcv, fetch_fundamentals, currency_symbol, detect_market
from core.indicators    import rsi, macd, ema, generate_signals
from core.risk_metrics  import compute_returns, annualised_volatility, var_historical
from utils.helpers      import inject_css, section_header, kpi_row, kpi_card, fmt_pct, fmt_large
import plotly.graph_objects as go

inject_css()

# ─── DEFAULT WATCHLIST ───────────────────────────────────────────────────────
DEFAULT_WATCHLIST = [
    {"ticker": "RELIANCE.NS", "name": "Reliance",   "qty": 10,  "avg_buy": 2800.0, "target": 3200.0, "stop": 2600.0},
    {"ticker": "TCS.NS",      "name": "TCS",         "qty": 5,   "avg_buy": 3700.0, "target": 4200.0, "stop": 3400.0},
    {"ticker": "INFY.NS",     "name": "Infosys",     "qty": 15,  "avg_buy": 1500.0, "target": 1750.0, "stop": 1380.0},
    {"ticker": "HDFCBANK.NS", "name": "HDFC Bank",   "qty": 8,   "avg_buy": 1650.0, "target": 1900.0, "stop": 1520.0},
    {"ticker": "AAPL",        "name": "Apple",       "qty": 10,  "avg_buy": 175.0,  "target": 210.0,  "stop": 160.0},
    {"ticker": "NVDA",        "name": "Nvidia",      "qty": 3,   "avg_buy": 780.0,  "target": 1100.0, "stop": 700.0},
]

# ─── SESSION STATE INIT ──────────────────────────────────────────────────────
if "watchlist" not in st.session_state:
    st.session_state["watchlist"] = DEFAULT_WATCHLIST.copy()

# ─── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style="font-family:'IBM Plex Mono',monospace;font-size:16px;
    font-weight:600;color:#3FB950;padding:8px 0 16px;">📈 StockPro
    <span style="font-size:10px;color:#8B949E;font-weight:400;display:block;
    letter-spacing:.1em;margin-top:2px;">ANALYTICS TERMINAL</span></div>""",
    unsafe_allow_html=True)

    st.markdown("**Add to Watchlist**")

    new_ticker  = st.text_input("Ticker",     placeholder="RELIANCE.NS · AAPL").upper().strip()
    new_name    = st.text_input("Label",      placeholder="Reliance Industries")
    new_qty     = st.number_input("Qty / Units",   value=1,     min_value=0, step=1)
    new_avg_buy = st.number_input("Avg Buy Price", value=0.0,   min_value=0.0, step=0.01, format="%.2f")
    new_target  = st.number_input("Target Price",  value=0.0,   min_value=0.0, step=0.01, format="%.2f",
                                  help="Alert when price reaches this level")
    new_stop    = st.number_input("Stop-Loss",      value=0.0,   min_value=0.0, step=0.01, format="%.2f",
                                  help="Alert when price drops below this level")

    add_btn = st.button("➕  Add to Watchlist", type="primary", use_container_width=True)

    if add_btn and new_ticker:
        existing = [w["ticker"] for w in st.session_state["watchlist"]]
        if new_ticker in existing:
            st.warning(f"{new_ticker} is already in your watchlist.")
        else:
            st.session_state["watchlist"].append({
                "ticker":   new_ticker,
                "name":     new_name or new_ticker,
                "qty":      new_qty,
                "avg_buy":  new_avg_buy,
                "target":   new_target,
                "stop":     new_stop,
            })
            st.success(f"✓ Added {new_ticker}")

    st.divider()

    if st.button("🔄  Reset to defaults", use_container_width=True):
        st.session_state["watchlist"] = DEFAULT_WATCHLIST.copy()

    st.divider()
    st.caption("NSE: add .NS  |  BSE: add .BO  |  US: plain symbol")
    st.caption("State persists within this session.")

# ─── HEADER ─────────────────────────────────────────────────────────────────
wl = st.session_state["watchlist"]

st.markdown(f"""<div style="font-family:'IBM Plex Mono',monospace;padding:10px 0 6px;
border-bottom:1px solid #30363D;margin-bottom:16px;display:flex;
align-items:baseline;justify-content:space-between;">
  <div>
    <span style="font-size:20px;font-weight:600;color:#C9D1D9">Watchlist</span>&nbsp;&nbsp;
    <span style="font-size:13px;color:#8B949E">{len(wl)} positions tracked</span>
  </div>
</div>""", unsafe_allow_html=True)

if not wl:
    st.info("Your watchlist is empty. Add tickers using the sidebar.")
    st.stop()

# ─── LOAD LIVE DATA ──────────────────────────────────────────────────────────
with st.spinner("Refreshing prices…"):
    live_rows = []
    for item in wl:
        t = item["ticker"]
        try:
            df  = fetch_ohlcv(t, "5d", "1d")
            if df.empty or len(df) < 2:
                continue
            last    = float(df["Close"].iloc[-1])
            prev    = float(df["Close"].iloc[-2])
            chg_1d  = (last - prev) / prev * 100
            mkt     = detect_market(t)
            sym     = currency_symbol("INR" if mkt in ("NSE","BSE") else "USD")
            flag    = "🇮🇳" if mkt in ("NSE","BSE") else "🇺🇸"

            # P&L
            avg     = item.get("avg_buy", 0)
            qty     = item.get("qty", 0)
            cost    = avg * qty
            value   = last * qty
            pnl     = value - cost
            pnl_pct = (pnl / cost * 100) if cost > 0 else 0

            # Signals
            df1y = fetch_ohlcv(t, "1y", "1d")
            sig  = generate_signals(df1y) if not df1y.empty else {"composite": "—", "buy_count": 0, "sell_count": 0}
            rsi_v = float(rsi(df1y["Close"]).iloc[-1]) if not df1y.empty else 0

            # Target / stop alerts
            tgt     = item.get("target", 0)
            stp     = item.get("stop", 0)
            at_tgt  = tgt > 0 and last >= tgt
            at_stop = stp > 0 and last <= stp

            live_rows.append({
                "ticker":     t,
                "flag":       flag,
                "name":       item.get("name", t),
                "sym":        sym,
                "last":       last,
                "chg_1d":     chg_1d,
                "qty":        qty,
                "avg_buy":    avg,
                "cost":       cost,
                "value":      value,
                "pnl":        pnl,
                "pnl_pct":    pnl_pct,
                "target":     tgt,
                "stop":       stp,
                "at_tgt":     at_tgt,
                "at_stop":    at_stop,
                "signal":     sig["composite"],
                "rsi":        rsi_v,
                "df":         df1y,
            })
        except Exception:
            continue

if not live_rows:
    st.warning("Could not load data for any watchlist items.")
    st.stop()

# ─── PORTFOLIO SUMMARY KPIS ─────────────────────────────────────────────────
section_header("Portfolio Summary")

total_cost  = sum(r["cost"]  for r in live_rows if r["cost"]  > 0)
total_value = sum(r["value"] for r in live_rows if r["cost"]  > 0)
total_pnl   = total_value - total_cost
total_pnl_p = (total_pnl / total_cost * 100) if total_cost > 0 else 0
alerts      = sum(1 for r in live_rows if r["at_tgt"] or r["at_stop"])
gainers     = sum(1 for r in live_rows if r["chg_1d"] >= 0)
losers      = len(live_rows) - gainers

kpi_row([
    kpi_card("Positions",      str(len(live_rows)),    "tracked"),
    kpi_card("Total Invested", f"{total_cost:,.0f}",   "across all positions"),
    kpi_card("Current Value",  f"{total_value:,.0f}",  "mark-to-market",
             "pos" if total_value >= total_cost else "neg"),
    kpi_card("Unrealised P&L", f"{total_pnl:+,.0f}",  f"{total_pnl_p:+.2f}%",
             "pos" if total_pnl >= 0 else "neg"),
    kpi_card("Today Gainers",  str(gainers),           f"{gainers}/{len(live_rows)}", "pos"),
    kpi_card("Today Losers",   str(losers),            f"{losers}/{len(live_rows)}", "neg" if losers > 0 else ""),
    kpi_card("Active Alerts",  str(alerts),            "target / stop triggered",
             "neg" if alerts > 0 else ""),
])

# ─── ALERTS BANNER ───────────────────────────────────────────────────────────
for r in live_rows:
    if r["at_tgt"]:
        st.markdown(f"""<div style="background:rgba(63,185,80,0.12);border:1px solid #3FB950;
        border-radius:6px;padding:10px 14px;margin-bottom:6px;
        font-family:'IBM Plex Mono',monospace;font-size:12px;color:#3FB950;">
        🎯 <b>{r['flag']} {r['ticker']}</b> — Price {r['sym']}{r['last']:,.2f} has
        reached your target of {r['sym']}{r['target']:,.2f}</div>""",
        unsafe_allow_html=True)
    if r["at_stop"]:
        st.markdown(f"""<div style="background:rgba(248,81,73,0.12);border:1px solid #F85149;
        border-radius:6px;padding:10px 14px;margin-bottom:6px;
        font-family:'IBM Plex Mono',monospace;font-size:12px;color:#F85149;">
        🛑 <b>{r['flag']} {r['ticker']}</b> — Price {r['sym']}{r['last']:,.2f} has
        breached your stop-loss of {r['sym']}{r['stop']:,.2f}</div>""",
        unsafe_allow_html=True)

# ─── WATCHLIST TABLE ─────────────────────────────────────────────────────────
section_header("Live Watchlist")

SIGNAL_COLOURS = {
    "STRONG BUY": "#3FB950", "BUY": "#3FB950",
    "NEUTRAL": "#8B949E",
    "SELL": "#F85149", "STRONG SELL": "#F85149",
}

rows_html = ""
for r in live_rows:
    c1d  = "#3FB950" if r["chg_1d"] >= 0 else "#F85149"
    cpnl = "#3FB950" if r["pnl"]    >= 0 else "#F85149"
    sc   = SIGNAL_COLOURS.get(r["signal"], "#8B949E")
    tgt_str  = f"{r['sym']}{r['target']:,.2f}"  if r["target"] > 0 else "—"
    stop_str = f"{r['sym']}{r['stop']:,.2f}"    if r["stop"]   > 0 else "—"
    qty_str  = str(int(r["qty"])) if r["qty"] > 0 else "—"
    pnl_str  = f"{r['sym']}{r['pnl']:+,.2f}" if r["cost"] > 0 else "—"
    pnlp_str = f"{r['pnl_pct']:+.2f}%"         if r["cost"] > 0 else "—"
    rsi_c    = "#F85149" if r["rsi"] > 70 else ("#3FB950" if r["rsi"] < 30 else "#C9D1D9")

    tgt_badge = ""
    if r["at_tgt"]:
        tgt_badge = '<span style="font-size:9px;color:#3FB950;margin-left:4px">🎯</span>'
    if r["at_stop"]:
        tgt_badge = '<span style="font-size:9px;color:#F85149;margin-left:4px">🛑</span>'

    rows_html += f"""<tr style="border-bottom:1px solid #21262D">
      <td style="padding:8px 10px;font-family:'IBM Plex Mono',monospace;font-size:12px;
                 font-weight:600;color:#C9D1D9">{r['flag']} {r['ticker']}
                 <br><span style="font-size:9px;color:#8B949E;font-weight:400">{r['name']}</span>
      </td>
      <td style="padding:8px 10px;font-family:'IBM Plex Mono',monospace;font-size:13px;
                 font-weight:600;color:#C9D1D9">{r['sym']}{r['last']:,.2f}</td>
      <td style="padding:8px 10px;font-family:'IBM Plex Mono',monospace;font-size:12px;
                 font-weight:600;color:{c1d}">{r['chg_1d']:+.2f}%</td>
      <td style="padding:8px 10px;font-family:'IBM Plex Mono',monospace;font-size:11px;
                 color:#8B949E">{qty_str}</td>
      <td style="padding:8px 10px;font-family:'IBM Plex Mono',monospace;font-size:11px;
                 color:#8B949E">{r['sym']}{r['avg_buy']:,.2f}</td>
      <td style="padding:8px 10px;font-family:'IBM Plex Mono',monospace;font-size:12px;
                 color:{cpnl};font-weight:600">{pnl_str}
                 <br><span style="font-size:10px">{pnlp_str}</span>
      </td>
      <td style="padding:8px 10px;font-family:'IBM Plex Mono',monospace;font-size:11px;
                 color:#8B949E">{tgt_str}{tgt_badge}</td>
      <td style="padding:8px 10px;font-family:'IBM Plex Mono',monospace;font-size:11px;
                 color:#8B949E">{stop_str}</td>
      <td style="padding:8px 10px;font-family:'IBM Plex Mono',monospace;font-size:11px;
                 color:{rsi_c}">{r['rsi']:.0f}</td>
      <td style="padding:8px 10px">
        <span style="font-family:'IBM Plex Mono',monospace;font-size:10px;font-weight:600;
                     color:{sc};border:1px solid {sc};border-radius:4px;
                     padding:2px 7px;letter-spacing:.05em">{r['signal']}</span>
      </td>
    </tr>"""

headers = ["Ticker","Last","1D %","Qty","Avg Buy","P&L","Target","Stop","RSI","Signal"]
headers_html = "".join(
    f'<th style="padding:8px 10px;text-align:left;font-family:\'IBM Plex Mono\',monospace;'
    f'font-size:9px;color:#8B949E;text-transform:uppercase;letter-spacing:.08em;'
    f'white-space:nowrap">{h}</th>' for h in headers
)
st.markdown(f"""
<div style="overflow-x:auto;margin-bottom:20px">
<table style="width:100%;border-collapse:collapse;background:#161B22;
              border:1px solid #30363D;border-radius:8px;overflow:hidden;min-width:800px">
  <thead><tr style="background:#21262D">{headers_html}</tr></thead>
  <tbody>{rows_html}</tbody>
</table>
</div>""", unsafe_allow_html=True)

# ─── REMOVE TICKERS ─────────────────────────────────────────────────────────
section_header("Manage Positions")
col_rm1, col_rm2 = st.columns([2, 1])
with col_rm1:
    all_tickers = [w["ticker"] for w in st.session_state["watchlist"]]
    to_remove   = st.multiselect("Select tickers to remove", all_tickers)
with col_rm2:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️  Remove selected", use_container_width=True) and to_remove:
        st.session_state["watchlist"] = [
            w for w in st.session_state["watchlist"]
            if w["ticker"] not in to_remove
        ]
        st.success(f"Removed: {', '.join(to_remove)}")
        st.rerun()

# ─── MINI SPARKLINES ────────────────────────────────────────────────────────
section_header("Price Sparklines  — 6 Months")

palette = ["#3FB950","#58A6FF","#E3B341","#BC8CFF","#FFA657",
           "#79C0FF","#F85149","#3FB950","#58A6FF","#E3B341"]

# Grid: up to 3 per row
rows_sp = [live_rows[i:i+3] for i in range(0, len(live_rows), 3)]
for row_group in rows_sp:
    cols = st.columns(len(row_group))
    for col, r, colour in zip(cols, row_group, palette):
        with col:
            df_sp = r["df"]
            if df_sp is None or df_sp.empty:
                st.markdown(f"*{r['ticker']} — no data*")
                continue
            c_ser = df_sp["Close"]
            chg   = (float(c_ser.iloc[-1]) / float(c_ser.iloc[0]) - 1) * 100
            line_c = "#3FB950" if chg >= 0 else "#F85149"

            fig_sp = go.Figure()
            fig_sp.add_trace(go.Scatter(
                x=c_ser.index, y=c_ser.values,
                line=dict(color=line_c, width=1.5),
                fill="tozeroy",
                fillcolor=line_c.replace("#3FB950", "rgba(63,185,80,0.07)")
                                 .replace("#F85149", "rgba(248,81,73,0.07)"),
                showlegend=False,
            ))
            # Avg buy line
            if r["avg_buy"] > 0:
                fig_sp.add_hline(
                    y=r["avg_buy"], line_color="#E3B341",
                    line_dash="dot", line_width=1,
                    annotation_text="avg buy",
                    annotation_font_size=8,
                    annotation_font_color="#E3B341",
                )
            # Target line
            if r["target"] > 0:
                fig_sp.add_hline(
                    y=r["target"], line_color="#3FB950",
                    line_dash="dash", line_width=1,
                    annotation_text="target",
                    annotation_font_size=8,
                    annotation_font_color="#3FB950",
                )
            # Stop line
            if r["stop"] > 0:
                fig_sp.add_hline(
                    y=r["stop"], line_color="#F85149",
                    line_dash="dash", line_width=1,
                    annotation_text="stop",
                    annotation_font_size=8,
                    annotation_font_color="#F85149",
                )

            fig_sp.update_layout(
                plot_bgcolor="#0D1117", paper_bgcolor="#161B22",
                height=180, margin=dict(l=6, r=6, t=28, b=6),
                xaxis=dict(gridcolor="#21262D", showticklabels=False,
                           showgrid=False, zeroline=False),
                yaxis=dict(gridcolor="#21262D", showticklabels=True,
                           tickfont=dict(size=8, family="IBM Plex Mono,monospace"),
                           zeroline=False),
                title=dict(
                    text=f"{r['flag']} {r['ticker']}  {chg:+.1f}%",
                    font=dict(size=10, family="IBM Plex Mono,monospace",
                              color="#C9D1D9"),
                    x=0.01,
                ),
            )
            st.plotly_chart(fig_sp, use_container_width=True,
                            config={"displayModeBar": False})

# ─── COMPARATIVE CHART ───────────────────────────────────────────────────────
section_header("Normalised Performance Comparison")

fig_cmp = go.Figure()
for i, r in enumerate(live_rows):
    df_c = r.get("df")
    if df_c is None or df_c.empty or len(df_c) < 2:
        continue
    normed = df_c["Close"] / df_c["Close"].iloc[0] * 100
    colour = palette[i % len(palette)]
    fig_cmp.add_trace(go.Scatter(
        x=normed.index, y=normed.values,
        name=f"{r['flag']} {r['ticker']}",
        line=dict(color=colour, width=1.8),
    ))
fig_cmp.add_hline(y=100, line_color="#8B949E", line_dash="dot", line_width=0.7)
fig_cmp.update_layout(
    plot_bgcolor="#0D1117", paper_bgcolor="#0D1117",
    height=380, margin=dict(l=12, r=12, t=36, b=12),
    font=dict(family="IBM Plex Mono, monospace", color="#C9D1D9", size=11),
    xaxis=dict(gridcolor="#21262D", zeroline=False),
    yaxis=dict(gridcolor="#21262D", zeroline=False, title_text="Indexed (100 = start)"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font_size=10),
    title=dict(text="Watchlist Performance — Indexed to 100 (6-month)", font_size=12),
)
st.plotly_chart(fig_cmp, use_container_width=True, config={"displayModeBar": False})

# ─── RISK OVERVIEW ───────────────────────────────────────────────────────────
section_header("Risk Snapshot")

risk_rows = []
for r in live_rows:
    df_r = r.get("df")
    if df_r is None or df_r.empty or len(df_r) < 30:
        continue
    ret_s  = compute_returns(df_r["Close"])
    vol    = annualised_volatility(ret_s) * 100
    var95  = var_historical(ret_s, 0.95) * 100
    rsi_r  = rsi(df_r["Close"]).iloc[-1]
    risk_rows.append({
        "Ticker":     f"{r['flag']} {r['ticker']}",
        "Ann. Vol %": round(vol, 1),
        "VaR 95%":    round(var95, 2),
        "RSI (14)":   round(rsi_r, 1),
        "Signal":     r["signal"],
        "P&L %":      round(r["pnl_pct"], 2) if r["cost"] > 0 else None,
    })

if risk_rows:
    risk_df = pd.DataFrame(risk_rows).set_index("Ticker")

    # Colour helper
    def colour_signal(val):
        c = {"STRONG BUY": "#3FB950","BUY": "#3FB950",
             "SELL": "#F85149","STRONG SELL": "#F85149"}.get(str(val), "#8B949E")
        return f"color: {c}; font-family: IBM Plex Mono, monospace; font-size:12px"

    def colour_pnl(val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return ""
        c = "#3FB950" if val >= 0 else "#F85149"
        return f"color: {c}; font-family: IBM Plex Mono, monospace; font-size:12px"

    styled = (
        risk_df.style
        .format({
            "Ann. Vol %": "{:.1f}%",
            "VaR 95%":    "{:.2f}%",
            "RSI (14)":   "{:.1f}",
            "P&L %":      lambda v: f"{v:+.2f}%" if v is not None and not (isinstance(v, float) and np.isnan(v)) else "—",
        })
        .applymap(colour_signal, subset=["Signal"])
        .applymap(colour_pnl,    subset=["P&L %"])
        .background_gradient(subset=["Ann. Vol %"],
                             cmap="RdYlGn_r", vmin=0, vmax=60)
        .background_gradient(subset=["RSI (14)"],
                             cmap="RdYlGn",   vmin=20, vmax=80)
    )
    st.dataframe(styled, use_container_width=True)

# ─── EXPORT ─────────────────────────────────────────────────────────────────
section_header("Export Watchlist")
export_data = []
for r in live_rows:
    export_data.append({
        "Ticker":    r["ticker"],
        "Name":      r["name"],
        "Last":      r["last"],
        "1D %":      r["chg_1d"],
        "Qty":       r["qty"],
        "Avg Buy":   r["avg_buy"],
        "P&L":       r["pnl"],
        "P&L %":     r["pnl_pct"],
        "Target":    r["target"],
        "Stop":      r["stop"],
        "RSI":       r["rsi"],
        "Signal":    r["signal"],
    })
csv = pd.DataFrame(export_data).to_csv(index=False)
st.download_button(
    label="⬇  Download Watchlist CSV",
    data=csv,
    file_name="watchlist.csv",
    mime="text/csv",
)
