"""Page 8 — Watchlist: track positions with targets, stop-loss, live P&L, sparklines."""
import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

st.set_page_config(page_title="Watchlist · StockPro", page_icon="⭐", layout="wide")

from core.data_fetcher  import fetch_ohlcv, currency_symbol, detect_market
from core.indicators    import rsi, generate_signals
from core.risk_metrics  import compute_returns, annualised_volatility, var_historical
from utils.helpers      import inject_css, section_header, kpi_row, kpi_card, fmt_pct, esc
from utils.charts       import T, BASE, COLORS
import plotly.graph_objects as go
inject_css()

DEFAULT_WL = [
    {"ticker":"RELIANCE.NS","name":"Reliance",  "qty":10, "avg_buy":2800.0,"target":3200.0,"stop":2600.0},
    {"ticker":"TCS.NS",     "name":"TCS",       "qty":5,  "avg_buy":3700.0,"target":4200.0,"stop":3400.0},
    {"ticker":"INFY.NS",    "name":"Infosys",   "qty":15, "avg_buy":1500.0,"target":1750.0,"stop":1380.0},
    {"ticker":"HDFCBANK.NS","name":"HDFC Bank", "qty":8,  "avg_buy":1650.0,"target":1900.0,"stop":1520.0},
    {"ticker":"AAPL",       "name":"Apple",     "qty":10, "avg_buy":175.0, "target":210.0, "stop":160.0},
    {"ticker":"NVDA",       "name":"Nvidia",    "qty":3,  "avg_buy":780.0, "target":1100.0,"stop":700.0},
]

if "watchlist" not in st.session_state:
    st.session_state["watchlist"] = DEFAULT_WL.copy()

with st.sidebar:
    st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:16px;font-weight:600;color:#3FB950;padding:8px 0 16px;">📈 StockPro<span style="font-size:10px;color:#8B949E;font-weight:400;display:block;letter-spacing:.1em;margin-top:2px;">ANALYTICS TERMINAL</span></div>', unsafe_allow_html=True)
    st.markdown("**Add Position**")
    new_t  = st.text_input("Ticker",     placeholder="RELIANCE.NS · AAPL").upper().strip()
    new_n  = st.text_input("Label",      placeholder="Display name")
    new_q  = st.number_input("Qty",      value=1,   min_value=0, step=1)
    new_b  = st.number_input("Avg Buy",  value=0.0, min_value=0.0, step=0.01, format="%.2f")
    new_tg = st.number_input("Target",   value=0.0, min_value=0.0, step=0.01, format="%.2f", help="Alert when reached")
    new_sl = st.number_input("Stop-Loss",value=0.0, min_value=0.0, step=0.01, format="%.2f", help="Alert when breached")
    if st.button("➕  Add", type="primary", use_container_width=True) and new_t:
        existing=[w["ticker"] for w in st.session_state["watchlist"]]
        if new_t in existing: st.warning(f"{new_t} already in watchlist.")
        else:
            st.session_state["watchlist"].append({"ticker":new_t,"name":new_n or new_t,"qty":new_q,"avg_buy":new_b,"target":new_tg,"stop":new_sl})
            st.success(f"✓ Added {new_t}")
    st.divider()
    if st.button("🔄  Reset defaults", use_container_width=True):
        st.session_state["watchlist"] = DEFAULT_WL.copy(); st.rerun()
    st.divider()
    st.caption("NSE: add .NS  |  BSE: add .BO  |  US: plain symbol")

wl = st.session_state["watchlist"]
st.markdown(
    f'<div style="font-family:\'IBM Plex Mono\',monospace;padding:10px 0 6px;'
    f'border-bottom:1px solid #30363D;margin-bottom:16px">'
    f'<span style="font-size:20px;font-weight:600;color:#C9D1D9">Watchlist</span>'
    f'&nbsp;&nbsp;<span style="font-size:13px;color:#8B949E">{len(wl)} positions tracked</span>'
    f'</div>', unsafe_allow_html=True)

if not wl:
    st.info("Your watchlist is empty. Add tickers using the sidebar.")
    st.stop()

# ── Load live data ─────────────────────────────────────────────────────────
with st.spinner("Refreshing prices…"):
    live = []
    for item in wl:
        t = item["ticker"]
        try:
            df = fetch_ohlcv(t,"6mo","1d")
            if df.empty or len(df)<2: continue
            last=float(df["Close"].iloc[-1]); prev=float(df["Close"].iloc[-2])
            chg_1d=(last-prev)/prev*100
            mkt=detect_market(t); sym=currency_symbol("INR" if mkt in ("NSE","BSE") else "USD")
            flag="🇮🇳" if mkt in ("NSE","BSE") else "🇺🇸"
            avg=item.get("avg_buy",0); qty=item.get("qty",0)
            cost=avg*qty; value=last*qty; pnl=value-cost
            pnl_pct=(pnl/cost*100) if cost>0 else 0
            sig=generate_signals(df); rsi_v=float(rsi(df["Close"]).iloc[-1])
            tgt=item.get("target",0); stp=item.get("stop",0)
            live.append({"ticker":t,"flag":flag,"name":item.get("name",t),"sym":sym,
                "last":last,"chg_1d":chg_1d,"qty":qty,"avg_buy":avg,
                "cost":cost,"value":value,"pnl":pnl,"pnl_pct":pnl_pct,
                "target":tgt,"stop":stp,"at_tgt":tgt>0 and last>=tgt,
                "at_stop":stp>0 and last<=stp,"signal":sig["composite"],
                "rsi":rsi_v,"df":df})
        except: continue

if not live: st.warning("Could not load data."); st.stop()

# ── Summary KPIs ───────────────────────────────────────────────────────────
section_header("Portfolio Summary")
tot_cost=sum(r["cost"]  for r in live if r["cost"]>0)
tot_val =sum(r["value"] for r in live if r["cost"]>0)
tot_pnl =tot_val-tot_cost; tot_pnl_p=(tot_pnl/tot_cost*100) if tot_cost>0 else 0
alerts  =sum(1 for r in live if r["at_tgt"] or r["at_stop"])
gainers =sum(1 for r in live if r["chg_1d"]>=0)
kpi_row([
    kpi_card("Positions",     str(len(live)),      "tracked"),
    kpi_card("Current Value", f"{tot_val:,.0f}",   f"Invested {tot_cost:,.0f}"),
    kpi_card("Unrealised P&L",f"{tot_pnl:+,.0f}",f"{tot_pnl_p:+.2f}%","pos" if tot_pnl>=0 else "neg"),
    kpi_card("Today Gainers", str(gainers),         f"of {len(live)}","pos"),
    kpi_card("Today Losers",  str(len(live)-gainers),f"of {len(live)}","neg" if (len(live)-gainers)>0 else ""),
    kpi_card("Active Alerts", str(alerts),          "target/stop","neg" if alerts>0 else ""),
])

# ── Alert banners ──────────────────────────────────────────────────────────
for r in live:
    if r["at_tgt"]:
        st.markdown(f'<div style="background:rgba(63,185,80,0.12);border:1px solid #3FB950;border-radius:6px;padding:10px 14px;margin-bottom:6px;font-family:\'IBM Plex Mono\',monospace;font-size:12px;color:#3FB950">🎯 <b>{r["flag"]} {esc(r["ticker"])}</b> — {r["sym"]}{r["last"]:,.2f} reached target {r["sym"]}{r["target"]:,.2f}</div>',unsafe_allow_html=True)
    if r["at_stop"]:
        st.markdown(f'<div style="background:rgba(248,81,73,0.12);border:1px solid #F85149;border-radius:6px;padding:10px 14px;margin-bottom:6px;font-family:\'IBM Plex Mono\',monospace;font-size:12px;color:#F85149">🛑 <b>{r["flag"]} {esc(r["ticker"])}</b> — {r["sym"]}{r["last"]:,.2f} breached stop {r["sym"]}{r["stop"]:,.2f}</div>',unsafe_allow_html=True)

# ── Watchlist table ────────────────────────────────────────────────────────
section_header("Live Watchlist")
SIG_COL={"STRONG BUY":"#3FB950","BUY":"#3FB950","NEUTRAL":"#8B949E","SELL":"#F85149","STRONG SELL":"#F85149"}
td="border-bottom:1px solid #21262D;font-family:'IBM Plex Mono',monospace"
rows_h=""
for r in live:
    c1d="#3FB950" if r["chg_1d"]>=0 else "#F85149"; cpnl="#3FB950" if r["pnl"]>=0 else "#F85149"
    sc=SIG_COL.get(r["signal"],"#8B949E")
    tgt_s=f'{r["sym"]}{r["target"]:,.2f}' if r["target"]>0 else "—"
    stp_s=f'{r["sym"]}{r["stop"]:,.2f}'   if r["stop"]>0   else "—"
    badge="🎯" if r["at_tgt"] else ("🛑" if r["at_stop"] else "")
    rsi_c="#F85149" if r["rsi"]>70 else ("#3FB950" if r["rsi"]<30 else "#C9D1D9")
    rows_h+=(f'<tr style="border-bottom:1px solid #21262D">'
             f'<td style="padding:8px 10px;{td};font-size:12px;font-weight:600;color:#C9D1D9">{r["flag"]} {esc(r["ticker"])}<br><span style="font-size:9px;color:#8B949E;font-weight:400">{esc(r["name"])}</span></td>'
             f'<td style="padding:8px 10px;{td};font-size:13px;font-weight:600;color:#C9D1D9">{r["sym"]}{r["last"]:,.2f}</td>'
             f'<td style="padding:8px 10px;{td};font-size:12px;font-weight:600;color:{c1d}">{r["chg_1d"]:+.2f}%</td>'
             f'<td style="padding:8px 10px;{td};font-size:11px;color:#8B949E">{int(r["qty"]) if r["qty"]>0 else "—"}</td>'
             f'<td style="padding:8px 10px;{td};font-size:11px;color:#8B949E">{r["sym"]}{r["avg_buy"]:,.2f}</td>'
             f'<td style="padding:8px 10px;{td};font-size:12px;color:{cpnl};font-weight:600">{r["sym"]}{r["pnl"]:+,.0f}<br><span style="font-size:10px">{r["pnl_pct"]:+.2f}%</span></td>'
             f'<td style="padding:8px 10px;{td};font-size:11px;color:#8B949E">{tgt_s} {badge}</td>'
             f'<td style="padding:8px 10px;{td};font-size:11px;color:#8B949E">{stp_s}</td>'
             f'<td style="padding:8px 10px;{td};font-size:11px;color:{rsi_c}">{r["rsi"]:.0f}</td>'
             f'<td style="padding:8px 10px;{td}"><span style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;font-weight:600;color:{sc};border:1px solid {sc};border-radius:4px;padding:2px 7px">{r["signal"]}</span></td>'
             f'</tr>')
th="padding:8px 10px;text-align:left;font-family:'IBM Plex Mono',monospace;font-size:9px;color:#8B949E;text-transform:uppercase;white-space:nowrap"
st.markdown(f'<div style="overflow-x:auto;margin-bottom:20px"><table style="width:100%;border-collapse:collapse;background:#161B22;border:1px solid #30363D;border-radius:8px;overflow:hidden;min-width:800px"><thead><tr style="background:#21262D"><th style="{th}">Ticker</th><th style="{th}">Last</th><th style="{th}">1D%</th><th style="{th}">Qty</th><th style="{th}">Avg Buy</th><th style="{th}">P&amp;L</th><th style="{th}">Target</th><th style="{th}">Stop</th><th style="{th}">RSI</th><th style="{th}">Signal</th></tr></thead><tbody>{rows_h}</tbody></table></div>',unsafe_allow_html=True)

# ── Remove positions ───────────────────────────────────────────────────────
section_header("Manage Positions")
col_rm1,col_rm2=st.columns([2,1])
with col_rm1:
    to_remove=st.multiselect("Remove tickers",[w["ticker"] for w in st.session_state["watchlist"]])
with col_rm2:
    st.markdown("<br>",unsafe_allow_html=True)
    if st.button("🗑️  Remove selected",use_container_width=True) and to_remove:
        st.session_state["watchlist"]=[w for w in st.session_state["watchlist"] if w["ticker"] not in to_remove]
        st.rerun()

# ── Sparklines ─────────────────────────────────────────────────────────────
section_header("Price Sparklines — 6 Months")
rows_sp=[live[i:i+3] for i in range(0,len(live),3)]
for row_g in rows_sp:
    cols=st.columns(len(row_g))
    for col,r,colour in zip(cols,row_g,COLORS):
        with col:
            df_sp=r.get("df"); 
            if df_sp is None or df_sp.empty: st.markdown(f"*{r['ticker']} — no data*"); continue
            c_ser=df_sp["Close"]; chg=(float(c_ser.iloc[-1])/float(c_ser.iloc[0])-1)*100
            lc="#3FB950" if chg>=0 else "#F85149"
            fig_sp=go.Figure()
            fig_sp.add_trace(go.Scatter(x=c_ser.index,y=c_ser.values,line=dict(color=lc,width=1.5),fill="tozeroy",fillcolor=f"rgba({'63,185,80' if chg>=0 else '248,81,73'},0.07)",showlegend=False))
            if r["avg_buy"]>0: fig_sp.add_hline(y=r["avg_buy"],line_color="#E3B341",line_dash="dot",line_width=1,annotation_text="avg",annotation_font_size=8,annotation_font_color="#E3B341")
            if r["target"]>0: fig_sp.add_hline(y=r["target"],line_color="#3FB950",line_dash="dash",line_width=1,annotation_text="tgt",annotation_font_size=8,annotation_font_color="#3FB950")
            if r["stop"]>0:   fig_sp.add_hline(y=r["stop"],  line_color="#F85149",line_dash="dash",line_width=1,annotation_text="stp",annotation_font_size=8,annotation_font_color="#F85149")
            fig_sp.update_layout(plot_bgcolor="#0D1117",paper_bgcolor="#161B22",height=180,margin=dict(l=6,r=6,t=28,b=6),
                xaxis=dict(gridcolor="#21262D",showticklabels=False,showgrid=False,zeroline=False),
                yaxis=dict(gridcolor="#21262D",tickfont=dict(size=8,family="IBM Plex Mono,monospace"),zeroline=False),
                title=dict(text=f"{r['flag']} {esc(r['ticker'])}  {chg:+.1f}%",font=dict(size=10,family="IBM Plex Mono,monospace",color="#C9D1D9"),x=0.01))
            st.plotly_chart(fig_sp,use_container_width=True,config={"displayModeBar":False})

# ── Comparison chart ────────────────────────────────────────────────────────
section_header("Normalised Performance Comparison")
fig_cmp=go.Figure()
for i,r in enumerate(live):
    df_c=r.get("df")
    if df_c is None or df_c.empty or len(df_c)<2: continue
    normed=df_c["Close"]/df_c["Close"].iloc[0]*100
    fig_cmp.add_trace(go.Scatter(x=normed.index,y=normed.values,name=f"{r['flag']} {r['ticker']}",line=dict(color=COLORS[i%len(COLORS)],width=1.8)))
fig_cmp.add_hline(y=100,line_color=T["dim"],line_dash="dot",line_width=0.7)
fig_cmp.update_layout(**{**BASE,"height":360,"title":dict(text="Watchlist Performance (Indexed to 100)",font_size=12)},yaxis_title="Indexed (100 = start)")
st.plotly_chart(fig_cmp,use_container_width=True,config={"displayModeBar":False})

# ── Export ─────────────────────────────────────────────────────────────────
section_header("Export")
csv=pd.DataFrame([{"Ticker":r["ticker"],"Name":r["name"],"Last":r["last"],"1D %":r["chg_1d"],"Qty":r["qty"],"Avg Buy":r["avg_buy"],"P&L":r["pnl"],"P&L %":r["pnl_pct"],"Target":r["target"],"Stop":r["stop"],"RSI":r["rsi"],"Signal":r["signal"]} for r in live]).to_csv(index=False)
st.download_button("⬇  Download Watchlist CSV", csv, "watchlist.csv", "text/csv")
