"""Page 7 — Market Overview: global indices, NSE/US movers, sector heatmaps, VIX."""
import streamlit as st
import pandas as pd
import numpy as np
import sys, os
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.data_fetcher import fetch_ohlcv, currency_symbol
from utils.helpers     import inject_css, section_header, esc, sidebar_brand, footer_bar
from utils.charts      import T, BASE
import plotly.graph_objects as go
inject_css()

GLOBAL_INDICES = [
    ("Nifty 50",     "^NSEI",   "INR", "🇮🇳 India"),
    ("Bank Nifty",   "^NSEBANK","INR", "🇮🇳 India"),
    ("Sensex",       "^BSESN",  "INR", "🇮🇳 India"),
    ("S&P 500",      "^GSPC",   "USD", "🇺🇸 US"),
    ("Nasdaq 100",   "^NDX",    "USD", "🇺🇸 US"),
    ("Dow Jones",    "^DJI",    "USD", "🇺🇸 US"),
    ("FTSE 100",     "^FTSE",   "GBP", "🇬🇧 UK"),
    ("DAX",          "^GDAXI",  "EUR", "🇩🇪 Germany"),
    ("Nikkei 225",   "^N225",   "JPY", "🇯🇵 Japan"),
    ("Hang Seng",    "^HSI",    "HKD", "🇭🇰 HK"),
    ("VIX",          "^VIX",    "USD", "🌐 Volatility"),
]

NSE_MOVERS = [
    ("RELIANCE.NS","Reliance","Energy"),("TCS.NS","TCS","IT"),
    ("HDFCBANK.NS","HDFC Bank","Banking"),("INFY.NS","Infosys","IT"),
    ("ICICIBANK.NS","ICICI Bank","Banking"),("HINDUNILVR.NS","HUL","FMCG"),
    ("ITC.NS","ITC","FMCG"),("SBIN.NS","SBI","Banking"),
    ("BHARTIARTL.NS","Airtel","Telecom"),("KOTAKBANK.NS","Kotak Bank","Banking"),
    ("LT.NS","L&T","Industrials"),("AXISBANK.NS","Axis Bank","Banking"),
    ("MARUTI.NS","Maruti","Auto"),("HCLTECH.NS","HCL Tech","IT"),
    ("SUNPHARMA.NS","Sun Pharma","Pharma"),("TITAN.NS","Titan","Consumer"),
    ("BAJFINANCE.NS","Bajaj Fin","NBFC"),("WIPRO.NS","Wipro","IT"),
    ("TATAMOTORS.NS","Tata Motors","Auto"),("JSWSTEEL.NS","JSW Steel","Metals"),
]
US_MOVERS = [
    ("AAPL","Apple","Technology"),("MSFT","Microsoft","Technology"),
    ("NVDA","Nvidia","Technology"),("GOOGL","Alphabet","Technology"),
    ("META","Meta","Communication"),("AMZN","Amazon","Consumer"),
    ("TSLA","Tesla","Consumer"),("JPM","JP Morgan","Financials"),
    ("V","Visa","Financials"),("XOM","ExxonMobil","Energy"),
    ("UNH","UnitedHlth","Healthcare"),("JNJ","J&J","Healthcare"),
    ("WMT","Walmart","Staples"),("HD","Home Depot","Consumer"),
    ("BAC","BofA","Financials"),("NFLX","Netflix","Communication"),
    ("AMD","AMD","Technology"),("INTC","Intel","Technology"),
    ("BA","Boeing","Industrials"),("GS","Goldman","Financials"),
]
NSE_SECTORS = ["IT","Banking","FMCG","Pharma","Auto","Energy","Industrials","NBFC","Consumer","Telecom","Metals"]
US_SECTORS  = ["Technology","Financials","Healthcare","Consumer","Communication","Energy","Industrials","Staples"]

with st.sidebar:
    sidebar_brand()
    st.divider()
    st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:16px;font-weight:600;color:#3FB950;padding:8px 0 16px;">📈 StockPro<span style="font-size:10px;color:#8B949E;font-weight:400;display:block;letter-spacing:.1em;margin-top:2px;">ANALYTICS TERMINAL</span></div>', unsafe_allow_html=True)
    market_focus = st.radio("Market Focus", ["🌍 Global","🇮🇳 India (NSE)","🇺🇸 US"], index=0)
    period_label = st.selectbox("Chart Period", ["1 Month","3 Months","6 Months","1 Year"], index=2)
    period_map   = {"1 Month":"1mo","3 Months":"3mo","6 Months":"6mo","1 Year":"1y"}
    period       = period_map[period_label]
    n_movers     = st.slider("Top movers to show", 5, 20, 10, 5)
    st.divider()
    st.caption("Data via Yahoo Finance · Not financial advice")

now_str = datetime.utcnow().strftime("%d %b %Y  %H:%M UTC")
st.markdown(
    f'<div style="font-family:\'IBM Plex Mono\',monospace;padding:10px 0 6px;'
    f'border-bottom:1px solid #30363D;margin-bottom:16px;display:flex;'
    f'align-items:baseline;justify-content:space-between">'
    f'<div><span style="font-size:20px;font-weight:600;color:#C9D1D9">Market Overview</span>'
    f'&nbsp;&nbsp;<span style="font-size:13px;color:#8B949E">{market_focus}</span></div>'
    f'<span style="font-size:11px;color:#8B949E">Updated: {now_str}</span>'
    f'</div>', unsafe_allow_html=True)

# ── Index cards ────────────────────────────────────────────────────────────
section_header("Global Indices")

@st.cache_data(ttl=300, show_spinner=False)
def load_idx(sym):
    try:
        df = fetch_ohlcv(sym,"5d","1d")
        if df.empty or len(df)<2: return {}
        last=float(df["Close"].iloc[-1]); prev=float(df["Close"].iloc[-2])
        return {"last":last,"chg":(last-prev)/prev*100,"series":df["Close"]}
    except: return {}

idx_data = {(n,s,c,r): load_idx(s) for n,s,c,r in GLOBAL_INDICES}
cards_html = ""
for (name,sym,curr,region), d in idx_data.items():
    if not d: continue
    last=d["last"]; chg=d["chg"]; sym_c=currency_symbol(curr)
    arrow="▲" if chg>=0 else "▼"; col="#3FB950" if chg>=0 else "#F85149"
    fmt=f"{last:,.0f}" if last>999 else f"{last:,.2f}"
    cards_html += (
        f'<div style="background:#161B22;border:1px solid #30363D;border-radius:8px;'
        f'padding:12px 14px;min-width:135px;border-top:2px solid {col}">'
        f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:9px;color:#8B949E;'
        f'text-transform:uppercase;letter-spacing:.07em;margin-bottom:4px">{esc(region)}</div>'
        f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:12px;font-weight:600;'
        f'color:#C9D1D9;margin-bottom:6px">{esc(name)}</div>'
        f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:16px;font-weight:600;'
        f'color:#C9D1D9">{fmt}</div>'
        f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:12px;color:{col};margin-top:3px">'
        f'{arrow} {chg:+.2f}%</div></div>')

st.markdown(f'<div style="display:flex;flex-wrap:wrap;gap:10px;margin-bottom:20px">{cards_html}</div>', unsafe_allow_html=True)

# ── Performance chart ───────────────────────────────────────────────────────
section_header(f"Index Performance — {period_label} (Indexed to 100)")
if "India" in market_focus:
    chart_idx=[("Nifty 50","^NSEI","#3FB950"),("Bank Nifty","^NSEBANK","#58A6FF"),("Sensex","^BSESN","#E3B341")]
elif "US" in market_focus:
    chart_idx=[("S&P 500","^GSPC","#3FB950"),("Nasdaq 100","^NDX","#58A6FF"),("Dow Jones","^DJI","#E3B341"),("Russell 2000","^RUT","#BC8CFF")]
else:
    chart_idx=[("Nifty 50","^NSEI","#3FB950"),("S&P 500","^GSPC","#58A6FF"),("Nasdaq","^NDX","#E3B341"),("FTSE 100","^FTSE","#BC8CFF"),("Nikkei","^N225","#FFA657")]

@st.cache_data(ttl=300, show_spinner=False)
def load_series(sym, period):
    try:
        df=fetch_ohlcv(sym,period,"1d"); return df["Close"] if not df.empty else pd.Series(dtype=float)
    except: return pd.Series(dtype=float)

fig_idx=go.Figure()
for name,sym,colour in chart_idx:
    s=load_series(sym,period)
    if s.empty or len(s)<2: continue
    normed=s/s.iloc[0]*100
    fig_idx.add_trace(go.Scatter(x=normed.index,y=normed.values,name=name,line=dict(color=colour,width=2)))
fig_idx.add_hline(y=100,line_color=T["dim"],line_dash="dot",line_width=0.8)
fig_idx.update_layout(**{**BASE,"height":380,"title":dict(text="Performance Indexed to 100",font_size=12)},yaxis_title="Indexed (100 = start)")
st.plotly_chart(fig_idx,use_container_width=True,config={"displayModeBar":False})

# ── Top movers ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def get_movers(watchlist, period, n):
    rows=[]
    for sym,name,sector in watchlist:
        try:
            df=fetch_ohlcv(sym,period,"1d")
            if df.empty or len(df)<2: continue
            last=float(df["Close"].iloc[-1]); prev=float(df["Close"].iloc[-2])
            chg_1d=(last-prev)/prev*100
            chg_p=(last/float(df["Close"].iloc[0])-1)*100
            vr=float(df["Volume"].iloc[-1]/df["Volume"].mean()) if df["Volume"].mean()>0 else 1
            rows.append({"Symbol":sym,"Name":name,"Sector":sector,
                         "Last":last,"1D %":round(chg_1d,2),"Period %":round(chg_p,2),"VolR":round(vr,2)})
        except: continue
    return pd.DataFrame(rows).sort_values("1D %",ascending=False) if rows else pd.DataFrame()

tabs_m=st.tabs(["  🇮🇳 NSE Movers  ","  🇺🇸 US Movers  ","  📊 Sector Heatmaps  "])

def movers_table(df_m, sym_prefix="₹"):
    if df_m.empty: st.info("Could not load data."); return
    top_g=df_m.head(n_movers//2); top_l=df_m.tail(n_movers//2).sort_values("1D %")
    col1,col2=st.columns(2)
    for col,title,subdf,tc in [(col1,"🟢 Top Gainers",top_g,"#3FB950"),(col2,"🔴 Top Losers",top_l,"#F85149")]:
        with col:
            st.markdown(f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;color:{tc};text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px">{title}</div>',unsafe_allow_html=True)
            td="border-bottom:1px solid #21262D;font-family:'IBM Plex Mono',monospace"
            rows="".join(f'<tr><td style="padding:6px 10px;{td};font-size:11px;font-weight:600;color:#C9D1D9">{esc(r["Name"])}</td><td style="padding:6px 10px;{td};font-size:10px;color:#8B949E">{esc(r["Sector"])}</td><td style="padding:6px 10px;{td};font-size:11px;color:#C9D1D9">{sym_prefix}{r["Last"]:,.2f}</td><td style="padding:6px 10px;{td};font-size:12px;font-weight:600;color:{"#3FB950" if r["1D %"]>=0 else "#F85149"}">{r["1D %"]:+.2f}%</td><td style="padding:6px 10px;{td};font-size:10px;color:#8B949E">{r["VolR"]:.1f}x</td></tr>' for _,r in subdf.iterrows())
            th="padding:7px 10px;text-align:left;font-family:'IBM Plex Mono',monospace;font-size:9px;color:#8B949E;text-transform:uppercase"
            st.markdown(f'<table style="width:100%;border-collapse:collapse;background:#161B22;border:1px solid #30363D;border-radius:6px;overflow:hidden"><thead><tr style="background:#21262D"><th style="{th}">Name</th><th style="{th}">Sector</th><th style="{th}">Price</th><th style="{th}">1D</th><th style="{th}">Vol</th></tr></thead><tbody>{rows}</tbody></table>',unsafe_allow_html=True)

    # Bar chart
    top_n=df_m.head(n_movers).sort_values("1D %")
    bar_c=["#3FB950" if v>=0 else "#F85149" for v in top_n["1D %"]]
    fig=go.Figure(go.Bar(x=top_n["1D %"],y=top_n["Name"],orientation="h",marker_color=bar_c,opacity=0.85,text=[f"{v:+.2f}%" for v in top_n["1D %"]],textposition="outside",textfont=dict(size=9,family="IBM Plex Mono, monospace",color="#C9D1D9")))
    fig.add_vline(x=0,line_color=T["dim"],line_dash="dot",line_width=0.8)
    from utils.charts import safe_layout
    fig.update_layout(**safe_layout(
        {"xaxis_title": "1D Return (%)", "margin": dict(l=8,r=60,t=36,b=8)},
        height=320, title="1-Day Returns (%)"))
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

with tabs_m[0]:
    section_header("NSE — Nifty 50 Movers")
    with st.spinner("Loading NSE data…"):
        nse_df=get_movers(NSE_MOVERS,period,n_movers)
    movers_table(nse_df,"₹")

with tabs_m[1]:
    section_header("US — S&P 500 Sample Movers")
    with st.spinner("Loading US data…"):
        us_df=get_movers(US_MOVERS,period,n_movers)
    movers_table(us_df,"$")

with tabs_m[2]:
    section_header("Sector Performance Heatmaps")
    def sector_heatmap(watchlist, sectors, title):
        sector_ret={s:[] for s in sectors}
        for sym,name,sector in watchlist:
            try:
                df=fetch_ohlcv(sym,"5d","1d")
                if df.empty or len(df)<2: continue
                chg=(float(df["Close"].iloc[-1])-float(df["Close"].iloc[-2]))/float(df["Close"].iloc[-2])*100
                if sector in sector_ret: sector_ret[sector].append(chg)
            except: continue
        avgs={s:np.mean(v) for s,v in sector_ret.items() if v}
        if not avgs: return None
        labs,vals=list(avgs.keys()),list(avgs.values())
        fc=["rgba(63,185,80,0.8)" if v>=0 else "rgba(248,81,73,0.8)" for v in vals]
        fig=go.Figure(go.Bar(x=vals,y=labs,orientation="h",marker_color=fc,text=[f"{v:+.2f}%" for v in vals],textposition="outside",textfont=dict(size=10,family="IBM Plex Mono, monospace",color="#C9D1D9")))
        fig.add_vline(x=0,line_color=T["dim"],line_dash="dot",line_width=0.8)
        from utils.charts import safe_layout
        fig.update_layout(**safe_layout(
            {"xaxis_title": "Avg 1D Return (%)", "margin": dict(l=8,r=70,t=36,b=8)},
            height=360, title=title))
        return fig

    col1,col2=st.columns(2)
    with col1:
        with st.spinner("NSE heatmap…"):
            fig_n=sector_heatmap(NSE_MOVERS,NSE_SECTORS,"NSE Sector Avg 1D Return (%)")
        if fig_n: st.plotly_chart(fig_n,use_container_width=True,config={"displayModeBar":False})
        else: st.info("Insufficient data for NSE heatmap.")
    with col2:
        with st.spinner("US heatmap…"):
            fig_u=sector_heatmap(US_MOVERS,US_SECTORS,"US Sector Avg 1D Return (%)")
        if fig_u: st.plotly_chart(fig_u,use_container_width=True,config={"displayModeBar":False})
        else: st.info("Insufficient data for US heatmap.")

# ── VIX ────────────────────────────────────────────────────────────────────
section_header("Volatility Pulse")
col_v1,col_v2=st.columns(2)
with col_v1:
    vix_s=load_series("^VIX",period)
    if not vix_s.empty:
        vix_now=float(vix_s.iloc[-1])
        vix_col="#F85149" if vix_now>25 else ("#E3B341" if vix_now>18 else "#3FB950")
        vix_lbl="High Fear" if vix_now>25 else ("Elevated" if vix_now>18 else "Low")
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=vix_s.index,y=vix_s.values,name="VIX",line=dict(color=vix_col,width=2),fill="tozeroy",fillcolor=f"rgba({','.join(str(int(vix_col.lstrip('#')[i:i+2],16)) for i in (0,2,4))},0.08)"))
        fig.add_hline(y=25,line_color="#F85149",line_dash="dot",line_width=1,annotation_text=" Fear >25")
        fig.add_hline(y=18,line_color="#E3B341",line_dash="dot",line_width=1,annotation_text=" Elevated >18")
        fig.update_layout(**{**BASE,"height":280,"title":dict(text=f"CBOE VIX — {vix_now:.1f} ({vix_lbl})",font_size=12)})
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

with col_v2:
    nsei_s=load_series("^NSEI",period)
    if not nsei_s.empty:
        nsei_hv=nsei_s.pct_change().rolling(20).std()*np.sqrt(252)*100
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=nsei_hv.index,y=nsei_hv.values,name="Nifty 20D HV",line=dict(color=T["amber"],width=2),fill="tozeroy",fillcolor="rgba(227,179,65,0.07)"))
        fig.update_layout(**{**BASE,"height":280,"title":dict(text="Nifty Historical Volatility 20D (Ann. %)",font_size=12)})
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

footer_bar()
