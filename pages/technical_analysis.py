"""Page 1 — Technical Analysis: 25+ indicators, multi-panel charts, pivot levels."""
import streamlit as st, pandas as pd, numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__),".."))
from core.data_fetcher import fetch_ohlcv,validate_ticker,PERIOD_MAP,detect_market,currency_symbol
from core.indicators   import (rsi,macd,bollinger_bands,stochastic,atr,keltner_channels,
                                donchian_channels,adx,williams_r,cci,obv,money_flow_index,
                                chaikin_money_flow,historical_volatility,parabolic_sar,
                                volume_ratio,ema,sma,generate_signals)
from utils.helpers     import (inject_css,section_header,signal_badge,signals_table,
                                esc,kpi_row,kpi_card,fmt_price,sidebar_brand,footer_bar)
from utils.charts      import multi_panel_chart, COLORS, T, BASE
import plotly.graph_objects as go
from plotly.subplots import make_subplots
inject_css()

with st.sidebar:
    sidebar_brand()
    st.divider()
    st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:16px;font-weight:600;color:#3FB950;padding:8px 0 16px;">📈 StockPro<span style="font-size:10px;color:#8B949E;font-weight:400;display:block;letter-spacing:.1em;margin-top:2px;">ANALYTICS TERMINAL</span></div>',unsafe_allow_html=True)
    ticker=st.text_input("Ticker Symbol",value="AAPL",placeholder="AAPL · RELIANCE.NS").upper().strip()
    period_label=st.selectbox("Time Period",list(PERIOD_MAP.keys()),index=3)
    period,interval=PERIOD_MAP[period_label]
    st.divider()
    st.markdown("**Chart Settings**")
    show_rsi=st.checkbox("RSI Panel",value=True); show_macd=st.checkbox("MACD Panel",value=True)
    show_bb=st.checkbox("Bollinger Bands",value=True); show_ema=st.checkbox("EMA Cross 20/50",value=True)
    st.divider()
    st.markdown("**Parameters**")
    rsi_p=st.slider("RSI Period",7,30,14); bb_w=st.slider("BB Window",10,50,20)
    bb_s=st.slider("BB Std Dev",1.0,3.0,2.0,0.5)
    st.divider(); st.caption("Data via Yahoo Finance · Not financial advice")

if not ticker: st.info("Enter a ticker."); st.stop()
with st.spinner(f"Loading {ticker}…"):
    valid,err=validate_ticker(ticker)
    if not valid: st.error(f"**{ticker}** — {err}"); st.stop()
    df=fetch_ohlcv(ticker,period,interval)
if df.empty: st.error("No data returned."); st.stop()

mkt=detect_market(ticker); flag="🇮🇳" if mkt in ("NSE","BSE") else "🇺🇸"; c=df["Close"]
st.markdown(f'<div style="font-family:\'IBM Plex Mono\',monospace;padding:10px 0 6px;border-bottom:1px solid #30363D;margin-bottom:16px"><span style="font-size:20px;font-weight:600;color:#C9D1D9">{esc(ticker)}</span>&nbsp;<span style="font-size:11px;color:#E3B341">{flag} {mkt}</span>&nbsp;&nbsp;<span style="font-size:13px;color:#8B949E">Technical Analysis</span><span style="float:right;font-size:12px;color:#3FB950">{len(df)} sessions · {df.index[0].strftime("%d %b %Y")} → {df.index[-1].strftime("%d %b %Y")}</span></div>',unsafe_allow_html=True)

# KPIs
section_header("Current Indicators")
rsi_v=float(rsi(c,rsi_p).iloc[-1]); macd_h=float(macd(c)["Hist"].iloc[-1])
bb=bollinger_bands(c,bb_w,bb_s); bb_pct=float(bb["BB_%B"].iloc[-1])
atr_v=float(atr(df).iloc[-1]); hv_v=float(historical_volatility(c,20).iloc[-1]*100)
stoch=stochastic(df); k_v=float(stoch["%K"].iloc[-1]); mfi_v=float(money_flow_index(df).iloc[-1])
cci_v=float(cci(df).iloc[-1]); wr_v=float(williams_r(df).iloc[-1]); vr_v=float(volume_ratio(df).iloc[-1])
def rlbl(v): return "OVERSOLD" if v<30 else ("OVERBOUGHT" if v>70 else "NEUTRAL")
kpi_row([kpi_card("RSI",f"{rsi_v:.1f}",rlbl(rsi_v),"neg" if rsi_v>70 else ("pos" if rsi_v<30 else "")),
         kpi_card("MACD Hist",f"{macd_h:+.4f}","","pos" if macd_h>=0 else "neg"),
         kpi_card("BB %B",f"{bb_pct:.2f}","<0 oversold >1 overbought","pos" if bb_pct<0.2 else ("neg" if bb_pct>0.8 else "")),
         kpi_card("ATR (14)",f"{atr_v:.2f}","Avg True Range"),
         kpi_card("Hist. Vol",f"{hv_v:.1f}%","20-day ann."),
         kpi_card("Stoch %K",f"{k_v:.1f}","Stochastic","pos" if k_v<20 else ("neg" if k_v>80 else "")),
         kpi_card("MFI (14)",f"{mfi_v:.1f}","Money Flow","pos" if mfi_v<20 else ("neg" if mfi_v>80 else "")),
         kpi_card("CCI (20)",f"{cci_v:.0f}","","pos" if cci_v<-100 else ("neg" if cci_v>100 else "")),
         kpi_card("Williams%R",f"{wr_v:.1f}","<-80 oversold","pos" if wr_v<-80 else ("neg" if wr_v>-20 else "")),
         kpi_card("Vol Ratio",f"{vr_v:.2f}x","vs 14D avg","pos" if vr_v>1.5 else ""),])

section_header("Price & Indicators")
st.plotly_chart(multi_panel_chart(df,ticker,show_rsi=show_rsi,show_macd=show_macd),
                use_container_width=True,config={"displayModeBar":False})

# Tabs: Volatility, Oscillators, Volume, Trend, Pivots
tabs=st.tabs(["  Volatility  ","  Oscillators  ","  Volume  ","  Trend  ","  Pivot Levels  "])
L={**BASE,"height":300,"margin":dict(l=8,r=8,t=36,b=8)}

with tabs[0]:
    col1,col2=st.columns(2)
    with col1:
        fig=go.Figure()
        bb_=bollinger_bands(c,bb_w,bb_s)
        fig.add_trace(go.Scatter(x=list(bb_.index)+list(bb_.index[::-1]),y=list(bb_["BB_Upper"])+list(bb_["BB_Lower"][::-1]),fill="toself",fillcolor="rgba(88,166,255,0.07)",line=dict(color="rgba(0,0,0,0)"),name="BB Band"))
        for col_,col_v,dash_ in [("BB_Upper",T["blue"],"dot"),("BB_Mid",T["amber"],"solid"),("BB_Lower",T["blue"],"dot")]:
            fig.add_trace(go.Scatter(x=bb_.index,y=bb_[col_],name=col_.replace("BB_","BB "),line=dict(color=col_v,width=1.2,dash=dash_)))
        fig.add_trace(go.Scatter(x=c.index,y=c,name="Close",line=dict(color=T["text"],width=1.5)))
        fig.update_layout(**{**L,"title":dict(text=f"Bollinger Bands ({bb_w},{bb_s})",font_size=12)})
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
    with col2:
        hv20=historical_volatility(c,20)*100; hv50=historical_volatility(c,50)*100
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=hv20.index,y=hv20,name="HV 20",line=dict(color=T["blue"],width=1.5)))
        fig.add_trace(go.Scatter(x=hv50.index,y=hv50,name="HV 50",line=dict(color=T["amber"],width=1.5)))
        fig.update_layout(**{**L,"title":dict(text="Historical Volatility (Ann. %)",font_size=12)})
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

with tabs[1]:
    col1,col2=st.columns(2)
    with col1:
        rsi_s=rsi(c,rsi_p); fig=go.Figure()
        fig.add_hrect(y0=70,y1=100,fillcolor="rgba(248,81,73,0.05)",line_width=0)
        fig.add_hrect(y0=0,y1=30,fillcolor="rgba(63,185,80,0.05)",line_width=0)
        fig.add_trace(go.Scatter(x=rsi_s.index,y=rsi_s,name=f"RSI {rsi_p}",line=dict(color="#79C0FF",width=1.8)))
        for y_,col_ in [(70,T["red"]),(30,T["green"]),(50,T["dim"])]:
            fig.add_hline(y=y_,line_color=col_,line_dash="dot",line_width=1)
        fig.update_yaxes(range=[0,100]); fig.update_layout(**{**L,"title":dict(text=f"RSI ({rsi_p})",font_size=12)})
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
    with col2:
        stch=stochastic(df); fig=go.Figure()
        fig.add_hrect(y0=80,y1=100,fillcolor="rgba(248,81,73,0.05)",line_width=0)
        fig.add_hrect(y0=0,y1=20,fillcolor="rgba(63,185,80,0.05)",line_width=0)
        fig.add_trace(go.Scatter(x=stch.index,y=stch["%K"],name="%K",line=dict(color=T["amber"],width=1.8)))
        fig.add_trace(go.Scatter(x=stch.index,y=stch["%D"],name="%D",line=dict(color=T["purple"],width=1.5,dash="dot")))
        for y_,col_ in [(80,T["red"]),(20,T["green"])]:
            fig.add_hline(y=y_,line_color=col_,line_dash="dot",line_width=1)
        fig.update_yaxes(range=[0,100]); fig.update_layout(**{**L,"title":dict(text="Stochastic (14,3)",font_size=12)})
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

with tabs[2]:
    col1,col2=st.columns(2)
    with col1:
        obv_s=obv(df); fig=go.Figure()
        fig.add_trace(go.Scatter(x=obv_s.index,y=obv_s,name="OBV",line=dict(color=T["blue"],width=1.8),fill="tozeroy",fillcolor="rgba(88,166,255,0.06)"))
        fig.update_layout(**{**L,"title":dict(text="On-Balance Volume",font_size=12)})
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
    with col2:
        mfi_s=money_flow_index(df); fig=go.Figure()
        fig.add_hrect(y0=80,y1=100,fillcolor="rgba(248,81,73,0.05)",line_width=0)
        fig.add_hrect(y0=0,y1=20,fillcolor="rgba(63,185,80,0.05)",line_width=0)
        fig.add_trace(go.Scatter(x=mfi_s.index,y=mfi_s,name="MFI 14",line=dict(color=T["green"],width=1.8)))
        for y_,col_ in [(80,T["red"]),(20,T["green"])]:
            fig.add_hline(y=y_,line_color=col_,line_dash="dot",line_width=1)
        fig.update_yaxes(range=[0,100]); fig.update_layout(**{**L,"title":dict(text="Money Flow Index (14)",font_size=12)})
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

with tabs[3]:
    col1,col2=st.columns(2)
    with col1:
        adx_df=adx(df); fig=go.Figure()
        fig.add_trace(go.Scatter(x=adx_df.index,y=adx_df["ADX"],name="ADX",line=dict(color=T["amber"],width=2)))
        fig.add_trace(go.Scatter(x=adx_df.index,y=adx_df["DI+"],name="+DI",line=dict(color=T["green"],width=1.5)))
        fig.add_trace(go.Scatter(x=adx_df.index,y=adx_df["DI-"],name="-DI",line=dict(color=T["red"],width=1.5)))
        fig.add_hline(y=25,line_color=T["dim"],line_dash="dot",annotation_text=" Trend threshold")
        fig.update_layout(**{**L,"height":320,"title":dict(text="ADX — Trend Strength",font_size=12)})
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
    with col2:
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=c.index,y=c,name="Close",line=dict(color=T["text"],width=1.5)))
        for span_,col_ in [(20,T["green"]),(50,T["blue"]),(200,T["amber"])]:
            if len(df)>=span_:
                fig.add_trace(go.Scatter(x=c.index,y=ema(c,span_),name=f"EMA {span_}",line=dict(color=col_,width=1.2,dash="dot")))
        fig.update_layout(**{**L,"height":320,"title":dict(text="EMA Ribbon",font_size=12)})
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

with tabs[4]:
    H,L,C_p=float(df["High"].iloc[-2]),float(df["Low"].iloc[-2]),float(df["Close"].iloc[-2])
    PP=(H+L+C_p)/3; R1=2*PP-L; S1=2*PP-H; R2=PP+(H-L); S2=PP-(H-L)
    rng=H-L; fR1=PP+0.382*rng; fR2=PP+0.618*rng; fS1=PP-0.382*rng; fS2=PP-0.618*rng
    curr_s=currency_symbol("INR" if detect_market(ticker) in ("NSE","BSE") else "USD")
    col_p,col_f=st.columns(2)
    for col_,title_,levels_ in [
        (col_p,"Classic Pivots",[("Resistance 2",R2,"#3FB950"),("Resistance 1",R1,"#3FB950"),
                                   ("Pivot",PP,"#E3B341"),("Support 1",S1,"#F85149"),("Support 2",S2,"#F85149")]),
        (col_f,"Fibonacci Pivots",[("Fib R2 (61.8%)",fR2,"#3FB950"),("Fib R1 (38.2%)",fR1,"#3FB950"),
                                    ("Pivot",PP,"#E3B341"),("Fib S1 (38.2%)",fS1,"#F85149"),("Fib S2 (61.8%)",fS2,"#F85149")]),
    ]:
        with col_:
            section_header(title_)
            td_k="padding:7px 10px;border-bottom:1px solid #30363D;font-family:'IBM Plex Mono',monospace;font-size:11px"
            rows="".join(
                f'<tr><td style="{td_k};color:{col}">{lbl}</td>'
                f'<td style="{td_k};color:#C9D1D9">{curr_s}{val:.2f}</td>'
                f'<td style="{td_k};color:{"#3FB950" if val>float(c.iloc[-1]) else "#F85149"}">'
                f'{(val-float(c.iloc[-1]))/float(c.iloc[-1])*100:+.2f}%</td></tr>'
                for lbl,val,col in levels_)
            st.markdown(f'<table style="width:100%;border-collapse:collapse;background:#161B22;border:1px solid #30363D;border-radius:6px;overflow:hidden"><thead><tr style="background:#21262D"><th style="{td_k};color:#8B949E;text-transform:uppercase">Level</th><th style="{td_k};color:#8B949E;text-transform:uppercase">Price</th><th style="{td_k};color:#8B949E;text-transform:uppercase">Distance</th></tr></thead><tbody>{rows}</tbody></table>',unsafe_allow_html=True)

section_header("Signal Summary")
sig=generate_signals(df); comp=sig["composite"]
st.markdown(f'<p style="margin-bottom:12px">Composite: {signal_badge(comp)}&nbsp;&nbsp;<span style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;color:#8B949E">{sig["buy_count"]} BUY · {sig["sell_count"]} SELL of 8 indicators</span></p>',unsafe_allow_html=True)
signals_table(sig["indicators"])

footer_bar()
