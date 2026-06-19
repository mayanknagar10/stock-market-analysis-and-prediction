"""Professional Plotly chart library — Bloomberg dark terminal theme."""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, Dict

T = dict(
    bg="#0D1117", card="#161B22", grid="#21262D", text="#C9D1D9",
    dim="#8B949E", green="#3FB950", green2="#26A641", red="#F85149",
    blue="#58A6FF", amber="#E3B341", purple="#BC8CFF", orange="#FFA657",
)
COLORS = [T["green"],T["blue"],T["amber"],T["purple"],T["orange"],
          "#79C0FF","#F0883E","#D29922","#58A6FF","#3FB950"]

BASE = dict(
    template="plotly_dark", plot_bgcolor=T["bg"], paper_bgcolor=T["bg"],
    font=dict(family="IBM Plex Mono, monospace", color=T["text"], size=11),
    margin=dict(l=12,r=12,t=40,b=12),
    legend=dict(bgcolor="rgba(0,0,0,0)",font_size=10,bordercolor=T["grid"],borderwidth=1),
    xaxis=dict(gridcolor=T["grid"],zeroline=False,showspikes=True,spikecolor=T["dim"]),
    yaxis=dict(gridcolor=T["grid"],zeroline=False,showspikes=True,spikecolor=T["dim"]),
    hoverlabel=dict(bgcolor=T["card"],font_size=12,font_family="IBM Plex Mono, monospace"),
)

def _apply(fig, title="", height=500):
    layout = {**BASE,"height":height}
    if title: layout["title"]=dict(text=title,font_size=13,x=0.01,xanchor="left")
    fig.update_layout(**layout); return fig


def candlestick_chart(df, ticker, overlays:Optional[Dict]=None, volume=True, height=580):
    rows=2 if volume else 1; rh=[0.72,0.28] if volume else [1.0]
    fig=make_subplots(rows=rows,cols=1,shared_xaxes=True,
                      vertical_spacing=0.02,row_heights=rh)
    fig.add_trace(go.Candlestick(x=df.index,open=df["Open"],high=df["High"],
        low=df["Low"],close=df["Close"],name=ticker,
        increasing_line_color=T["green"],decreasing_line_color=T["red"],
        increasing_fillcolor=T["green"],decreasing_fillcolor=T["red"],
        line_width=1),row=1,col=1)
    if overlays:
        for i,(name,series) in enumerate(overlays.items()):
            dash="dash" if "BB" in name else "solid"
            fig.add_trace(go.Scatter(x=series.index,y=series.values,name=name,
                line=dict(color=COLORS[i%len(COLORS)],width=1.5,dash=dash),opacity=0.85),row=1,col=1)
    if volume:
        clrs=np.where(df["Close"]>=df["Open"],T["green2"],T["red"])
        fig.add_trace(go.Bar(x=df.index,y=df["Volume"],name="Volume",
            marker_color=list(clrs),opacity=0.6,showlegend=False),row=2,col=1)
        fig.update_yaxes(title_text="Volume",row=2,col=1,tickfont_size=9)
    fig.update_xaxes(rangeslider_visible=False)
    return _apply(fig,f"{ticker} — Price & Volume",height)


def multi_panel_chart(df, ticker, show_rsi=True, show_macd=True, height=750):
    from core.indicators import rsi, macd, bollinger_bands, ema
    n=1+int(show_rsi)+int(show_macd)
    rh=([0.55]+[0.225]*(n-1)) if n>1 else [1.0]
    fig=make_subplots(rows=n,cols=1,shared_xaxes=True,vertical_spacing=0.025,row_heights=rh)
    c=df["Close"]
    fig.add_trace(go.Candlestick(x=df.index,open=df["Open"],high=df["High"],
        low=df["Low"],close=df["Close"],name="OHLC",
        increasing_line_color=T["green"],decreasing_line_color=T["red"],
        increasing_fillcolor=T["green"],decreasing_fillcolor=T["red"]),row=1,col=1)
    bb=bollinger_bands(c)
    for col,colour,fill in [("BB_Upper",T["blue"],True),("BB_Lower",T["blue"],None),("BB_Mid",T["amber"],None)]:
        fig.add_trace(go.Scatter(x=bb.index,y=bb[col],
            name=col.replace("BB_","BB "),line=dict(color=colour,width=1,dash="dot"),
            fill="tonexty" if col=="BB_Upper" else None,
            fillcolor="rgba(88,166,255,0.04)",showlegend=col!="BB_Upper"),row=1,col=1)
    for span,colour in [(20,T["amber"]),(50,T["purple"])]:
        fig.add_trace(go.Scatter(x=c.index,y=ema(c,span),name=f"EMA {span}",
            line=dict(color=colour,width=1.5)),row=1,col=1)
    row=2
    if show_rsi:
        rv=rsi(c)
        fig.add_trace(go.Scatter(x=rv.index,y=rv,name="RSI 14",
            line=dict(color="#79C0FF",width=1.5)),row=row,col=1)
        for y_,col_ in [(70,T["red"]),(30,T["green"])]:
            fig.add_hline(y=y_,line_color=col_,line_dash="dot",line_width=1,row=row,col=1)
        fig.update_yaxes(title_text="RSI",range=[0,100],row=row,col=1,tickfont_size=9)
        row+=1
    if show_macd:
        md=macd(c)
        fig.add_trace(go.Scatter(x=md.index,y=md["MACD"],name="MACD",
            line=dict(color=T["blue"],width=1.5)),row=row,col=1)
        fig.add_trace(go.Scatter(x=md.index,y=md["Signal"],name="Signal",
            line=dict(color=T["orange"],width=1.5)),row=row,col=1)
        hc=np.where(md["Hist"]>=0,T["green2"],T["red"])
        fig.add_trace(go.Bar(x=md.index,y=md["Hist"],name="Histogram",
            marker_color=list(hc),opacity=0.6,showlegend=False),row=row,col=1)
        fig.update_yaxes(title_text="MACD",row=row,col=1,tickfont_size=9)
    fig.update_xaxes(rangeslider_visible=False)
    return _apply(fig,f"{ticker} — Technical Analysis",height)


def prediction_chart(historical, forecast_df, backtest_df=None, ticker="", height=500):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=historical.index,y=historical.values,
        name="Historical",line=dict(color=T["text"],width=1.5)))
    if backtest_df is not None and not backtest_df.empty and "Predicted" in backtest_df.columns:
        aligned=backtest_df["Predicted"].reindex(historical.index).dropna()
        if not aligned.empty:
            fig.add_trace(go.Scatter(x=aligned.index,y=aligned.values,
                name="Backtest Fit",line=dict(color=T["amber"],width=1,dash="dot"),opacity=0.8))
    if "Upper_80" in forecast_df.columns:
        fig.add_trace(go.Scatter(
            x=list(forecast_df.index)+list(forecast_df.index[::-1]),
            y=list(forecast_df["Upper_80"])+list(forecast_df["Lower_80"][::-1]),
            fill="toself",fillcolor="rgba(63,185,80,0.10)",
            line=dict(color="rgba(0,0,0,0)"),name="80% CI (GBM cone)"))
        for col,name in [("Upper_80","Upper 80%"),("Lower_80","Lower 80%")]:
            fig.add_trace(go.Scatter(x=forecast_df.index,y=forecast_df[col],
                name=name,line=dict(color=T["green"],width=1,dash="dot"),opacity=0.7))
    fig.add_trace(go.Scatter(x=forecast_df.index,y=forecast_df["Forecast"],
        name="Forecast",line=dict(color=T["green"],width=2.5),
        mode="lines+markers",marker=dict(size=5,color=T["green"])))
    fig.add_vline(x=str(historical.index[-1]),line_color=T["dim"],
                  line_dash="dot",line_width=1,
                  annotation_text=" Today",annotation_font_color=T["dim"],annotation_font_size=10)
    return _apply(fig,f"{ticker} — Price Forecast",height)


def returns_distribution(returns, ticker, height=380):
    fig=go.Figure()
    fig.add_trace(go.Histogram(x=returns*100,nbinsx=60,
        name="Daily Returns",marker_color=T["blue"],opacity=0.75))
    mu=returns.mean()*100; sig=returns.std()*100
    x=np.linspace(mu-4*sig,mu+4*sig,300)
    from scipy.stats import norm
    pdf=norm.pdf(x,mu,sig)*len(returns)*(8*sig/60)
    fig.add_trace(go.Scatter(x=x,y=pdf,name="Normal Fit",
        line=dict(color=T["amber"],width=2)))
    fig.add_vline(x=0,line_color=T["dim"],line_dash="dot",line_width=1)
    fig.update_xaxes(title_text="Daily Return (%)")
    fig.update_yaxes(title_text="Count")
    return _apply(fig,f"{ticker} — Return Distribution",height)


def drawdown_chart(dd_series, ticker, height=280):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=dd_series.index,y=dd_series*100,
        fill="tozeroy",fillcolor="rgba(248,81,73,0.25)",
        line=dict(color=T["red"],width=1.5),name="Drawdown"))
    fig.update_yaxes(title_text="Drawdown (%)")
    return _apply(fig,f"{ticker} — Drawdown",height)


def monte_carlo_chart(historical, sim_df, ticker, height=440):
    fig=go.Figure()
    sample=min(150,sim_df.shape[1])
    for col in sim_df.columns[:sample]:
        fig.add_trace(go.Scatter(x=sim_df.index,y=sim_df[col],mode="lines",
            line=dict(width=0.4,color="rgba(88,166,255,0.12)"),showlegend=False))
    p5=sim_df.quantile(0.05,axis=1); p50=sim_df.quantile(0.50,axis=1); p95=sim_df.quantile(0.95,axis=1)
    fig.add_trace(go.Scatter(
        x=list(sim_df.index)+list(sim_df.index[::-1]),
        y=list(p95)+list(p5[::-1]),fill="toself",
        fillcolor="rgba(88,166,255,0.08)",line=dict(color="rgba(0,0,0,0)"),name="5–95th pct"))
    fig.add_trace(go.Scatter(x=sim_df.index,y=p50,name="Median",
        line=dict(color=T["amber"],width=2)))
    fig.add_trace(go.Scatter(x=historical.index[-30:],y=historical.values[-30:],
        name="Historical",line=dict(color=T["text"],width=2)))
    return _apply(fig,f"{ticker} — Monte Carlo ({sim_df.shape[1]} paths)",height)


def correlation_heatmap(returns_df, height=420):
    corr=returns_df.corr()
    fig=go.Figure(go.Heatmap(z=corr.values,x=corr.columns,y=corr.index,
        colorscale=[[0,T["red"]],[0.5,T["card"]],[1,T["green"]]],
        zmid=0,zmin=-1,zmax=1,text=np.round(corr.values,2),
        texttemplate="%{text}",colorbar=dict(tickfont_size=9)))
    return _apply(fig,"Correlation Matrix — Daily Returns",height)


def portfolio_performance_chart(perf_df, height=420):
    fig=go.Figure()
    for i,col in enumerate(perf_df.columns):
        normed=perf_df[col]/perf_df[col].iloc[0]*100
        fig.add_trace(go.Scatter(x=normed.index,y=normed,name=col,
            line=dict(color=COLORS[i%len(COLORS)],width=2)))
    fig.add_hline(y=100,line_color=T["dim"],line_dash="dot",line_width=1)
    fig.update_yaxes(title_text="Indexed to 100")
    return _apply(fig,"Portfolio Performance (Indexed)",height)
