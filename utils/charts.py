"""
Professional charting utilities.
All charts use a Bloomberg-inspired dark terminal theme with consistent styling.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, List


# ─── THEME ───────────────────────────────────────────────────────────────────

THEME = {
    "bg":          "#0D1117",
    "bg_card":     "#161B22",
    "bg_paper":    "#0D1117",
    "grid":        "#21262D",
    "text":        "#C9D1D9",
    "text_dim":    "#8B949E",
    "green":       "#3FB950",
    "green_light": "#26A641",
    "red":         "#F85149",
    "blue":        "#58A6FF",
    "amber":       "#E3B341",
    "purple":      "#BC8CFF",
    "orange":      "#FFA657",
    "cyan":        "#79C0FF",
}

LAYOUT_BASE = dict(
    template="plotly_dark",
    plot_bgcolor  = THEME["bg"],
    paper_bgcolor = THEME["bg_paper"],
    font          = dict(family="IBM Plex Mono, monospace", color=THEME["text"], size=11),
    margin        = dict(l=12, r=12, t=40, b=12),
    legend        = dict(bgcolor="rgba(0,0,0,0)", font_size=10,
                         bordercolor=THEME["grid"], borderwidth=1),
    xaxis         = dict(gridcolor=THEME["grid"], zeroline=False,
                         showspikes=True, spikecolor=THEME["text_dim"]),
    yaxis         = dict(gridcolor=THEME["grid"], zeroline=False,
                         showspikes=True, spikecolor=THEME["text_dim"]),
    hoverlabel    = dict(bgcolor=THEME["bg_card"], font_size=12,
                         font_family="IBM Plex Mono, monospace"),
)


def _apply_theme(fig: go.Figure, title: str = "", height: int = 500) -> go.Figure:
    layout = {**LAYOUT_BASE, "height": height}
    if title:
        layout["title"] = dict(text=title, font_size=13, x=0.01, xanchor="left")
    fig.update_layout(**layout)
    return fig


# ─── CANDLESTICK CHART ───────────────────────────────────────────────────────

def candlestick_chart(
    df: pd.DataFrame,
    ticker: str,
    overlays: Optional[dict] = None,     # {"SMA 20": series, ...}
    volume: bool = True,
    height: int = 580,
) -> go.Figure:
    """
    Full OHLCV candlestick chart with optional MA overlays and volume subplot.
    """
    rows = 2 if volume else 1
    row_heights = [0.72, 0.28] if volume else [1.0]

    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                        vertical_spacing=0.02, row_heights=row_heights)

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name=ticker,
        increasing_line_color=THEME["green"],
        decreasing_line_color=THEME["red"],
        increasing_fillcolor=THEME["green"],
        decreasing_fillcolor=THEME["red"],
        line_width=1,
    ), row=1, col=1)

    # Overlays (moving averages, Bollinger, etc.)
    overlay_colours = [THEME["amber"], THEME["blue"], THEME["purple"],
                       THEME["orange"], THEME["cyan"]]
    if overlays:
        for i, (name, series) in enumerate(overlays.items()):
            dash = "dash" if "BB" in name else "solid"
            width = 1 if "BB" in name else 1.5
            fig.add_trace(go.Scatter(
                x=series.index, y=series.values, name=name,
                line=dict(color=overlay_colours[i % len(overlay_colours)],
                          width=width, dash=dash),
                opacity=0.85,
            ), row=1, col=1)

    # Volume
    if volume:
        colours = np.where(df["Close"] >= df["Open"],
                           THEME["green_light"], THEME["red"])
        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"], name="Volume",
            marker_color=list(colours), opacity=0.7,
            showlegend=False,
        ), row=2, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1,
                         tickfont_size=9)

    fig.update_xaxes(rangeslider_visible=False)
    return _apply_theme(fig, f"{ticker} — Price & Volume", height)


# ─── INDICATOR PANELS ────────────────────────────────────────────────────────

def multi_panel_chart(
    df: pd.DataFrame,
    ticker: str,
    show_rsi: bool = True,
    show_macd: bool = True,
    height: int = 750,
) -> go.Figure:
    """Candlestick + RSI + MACD multi-panel chart."""
    from core.indicators import rsi, macd, bollinger_bands, ema

    n_panels = 1 + int(show_rsi) + int(show_macd)
    heights  = ([0.55] + [0.225] * (n_panels - 1)) if n_panels > 1 else [1.0]
    fig = make_subplots(rows=n_panels, cols=1, shared_xaxes=True,
                        vertical_spacing=0.025, row_heights=heights)

    # Panel 1: candlestick + overlays
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="OHLC",
        increasing_line_color=THEME["green"],  decreasing_line_color=THEME["red"],
        increasing_fillcolor=THEME["green"],   decreasing_fillcolor=THEME["red"],
    ), row=1, col=1)

    # Bollinger Bands
    bb = bollinger_bands(df["Close"])
    for col, colour, fill in [
        ("BB_Upper", THEME["blue"],   "toself"),
        ("BB_Lower", THEME["blue"],   None),
        ("BB_Mid",   THEME["amber"],  None),
    ]:
        fig.add_trace(go.Scatter(
            x=bb.index, y=bb[col], name=col.replace("BB_", "BB "),
            line=dict(color=colour, width=1, dash="dot"),
            fill="tonexty" if col == "BB_Upper" else None,
            fillcolor="rgba(88,166,255,0.04)",
            showlegend=col != "BB_Upper",
        ), row=1, col=1)

    # EMA 20 / 50
    for span, colour in [(20, THEME["amber"]), (50, THEME["purple"])]:
        fig.add_trace(go.Scatter(
            x=df.index, y=ema(df["Close"], span), name=f"EMA {span}",
            line=dict(color=colour, width=1.5),
        ), row=1, col=1)

    row = 2
    # RSI panel
    if show_rsi:
        rsi_vals = rsi(df["Close"])
        fig.add_trace(go.Scatter(x=rsi_vals.index, y=rsi_vals,
                                 name="RSI 14", line=dict(color=THEME["cyan"], width=1.5)),
                      row=row, col=1)
        fig.add_hline(y=70, line_color=THEME["red"],    line_dash="dot", line_width=1, row=row, col=1)
        fig.add_hline(y=30, line_color=THEME["green"],  line_dash="dot", line_width=1, row=row, col=1)
        fig.add_hline(y=50, line_color=THEME["text_dim"], line_dash="dot", line_width=0.5, row=row, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=row, col=1, tickfont_size=9)
        row += 1

    # MACD panel
    if show_macd:
        macd_df = macd(df["Close"])
        fig.add_trace(go.Scatter(x=macd_df.index, y=macd_df["MACD"],
                                 name="MACD", line=dict(color=THEME["blue"], width=1.5)),
                      row=row, col=1)
        fig.add_trace(go.Scatter(x=macd_df.index, y=macd_df["Signal"],
                                 name="Signal", line=dict(color=THEME["orange"], width=1.5)),
                      row=row, col=1)
        hist_colours = np.where(macd_df["Hist"] >= 0, THEME["green_light"], THEME["red"])
        fig.add_trace(go.Bar(x=macd_df.index, y=macd_df["Hist"],
                             name="Histogram", marker_color=list(hist_colours),
                             opacity=0.6, showlegend=False),
                      row=row, col=1)
        fig.update_yaxes(title_text="MACD", row=row, col=1, tickfont_size=9)

    fig.update_xaxes(rangeslider_visible=False)
    return _apply_theme(fig, f"{ticker} — Technical Analysis", height)


# ─── PREDICTION CHART ────────────────────────────────────────────────────────

def prediction_chart(
    historical: pd.Series,
    forecast_df: pd.DataFrame,
    backtest_df: Optional[pd.DataFrame] = None,
    ticker: str = "",
    height: int = 500,
) -> go.Figure:
    fig = go.Figure()

    # Historical close
    fig.add_trace(go.Scatter(
        x=historical.index, y=historical.values,
        name="Historical",
        line=dict(color=THEME["text"], width=1.5),
    ))

    # Backtest predictions
    if backtest_df is not None and not backtest_df.empty:
        fig.add_trace(go.Scatter(
            x=backtest_df.index, y=backtest_df["Predicted"],
            name="Backtest Fit",
            line=dict(color=THEME["amber"], width=1, dash="dot"),
        ))

    # Forecast band
    if "Upper_80" in forecast_df.columns and "Lower_80" in forecast_df.columns:
        fig.add_trace(go.Scatter(
            x=list(forecast_df.index) + list(forecast_df.index[::-1]),
            y=list(forecast_df["Upper_80"]) + list(forecast_df["Lower_80"][::-1]),
            fill="toself", fillcolor="rgba(63,185,80,0.12)",
            line=dict(color="rgba(0,0,0,0)"), name="80% CI", showlegend=True,
        ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=forecast_df.index, y=forecast_df["Forecast"],
        name="Forecast",
        line=dict(color=THEME["green"], width=2.5, dash="dash"),
        mode="lines+markers",
        marker=dict(size=5, color=THEME["green"]),
    ))

    # Vertical divider
    fig.add_vline(x=str(historical.index[-1]), line_color=THEME["text_dim"],
                  line_dash="dot", line_width=1)

    return _apply_theme(fig, f"{ticker} — Price Forecast", height)


# ─── RISK CHARTS ─────────────────────────────────────────────────────────────

def returns_distribution(returns: pd.Series, ticker: str, height: int = 380) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=returns * 100, nbinsx=60, name="Daily Returns",
        marker_color=THEME["blue"], opacity=0.75,
    ))
    mu  = returns.mean() * 100
    sig = returns.std()  * 100
    x   = np.linspace(mu - 4*sig, mu + 4*sig, 300)
    from scipy.stats import norm
    pdf = norm.pdf(x, mu, sig) * len(returns) * (8 * sig / 60)
    fig.add_trace(go.Scatter(x=x, y=pdf, name="Normal Fit",
                             line=dict(color=THEME["amber"], width=2)))
    fig.add_vline(x=0, line_color=THEME["text_dim"], line_dash="dot", line_width=1)
    fig.update_xaxes(title_text="Daily Return (%)")
    fig.update_yaxes(title_text="Count")
    return _apply_theme(fig, f"{ticker} — Return Distribution", height)


def drawdown_chart(dd_series: pd.Series, ticker: str, height: int = 280) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd_series.index, y=dd_series * 100,
        fill="tozeroy", fillcolor="rgba(248,81,73,0.25)",
        line=dict(color=THEME["red"], width=1.5), name="Drawdown",
    ))
    fig.update_yaxes(title_text="Drawdown (%)")
    return _apply_theme(fig, f"{ticker} — Drawdown", height)


def monte_carlo_chart(
    historical: pd.Series,
    sim_df: pd.DataFrame,
    ticker: str,
    height: int = 440,
) -> go.Figure:
    fig = go.Figure()

    # Show a subset of paths
    sample = min(150, sim_df.shape[1])
    for col in sim_df.columns[:sample]:
        fig.add_trace(go.Scatter(
            x=sim_df.index, y=sim_df[col],
            mode="lines", line=dict(width=0.4, color=f"rgba(88,166,255,0.12)"),
            showlegend=False,
        ))

    # Percentile bands
    p5  = sim_df.quantile(0.05, axis=1)
    p50 = sim_df.quantile(0.50, axis=1)
    p95 = sim_df.quantile(0.95, axis=1)

    fig.add_trace(go.Scatter(
        x=list(sim_df.index) + list(sim_df.index[::-1]),
        y=list(p95) + list(p5[::-1]),
        fill="toself", fillcolor="rgba(88,166,255,0.08)",
        line=dict(color="rgba(0,0,0,0)"), name="5–95th percentile",
    ))
    fig.add_trace(go.Scatter(x=sim_df.index, y=p50, name="Median",
                             line=dict(color=THEME["amber"], width=2)))
    fig.add_trace(go.Scatter(
        x=historical.index[-30:], y=historical.values[-30:],
        name="Historical", line=dict(color=THEME["text"], width=2),
    ))
    return _apply_theme(fig, f"{ticker} — Monte Carlo ({sim_df.shape[1]} paths)", height)


def correlation_heatmap(returns_df: pd.DataFrame, height: int = 420) -> go.Figure:
    corr = returns_df.corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale=[
            [0.0, THEME["red"]],
            [0.5, THEME["bg_card"]],
            [1.0, THEME["green"]],
        ],
        zmid=0, zmin=-1, zmax=1,
        text=np.round(corr.values, 2), texttemplate="%{text}",
        colorbar=dict(tickfont_size=9),
    ))
    return _apply_theme(fig, "Correlation Matrix — Daily Returns", height)


# ─── PORTFOLIO CHARTS ────────────────────────────────────────────────────────

def portfolio_performance_chart(perf_df: pd.DataFrame, height: int = 420) -> go.Figure:
    """Normalised cumulative return comparison chart."""
    fig = go.Figure()
    colours = [THEME["green"], THEME["blue"], THEME["amber"], THEME["purple"],
               THEME["orange"], THEME["cyan"], THEME["red"]]
    for i, col in enumerate(perf_df.columns):
        normed = perf_df[col] / perf_df[col].iloc[0] * 100
        fig.add_trace(go.Scatter(
            x=normed.index, y=normed,
            name=col, line=dict(color=colours[i % len(colours)], width=2),
        ))
    fig.add_hline(y=100, line_color=THEME["text_dim"], line_dash="dot", line_width=1)
    fig.update_yaxes(title_text="Indexed to 100")
    return _apply_theme(fig, "Portfolio Performance (Indexed)", height)
