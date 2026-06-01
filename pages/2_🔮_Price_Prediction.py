"""
Page 2 — Price Prediction
Ensemble ML forecast: XGBoost + LightGBM with walk-forward backtesting,
confidence intervals, feature importance, and performance metrics.
Pure Streamlit — no external AI APIs.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

st.set_page_config(page_title="Price Prediction · StockPro", page_icon="🔮",
                   layout="wide", initial_sidebar_state="expanded")

from core.data_fetcher  import fetch_ohlcv, validate_ticker, PERIOD_MAP, detect_market, currency_symbol
from core.models        import forecast_future, walk_forward_backtest
from utils.helpers      import (inject_css, section_header, kpi_row, kpi_card,
                                fmt_price, fmt_pct)
from utils.charts       import prediction_chart, THEME
import plotly.graph_objects as go

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
        help="US: AAPL  |  NSE: RELIANCE.NS  |  BSE: RELIANCE.BO"
    ).upper().strip()
    period_label = st.selectbox("Training Data", list(PERIOD_MAP.keys()), index=4,
                                help="More data = better-trained model")
    period, interval = PERIOD_MAP[period_label]

    st.divider()
    st.markdown("**Forecast Settings**")
    horizon = st.slider("Forecast Horizon (days)", 5, 30, 10,
                        help="Business days ahead to forecast")
    n_folds = st.slider("Backtest Folds", 3, 8, 5,
                        help="Walk-forward validation folds")
    run_backtest = st.checkbox("Run Backtest", value=True,
                               help="Walk-forward backtest (takes ~20s for large datasets)")

    st.divider()
    st.markdown("""<div style="background:#161B22;border:1px solid #30363D;
    border-radius:6px;padding:10px 12px;font-size:11px;color:#8B949E;
    font-family:'IBM Plex Mono',monospace;line-height:1.6;">
    <b style="color:#E3B341">Models Used</b><br>
    · XGBoost (mid / q10 / q90)<br>
    · LightGBM (mid)<br>
    · 60+ engineered features<br>
    · Walk-forward CV<br>
    · No data leakage
    </div>""", unsafe_allow_html=True)

    st.divider()
    st.caption("⚠️ Forecasts are statistical estimates, not financial advice.")
    run_btn = st.button("▶  Run Forecast", type="primary", use_container_width=True)

# ─── MAIN ───────────────────────────────────────────────────────────────────
if not ticker:
    st.info("Enter a ticker symbol in the sidebar and click **Run Forecast**.")
    st.stop()

if not run_btn and "pred_result" not in st.session_state:
    st.markdown(f"""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
    min-height:40vh;text-align:center;padding:40px;">
      <div style="font-size:48px;margin-bottom:16px;">🔮</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:20px;
                  font-weight:600;color:#C9D1D9;margin-bottom:8px;">
        Price Prediction Engine</div>
      <div style="font-size:13px;color:#8B949E;max-width:480px;line-height:1.7;">
        XGBoost + LightGBM ensemble trained on 60+ technical indicators.
        Walk-forward backtesting ensures zero data leakage.
        Configure your settings in the sidebar and click <b style="color:#3FB950">Run Forecast</b>.
      </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ─── RUN / LOAD ─────────────────────────────────────────────────────────────
if run_btn:
    valid, err = validate_ticker(ticker)
    if not valid:
        st.error(f"**{ticker}** — {err}")
        st.stop()

    with st.spinner("Fetching data…"):
        df = fetch_ohlcv(ticker, period, interval)
    if df.empty or len(df) < 60:
        st.error("Not enough data. Try a longer training period (≥ 1 Year).")
        st.stop()

    progress = st.progress(0, text="Engineering features…")

    with st.spinner("Training XGBoost + LightGBM ensemble…"):
        progress.progress(20, text="Training models…")
        try:
            result = forecast_future(df, horizon=horizon)
            progress.progress(65, text="Forecast generated…")
        except Exception as e:
            st.error(f"Model training failed: {e}")
            st.stop()

    bt_result = None
    if run_backtest:
        with st.spinner(f"Running {n_folds}-fold walk-forward backtest…"):
            progress.progress(75, text="Running backtest…")
            try:
                bt_result = walk_forward_backtest(df, horizon=1, n_folds=n_folds)
                progress.progress(95, text="Almost done…")
            except Exception as e:
                st.warning(f"Backtest skipped: {e}")

    progress.progress(100, text="Done!")
    progress.empty()

    st.session_state["pred_result"] = result
    st.session_state["bt_result"]   = bt_result
    st.session_state["pred_df"]     = df
    st.session_state["pred_ticker"] = ticker

# ─── DISPLAY RESULTS ────────────────────────────────────────────────────────
result    = st.session_state.get("pred_result")
bt_result = st.session_state.get("bt_result")
df        = st.session_state.get("pred_df")
tkr       = st.session_state.get("pred_ticker", ticker)

if result is None or df is None:
    st.info("Click **Run Forecast** in the sidebar to generate predictions.")
    st.stop()

fc      = result["forecast"]
last_p  = float(df["Close"].iloc[-1])
f_end   = float(fc["Forecast"].iloc[-1])
f_chg   = (f_end - last_p) / last_p
f_lo    = float(fc["Lower_80"].iloc[-1])
f_hi    = float(fc["Upper_80"].iloc[-1])
n_feat  = result["n_features"]
n_train = result["n_train"]
models  = result["models_available"]

# Detect currency for this ticker
_mkt   = detect_market(tkr)
_curr  = "INR" if _mkt in ("NSE", "BSE") else "USD"
_sym   = currency_symbol(_curr)
_flag  = "🇮🇳" if _mkt in ("NSE","BSE") else "🇺🇸"

# ─── HEADER ─────────────────────────────────────────────────────────────────
st.markdown(f"""<div style="font-family:'IBM Plex Mono',monospace;padding:10px 0 6px;
border-bottom:1px solid #30363D;margin-bottom:16px;">
<span style="font-size:20px;font-weight:600;color:#C9D1D9">{tkr}</span>&nbsp;&nbsp;
<span style="font-size:11px;color:#E3B341;font-family:'IBM Plex Mono',monospace">{_flag} {_mkt}</span>&nbsp;&nbsp;
<span style="font-size:13px;color:#8B949E">Price Forecast — {horizon}-Day Horizon</span>
<span style="float:right;font-size:12px;color:#3FB950">
  XGBoost {'✓' if models['xgboost'] else '✗'} &nbsp;
  LightGBM {'✓' if models['lightgbm'] else '✗'} &nbsp;
  {n_feat} features · {n_train} training samples
</span></div>""", unsafe_allow_html=True)

# ─── KPI ROW ────────────────────────────────────────────────────────────────
section_header("Forecast Summary")
kpi_row([
    kpi_card("Current Price",    fmt_price(last_p, currency=_sym),       "Last Close"),
    kpi_card(f"Day {horizon} Forecast", fmt_price(f_end, currency=_sym), f"{horizon}-day projection",
             "pos" if f_chg >= 0 else "neg"),
    kpi_card("Expected Δ",       fmt_pct(f_chg),          "vs Current",
             "pos" if f_chg >= 0 else "neg"),
    kpi_card("80% CI Lower",     fmt_price(f_lo, currency=_sym),          "Downside bound"),
    kpi_card("80% CI Upper",     fmt_price(f_hi, currency=_sym),          "Upside bound"),
    kpi_card("In-Sample MAE",    f"{_sym}{result['in_sample_mae']:.3f}",  "Mean Abs Error"),
    kpi_card("In-Sample RMSE",   f"{_sym}{result['in_sample_rmse']:.3f}", "Root Mean Sq Error"),
    kpi_card("In-Sample MAPE",   f"{result['in_sample_mape']:.2f}%", "Mean Abs Pct Error"),
])

# ─── FORECAST CHART ─────────────────────────────────────────────────────────
section_header("Price Forecast Chart")

bt_df = bt_result["predictions"] if bt_result else None

# Show last 120 days of history + forecast
hist = df["Close"].iloc[-120:]
fig  = prediction_chart(hist, fc, backtest_df=bt_df, ticker=tkr, height=520)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ─── FORECAST TABLE ─────────────────────────────────────────────────────────
col_tbl, col_bt = st.columns([1, 1])

with col_tbl:
    section_header("Day-by-Day Forecast")
    fc_disp = fc.copy()
    fc_disp.index = fc_disp.index.strftime("%d %b %Y")
    rows = ""
    for date, row in fc_disp.iterrows():
        chg  = (row["Forecast"] - last_p) / last_p * 100
        sign = "+" if chg >= 0 else ""
        colour = "#3FB950" if chg >= 0 else "#F85149"
        rows += f"""<tr>
          <td style="padding:6px 10px;border-bottom:1px solid #30363D;
                     font-family:'IBM Plex Mono',monospace;font-size:11px;
                     color:#8B949E">{date}</td>
          <td style="padding:6px 10px;border-bottom:1px solid #30363D;
                     font-family:'IBM Plex Mono',monospace;font-size:12px;
                     font-weight:600;color:#C9D1D9">{_sym}{row['Forecast']:.2f}</td>
          <td style="padding:6px 10px;border-bottom:1px solid #30363D;
                     font-family:'IBM Plex Mono',monospace;font-size:11px;
                     color:{colour}">{sign}{chg:.2f}%</td>
          <td style="padding:6px 10px;border-bottom:1px solid #30363D;
                     font-family:'IBM Plex Mono',monospace;font-size:11px;
                     color:#8B949E">{_sym}{row['Lower_80']:.2f} – {_sym}{row['Upper_80']:.2f}</td>
        </tr>"""
    st.markdown(f"""<table style="width:100%;border-collapse:collapse;
      background:#161B22;border:1px solid #30363D;border-radius:6px;overflow:hidden">
      <thead><tr style="background:#21262D">
        <th style="padding:8px 10px;text-align:left;font-family:'IBM Plex Mono',monospace;
           font-size:10px;color:#8B949E;text-transform:uppercase">Date</th>
        <th style="padding:8px 10px;text-align:left;font-family:'IBM Plex Mono',monospace;
           font-size:10px;color:#8B949E;text-transform:uppercase">Forecast</th>
        <th style="padding:8px 10px;text-align:left;font-family:'IBM Plex Mono',monospace;
           font-size:10px;color:#8B949E;text-transform:uppercase">Δ vs Now</th>
        <th style="padding:8px 10px;text-align:left;font-family:'IBM Plex Mono',monospace;
           font-size:10px;color:#8B949E;text-transform:uppercase">80% CI Range</th>
      </tr></thead><tbody>{rows}</tbody></table>""", unsafe_allow_html=True)

# ─── BACKTEST METRICS ───────────────────────────────────────────────────────
with col_bt:
    section_header("Walk-Forward Backtest")
    if bt_result:
        agg = bt_result["aggregate"]
        fold_df = bt_result["fold_metrics"]

        kpi_row([
            kpi_card("MAE",  f"${agg.get('MAE_mean','—'):.3f}",  f"±{agg.get('MAE_std',0):.3f}"),
            kpi_card("RMSE", f"${agg.get('RMSE_mean','—'):.3f}", f"±{agg.get('RMSE_std',0):.3f}"),
            kpi_card("MAPE", f"{agg.get('MAPE (%)_mean','—'):.2f}%",
                     f"±{agg.get('MAPE (%)_std',0):.2f}%"),
            kpi_card("Dir. Accuracy",
                     f"{agg.get('Dir. Accuracy (%)_mean','—'):.1f}%",
                     f"±{agg.get('Dir. Accuracy (%)_std',0):.1f}%",
                     "pos" if agg.get("Dir. Accuracy (%)_mean",50) >= 55 else ""),
        ])

        # Per-fold bar chart
        fig_folds = go.Figure()
        colours = ["#3FB950" if v >= 55 else "#F85149"
                   for v in fold_df["Dir. Accuracy (%)"]]
        fig_folds.add_trace(go.Bar(
            x=[f"Fold {i}" for i in fold_df["Fold"]],
            y=fold_df["Dir. Accuracy (%)"],
            marker_color=colours, opacity=0.85, name="Dir. Accuracy",
        ))
        fig_folds.add_hline(y=50, line_color="#8B949E", line_dash="dot",
                            annotation_text="50% (random)")
        fig_folds.update_layout(
            plot_bgcolor="#0D1117", paper_bgcolor="#0D1117", height=240,
            font=dict(family="IBM Plex Mono, monospace", color="#C9D1D9", size=11),
            margin=dict(l=8, r=8, t=30, b=8),
            xaxis=dict(gridcolor="#21262D"),
            yaxis=dict(gridcolor="#21262D", title_text="Directional Accuracy (%)"),
            showlegend=False,
            title=dict(text="Directional Accuracy per Fold", font_size=12),
        )
        st.plotly_chart(fig_folds, use_container_width=True,
                        config={"displayModeBar": False})

        # Fold table
        st.dataframe(
            fold_df.set_index("Fold").style
            .format({"MAE": "${:.4f}", "RMSE": "${:.4f}",
                     "MAPE (%)": "{:.2f}%", "Dir. Accuracy (%)": "{:.1f}%"})
            .background_gradient(subset=["Dir. Accuracy (%)"],
                                 cmap="RdYlGn", vmin=40, vmax=70),
            use_container_width=True,
        )
    else:
        st.info("Enable **Run Backtest** in the sidebar to see walk-forward validation results.")

# ─── FEATURE IMPORTANCE ─────────────────────────────────────────────────────
section_header("Feature Importance")
feat_imp = result.get("feature_importance", pd.DataFrame())

if feat_imp is not None and not feat_imp.empty:
    top_n  = 20
    top_fi = feat_imp.head(top_n).copy()
    # Normalise to %
    top_fi["Pct"] = top_fi["Importance"] / top_fi["Importance"].sum() * 100

    fig_fi = go.Figure(go.Bar(
        x=top_fi["Pct"],
        y=top_fi["Feature"],
        orientation="h",
        marker=dict(
            color=top_fi["Pct"],
            colorscale=[[0, "#161B22"], [0.4, "#E3B341"], [1.0, "#3FB950"]],
            showscale=False,
        ),
    ))
    fig_fi.update_layout(
        plot_bgcolor="#0D1117", paper_bgcolor="#0D1117", height=480,
        font=dict(family="IBM Plex Mono, monospace", color="#C9D1D9", size=10),
        margin=dict(l=8, r=20, t=30, b=8),
        xaxis=dict(gridcolor="#21262D", title_text="Importance (%)"),
        yaxis=dict(gridcolor="#21262D", autorange="reversed"),
        title=dict(text=f"Top {top_n} Feature Importances (XGBoost + LightGBM ensemble)",
                   font_size=12),
    )
    st.plotly_chart(fig_fi, use_container_width=True, config={"displayModeBar": False})
else:
    st.info("Feature importance requires XGBoost or LightGBM to be installed.")

# ─── DISCLAIMER ─────────────────────────────────────────────────────────────
st.markdown("""<div style="background:#161B22;border:1px solid #30363D;border-radius:6px;
padding:12px 16px;margin-top:20px;font-size:11px;color:#8B949E;
font-family:'IBM Plex Mono',monospace;line-height:1.7;">
⚠️ <b style="color:#E3B341">Disclaimer</b> — These forecasts are generated by statistical
machine learning models trained on historical price and technical indicator data.
They do <b>not</b> incorporate fundamental analysis, earnings surprises, macroeconomic events,
or market sentiment. Past model performance does not guarantee future accuracy.
This is not financial advice. Always consult a qualified professional before making
investment decisions.</div>""", unsafe_allow_html=True)
