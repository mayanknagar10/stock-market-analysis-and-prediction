"""Page 2 — Price Prediction: XGBoost + LightGBM + LSTM, log-return target, GBM cone CI."""
import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

st.set_page_config(page_title="Price Prediction · StockPro", page_icon="🔮", layout="wide")

from core.data_fetcher import fetch_ohlcv, validate_ticker, PERIOD_MAP, detect_market, currency_symbol
from core.models       import forecast_future, walk_forward_backtest
from utils.helpers     import inject_css, section_header, kpi_row, kpi_card, fmt_price, fmt_pct, esc
from utils.charts      import prediction_chart, T, BASE
import plotly.graph_objects as go
inject_css()

SIDEBAR_LOGO = ('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:16px;font-weight:600;'
                'color:#3FB950;padding:8px 0 16px;">📈 StockPro'
                '<span style="font-size:10px;color:#8B949E;font-weight:400;display:block;'
                'letter-spacing:.1em;margin-top:2px;">ANALYTICS TERMINAL</span></div>')

with st.sidebar:
    st.markdown(SIDEBAR_LOGO, unsafe_allow_html=True)
    ticker = st.text_input("Ticker Symbol", value="AAPL",
                           placeholder="AAPL · RELIANCE.NS").upper().strip()
    period_label = st.selectbox("Training Data", list(PERIOD_MAP.keys()), index=4)
    period, interval = PERIOD_MAP[period_label]
    st.divider()
    st.markdown("**Forecast Settings**")
    horizon = st.slider("Forecast Horizon (days)", 5, 30, 10)
    n_folds = st.slider("Backtest Folds", 3, 8, 5)
    run_bt  = st.checkbox("Run Backtest", value=True)
    st.divider()
    st.markdown(
        '<div style="background:#161B22;border:1px solid #30363D;border-radius:6px;'
        'padding:10px 12px;font-size:11px;color:#8B949E;font-family:\'IBM Plex Mono\','
        'monospace;line-height:1.7"><b style="color:#E3B341">How it works</b><br>'
        '· Target = <b style="color:#C9D1D9">log return</b> (stationary)<br>'
        '· Price = P·exp(Σ r)<br>'
        '· CI = GBM cone: σ·√t<br>'
        '· XGBoost + LightGBM + LSTM<br>'
        '· 60+ engineered features<br>'
        '· Zero data leakage backtest</div>', unsafe_allow_html=True)
    st.divider()
    st.caption("Statistical estimates — not financial advice.")
    run_btn = st.button("▶  Run Forecast", type="primary", use_container_width=True)

if not run_btn and "pred_result" not in st.session_state:
    st.markdown(
        '<div style="display:flex;flex-direction:column;align-items:center;'
        'justify-content:center;min-height:42vh;text-align:center;padding:40px">'
        '<div style="font-size:52px;margin-bottom:16px">🔮</div>'
        '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:22px;font-weight:600;'
        'color:#C9D1D9;margin-bottom:10px">Price Prediction Engine</div>'
        '<div style="font-size:13px;color:#8B949E;max-width:520px;line-height:1.8">'
        'Ensemble: <b style="color:#C9D1D9">XGBoost + LightGBM + LSTM</b><br>'
        'Trains on <b style="color:#C9D1D9">log returns</b> — stationary, no bias.<br>'
        'CI = <b style="color:#C9D1D9">GBM volatility cone</b> (σ·√t).<br><br>'
        '<span style="color:#3FB950">Configure sidebar → Run Forecast</span></div></div>',
        unsafe_allow_html=True)
    st.stop()

if run_btn:
    valid, err = validate_ticker(ticker)
    if not valid:
        st.error(f"**{ticker}** — {err}"); st.stop()
    df = fetch_ohlcv(ticker, period, interval)
    if df.empty or len(df) < 80:
        st.error("Not enough data. Use 2+ Years."); st.stop()
    prog = st.progress(0, "Engineering features…")
    try:
        prog.progress(20, "Training ensemble…")
        result = forecast_future(df, horizon=horizon)
        prog.progress(70, "Forecast done…")
    except Exception as e:
        st.error(f"Training failed: {e}"); st.stop()
    bt_result = None
    if run_bt:
        try:
            prog.progress(75, "Backtesting…")
            bt_result = walk_forward_backtest(df, horizon=1, n_folds=n_folds)
        except Exception as e:
            st.warning(f"Backtest skipped: {e}")
    prog.progress(100, "Done!"); prog.empty()
    st.session_state.update({"pred_result": result, "bt_result": bt_result,
                              "pred_df": df, "pred_ticker": ticker})

result    = st.session_state.get("pred_result")
bt_result = st.session_state.get("bt_result")
df        = st.session_state.get("pred_df")
tkr       = st.session_state.get("pred_ticker", ticker)
if result is None:
    st.info("Click **Run Forecast** in the sidebar."); st.stop()

fc       = result["forecast"]
last_p   = float(df["Close"].iloc[-1])
f_end    = float(fc["Forecast"].iloc[-1])
f_chg    = (f_end - last_p) / last_p
_mkt     = detect_market(tkr)
_sym     = currency_symbol("INR" if _mkt in ("NSE","BSE") else "USD")
_flag    = "🇮🇳" if _mkt in ("NSE","BSE") else "🇺🇸"
lstm_u   = result["lstm_used"]
models   = result["models_available"]
daily_v  = result["daily_volatility"]
in_mae   = result["in_sample_mae"]
in_mape  = result["in_sample_mape"]
in_dir   = result["in_sample_dir_acc"]
lstm_w   = result["lstm_weight"]
f_lo     = float(fc["Lower_80"].iloc[-1])
f_hi     = float(fc["Upper_80"].iloc[-1])
xgb_ok   = "✓" if models["xgboost"]  else "✗"
lgb_ok   = "✓" if models["lightgbm"] else "✗"

lstm_badge = ('<span style="background:rgba(63,185,80,0.15);color:#3FB950;border:1px solid #3FB950;'
              'border-radius:4px;padding:2px 8px;font-size:10px;margin-left:8px">LSTM ✓</span>'
              if lstm_u else
              '<span style="background:rgba(139,148,158,0.1);color:#8B949E;border:1px solid #30363D;'
              'border-radius:4px;padding:2px 8px;font-size:10px;margin-left:8px">LSTM —</span>')

st.markdown(
    f'<div style="font-family:\'IBM Plex Mono\',monospace;padding:10px 0 6px;'
    f'border-bottom:1px solid #30363D;margin-bottom:16px">'
    f'<span style="font-size:20px;font-weight:600;color:#C9D1D9">{esc(tkr)}</span>'
    f'&nbsp;<span style="font-size:11px;color:#E3B341">{_flag} {_mkt}</span>'
    f'&nbsp;&nbsp;<span style="font-size:13px;color:#8B949E">'
    f'Price Forecast — {horizon}-Day Horizon</span>{lstm_badge}'
    f'<span style="float:right;font-size:11px;color:#3FB950">'
    f'XGB {xgb_ok} · LGB {lgb_ok} · {result["n_features"]} features · {result["n_train"]} samples'
    f'</span></div>', unsafe_allow_html=True)

section_header("Forecast Summary")
kpi_row([
    kpi_card("Current Price",           fmt_price(last_p, currency=_sym),  "Last close"),
    kpi_card(f"Day {horizon} Forecast", fmt_price(f_end, currency=_sym),   f"{horizon}-day projection",
             "pos" if f_chg >= 0 else "neg"),
    kpi_card("Expected Move",           fmt_pct(f_chg),                    "vs current",
             "pos" if f_chg >= 0 else "neg"),
    kpi_card("80% CI Lower",            fmt_price(f_lo, currency=_sym),    "GBM floor (1.28σ√t)"),
    kpi_card("80% CI Upper",            fmt_price(f_hi, currency=_sym),    "GBM ceiling (1.28σ√t)"),
    kpi_card("Daily Volatility",        f"{daily_v:.3f}%",                 "σ of log returns"),
    kpi_card("In-Sample MAE",           fmt_price(in_mae, currency=_sym),  ""),
    kpi_card("MAPE",                    f"{in_mape:.2f}%",                 "Mean abs pct error"),
    kpi_card("Dir. Accuracy",           f"{in_dir:.1f}%",                  "Correct direction",
             "pos" if in_dir >= 55 else ""),
    kpi_card("LSTM Weight",
             f"{lstm_w*100:.0f}%" if lstm_u else "—",
             "In ensemble" if lstm_u else "Not available"),
])

section_header("Price Forecast Chart")
hist  = df["Close"].iloc[-120:]
bt_df = bt_result["predictions"] if bt_result and "predictions" in bt_result else None
st.plotly_chart(prediction_chart(hist, fc, backtest_df=bt_df, ticker=tkr, height=500),
                use_container_width=True, config={"displayModeBar": False})

with st.expander("📐 How the confidence interval is calculated"):
    st.markdown(
        f"**GBM Volatility Cone**: `Lower/Upper = P₀ × exp(±1.28 × σ × √t)`  \n"
        f"- **P₀** = current price **{_sym}{last_p:,.2f}**  \n"
        f"- **σ** = daily log-return std = **{daily_v:.4f}%**  \n"
        f"- **t** = days ahead (1 → {horizon})  \n"
        f"- Width grows as **√t** (not linearly) — consistent with GBM theory")

col_tbl, col_bt = st.columns([1, 1])
with col_tbl:
    section_header("Day-by-Day Forecast")
    th_ = ("padding:8px 10px;text-align:left;font-family:'IBM Plex Mono',monospace;"
           "font-size:9px;color:#8B949E;text-transform:uppercase")
    td_ = "padding:6px 10px;border-bottom:1px solid #30363D;font-family:'IBM Plex Mono',monospace"
    rows_h = ""
    for i, (dt, row) in enumerate(fc.iterrows(), 1):
        chg_v = (row["Forecast"] - last_p) / last_p * 100
        col_  = "#3FB950" if chg_v >= 0 else "#F85149"
        lr_v  = row.get("Log_Return", 0) * 100
        rows_h += (
            f'<tr><td style="{td_};font-size:10px;color:#8B949E">'
            f'Day {i} · {pd.Timestamp(dt).strftime("%d %b")}</td>'
            f'<td style="{td_};font-size:12px;font-weight:600;color:#C9D1D9">'
            f'{_sym}{row["Forecast"]:,.2f}</td>'
            f'<td style="{td_};font-size:11px;color:{col_}">{chg_v:+.2f}%</td>'
            f'<td style="{td_};font-size:10px;color:#8B949E">'
            f'{_sym}{row["Lower_80"]:,.2f}–{_sym}{row["Upper_80"]:,.2f}</td>'
            f'<td style="{td_};font-size:10px;color:#8B949E">{lr_v:+.4f}%</td></tr>')
    st.markdown(
        f'<table style="width:100%;border-collapse:collapse;background:#161B22;'
        f'border:1px solid #30363D;border-radius:6px;overflow:hidden">'
        f'<thead><tr style="background:#21262D">'
        f'<th style="{th_}">Day</th><th style="{th_}">Forecast</th>'
        f'<th style="{th_}">Δ Now</th><th style="{th_}">80% CI</th>'
        f'<th style="{th_}">LogRet/Day</th>'
        f'</tr></thead><tbody>{rows_h}</tbody></table>',
        unsafe_allow_html=True)

with col_bt:
    section_header("Walk-Forward Backtest")
    if bt_result and not bt_result["fold_metrics"].empty:
        agg     = bt_result["aggregate"]
        fold_df = bt_result["fold_metrics"]
        mae_m   = agg.get("MAE_mean", 0);       mae_s  = agg.get("MAE_std", 0)
        mape_m  = agg.get("MAPE (%)_mean", 0);  mape_s = agg.get("MAPE (%)_std", 0)
        dir_m   = agg.get("Dir. Accuracy (%)_mean", 0)
        dir_s   = agg.get("Dir. Accuracy (%)_std", 0)
        kpi_row([
            kpi_card("MAE",  fmt_price(mae_m, currency=_sym), f"+/-{_sym}{mae_s:.2f}"),
            kpi_card("MAPE", f"{mape_m:.2f}%",                f"+/-{mape_s:.2f}%"),
            kpi_card("Dir. Acc.", f"{dir_m:.1f}%",            f"+/-{dir_s:.1f}%",
                     "pos" if dir_m >= 54 else ""),
        ])
        bar_c = ["#3FB950" if v >= 55 else ("#E3B341" if v >= 50 else "#F85149")
                 for v in fold_df["Dir. Accuracy (%)"]]
        fig = go.Figure(go.Bar(
            x=[f"Fold {i}" for i in fold_df["Fold"]], y=fold_df["Dir. Accuracy (%)"],
            marker_color=bar_c, opacity=0.85,
            text=[f"{v:.1f}%" for v in fold_df["Dir. Accuracy (%)"]],
            textposition="outside",
            textfont=dict(size=9, family="IBM Plex Mono, monospace", color="#C9D1D9")))
        fig.add_hline(y=50, line_color=T["dim"], line_dash="dot",
                      annotation_text=" 50% = random", annotation_font_size=9)
        fig.update_layout(**{**BASE, "height": 260,
            "title": dict(text="Directional Accuracy per Fold", font_size=12),
            "yaxis": dict(gridcolor=T["grid"], range=[0, 100]), "showlegend": False})
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        pred_df = bt_result.get("predictions", pd.DataFrame())
        if not pred_df.empty and "Actual" in pred_df.columns:
            mn = min(pred_df["Actual"].min(), pred_df["Predicted"].min())
            mx = max(pred_df["Actual"].max(), pred_df["Predicted"].max())
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode="lines",
                name="Perfect", line=dict(color=T["dim"], dash="dot", width=1)))
            fig2.add_trace(go.Scatter(x=pred_df["Actual"], y=pred_df["Predicted"],
                mode="markers", name="Predictions",
                marker=dict(color=T["blue"], size=3, opacity=0.5)))
            fig2.update_layout(**{**BASE, "height": 280,
                "title": dict(text="Actual vs Predicted", font_size=12)},
                xaxis_title=f"Actual ({_sym})", yaxis_title=f"Predicted ({_sym})")
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("Enable **Run Backtest** in the sidebar.")

section_header("Feature Importance (Top 20)")
fi = result.get("feature_importance", pd.DataFrame())
if fi is not None and not fi.empty:
    top = fi.head(20).copy()
    top["Pct"] = top["Importance"] / top["Importance"].sum() * 100
    fig_fi = go.Figure(go.Bar(
        x=top["Pct"], y=top["Feature"], orientation="h",
        marker=dict(color=top["Pct"],
                    colorscale=[[0, "#21262D"], [0.4, T["amber"]], [1.0, T["green"]]],
                    showscale=False),
        text=[f"{v:.1f}%" for v in top["Pct"]], textposition="outside",
        textfont=dict(size=9, family="IBM Plex Mono, monospace", color="#C9D1D9")))
    fig_fi.update_layout(**{**BASE, "height": 500,
        "title": dict(text="Top 20 Feature Importances", font_size=12)},
        xaxis_title="Importance (%)",
        yaxis=dict(autorange="reversed", gridcolor=T["grid"]),
        margin=dict(l=8, r=60, t=40, b=8))
    st.plotly_chart(fig_fi, use_container_width=True, config={"displayModeBar": False})
else:
    st.info("Feature importance requires XGBoost or LightGBM.")

st.markdown(
    '<div style="background:#161B22;border:1px solid #30363D;border-radius:6px;'
    'padding:12px 16px;margin-top:20px;font-size:11px;color:#8B949E;'
    'font-family:\'IBM Plex Mono\',monospace;line-height:1.7">'
    '⚠️ <b style="color:#E3B341">Disclaimer</b> — Statistical forecasts based on '
    'historical patterns only. Does not incorporate fundamentals, earnings, or macro. '
    'Past accuracy does not guarantee future results. <b>Not financial advice.</b></div>',
    unsafe_allow_html=True)
