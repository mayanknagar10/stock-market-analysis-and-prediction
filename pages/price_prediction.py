"""
Page 2 — Price Prediction
Universal checkpoint architecture: a single XGBoost+LightGBM model trained
once on a diverse cross-section of stocks, loaded instantly for inference
on any ticker. No per-request training in the normal path — typically
completes in well under a second. Falls back to a fast single-ticker
model only if no checkpoint has been trained yet.
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.data_fetcher import fetch_ohlcv, fetch_fundamentals, validate_ticker, PERIOD_MAP, detect_market, currency_symbol
from core.models import (forecast_future, walk_forward_backtest,
                          universal_model_available, universal_model_metadata,
                          train_universal_model, reload_universal_predictor,
                          DEFAULT_TRAIN_UNIVERSE)
from utils.helpers import inject_css, section_header, kpi_row, kpi_card, fmt_price, fmt_pct, esc, footer_bar, top_bar
from utils.charts import prediction_chart, safe_layout, T
import plotly.graph_objects as go
inject_css()

with st.sidebar:
    ticker = st.text_input("Ticker Symbol", value="AAPL",
                           placeholder="AAPL · RELIANCE.NS").upper().strip()
    period_label = st.selectbox("Data Period", list(PERIOD_MAP.keys()), index=3,
                                help="History fetched for this ticker's own volatility & features")
    period, interval = PERIOD_MAP[period_label]
    st.divider()
    st.markdown("**Forecast Settings**")
    horizon = st.slider("Forecast Horizon (days)", 5, 30, 10)
    n_folds = st.slider("Backtest Folds", 3, 8, 5)
    st.divider()

    model_ready = universal_model_available()
    if model_ready:
        meta = universal_model_metadata()
        st.markdown(
            f'<div style="background:rgba(63,185,80,0.1);border:1px solid #3FB950;'
            f'border-radius:6px;padding:10px 12px;font-size:11px;color:#C9D1D9;'
            f'font-family:\'IBM Plex Mono\',monospace;line-height:1.7">'
            f'<b style="color:#3FB950">🌐 Universal Model Active</b><br>'
            f'· Trained on {meta.get("n_tickers_used","?")} companies<br>'
            f'· {meta.get("n_train_rows",0):,} training rows<br>'
            f'· Test dir. accuracy: {meta.get("test_directional_accuracy","?")}%<br>'
            f'· Inference: instant (no per-request training)'
            f'</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="background:rgba(227,179,65,0.1);border:1px solid #E3B341;'
            'border-radius:6px;padding:10px 12px;font-size:11px;color:#C9D1D9;'
            'font-family:\'IBM Plex Mono\',monospace;line-height:1.7">'
            '<b style="color:#E3B341">⚠️ No Universal Model Yet</b><br>'
            'Using a fast per-ticker fallback model.<br>'
            'Train the universal model below for<br>'
            'better accuracy and instant inference on<br>'
            'every future request.</div>', unsafe_allow_html=True)

    st.divider()
    st.caption("Statistical estimates — not financial advice.")
    run_btn = st.button("▶  Run Forecast", type="primary", use_container_width=True)

# ── Training panel (always available, collapsed by default) ─────────────────
with st.expander("🔧 Train / Retrain Universal Model", expanded=not model_ready):
    st.markdown(
        "Trains **one model** on a diverse cross-section of companies "
        "(default: 20 NSE + 20 US stocks across sectors). The resulting "
        "checkpoint is then used for **every ticker** on this page — no "
        "retraining needed per stock. Requires internet access to Yahoo "
        "Finance; takes roughly 2–5 minutes."
    )
    col_t1, col_t2 = st.columns([3, 1])
    with col_t1:
        custom_universe = st.text_area(
            "Training universe (comma-separated, optional)",
            value="", placeholder=f"Leave blank to use the default {len(DEFAULT_TRAIN_UNIVERSE)}-ticker universe",
            height=70)
    with col_t2:
        train_period = st.selectbox("History", ["2y", "5y", "max"], index=1)
        train_btn = st.button("🚀  Train Now", use_container_width=True)

    if train_btn:
        universe = ([t.strip().upper() for t in custom_universe.split(",") if t.strip()]
                    if custom_universe.strip() else None)
        prog_bar = st.progress(0, "Starting…")
        def _update(pct, msg):
            prog_bar.progress(min(pct, 1.0), msg)
        try:
            meta = train_universal_model(universe=universe, period=train_period,
                                         progress_callback=_update)
            reload_universal_predictor()
            st.success(
                f"✓ Trained on {meta['n_tickers_used']} tickers, "
                f"{meta['n_train_rows']:,} rows. "
                f"Test directional accuracy: {meta['test_directional_accuracy']:.1f}%"
            )
            st.info(
                "To make this permanent across Streamlit Cloud redeploys, "
                "download the `models/` folder and commit it to your GitHub "
                "repo (`git add models/ && git commit && git push`). "
                "Otherwise this checkpoint only lasts until the app restarts."
            )
            st.rerun()
        except Exception as e:
            st.error(
                f"Training failed: {e}\n\n"
                "Most common cause: this environment can't reach Yahoo Finance. "
                "Training works on Streamlit Cloud and local machines with normal "
                "internet access."
            )

# ── Landing state ─────────────────────────────────────────────────────────
if not run_btn and "pred_result" not in st.session_state:
    st.markdown(
        '<div style="display:flex;flex-direction:column;align-items:center;'
        'justify-content:center;min-height:38vh;text-align:center;padding:40px">'
        '<div style="font-size:52px;margin-bottom:16px">🔮</div>'
        '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:22px;font-weight:600;'
        'color:#C9D1D9;margin-bottom:10px">Price Prediction Engine</div>'
        '<div style="font-size:13px;color:#8B949E;max-width:540px;line-height:1.8">'
        'One model, trained once, works on <b style="color:#C9D1D9">any company</b>.<br>'
        'Predicts <b style="color:#C9D1D9">log returns</b> (stationary) using 56 '
        'scale-free technical features.<br>'
        'CI = <b style="color:#C9D1D9">GBM volatility cone</b> (σ·√t), computed live '
        'from this ticker\'s own history.<br><br>'
        '<span style="color:#3FB950">Configure sidebar → Run Forecast</span></div></div>',
        unsafe_allow_html=True)
    st.stop()

# ── Run ────────────────────────────────────────────────────────────────────
if run_btn:
    valid, err = validate_ticker(ticker)
    if not valid:
        st.error(f"**{ticker}** — {err}")
        st.stop()
    df = fetch_ohlcv(ticker, period, interval)
    if df.empty or len(df) < 80:
        st.error("Not enough data. Use a longer period (≥ 1 Year).")
        st.stop()
    try:
        result = forecast_future(df, horizon=horizon)
    except Exception as e:
        st.error(f"Forecast failed: {e}")
        st.stop()
    try:
        bt_result = walk_forward_backtest(df, horizon=1, n_folds=n_folds)
    except Exception as e:
        bt_result = None
        st.warning(f"Backtest skipped: {e}")
    st.session_state.update({"pred_result": result, "bt_result": bt_result,
                              "pred_df": df, "pred_ticker": ticker})

result = st.session_state.get("pred_result")
bt_result = st.session_state.get("bt_result")
df = st.session_state.get("pred_df")
tkr = st.session_state.get("pred_ticker", ticker)
if result is None:
    st.info("Click **Run Forecast** in the sidebar.")
    st.stop()

fc = result["forecast"]
last_p = float(df["Close"].iloc[-1])
f_end = float(fc["Forecast"].iloc[-1])
f_chg = (f_end - last_p) / last_p
_mkt = detect_market(tkr)
_sym = currency_symbol("INR" if _mkt in ("NSE","BSE") else "USD")
_flag = "🇮🇳" if _mkt in ("NSE","BSE") else "🇺🇸"
mode = result["mode"]
daily_v = result["daily_volatility"]
elapsed = result.get("elapsed_seconds", 0)
f_lo = float(fc["Lower_80"].iloc[-1])
f_hi = float(fc["Upper_80"].iloc[-1])

mode_badge = (
    '<span style="background:rgba(63,185,80,0.15);color:#3FB950;border:1px solid #3FB950;'
    'border-radius:4px;padding:2px 8px;font-size:10px;margin-left:8px">🌐 Universal Model</span>'
    if mode == "universal" else
    '<span style="background:rgba(227,179,65,0.15);color:#E3B341;border:1px solid #E3B341;'
    'border-radius:4px;padding:2px 8px;font-size:10px;margin-left:8px">⚠️ Fallback (per-ticker)</span>'
)

info = fetch_fundamentals(tkr)
prev_p = float(df["Close"].iloc[-2]) if len(df) > 1 else last_p
day_chg = last_p - prev_p
day_chg_pct = (day_chg / prev_p * 100) if prev_p else 0.0
top_bar(tkr, info.get("name", tkr), last_p, day_chg, day_chg_pct, _sym, _mkt, info.get("logo_url", ""))
st.markdown(
    f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:12px;'
    f'color:#8B949E;margin:-10px 0 16px">'
    f'Price Forecast — {horizon}-Day Horizon {mode_badge}'
    f'<span style="float:right;font-size:11px;color:#3FB950">'
    f'Inference time: {elapsed}s</span></div>', unsafe_allow_html=True)

if mode == "fallback":
    st.warning(
        "No universal checkpoint is trained yet, so this forecast uses a quick "
        "single-ticker model fitted just to this stock's own history. This is "
        "fast but more prone to overfitting on small datasets. Train the "
        "universal model above (once) for better, faster predictions on every "
        "ticker going forward."
    )

# ── KPI row ────────────────────────────────────────────────────────────────
section_header("Forecast Summary")
in_mae = result.get("in_sample_mae")
in_dir = result.get("in_sample_dir_acc")
mae_label = "Test MAE (log ret)" if mode == "universal" else "In-sample MAE (overfit-prone)"
dir_label = "Test dir. accuracy" if mode == "universal" else "In-sample dir. accuracy (overfit-prone)"

kpi_row([
    kpi_card("Current Price", fmt_price(last_p, currency=_sym), "Last close"),
    kpi_card(f"Day {horizon} Forecast", fmt_price(f_end, currency=_sym),
             f"{horizon}-day projection", "pos" if f_chg >= 0 else "neg"),
    kpi_card("Expected Move", fmt_pct(f_chg), "vs current",
             "pos" if f_chg >= 0 else "neg"),
    kpi_card("80% CI Lower", fmt_price(f_lo, currency=_sym), "GBM floor (1.28σ√t)"),
    kpi_card("80% CI Upper", fmt_price(f_hi, currency=_sym), "GBM ceiling (1.28σ√t)"),
    kpi_card("Daily Volatility", f"{daily_v:.3f}%", "σ of log returns (this ticker)"),
    kpi_card(mae_label,
             f"{in_mae:.5f}" if in_mae is not None else "—", ""),
    kpi_card(dir_label,
             f"{in_dir:.1f}%" if in_dir is not None else "—", "",
             "pos" if (in_dir and in_dir >= 53 and mode == "universal") else ""),
    kpi_card("Features", str(result["n_features"]), "Scale-free, cross-sectional"),
    kpi_card("Trained On",
             f"{result.get('universe_size','—')} tickers" if mode == "universal" else "This ticker only",
             ""),
])

# ── Forecast chart ─────────────────────────────────────────────────────────
section_header("Price Forecast Chart")
hist = df["Close"].iloc[-120:]
bt_df = bt_result["predictions"] if bt_result and "predictions" in bt_result and not bt_result["predictions"].empty else None
st.plotly_chart(prediction_chart(hist, fc, backtest_df=bt_df, ticker=tkr, height=500),
                use_container_width=True, config={"displayModeBar": False})

with st.expander("📐 How this forecast is calculated"):
    st.markdown(
        f"**1. Direction & magnitude** — the {'universal checkpoint (trained once on '+str(result.get('universe_size','many'))+' companies)' if mode=='universal' else 'per-ticker fallback model'} "
        f"predicts tomorrow's expected log return from 56 scale-free technical "
        f"features (RSI, MACD, Bollinger %B, price-distance-from-moving-averages, "
        f"volatility, volume flow, etc).\n\n"
        f"**2. Multi-day compounding** — the 1-day expected return is compounded "
        f"forward: `Price_t = Price_0 × exp(t × expected_daily_return)`.\n\n"
        f"**3. Confidence interval** — uses a GBM volatility cone computed from "
        f"**this ticker's own** historical daily volatility (σ = {daily_v:.4f}%):  \n"
        f"`Lower/Upper = P₀ × exp(±1.28 × σ × √t)` — width grows as √t, "
        f"consistent with random-walk theory."
    )

# ── Day-by-day + backtest ───────────────────────────────────────────────────
col_tbl, col_bt = st.columns([1, 1])
with col_tbl:
    section_header("Day-by-Day Forecast")
    th_ = ("padding:8px 10px;text-align:left;font-family:'IBM Plex Mono',monospace;"
           "font-size:9px;color:#8B949E;text-transform:uppercase")
    td_ = "padding:6px 10px;border-bottom:1px solid #30363D;font-family:'IBM Plex Mono',monospace"
    rows_h = ""
    for i, (dt, row) in enumerate(fc.iterrows(), 1):
        chg_v = (row["Forecast"] - last_p) / last_p * 100
        col_ = "#3FB950" if chg_v >= 0 else "#F85149"
        rows_h += (
            f'<tr><td style="{td_};font-size:10px;color:#8B949E">'
            f'Day {i} · {pd.Timestamp(dt).strftime("%d %b")}</td>'
            f'<td style="{td_};font-size:12px;font-weight:600;color:#C9D1D9">'
            f'{_sym}{row["Forecast"]:,.2f}</td>'
            f'<td style="{td_};font-size:11px;color:{col_}">{chg_v:+.2f}%</td>'
            f'<td style="{td_};font-size:10px;color:#8B949E">'
            f'{_sym}{row["Lower_80"]:,.2f}–{_sym}{row["Upper_80"]:,.2f}</td></tr>')
    st.markdown(
        f'<table style="width:100%;border-collapse:collapse;background:#161B22;'
        f'border:1px solid #30363D;border-radius:6px;overflow:hidden">'
        f'<thead><tr style="background:#21262D">'
        f'<th style="{th_}">Day</th><th style="{th_}">Forecast</th>'
        f'<th style="{th_}">Δ Now</th><th style="{th_}">80% CI</th>'
        f'</tr></thead><tbody>{rows_h}</tbody></table>',
        unsafe_allow_html=True)

with col_bt:
    section_header("Walk-Forward Evaluation")
    if bt_result and not bt_result["fold_metrics"].empty:
        agg = bt_result["aggregate"]
        fold_df = bt_result["fold_metrics"]
        bt_mode = bt_result.get("mode", mode)
        if bt_mode == "universal":
            st.caption("Pure inference across rolling windows — no retraining, so this runs instantly.")
        mae_m = agg.get("MAE_mean", 0); mae_s = agg.get("MAE_std", 0)
        mape_m = agg.get("MAPE (%)_mean", 0); mape_s = agg.get("MAPE (%)_std", 0)
        dir_m = agg.get("Dir. Accuracy (%)_mean", 0); dir_s = agg.get("Dir. Accuracy (%)_std", 0)
        kpi_row([
            kpi_card("MAE", fmt_price(mae_m, currency=_sym), f"+/-{_sym}{mae_s:.2f}"),
            kpi_card("MAPE", f"{mape_m:.2f}%", f"+/-{mape_s:.2f}%"),
            kpi_card("Dir. Acc.", f"{dir_m:.1f}%", f"+/-{dir_s:.1f}%",
                     "pos" if dir_m >= 53 else ""),
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
        fig.update_layout(**safe_layout(
            {"yaxis": dict(range=[0, 100]), "showlegend": False},
            height=260, title="Directional Accuracy per Fold"))
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
            fig2.update_layout(**safe_layout(
                {"xaxis_title": f"Actual ({_sym})", "yaxis_title": f"Predicted ({_sym})"},
                height=280, title="Actual vs Predicted"))
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("Not enough history for a backtest on this ticker.")

st.markdown(
    '<div style="background:#161B22;border:1px solid #30363D;border-radius:6px;'
    'padding:12px 16px;margin-top:20px;font-size:11px;color:#8B949E;'
    'font-family:\'IBM Plex Mono\',monospace;line-height:1.7">'
    '⚠️ <b style="color:#E3B341">Disclaimer</b> — Statistical forecasts based on '
    'historical technical patterns only. Does not incorporate fundamentals, '
    'earnings, or macro events. Past accuracy does not guarantee future results. '
    '<b>Not financial advice.</b></div>',
    unsafe_allow_html=True)

footer_bar()
