"""
Factor Analysis — Fama-French exposures + quantitative factor signals.

Two tools in one page:
  1. Single-stock factor exposure regression (market beta, size, value,
     profitability, investment tilts) against free public Fama-French
     data — no API key needed.
  2. Cross-sectional quant factor screening (Value, Momentum, Quality,
     Low-Volatility) across a universe, computed from data already in
     the app — ranks stocks by a composite factor score.
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.data_fetcher import fetch_ohlcv, fetch_fundamentals, validate_ticker, currency_symbol, detect_market
from core.factor_models import (fetch_fama_french_factors, compute_factor_exposures,
                                interpret_factor_betas, compute_composite_factor_scores,
                                FF_FACTOR_SETS)
from utils.helpers import (inject_css, section_header, kpi_row, kpi_card,
                           esc, footer_bar, sidebar_brand)
from utils.charts import T, BASE, safe_layout
import plotly.graph_objects as go
inject_css()

NIFTY50_UNIVERSE = [
    ("RELIANCE.NS","Reliance","Energy"), ("TCS.NS","TCS","IT"),
    ("HDFCBANK.NS","HDFC Bank","Banking"), ("INFY.NS","Infosys","IT"),
    ("ICICIBANK.NS","ICICI Bank","Banking"), ("HINDUNILVR.NS","HUL","FMCG"),
    ("ITC.NS","ITC","FMCG"), ("SBIN.NS","SBI","Banking"),
    ("BHARTIARTL.NS","Airtel","Telecom"), ("KOTAKBANK.NS","Kotak Bank","Banking"),
]
SP500_UNIVERSE = [
    ("AAPL","Apple","Technology"), ("MSFT","Microsoft","Technology"),
    ("NVDA","Nvidia","Technology"), ("GOOGL","Alphabet","Technology"),
    ("META","Meta","Communication"), ("AMZN","Amazon","Consumer"),
    ("JPM","JP Morgan","Financials"), ("V","Visa","Financials"),
    ("UNH","UnitedHlth","Healthcare"), ("XOM","ExxonMobil","Energy"),
]

with st.sidebar:
    sidebar_brand()
    st.divider()
    mode = st.radio("Analysis Type", ["📈 Factor Exposures (single stock)",
                                      "🔍 Quant Factor Screening (universe)"])
    st.divider()
    if "Exposures" in mode:
        ticker = st.text_input("Ticker Symbol", value="AAPL",
                               placeholder="AAPL · RELIANCE.NS").upper().strip()
        factor_set = st.selectbox("Factor Model", list(FF_FACTOR_SETS.keys()), index=1)
        years_back = st.slider("History (years)", 2, 10, 5)
        run_btn = st.button("▶  Run Factor Analysis", type="primary", use_container_width=True)
    else:
        universe_choice = st.radio("Universe", ["🇮🇳 NSE Nifty 50 (sample)", "🇺🇸 US S&P 500 (sample)"])
        run_btn = st.button("▶  Run Screening", type="primary", use_container_width=True)
    st.caption("Fama-French data: free public download, no API key.")

st.markdown(
    f'<div style="font-family:\'IBM Plex Mono\',monospace;padding:10px 0 6px;'
    f'border-bottom:1px solid #30363D;margin-bottom:16px">'
    f'<span style="font-size:20px;font-weight:600;color:#C9D1D9">Factor Analysis</span>'
    f'&nbsp;&nbsp;<span style="font-size:13px;color:#8B949E">{mode}</span></div>',
    unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# MODE 1 — SINGLE-STOCK FACTOR EXPOSURES
# ═══════════════════════════════════════════════════════════════════════════
if "Exposures" in mode:
    if not run_btn and "factor_exposure_result" not in st.session_state:
        st.markdown(
            '<div style="display:flex;flex-direction:column;align-items:center;'
            'justify-content:center;min-height:38vh;text-align:center;padding:40px">'
            '<div style="font-size:52px;margin-bottom:16px">📈</div>'
            '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:22px;font-weight:600;'
            'color:#C9D1D9;margin-bottom:10px">Fama-French Factor Exposures</div>'
            '<div style="font-size:13px;color:#8B949E;max-width:540px;line-height:1.8">'
            'Regress a stock\'s returns against Market, Size (SMB), Value (HML), '
            'Profitability (RMW) and Investment (CMA) factors to see what '
            'actually drives its returns.<br><br>'
            '<span style="color:#3FB950">Configure sidebar → Run Factor Analysis</span></div></div>',
            unsafe_allow_html=True)
        st.stop()

    if run_btn:
        valid, err = validate_ticker(ticker)
        if not valid:
            st.error(f"**{ticker}** — {err}")
            st.stop()
        with st.spinner("Fetching price history…"):
            df = fetch_ohlcv(ticker, f"{years_back}y", "1d")
        if df.empty or len(df) < 260:
            st.error("Not enough price history for a meaningful factor regression (need 2+ years).")
            st.stop()

        with st.spinner("Downloading Fama-French factor data (free, no key)…"):
            factors_df = fetch_fama_french_factors(factor_set, start=str(df.index[0].date()))

        if factors_df.empty:
            st.error(
                "Could not download Fama-French factor data. This needs internet "
                "access to Kenneth French's Dartmouth data library — works on "
                "Streamlit Cloud / local machines, may fail in restricted sandboxes."
            )
            st.stop()

        # Resample daily prices to monthly returns to match FF factor frequency
        monthly_close = df["Close"].resample("ME").last()
        monthly_returns = monthly_close.pct_change().dropna()

        result = compute_factor_exposures(monthly_returns, factors_df)
        st.session_state["factor_exposure_result"] = result
        st.session_state["factor_exposure_ticker"] = ticker

    result = st.session_state.get("factor_exposure_result")
    tkr = st.session_state.get("factor_exposure_ticker", ticker)

    if result and "error" not in result:
        section_header(f"Factor Exposure Report — {tkr}")
        kpi_row([
            kpi_card("Alpha (Annualised)", f"{result['alpha_annualised_pct']:+.2f}%",
                     f"t-stat: {result['alpha_t_stat']:.2f}",
                     "pos" if result['alpha_annualised_pct'] >= 0 else "neg"),
            kpi_card("R-Squared", f"{result['r_squared']:.3f}", "Variance explained"),
            kpi_card("Observations", str(result['n_observations']), "months"),
            kpi_card("Significant?", "Yes" if result['alpha_p_value'] < 0.05 else "No",
                     f"p={result['alpha_p_value']:.3f}"),
        ])

        section_header("Factor Betas")
        betas = result["betas"]
        t_stats = result["t_stats"]
        fig = go.Figure(go.Bar(
            x=list(betas.values()), y=list(betas.keys()), orientation="h",
            marker_color=["#3FB950" if v >= 0 else "#F85149" for v in betas.values()],
            text=[f"{v:+.3f}" for v in betas.values()], textposition="outside",
            textfont=dict(size=11, family="IBM Plex Mono, monospace", color="#C9D1D9")))
        fig.add_vline(x=0, line_color=T["dim"], line_dash="dot")
        fig.update_layout(**safe_layout({}, height=300, title="Factor Loadings (Betas)"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        col_t, col_i = st.columns(2)
        with col_t:
            section_header("Statistical Detail")
            td = "border-bottom:1px solid #21262D;font-family:'IBM Plex Mono',monospace"
            rows = "".join(
                f'<tr><td style="padding:6px 10px;{td};font-size:11px;color:#C9D1D9">{f}</td>'
                f'<td style="padding:6px 10px;{td};font-size:11px;color:#C9D1D9">{betas[f]:+.4f}</td>'
                f'<td style="padding:6px 10px;{td};font-size:11px;color:#8B949E">{t_stats[f]:.2f}</td>'
                f'<td style="padding:6px 10px;{td};font-size:11px;color:{"#3FB950" if result["p_values"][f]<0.05 else "#8B949E"}">{result["p_values"][f]:.3f}</td></tr>'
                for f in betas.keys())
            th_ = "padding:7px 10px;text-align:left;font-family:'IBM Plex Mono',monospace;font-size:9px;color:#8B949E;text-transform:uppercase"
            st.markdown(
                f'<table style="width:100%;border-collapse:collapse;background:#161B22;'
                f'border:1px solid #30363D;border-radius:6px;overflow:hidden">'
                f'<thead><tr style="background:#21262D"><th style="{th_}">Factor</th>'
                f'<th style="{th_}">Beta</th><th style="{th_}">t-stat</th>'
                f'<th style="{th_}">p-value</th></tr></thead><tbody>{rows}</tbody></table>',
                unsafe_allow_html=True)
        with col_i:
            section_header("What This Means")
            notes = interpret_factor_betas(betas)
            for note in notes:
                st.markdown(f"• {note}")
            st.caption("Statistically significant factors (p < 0.05) are more reliable than others.")
    elif result and "error" in result:
        st.error(result["error"])

# ═══════════════════════════════════════════════════════════════════════════
# MODE 2 — QUANT FACTOR SCREENING
# ═══════════════════════════════════════════════════════════════════════════
else:
    if not run_btn and "factor_screen_result" not in st.session_state:
        st.markdown(
            '<div style="display:flex;flex-direction:column;align-items:center;'
            'justify-content:center;min-height:38vh;text-align:center;padding:40px">'
            '<div style="font-size:52px;margin-bottom:16px">🔍</div>'
            '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:22px;font-weight:600;'
            'color:#C9D1D9;margin-bottom:10px">Quant Factor Screening</div>'
            '<div style="font-size:13px;color:#8B949E;max-width:540px;line-height:1.8">'
            'Ranks stocks by Value, Momentum, Quality and Low-Volatility factors — '
            'the same style tilts used in quantitative equity investing.<br><br>'
            '<span style="color:#3FB950">Configure sidebar → Run Screening</span></div></div>',
            unsafe_allow_html=True)
        st.stop()

    if run_btn:
        universe = NIFTY50_UNIVERSE if "NSE" in universe_choice else SP500_UNIVERSE
        universe_data = []
        prog = st.progress(0, "Fetching data…")
        for i, (sym, name, sector) in enumerate(universe):
            prog.progress((i + 1) / len(universe), f"[{i+1}/{len(universe)}] {sym}")
            try:
                df = fetch_ohlcv(sym, "2y", "1d")
                if df.empty or len(df) < 260:
                    continue
                info = fetch_fundamentals(sym)
                universe_data.append({
                    "ticker": sym, "name": name, "sector": sector,
                    "close_series": df["Close"],
                    "pe_ratio": info.get("pe_ttm"),
                    "roe": info.get("roe"),
                    "gross_margin": info.get("gross_margin"),
                    "operating_margin": info.get("operating_margin"),
                })
            except Exception:
                continue
        prog.empty()
        scores_df = compute_composite_factor_scores(universe_data)
        st.session_state["factor_screen_result"] = scores_df
        st.session_state["factor_screen_universe"] = universe_data

    scores_df = st.session_state.get("factor_screen_result")
    if scores_df is not None and not scores_df.empty:
        section_header("Composite Factor Ranking")
        st.dataframe(
            scores_df.style.format("{:+.3f}", na_rep="—")
            .background_gradient(subset=["Composite"], cmap="RdYlGn", vmin=-1.5, vmax=1.5),
            use_container_width=True)

        section_header("Factor Score Breakdown")
        fig = go.Figure()
        for col, color in [("Momentum", T["blue"]), ("Value", T["green"]),
                          ("Quality", T["amber"]), ("LowVol", T["purple"])]:
            fig.add_trace(go.Bar(name=col, x=scores_df.index, y=scores_df[col], marker_color=color))
        fig.update_layout(**safe_layout({"barmode": "group"}, height=380,
                                        title="Factor Z-Scores by Stock"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    elif scores_df is not None:
        st.warning("No stocks had enough data to compute factor scores. Try a longer history or different universe.")

footer_bar()
