"""
Global Data — Crypto, FX Rates, Macro Indicators, SEC Filings.

Every data source on this page requires ZERO login and ZERO API key:
  - CoinGecko       -> crypto prices & history
  - Frankfurter.app -> live + historical FX rates
  - World Bank      -> GDP / inflation / interest rates
  - SEC EDGAR       -> US company filings (10-K, 10-Q, 8-K)
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.external_data import (
    fetch_crypto_price, fetch_crypto_history, fetch_crypto_top_movers,
    fetch_fx_rate, fetch_fx_history,
    fetch_macro_indicator, WORLDBANK_INDICATORS, WORLDBANK_COUNTRIES,
    fetch_sec_filings, fetch_sec_cik,
    _COINGECKO_IDS,
)
from utils.helpers import (inject_css, section_header, kpi_row, kpi_card,
                            esc, top_bar_simple, footer_bar)
from utils.charts import T, BASE
import plotly.graph_objects as go
inject_css()

with st.sidebar:
    st.divider()
    st.markdown(
        '<div style="background:rgba(63,185,80,0.08);border:1px solid #3FB950;'
        'border-radius:6px;padding:10px 12px;font-size:11px;color:#C9D1D9;'
        'font-family:\'IBM Plex Mono\',monospace;line-height:1.7">'
        '<b style="color:#3FB950">🔓 Zero-login data</b><br>'
        'Every source on this page works with<br>'
        'no account, no API key, no sign-up.'
        '</div>', unsafe_allow_html=True)

top_bar_simple("Global Data", "Crypto · FX · Macro · SEC Filings — all key-free")

tabs = st.tabs(["  💰 Crypto  ", "  💱 FX Rates  ", "  🌍 Macro Indicators  ", "  📄 SEC Filings  "])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — CRYPTO (CoinGecko, no key)
# ═══════════════════════════════════════════════════════════════════════════
with tabs[0]:
    section_header("Crypto Lookup")
    col_in, col_kpi = st.columns([1, 3])
    with col_in:
        crypto_sym = st.selectbox("Coin", sorted(_COINGECKO_IDS.keys()), index=0)

    with st.spinner(f"Loading {crypto_sym}…"):
        price_data = fetch_crypto_price(crypto_sym)

    if price_data:
        chg = price_data.get("change_24h_pct") or 0
        with col_kpi:
            kpi_row([
                kpi_card("Price (USD)", f"${price_data.get('price_usd', 0):,.2f}", ""),
                kpi_card("Price (INR)", f"₹{price_data.get('price_inr', 0):,.2f}", ""),
                kpi_card("24h Change", f"{chg:+.2f}%", "",
                         "pos" if chg >= 0 else "neg"),
                kpi_card("Market Cap", f"${price_data.get('market_cap_usd', 0):,.0f}", "USD"),
                kpi_card("24h Volume", f"${price_data.get('volume_24h_usd', 0):,.0f}", "USD"),
            ])

        section_header(f"{crypto_sym} — 1 Year Price History")
        hist_days = st.slider("History (days)", 7, 365, 90, key="crypto_days")
        with st.spinner("Loading history…"):
            hist_df = fetch_crypto_history(crypto_sym, days=hist_days)
        if not hist_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist_df.index, y=hist_df["Close"], name=crypto_sym,
                line=dict(color=T["green"] if chg >= 0 else T["red"], width=1.8),
                fill="tozeroy",
                fillcolor="rgba(63,185,80,0.06)" if chg >= 0 else "rgba(248,81,73,0.06)"))
            fig.update_layout(**{**BASE, "height": 380,
                "title": dict(text=f"{crypto_sym}/USD — {hist_days}D", font_size=12)},
                yaxis_title="Price (USD)")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("History temporarily unavailable from CoinGecko.")
    else:
        st.warning(f"Could not load live price for {crypto_sym}. CoinGecko may be rate-limiting — try again in a moment.")

    section_header("Top 20 Coins by Market Cap")
    with st.spinner("Loading top movers…"):
        movers_df = fetch_crypto_top_movers(20)
    if not movers_df.empty:
        gainers = movers_df.sort_values("24h %", ascending=False)
        td = "border-bottom:1px solid #21262D;font-family:'IBM Plex Mono',monospace"
        rows_h = ""
        for _, r in gainers.iterrows():
            c_ = "#3FB950" if (r["24h %"] or 0) >= 0 else "#F85149"
            rows_h += (
                f'<tr><td style="padding:6px 10px;{td};font-size:12px;font-weight:600;'
                f'color:#C9D1D9">{esc(r["Symbol"])}</td>'
                f'<td style="padding:6px 10px;{td};font-size:11px;color:#8B949E">{esc(r["Name"])}</td>'
                f'<td style="padding:6px 10px;{td};font-size:12px;color:#C9D1D9">${r["Price USD"]:,.4g}</td>'
                f'<td style="padding:6px 10px;{td};font-size:12px;font-weight:600;color:{c_}">'
                f'{r["24h %"]:+.2f}%</td>'
                f'<td style="padding:6px 10px;{td};font-size:11px;color:#8B949E">${r["Market Cap"]:,.0f}</td></tr>')
        th_ = ("padding:7px 10px;text-align:left;font-family:'IBM Plex Mono',monospace;"
               "font-size:9px;color:#8B949E;text-transform:uppercase")
        st.markdown(
            f'<table style="width:100%;border-collapse:collapse;background:#161B22;'
            f'border:1px solid #30363D;border-radius:6px;overflow:hidden">'
            f'<thead><tr style="background:#21262D">'
            f'<th style="{th_}">Symbol</th><th style="{th_}">Name</th>'
            f'<th style="{th_}">Price</th><th style="{th_}">24h</th>'
            f'<th style="{th_}">Market Cap</th></tr></thead>'
            f'<tbody>{rows_h}</tbody></table>',
            unsafe_allow_html=True)
    else:
        st.info("Top movers temporarily unavailable.")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — FX RATES (Frankfurter, no key)
# ═══════════════════════════════════════════════════════════════════════════
with tabs[1]:
    section_header("Live Exchange Rates")
    COMMON_PAIRS = [("USD", "INR"), ("EUR", "INR"), ("GBP", "INR"),
                    ("USD", "EUR"), ("USD", "GBP"), ("USD", "JPY")]

    cols = st.columns(3)
    for i, (base, quote) in enumerate(COMMON_PAIRS):
        with cols[i % 3]:
            rate = fetch_fx_rate(base, quote)
            st.markdown(
                f'<div style="background:#161B22;border:1px solid #30363D;'
                f'border-radius:8px;padding:14px;text-align:center;margin-bottom:10px">'
                f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;'
                f'color:#8B949E">{base}/{quote}</div>'
                f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:22px;'
                f'font-weight:600;color:#C9D1D9">'
                f'{f"{rate:,.4f}" if rate else "—"}</div></div>',
                unsafe_allow_html=True)

    section_header("Custom Pair + History")
    col_a, col_b, col_c = st.columns([1, 1, 2])
    with col_a:
        base_in = st.text_input("From", value="USD", max_chars=3).upper().strip()
    with col_b:
        quote_in = st.text_input("To", value="INR", max_chars=3).upper().strip()
    with col_c:
        fx_days = st.slider("History (days)", 7, 365, 90, key="fx_days")

    rate_now = fetch_fx_rate(base_in, quote_in)
    if rate_now:
        st.markdown(
            f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:13px;'
            f'color:#3FB950;margin:8px 0 16px">'
            f'1 {base_in} = {rate_now:,.4f} {quote_in}</div>',
            unsafe_allow_html=True)
        with st.spinner("Loading FX history…"):
            fx_hist = fetch_fx_history(base_in, quote_in, days=fx_days)
        if not fx_hist.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fx_hist.index, y=fx_hist.values, name=f"{base_in}/{quote_in}",
                line=dict(color=T["blue"], width=1.8),
                fill="tozeroy", fillcolor="rgba(88,166,255,0.06)"))
            fig.update_layout(**{**BASE, "height": 360,
                "title": dict(text=f"{base_in}/{quote_in} — {fx_days}D (ECB data)", font_size=12)},
                yaxis_title=f"{quote_in} per {base_in}")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.warning(f"Could not fetch {base_in}/{quote_in} — check the currency codes are valid ISO-3 (e.g. USD, INR, EUR).")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — MACRO INDICATORS (World Bank, no key)
# ═══════════════════════════════════════════════════════════════════════════
with tabs[2]:
    section_header("Country Macro Indicators")
    col_country, col_indicator = st.columns(2)
    with col_country:
        country_name = st.selectbox("Country", list(WORLDBANK_COUNTRIES.keys()), index=0)
    with col_indicator:
        indicator_name = st.selectbox("Indicator", list(WORLDBANK_INDICATORS.keys()), index=0)

    country_code   = WORLDBANK_COUNTRIES[country_name]
    indicator_code = WORLDBANK_INDICATORS[indicator_name]

    with st.spinner("Loading World Bank data…"):
        macro_series = fetch_macro_indicator(country_code, indicator_code)

    if not macro_series.empty:
        latest_year = macro_series.index.max()
        latest_val  = macro_series.iloc[-1]
        kpi_row([
            kpi_card(f"{country_name} — {indicator_name}",
                     f"{latest_val:,.2f}", f"as of {latest_year}"),
        ])
        fig = go.Figure(go.Bar(
            x=[str(y) for y in macro_series.index], y=macro_series.values,
            marker_color=T["amber"], opacity=0.85))
        fig.update_layout(**{**BASE, "height": 380,
            "title": dict(text=f"{country_name} — {indicator_name}", font_size=12)})
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No data returned for this country/indicator combination.")

    section_header("Compare Across Countries")
    compare_countries = st.multiselect(
        "Countries to compare", list(WORLDBANK_COUNTRIES.keys()),
        default=["India", "United States"])
    if compare_countries:
        fig2 = go.Figure()
        palette = [T["green"], T["blue"], T["amber"], T["purple"], T["orange"], T["red"]]
        for i, cname in enumerate(compare_countries):
            ccode = WORLDBANK_COUNTRIES[cname]
            s = fetch_macro_indicator(ccode, indicator_code)
            if not s.empty:
                fig2.add_trace(go.Scatter(
                    x=[str(y) for y in s.index], y=s.values, name=cname,
                    line=dict(color=palette[i % len(palette)], width=2)))
        fig2.update_layout(**{**BASE, "height": 380,
            "title": dict(text=f"{indicator_name} — Country Comparison", font_size=12)})
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — SEC FILINGS (EDGAR, no key, US tickers only)
# ═══════════════════════════════════════════════════════════════════════════
with tabs[3]:
    section_header("SEC EDGAR Filings Lookup")
    st.caption("US-listed companies only. Pulls directly from SEC's public EDGAR system — no key needed.")
    sec_ticker = st.text_input("US Ticker", value="AAPL", max_chars=10).upper().strip()
    form_filter = st.multiselect(
        "Filter by form type", ["10-K", "10-Q", "8-K", "DEF 14A", "4", "13F-HR"],
        default=["10-K", "10-Q"])

    if sec_ticker:
        with st.spinner(f"Looking up {sec_ticker} on SEC EDGAR…"):
            cik = fetch_sec_cik(sec_ticker)
        if cik:
            st.caption(f"CIK: {cik}")
            with st.spinner("Loading filings…"):
                filings_df = fetch_sec_filings(
                    sec_ticker, form_types=form_filter or None, limit=20)
            if not filings_df.empty:
                td = "border-bottom:1px solid #21262D;font-family:'IBM Plex Mono',monospace"
                rows_h = ""
                for _, r in filings_df.iterrows():
                    rows_h += (
                        f'<tr><td style="padding:7px 10px;{td};font-size:11px;'
                        f'font-weight:600;color:#3FB950">{esc(r["Form"])}</td>'
                        f'<td style="padding:7px 10px;{td};font-size:11px;color:#8B949E">{esc(r["Date"])}</td>'
                        f'<td style="padding:7px 10px;{td};font-size:11px;color:#C9D1D9">'
                        f'<a href="{r["URL"]}" target="_blank" style="color:#58A6FF;text-decoration:none">'
                        f'{esc(r["Description"] or r["Document"])}</a></td></tr>')
                th_ = ("padding:8px 10px;text-align:left;font-family:'IBM Plex Mono',monospace;"
                       "font-size:9px;color:#8B949E;text-transform:uppercase")
                st.markdown(
                    f'<table style="width:100%;border-collapse:collapse;background:#161B22;'
                    f'border:1px solid #30363D;border-radius:6px;overflow:hidden">'
                    f'<thead><tr style="background:#21262D">'
                    f'<th style="{th_}">Form</th><th style="{th_}">Date</th>'
                    f'<th style="{th_}">Document</th></tr></thead>'
                    f'<tbody>{rows_h}</tbody></table>',
                    unsafe_allow_html=True)
            else:
                st.info("No filings matched the selected form types.")
        else:
            st.warning(f"Could not find {sec_ticker} in SEC EDGAR. This works for US-listed companies only — NSE/BSE stocks file with SEBI, not the SEC.")

footer_bar()
