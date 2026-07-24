"""
Insights — Sentiment, Alternative Data, and the Rule-Based Assistant.

Three zero-signup features in one page:
  1. News sentiment analysis (VADER, offline, no API key)
  2. SEC EDGAR filing sentiment (free public filings + local NLP)
  3. Rule-based Q&A assistant that answers from real computed data
     (not an LLM — see core/assistant.py docstring for why)

Also surfaces personalized stock recommendations for logged-in users,
based on their own viewing history (see core/personalization.py).
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.data_fetcher import fetch_ohlcv, fetch_news, validate_ticker, currency_symbol, detect_market
from core.external_data import fetch_sec_filings, fetch_sec_cik
from core.sentiment import analyze_news_sentiment, analyze_filing_sentiment, sentiment_time_series
from core.assistant import answer_question, SUGGESTED_QUESTIONS
from core.personalization import (track_view, get_favorite_sectors,
                                  recommend_similar_stocks, get_user_stats)
from utils.helpers import (inject_css, section_header, kpi_row, kpi_card,
                           esc, footer_bar, top_bar_simple)
from utils.charts import T, BASE, safe_layout
import plotly.graph_objects as go
inject_css()

NIFTY50_SAMPLE = [
    {"ticker": "RELIANCE.NS", "name": "Reliance", "sector": "Energy"},
    {"ticker": "TCS.NS", "name": "TCS", "sector": "IT"},
    {"ticker": "HDFCBANK.NS", "name": "HDFC Bank", "sector": "Banking"},
    {"ticker": "INFY.NS", "name": "Infosys", "sector": "IT"},
    {"ticker": "ICICIBANK.NS", "name": "ICICI Bank", "sector": "Banking"},
    {"ticker": "HCLTECH.NS", "name": "HCL Tech", "sector": "IT"},
    {"ticker": "WIPRO.NS", "name": "Wipro", "sector": "IT"},
    {"ticker": "AXISBANK.NS", "name": "Axis Bank", "sector": "Banking"},
    {"ticker": "SUNPHARMA.NS", "name": "Sun Pharma", "sector": "Pharma"},
    {"ticker": "TITAN.NS", "name": "Titan", "sector": "Consumer"},
]

with st.sidebar:
    st.divider()
    ticker = st.text_input("Ticker Symbol", value="AAPL",
                           placeholder="AAPL · RELIANCE.NS").upper().strip()
    st.divider()
    st.caption("VADER sentiment: offline, no API key. SEC EDGAR: free public filings, US tickers only.")

mkt = detect_market(ticker) if ticker else "US"
_sym = currency_symbol("INR" if mkt in ("NSE", "BSE") else "USD")
_flag = "🇮🇳" if mkt in ("NSE", "BSE") else "🇺🇸"

top_bar_simple("Insights", f"{_flag} {ticker} · Sentiment · Assistant · Recommendations")

if not ticker:
    st.info("Enter a ticker symbol in the sidebar.")
    st.stop()

valid, err = validate_ticker(ticker)
if not valid:
    st.error(f"**{ticker}** — {err}")
    st.stop()

with st.spinner(f"Loading {ticker}…"):
    df = fetch_ohlcv(ticker, "1y", "1d")
    news = fetch_news(ticker, max_items=15)

if df.empty:
    st.error("No price data available.")
    st.stop()

user = st.session_state.get("user")
username = user.get("username") if user else None
sector_lookup = {item["ticker"]: item["sector"] for item in NIFTY50_SAMPLE}
track_view(username, ticker, sector_lookup.get(ticker, "—"))

tabs = st.tabs(["  💬 News Sentiment  ", "  📄 SEC Filings  ", "  🤖 Ask the Assistant  ", "  ⭐ For You  "])

with tabs[0]:
    section_header("News Sentiment Analysis")
    if not news:
        st.info(f"No recent news headlines available for {ticker}.")
    else:
        report = analyze_news_sentiment(news)
        label_color = {"Positive": "#3FB950", "Negative": "#F85149", "Neutral": "#8B949E"}
        kpi_row([
            kpi_card("Overall Sentiment", report["overall_label"], f"score: {report['overall_score']:+.2f}",
                     "pos" if report["overall_label"] == "Positive" else
                     ("neg" if report["overall_label"] == "Negative" else "")),
            kpi_card("Headlines Analyzed", str(report["n_items"]), ""),
            kpi_card("Positive", str(report["pos_count"]), "", "pos"),
            kpi_card("Negative", str(report["neg_count"]), "", "neg" if report["neg_count"] else ""),
            kpi_card("Neutral", str(report["neu_count"]), ""),
        ])

        col_chart, col_list = st.columns([1, 1])
        with col_chart:
            section_header("Sentiment Over Time")
            ts = sentiment_time_series(news)
            if not ts.empty and len(ts) > 1:
                colors = ["#3FB950" if v >= 0 else "#F85149" for v in ts["score"]]
                fig = go.Figure(go.Bar(x=ts.index, y=ts["score"], marker_color=colors))
                fig.add_hline(y=0, line_color=T["dim"], line_dash="dot")
                fig.update_layout(**safe_layout({}, height=320, title="Daily Average Sentiment"))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("Not enough dated headlines to show a trend.")

        with col_list:
            section_header("Headlines Ranked by Sentiment Strength")
            for item in report["items"][:8]:
                color = label_color.get(item["label"], "#8B949E")
                st.markdown(
                    f'<div style="background:#161B22;border-left:3px solid {color};'
                    f'border-radius:4px;padding:8px 10px;margin-bottom:6px">'
                    f'<div style="font-size:12px;color:#C9D1D9">{esc(item["title"])}</div>'
                    f'<div style="font-size:10px;color:{color};margin-top:3px;'
                    f'font-family:\'IBM Plex Mono\',monospace">{item["label"]} '
                    f'({item["compound"]:+.2f}) · {esc(item["publisher"])}</div></div>',
                    unsafe_allow_html=True)

with tabs[1]:
    section_header("SEC EDGAR Filing Sentiment")
    if mkt != "US":
        st.info("SEC EDGAR only covers US-listed companies (NSE/BSE stocks file with SEBI, not the SEC).")
    else:
        with st.spinner("Looking up SEC filings…"):
            cik = fetch_sec_cik(ticker)
        if not cik:
            st.warning(f"Could not find {ticker} in SEC EDGAR.")
        else:
            with st.spinner("Loading recent filings…"):
                filings_df = fetch_sec_filings(ticker, form_types=["10-K", "10-Q"], limit=5)
            if filings_df.empty:
                st.info("No recent 10-K/10-Q filings found.")
            else:
                st.caption(f"CIK: {cik} · Showing {len(filings_df)} most recent filings")
                for _, row in filings_df.iterrows():
                    st.markdown(
                        f'<div style="background:#161B22;border:1px solid #30363D;border-radius:6px;'
                        f'padding:10px 14px;margin-bottom:8px">'
                        f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:12px;'
                        f'font-weight:600;color:#3FB950">{esc(row["Form"])}</span>'
                        f'<span style="font-size:11px;color:#8B949E;margin-left:10px">{esc(row["Date"])}</span>'
                        f'<br><a href="{row["URL"]}" target="_blank" style="font-size:11px;color:#58A6FF;'
                        f'text-decoration:none">{esc(row["Description"] or row["Document"])}</a>'
                        f'</div>', unsafe_allow_html=True)
                st.caption(
                    "Sentiment scoring on full filing text requires fetching and parsing the "
                    "linked document (large HTML/XBRL files) — click through to read directly, "
                    "or paste an excerpt below for a quick sentiment read."
                )
                excerpt = st.text_area("Paste a filing excerpt to analyze (e.g. Risk Factors section)",
                                       height=150, placeholder="Paste text here...")
                if excerpt.strip():
                    filing_sentiment = analyze_filing_sentiment(excerpt)
                    kpi_row([
                        kpi_card("Sentiment", filing_sentiment["overall_label"],
                                f"score: {filing_sentiment['overall_score']:+.2f}"),
                        kpi_card("Sentences Analyzed", str(filing_sentiment["n_sentences"]), ""),
                    ])
                    if filing_sentiment["n_sentences"] > 0:
                        st.markdown(f"**Most positive:** {filing_sentiment['most_positive_sentence']}")
                        st.markdown(f"**Most negative:** {filing_sentiment['most_negative_sentence']}")

with tabs[2]:
    section_header("Ask About This Stock")
    st.caption(
        "🤖 Rule-based assistant — answers come directly from this app's own "
        "computed indicators, risk metrics, and forecasts. Not an LLM, so it "
        "only understands a specific set of question types (see suggestions below), "
        "but every number it gives you is real, not generated."
    )

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    cols = st.columns(3)
    for i, sq in enumerate(SUGGESTED_QUESTIONS):
        with cols[i % 3]:
            if st.button(sq, key=f"suggested_{i}", use_container_width=True):
                st.session_state["pending_question"] = sq

    question = st.chat_input("Ask a question about this stock…")
    if "pending_question" in st.session_state:
        question = st.session_state.pop("pending_question")

    if question:
        result = answer_question(question, df, ticker, currency_sym=_sym, news_items=news)
        st.session_state["chat_history"].append({"q": question, "a": result["answer"]})

    for exchange in reversed(st.session_state["chat_history"][-10:]):
        with st.chat_message("user"):
            st.write(exchange["q"])
        with st.chat_message("assistant"):
            st.write(exchange["a"])

    if st.session_state["chat_history"] and st.button("🗑️ Clear conversation"):
        st.session_state["chat_history"] = []
        st.rerun()

with tabs[3]:
    section_header("Recommended For You")
    if not username:
        st.info(
            "🔐 Log in from the sidebar to get personalized recommendations "
            "based on the sectors and stocks you view most often."
        )
    else:
        stats = get_user_stats(username)
        kpi_row([
            kpi_card("Stocks Viewed", str(stats["total_views"]), "total views"),
            kpi_card("Unique Tickers", str(stats["unique_tickers"]), ""),
            kpi_card("Top Sector", stats["top_sectors"][0] if stats["top_sectors"] else "—", ""),
        ])

        if not stats["top_sectors"]:
            st.info("Browse a few more stocks (Overview, Technical Analysis, etc.) and recommendations will appear here.")
        else:
            st.caption(f"Based on your interest in: {', '.join(stats['top_sectors'])}")
            recs = recommend_similar_stocks(username, NIFTY50_SAMPLE, limit=6)
            if not recs:
                st.info("No new recommendations right now — you may have already viewed everything in your favorite sectors.")
            else:
                cols = st.columns(3)
                for i, rec in enumerate(recs):
                    with cols[i % 3]:
                        st.markdown(
                            f'<div style="background:#161B22;border:1px solid #30363D;'
                            f'border-radius:8px;padding:14px;margin-bottom:10px">'
                            f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:14px;'
                            f'font-weight:600;color:#C9D1D9">{esc(rec["ticker"])}</div>'
                            f'<div style="font-size:11px;color:#8B949E;margin:4px 0">{esc(rec["name"])}</div>'
                            f'<div style="font-size:10px;color:#3FB950">{esc(rec["match_reason"])}</div>'
                            f'</div>', unsafe_allow_html=True)

footer_bar()
