"""Shared dated market snapshots for app consumers; fetching never rewrites source age."""
import streamlit as st
from core.data.providers import YahooMarketDataProvider

@st.cache_data(ttl=300,show_spinner=False)
def market_snapshot(ticker):
    return YahooMarketDataProvider().capture_history(ticker)
