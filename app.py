"""
StockPro Analytics — Application Entry Point.

This file owns:
  - Global page config (set_page_config called exactly once)
  - Navigation definition via st.navigation() — gives clean labels
    regardless of what the page file is named on disk
  - Shared sidebar chrome: brand logo + login placeholder
  - Global CSS injection

Each page runs as its own script via pg.run(); see pages/*.py.
"""

import streamlit as st
import sys
sys.path.insert(0, ".")

st.set_page_config(
    page_title="StockPro Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Professional Stock Market Analysis Platform · v6"},
)

from utils.helpers import inject_css, sidebar_brand, sidebar_user, auth_widget
from core.notifications import notification_bell
inject_css()

# ── Sidebar: brand + notifications ────────────────────────────────────────────
with st.sidebar:
    sidebar_brand()
    notification_bell()
    st.divider()

# ── Navigation (clean labels, grouped sections, emoji icons) ─────────────────
# st.navigation() overrides the auto-generated sidebar nav that Streamlit
# normally builds from filenames — so the emoji-bytes encoding issue that
# caused garbled labels like "≡ƒô⌐ Technical Analysis" is gone.
pg = st.navigation(
    {
        "Overview": [
            st.Page("pages/overview.py",
                    title="Dashboard", icon="📊", default=True),
        ],
        "Analysis": [
            st.Page("pages/technical_analysis.py",
                    title="Technical Analysis", icon="📈"),
            st.Page("pages/price_prediction.py",
                    title="Price Prediction",   icon="🔮"),
            st.Page("pages/risk_analysis.py",
                    title="Risk Analysis",       icon="⚠️"),
        ],
        "Quant": [
            st.Page("pages/backtester.py",
                    title="Strategy Backtester", icon="📊"),
            st.Page("pages/factor_analysis.py",
                    title="Factor Analysis",     icon="🧮"),
            st.Page("pages/insights.py",
                    title="Insights",            icon="🤖"),
        ],
        "Portfolio": [
            st.Page("pages/portfolio.py",
                    title="Portfolio Tracker", icon="💼"),
            st.Page("pages/watchlist.py",
                    title="Watchlist",         icon="⭐"),
        ],
        "Markets": [
            st.Page("pages/screener.py",
                    title="Screener",          icon="🔍"),
            st.Page("pages/compare.py",
                    title="Compare",           icon="⚖️"),
            st.Page("pages/market_overview.py",
                    title="Market Overview",   icon="🌍"),
            st.Page("pages/global_data.py",
                    title="Global Data",       icon="🔓"),
        ],
    },
    position="sidebar",
)

pg.run()

# ── Sidebar: user / login (rendered AFTER pg.run() so it appears below nav) ──
with st.sidebar:
    st.divider()
    # ── Local auth — zero external accounts, zero API keys ─────────────────
    # Credentials live in data/users.json (hashlib-based, see core/auth.py).
    # Streamlit Cloud's filesystem is ephemeral: self-registered accounts
    # persist only until the app restarts, unless you commit data/users.json
    # to git (same pattern as the ML checkpoint in models/). See core/auth.py
    # module docstring for details and the upgrade path to a real database.
    user = st.session_state.get("user")
    if user:
        sidebar_user(
            name=user.get("name", "User"),
            role=user.get("role", "Free tier"),
            initials=(user.get("name", "U")[:2].upper()),
        )
        if st.button("🚪  Logout", use_container_width=True):
            del st.session_state["user"]
            st.rerun()
    else:
        auth_widget()

    st.caption("Data via Yahoo Finance · Not financial advice")
