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

from utils.helpers import inject_css, sidebar_brand, sidebar_user
inject_css()

# ── Sidebar: brand + user section ────────────────────────────────────────────
with st.sidebar:
    sidebar_brand()
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
    # ── Auth integration point ──────────────────────────────────────────────
    # To add real authentication, replace sidebar_user() with your chosen
    # provider (Streamlit-Authenticator, Auth0, Google OAuth, etc.) and
    # store the user object in st.session_state["user"].
    #
    # Example with streamlit-authenticator:
    #   import streamlit_authenticator as stauth
    #   authenticator = stauth.Authenticate(credentials, ...)
    #   name, auth_status, username = authenticator.login('Login', 'sidebar')
    #   if auth_status:
    #       sidebar_user(name=name, role="Pro", initials=name[:2].upper())
    #   else:
    #       sidebar_user()  # shows guest login button
    # ────────────────────────────────────────────────────────────────────────
    user = st.session_state.get("user")
    if user:
        sidebar_user(
            name=user.get("name", "User"),
            role=user.get("role", "Pro"),
            initials=(user.get("name", "U")[:2].upper()),
        )
    else:
        sidebar_user()   # Guest view with Login placeholder button

    st.caption("Data via Yahoo Finance · Not financial advice")
