"""
Page 7 — Market Overview
Global indices snapshot, NSE & US top movers, sector performance heatmap,
and intraday volume breadth. Pure Streamlit — no external AI APIs.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

st.set_page_config(
    page_title="Market Overview · StockPro", page_icon="🌍",
    layout="wide", initial_sidebar_state="expanded"
)

from core.data_fetcher import fetch_ohlcv, currency_symbol
from utils.helpers     import inject_css, section_header, kpi_row, kpi_card
import plotly.graph_objects as go
from plotly.subplots import make_subplots

inject_css()

# ── Constants ────────────────────────────────────────────────────────────────
GLOBAL_INDICES = [
    # (display name, yf symbol, currency, region)
    ("Nifty 50",     "^NSEI",   "INR", "🇮🇳 India"),
    ("Bank Nifty",   "^NSEBANK","INR", "🇮🇳 India"),
    ("Sensex",       "^BSESN",  "INR", "🇮🇳 India"),
    ("S&P 500",      "^GSPC",   "USD", "🇺🇸 US"),
    ("Nasdaq 100",   "^NDX",    "USD", "🇺🇸 US"),
    ("Dow Jones",    "^DJI",    "USD", "🇺🇸 US"),
    ("Russell 2000", "^RUT",    "USD", "🇺🇸 US"),
    ("FTSE 100",     "^FTSE",   "GBP", "🇬🇧 UK"),
    ("DAX",          "^GDAXI",  "EUR", "🇩🇪 Germany"),
    ("Nikkei 225",   "^N225",   "JPY", "🇯🇵 Japan"),
    ("Hang Seng",    "^HSI",    "HKD", "🇭🇰 HK"),
    ("VIX",          "^VIX",    "USD", "🌐 Volatility"),
]

NSE_NIFTY50 = [
    ("RELIANCE.NS", "Reliance",      "Energy"),
    ("TCS.NS",      "TCS",           "IT"),
    ("HDFCBANK.NS", "HDFC Bank",     "Banking"),
    ("INFY.NS",     "Infosys",       "IT"),
    ("ICICIBANK.NS","ICICI Bank",    "Banking"),
    ("HINDUNILVR.NS","HUL",          "FMCG"),
    ("ITC.NS",      "ITC",           "FMCG"),
    ("SBIN.NS",     "SBI",           "Banking"),
    ("BHARTIARTL.NS","Airtel",       "Telecom"),
    ("KOTAKBANK.NS","Kotak Bank",    "Banking"),
    ("LT.NS",       "L&T",           "Industrials"),
    ("AXISBANK.NS", "Axis Bank",     "Banking"),
    ("ASIANPAINT.NS","Asian Paints", "Consumer"),
    ("MARUTI.NS",   "Maruti",        "Auto"),
    ("HCLTECH.NS",  "HCL Tech",      "IT"),
    ("SUNPHARMA.NS","Sun Pharma",    "Pharma"),
    ("TITAN.NS",    "Titan",         "Consumer"),
    ("BAJFINANCE.NS","Bajaj Fin",    "NBFC"),
    ("WIPRO.NS",    "Wipro",         "IT"),
    ("TATAMOTORS.NS","Tata Motors",  "Auto"),
]

US_WATCHLIST = [
    ("AAPL",  "Apple",      "Technology"),
    ("MSFT",  "Microsoft",  "Technology"),
    ("NVDA",  "Nvidia",     "Technology"),
    ("GOOGL", "Alphabet",   "Technology"),
    ("META",  "Meta",       "Communication"),
    ("AMZN",  "Amazon",     "Consumer"),
    ("TSLA",  "Tesla",      "Consumer"),
    ("JPM",   "JP Morgan",  "Financials"),
    ("V",     "Visa",       "Financials"),
    ("XOM",   "ExxonMobil", "Energy"),
    ("UNH",   "UnitedHlth", "Healthcare"),
    ("JNJ",   "J&J",        "Healthcare"),
    ("WMT",   "Walmart",    "Staples"),
    ("HD",    "Home Depot", "Consumer"),
    ("BAC",   "BofA",       "Financials"),
    ("NFLX",  "Netflix",    "Communication"),
    ("AMD",   "AMD",        "Technology"),
    ("INTC",  "Intel",      "Technology"),
    ("BA",    "Boeing",     "Industrials"),
    ("GS",    "Goldman",    "Financials"),
]

NSE_SECTORS = ["IT","Banking","FMCG","Pharma","Auto","Energy","Industrials","NBFC","Consumer","Telecom"]
US_SECTORS  = ["Technology","Financials","Healthcare","Consumer","Communication","Energy","Industrials","Staples"]

LAYOUT = dict(
    plot_bgcolor="#0D1117", paper_bgcolor="#0D1117",
    font=dict(family="IBM Plex Mono, monospace", color="#C9D1D9", size=11),
    margin=dict(l=12, r=12, t=36, b=12),
    xaxis=dict(gridcolor="#21262D", zeroline=False),
    yaxis=dict(gridcolor="#21262D", zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style="font-family:'IBM Plex Mono',monospace;font-size:16px;
    font-weight:600;color:#3FB950;padding:8px 0 16px;">📈 StockPro
    <span style="font-size:10px;color:#8B949E;font-weight:400;display:block;
    letter-spacing:.1em;margin-top:2px;">ANALYTICS TERMINAL</span></div>""",
    unsafe_allow_html=True)

    market_focus = st.radio(
        "Market Focus",
        ["🌍 Global", "🇮🇳 India (NSE)", "🇺🇸 US"],
        index=0,
    )
    period_label = st.selectbox(
        "Chart Period",
        ["1 Month", "3 Months", "6 Months", "1 Year"],
        index=2,
    )
    period_map = {"1 Month": "1mo", "3 Months": "3mo",
                  "6 Months": "6mo", "1 Year": "1y"}
    period = period_map[period_label]

    n_movers = st.slider("Top movers to show", 5, 20, 10, 5)

    st.divider()
    st.caption("Data via Yahoo Finance · Not financial advice")

# ── Header ────────────────────────────────────────────────────────────────────
from datetime import datetime
now_str = datetime.utcnow().strftime("%d %b %Y  %H:%M UTC")
st.markdown(f"""<div style="font-family:'IBM Plex Mono',monospace;padding:10px 0 6px;
border-bottom:1px solid #30363D;margin-bottom:16px;display:flex;
align-items:baseline;justify-content:space-between;">
  <div>
    <span style="font-size:20px;font-weight:600;color:#C9D1D9">Market Overview</span>&nbsp;&nbsp;
    <span style="font-size:13px;color:#8B949E">{market_focus}</span>
  </div>
  <span style="font-size:11px;color:#8B949E;font-family:'IBM Plex Mono',monospace">
    Last updated: {now_str}
  </span>
</div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — GLOBAL INDICES TICKER STRIP
# ═══════════════════════════════════════════════════════════════════════════════
section_header("Global Indices")

@st.cache_data(ttl=300, show_spinner=False)
def load_index(symbol: str, period: str = "5d") -> dict:
    try:
        df = fetch_ohlcv(symbol, period, "1d")
        if df.empty or len(df) < 2:
            return {}
        last  = float(df["Close"].iloc[-1])
        prev  = float(df["Close"].iloc[-2])
        chg   = (last - prev) / prev * 100
        hi52  = float(df["Close"].max())  # rough approximation within period
        lo52  = float(df["Close"].min())
        return {"last": last, "chg": chg, "hi52": hi52, "lo52": lo52,
                "series": df["Close"]}
    except Exception:
        return {}

# Load all indices (cached)
with st.spinner("Loading indices…"):
    idx_data = {}
    for name, sym, curr, region in GLOBAL_INDICES:
        d = load_index(sym)
        if d:
            idx_data[(name, sym, curr, region)] = d

# Render as a responsive card grid
cards_html = ""
for (name, sym, curr, region), d in idx_data.items():
    last  = d["last"]
    chg   = d["chg"]
    sym_c = currency_symbol(curr)
    arrow = "▲" if chg >= 0 else "▼"
    col   = "#3FB950" if chg >= 0 else "#F85149"
    fmt   = f"{last:,.0f}" if last > 999 else f"{last:,.2f}"
    bg    = "rgba(63,185,80,0.05)" if chg >= 0 else "rgba(248,81,73,0.05)"
    cards_html += f"""
    <div style="background:#161B22;border:1px solid #30363D;border-radius:8px;
    padding:12px 14px;min-width:140px;border-top:2px solid {col};">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:#8B949E;
                  text-transform:uppercase;letter-spacing:.07em;margin-bottom:4px">
        {region}</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:12px;font-weight:600;
                  color:#C9D1D9;margin-bottom:6px">{name}</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:16px;font-weight:600;
                  color:#C9D1D9">{fmt}</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:12px;color:{col};margin-top:3px">
        {arrow} {chg:+.2f}%</div>
    </div>"""

st.markdown(f"""
<div style="display:flex;flex-wrap:wrap;gap:10px;margin-bottom:20px">
  {cards_html}
</div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PERFORMANCE CHART
# ═══════════════════════════════════════════════════════════════════════════════
section_header(f"Index Performance — {period_label} (Indexed to 100)")

# Pick which indices to chart based on market focus
if "India" in market_focus:
    chart_indices = [("Nifty 50","^NSEI","#3FB950"),
                     ("Bank Nifty","^NSEBANK","#58A6FF"),
                     ("Sensex","^BSESN","#E3B341")]
elif "US" in market_focus:
    chart_indices = [("S&P 500","^GSPC","#3FB950"),
                     ("Nasdaq","^NDX","#58A6FF"),
                     ("Dow Jones","^DJI","#E3B341"),
                     ("Russell 2000","^RUT","#BC8CFF")]
else:
    chart_indices = [("Nifty 50","^NSEI","#3FB950"),
                     ("S&P 500","^GSPC","#58A6FF"),
                     ("Nasdaq","^NDX","#E3B341"),
                     ("FTSE 100","^FTSE","#BC8CFF"),
                     ("Nikkei 225","^N225","#FFA657")]

@st.cache_data(ttl=300, show_spinner=False)
def load_series(symbol: str, period: str) -> pd.Series:
    try:
        df = fetch_ohlcv(symbol, period, "1d")
        if df.empty:
            return pd.Series(dtype=float)
        return df["Close"]
    except Exception:
        return pd.Series(dtype=float)

fig_idx = go.Figure()
for name, sym, colour in chart_indices:
    s = load_series(sym, period)
    if s.empty or len(s) < 2:
        continue
    normed = s / s.iloc[0] * 100
    fig_idx.add_trace(go.Scatter(
        x=normed.index, y=normed.values,
        name=name, line=dict(color=colour, width=2),
    ))
fig_idx.add_hline(y=100, line_color="#8B949E", line_dash="dot", line_width=0.8)
fig_idx.update_layout(**{**LAYOUT, "height": 380,
    "title": dict(text="Performance Indexed to 100 (start of period)", font_size=12)},
    yaxis_title="Indexed (100 = start)")
st.plotly_chart(fig_idx, use_container_width=True, config={"displayModeBar": False})

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — TOP MOVERS
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=300, show_spinner=False)
def compute_movers(watchlist: list, period: str, n: int) -> pd.DataFrame:
    rows = []
    for sym, name, sector in watchlist:
        try:
            df = fetch_ohlcv(sym, period, "1d")
            if df.empty or len(df) < 2:
                continue
            last    = float(df["Close"].iloc[-1])
            prev    = float(df["Close"].iloc[-2])
            pct_1d  = (last - prev) / prev * 100
            pct_per = (last / float(df["Close"].iloc[0]) - 1) * 100
            vol_r   = (float(df["Volume"].iloc[-1]) /
                       float(df["Volume"].mean()) if df["Volume"].mean() > 0 else 1)
            rows.append({"Symbol": sym, "Name": name, "Sector": sector,
                         "Last": last, "1D %": round(pct_1d, 2),
                         "Period %": round(pct_per, 2),
                         "Vol Ratio": round(vol_r, 2)})
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    df_out = pd.DataFrame(rows)
    return df_out.sort_values("1D %", ascending=False)

tabs_m = st.tabs([
    "  🇮🇳 NSE Top Movers  ",
    "  🇺🇸 US Top Movers  ",
    "  📊 Sector Heatmap  ",
])

# ── TAB 1: NSE Movers ────────────────────────────────────────────────────────
with tabs_m[0]:
    section_header("NSE — Nifty 50 Movers")
    with st.spinner("Loading NSE data…"):
        nse_df = compute_movers(NSE_NIFTY50, period, n_movers)

    if nse_df.empty:
        st.warning("Could not load NSE data. Check your internet connection.")
    else:
        top_g   = nse_df.head(n_movers // 2)
        top_l   = nse_df.tail(n_movers // 2).sort_values("1D %")

        col1, col2 = st.columns(2)
        for col, title, subdf, colour in [
            (col1, "🟢 Top Gainers",  top_g, "#3FB950"),
            (col2, "🔴 Top Losers",   top_l, "#F85149"),
        ]:
            with col:
                st.markdown(f"""<div style="font-family:'IBM Plex Mono',monospace;
                font-size:11px;color:{colour};text-transform:uppercase;
                letter-spacing:.08em;margin-bottom:8px">{title}</div>""",
                unsafe_allow_html=True)
                rows_h = ""
                for _, row in subdf.iterrows():
                    c   = "#3FB950" if row["1D %"] >= 0 else "#F85149"
                    sgn = "+" if row["1D %"] >= 0 else ""
                    rows_h += f"""<tr>
                      <td style="padding:6px 10px;border-bottom:1px solid #21262D;
                                 font-family:'IBM Plex Mono',monospace;font-size:11px;
                                 font-weight:600;color:#C9D1D9">{row['Name']}</td>
                      <td style="padding:6px 10px;border-bottom:1px solid #21262D;
                                 font-size:10px;color:#8B949E">{row['Sector']}</td>
                      <td style="padding:6px 10px;border-bottom:1px solid #21262D;
                                 font-family:'IBM Plex Mono',monospace;font-size:11px;
                                 color:#C9D1D9">₹{row['Last']:,.2f}</td>
                      <td style="padding:6px 10px;border-bottom:1px solid #21262D;
                                 font-family:'IBM Plex Mono',monospace;font-size:12px;
                                 font-weight:600;color:{c}">{sgn}{row['1D %']:.2f}%</td>
                      <td style="padding:6px 10px;border-bottom:1px solid #21262D;
                                 font-family:'IBM Plex Mono',monospace;font-size:10px;
                                 color:#8B949E">{row['Vol Ratio']:.1f}x vol</td>
                    </tr>"""
                st.markdown(f"""<table style="width:100%;border-collapse:collapse;
                background:#161B22;border:1px solid #30363D;border-radius:6px;overflow:hidden">
                <thead><tr style="background:#21262D">
                  <th style="padding:7px 10px;text-align:left;font-family:'IBM Plex Mono',monospace;
                     font-size:9px;color:#8B949E;text-transform:uppercase">Name</th>
                  <th style="padding:7px 10px;text-align:left;font-family:'IBM Plex Mono',monospace;
                     font-size:9px;color:#8B949E;text-transform:uppercase">Sector</th>
                  <th style="padding:7px 10px;text-align:left;font-family:'IBM Plex Mono',monospace;
                     font-size:9px;color:#8B949E;text-transform:uppercase">Price</th>
                  <th style="padding:7px 10px;text-align:left;font-family:'IBM Plex Mono',monospace;
                     font-size:9px;color:#8B949E;text-transform:uppercase">1D</th>
                  <th style="padding:7px 10px;text-align:left;font-family:'IBM Plex Mono',monospace;
                     font-size:9px;color:#8B949E;text-transform:uppercase">Volume</th>
                </tr></thead>
                <tbody>{rows_h}</tbody></table>""", unsafe_allow_html=True)

        # Movers bar chart
        st.markdown("<br>", unsafe_allow_html=True)
        top_n = nse_df.head(n_movers).sort_values("1D %")
        bar_colours = ["#3FB950" if v >= 0 else "#F85149" for v in top_n["1D %"]]
        fig_bar = go.Figure(go.Bar(
            x=top_n["1D %"],
            y=top_n["Name"],
            orientation="h",
            marker_color=bar_colours, opacity=0.85,
            text=[f"{v:+.2f}%" for v in top_n["1D %"]],
            textposition="outside",
            textfont=dict(size=9, family="IBM Plex Mono, monospace", color="#C9D1D9"),
        ))
        fig_bar.add_vline(x=0, line_color="#8B949E", line_dash="dot", line_width=0.8)
        fig_bar.update_layout(**{**LAYOUT, "height": 340,
            "title": dict(text="NSE 1-Day Returns (%)", font_size=12)},
            xaxis_title="1D Return (%)", margin=dict(l=8, r=60, t=36, b=8))
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

# ── TAB 2: US Movers ─────────────────────────────────────────────────────────
with tabs_m[1]:
    section_header("US — S&P 500 Sample Movers")
    with st.spinner("Loading US data…"):
        us_df = compute_movers(US_WATCHLIST, period, n_movers)

    if us_df.empty:
        st.warning("Could not load US data.")
    else:
        top_g = us_df.head(n_movers // 2)
        top_l = us_df.tail(n_movers // 2).sort_values("1D %")

        col1, col2 = st.columns(2)
        for col, title, subdf, colour in [
            (col1, "🟢 Top Gainers", top_g, "#3FB950"),
            (col2, "🔴 Top Losers",  top_l, "#F85149"),
        ]:
            with col:
                st.markdown(f"""<div style="font-family:'IBM Plex Mono',monospace;
                font-size:11px;color:{colour};text-transform:uppercase;
                letter-spacing:.08em;margin-bottom:8px">{title}</div>""",
                unsafe_allow_html=True)
                rows_h = ""
                for _, row in subdf.iterrows():
                    c   = "#3FB950" if row["1D %"] >= 0 else "#F85149"
                    sgn = "+" if row["1D %"] >= 0 else ""
                    rows_h += f"""<tr>
                      <td style="padding:6px 10px;border-bottom:1px solid #21262D;
                                 font-family:'IBM Plex Mono',monospace;font-size:11px;
                                 font-weight:600;color:#C9D1D9">{row['Symbol']}</td>
                      <td style="padding:6px 10px;border-bottom:1px solid #21262D;
                                 font-size:10px;color:#8B949E">{row['Sector']}</td>
                      <td style="padding:6px 10px;border-bottom:1px solid #21262D;
                                 font-family:'IBM Plex Mono',monospace;font-size:11px;
                                 color:#C9D1D9">${row['Last']:,.2f}</td>
                      <td style="padding:6px 10px;border-bottom:1px solid #21262D;
                                 font-family:'IBM Plex Mono',monospace;font-size:12px;
                                 font-weight:600;color:{c}">{sgn}{row['1D %']:.2f}%</td>
                      <td style="padding:6px 10px;border-bottom:1px solid #21262D;
                                 font-family:'IBM Plex Mono',monospace;font-size:10px;
                                 color:#8B949E">{row['Vol Ratio']:.1f}x vol</td>
                    </tr>"""
                st.markdown(f"""<table style="width:100%;border-collapse:collapse;
                background:#161B22;border:1px solid #30363D;border-radius:6px;overflow:hidden">
                <thead><tr style="background:#21262D">
                  <th style="padding:7px 10px;text-align:left;font-family:'IBM Plex Mono',monospace;
                     font-size:9px;color:#8B949E;text-transform:uppercase">Symbol</th>
                  <th style="padding:7px 10px;text-align:left;font-family:'IBM Plex Mono',monospace;
                     font-size:9px;color:#8B949E;text-transform:uppercase">Sector</th>
                  <th style="padding:7px 10px;text-align:left;font-family:'IBM Plex Mono',monospace;
                     font-size:9px;color:#8B949E;text-transform:uppercase">Price</th>
                  <th style="padding:7px 10px;text-align:left;font-family:'IBM Plex Mono',monospace;
                     font-size:9px;color:#8B949E;text-transform:uppercase">1D</th>
                  <th style="padding:7px 10px;text-align:left;font-family:'IBM Plex Mono',monospace;
                     font-size:9px;color:#8B949E;text-transform:uppercase">Volume</th>
                </tr></thead>
                <tbody>{rows_h}</tbody></table>""", unsafe_allow_html=True)

        top_n = us_df.head(n_movers).sort_values("1D %")
        bar_colours = ["#3FB950" if v >= 0 else "#F85149" for v in top_n["1D %"]]
        fig_bar2 = go.Figure(go.Bar(
            x=top_n["1D %"], y=top_n["Symbol"],
            orientation="h",
            marker_color=bar_colours, opacity=0.85,
            text=[f"{v:+.2f}%" for v in top_n["1D %"]],
            textposition="outside",
            textfont=dict(size=9, family="IBM Plex Mono, monospace", color="#C9D1D9"),
        ))
        fig_bar2.add_vline(x=0, line_color="#8B949E", line_dash="dot", line_width=0.8)
        fig_bar2.update_layout(**{**LAYOUT, "height": 340,
            "title": dict(text="US 1-Day Returns (%)", font_size=12)},
            xaxis_title="1D Return (%)", margin=dict(l=8, r=60, t=36, b=8))
        st.plotly_chart(fig_bar2, use_container_width=True, config={"displayModeBar": False})

# ── TAB 3: Sector Heatmap ────────────────────────────────────────────────────
with tabs_m[2]:
    col_h1, col_h2 = st.columns(2)

    def build_sector_heatmap(watchlist, sectors, title, sym_prefix=""):
        """Compute avg 1D return per sector and render as treemap-style chart."""
        sector_returns = {s: [] for s in sectors}
        for sym, name, sector in watchlist:
            try:
                df_s = fetch_ohlcv(sym, "5d", "1d")
                if df_s.empty or len(df_s) < 2:
                    continue
                chg = (float(df_s["Close"].iloc[-1]) -
                       float(df_s["Close"].iloc[-2])) / float(df_s["Close"].iloc[-2]) * 100
                if sector in sector_returns:
                    sector_returns[sector].append(chg)
            except Exception:
                continue

        avgs   = {s: np.mean(v) for s, v in sector_returns.items() if v}
        labels = list(avgs.keys())
        values = list(avgs.values())
        if not labels:
            return None

        colours = ["rgba(63,185,80,0.8)" if v >= 0 else "rgba(248,81,73,0.8)"
                   for v in values]
        alphas  = [min(abs(v) / 3, 1.0) for v in values]
        final_c = []
        for i, (v, a) in enumerate(zip(values, alphas)):
            r, g, b = (63, 185, 80) if v >= 0 else (248, 81, 73)
            final_c.append(f"rgba({r},{g},{b},{0.2 + 0.7*a})")

        fig_hm = go.Figure(go.Bar(
            x=values, y=labels,
            orientation="h",
            marker_color=final_c,
            text=[f"{v:+.2f}%" for v in values],
            textposition="outside",
            textfont=dict(size=10, family="IBM Plex Mono, monospace", color="#C9D1D9"),
        ))
        fig_hm.add_vline(x=0, line_color="#8B949E", line_dash="dot", line_width=0.8)
        fig_hm.update_layout(
            **{**LAYOUT, "height": 360,
               "title": dict(text=title, font_size=12)},
            xaxis_title="Avg 1D Return (%)",
            margin=dict(l=8, r=70, t=36, b=8),
        )
        return fig_hm

    with col_h1:
        section_header("NSE Sector Performance")
        with st.spinner("Building NSE heatmap…"):
            fig_nse_h = build_sector_heatmap(NSE_NIFTY50, NSE_SECTORS,
                                             "NSE Sector Avg 1D Return (%)")
        if fig_nse_h:
            st.plotly_chart(fig_nse_h, use_container_width=True,
                            config={"displayModeBar": False})
        else:
            st.info("Insufficient data for NSE sector heatmap.")

    with col_h2:
        section_header("US Sector Performance")
        with st.spinner("Building US heatmap…"):
            fig_us_h = build_sector_heatmap(US_WATCHLIST, US_SECTORS,
                                            "US Sector Avg 1D Return (%)")
        if fig_us_h:
            st.plotly_chart(fig_us_h, use_container_width=True,
                            config={"displayModeBar": False})
        else:
            st.info("Insufficient data for US sector heatmap.")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — VOLATILITY PULSE  (VIX + historical vol spread)
# ═══════════════════════════════════════════════════════════════════════════════
section_header("Volatility Pulse")
col_v1, col_v2 = st.columns(2)

with col_v1:
    vix_s = load_series("^VIX", period)
    if not vix_s.empty:
        vix_now = float(vix_s.iloc[-1])
        vix_col = ("#F85149" if vix_now > 25
                   else ("#E3B341" if vix_now > 18 else "#3FB950"))
        fig_vix = go.Figure()
        fig_vix.add_trace(go.Scatter(
            x=vix_s.index, y=vix_s.values,
            name="VIX", line=dict(color=vix_col, width=2),
            fill="tozeroy", fillcolor=vix_col.replace(")", ",0.08)").replace("rgba(", "rgba(")
                            if vix_col.startswith("rgba") else
                            f"rgba({','.join(str(int(vix_col.lstrip('#')[i:i+2], 16)) for i in (0,2,4))},0.08)",
        ))
        fig_vix.add_hline(y=25, line_color="#F85149", line_dash="dot",
                          line_width=1, annotation_text=" Fear > 25")
        fig_vix.add_hline(y=18, line_color="#E3B341", line_dash="dot",
                          line_width=1, annotation_text=" Caution > 18")
        fig_vix.update_layout(**{**LAYOUT, "height": 280,
            "title": dict(
                text=f"CBOE VIX — Current: {vix_now:.1f} ({'High Fear' if vix_now>25 else 'Elevated' if vix_now>18 else 'Low'})",
                font_size=12)})
        st.plotly_chart(fig_vix, use_container_width=True,
                        config={"displayModeBar": False})

with col_v2:
    # India VIX — India's fear gauge
    india_vix_s = load_series("^INDIAVIX", period)
    nsei_s      = load_series("^NSEI", period)

    if not nsei_s.empty:
        nsei_hv = nsei_s.pct_change().rolling(20).std() * np.sqrt(252) * 100
        fig_ihv = go.Figure()
        fig_ihv.add_trace(go.Scatter(
            x=nsei_hv.index, y=nsei_hv.values,
            name="Nifty 20D HV", line=dict(color="#E3B341", width=2),
            fill="tozeroy", fillcolor="rgba(227,179,65,0.07)",
        ))
        if not india_vix_s.empty:
            fig_ihv.add_trace(go.Scatter(
                x=india_vix_s.index, y=india_vix_s.values,
                name="India VIX", line=dict(color="#F85149", width=1.8),
            ))
        fig_ihv.update_layout(**{**LAYOUT, "height": 280,
            "title": dict(text="Nifty Historical Volatility (20D Ann.%)", font_size=12)})
        st.plotly_chart(fig_ihv, use_container_width=True,
                        config={"displayModeBar": False})
