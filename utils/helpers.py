"""
UI helpers: formatters, CSS injection, metric cards, signal badges.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional


# ─────────────────────────────────────────
# PROFESSIONAL CSS
# ─────────────────────────────────────────

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

/* ── Root variables ── */
:root {
  --bg:       #0D1117;
  --bg-card:  #161B22;
  --bg-hover: #1C2128;
  --border:   #30363D;
  --green:    #3FB950;
  --red:      #F85149;
  --blue:     #58A6FF;
  --amber:    #E3B341;
  --purple:   #BC8CFF;
  --text:     #C9D1D9;
  --text-dim: #8B949E;
  --mono:     'IBM Plex Mono', monospace;
  --sans:     'IBM Plex Sans', sans-serif;
}

/* ── App shell ── */
.stApp { background-color: var(--bg); font-family: var(--sans); }
.block-container { padding: 1.25rem 2rem 2rem; max-width: 1400px; }
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--bg-card);
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stTextInput label {
  font-size: 11px;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: .06em;
  font-family: var(--mono);
}

/* ── Top ticker bar ── */
.ticker-bar {
  display: flex; align-items: baseline; gap: 16px;
  padding: 12px 0 8px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 16px;
}
.ticker-symbol {
  font-family: var(--mono); font-size: 22px; font-weight: 600; color: var(--text);
}
.ticker-name {
  font-size: 13px; color: var(--text-dim);
}
.ticker-price {
  font-family: var(--mono); font-size: 28px; font-weight: 600; color: var(--text);
  margin-left: auto;
}
.change-pos { color: var(--green); font-family: var(--mono); font-size: 14px; }
.change-neg { color: var(--red);   font-family: var(--mono); font-size: 14px; }

/* ── KPI metric card ── */
.kpi-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
  gap: 10px; margin-bottom: 20px;
}
.kpi-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px 14px;
}
.kpi-label {
  font-family: var(--mono); font-size: 10px; color: var(--text-dim);
  text-transform: uppercase; letter-spacing: .08em; margin-bottom: 6px;
}
.kpi-value {
  font-family: var(--mono); font-size: 18px; font-weight: 600; color: var(--text);
}
.kpi-value.pos { color: var(--green); }
.kpi-value.neg { color: var(--red); }
.kpi-sub {
  font-size: 10px; color: var(--text-dim); margin-top: 3px;
}

/* ── Signal badge ── */
.sig-badge {
  display: inline-block;
  font-family: var(--mono); font-size: 11px; font-weight: 600;
  letter-spacing: .08em; padding: 3px 10px; border-radius: 4px;
}
.sig-strong-buy  { background:rgba(63,185,80,.15);  color:var(--green);  border:1px solid var(--green); }
.sig-buy         { background:rgba(63,185,80,.08);  color:var(--green);  border:1px solid rgba(63,185,80,.35); }
.sig-neutral     { background:rgba(139,148,158,.1); color:var(--text-dim);border:1px solid var(--border); }
.sig-sell        { background:rgba(248,81,73,.08);  color:var(--red);    border:1px solid rgba(248,81,73,.35); }
.sig-strong-sell { background:rgba(248,81,73,.15);  color:var(--red);    border:1px solid var(--red); }

/* ── Signal table row colours ── */
.st-buy  td { color: var(--green) !important; }
.st-sell td { color: var(--red)   !important; }

/* ── Section divider ── */
.section-header {
  font-family: var(--mono); font-size: 11px; font-weight: 600; color: var(--text-dim);
  text-transform: uppercase; letter-spacing: .1em;
  border-bottom: 1px solid var(--border); padding-bottom: 6px; margin: 22px 0 12px;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  gap: 2px; background: var(--bg-card);
  border-bottom: 1px solid var(--border);
  padding: 0 4px;
}
.stTabs [data-baseweb="tab"] {
  background: transparent; border: none; border-bottom: 2px solid transparent;
  color: var(--text-dim); font-family: var(--mono); font-size: 12px;
  padding: 10px 16px;
}
.stTabs [aria-selected="true"] {
  color: #3FB950; border-bottom: 2px solid #3FB950;
}

/* ── Plotly chart containers ── */
.element-container .stPlotlyChart {
  border: 1px solid var(--border);
  border-radius: 6px;
  overflow: hidden;
}

/* ── News card ── */
.news-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 12px 14px;
  margin-bottom: 8px;
}
.news-title { font-size: 13px; color: var(--text); font-weight: 500; }
.news-meta  { font-size: 10px; color: var(--text-dim); font-family: var(--mono); margin-top: 4px; }

/* ── Risk meter ── */
.risk-meter {
  display:flex; align-items:center; gap:12px; padding:14px;
  background: var(--bg-card); border-radius:8px; border:1px solid var(--border);
  margin-bottom: 12px;
}
.risk-label { font-family: var(--mono); font-size: 10px; color: var(--text-dim);
              text-transform: uppercase; }
.risk-val   { font-family: var(--mono); font-size: 20px; font-weight:600; }
</style>
"""


def inject_css():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────
# FORMATTERS
# ─────────────────────────────────────────

def fmt_price(v, decimals: int = 2, currency: str = "$") -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    # Accept ISO code (USD, INR…) or a symbol directly ($, ₹…)
    from core.data_fetcher import currency_symbol
    sym = currency_symbol(currency) if len(currency) == 3 and currency.isalpha() else currency
    return f"{sym}{v:,.{decimals}f}"


def fmt_pct(v, decimals: int = 2) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v * 100:+.{decimals}f}%"


def fmt_pct_plain(v, decimals: int = 2) -> str:
    """Percentage without sign."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v * 100:.{decimals}f}%"


def fmt_large(v) -> str:
    """Compact large number: 1.23T, 456.7B, 12.3M, 987K."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    v = float(v)
    if   abs(v) >= 1e12: return f"{v/1e12:.2f}T"
    elif abs(v) >= 1e9:  return f"{v/1e9:.2f}B"
    elif abs(v) >= 1e6:  return f"{v/1e6:.2f}M"
    elif abs(v) >= 1e3:  return f"{v/1e3:.1f}K"
    return f"{v:.2f}"


def fmt_ratio(v, decimals: int = 2) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{decimals}f}x"


def signed_colour(v) -> str:
    """Return CSS class name for positive/negative colouring."""
    if v is None:
        return ""
    return "pos" if float(v) >= 0 else "neg"


# ─────────────────────────────────────────
# UI COMPONENTS
# ─────────────────────────────────────────

def kpi_card(label: str, value: str, sub: str = "", colour: str = "") -> str:
    colour_cls = f" {colour}" if colour else ""
    sub_html   = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value{colour_cls}">{value}</div>
      {sub_html}
    </div>"""


def kpi_row(cards: list):
    """Render a flex row of KPI cards from a list of kpi_card() strings."""
    inner = "".join(cards)
    st.markdown(f'<div class="kpi-grid">{inner}</div>', unsafe_allow_html=True)


def signal_badge(signal: str) -> str:
    cls_map = {
        "STRONG BUY":  "sig-strong-buy",
        "BUY":         "sig-buy",
        "NEUTRAL":     "sig-neutral",
        "SELL":        "sig-sell",
        "STRONG SELL": "sig-strong-sell",
    }
    cls = cls_map.get(signal.upper(), "sig-neutral")
    return f'<span class="sig-badge {cls}">{signal}</span>'


def section_header(title: str):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def ticker_bar(symbol: str, name: str, price: float, change: float, change_pct: float,
               currency: str = "$"):
    sign      = "+" if change >= 0 else ""
    chg_cls   = "change-pos" if change >= 0 else "change-neg"
    chg_arrow = "▲" if change >= 0 else "▼"
    st.markdown(f"""
    <div class="ticker-bar">
      <span class="ticker-symbol">{symbol}</span>
      <span class="ticker-name">{name}</span>
      <span class="ticker-price">{currency}{price:,.2f}</span>
      <span class="{chg_cls}">{chg_arrow} {sign}{change:.2f}
        &nbsp;({sign}{change_pct:.2f}%)</span>
    </div>
    """, unsafe_allow_html=True)


def signals_table(signals_dict: dict):
    """Render indicator signals as a styled table."""
    rows = ""
    for name, data in signals_dict.items():
        sig = data["signal"]
        if sig == "BUY":
            colour = "#3FB950"
        elif sig == "SELL":
            colour = "#F85149"
        else:
            colour = "#8B949E"
        rows += f"""
        <tr>
          <td style="padding:7px 12px;border-bottom:1px solid #30363D;
                     font-family:'IBM Plex Mono',monospace;font-size:12px;
                     color:#C9D1D9;">{name}</td>
          <td style="padding:7px 12px;border-bottom:1px solid #30363D;
                     font-family:'IBM Plex Mono',monospace;font-size:12px;
                     color:{colour};font-weight:600;">{sig}</td>
          <td style="padding:7px 12px;border-bottom:1px solid #30363D;
                     font-family:'IBM Plex Mono',monospace;font-size:12px;
                     color:#C9D1D9;">{data["value"]}</td>
          <td style="padding:7px 12px;border-bottom:1px solid #30363D;
                     font-size:11px;color:#8B949E;">{data["note"]}</td>
        </tr>"""
    st.markdown(f"""
    <table style="width:100%;border-collapse:collapse;background:#161B22;
                  border-radius:6px;overflow:hidden;border:1px solid #30363D">
      <thead>
        <tr style="background:#21262D">
          <th style="padding:8px 12px;text-align:left;font-family:'IBM Plex Mono',monospace;
                     font-size:10px;color:#8B949E;text-transform:uppercase;letter-spacing:.08em">
            Indicator</th>
          <th style="padding:8px 12px;text-align:left;font-family:'IBM Plex Mono',monospace;
                     font-size:10px;color:#8B949E;text-transform:uppercase;letter-spacing:.08em">
            Signal</th>
          <th style="padding:8px 12px;text-align:left;font-family:'IBM Plex Mono',monospace;
                     font-size:10px;color:#8B949E;text-transform:uppercase;letter-spacing:.08em">
            Value</th>
          <th style="padding:8px 12px;text-align:left;font-family:'IBM Plex Mono',monospace;
                     font-size:10px;color:#8B949E;text-transform:uppercase;letter-spacing:.08em">
            Note</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>""", unsafe_allow_html=True)


def news_item(title: str, publisher: str, timestamp: int):
    from datetime import datetime
    try:
        dt_str = datetime.utcfromtimestamp(timestamp).strftime("%d %b %Y  %H:%M UTC")
    except Exception:
        dt_str = ""
    st.markdown(f"""
    <div class="news-card">
      <div class="news-title">{title}</div>
      <div class="news-meta">{publisher} &nbsp;·&nbsp; {dt_str}</div>
    </div>""", unsafe_allow_html=True)
