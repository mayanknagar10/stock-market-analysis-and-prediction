"""UI helpers: formatters, CSS, metric cards, signal badges, top/bottom bars."""

import html as _html
import numpy as np
import streamlit as st

_FALLBACK_SYMBOLS = {
    "INR": "₹", "USD": "$", "EUR": "€", "GBP": "£",
    "JPY": "¥", "CNY": "¥", "AUD": "A$", "CAD": "C$",
    "HKD": "HK$", "SGD": "S$",
}

def _sym(currency: str) -> str:
    s = str(currency).strip()
    if len(s) == 3 and s.isalpha():
        try:
            from core.data_fetcher import currency_symbol
            return currency_symbol(s)
        except Exception:
            return _FALLBACK_SYMBOLS.get(s.upper(), s)
    return s


# ─────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

:root {
  --bg:        #0D1117;
  --bg-card:   #161B22;
  --bg-nav:    #0D1117;
  --border:    #30363D;
  --green:     #3FB950;
  --red:       #F85149;
  --blue:      #58A6FF;
  --amber:     #E3B341;
  --purple:    #BC8CFF;
  --text:      #C9D1D9;
  --text-dim:  #8B949E;
  --mono:      'IBM Plex Mono', monospace;
  --sans:      'IBM Plex Sans', sans-serif;
  --radius:    8px;
}

/* ── Base ── */
.stApp { background-color: var(--bg); font-family: var(--sans); }
.block-container { padding: 0 2rem 2rem !important; max-width: 1400px; }
#MainMenu, footer { visibility: hidden; }
[data-testid="stHeader"] { background: transparent; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--bg-card) !important;
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stTextInput label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stCheckbox label {
  font-size: 11px !important;
  color: var(--text-dim) !important;
  text-transform: uppercase;
  letter-spacing: .06em;
  font-family: var(--mono);
}

/* ── Nav links (st.navigation) ── */
[data-testid="stSidebarNavItems"] a {
  font-family: var(--mono) !important;
  font-size: 12px !important;
  color: var(--text-dim) !important;
  border-radius: 6px;
  transition: background .15s, color .15s;
}
[data-testid="stSidebarNavItems"] a:hover,
[data-testid="stSidebarNavItems"] a[aria-selected="true"] {
  background: rgba(63,185,80,0.1) !important;
  color: var(--green) !important;
}
[data-testid="stSidebarNavSectionHeader"] {
  font-family: var(--mono) !important;
  font-size: 9px !important;
  color: var(--text-dim) !important;
  text-transform: uppercase;
  letter-spacing: .12em;
  padding-top: 12px !important;
}

/* ── Top bar ── */
.sp-topbar {
  position: sticky;
  top: 0;
  z-index: 999;
  background: var(--bg-card);
  border-bottom: 1px solid var(--border);
  padding: 10px 0 10px;
  margin-bottom: 18px;
  display: flex;
  align-items: center;
  gap: 14px;
}
.sp-topbar-logo {
  width: 36px; height: 36px; border-radius: 6px;
  object-fit: contain;
  background: rgba(255,255,255,0.06);
  padding: 4px;
  border: 1px solid rgba(255,255,255,0.08);
}
.sp-topbar-logo-placeholder {
  width: 36px; height: 36px; border-radius: 6px;
  background: linear-gradient(135deg,#1f6feb,#3fb950);
  display: flex; align-items: center; justify-content: center;
  font-family: var(--mono); font-size: 13px; font-weight: 700;
  color: #fff; flex-shrink: 0;
}
.sp-topbar-ticker  { font-family: var(--mono); font-size: 20px; font-weight: 700; color: var(--text); }
.sp-topbar-name    { font-size: 12px; color: var(--text-dim); margin-top: 1px; }
.sp-topbar-price   { font-family: var(--mono); font-size: 26px; font-weight: 700; color: var(--text); margin-left: auto; }
.sp-topbar-up      { color: var(--green); font-family: var(--mono); font-size: 13px; }
.sp-topbar-down    { color: var(--red);   font-family: var(--mono); font-size: 13px; }
.sp-topbar-badge   { font-family: var(--mono); font-size: 10px; padding: 2px 8px;
                      border-radius: 4px; border: 1px solid var(--border);
                      color: var(--text-dim); background: rgba(255,255,255,.04); }

/* ── Footer ── */
.sp-footer {
  margin-top: 40px;
  padding: 12px 0;
  border-top: 1px solid var(--border);
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-family: var(--mono);
  font-size: 10px;
  color: var(--text-dim);
}
.sp-footer a { color: var(--text-dim); text-decoration: none; }
.sp-footer a:hover { color: var(--text); }

/* ── KPI cards ── */
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fill,minmax(155px,1fr)); gap: 10px; margin-bottom: 20px; }
.kpi-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 12px 14px; }
.kpi-label { font-family: var(--mono); font-size: 10px; color: var(--text-dim); text-transform: uppercase; letter-spacing: .08em; margin-bottom: 6px; }
.kpi-value { font-family: var(--mono); font-size: 18px; font-weight: 600; color: var(--text); }
.kpi-value.pos { color: var(--green); }
.kpi-value.neg { color: var(--red); }
.kpi-sub { font-size: 10px; color: var(--text-dim); margin-top: 3px; }

/* ── Section header ── */
.section-header {
  font-family: var(--mono); font-size: 11px; font-weight: 600; color: var(--text-dim);
  text-transform: uppercase; letter-spacing: .1em;
  border-bottom: 1px solid var(--border); padding-bottom: 6px; margin: 22px 0 12px;
}

/* ── Signal badges ── */
.sig-badge { display:inline-block; font-family:var(--mono); font-size:11px;
  font-weight:600; letter-spacing:.08em; padding:3px 10px; border-radius:4px; }
.sig-strong-buy  { background:rgba(63,185,80,.15);  color:#3FB950; border:1px solid #3FB950; }
.sig-buy         { background:rgba(63,185,80,.08);  color:#3FB950; border:1px solid rgba(63,185,80,.35); }
.sig-neutral     { background:rgba(139,148,158,.1); color:#8B949E; border:1px solid #30363D; }
.sig-sell        { background:rgba(248,81,73,.08);  color:#F85149; border:1px solid rgba(248,81,73,.35); }
.sig-strong-sell { background:rgba(248,81,73,.15);  color:#F85149; border:1px solid #F85149; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"]     { gap:2px; background:var(--bg-card); border-bottom:1px solid var(--border); padding:0 4px; }
.stTabs [data-baseweb="tab"]          { background:transparent; border:none; border-bottom:2px solid transparent; color:var(--text-dim); font-family:var(--mono); font-size:12px; padding:10px 16px; }
.stTabs [aria-selected="true"]        { color:#3FB950; border-bottom:2px solid #3FB950; }

/* ── Charts ── */
.element-container .stPlotlyChart     { border:1px solid var(--border); border-radius:6px; overflow:hidden; }

/* ── News ── */
.news-card { background:var(--bg-card); border:1px solid var(--border); border-radius:6px; padding:12px 14px; margin-bottom:8px; }
.news-title { font-size:13px; color:var(--text); font-weight:500; }
.news-meta  { font-size:10px; color:var(--text-dim); font-family:var(--mono); margin-top:4px; }

/* ── Login area in sidebar ── */
.sp-user-card {
  background: rgba(255,255,255,.04);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 10px 12px;
  display: flex;
  align-items: center;
  gap: 10px;
}
.sp-user-avatar {
  width: 30px; height: 30px; border-radius: 50%;
  background: linear-gradient(135deg,#3fb950,#1f6feb);
  display: flex; align-items: center; justify-content: center;
  font-family: var(--mono); font-size: 12px; color: #fff;
  flex-shrink: 0;
}
.sp-user-name  { font-family: var(--mono); font-size: 11px; color: var(--text); }
.sp-user-role  { font-family: var(--mono); font-size: 9px; color: var(--text-dim); }
</style>
"""


def inject_css():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# FORMATTERS
# ─────────────────────────────────────────────────────────────────

def esc(text: str) -> str:
    return _html.escape(str(text), quote=False)


def fmt_price(v, decimals: int = 2, currency: str = "$") -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{_sym(currency)}{float(v):,.{decimals}f}"


def fmt_pct(v, decimals: int = 2) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{float(v) * 100:+.{decimals}f}%"


def fmt_pct_plain(v, decimals: int = 2) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{float(v) * 100:.{decimals}f}%"


def fmt_large(v) -> str:
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
    return f"{float(v):.{decimals}f}x"


# ─────────────────────────────────────────────────────────────────
# SHARED SIDEBAR HEADER  (call at top of every page's sidebar)
# ─────────────────────────────────────────────────────────────────

def sidebar_brand():
    """Renders the StockPro brand logo at the top of the sidebar."""
    st.markdown(
        '<div style="padding: 6px 0 16px;">'
        '<span style="font-family:\'IBM Plex Mono\',monospace;font-size:20px;'
        'font-weight:700;color:#3FB950;letter-spacing:-.5px;">📈 StockPro</span>'
        '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:9px;'
        'color:#8B949E;text-transform:uppercase;letter-spacing:.18em;'
        'margin-top:2px;">Analytics Terminal</div>'
        '</div>',
        unsafe_allow_html=True)


def sidebar_user(name: str = "Guest", role: str = "Free tier",
                 initials: str = "G", on_login=None):
    """
    Renders a user card at the bottom of the sidebar.
    Pass on_login callback (or leave None for a placeholder Login button).
    """
    st.markdown(
        f'<div class="sp-user-card">'
        f'<div class="sp-user-avatar">{esc(initials)}</div>'
        f'<div><div class="sp-user-name">{esc(name)}</div>'
        f'<div class="sp-user-role">{esc(role)}</div></div>'
        f'</div>',
        unsafe_allow_html=True)
    if on_login:
        if st.button("🔐  Login / Sign up", use_container_width=True):
            on_login()
    else:
        # Placeholder — wire up your auth provider here
        st.button("🔐  Login / Sign up", use_container_width=True,
                  help="Auth integration point — connect your provider here")


# ─────────────────────────────────────────────────────────────────
# TOP BAR  (call at very top of each page's main content area)
# ─────────────────────────────────────────────────────────────────

def top_bar(ticker: str, name: str, price: float, change: float,
            change_pct: float, currency: str = "$",
            market: str = "US", logo_url: str = ""):
    """
    Renders the sticky top bar with company logo, ticker, price and change.
    Appears at the top of the main content area on every page that loads
    a specific ticker.
    """
    sym    = _sym(currency)
    sign   = "+" if change >= 0 else ""
    cls    = "sp-topbar-up" if change >= 0 else "sp-topbar-down"
    arrow  = "▲" if change >= 0 else "▼"
    flag   = "🇮🇳" if market in ("NSE", "BSE") else "🇺🇸"
    initials = esc((name or ticker)[:2].upper())

    # Strategy: show the initials badge immediately (never breaks),
    # then swap to the real logo in the background only if it loads.
    # This avoids any broken-image icon regardless of network conditions.
    if logo_url:
        logo_html = (
            # Initials badge shown by default
            f'<div class="sp-topbar-logo-placeholder" id="sp-logo-ph">{initials}</div>'
            # Hidden img — swaps in on success, silently removed on failure
            f'<img src="{logo_url}" '
            f'style="display:none;width:36px;height:36px;border-radius:6px;'
            f'object-fit:contain;padding:4px;'
            f'background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.08)" '
            f'onload="this.style.display=\'block\';'
            f'var ph=document.getElementById(\'sp-logo-ph\');'
            f'if(ph)ph.style.display=\'none\';" '
            f'onerror="this.remove();" />'
        )
    else:
        logo_html = f'<div class="sp-topbar-logo-placeholder">{initials}</div>'

    st.markdown(
        f'<div class="sp-topbar">'
        f'{logo_html}'
        f'<div>'
        f'<div class="sp-topbar-ticker">{esc(ticker)}</div>'
        f'<div class="sp-topbar-name">{esc(name)} &nbsp; {flag} {esc(market)}</div>'
        f'</div>'
        f'<div class="sp-topbar-price">{sym}{price:,.2f}</div>'
        f'<div class="{cls}">{arrow} {sign}{change:.2f} ({sign}{change_pct:.2f}%)</div>'
        f'</div>',
        unsafe_allow_html=True)


def top_bar_simple(title: str, subtitle: str = ""):
    """Top bar for pages that don't have a single ticker (Screener, Overview etc.)."""
    st.markdown(
        f'<div class="sp-topbar">'
        f'<div>'
        f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:20px;'
        f'font-weight:700;color:#C9D1D9">{esc(title)}</span>'
        + (f'&nbsp;&nbsp;<span style="font-size:13px;color:#8B949E">{esc(subtitle)}</span>' if subtitle else "")
        + f'</div></div>',
        unsafe_allow_html=True)


def footer_bar():
    """Renders the app footer with data attribution and disclaimer."""
    from datetime import datetime
    now = datetime.utcnow().strftime("%d %b %Y %H:%M UTC")
    st.markdown(
        f'<div class="sp-footer">'
        f'<span>Data via <a href="https://finance.yahoo.com" target="_blank">'
        f'Yahoo Finance</a> · may be delayed</span>'
        f'<span>Last updated: {now}</span>'
        f'<span style="color:#F85149">⚠️ Not financial advice</span>'
        f'</div>',
        unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# EXISTING COMPONENTS (unchanged interface)
# ─────────────────────────────────────────────────────────────────

def kpi_card(label: str, value: str, sub: str = "", colour: str = "") -> str:
    cls      = f" {colour}" if colour else ""
    sub_html = f'<div class="kpi-sub">{esc(sub)}</div>' if sub else ""
    return (f'<div class="kpi-card">'
            f'<div class="kpi-label">{esc(label)}</div>'
            f'<div class="kpi-value{cls}">{esc(str(value))}</div>'
            f'{sub_html}</div>')


def kpi_row(cards: list):
    st.markdown(f'<div class="kpi-grid">{"".join(cards)}</div>',
                unsafe_allow_html=True)


def signal_badge(signal: str) -> str:
    cls = {
        "STRONG BUY": "sig-strong-buy", "BUY": "sig-buy",
        "NEUTRAL":    "sig-neutral",
        "SELL":       "sig-sell",       "STRONG SELL": "sig-strong-sell",
    }.get(signal.upper(), "sig-neutral")
    return f'<span class="sig-badge {cls}">{esc(signal)}</span>'


def section_header(title: str):
    st.markdown(f'<div class="section-header">{esc(title)}</div>',
                unsafe_allow_html=True)


def ticker_bar(symbol: str, name: str, price: float,
               change: float, change_pct: float, currency: str = "$"):
    """Legacy component — use top_bar() for new pages."""
    sym   = _sym(currency)
    sign  = "+" if change >= 0 else ""
    cls   = "change-pos" if change >= 0 else "change-neg"
    arrow = "▲" if change >= 0 else "▼"
    st.markdown(
        f'<div style="display:flex;align-items:baseline;gap:16px;'
        f'padding:12px 0 8px;border-bottom:1px solid #30363D;margin-bottom:16px">'
        f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:22px;'
        f'font-weight:600;color:#C9D1D9">{esc(symbol)}</span>'
        f'<span style="font-size:13px;color:#8B949E">{esc(name)}</span>'
        f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:28px;'
        f'font-weight:600;color:#C9D1D9;margin-left:auto">{sym}{price:,.2f}</span>'
        f'<span class="{cls}">{arrow} {sign}{change:.2f} ({sign}{change_pct:.2f}%)</span>'
        f'</div>',
        unsafe_allow_html=True)


def signals_table(signals_dict: dict):
    th = ("padding:8px 12px;text-align:left;font-family:'IBM Plex Mono',monospace;"
          "font-size:10px;color:#8B949E;text-transform:uppercase;letter-spacing:.08em")
    td = "padding:7px 12px;border-bottom:1px solid #30363D;font-family:'IBM Plex Mono',monospace"
    rows = ""
    for name, data in signals_dict.items():
        sig    = data["signal"]
        colour = "#3FB950" if sig == "BUY" else ("#F85149" if sig == "SELL" else "#8B949E")
        rows += (
            f'<tr>'
            f'<td style="{td};font-size:12px;color:#C9D1D9">{esc(name)}</td>'
            f'<td style="{td};font-size:12px;color:{colour};font-weight:600">{esc(sig)}</td>'
            f'<td style="{td};font-size:12px;color:#C9D1D9">{esc(data["value"])}</td>'
            f'<td style="{td};font-size:11px;color:#8B949E">{esc(data["note"])}</td>'
            f'</tr>')
    st.markdown(
        f'<table style="width:100%;border-collapse:collapse;background:#161B22;'
        f'border-radius:6px;overflow:hidden;border:1px solid #30363D">'
        f'<thead><tr style="background:#21262D">'
        f'<th style="{th}">Indicator</th><th style="{th}">Signal</th>'
        f'<th style="{th}">Value</th><th style="{th}">Note</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>',
        unsafe_allow_html=True)