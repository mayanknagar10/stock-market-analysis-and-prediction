"""
UI helpers: formatters, CSS, metric cards, signal badges.
All user-facing strings are HTML-escaped via esc().
"""

import html as _html
import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Currency helpers
# ---------------------------------------------------------------------------
_FALLBACK_SYMBOLS = {
    "INR": "₹", "USD": "$", "EUR": "€", "GBP": "£",
    "JPY": "¥", "CNY": "¥", "AUD": "A$", "CAD": "C$",
    "HKD": "HK$", "SGD": "S$",
}

def _sym(currency: str) -> str:
    """
    Convert a currency string to its display symbol.
    - 3-letter ISO code → table lookup (e.g. 'USD' → '$')
    - Anything else → returned as-is (e.g. '$' → '$', '₹' → '₹')
    No trailing spaces are ever added.
    """
    s = str(currency).strip()
    if len(s) == 3 and s.isalpha():
        try:
            from core.data_fetcher import currency_symbol
            return currency_symbol(s)
        except Exception:
            return _FALLBACK_SYMBOLS.get(s.upper(), s)
    return s   # already a symbol — pass through unchanged


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
:root{--bg:#0D1117;--bg-card:#161B22;--border:#30363D;--green:#3FB950;
      --red:#F85149;--blue:#58A6FF;--amber:#E3B341;--purple:#BC8CFF;
      --text:#C9D1D9;--text-dim:#8B949E;
      --mono:'IBM Plex Mono',monospace;--sans:'IBM Plex Sans',sans-serif}
.stApp{background-color:var(--bg);font-family:var(--sans)}
.block-container{padding:1.25rem 2rem 2rem;max-width:1400px}
#MainMenu,footer,header{visibility:hidden}
[data-testid="stSidebar"]{background:var(--bg-card);border-right:1px solid var(--border)}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stTextInput label{
  font-size:11px;color:var(--text-dim);text-transform:uppercase;
  letter-spacing:.06em;font-family:var(--mono)}
.ticker-bar{display:flex;align-items:baseline;gap:16px;padding:12px 0 8px;
  border-bottom:1px solid var(--border);margin-bottom:16px}
.ticker-symbol{font-family:var(--mono);font-size:22px;font-weight:600;color:var(--text)}
.ticker-name{font-size:13px;color:var(--text-dim)}
.ticker-price{font-family:var(--mono);font-size:28px;font-weight:600;
  color:var(--text);margin-left:auto}
.change-pos{color:var(--green);font-family:var(--mono);font-size:14px}
.change-neg{color:var(--red);font-family:var(--mono);font-size:14px}
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(155px,1fr));
  gap:10px;margin-bottom:20px}
.kpi-card{background:var(--bg-card);border:1px solid var(--border);
  border-radius:8px;padding:12px 14px}
.kpi-label{font-family:var(--mono);font-size:10px;color:var(--text-dim);
  text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px}
.kpi-value{font-family:var(--mono);font-size:18px;font-weight:600;color:var(--text)}
.kpi-value.pos{color:var(--green)}.kpi-value.neg{color:var(--red)}
.kpi-sub{font-size:10px;color:var(--text-dim);margin-top:3px}
.sig-badge{display:inline-block;font-family:var(--mono);font-size:11px;
  font-weight:600;letter-spacing:.08em;padding:3px 10px;border-radius:4px}
.sig-strong-buy{background:rgba(63,185,80,.15);color:#3FB950;border:1px solid #3FB950}
.sig-buy{background:rgba(63,185,80,.08);color:#3FB950;border:1px solid rgba(63,185,80,.35)}
.sig-neutral{background:rgba(139,148,158,.1);color:#8B949E;border:1px solid #30363D}
.sig-sell{background:rgba(248,81,73,.08);color:#F85149;border:1px solid rgba(248,81,73,.35)}
.sig-strong-sell{background:rgba(248,81,73,.15);color:#F85149;border:1px solid #F85149}
.section-header{font-family:var(--mono);font-size:11px;font-weight:600;
  color:var(--text-dim);text-transform:uppercase;letter-spacing:.1em;
  border-bottom:1px solid var(--border);padding-bottom:6px;margin:22px 0 12px}
.stTabs [data-baseweb="tab-list"]{gap:2px;background:var(--bg-card);
  border-bottom:1px solid var(--border);padding:0 4px}
.stTabs [data-baseweb="tab"]{background:transparent;border:none;
  border-bottom:2px solid transparent;color:var(--text-dim);
  font-family:var(--mono);font-size:12px;padding:10px 16px}
.stTabs [aria-selected="true"]{color:#3FB950;border-bottom:2px solid #3FB950}
.element-container .stPlotlyChart{border:1px solid var(--border);
  border-radius:6px;overflow:hidden}
.news-card{background:var(--bg-card);border:1px solid var(--border);
  border-radius:6px;padding:12px 14px;margin-bottom:8px}
.news-title{font-size:13px;color:var(--text);font-weight:500}
.news-meta{font-size:10px;color:var(--text-dim);font-family:var(--mono);margin-top:4px}
</style>
"""

def inject_css():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def esc(text: str) -> str:
    """HTML-escape a string so & < > don't break inline HTML."""
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


# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------

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
    sym   = _sym(currency)
    sign  = "+" if change >= 0 else ""
    cls   = "change-pos" if change >= 0 else "change-neg"
    arrow = "▲" if change >= 0 else "▼"
    st.markdown(
        f'<div class="ticker-bar">'
        f'<span class="ticker-symbol">{esc(symbol)}</span>'
        f'<span class="ticker-name">{esc(name)}</span>'
        f'<span class="ticker-price">{sym}{price:,.2f}</span>'
        f'<span class="{cls}">{arrow} {sign}{change:.2f}'
        f' ({sign}{change_pct:.2f}%)</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def signals_table(signals_dict: dict):
    th = ("padding:8px 12px;text-align:left;font-family:'IBM Plex Mono',monospace;"
          "font-size:10px;color:#8B949E;text-transform:uppercase;letter-spacing:.08em")
    td_base = "padding:7px 12px;border-bottom:1px solid #30363D;font-family:'IBM Plex Mono',monospace"

    rows = ""
    for name, data in signals_dict.items():
        sig    = data["signal"]
        colour = "#3FB950" if sig == "BUY" else ("#F85149" if sig == "SELL" else "#8B949E")
        rows += (
            f'<tr>'
            f'<td style="{td_base};font-size:12px;color:#C9D1D9">{esc(name)}</td>'
            f'<td style="{td_base};font-size:12px;color:{colour};font-weight:600">{esc(sig)}</td>'
            f'<td style="{td_base};font-size:12px;color:#C9D1D9">{esc(data["value"])}</td>'
            f'<td style="{td_base};font-size:11px;color:#8B949E">{esc(data["note"])}</td>'
            f'</tr>'
        )
    st.markdown(
        f'<table style="width:100%;border-collapse:collapse;background:#161B22;'
        f'border-radius:6px;overflow:hidden;border:1px solid #30363D">'
        f'<thead><tr style="background:#21262D">'
        f'<th style="{th}">Indicator</th><th style="{th}">Signal</th>'
        f'<th style="{th}">Value</th><th style="{th}">Note</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>',
        unsafe_allow_html=True,
    )
