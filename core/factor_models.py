"""
Factor Models — Fama-French exposures + quantitative factor signals.

Fama-French factor data (Market, SMB, HML, RMW, CMA) is a FREE public
download from Kenneth French's Dartmouth data library — no API key, no
account, no login. `pandas_datareader` has a built-in reader for it
(source="famafrench"), which is what fetch_fama_french_factors() uses.

Two distinct things live in this module:

  1. Factor EXPOSURES — regress a stock's excess returns against the
     Fama-French factors to see how much of its return is explained by
     each factor (market beta, size tilt, value tilt, profitability
     tilt, investment tilt) vs. stock-specific alpha. This is standard
     academic factor analysis (Fama & French 1993, 2015).

  2. Quant factor SIGNALS — independent of Fama-French, these are
     simple long-only scoring signals (Value, Momentum, Quality,
     Low-Volatility) computed directly from data already in the app
     (fundamentals + price history), used for factor-based screening
     and the composite score shown per stock.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
import warnings
warnings.filterwarnings("ignore")

try:
    import pandas_datareader.data as web
    _PDR_AVAILABLE = True
except ImportError:
    _PDR_AVAILABLE = False

try:
    import statsmodels.api as sm
    _SM_AVAILABLE = True
except ImportError:
    _SM_AVAILABLE = False


FF_FACTOR_SETS = {
    "3-Factor (Mkt, SMB, HML)": "F-F_Research_Data_Factors",
    "5-Factor (+ RMW, CMA)":    "F-F_Research_Data_5_Factors_2x3",
}


# ─────────────────────────────────────────────────────────────────
# FAMA-FRENCH FACTOR DATA — free public download, no key
# ─────────────────────────────────────────────────────────────────

def fetch_fama_french_factors(factor_set: str = "5-Factor (+ RMW, CMA)",
                              start: Optional[str] = None) -> pd.DataFrame:
    """
    Monthly Fama-French factor returns, in decimal (0.0123 = 1.23%).
    Columns are a subset of: Mkt-RF, SMB, HML, RMW, CMA, RF.

    Returns empty DataFrame if pandas_datareader isn't installed or the
    download fails (e.g. no internet access) — callers should handle
    that gracefully, same pattern as every other external_data fetcher
    in this app.
    """
    if not _PDR_AVAILABLE:
        return pd.DataFrame()
    dataset_name = FF_FACTOR_SETS.get(factor_set, "F-F_Research_Data_5_Factors_2x3")
    try:
        raw = web.DataReader(dataset_name, "famafrench", start=start)
        df = raw[0].copy()
        df.index = df.index.to_timestamp()
        df = df / 100.0
        return df
    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────
# FACTOR EXPOSURE REGRESSION
# ─────────────────────────────────────────────────────────────────

def compute_factor_exposures(stock_monthly_returns: pd.Series,
                             factors_df: pd.DataFrame) -> Dict:
    """
    OLS regression: (stock_return - RF) ~ Mkt-RF + SMB + HML [+ RMW + CMA]

    Returns alpha (annualised), factor betas, t-stats, R-squared — the
    standard Fama-French factor-exposure report. Uses statsmodels for
    proper statistical inference (t-stats, p-values), not just point
    estimates.
    """
    if not _SM_AVAILABLE or factors_df.empty or stock_monthly_returns.empty:
        return {"error": "Missing dependency or empty data."}

    factor_cols = [c for c in factors_df.columns if c != "RF"]
    aligned = pd.concat([stock_monthly_returns.rename("stock_ret"), factors_df], axis=1).dropna()
    if len(aligned) < 12:
        return {"error": "Not enough overlapping months (need >= 12) for a meaningful regression."}

    y = aligned["stock_ret"] - aligned["RF"]
    X = aligned[factor_cols]
    X = sm.add_constant(X)

    try:
        model = sm.OLS(y, X).fit()
    except Exception as e:
        return {"error": f"Regression failed: {e}"}

    alpha_monthly = float(model.params.get("const", 0))
    alpha_annualised = (1 + alpha_monthly) ** 12 - 1

    betas = {c: round(float(model.params.get(c, 0)), 4) for c in factor_cols}
    t_stats = {c: round(float(model.tvalues.get(c, 0)), 3) for c in factor_cols}
    p_values = {c: round(float(model.pvalues.get(c, 1)), 4) for c in factor_cols}

    return {
        "alpha_monthly_pct": round(alpha_monthly * 100, 4),
        "alpha_annualised_pct": round(alpha_annualised * 100, 3),
        "alpha_t_stat": round(float(model.tvalues.get("const", 0)), 3),
        "alpha_p_value": round(float(model.pvalues.get("const", 1)), 4),
        "betas": betas,
        "t_stats": t_stats,
        "p_values": p_values,
        "r_squared": round(float(model.rsquared), 4),
        "adj_r_squared": round(float(model.rsquared_adj), 4),
        "n_observations": int(len(aligned)),
    }


def interpret_factor_betas(betas: Dict[str, float]) -> List[str]:
    """Plain-English interpretation of factor loadings."""
    notes = []
    if "Mkt-RF" in betas:
        b = betas["Mkt-RF"]
        if b > 1.2: notes.append(f"High market beta ({b:.2f}) — more volatile than the market.")
        elif b < 0.8: notes.append(f"Low market beta ({b:.2f}) — less volatile than the market.")
        else: notes.append(f"Market beta near 1.0 ({b:.2f}) — moves roughly with the market.")
    if "SMB" in betas:
        b = betas["SMB"]
        if b > 0.2: notes.append(f"Positive size tilt ({b:.2f}) — behaves like a small-cap stock.")
        elif b < -0.2: notes.append(f"Negative size tilt ({b:.2f}) — behaves like a large-cap stock.")
    if "HML" in betas:
        b = betas["HML"]
        if b > 0.2: notes.append(f"Positive value tilt ({b:.2f}) — behaves like a value stock.")
        elif b < -0.2: notes.append(f"Negative value tilt ({b:.2f}) — behaves like a growth stock.")
    if "RMW" in betas:
        b = betas["RMW"]
        if b > 0.2: notes.append(f"Positive profitability tilt ({b:.2f}) — robust-profitability exposure.")
        elif b < -0.2: notes.append(f"Negative profitability tilt ({b:.2f}) — weak-profitability exposure.")
    if "CMA" in betas:
        b = betas["CMA"]
        if b > 0.2: notes.append(f"Conservative investment tilt ({b:.2f}) — low-investment-growth exposure.")
        elif b < -0.2: notes.append(f"Aggressive investment tilt ({b:.2f}) — high-investment-growth exposure.")
    return notes


# ─────────────────────────────────────────────────────────────────
# QUANT FACTOR SIGNALS — independent of Fama-French, computed locally
# ─────────────────────────────────────────────────────────────────

def compute_momentum_signal(close: pd.Series, skip_recent_days: int = 21) -> Optional[float]:
    """Classic 12-1 momentum: 12-month return, EXCLUDING the most recent
    month — standard in academic momentum research since the most recent
    month tends to mean-revert rather than continue trending."""
    if len(close) < 252 + skip_recent_days:
        return None
    price_12m_ago = float(close.iloc[-(252 + skip_recent_days)])
    price_1m_ago = float(close.iloc[-skip_recent_days])
    if price_12m_ago <= 0:
        return None
    return (price_1m_ago / price_12m_ago) - 1


def compute_lowvol_signal(close: pd.Series, window: int = 252) -> Optional[float]:
    """Negative of annualised volatility — higher score = lower vol (the
    'low volatility anomaly': low-vol stocks have historically earned
    better risk-adjusted returns than high-vol stocks)."""
    if len(close) < window:
        return None
    ann_vol = float(close.pct_change().tail(window).std() * np.sqrt(252))
    return -ann_vol


def compute_value_signal(pe_ratio: Optional[float]) -> Optional[float]:
    """Negative of P/E — lower P/E = higher value score. A simple but
    widely used practical proxy for the academic HML factor's book-to-
    market when per-share book value isn't readily available."""
    if pe_ratio is None or pe_ratio <= 0:
        return None
    return -pe_ratio


def compute_quality_signal(roe: Optional[float], gross_margin: Optional[float],
                           operating_margin: Optional[float]) -> Optional[float]:
    """Composite of profitability metrics — mirrors the intent of the
    RMW (robust-minus-weak profitability) factor using metrics already
    available from fetch_fundamentals."""
    parts = [v for v in [roe, gross_margin, operating_margin] if v is not None]
    if not parts:
        return None
    return float(np.mean(parts))


def compute_composite_factor_scores(universe_data: List[Dict]) -> pd.DataFrame:
    """
    universe_data: list of dicts, each with keys:
        ticker, close_series (pd.Series), pe_ratio, roe, gross_margin, operating_margin

    Returns a DataFrame with raw signals + cross-sectional z-scores per
    factor + an equal-weighted composite score, ranked best-first.
    Cross-sectional ranking only makes sense across a universe, so this
    always operates on a list of tickers at once.
    """
    rows = []
    for item in universe_data:
        mom = compute_momentum_signal(item["close_series"])
        lowvol = compute_lowvol_signal(item["close_series"])
        value = compute_value_signal(item.get("pe_ratio"))
        quality = compute_quality_signal(
            item.get("roe"), item.get("gross_margin"), item.get("operating_margin"))
        rows.append({
            "Ticker": item["ticker"], "Momentum": mom, "LowVol": lowvol,
            "Value": value, "Quality": quality,
        })

    df = pd.DataFrame(rows).set_index("Ticker")
    if df.empty:
        return df

    z_df = df.copy()
    for col in ["Momentum", "LowVol", "Value", "Quality"]:
        valid = df[col].dropna()
        if len(valid) >= 2 and valid.std() > 0:
            z_df[col] = (df[col] - valid.mean()) / valid.std()
        else:
            z_df[col] = np.nan

    z_df["Composite"] = z_df[["Momentum", "LowVol", "Value", "Quality"]].mean(axis=1, skipna=True)
    z_df = z_df.sort_values("Composite", ascending=False)
    return z_df.round(3)
