"""
Professional risk metrics module.
Covers return analytics, VaR/CVaR (3 methods), risk-adjusted ratios,
drawdown analysis, CAPM stats, and Monte Carlo simulation.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Dict, Optional


TRADING_DAYS = 252
RISK_FREE_RATE = 0.045  # annualised; update as needed


# ─────────────────────────────────────────
# RETURN ANALYTICS
# ─────────────────────────────────────────

def compute_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change().dropna()


def annualised_return(returns: pd.Series) -> float:
    if len(returns) == 0:
        return 0.0
    total_return = (1 + returns).prod()
    n_years = len(returns) / TRADING_DAYS
    return float(total_return ** (1 / n_years) - 1) if n_years > 0 else 0.0


def annualised_volatility(returns: pd.Series) -> float:
    return float(returns.std() * np.sqrt(TRADING_DAYS))


def cumulative_return(prices: pd.Series) -> float:
    if len(prices) < 2:
        return 0.0
    return float((prices.iloc[-1] / prices.iloc[0]) - 1)


# ─────────────────────────────────────────
# VALUE AT RISK  (3 METHODS)
# ─────────────────────────────────────────

def var_historical(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical simulation VaR — no distribution assumption."""
    return float(-np.percentile(returns.dropna(), (1 - confidence) * 100))


def var_parametric(returns: pd.Series, confidence: float = 0.95) -> float:
    """Parametric (Gaussian) VaR."""
    mu, sigma = returns.mean(), returns.std()
    return float(-(mu + stats.norm.ppf(1 - confidence) * sigma))


def var_cornish_fisher(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Modified VaR using Cornish-Fisher expansion — accounts for skew and excess kurtosis.
    Better for fat-tailed return distributions.
    """
    r = returns.dropna()
    mu, sigma = r.mean(), r.std()
    skew  = float(r.skew())
    kurt  = float(r.kurtosis())          # excess kurtosis
    z     = stats.norm.ppf(1 - confidence)
    z_cf  = (z + (z**2 - 1) * skew / 6
               + (z**3 - 3*z) * kurt / 24
               - (2*z**3 - 5*z) * skew**2 / 36)
    return float(-(mu + z_cf * sigma))


def cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """Expected Shortfall (CVaR / ES) — expected loss beyond VaR."""
    r = returns.dropna()
    cutoff = np.percentile(r, (1 - confidence) * 100)
    tail   = r[r <= cutoff]
    return float(-tail.mean()) if len(tail) > 0 else 0.0


# ─────────────────────────────────────────
# RISK-ADJUSTED RATIOS
# ─────────────────────────────────────────

def sharpe_ratio(returns: pd.Series, risk_free: float = RISK_FREE_RATE) -> float:
    annual_ret = annualised_return(returns)
    annual_vol = annualised_volatility(returns)
    if annual_vol == 0:
        return 0.0
    daily_rf = (1 + risk_free) ** (1 / TRADING_DAYS) - 1
    excess   = returns - daily_rf
    return float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS))


def sortino_ratio(returns: pd.Series, risk_free: float = RISK_FREE_RATE) -> float:
    """Uses downside deviation instead of total std dev."""
    daily_rf = (1 + risk_free) ** (1 / TRADING_DAYS) - 1
    excess   = returns - daily_rf
    downside = excess[excess < 0].std() * np.sqrt(TRADING_DAYS)
    ann_ret  = annualised_return(returns)
    if downside == 0:
        return 0.0
    return float((ann_ret - risk_free) / downside)


def calmar_ratio(prices: pd.Series, returns: pd.Series) -> float:
    """Annual return / Max Drawdown."""
    ann_ret = annualised_return(returns)
    _, mdd, _ = drawdown_analysis(prices)
    if mdd == 0:
        return 0.0
    return float(ann_ret / abs(mdd))


def information_ratio(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Active return / Tracking error."""
    active = returns - benchmark_returns.reindex(returns.index).fillna(0)
    te = active.std() * np.sqrt(TRADING_DAYS)
    if te == 0:
        return 0.0
    return float(active.mean() * TRADING_DAYS / te)


# ─────────────────────────────────────────
# DRAWDOWN ANALYSIS
# ─────────────────────────────────────────

def drawdown_series(prices: pd.Series) -> pd.Series:
    """Rolling drawdown from peak."""
    peak = prices.cummax()
    return (prices - peak) / peak


def drawdown_analysis(prices: pd.Series) -> Tuple[pd.Series, float, int]:
    """
    Returns:
      - drawdown Series
      - max drawdown (negative float)
      - max drawdown duration (trading days)
    """
    dd = drawdown_series(prices)
    max_dd = float(dd.min())

    # Duration calculation
    in_dd = dd < 0
    groups = (in_dd != in_dd.shift()).cumsum()
    durations = in_dd.groupby(groups).cumsum()
    max_dur = int(durations.max()) if in_dd.any() else 0

    return dd, max_dd, max_dur


# ─────────────────────────────────────────
# CAPM  (Beta, Alpha, R²)
# ─────────────────────────────────────────

def capm_stats(stock_returns: pd.Series, benchmark_returns: pd.Series,
               risk_free: float = RISK_FREE_RATE
               ) -> Dict[str, float]:
    """
    Computes Beta, Alpha, R² versus a benchmark.
    Aligns on common dates.
    """
    sr = stock_returns.dropna()
    br = benchmark_returns.dropna()
    aligned = pd.concat([sr, br], axis=1, join="inner").dropna()
    if len(aligned) < 10:
        return {"beta": 1.0, "alpha": 0.0, "r_squared": 0.0, "treynor": 0.0}

    y = aligned.iloc[:, 0].values
    x = aligned.iloc[:, 1].values

    slope, intercept, r_val, _, _ = stats.linregress(x, y)
    daily_rf  = (1 + risk_free) ** (1 / TRADING_DAYS) - 1
    ann_alpha = (intercept * TRADING_DAYS)  # Jensen's alpha annualised
    treynor   = ((y.mean() - daily_rf) * TRADING_DAYS) / slope if slope != 0 else 0.0

    return {
        "beta":      round(float(slope), 4),
        "alpha":     round(float(ann_alpha), 4),
        "r_squared": round(float(r_val**2), 4),
        "treynor":   round(float(treynor), 4),
    }


# ─────────────────────────────────────────
# MONTE CARLO SIMULATION
# ─────────────────────────────────────────

def monte_carlo(prices: pd.Series, n_simulations: int = 500,
                n_days: int = 30, seed: int = 42) -> pd.DataFrame:
    """
    Geometric Brownian Motion Monte Carlo simulation.
    Returns DataFrame of simulated price paths (shape: n_days × n_simulations).
    """
    rng     = np.random.default_rng(seed)
    returns = np.log(prices / prices.shift(1)).dropna()
    mu      = returns.mean()
    sigma   = returns.std()
    last_price = float(prices.iloc[-1])

    # GBM: dS = S * (mu*dt + sigma*dW)
    dt    = 1.0
    shock = rng.normal(loc=(mu - 0.5 * sigma**2) * dt,
                       scale=sigma * np.sqrt(dt),
                       size=(n_days, n_simulations))
    paths = np.exp(np.vstack([np.zeros(n_simulations), shock.cumsum(axis=0)]))
    paths = last_price * paths

    future_dates = pd.bdate_range(prices.index[-1], periods=n_days + 1)[1:]
    return pd.DataFrame(paths[1:], index=future_dates)


# ─────────────────────────────────────────
# FULL RISK REPORT
# ─────────────────────────────────────────

def full_risk_report(prices: pd.Series,
                     benchmark_prices: Optional[pd.Series] = None
                     ) -> Dict:
    """Compute all risk metrics and return as a structured dict."""
    returns = compute_returns(prices)
    dd, mdd, mdd_dur = drawdown_analysis(prices)

    report = {
        # Return metrics
        "cumulative_return":     cumulative_return(prices),
        "annualised_return":     annualised_return(returns),
        "annualised_volatility": annualised_volatility(returns),
        "daily_return_mean":     float(returns.mean()),
        "daily_return_std":      float(returns.std()),
        "skewness":              float(returns.skew()),
        "kurtosis":              float(returns.kurtosis()),

        # VaR / CVaR (95%)
        "var_95_historical":     var_historical(returns, 0.95),
        "var_95_parametric":     var_parametric(returns, 0.95),
        "var_95_cf":             var_cornish_fisher(returns, 0.95),
        "cvar_95":               cvar(returns, 0.95),

        # VaR (99%)
        "var_99_historical":     var_historical(returns, 0.99),
        "var_99_parametric":     var_parametric(returns, 0.99),
        "cvar_99":               cvar(returns, 0.99),

        # Risk-adjusted
        "sharpe_ratio":  sharpe_ratio(returns),
        "sortino_ratio": sortino_ratio(returns),

        # Drawdown
        "max_drawdown":          mdd,
        "max_drawdown_duration": mdd_dur,
        "drawdown_series":       dd,
    }

    if benchmark_prices is not None:
        bench_ret = compute_returns(benchmark_prices)
        capm      = capm_stats(returns, bench_ret)
        report.update(capm)
        report["calmar_ratio"]      = calmar_ratio(prices, returns)
        report["information_ratio"] = information_ratio(returns, bench_ret)

    return report
