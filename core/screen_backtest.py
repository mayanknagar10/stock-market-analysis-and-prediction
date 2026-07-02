"""
Screen Backtester — "If I had run this screen N months ago, how would the
matched stocks have performed?"

This directly addresses the MVP requirement from the roadmap:
  "Basic Backtesting: Minimal support (e.g. result statistics on saved
   screens using past price data)."

Design constraints (deliberate, and important to be honest about):
  - Uses TECHNICAL filters only (RSI, MA trend, volume ratio) for the
    point-in-time evaluation. Free data APIs (yfinance/Stooq) do not give
    point-in-time fundamental snapshots — only the LATEST P/E, beta, etc.
    Using today's fundamentals to decide "would this have matched 6
    months ago" would be look-ahead bias, so fundamental filters are
    intentionally excluded from the backtest engine even though they're
    available in the live Screener.
  - Each rebalance period, the screen is evaluated using ONLY price data
    up to that date — never future data. This is what makes the backtest
    honest rather than just curve-fitted hindsight.
  - Equal-weighted portfolio of all matches at each rebalance; held until
    the next rebalance, then re-evaluated and re-balanced.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

TRADING_DAYS = 252


def _evaluate_technical_filters_at(
    df: pd.DataFrame, as_of_idx: int,
    rsi_min: float, rsi_max: float,
    ma_trend: str, min_vol_ratio: float,
) -> Optional[bool]:
    """
    Evaluate whether a stock's technical state AT a specific past bar
    (as_of_idx) would have matched the filter criteria — using only data
    up to and including that bar (no look-ahead).

    Returns None if there isn't enough history yet at that point to
    compute the indicators (e.g. too early in the series for EMA50).
    """
    from core.indicators import rsi as rsi_fn, ema as ema_fn, volume_ratio as vr_fn

    if as_of_idx < 55:  # need enough bars for EMA50 + RSI warmup
        return None

    window = df.iloc[: as_of_idx + 1]
    close = window["Close"]

    rsi_val = float(rsi_fn(close).iloc[-1])
    if np.isnan(rsi_val) or not (rsi_min <= rsi_val <= rsi_max):
        return False

    if ma_trend != "Any":
        e20 = float(ema_fn(close, 20).iloc[-1])
        e50 = float(ema_fn(close, 50).iloc[-1])
        last = float(close.iloc[-1])
        if ma_trend == "Price > EMA20" and not (last > e20):
            return False
        if ma_trend == "Price > EMA50" and not (last > e50):
            return False
        if ma_trend == "EMA20 > EMA50 (Bullish)" and not (e20 > e50):
            return False
        if ma_trend == "EMA20 < EMA50 (Bearish)" and not (e20 < e50):
            return False

    if min_vol_ratio > 0:
        vr_val = float(vr_fn(window).iloc[-1])
        if np.isnan(vr_val) or vr_val < min_vol_ratio:
            return False

    return True


def run_screen_backtest(
    price_data: Dict[str, pd.DataFrame],
    rebalance_freq: str = "1M",
    lookback_months: int = 12,
    rsi_min: float = 0, rsi_max: float = 100,
    ma_trend: str = "Any",
    min_vol_ratio: float = 0.0,
    max_positions: int = 20,
) -> Dict:
    """
    Run a point-in-time backtest of a technical screen across a universe
    of stocks.

    Parameters
    ----------
    price_data : dict of {ticker: OHLCV DataFrame}, all sharing roughly
        the same date range (already fetched — this function does no
        network calls itself, keeping it fast and side-effect free).
    rebalance_freq : pandas offset alias, e.g. "1M", "2W", "1W".
    lookback_months : how far back the backtest window starts.
    rsi_min/rsi_max/ma_trend/min_vol_ratio : the screen's filter criteria
        (mirrors the live Screener page's technical filters).
    max_positions : cap on simultaneous holdings (largest matches by
        recent volume, to avoid an unrealistically huge equal-weight
        basket).

    Returns
    -------
    dict with:
        portfolio_returns : pd.Series of period returns
        portfolio_value   : pd.Series cumulative value (start = 100)
        rebalance_log      : list of {date, matched_tickers, n_matched}
        metrics            : dict of summary stats
    """
    if not price_data:
        return {"portfolio_returns": pd.Series(dtype=float),
                "portfolio_value": pd.Series(dtype=float),
                "rebalance_log": [], "metrics": {}}

    # Common date index across all tickers (intersection keeps it honest —
    # every ticker has data on every rebalance date we test)
    all_dates = None
    for df in price_data.values():
        idx = df.index
        all_dates = idx if all_dates is None else all_dates.intersection(idx)
    if all_dates is None or len(all_dates) < 80:
        return {"portfolio_returns": pd.Series(dtype=float),
                "portfolio_value": pd.Series(dtype=float),
                "rebalance_log": [], "metrics": {}}
    all_dates = all_dates.sort_values()

    end_date = all_dates[-1]
    start_date = end_date - pd.DateOffset(months=lookback_months)
    test_dates = all_dates[all_dates >= start_date]
    if len(test_dates) < 40:
        # Not enough history for the requested lookback — use what's available
        test_dates = all_dates

    # Normalize aliases: pandas 2.2+ dropped 'M','Q','Y' — use 'ME','QE','YE'
    _alias_map = {"1M":"1ME","2M":"2ME","3M":"3ME","6M":"6ME",
                  "M":"ME","Q":"QE","Y":"YE",
                  "1W":"1W","2W":"2W"}
    _freq = _alias_map.get(rebalance_freq.upper(), rebalance_freq)

    rebal_dates = pd.date_range(test_dates[0], test_dates[-1], freq=_freq)
    rebal_dates = [d for d in rebal_dates if d in set(all_dates)]
    if len(rebal_dates) < 2:
        # Fall back to nearest available trading dates
        rebal_dates = list(
            pd.Series(test_dates).iloc[:: max(len(test_dates) // 12, 1)]
        )
    if len(rebal_dates) < 2:
        return {"portfolio_returns": pd.Series(dtype=float),
                "portfolio_value": pd.Series(dtype=float),
                "rebalance_log": [], "metrics": {}}

    rebalance_log: List[Dict] = []
    period_returns: List[Tuple[pd.Timestamp, float]] = []

    for i in range(len(rebal_dates) - 1):
        d0, d1 = rebal_dates[i], rebal_dates[i + 1]

        matched: List[str] = []
        for ticker, df in price_data.items():
            if d0 not in df.index:
                continue
            as_of_idx = df.index.get_loc(d0)
            result = _evaluate_technical_filters_at(
                df, as_of_idx, rsi_min, rsi_max, ma_trend, min_vol_ratio)
            if result:
                matched.append(ticker)

        # Cap basket size by recent dollar volume (proxy for liquidity/size)
        if len(matched) > max_positions:
            vol_scores = {}
            for t in matched:
                df = price_data[t]
                idx = df.index.get_loc(d0)
                recent = df.iloc[max(0, idx - 10): idx + 1]
                vol_scores[t] = float((recent["Close"] * recent["Volume"]).mean())
            matched = sorted(matched, key=lambda t: -vol_scores.get(t, 0))[:max_positions]

        rebalance_log.append({
            "date": d0, "matched_tickers": matched, "n_matched": len(matched),
        })

        if not matched:
            period_returns.append((d1, 0.0))  # cash / no position this period
            continue

        # Equal-weight return of all matched tickers from d0 -> d1
        rets = []
        for t in matched:
            df = price_data[t]
            if d0 not in df.index or d1 not in df.index:
                continue
            p0 = float(df.loc[d0, "Close"])
            p1 = float(df.loc[d1, "Close"])
            if p0 > 0:
                rets.append(p1 / p0 - 1)
        period_ret = float(np.mean(rets)) if rets else 0.0
        period_returns.append((d1, period_ret))

    if not period_returns:
        return {"portfolio_returns": pd.Series(dtype=float),
                "portfolio_value": pd.Series(dtype=float),
                "rebalance_log": rebalance_log, "metrics": {}}

    ret_dates, ret_vals = zip(*period_returns)
    portfolio_returns = pd.Series(ret_vals, index=pd.DatetimeIndex(ret_dates))
    portfolio_value = (1 + portfolio_returns).cumprod() * 100

    # Summary metrics
    from core.risk_metrics import sharpe_ratio, drawdown_analysis
    total_periods = len(portfolio_returns)
    n_per_year = {"1W": 52, "2W": 26, "1ME": 12, "1M": 12, "2ME": 6,
                  "2M": 6, "3ME": 4, "3M": 4, "ME": 12, "QE": 4}.get(
        _freq.upper().replace("MS", "M"), 12)
    cum_ret = float(portfolio_value.iloc[-1] / 100 - 1) if len(portfolio_value) else 0.0
    years = total_periods / n_per_year if n_per_year else 1
    cagr = float((1 + cum_ret) ** (1 / years) - 1) if years > 0 and (1 + cum_ret) > 0 else 0.0
    ann_vol = float(portfolio_returns.std() * np.sqrt(n_per_year)) if total_periods > 1 else 0.0
    sharpe = (float((portfolio_returns.mean() * n_per_year - 0.045) / ann_vol)
             if ann_vol > 0 else 0.0)
    _, max_dd, _ = drawdown_analysis(portfolio_value)
    avg_matches = float(np.mean([r["n_matched"] for r in rebalance_log])) if rebalance_log else 0
    win_rate = float((portfolio_returns > 0).mean() * 100) if total_periods > 0 else 0.0

    metrics = {
        "total_return_pct": round(cum_ret * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "ann_volatility_pct": round(ann_vol * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "n_rebalances": total_periods,
        "avg_matches_per_rebalance": round(avg_matches, 1),
        "win_rate_pct": round(win_rate, 1),
    }

    return {
        "portfolio_returns": portfolio_returns,
        "portfolio_value": portfolio_value,
        "rebalance_log": rebalance_log,
        "metrics": metrics,
    }


def run_benchmark_comparison(benchmark_df: pd.DataFrame,
                             portfolio_value: pd.Series) -> pd.Series:
    """Indexed benchmark series aligned to the same dates as portfolio_value,
    for an apples-to-apples comparison chart."""
    if portfolio_value.empty or benchmark_df.empty:
        return pd.Series(dtype=float)
    aligned = benchmark_df["Close"].reindex(portfolio_value.index, method="ffill")
    aligned = aligned.dropna()
    if aligned.empty:
        return pd.Series(dtype=float)
    return aligned / aligned.iloc[0] * 100
