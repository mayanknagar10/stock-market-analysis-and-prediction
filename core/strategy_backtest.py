"""
Strategy Backtester — vectorized, using vectorbt.

Unlike core/screen_backtest.py (which backtests a stock-SCREEN's matches
over time), this module backtests a TRADING STRATEGY's entry/exit rules
on a single ticker's full price history — the "MarketInOut"-style
backtester from the mid-term roadmap.

Uses vectorbt for genuinely vectorized signal processing and portfolio
simulation (fast even on years of daily data), not a manual bar-by-bar
Python loop.

All strategies are self-contained functions that take a Close price
series (+ any needed OHLCV columns) and return (entries, exits) boolean
Series — vectorbt handles the rest (position sizing, fees, equity curve,
trade log, every standard performance metric).
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Callable
import warnings
warnings.filterwarnings("ignore")

try:
    import vectorbt as vbt
    _VBT_AVAILABLE = True
except ImportError:
    _VBT_AVAILABLE = False


STRATEGY_REGISTRY: Dict[str, Dict] = {}


def register_strategy(name: str, description: str, params: Dict):
    """Decorator that registers a strategy function + its tunable params
    so the UI can build controls automatically without hardcoding a form
    per strategy."""
    def decorator(fn: Callable):
        STRATEGY_REGISTRY[name] = {
            "fn": fn, "description": description, "params": params,
        }
        return fn
    return decorator


# ─────────────────────────────────────────────────────────────────
# BUILT-IN STRATEGIES
# ─────────────────────────────────────────────────────────────────

@register_strategy(
    "MA Crossover", "Buy when fast MA crosses above slow MA, sell on cross below.",
    {"fast": (5, 50, 10), "slow": (20, 200, 30)})
def strategy_ma_crossover(df: pd.DataFrame, fast: int = 10, slow: int = 30) -> Tuple[pd.Series, pd.Series]:
    close = df["Close"]
    fast_ma = vbt.MA.run(close, fast)
    slow_ma = vbt.MA.run(close, slow)
    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)
    return entries, exits


@register_strategy(
    "RSI Mean Reversion", "Buy when RSI drops below oversold, sell above overbought.",
    {"rsi_period": (7, 30, 14), "oversold": (10, 40, 30), "overbought": (60, 90, 70)})
def strategy_rsi_reversion(df: pd.DataFrame, rsi_period: int = 14,
                           oversold: int = 30, overbought: int = 70) -> Tuple[pd.Series, pd.Series]:
    close = df["Close"]
    rsi = vbt.RSI.run(close, rsi_period).rsi
    entries = rsi < oversold
    exits = rsi > overbought
    return entries, exits


@register_strategy(
    "MACD Signal Cross", "Buy when MACD crosses above signal line, sell on cross below.",
    {"fast": (5, 20, 12), "slow": (20, 40, 26), "signal": (5, 15, 9)})
def strategy_macd_cross(df: pd.DataFrame, fast: int = 12, slow: int = 26,
                        signal: int = 9) -> Tuple[pd.Series, pd.Series]:
    close = df["Close"]
    macd = vbt.MACD.run(close, fast_window=fast, slow_window=slow, signal_window=signal)
    entries = macd.macd_crossed_above(macd.signal)
    exits = macd.macd_crossed_below(macd.signal)
    return entries, exits


@register_strategy(
    "Bollinger Band Bounce", "Buy at lower band touch, sell at upper band touch.",
    {"window": (10, 50, 20), "std_dev": (1.0, 3.0, 2.0)})
def strategy_bollinger_bounce(df: pd.DataFrame, window: int = 20,
                              std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series]:
    close = df["Close"]
    bb = vbt.BBANDS.run(close, window=window, alpha=std_dev)
    entries = close < bb.lower
    exits = close > bb.upper
    return entries, exits


@register_strategy(
    "Buy & Hold", "Benchmark — buy on day 1, hold to the end.", {})
def strategy_buy_hold(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    close = df["Close"]
    entries = pd.Series(False, index=close.index)
    exits = pd.Series(False, index=close.index)
    entries.iloc[0] = True
    return entries, exits


@register_strategy(
    "Donchian Breakout", "Buy on N-day high breakout, sell on N-day low breakdown.",
    {"window": (10, 60, 20)})
def strategy_donchian_breakout(df: pd.DataFrame, window: int = 20) -> Tuple[pd.Series, pd.Series]:
    close = df["Close"]
    upper = df["High"].rolling(window).max()
    lower = df["Low"].rolling(window).min()
    entries = close >= upper.shift(1)
    exits = close <= lower.shift(1)
    return entries, exits


def list_strategies() -> Dict[str, Dict]:
    return {name: {"description": v["description"], "params": v["params"]}
            for name, v in STRATEGY_REGISTRY.items()}


# ─────────────────────────────────────────────────────────────────
# RUN A BACKTEST
# ─────────────────────────────────────────────────────────────────

def run_strategy_backtest(
    df: pd.DataFrame, strategy_name: str, strategy_params: Optional[Dict] = None,
    init_cash: float = 100_000, fees_pct: float = 0.001, slippage_pct: float = 0.0005,
) -> Dict:
    """
    Runs a registered strategy against OHLCV data and returns a full
    performance report: equity curve, benchmark comparison, every
    standard metric, and a trade-by-trade log.

    fees_pct / slippage_pct: applied per trade, mirrors real transaction
    costs (unlike screen_backtest.py, which deliberately ignores them for
    the simpler screen-basket use case).
    """
    if not _VBT_AVAILABLE:
        return {"error": "vectorbt is not installed."}
    if strategy_name not in STRATEGY_REGISTRY:
        return {"error": f"Unknown strategy: {strategy_name}"}
    if df.empty or len(df) < 30:
        return {"error": "Not enough price history to backtest."}

    strategy_params = strategy_params or {}
    fn = STRATEGY_REGISTRY[strategy_name]["fn"]

    try:
        entries, exits = fn(df, **strategy_params)
    except Exception as e:
        return {"error": f"Strategy computation failed: {e}"}

    entries = entries.fillna(False)
    exits = exits.fillna(False)

    try:
        pf = vbt.Portfolio.from_signals(
            df["Close"], entries, exits,
            init_cash=init_cash, fees=fees_pct, slippage=slippage_pct,
            freq="D",  # explicit freq avoids a vectorbt/pandas BusinessDay bug
        )
    except Exception as e:
        return {"error": f"Portfolio simulation failed: {e}"}

    equity = pf.value()
    benchmark = pf.benchmark_value()

    trades_df = pf.trades.records_readable
    n_trades = len(trades_df)

    def _safe(fn_call, default=0.0):
        try:
            v = fn_call()
            return float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else default
        except Exception:
            return default

    metrics = {
        "total_return_pct":   round(_safe(pf.total_return) * 100, 2),
        "benchmark_return_pct": round(_safe(lambda: (benchmark.iloc[-1] / benchmark.iloc[0] - 1)) * 100, 2),
        "cagr_pct":            round(_safe(pf.annualized_return) * 100, 2),
        "sharpe_ratio":        round(_safe(pf.sharpe_ratio), 3),
        "sortino_ratio":       round(_safe(pf.sortino_ratio), 3),
        "calmar_ratio":        round(_safe(pf.calmar_ratio), 3),
        "max_drawdown_pct":    round(_safe(pf.max_drawdown) * 100, 2),
        "n_trades":            n_trades,
        "win_rate_pct":        round(_safe(pf.trades.win_rate) * 100, 2) if n_trades else 0.0,
        "profit_factor":       round(_safe(pf.trades.profit_factor), 3) if n_trades else 0.0,
        "avg_trade_pct":       round(float(trades_df["Return"].mean()) * 100, 2) if n_trades else 0.0,
        "best_trade_pct":      round(float(trades_df["Return"].max()) * 100, 2) if n_trades else 0.0,
        "worst_trade_pct":     round(float(trades_df["Return"].min()) * 100, 2) if n_trades else 0.0,
        "final_value":         round(float(equity.iloc[-1]), 2),
    }

    return {
        "equity_curve": equity,
        "benchmark_curve": benchmark,
        "trades": trades_df,
        "entries": entries,
        "exits": exits,
        "metrics": metrics,
        "strategy_name": strategy_name,
        "strategy_params": strategy_params,
    }


def optimize_strategy(
    df: pd.DataFrame, strategy_name: str, param_grid: Dict[str, list],
    init_cash: float = 100_000, fees_pct: float = 0.001,
    optimize_for: str = "sharpe_ratio",
) -> pd.DataFrame:
    """
    Brute-force grid search over a strategy's parameters, ranked by the
    chosen metric. Returns a DataFrame — one row per parameter
    combination, sorted best-first. Uses the same run_strategy_backtest
    per combination (not a custom vectorbt param-sweep) to keep the
    metrics computation identical and trustworthy between a single run
    and an optimization sweep.
    """
    import itertools

    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))
    if len(combos) > 200:
        combos = combos[:200]  # safety cap — grid search cost scales fast

    rows = []
    for combo in combos:
        params = dict(zip(keys, combo))
        result = run_strategy_backtest(df, strategy_name, params, init_cash, fees_pct)
        if "error" in result:
            continue
        row = {**params, **result["metrics"]}
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    result_df = pd.DataFrame(rows)
    sort_col = optimize_for if optimize_for in result_df.columns else "sharpe_ratio"
    return result_df.sort_values(sort_col, ascending=False).reset_index(drop=True)
