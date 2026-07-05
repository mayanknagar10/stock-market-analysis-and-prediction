"""
Strategy Backtester — vectorized backtesting for trading strategies.

Unlike the Screener's "backtest this screen" feature (which tests a
stock-selection filter across many tickers over time), this page
backtests actual entry/exit TRADING RULES on a single ticker's full
price history, with realistic fees/slippage and a full trade log —
the "MarketInOut"-style backtester from the roadmap.
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.data_fetcher import fetch_ohlcv, validate_ticker, PERIOD_MAP, detect_market, currency_symbol
from core.strategy_backtest import (run_strategy_backtest, optimize_strategy,
                                    list_strategies)
from utils.helpers import (inject_css, section_header, kpi_row, kpi_card,
                           fmt_price, esc, footer_bar, sidebar_brand)
from utils.charts import T, BASE, safe_layout
import plotly.graph_objects as go
inject_css()

with st.sidebar:
    sidebar_brand()
    st.divider()
    ticker = st.text_input("Ticker Symbol", value="AAPL",
                           placeholder="AAPL · RELIANCE.NS").upper().strip()
    period_label = st.selectbox("History", list(PERIOD_MAP.keys()), index=4)
    period, interval = PERIOD_MAP[period_label]

    st.divider()
    strategies = list_strategies()
    strategy_name = st.selectbox("Strategy", list(strategies.keys()))
    st.caption(strategies[strategy_name]["description"])

    strategy_params = {}
    param_spec = strategies[strategy_name]["params"]
    if param_spec:
        st.markdown("**Parameters**")
        for pname, spec in param_spec.items():
            lo, hi, default = spec
            if isinstance(default, float):
                strategy_params[pname] = st.slider(pname, float(lo), float(hi), float(default), 0.1)
            else:
                strategy_params[pname] = st.slider(pname, int(lo), int(hi), int(default))

    st.divider()
    st.markdown("**Trading Costs**")
    init_cash = st.number_input("Initial Capital", value=100_000, min_value=1_000, step=10_000)
    fees_pct = st.slider("Fees per trade (%)", 0.0, 1.0, 0.1, 0.05) / 100
    slippage_pct = st.slider("Slippage per trade (%)", 0.0, 1.0, 0.05, 0.05) / 100

    st.divider()
    run_btn = st.button("▶  Run Backtest", type="primary", use_container_width=True)
    optimize_btn = st.button("🔍  Optimize Parameters", use_container_width=True)
    st.caption("Vectorized via vectorbt — fast even on years of daily data.")

mkt = detect_market(ticker) if ticker else "US"
_sym = currency_symbol("INR" if mkt in ("NSE", "BSE") else "USD")
_flag = "🇮🇳" if mkt in ("NSE", "BSE") else "🇺🇸"

st.markdown(
    f'<div style="font-family:\'IBM Plex Mono\',monospace;padding:10px 0 6px;'
    f'border-bottom:1px solid #30363D;margin-bottom:16px">'
    f'<span style="font-size:20px;font-weight:600;color:#C9D1D9">Strategy Backtester</span>'
    f'&nbsp;&nbsp;<span style="font-size:13px;color:#8B949E">{_flag} {esc(ticker)} · {strategy_name}</span>'
    f'</div>', unsafe_allow_html=True)

if not run_btn and not optimize_btn and "bt_strategy_result" not in st.session_state:
    st.markdown(
        '<div style="display:flex;flex-direction:column;align-items:center;'
        'justify-content:center;min-height:38vh;text-align:center;padding:40px">'
        '<div style="font-size:52px;margin-bottom:16px">📊</div>'
        '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:22px;font-weight:600;'
        'color:#C9D1D9;margin-bottom:10px">Vectorized Strategy Backtester</div>'
        '<div style="font-size:13px;color:#8B949E;max-width:540px;line-height:1.8">'
        'Test real entry/exit trading rules — MA crossovers, RSI reversion, '
        'MACD signals, Bollinger bounces, Donchian breakouts — with realistic '
        'fees, slippage, and a full trade log.<br><br>'
        '<span style="color:#3FB950">Configure sidebar → Run Backtest</span></div></div>',
        unsafe_allow_html=True)
    st.stop()

if run_btn or optimize_btn:
    valid, err = validate_ticker(ticker)
    if not valid:
        st.error(f"**{ticker}** — {err}")
        st.stop()
    df = fetch_ohlcv(ticker, period, interval)
    if df.empty or len(df) < 60:
        st.error("Not enough data. Use a longer history period.")
        st.stop()
    st.session_state["bt_strategy_df"] = df
    st.session_state["bt_strategy_ticker"] = ticker

    if run_btn:
        result = run_strategy_backtest(df, strategy_name, strategy_params,
                                       init_cash, fees_pct, slippage_pct)
        st.session_state["bt_strategy_result"] = result
        st.session_state.pop("bt_optimize_result", None)

    if optimize_btn:
        param_spec = strategies[strategy_name]["params"]
        if not param_spec:
            st.warning(f"{strategy_name} has no tunable parameters to optimize.")
        else:
            grid = {}
            for pname, (lo, hi, default) in param_spec.items():
                if isinstance(default, float):
                    grid[pname] = list(np.round(np.linspace(lo, hi, 4), 2))
                else:
                    grid[pname] = list(np.linspace(lo, hi, 5, dtype=int))
            with st.spinner(f"Testing {np.prod([len(v) for v in grid.values()])} parameter combinations…"):
                opt_df = optimize_strategy(df, strategy_name, grid, init_cash, fees_pct)
            st.session_state["bt_optimize_result"] = opt_df

result = st.session_state.get("bt_strategy_result")
df = st.session_state.get("bt_strategy_df")
tkr = st.session_state.get("bt_strategy_ticker", ticker)
opt_result = st.session_state.get("bt_optimize_result")

if opt_result is not None and not opt_result.empty:
    section_header(f"🔍 Parameter Optimization — {strategy_name}")
    st.caption(f"Top combinations ranked by Sharpe ratio ({len(opt_result)} tested)")
    display_cols = [c for c in opt_result.columns if c in
                    list(strategies[strategy_name]["params"].keys()) +
                    ["sharpe_ratio", "total_return_pct", "max_drawdown_pct", "n_trades", "win_rate_pct"]]
    st.dataframe(
        opt_result[display_cols].head(15).style
        .format({"sharpe_ratio": "{:.3f}", "total_return_pct": "{:+.2f}%",
                "max_drawdown_pct": "{:.2f}%", "win_rate_pct": "{:.1f}%"})
        .background_gradient(subset=["sharpe_ratio"], cmap="RdYlGn", vmin=-1, vmax=2),
        use_container_width=True)
    best = opt_result.iloc[0]
    st.info(f"Best params: {', '.join(f'{k}={best[k]}' for k in strategies[strategy_name]['params'].keys())} "
           f"→ Sharpe {best['sharpe_ratio']:.2f}, Return {best['total_return_pct']:+.2f}%")

if result is None or df is None:
    st.info("Click **Run Backtest** in the sidebar.")
    st.stop()

if "error" in result:
    st.error(result["error"])
    st.stop()

m = result["metrics"]
equity = result["equity_curve"]
benchmark = result["benchmark_curve"]

section_header("Performance Summary")
kpi_row([
    kpi_card("Total Return", f"{m['total_return_pct']:+.2f}%", "Strategy",
             "pos" if m['total_return_pct'] >= 0 else "neg"),
    kpi_card("Buy & Hold", f"{m['benchmark_return_pct']:+.2f}%", "Benchmark",
             "pos" if m['benchmark_return_pct'] >= 0 else "neg"),
    kpi_card("CAGR", f"{m['cagr_pct']:+.2f}%", "Annualised",
             "pos" if m['cagr_pct'] >= 0 else "neg"),
    kpi_card("Sharpe Ratio", f"{m['sharpe_ratio']:.2f}", ">1 = good",
             "pos" if m['sharpe_ratio'] >= 1 else ""),
    kpi_card("Sortino Ratio", f"{m['sortino_ratio']:.2f}", "Downside-adj."),
    kpi_card("Calmar Ratio", f"{m['calmar_ratio']:.2f}", "Return/MaxDD"),
    kpi_card("Max Drawdown", f"{m['max_drawdown_pct']:.2f}%", "", "neg"),
    kpi_card("Trades", str(m['n_trades']), ""),
    kpi_card("Win Rate", f"{m['win_rate_pct']:.1f}%", "" if m['n_trades'] else "no trades"),
    kpi_card("Profit Factor", f"{m['profit_factor']:.2f}", "gross win/loss"),
    kpi_card("Final Value", fmt_price(m['final_value'], currency=_sym), ""),
])

section_header("Equity Curve vs Buy & Hold")
fig = go.Figure()
fig.add_trace(go.Scatter(x=equity.index, y=equity.values, name="Strategy",
                         line=dict(color=T["green"], width=2.2)))
fig.add_trace(go.Scatter(x=benchmark.index, y=benchmark.values, name="Buy & Hold",
                         line=dict(color=T["dim"], width=1.5, dash="dot")))
fig.update_layout(**safe_layout(
    {"yaxis_title": f"Portfolio Value ({_sym})"},
    height=420, title=f"{strategy_name} — {tkr}"))
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

col_price, col_trades = st.columns([3, 2])
with col_price:
    section_header("Price Chart with Entry/Exit Signals")
    entries = result["entries"]
    exits = result["exits"]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close",
                              line=dict(color=T["text"], width=1.3)))
    entry_pts = df.loc[entries[entries].index] if entries.any() else pd.DataFrame()
    exit_pts = df.loc[exits[exits].index] if exits.any() else pd.DataFrame()
    if not entry_pts.empty:
        fig2.add_trace(go.Scatter(x=entry_pts.index, y=entry_pts["Close"], mode="markers",
                                  name="Entry", marker=dict(color=T["green"], size=8, symbol="triangle-up")))
    if not exit_pts.empty:
        fig2.add_trace(go.Scatter(x=exit_pts.index, y=exit_pts["Close"], mode="markers",
                                  name="Exit", marker=dict(color=T["red"], size=8, symbol="triangle-down")))
    fig2.update_layout(**safe_layout({}, height=380, title="Entry/Exit Points"))
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

with col_trades:
    section_header("Trade Log")
    trades = result["trades"]
    if trades.empty:
        st.info("No trades were triggered by this strategy over the selected period.")
    else:
        display_trades = trades[["Entry Timestamp", "Avg Entry Price", "Exit Timestamp",
                                 "Avg Exit Price", "PnL", "Return"]].copy()
        display_trades.columns = ["Entry", "Entry Px", "Exit", "Exit Px", "PnL", "Return %"]
        display_trades["Return %"] = display_trades["Return %"] * 100
        st.dataframe(
            display_trades.style.format({
                "Entry Px": f"{_sym}{{:.2f}}", "Exit Px": f"{_sym}{{:.2f}}",
                "PnL": f"{_sym}{{:+.2f}}", "Return %": "{:+.2f}%",
            }).background_gradient(subset=["Return %"], cmap="RdYlGn", vmin=-10, vmax=10),
            use_container_width=True, height=380)
        csv = display_trades.to_csv(index=False)
        st.download_button("⬇ Download Trade Log", csv, f"{tkr}_trades.csv", "text/csv")

st.markdown(
    '<div style="background:#161B22;border:1px solid #30363D;border-radius:6px;'
    'padding:12px 16px;margin-top:20px;font-size:11px;color:#8B949E;'
    'font-family:\'IBM Plex Mono\',monospace;line-height:1.7">'
    '⚠️ <b style="color:#E3B341">Disclaimer</b> — Historical backtest performance does not '
    'guarantee future results. Fees/slippage are estimated, not exact. '
    '<b>Not financial advice.</b></div>', unsafe_allow_html=True)

footer_bar()
