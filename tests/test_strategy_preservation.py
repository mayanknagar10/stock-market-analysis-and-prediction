import numpy as np
import pytest
from core.strategy_backtest import run_strategy_backtest, list_strategies

def test_buy_hold_costs_reduce_result_without_changing_price_data(candles):
    before=candles.copy()
    gross=run_strategy_backtest(candles,'Buy & Hold',fees_pct=0,slippage_pct=0)
    net=run_strategy_backtest(candles,'Buy & Hold',fees_pct=.001,slippage_pct=.0005)
    assert 'error' not in gross,gross
    assert 'error' not in net,net
    assert net['metrics']['final_value']<gross['metrics']['final_value']
    assert net['entries'].sum()==1
    assert net['exits'].sum()==0
    assert candles.equals(before)

def test_donchian_breakout_does_not_change_past_signals(candles):
    from core.strategy_backtest import strategy_donchian_breakout
    entries,exits=strategy_donchian_breakout(candles)
    past_entries,past_exits=strategy_donchian_breakout(candles.iloc[:300])
    assert entries.iloc[:300].equals(past_entries)
    assert exits.iloc[:300].equals(past_exits)

def test_unknown_strategy_is_an_explicit_error(candles):
    result=run_strategy_backtest(candles,'missing')
    assert 'error' in result
    assert 'metrics' not in result
