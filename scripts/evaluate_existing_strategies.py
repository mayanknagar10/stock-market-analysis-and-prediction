"""Persist preserved V4 strategy metrics separately from forecast-model evaluation."""
import argparse
import json
from pathlib import Path
import sys
import math
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.evaluate_v4 import read_snapshot
from core.strategy_backtest import list_strategies,run_strategy_backtest

def finite(value):
    if isinstance(value,dict): return {k:finite(v) for k,v in value.items()}
    if isinstance(value,list): return [finite(v) for v in value]
    if isinstance(value,(float,np.floating)) and not math.isfinite(value): return None
    return value

def main():
    import pandas as pd
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'reports/baseline/V4_STRATEGIES.json')
    args=parser.parse_args()
    manifest,frames=read_snapshot(args.manifest)
    protocol=json.loads((ROOT/'config/v5_validation.json').read_text(encoding='utf-8'))
    report={'kind':'existing_strategy_research_diagnostic','final_test_evaluated':False,
            'manifest':args.manifest.name,'final_test_start':protocol['final_test_start'],
            'fees_pct':.001,'slippage_pct':.0005,'initial_cash':100000,
            'limitations':['Default strategy parameters; no optimization',
                'Retrospective current-universe snapshot; not verified point-in-time',
                'Legacy close-based signals/close execution may be economically unattainable',
                'Legacy vectorbt freq=D annualization retained; not corrected to exchange sessions',
                'These strategy returns are not forecast-model performance'], 'results':{}}
    for ticker,full in frames.items():
        frame=full.loc[full.index<pd.Timestamp(protocol['final_test_start'])]
        for strategy in list_strategies():
            result=run_strategy_backtest(frame,strategy,fees_pct=.001,slippage_pct=.0005)
            if 'error' in result:
                report['results'][ticker+' / '+strategy]={'error':result['error']}
                continue
            metrics=dict(result['metrics'])
            trades=result['trades']
            entry_col=next((c for c in ['Avg Entry Price','Entry Price'] if c in trades),None)
            exit_col=next((c for c in ['Avg Exit Price','Exit Price'] if c in trades),None)
            if entry_col and exit_col and 'Size' in trades and 'Status' in trades:
                entry_notional=float((trades['Size']*trades[entry_col]).abs().sum())
                closed=trades['Status'].astype(str).str.lower()=='closed'
                exit_notional=float((trades.loc[closed,'Size']*trades.loc[closed,exit_col]).abs().sum())
                metrics['trade_based_turnover_ratio']=(entry_notional+exit_notional)/100000
            else: metrics['trade_based_turnover_ratio']=None
            report['results'][ticker+' / '+strategy]={'sector':manifest['assets'][ticker]['sector'],
                'family':manifest['assets'][ticker]['family'],'metrics_after_costs':finite(metrics)}
        print(ticker,'strategy diagnostics complete',flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream: json.dump(report,stream,indent=2,allow_nan=False)
    print('Persisted',args.output,flush=True)

if __name__=='__main__': main()
