"""Capture raw free-provider bars once; immutable bytes, including held-out period.

This is a retrospective research snapshot, not proof of historical availability.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tickers', default='RELIANCE.NS,TCS.NS,HDFCBANK.NS,AAPL,JPM,XOM')
    parser.add_argument('--period', default='5y')
    parser.add_argument('--output', type=Path, default=ROOT/'data'/'snapshots')
    args = parser.parse_args()
    import yfinance as yf
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {'format_version':1, 'provider':'yahoo/yfinance', 'provider_version':yf.__version__,
                'fetched_at':datetime.now(timezone.utc).isoformat(),
                'point_in_time_verified':False,
                'limitation':'Retrospective adjusted-price vintage; exact original publication times unavailable',
                'requested_period':args.period, 'requested_interval':'1d', 'assets':{}, 'failures':{}}
    sectors = {'RELIANCE.NS':'Energy','TCS.NS':'IT','HDFCBANK.NS':'Bank',
               'AAPL':'Technology','JPM':'Financials','XOM':'Energy'}
    for ticker in [t.strip().upper() for t in args.tickers.split(',') if t.strip()]:
        try:
            bars = yf.Ticker(ticker).history(period=args.period, interval='1d', auto_adjust=False, actions=True)
            if bars.empty or len(bars)<300:
                raise RuntimeError('Fewer than 300 daily candles returned')
            required = ['Open','High','Low','Close','Adj Close','Volume','Dividends','Stock Splits']
            missing = set(required)-set(bars.columns)
            if missing: raise ValueError('Missing provider fields: '+','.join(sorted(missing)))
            bars.index.name = 'session'
            payload = bars[required].to_csv(float_format='%.12g').encode('utf-8')
            digest = hashlib.sha256(payload).hexdigest()
            file = args.output/(digest+'.csv')
            if not file.exists(): file.write_bytes(payload)
            elif file.read_bytes()!=payload: raise RuntimeError('Snapshot hash collision')
            metadata['assets'][ticker] = {'file':file.name, 'sha256':digest, 'rows':len(bars),
                'exchange_timezone':str(bars.index.tz), 'sector':sectors.get(ticker,'unknown'),
                'family':'india_equity' if ticker.endswith(('.NS','.BO')) else 'us_equity',
                'adjustment_policy':'Yahoo Adj Close; raw OHLC/actions preserved',
                'available_at_policy':'Session close + conservative delivery lag; reconstructed, not verified vintage'}
            print(ticker, len(bars), digest[:12], flush=True)
        except Exception as exc:
            metadata['failures'][ticker] = str(exc)
            print(ticker, 'FAILED', str(exc), flush=True)
    payload = json.dumps(metadata, indent=2, sort_keys=True).encode('utf-8')
    digest = hashlib.sha256(payload).hexdigest()
    file = args.output/(digest+'.json')
    with file.open('xb') as stream: stream.write(payload)
    print('MANIFEST',file, flush=True)
    if not metadata['assets']: raise SystemExit(2)

if __name__=='__main__': main()
