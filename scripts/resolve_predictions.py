"""Explicit outcome job; no scheduled writes or automatic model training."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from core.data.providers import YahooMarketDataProvider
from core.data.snapshots import SnapshotStore
from core.forecasting.ledger import PredictionLedger,resolve_outcome
from core.forecasting.service import independent_research_calendar

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--as-of',default=None)
    args=parser.parse_args()
    asof=args.as_of or datetime.now(timezone.utc).isoformat()
    ledger=PredictionLedger(ROOT/'data/v5/predictions.sqlite')
    store=SnapshotStore(ROOT/'data/v5/snapshots')
    provider=YahooMarketDataProvider()
    cached={}
    for record in ledger.rows():
        if record['outcome'] is not None: continue
        try:
            ticker=record['ticker']
            if ticker not in cached: cached[ticker]=provider.capture_history(ticker)
            snapshot=cached[ticker]
            dates=snapshot.frame.index.tz_convert(snapshot.metadata.exchange_timezone).tz_localize(None).normalize()
            calendar=independent_research_calendar(record['market'],dates.min(),asof,exclude_ticker=ticker)
            outcome=resolve_outcome(record,snapshot.frame,asof,expected_sessions=calendar)
            if outcome is None: continue
            outcome['outcome_snapshot_id']=store.save(snapshot)
            outcome['provider_versions']={snapshot.metadata.provider_id:snapshot.metadata.provider_version}
            outcome['calendar_status']='INDEPENDENT_RESEARCH_PROXY_UNVERIFIED'
            ledger.attach_outcome(record['prediction_id'],outcome)
            print(record['prediction_id'],outcome['status'],flush=True)
        except Exception as error: print(record['prediction_id'],'unresolved:',str(error),flush=True)

if __name__=='__main__': main()
