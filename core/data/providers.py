"""Free-first V5 market adapter; never silently substitutes incompatible feeds."""
from datetime import datetime, timezone
import pandas as pd
from core.data.contracts import DataSnapshot,SourceMetadata
from core.validation.alignment import aware_index

class ProviderUnavailable(RuntimeError): pass

def equity_family(symbol):
    symbol=symbol.upper().strip()
    if symbol.startswith('^') or '=' in symbol or symbol.endswith(('-USD','-USDT')):
        raise ValueError('Indices, FX, futures and crypto require separate model families')
    if symbol.endswith(('.NS','.BO')): return 'india_equity'
    if '.' in symbol or not symbol: raise ValueError('Unsupported or ambiguous equity market')
    return 'us_equity'

class YahooMarketDataProvider:
    def __init__(self,transport=None): self.transport=transport

    def capture_history(self,symbol,period='5y',interval='1d',reference=False):
        try:
            import yfinance as yf
            if self.transport is None:
                frame=yf.Ticker(symbol).history(period=period,interval=interval,auto_adjust=False,actions=True)
            else: frame=self.transport(symbol,period,interval)
            if frame is None or frame.empty: raise ValueError('No source bars')
            required={'Open','High','Low','Close','Adj Close','Volume'}
            if not reference: required |= {'Dividends','Stock Splits'}
            if not required.issubset(frame): raise ValueError('Raw OHLCV, adjusted close and actions required')
            if frame.index.tz is None: raise ValueError('Source timezone missing')
            zone=str(frame.index.tz)
            frame=frame.copy()
            frame.index=aware_index(frame.index,'Yahoo source session labels')
            fetched=datetime.now(timezone.utc).isoformat()
            metadata=SourceMetadata(provider_id='yahoo/yfinance',provider_version=yf.__version__,
                symbol=symbol.upper().strip(),family='research_reference' if reference else equity_family(symbol),exchange_timezone=zone,interval=interval,
                adjustment_policy='Yahoo auto_adjust=False vendor OHLCV plus total-return Adj Close and actions',source_timestamp=frame.index.max().isoformat(),
                available_at=fetched,fetched_at=fetched,vintage_at=fetched,
                availability_verified=False,revision_history_verified=False)
            return DataSnapshot(frame,metadata)
        except Exception as exc:
            raise ProviderUnavailable('Yahoo capture failed: '+str(exc)) from exc
