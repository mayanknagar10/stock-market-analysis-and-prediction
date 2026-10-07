"""Provider and snapshot contracts. Availability and revision time are distinct."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Protocol
import pandas as pd
from core.validation.alignment import aware_index

class PointInTimeError(ValueError):
    """Historical information cannot be proved available at the requested cutoff."""

@dataclass(frozen=True)
class SourceMetadata:
    provider_id: str
    provider_version: str
    symbol: str
    family: str
    exchange_timezone: str
    interval: str
    adjustment_policy: str
    source_timestamp: str
    available_at: str
    fetched_at: str
    vintage_at: str
    availability_verified: bool = False
    revision_history_verified: bool = False
    fallback_from: str | None = None

    def __post_init__(self):
        if type(self.availability_verified) is not bool or type(self.revision_history_verified) is not bool:
            raise ValueError('Verification flags must be actual boolean values')
        for name in ['provider_id','provider_version','symbol','family','exchange_timezone',
                     'interval','adjustment_policy']:
            if not getattr(self,name): raise ValueError('Missing provider metadata: '+name)
        pd.Timestamp('2024-01-01',tz=self.exchange_timezone)
        observed,available,fetched,vintage = aware_index([self.source_timestamp,self.available_at,
                                                        self.fetched_at,self.vintage_at],'source metadata')
        if observed>available or available>fetched or vintage>fetched:
            raise ValueError('Inconsistent provider source/availability/fetch/vintage clock')

    def to_dict(self): return asdict(self)

@dataclass(frozen=True, init=False)
class DataSnapshot:
    _frame: pd.DataFrame
    metadata: SourceMetadata

    def __init__(self,frame,metadata):
        object.__setattr__(self,'_frame',frame.copy(deep=True))
        object.__setattr__(self,'metadata',metadata)
        self.__post_init__()

    @property
    def frame(self):
        """Consumers receive a defensive copy of the frozen source values."""
        return self._frame.copy(deep=True)

    def __post_init__(self):
        index = aware_index(self._frame.index,'snapshot source timestamps')
        if not index.is_unique or not index.is_monotonic_increasing:
            raise ValueError('Snapshot observations must be unique and ordered')
        if len(index) and index.max()>aware_index([self.metadata.source_timestamp])[0]:
            raise ValueError('Snapshot metadata precedes its newest observation')

    def assert_as_of(self,as_of,require_verified=True):
        self.__post_init__()
        cutoff = aware_index([as_of],'as_of')[0]
        if require_verified and not (self.metadata.availability_verified and self.metadata.revision_history_verified):
            raise PointInTimeError('Point-in-time availability and revision history must be verified')
        for name in ['source_timestamp','available_at','vintage_at']:
            if aware_index([getattr(self.metadata,name)],name)[0]>cutoff:
                raise PointInTimeError(name+' is later than requested information cutoff')
        for name in ['source_timestamp','available_at','vintage_at']:
            if name in self._frame and (aware_index(self._frame[name],name)>cutoff).any():
                raise PointInTimeError(name+' row is later than information cutoff')
        # A verified historical archive can be retrieved later; fetched_at is
        # acquisition time, while available_at/vintage_at govern information.
        return True

class MarketDataProvider(Protocol):
    def history(self,symbol: str,as_of: str,interval: str='1d') -> DataSnapshot: ...

class FundamentalDataProvider(Protocol):
    def fundamentals(self,symbol: str,as_of: str) -> DataSnapshot: ...

class NewsDataProvider(Protocol):
    def news(self,symbol: str,as_of: str) -> DataSnapshot: ...

class MacroDataProvider(Protocol):
    def series(self,series_id: str,as_of: str) -> DataSnapshot: ...

class FactorDataProvider(Protocol):
    def factors(self,dataset: str,as_of: str) -> DataSnapshot: ...

class FilingsDataProvider(Protocol):
    def filings(self,symbol: str,as_of: str) -> DataSnapshot: ...

def assert_compatible_fallback(primary: SourceMetadata,fallback: SourceMetadata):
    for name in ['symbol','family','exchange_timezone','interval','adjustment_policy']:
        if getattr(primary,name)!=getattr(fallback,name):
            raise ValueError('Incompatible provider fallback: '+name)
    if fallback.fallback_from!=primary.provider_id:
        raise ValueError('Fallback provenance must name primary provider')
    return True
