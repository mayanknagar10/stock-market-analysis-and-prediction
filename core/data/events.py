"""Fail-safe Indian event interfaces. No unverified scraping or invented coverage."""
from dataclasses import dataclass
from typing import Protocol
from core.data.contracts import DataSnapshot

class CorporateEventsProvider(Protocol):
    def announcements(self,symbol: str,as_of: str) -> DataSnapshot: ...

@dataclass(frozen=True)
class EventCoverage:
    nse_announcements: str='NOT_CONFIGURED'
    bse_announcements: str='NOT_CONFIGURED'
    sebi: str='NOT_CONFIGURED'
    investor_relations: str='NOT_CONFIGURED'
    point_in_time_verified: bool=False
    forecast_usage: str='NONE'

EVENT_COVERAGE=EventCoverage()

class UnsupportedEventProvider:
    def announcements(self,symbol,as_of):
        raise RuntimeError('Reliable timestamped event provider not configured; no scraping fallback')
