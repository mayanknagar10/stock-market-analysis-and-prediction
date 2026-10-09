"""Registered NSE sessions. Unsupported dates fail closed; never infer holidays from quotes."""
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
from zoneinfo import ZoneInfo
from application.config import ROOT
IST=ZoneInfo("Asia/Kolkata")
CALENDAR_VERSION="nse-cm-2026-registered-v1"
class CalendarUnavailable(ValueError): pass
def aware(value):
    value=datetime.fromisoformat(value.replace("Z","+00:00")) if isinstance(value,str) else value
    if value.tzinfo is None or value.utcoffset() is None: raise ValueError("Timezone-aware timestamp required")
    return value.astimezone(timezone.utc)
class Calendar:
    version=CALENDAR_VERSION
    def __init__(self):
        file=ROOT/"config/collection/calendar_sources/nse-holidays-2026.json"
        payload=file.read_bytes()
        if hashlib.sha256(payload).hexdigest()!="933cf5dc6dd4efe3b8db8ce6ecd6ecb2582929f5e0b4faffdb88e11cf4efdfbe":
            raise CalendarUnavailable("Registered calendar source hash mismatch")
        self.holidays={datetime.strptime(r["tradingDate"],"%d-%b-%Y").date() for r in json.loads(payload)["CM"]}
    def session(self,day):
        day=date.fromisoformat(day) if isinstance(day,str) else day
        if day.year!=2026: raise CalendarUnavailable("No registered calendar for requested year")
        if day==date(2026,11,8): raise CalendarUnavailable("Muhurat timing not yet registered")
        if day==date(2026,2,1): return True
        return day.weekday()<5 and day not in self.holidays
    def close(self,day):
        day=date.fromisoformat(day) if isinstance(day,str) else day
        if not self.session(day): raise CalendarUnavailable("Not a registered trading session")
        return datetime.combine(day,time(15,30),IST).astimezone(timezone.utc)
    def cutoff(self,day):
        day=date.fromisoformat(day) if isinstance(day,str) else day
        return datetime.combine(day+timedelta(days=1),time(6,30),IST).astimezone(timezone.utc)
    def advance(self,day,horizon):
        day=date.fromisoformat(day) if isinstance(day,str) else day
        if not self.session(day): raise CalendarUnavailable("Invalid origin session")
        if horizon not in {1,5,10,20}: raise ValueError("Unsupported horizon")
        count=0
        while count<horizon:
            day+=timedelta(days=1)
            if self.session(day): count+=1
        return day
    def origin_for(self,now):
        return (aware(now).astimezone(IST).date()-timedelta(days=1)).isoformat()
