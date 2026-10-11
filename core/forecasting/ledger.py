"""Append-only ledger with strict provenance, price identities and explicit session maturity."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
from core.validation.alignment import aware_index,session_cutoffs

REQUIRED={'prediction_id','ticker','market','forecast_as_of','generated_at','information_cutoff','timezone','horizon',
    'current_price','central_log_return','central_price','quantiles','p_positive','confidence','abstain','model_agreement',
    'regime','data_quality','ood_score','model_version','feature_version','training_cutoff','provider_versions',
    'snapshot_ids','feature_schema_hash','code_version','scope','point_in_time_verified'}

class PredictionLedger:
    def __init__(self,path):
        self.path=Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self._connect() as connection:
            connection.executescript('''
                CREATE TABLE IF NOT EXISTS forecasts (prediction_id TEXT PRIMARY KEY, payload TEXT NOT NULL, payload_hash TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS outcomes (prediction_id TEXT PRIMARY KEY REFERENCES forecasts(prediction_id), payload TEXT NOT NULL, payload_hash TEXT NOT NULL);
                CREATE TRIGGER IF NOT EXISTS no_forecast_update BEFORE UPDATE ON forecasts BEGIN SELECT RAISE(ABORT,'immutable forecasts'); END;
                CREATE TRIGGER IF NOT EXISTS no_forecast_delete BEFORE DELETE ON forecasts BEGIN SELECT RAISE(ABORT,'immutable forecasts'); END;
                CREATE TRIGGER IF NOT EXISTS no_outcome_update BEFORE UPDATE ON outcomes BEGIN SELECT RAISE(ABORT,'immutable outcomes'); END;
                CREATE TRIGGER IF NOT EXISTS no_outcome_delete BEFORE DELETE ON outcomes BEGIN SELECT RAISE(ABORT,'immutable outcomes'); END;
            ''')
    def _connect(self):
        connection=sqlite3.connect(self.path,timeout=30)
        connection.execute('PRAGMA foreign_keys=ON')
        return connection
    @staticmethod
    def _payload(record):
        text=json.dumps(record,sort_keys=True,separators=(',',':'),allow_nan=False)
        return text,hashlib.sha256(text.encode('utf-8')).hexdigest()
    def _validated_payload(self,record):
        if not REQUIRED.issubset(record): raise ValueError('Incomplete prediction provenance')
        if type(record['horizon']) is not int or record['horizon'] not in {1,5,10,20}: raise ValueError('Invalid horizon')
        cutoff,asof,generated,training=aware_index([record['information_cutoff'],record['forecast_as_of'],record['generated_at'],record['training_cutoff']])
        if cutoff>asof or asof>generated or training>=asof: raise ValueError('Invalid provenance timestamp bounds')
        if type(record['point_in_time_verified']) is not bool or type(record['abstain']) is not bool: raise ValueError('Invalid evidence booleans')
        if not record['snapshot_ids'] or not record['provider_versions']: raise ValueError('Missing source lineage')
        q=np.asarray(record['quantiles'],dtype=float)
        if q.shape!=(5,) or not np.isfinite(q).all() or np.any(np.diff(q)<0): raise ValueError('Invalid quantiles')
        if not 0<=record['p_positive']<=1 or record['current_price']<=0 or record['central_price']<=0: raise ValueError('Invalid forecast values')
        implied=float(record['current_price']*np.exp(record['central_log_return']))
        if not np.isfinite(implied) or not np.isclose(implied,record['central_price'],rtol=1e-8,atol=1e-6):
            raise ValueError('Inconsistent price/return forecast')
        if record['scope']=='research_only' and record['confidence']!='LOW': raise ValueError('Research confidence must remain LOW')
        return self._payload(record)
    def append(self,record):
        return self.append_many([record])[0]
    def append_many(self,records):
        prepared=[(record['prediction_id'],*self._validated_payload(record)) for record in records]
        try:
            with self._connect() as connection:
                connection.executemany('INSERT INTO forecasts VALUES (?,?,?)',prepared)
        except sqlite3.IntegrityError as exc: raise ValueError('Prediction already exists; batch rolled back') from exc
        return [row[0] for row in prepared]
    def attach_outcome(self,prediction_id,outcome):
        if outcome.get('status') not in {'matured','invalidated_due_to_bad_data'}: raise ValueError('Invalid outcome status')
        with self._connect() as connection:
            stored=connection.execute('SELECT payload FROM forecasts WHERE prediction_id=?',(prediction_id,)).fetchone()
        if stored is None: raise ValueError('Unknown prediction')
        prediction=json.loads(stored[0])
        if outcome['status']=='matured':
            fields={'actual_return','actual_log_return','actual_future_price','maturity_timestamp','resolved_at',
                    'prediction_error_log','direction_correct','interval_80_hit','interval_50_hit','outcome_snapshot_id'}
            if not fields.issubset(outcome): raise ValueError('Incomplete matured outcome')
            maturity,resolved,generated=aware_index([outcome['maturity_timestamp'],outcome['resolved_at'],prediction['generated_at']])
            if maturity<=generated or resolved<maturity: raise ValueError('Invalid outcome maturity bounds')
            values=[outcome['actual_return'],outcome['actual_log_return'],outcome['actual_future_price'],outcome['prediction_error_log']]
            if not np.isfinite(values).all() or outcome['actual_future_price']<=0: raise ValueError('Invalid realized values')
            if not np.isclose(np.expm1(outcome['actual_log_return']),outcome['actual_return'],rtol=1e-8,atol=1e-10):
                raise ValueError('Inconsistent outcome return units')
            if not np.isclose(outcome['actual_log_return']-prediction['central_log_return'],outcome['prediction_error_log'],atol=1e-10):
                raise ValueError('Inconsistent realized prediction error')
            actual=outcome['actual_log_return']
            expected_hits={'direction_correct':bool((actual>0)==(prediction['p_positive']>=.5)),
                'interval_80_hit':bool(prediction['quantiles'][0]<=actual<=prediction['quantiles'][4]),
                'interval_50_hit':bool(prediction['quantiles'][1]<=actual<=prediction['quantiles'][3])}
            if any(outcome[field]!=value for field,value in expected_hits.items()): raise ValueError('Inconsistent realized direction/coverage flags')
            for field in ['direction_correct','interval_80_hit','interval_50_hit']:
                if type(outcome[field]) is not bool: raise ValueError('Invalid realized outcome boolean')
        elif not outcome.get('reason'): raise ValueError('Invalidation must explain data failure')
        payload,digest=self._payload(outcome)
        try:
            with self._connect() as connection: connection.execute('INSERT INTO outcomes VALUES (?,?,?)',(prediction_id,payload,digest))
        except sqlite3.IntegrityError as exc: raise ValueError('Unknown prediction or outcome already attached') from exc
    def rows(self):
        with self._connect() as connection:
            records=connection.execute('SELECT f.payload,f.payload_hash,o.payload,o.payload_hash FROM forecasts f LEFT JOIN outcomes o USING(prediction_id) ORDER BY f.rowid').fetchall()
        result=[]
        for payload,digest,outcome,outcome_digest in records:
            if hashlib.sha256(payload.encode('utf-8')).hexdigest()!=digest: raise ValueError('Ledger prediction integrity failure')
            row=json.loads(payload)
            if outcome is not None:
                if hashlib.sha256(outcome.encode('utf-8')).hexdigest()!=outcome_digest: raise ValueError('Ledger outcome integrity failure')
                row['outcome']=json.loads(outcome)
            else: row['outcome']=None
            result.append(row)
        return result

def resolve_outcome(prediction,bars,as_of,benchmark=None,sector=None,expected_sessions=None):
    if expected_sessions is None or bars.empty or not {'Close','Adj Close'}.issubset(bars): return None
    times=aware_index(bars.index,'outcome source timestamps')
    if not times.is_unique or not times.is_monotonic_increasing: raise ValueError('Invalid outcome source order')
    family=prediction['market']
    zone='Asia/Kolkata' if family=='india_equity' else 'America/New_York'
    dates=times.tz_convert(zone).tz_localize(None).normalize()
    completed=session_cutoffs(dates,family)+pd.Timedelta('20min')
    information=aware_index([prediction['information_cutoff']])[0]
    if prediction.get('origin_session'):
        origin=pd.Timestamp(prediction['origin_session']).tz_localize(None).normalize()
    else:
        known=np.flatnonzero(completed<=information)
        if not len(known): return None
        origin=dates[int(known[-1])]
    positions=np.flatnonzero(dates==origin)
    if len(positions)!=1: return None
    start=int(positions[0])
    if completed[start]>information: raise ValueError('Origin price was not known at forecast cutoff')
    calendar=pd.DatetimeIndex([pd.Timestamp(t).date() for t in expected_sessions])
    if not calendar.is_unique or not calendar.is_monotonic_increasing: raise ValueError('Invalid independent calendar')
    location=np.flatnonzero(calendar==origin)
    if len(location)!=1 or location[0]+prediction['horizon']>=len(calendar): return None
    maturity_date=calendar[int(location[0])+prediction['horizon']]
    locations=np.flatnonzero(dates==maturity_date)
    if len(locations)!=1: return None
    end=int(locations[0])
    if completed[end]>aware_index([as_of])[0]: return None
    initial=float(bars['Adj Close'].iloc[start]); future=float(bars['Adj Close'].iloc[end])
    if not np.isfinite([initial,future]).all() or initial<=0 or future<=0:
        return {'status':'invalidated_due_to_bad_data','reason':'Invalid adjusted outcome prices'}
    actual_log=float(np.log(future/initial)); actual=float(np.expm1(actual_log))
    actions_known={'Dividends','Stock Splits'}.issubset(bars)
    actions=bool(((bars.iloc[start+1:end+1].get('Dividends',pd.Series(dtype=float))!=0).any()) or
                 ((bars.iloc[start+1:end+1].get('Stock Splits',pd.Series(dtype=float))!=0).any()))
    result={'status':'matured','resolved_at':datetime.now(timezone.utc).isoformat(),'maturity_timestamp':completed[end].isoformat(),
        'actual_return':actual,'actual_log_return':actual_log,'actual_future_price':float(bars.Close.iloc[end]),
        'prediction_error_log':actual_log-prediction['central_log_return'],
        'direction_correct':bool((actual_log>0)==(prediction['p_positive']>=.5)),
        'interval_80_hit':bool(prediction['quantiles'][0]<=actual_log<=prediction['quantiles'][4]),
        'interval_50_hit':bool(prediction['quantiles'][1]<=actual_log<=prediction['quantiles'][3]),
        'corporate_actions_status':'PRESENT' if actions else 'NONE' if actions_known else 'UNKNOWN',
        'price_error':None if actions or not actions_known else float(bars.Close.iloc[end]-prediction['central_price']),
        'confidence_scored':not prediction['abstain'],'alpha_market':None,'alpha_sector':None}
    for name,reference in [('market',benchmark),('sector',sector)]:
        if reference is not None:
            aligned=reference.reindex([origin,dates[end]])
            if aligned.notna().all() and (aligned>0).all():
                result['alpha_'+name]=actual_log-float(np.log(aligned.iloc[1]/aligned.iloc[0]))
    return result
