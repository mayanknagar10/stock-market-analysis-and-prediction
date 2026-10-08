"""Actual retrieval-time input archives; no historical publication certification."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
from core.data.contracts import DataSnapshot,SourceMetadata
from core.validation.alignment import aware_index

def encode(value):
    if isinstance(value,np.generic): value=value.item()
    if isinstance(value,float) and not np.isfinite(value): return {'nonfinite':'nan' if np.isnan(value) else 'inf' if value>0 else '-inf'}
    if value is None or isinstance(value,(str,int,float,bool)): return value
    if isinstance(value,pd.Timestamp): return {'timestamp':value.isoformat()}
    raise ValueError('Unsupported raw source scalar: '+type(value).__name__)

def decode(value):
    if isinstance(value,dict) and 'nonfinite' in value: return float(value['nonfinite'])
    if isinstance(value,dict) and 'timestamp' in value: return pd.Timestamp(value['timestamp'])
    return value

class RunArchive:
    def __init__(self,directory): self.directory=Path(directory)
    def put(self,kind,document):
        if kind not in {'runs','snapshots','forecasts','attempts','outcomes'}: raise ValueError('Invalid archive kind')
        payload=json.dumps(document,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')
        identifier=hashlib.sha256(payload).hexdigest()
        directory=self.directory/kind; directory.mkdir(parents=True,exist_ok=True)
        file=directory/(identifier+'.json')
        try:
            with file.open('xb') as stream: stream.write(payload)
        except FileExistsError:
            if file.read_bytes()!=payload: raise ValueError('Archive integrity collision')
        return identifier
    def get(self,kind,identifier):
        if not re.fullmatch('[0-9a-f]{64}',identifier): raise ValueError('Invalid archive identifier')
        payload=(self.directory/kind/(identifier+'.json')).read_bytes()
        if hashlib.sha256(payload).hexdigest()!=identifier: raise ValueError('Archive integrity failure')
        return json.loads(payload)
    def save_snapshot(self,snapshot):
        frame=snapshot.frame
        if not frame.columns.is_unique: raise ValueError('Duplicate source columns')
        return self.put('snapshots',{'format_version':1,'metadata':snapshot.metadata.to_dict(),
            'columns':list(frame.columns),'dtypes':[str(t) for t in frame.dtypes],
            'index':[t.isoformat() for t in frame.index],'index_timezone':str(frame.index.tz),'index_name':frame.index.name,
            'index_dtype':str(frame.index.dtype),'index_frequency':frame.index.freqstr,
            'values':[[encode(v) for v in row] for row in frame.itertuples(index=False,name=None)]})
    def load_snapshot(self,identifier):
        doc=self.get('snapshots',identifier)
        index=pd.DatetimeIndex(pd.to_datetime(doc['index'],utc=True)).tz_convert(doc['index_timezone']).astype(doc['index_dtype'])
        if doc['index_frequency'] is not None: index.freq=doc['index_frequency']
        frame=pd.DataFrame([[decode(v) for v in row] for row in doc['values']],columns=doc['columns'],index=index)
        frame.index.name=doc['index_name']
        for column,dtype in zip(doc['columns'],doc['dtypes']): frame[column]=frame[column].astype(dtype)
        return DataSnapshot(frame,SourceMetadata(**doc['metadata']))
    def save_run(self,snapshots,failures,cutoff,study,purpose='capture_only',universe_hash=None):
        asof=aware_index([cutoff])[0]
        identities={};sources={}
        for key,snapshot in snapshots.items():
            meta=snapshot.metadata
            if aware_index([meta.fetched_at])[0]>asof: raise ValueError('Source retrieval is after prediction cutoff')
            identities[key]=self.save_snapshot(snapshot)
            sources[key]={'retrieved_at':meta.fetched_at,'source_observation_timestamp':meta.source_timestamp,
                'observation_timestamp_kind':'provider session label; not an exact historical publication timestamp',
                'provider':meta.provider_id,'provider_version':meta.provider_version,
                'input_observed_before_prediction':True,'historical_availability_verified':False}
        return self.put('runs',{'format_version':1,'study_id':study,'purpose':purpose,'cutoff':asof.isoformat(),
            'archived_at':datetime.now(timezone.utc).isoformat(),'snapshots':identities,'sources':sources,'failures':failures,
            'universe_hash':universe_hash,'evidence_type':'prospective_input_capture','historical_point_in_time_certified':False})
    def load_run(self,identifier):
        document=self.get('runs',identifier)
        cutoff=aware_index([document['cutoff']])[0]
        for key,snapshot_id in document['snapshots'].items():
            source=self.load_snapshot(snapshot_id)
            if aware_index([source.metadata.fetched_at])[0]>cutoff: raise ValueError('Source retrieval exceeds stored cutoff')
        return document
