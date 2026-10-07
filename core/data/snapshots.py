"""Content-addressed immutable snapshot storage using strict UTF-8 JSON."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from core.data.contracts import DataSnapshot, SourceMetadata

_DIGEST = re.compile(r'^[0-9a-f]{64}$')

def _scalar(value):
    if isinstance(value,np.generic): value=value.item()
    if isinstance(value,pd.Timestamp): return {'timestamp':value.isoformat()}
    if value is None or isinstance(value,(str,int,float,bool)): return value
    raise ValueError('Unsupported snapshot scalar: '+type(value).__name__)

class SnapshotStore:
    def __init__(self,directory): self.directory=Path(directory)

    def save(self,snapshot: DataSnapshot):
        frame=snapshot.frame
        if any(isinstance(dtype,pd.CategoricalDtype) for dtype in frame.dtypes):
            raise ValueError('categorical dtype requires an explicitly versioned category schema; convert to scalar strings first')
        if any(not isinstance(c,str) for c in frame.columns) or not frame.columns.is_unique:
            raise ValueError('Snapshot columns must be unique strings')
        document={'schema_version':1,'metadata':snapshot.metadata.to_dict(),
                  'columns':frame.columns.tolist(),'dtypes':[str(t) for t in frame.dtypes],
                  'index':[t.isoformat() for t in frame.index],'index_name':frame.index.name,
                  'index_timezone':str(frame.index.tz),'index_dtype':str(frame.index.dtype),
                  'index_frequency':frame.index.freqstr,
                  'values':[[_scalar(v) for v in row] for row in frame.itertuples(index=False,name=None)]}
        payload=json.dumps(document,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')
        identifier=hashlib.sha256(payload).hexdigest()
        self.directory.mkdir(parents=True,exist_ok=True)
        file=self.directory/(identifier+'.json')
        try:
            with file.open('xb') as stream: stream.write(payload)
        except FileExistsError:
            if file.read_bytes()!=payload: raise ValueError('Snapshot integrity collision')
        return identifier

    def load(self,identifier):
        if not isinstance(identifier,str) or not _DIGEST.fullmatch(identifier):
            raise ValueError('Invalid snapshot identifier')
        payload=(self.directory/(identifier+'.json')).read_bytes()
        if hashlib.sha256(payload).hexdigest()!=identifier: raise ValueError('Snapshot integrity failure')
        document=json.loads(payload.decode('utf-8'))
        if document['schema_version']!=1: raise ValueError('Unsupported snapshot schema')
        values=[[pd.Timestamp(v['timestamp']) if isinstance(v,dict) and set(v)=={'timestamp'} else v
                 for v in row] for row in document['values']]
        index=pd.DatetimeIndex(pd.to_datetime(document['index'],utc=True)).tz_convert(document['index_timezone'])
        index=index.astype(document['index_dtype'])
        if document['index_frequency'] is not None: index.freq=document['index_frequency']
        frame=pd.DataFrame(values,columns=document['columns'],index=index)
        frame.index.name=document['index_name']
        for column,dtype in zip(document['columns'],document['dtypes']): frame[column]=frame[column].astype(dtype)
        return DataSnapshot(frame,SourceMetadata(**document['metadata']))
