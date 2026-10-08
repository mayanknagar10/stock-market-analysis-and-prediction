"""Freeze official membership once; failures never remove study members."""
import csv
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
URL='https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv'
PAGE='https://www.niftyindices.com/indices/equity/broad-based-indices/nifty-100'
def parse_membership(payload):
    reader=csv.DictReader(io.StringIO(payload.decode('utf-8-sig')))
    required={'Company Name','Industry','Symbol','ISIN Code'}
    if not required.issubset(reader.fieldnames or []): raise ValueError('Official membership schema missing')
    rows=[]
    for row in reader:
        symbol=row['Symbol'].strip().upper();isin=row['ISIN Code'].strip()
        if not symbol or not isin: raise ValueError('Missing member identity')
        rows.append({'ticker':symbol+'.NS','exchange_symbol':symbol,'isin':isin,'company':row['Company Name'].strip(),
            'industry':row['Industry'].strip(),'series':row.get('Series','').strip(),'status':'included_at_freeze'})
    if len({r['ticker'] for r in rows})!=len(rows) or len({r['isin'] for r in rows})!=len(rows): raise ValueError('Duplicate member identity')
    if not rows: raise ValueError('Empty membership')
    return sorted(rows,key=lambda r:r['ticker'])
def freeze(directory):
    import requests
    directory=Path(directory);target=directory/'UNIVERSE.json'
    if target.exists(): raise ValueError('Universe already frozen; refusal to replace membership')
    response=requests.get(URL,headers={'User-Agent':'Mozilla/5.0','Referer':PAGE},timeout=40)
    response.raise_for_status();payload=response.content
    rows=parse_membership(payload)
    if not 80<=len(rows)<=120: raise ValueError('Unexpected official Nifty 100 membership size')
    frozen=datetime.now(timezone.utc).isoformat();digest=hashlib.sha256(payload).hexdigest()
    document={'study_id':'prospective-india-20261008-v1','frozen_at':frozen,'membership_date':frozen[:10],
        'source_url':URL,'source_page':PAGE,'raw_membership_sha256':digest,'raw_file':'NIFTY100_AT_FREEZE.csv',
        'selection_rule':'All securities in official Nifty 100 constituent CSV retrieved at freeze; no performance or provider-success filtering',
        'retention_rule':'Keep failed/suspended/delisted/renamed members; append mapping/status events, never rewrite membership',
        'historical_survivorship_bias_resolved':False,'members':rows,'n_securities':len(rows),
        'industries':sorted({r['industry'] for r in rows})}
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/document['raw_file']).open('xb') as stream:stream.write(payload)
    with target.open('x',encoding='utf-8') as stream:json.dump(document,stream,indent=2)
    return document
