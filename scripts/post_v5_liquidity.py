"""Append observed liquidity diagnostics, never revise membership or price captures."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.post_v5.archive import RunArchive
from research.post_v5.shadow import DIRECTORY,STUDY
from research.post_v5.diagnostics import liquidity_diagnostics
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-id',required=True);args=parser.parse_args()
archive=RunArchive(DIRECTORY);run=archive.load_run(args.run_id)
universe=json.loads((ROOT/'config/research'/STUDY/'UNIVERSE.json').read_text(encoding='utf-8'))
record={'study_id':STUDY,'input_run_id':args.run_id,'measured_at':datetime.now(timezone.utc).isoformat(),'selection_effect':'none','members':{}}
for member in universe['members']:
    key=member['ticker']
    try:record['members'][key]=liquidity_diagnostics(archive.load_snapshot(run['snapshots'][key]))
    except Exception as error:record['members'][key]={'status':'unavailable','reason':str(error),'selection_effect':'none; member retained'}
identifier=archive.put('attempts',{**record,'status':'liquidity_diagnostics'})
file=ROOT/'reports/post_v5/liquidity'/f'{identifier}.json';file.parent.mkdir(parents=True,exist_ok=True)
with file.open('x',encoding='utf-8') as stream:json.dump(record,stream,indent=2,allow_nan=False)
print('LIQUIDITY_DIAGNOSTICS',file)
