"""Explicit prospective capture; every provider failure and original member is retained."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from core.data.providers import YahooMarketDataProvider
from research.post_v5.archive import RunArchive
STUDY='prospective-india-20261008-v1'
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--period',default='5y');parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    universe_file=ROOT/'config/research'/STUDY/'UNIVERSE.json'
    universe=json.loads(universe_file.read_text(encoding='utf-8'))
    settings=json.loads((ROOT/'config/context_sources.json').read_text(encoding='utf-8'))
    sources={r['ticker']:(r['ticker'],False) for r in universe['members']}
    sources.update({key:(symbol,True) for key,symbol in settings['sources'].items()})
    snapshots={};failures={}
    archive=RunArchive(ROOT/'data/research'/STUDY)
    def fetch(key,value):
        symbol,reference=value
        snapshot=YahooMarketDataProvider().capture_history(symbol,period=args.period,reference=reference)
        identifier=archive.save_snapshot(snapshot)
        return key,snapshot,identifier
    with ThreadPoolExecutor(max_workers=max(1,min(3,args.workers))) as pool:
        jobs={pool.submit(fetch,key,value):key for key,value in sources.items()}
        for future in as_completed(jobs):
            key=jobs[future]
            try:
                key,snapshot,identifier=future.result();snapshots[key]=snapshot
                print(key,'CAPTURED',len(snapshot.frame),identifier[:12],flush=True)
            except Exception as error:
                failures[key]=str(error);print(key,'FAILED',str(error)[:180],flush=True)
    cutoff=datetime.now(timezone.utc).isoformat()
    identifier=archive.save_run(snapshots,failures,cutoff,STUDY,universe_hash=hashlib.sha256(universe_file.read_bytes()).hexdigest())
    print('RUN_ID',identifier,flush=True)
    public={'run_id':identifier,'study_id':STUDY,'captured_at':cutoff,'n_members':len(universe['members']),
        'captured_equities':sum(k.endswith('.NS') for k in snapshots),'captured_references':sum(not k.endswith('.NS') for k in snapshots),
        'failures':failures,'input_archive_verified_at_capture':True,'historical_point_in_time_certified':False,
        'ledger_predictions':0,'prospective_matured_outcomes':0}
    directory=ROOT/'reports/post_v5/captures';directory.mkdir(parents=True,exist_ok=True)
    with (directory/(identifier+'.json')).open('x',encoding='utf-8') as stream:json.dump(public,stream,indent=2)
    if not snapshots: raise SystemExit(2)
if __name__=='__main__':main()
