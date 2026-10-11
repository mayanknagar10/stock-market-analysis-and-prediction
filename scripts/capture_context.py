"""Explicit free reference-data capture; never runs from a page load."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from core.data.providers import YahooMarketDataProvider
from core.data.snapshots import SnapshotStore

def main():
    settings=json.loads((ROOT/'config/context_sources.json').read_text(encoding='utf-8'))
    store=SnapshotStore(ROOT/'data/v5/snapshots')
    provider=YahooMarketDataProvider()
    result={'mode':'research_only','point_in_time_verified':False,'snapshots':{},'failures':{}}
    for key,symbol in settings['sources'].items():
        try:
            data=provider.capture_history(symbol,reference=True)
            result['snapshots'][key]=store.save(data)
            print(key,len(data.frame),'captured',flush=True)
        except Exception as exc:
            result['failures'][key]=str(exc)
            print(key,'unavailable:',str(exc),flush=True)
    file=ROOT/'data/v5/REFERENCE_CAPTURE.json'
    file.parent.mkdir(parents=True,exist_ok=True)
    with file.open('x',encoding='utf-8') as stream: json.dump(result,stream,indent=2)
    print(file,flush=True)

if __name__=='__main__': main()
