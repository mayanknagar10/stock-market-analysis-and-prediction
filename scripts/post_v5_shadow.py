"""Explicit batch shadow forecast; no new model fitting or production recommendations."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.post_v5.shadow import forecast_from_run,DIRECTORY,STUDY
from research.post_v5.archive import RunArchive
from core.forecasting.service import active_research,load_research_model
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-id',required=True)
parser.add_argument('--tickers',default=None);args=parser.parse_args()
universe=json.loads((ROOT/'config/research'/STUDY/'UNIVERSE.json').read_text(encoding='utf-8'))
tickers=args.tickers.split(',') if args.tickers else [row['ticker'] for row in universe['members']]
archive=RunArchive(DIRECTORY);pointer=active_research()
bundles={h:load_research_model('india_equity',h,pointer) for h in [1,5,10,20]}
results={'study_id':STUDY,'run_id':args.run_id,'issued_at':datetime.now(timezone.utc).isoformat(),'successes':{},'failures':{},
    'model_version':pointer['model_version'],'research_only':True,'new_final_candidate_trained':False}
for ticker in tickers:
    try:
        result=forecast_from_run(args.run_id,ticker,bundles=bundles)
        results['successes'][ticker]=[item['record']['prediction_id'] for item in result['forecasts']]
        print(ticker,'RECORDED',len(result['forecasts']),'LOW/ABSTAINED',flush=True)
    except Exception as error:
        results['failures'][ticker]=str(error)
        archive.put('attempts',{'run_id':args.run_id,'ticker':ticker,'attempted_at':datetime.now(timezone.utc).isoformat(),
            'status':'forecast_failed','error':str(error),'study_id':STUDY})
        print(ticker,'FAILED',str(error),flush=True)
identifier=archive.put('attempts',{**results,'status':'batch_complete'})
file=ROOT/'reports/post_v5/shadow'/f'{identifier}.json';file.parent.mkdir(parents=True,exist_ok=True)
with file.open('x',encoding='utf-8') as stream:json.dump(results,stream,indent=2)
print('SHADOW_REPORT',file,flush=True)
