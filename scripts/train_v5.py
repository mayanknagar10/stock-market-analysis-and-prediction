"""Explicit research model training; never promotes production or opens the final test."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.evaluate_v4 import read_snapshot
from core.data.snapshots import SnapshotStore
from core.forecasting.panel import build_panel
from core.forecasting.training import train_research_bundle
from core.forecasting.registry import save_bundle
from core.validation.splits import development_rows
from core.validation.metrics import forecast_metrics,grouped_metrics

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--references',type=Path,default=ROOT/'data/v5/REFERENCE_CAPTURE.json')
    args=parser.parse_args()
    source,frames=read_snapshot(args.manifest)
    references_manifest=json.loads(args.references.read_text(encoding='utf-8'))
    store=SnapshotStore(ROOT/'data/v5/snapshots')
    references={key:store.load(identifier) for key,identifier in references_manifest['snapshots'].items()}
    settings=json.loads((ROOT/'config/context_sources.json').read_text(encoding='utf-8'))
    protocol=json.loads((ROOT/'config/v5_validation.json').read_text(encoding='utf-8'))
    boundary=pd.Timestamp(protocol['final_test_start'],tz='UTC')
    version='research-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    report={'model_version':version,'scope':'research_only','point_in_time_verified':False,'final_test_evaluated':False,
        'final_test_start':protocol['final_test_start'],'sources':args.manifest.name,
        'reference_failures':references_manifest['failures'],'models':{}}
    paths={}
    records=[]
    for family in protocol['family_separation']:
        for horizon in protocol['horizons']:
            X,y,meta=build_panel(frames,source,horizon,family,references,settings)
            groups=X.attrs['feature_groups']
            eligible=development_rows(meta,boundary).index
            X,y,meta=X.loc[eligible].reset_index(drop=True),y.loc[eligible].reset_index(drop=True),meta.loc[eligible].reset_index(drop=True)
            provenance={'stock_manifest':args.manifest.name,'stock_snapshots':{k:v['sha256'] for k,v in source['assets'].items()},
                'references':references_manifest['snapshots'],'provider_version':source['provider_version'],
                'availability_policy':'Conservative session estimates; not verified historical publication/revision records'}
            from core.forecasting.relative import relative_targets
            alpha=relative_targets(meta,y,references,settings,family)
            bundle,rows=train_research_bundle(X,y,meta,family,horizon,version,provenance,groups,relative=alpha)
            directory=ROOT/'models/v5'/version/family/str(horizon)
            save_bundle(bundle,directory)
            key=family+'_'+str(horizon)
            paths[key]=str(directory.relative_to(ROOT)).replace('\\','/')
            report['models'][key]={'metadata':bundle['metadata'],'policy':bundle['policy'],
                'breakdowns':grouped_metrics(rows,['ticker','sector','year','market_regime']),
                'zero_return_metrics':forecast_metrics(rows.actual,[0.]*len(rows))}
            rows['horizon']=horizon
            records.append(rows)
            print(key,'trained; calibration accepted:',bundle['metadata']['calibration_accepted'],flush=True)
    output=ROOT/'reports/validation'/version
    output.mkdir(parents=True,exist_ok=False)
    (output/'DEVELOPMENT.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    pd.concat(records,ignore_index=True).to_csv(output/'predictions.csv',index=False)
    pointer=ROOT/'models/v5/ACTIVE_RESEARCH.json'
    temporary=pointer.with_suffix('.tmp')
    temporary.write_text(json.dumps({'model_version':version,'models':paths,'report':str(output.relative_to(ROOT))},indent=2),encoding='utf-8')
    temporary.replace(pointer)
    print('Research candidate saved:',version,flush=True)

if __name__=='__main__': main()
