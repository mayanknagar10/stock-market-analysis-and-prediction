"""Purged walk-forward direct-regression research; no final-test access or PIT certification."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.evaluate_v4 import read_snapshot
from core.forecasting.panel import build_panel
from core.forecasting.heads import DirectRegressor
from core.validation.splits import development_rows,purged_expanding_splits,assert_disjoint_label_intervals
from core.validation.metrics import forecast_metrics,grouped_metrics

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--folds',type=int,default=3)
    parser.add_argument('--output',type=Path,default=ROOT/'reports/validation/V5_DIRECT_REGRESSION.json')
    args=parser.parse_args()
    protocol=json.loads((ROOT/'config/v5_validation.json').read_text(encoding='utf-8'))
    boundary=pd.Timestamp(protocol['final_test_start'],tz='UTC')
    manifest,frames=read_snapshot(args.manifest)
    report={'mode':'research_only','point_in_time_verified':False,'final_test_evaluated':False,
            'final_test_start':protocol['final_test_start'],'snapshot_manifest':args.manifest.name,
            'assumptions':['Retrospective Yahoo adjusted vintage; historical availability unverified',
                'Origin timestamps estimated as standard session close +20min',
                'Horizons count observed sessions; verified exchange calendar still required for production',
                'Stock regime proxy is descriptive; contextual regime layer follows in phase6'], 'results':{}}
    records=[]
    for family in protocol['family_separation']:
        for horizon in protocol['horizons']:
            X,y,meta=build_panel(frames,manifest,horizon,family)
            admitted=development_rows(meta,boundary).index
            X,y,meta=X.loc[admitted].reset_index(drop=True),y.loc[admitted].reset_index(drop=True),meta.loc[admitted].reset_index(drop=True)
            folds=list(purged_expanding_splits(meta,protocol['minimum_train_sessions'],protocol['validation_sessions']))[-args.folds:]
            if len(folds)<2: raise ValueError('Not enough development folds')
            rows=[]
            for fold,(train,test) in enumerate(folds):
                assert_disjoint_label_intervals(meta.iloc[train],meta.iloc[test])
                cutoff=meta.iloc[test].feature_time.min()
                model=DirectRegressor(family,horizon).fit(X.iloc[train],y.iloc[train],
                    meta.iloc[train].feature_time,meta.iloc[train].label_end,cutoff)
                predicted=model.predict(X.iloc[test])['central_log_return']
                drift=float(y.iloc[train].mean())
                for position,index in enumerate(test):
                    base={**meta.iloc[index].to_dict(),'actual':float(y.iloc[index]),'horizon':horizon,'fold':fold}
                    for method,estimate in [('direct_xgb_lgb',predicted[position]),('zero_return',0.),('historical_drift',drift)]:
                        rows.append({**base,'method':method,'predicted':float(estimate)})
            frame=pd.DataFrame(rows)
            for method,group in frame.groupby('method'):
                report['results'][family+'_'+str(horizon)+'d_'+method]={
                    'aggregate':forecast_metrics(group.actual,group.predicted),
                    'breakdowns':grouped_metrics(group,['ticker','sector','year','stock_regime_proxy','fold'])}
            records.extend(rows)
            print(family,horizon,'direct target,',len(folds),'purged folds',flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream: json.dump(report,stream,indent=2,allow_nan=False)
    frame=pd.DataFrame(records)
    with args.output.with_suffix('.csv').open('x',encoding='utf-8',newline='') as stream: frame.to_csv(stream,index=False)
    print('Persisted',args.output,flush=True)

if __name__=='__main__': main()
