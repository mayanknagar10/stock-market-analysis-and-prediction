"""Replay native V5 only on exact pre-boundary fit/validation rows; never retrains."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import uuid
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.evaluate_v4 import read_snapshot
from core.data.snapshots import SnapshotStore
from core.data.contracts import DataSnapshot
from core.forecasting.service import active_research,load_research_model
from core.forecasting.panel import build_panel
from core.validation.splits import development_rows
from core.validation.metrics import forecast_metrics
from research.post_v5.development import clip_frame,BOUNDARY
from research.post_v5.preservation import verify

def metrics(bundle,X,y):
    predicted=bundle['regression'].predict(X)['central_log_return']
    distribution=bundle['distribution'].predict(bundle['regression'].scaled(X))
    probability=bundle['calibrator'].predict(distribution['probability']) if bundle['calibrator'] is not None else distribution['probability']
    quantiles=distribution['quantiles'].copy();quantiles[:,0]-=bundle['conformal_margin'];quantiles[:,4]+=bundle['conformal_margin']
    scores=forecast_metrics(y,predicted,probability,quantiles)
    scores['roc_auc']=float(roc_auc_score(np.asarray(y)>0,probability)) if len(np.unique(np.asarray(y)>0))>1 else None
    return scores

def main():
    verify(ROOT,ROOT/'reports/post_v5/V5_PRESERVATION.json')
    pointer=active_research();assert pointer['model_version']=='research-20261008T190859'
    bundle=load_research_model('india_equity',1,pointer);provenance=bundle['metadata']['source_provenance']
    manifest,frames=read_snapshot(ROOT/'data/snapshots'/provenance['stock_manifest'])
    frames={key:clip_frame(frame) for key,frame in frames.items()}
    store=SnapshotStore(ROOT/'data/v5/snapshots');references={}
    for key,identity in provenance['references'].items():
        snapshot=store.load(identity);frame=clip_frame(snapshot.frame)
        if not frame.empty:references[key]=DataSnapshot(frame,snapshot.metadata)
    settings=json.loads((ROOT/'config/context_sources.json').read_text(encoding='utf-8'))
    output={'diagnosis_id':'native-v5-development-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8],
        'model_version':pointer['model_version'],'created_at':datetime.now(timezone.utc).isoformat(),'development_cutoff':BOUNDARY.isoformat(),
        'source_manifest':provenance['stock_manifest'],'source_reference_ids':provenance['references'],'models':{},
        'model_training_performed':False,'final_period_read_or_evaluated':False,
        'note':'Train errors replay base-fitting rows with the released blend/calibration learned later in development; they are optimistic in-sample diagnostics, not OOS evidence.'}
    lines=['# Native V5 train/validation diagnostic addendum','',output['note'],
        '', 'Exact original snapshots were trimmed before feature/target transformations. Reconstructed fit matrices and labels must match saved analogue arrays. Validation replay must match stored development metrics. No new training/calibration, final data, model selection or architecture change occurs.',
        '', '| Family | Sessions | Base-fit N | Fit features | Train MAE | Validation MAE | MAE degradation | Train ROC-AUC | Validation ROC-AUC | Train Brier | Validation Brier |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for family in ['india_equity','us_equity']:
        for horizon in [1,5,10,20]:
            bundle=load_research_model(family,horizon,pointer);metadata=bundle['metadata']
            X,y,meta=build_panel(frames,manifest,horizon,family,references,settings)
            eligible=development_rows(meta,BOUNDARY).index
            X=X.loc[eligible].reset_index(drop=True);y=y.loc[eligible].reset_index(drop=True);meta=meta.loc[eligible].reset_index(drop=True)
            start=pd.Timestamp(metadata['blending_start'])
            base=np.flatnonzero((meta.feature_time<start)&(meta.label_end<start))
            validation=np.flatnonzero(meta.feature_time>=pd.Timestamp(metadata['validation_start']))
            names=list(bundle['regression'].feature_names);X=X[names]
            original=bundle['analogues']
            assert bundle['regression'].scaled(X.iloc[base]).shape==original['features'].shape
            assert np.allclose(bundle['regression'].scaled(X.iloc[base]),original['features'],rtol=1e-8,atol=1e-10)
            assert np.allclose(y.iloc[base],original['returns'],rtol=1e-10,atol=1e-12)
            train=metrics(bundle,X.iloc[base],y.iloc[base]);valid=metrics(bundle,X.iloc[validation],y.iloc[validation])
            for field in ['mae','rmse','brier','coverage_50','coverage_80']:
                assert np.isclose(valid[field],metadata['validation_metrics'][field],rtol=1e-8,atol=1e-10),(family,horizon,field)
            record={'base_n':len(base),'validation_n':len(validation),'feature_count':len(names),'parameters':metadata['parameters'],
                'training_metrics':train,'validation_metrics':valid,'base_feature_end':meta.iloc[base].feature_time.max().isoformat(),
                'base_label_end':meta.iloc[base].label_end.max().isoformat(),'validation_label_end':meta.iloc[validation].label_end.max().isoformat(),
                'original_fit_arrays_match':True,'original_validation_metrics_match':True,
                'independent_base_origin_dates':int(meta.iloc[base].feature_time.nunique()),
                'approx_nonoverlap_base_blocks':int(meta.iloc[base].feature_time.nunique()//horizon)}
            output['models'][family+'_'+str(horizon)]=record
            lines.append(f'| {family} | {horizon} | {len(base)} | {len(names)} | {train["mae"]:.6f} | {valid["mae"]:.6f} | {valid["mae"]/train["mae"]-1:+.1%} | {train["roc_auc"]:.3f} | {valid["roc_auc"]:.3f} | {train["brier"]:.4f} | {valid["brier"]:.4f} |')
            print(family,horizon,'native pre-boundary replay verified',flush=True)
    lines+=['','Depth3/120tree heads and many features fit only three stocks per family. The table measures in-sample optimism and later development degradation; it cannot identify a single causal failure mechanism. Correlated/overlapping labels reduce effective support below row counts.',
        '', 'The already-published failed V5 final gate remains historical context (1/0/2/7/3); its rows and report were not opened or rescored. The all100-member registered development study, rather than the consumed final result, governs the provisional next architecture. This addendum leaves original V5 and the first failure-analysis report unchanged.']
    directory=ROOT/'reports/post_v5'/output['diagnosis_id'];directory.mkdir(parents=True,exist_ok=False)
    with (directory/'NATIVE_TRAIN_DIAGNOSIS.json').open('x',encoding='utf-8') as stream:json.dump(output,stream,indent=2,allow_nan=False)
    text=chr(10).join(lines)+chr(10)
    with (directory/'NATIVE_TRAIN_DIAGNOSIS.md').open('x',encoding='utf-8') as stream:stream.write(text)
    with (ROOT/'reports/post_v5/V5_NATIVE_TRAIN_ADDENDUM.md').open('x',encoding='utf-8') as stream:stream.write(text)
    print('NATIVE_DIAGNOSIS',directory,flush=True)
if __name__=='__main__':main()
