"""Fixed all-member study: immutable measured outputs, no final candidate API."""
import argparse
import gzip
import io
import json
import sys
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from research.post_v5.development import load_development, BOUNDARY
from research.post_v5.experiments import build_panel,evaluate_panel,summarize,exclusive_json,source_hash,regression_metrics


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.generic):value=value.item()
    if isinstance(value,float) and not np.isfinite(value):return None
    if isinstance(value,(pd.Timestamp,datetime,Path)):return str(value)
    return value


def write_csv(path,frame):
    with Path(path).open('xb') as raw:
        with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as gz:
            with io.TextIOWrapper(gz,encoding='utf-8',newline='') as f:frame.to_csv(f,index=False)


def table(records,columns):
    def fmt(v):
        if v is None:return '—'
        if isinstance(v,(float,np.floating)):return f'{v:.6f}'
        return str(v).replace('|','/')
    return '| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(r.get(c)) for c in columns)+' |' for r in records)


def frozen_development_diagnosis():
    directory=ROOT/'reports/validation/research-20261008T190859'
    p=pd.read_csv(directory/'predictions.csv')
    for col in ['feature_time','label_end']:
        if pd.to_datetime(p[col],utc=True).max()>=BOUNDARY:raise ValueError('Frozen development predictions enter consumed period; stop')
    data=json.loads((directory/'DEVELOPMENT.json').read_text())
    rows=[]
    for (family,horizon),f in p.groupby(['family','horizon']):
        truth=f.actual.to_numpy();pred=f.predicted.to_numpy();prob=f.probability.to_numpy();label=(truth>0).astype(float)
        m=regression_metrics(truth,pred);z=regression_metrics(truth,np.zeros(len(truth)))
        meta=data['models'].get(str(family)+'_'+str(horizon),{}).get('metadata',{})
        vm=meta.get('validation_metrics',{})
        rows.append({'family':family,'horizon':int(horizon),**m,'zero_mae':z['mae'],'improvement_zero':1-m['mae']/z['mae'],'brier':float(np.mean((prob-label)**2)),'coverage_50':float(np.mean((truth>=f.q25)&(truth<=f.q75))),'coverage_80':float(np.mean((truth>=f.q10)&(truth<=f.q90))),'training_stocks':len(meta.get('training_universe',[])),'features':len(meta.get('feature_names',[])),'train_metrics_available':bool(meta.get('train_metrics') or meta.get('training_metrics')),'calibration_accepted':meta.get('calibration_accepted'),'reported_classifier_balanced_accuracy':vm.get('classifier_balanced_accuracy')})
    return {'rows':rows,'date_min':str(pd.to_datetime(p.feature_time,utc=True).min()),'date_max':str(pd.to_datetime(p.label_end,utc=True).max()),'source_hashes':{str(x.relative_to(ROOT)):source_hash(x) for x in [directory/'DEVELOPMENT.json',directory/'predictions.csv']},'final_data_read':False,'native_train_replay_performed':False}


def reports(run,summary,paired,targets,signals,diagnosis):
    common=f"Experiment {run['experiment_id']}; input capture {run['input_run_id']}. Four registered dated folds and all 100 fixed members were attempted at 1/5/10/20 sessions. Feature and label endpoints strictly precede 2025-10-01. Source hashes, bounds, failures, fit audits, predictions and date-block summaries are in {run['experiment_id']}/. Reconstructed historical research: current membership and revised prices remain survivorship/revision biased. No historical PIT certification, production claim or final successor artifact.\n\n"
    baseline=[x for x in summary if x['model'] in ['zero','historical_mean','momentum20','reversion5','ridge','elastic_net','shallow_boost','market_only','market_sector']]
    detail=[x for x in paired if x['model'] in ['ridge','elastic_net','shallow_boost','market_only','market_sector']]
    candidates=[];headline=[]
    for h in [1,5,10,20]:
        simple=min([x for x in baseline if x['horizon']==h and x['model'] in ['zero','historical_mean','momentum20','reversion5'] and x['mae'] is not None],key=lambda x:x['mae'])
        candidates.append({'horizon':h,'strongest_simple':simple['model'],'simple_mae':simple['mae']})
        pp=[x for x in paired if x['horizon']==h and x['baseline']==simple['model'] and x['model'] not in ['zero','historical_mean','momentum20','reversion5']]
        strong=[x for x in pp if x['mae_improvement']>0 and x['folds_improved']>=3 and x['stocks_improved_fraction']>=.6]
        headline.append({'horizon':h,'strongest_simple':simple['model'],'complex_models_passing_fold_stock_screen':','.join(x['model'] for x in strong) or 'NONE'})
    baseline_text='# Development baseline comparison\n\n'+common+table(baseline,['horizon','model','n','mae','rmse','direction_accuracy','ic','dates','nonoverlap_blocks','date_rank_ic'])+'\n\nErrors are cumulative-log-return MAE/RMSE; direction is positive versus nonpositive. Historical mean is fixed once per purged fold. Momentum = ret20 * horizon/20; reversion = -ret5 * horizon/5. Ridge alpha10, elastic-net alpha.001/l1.5, shallow boosting 50iterations/depth2/minleaf100/lr.03/seed42/early_stopping=False. All median/scale fitting is train-only; no search.\n\n'+table(candidates,['horizon','strongest_simple','simple_mae'])+'\n\nPaired cohort comparisons (positive improvement favors model):\n\n'+table(detail,['horizon','model','baseline','n','mae_improvement','stocks_improved_fraction','folds_improved','nonoverlap_blocks','block_delta_mean','block_delta_se'])+'\n\n'+table(headline,['horizon','strongest_simple','complex_models_passing_fold_stock_screen'])+'\n\nScreen requires aggregate improvement, three improving folds and 60% of stocks. Sector/regime consistency and prospective evidence remain required. Prefer simpler within 1% MAE. Overlapping labels and common shocks make pooled rows dependent: horizon-session date blocks are primary support. Block SE is descriptive dispersion, not a calibrated confidence interval. Full stock/industry/fold/regime/trend results are in STRATA.csv.gz. No trading backtest.\n'
    target_ret=[x for x in summary if x['model'] in ['ridge','vol_normalized','market_excess','sector_excess','decomposition']]
    target_text='# Target formulation comparison\n\n'+common+table(targets,['horizon','fold','target','metric_space','n','mae','rmse','ic','train_target_mae','roc_auc','balanced_accuracy','brier','train_rate_brier','train_roc_auc'])+'\n\nVolatility units, excess-log-return error and same-origin percentile errors are separate metric spaces, never return MAE. Normalize using origin-only daily20 volatility * sqrt(horizon); convert predictions back with this ex-ante scale. Rank target uses eligible stocks of the same origin only; its score is not automatically clipped or reconstructed into returns. Positive direction = expm1(log return) > .003 round-trip cost. Classifier scores are RAW and UNCALIBRATED.\n\nReconstructed return-space results:\n\n'+table(target_ret,['horizon','model','n','mae','rmse','ic','date_rank_ic'])+'\n\nExcess reconstruction adds independently predicted market and sector-minus-market components. Realized future references enter training/validation LABELS ONLY, never predictors or validation-time reconstruction. Market head uses one sample/date; sector-excess head uses train-only industry interactions. Three-head decomposition adds market + sector excess + stock alpha from past features on identical purged folds. Keep nothing automatically; paired negative results remain recorded. No new calibration was fit: held-out ROC-AUC .55 and balanced accuracy .52 with stable date ranking must first be established. Coin-flip Brier=.25 and training-rate Brier are explicit baselines.\n'
    feature_rows=[x for x in summary if x['model'].endswith('_only') or x['model'].startswith('without_') or x['model'] in ['ridge','reduced_stable','sector_specific','sector_interactions','sector_heldout','decomposition']]
    sg=pd.DataFrame(signals);aggregate=sg.groupby(['horizon','feature'])[['train_ic','validation_ic','train_residual_ic','validation_residual_ic']].mean().reset_index();aggregate['degradation']=aggregate.train_ic.abs()-aggregate.validation_ic.abs()
    top=aggregate.sort_values(['horizon','validation_ic'],ascending=[True,False]).groupby('horizon').head(8).to_dict('records')
    degradation=[{'horizon':x['horizon'],'fold':x['fold'],'model':x['model'],'train_mae':x['train_metrics']['mae'],'validation_mae':x['validation_metrics']['mae'],'ratio':x['validation_metrics']['mae']/x['train_metrics']['mae']} for x in run['attempts'] if x.get('train_metrics') and x['model'] in ['ridge','elastic_net','shallow_boost','reduced_stable','sector_interactions']]
    unsupported=[x for x in run['sector_support'] if not x['supported']]
    fp=[x for x in paired if x['baseline']=='ridge_paired_supported_industries']
    feature_text='# Feature, breadth and stability analysis\n\n'+common+table(feature_rows,['horizon','model','n','mae','rmse','ic','date_rank_ic','nonoverlap_blocks'])+'\n\nGroups: 56 technical stock ratios; domestic market; sector; global equity/volatility/yield; FX USD/INR/DXY; commodities Brent/WTI/gold. Group-only and removed-group ridge share origins. Train-only median imputation excludes all-missing/constant columns (FIT_AUDITS). Context availability is estimated following UTC day, not PIT-certified. For absent official sectors use equal-weight fixed-universe industry DAILY RETURNS excluding forecast stock with minimum2 observed peers. This is NOT an official sector index; survivorship bias remains. Failed/sparse study members remain in ELIGIBILITY.\n\nReduced train-stable set: train variance filter, greedy absolute training Pearson correlation >.95 pruning; consistent Spearman sign in chronological training halves, minimum half absolute IC .01; cap24 by training strength, fall back first5 decorrelated features if none. No validation selection. Train-only identity vocabularies; sector-specific support >=2000 train rows/252 dates/2stocks.\n\nPaired supported-sector versus pooled:\n\n'+table(fp,['horizon','model','baseline','n','mae_improvement'])+'\n\nUnsupported industries/folds:\n\n'+table(unsupported,['horizon','fold','industry','train_n','train_dates','train_stocks','test_n','supported'])+'\n\nLeave-industry-out evaluates every industry with all of its training stocks excluded. It supplements temporal checks; it is not prospective evidence.\n\nMeasured train/validation degradation:\n\n'+table(degradation,['horizon','fold','model','train_mae','validation_mae','ratio'])+'\n\nExploratory validation IC leaders (never used to reselect):\n\n'+table(top,['horizon','feature','train_ic','validation_ic','train_residual_ic','validation_residual_ic','degradation'])+'\n\nResidual IC uses train-fitted market-feature ridge residualization of both signal and target. Full IC in SIGNALS; sample/date/features in ATTEMPTS/FIT_AUDITS; strata in STRATA. Regime rules: past market trend +/-2% bull/bear/range, past rolling volatility median high/low, |past stock ret20|>2% trend/range. Stocks/industries/folds/horizons/regimes reported separately. Complexity sensitivity is fixed linear versus depth2/50 boosting and full versus reduced features; no depth search. Correlated stocks do not multiply effective dates.\n'
    diagnosis_text='# V5 failure analysis\n\n'+common+'Only frozen DEVELOPMENT.json and its predictions.csv were inspected after verifying every feature/label date preboundary. Native V5 validation (2025-07 onward) is not directly paired with this broader four-fold study. No V5 final report or final CSV was opened/rescored.\n\n'+table(diagnosis['rows'],['family','horizon','n','mae','zero_mae','improvement_zero','ic','direction_accuracy','brier','coverage_50','coverage_80','training_stocks','features','train_metrics_available','calibration_accepted'])+'\n\nMeasured: three stocks per exchange family, many correlated inputs relative to independent dates, horizon-varying IC/direction/error/interval coverage and calibration acceptance. Actual native V5 in-sample train metrics are absent; no native train replay was performed. V5 train/validation overfit magnitude is therefore UNMEASURED. The new study measures its own gaps, not V5 train errors.\n\nPlausible hypotheses, not causal findings: correlated indicators encourage unstable partitions/coefficients; three-stock sector coverage confuses common exposure with transferable alpha; revised histories/current membership inflate replay confidence; overlapping labels/small date-block support weaken calibration evidence. The new measured ablations, sector holdouts, target experiments and gaps test these directions without consumed final outcomes. Historical final gate facts are restricted to published 1/0/2/7/3; they did not select this architecture. No final degradation scalar was required.\n'
    architecture_text='# Provisional next architecture\n\n'+common+table(headline,['horizon','strongest_simple','complex_models_passing_fold_stock_screen'])+'\n\nUse the strongest simple development baseline per horizon as reference. No complexity expansion or final successor fit follows an aggregate average. Require three improving folds, 60% stocks, supported sector/regime consistency and prospective evidence; choose simpler within1% MAE.\n\nProvisional research architecture: immutable capture -> session-aligned source-health-aware features -> horizon-specific simple baseline plus train-stable pooled linear challenger -> raw direction/rank diagnostics -> abstaining output. Keep independent market/sector/alpha decomposition as a diagnostic unless paired evidence consistently beats the strongest baseline. Sector-specific/interaction models remain experimental and unsupported groups explicit. Normalized, excess and rank targets must earn their own measured benefit; do not infer absolute accuracy from normalized/rank IC. Features require bounded registered evidence and cannot be chosen from consumed final results.\n\nCalibration deferred until held-out ROC-AUC>=.55/balanced accuracy>=.52 and stable date-level ranking. Existing abstention safeguards retained. No new calibration, threshold, interval, production artifact or BUY/SELL output. NLP/news deferred.\n\nProspective development continues across fixed100 including failed/delisted members. Freeze by2027-10-10; untouched origin window2027-10-11–2028-10-10; score only after20-session maturity and not before2028-11-10. Protocol source/session/sector/block counts and future gates remain binding. All eight immediate deliverables must be reviewed before any successor fit.\n'
    mapping={'BASELINE_COMPARISON.md':baseline_text,'TARGET_FORMULATIONS.md':target_text,'FEATURE_ABLATIONS.md':feature_text,'V5_FAILURE_ANALYSIS.md':diagnosis_text,'NEXT_CANDIDATE_ARCHITECTURE.md':architecture_text}
    for name,text in mapping.items():
        with (ROOT/'reports/post_v5'/run['experiment_id']/name).open('x',encoding='utf-8') as f:f.write(text)
        canonical=ROOT/'reports/post_v5'/name
        if not canonical.exists():
            with canonical.open('x',encoding='utf-8') as f:f.write(text)
    return headline


def run_study(run_id):
    start=datetime.now(timezone.utc);identifier='development-'+start.strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8]
    directory=ROOT/'reports/post_v5'/identifier;directory.mkdir(parents=True,exist_ok=False)
    manifest={'experiment_id':identifier,'input_run_id':run_id,'started_at':start.isoformat(),'scope':'reconstructed_historical_development_only','final_data_used':False,'candidate_artifacts_saved':False,'pre_transform_boundary':'2025-10-01T00:00:00Z','max_threads':2,'source_hashes':{str(Path(x).relative_to(ROOT)):source_hash(x) for x in [ROOT/'research/post_v5/experiments.py',ROOT/'research/post_v5/development.py',ROOT/'scripts/post_v5_research.py',ROOT/'config/research/prospective-india-20261008-v1/PROTOCOL.json',ROOT/'config/research/prospective-india-20261008-v1/UNIVERSE.json',ROOT/'config/context_sources.json']}}
    capture_path=ROOT/'data/research/prospective-india-20261008-v1/runs'/(run_id+'.json')
    manifest['source_hashes'][str(capture_path.relative_to(ROOT))]=source_hash(capture_path)
    manifest['snapshot_ids']=json.loads(capture_path.read_text())['snapshots']
    for extra in ['core/indicators.py','core/forecasting/context.py','core/validation/alignment.py','core/data/contracts.py']:
        manifest['source_hashes'][extra]=source_hash(ROOT/extra)
    import sklearn
    manifest['versions']={'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__}
    exclusive_json(directory/'REGISTRATION.json',manifest)
    frames,refs,settings,universe,protocol=load_development(run_id)
    manifest['protocol']=protocol;manifest['universe_members']=len(universe['members']);manifest['captured_frames']=len(frames);manifest['captured_references']=len(refs)
    attempts=[];audits=[];signals=[];targets=[];eligibility=[];support=[];summaries=[];strata=[];pairs=[];bounds=[];warnings_seen=[]
    try:
        for horizon in protocol['horizons']:
            print('Building full fixed-universe panel horizon '+str(horizon),flush=True)
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter('always')
                panel,eligible=build_panel(frames,refs,settings,universe,horizon)
                bound={'horizon':horizon,'panel_rows':len(panel),'stocks':panel.ticker.nunique(),'industries':panel.industry.nunique(),'feature_min':str(pd.to_datetime(panel.feature_time,utc=True).min()),'feature_max':str(pd.to_datetime(panel.feature_time,utc=True).max()),'label_max':str(pd.to_datetime(panel.label_end,utc=True).max()),'feature_groups':panel.attrs['feature_groups']}
                bounds.append(bound);print(json.dumps(clean(bound)),flush=True)
                def log_attempt(attempt):
                    filename='H'+str(horizon)+'_F'+str(attempt['fold'])+'_'+attempt['model']+'_'+attempt['status']+'.json'
                    exclusive_json(directory/'attempts'/filename,clean(attempt))
                out=evaluate_panel(panel,protocol['folds'],horizon,progress_callback=log_attempt)
                summary,st,pair=summarize(out['predictions'])
                write_csv(directory/('PREDICTIONS_H'+str(horizon)+'.csv.gz'),out['predictions'])
                exclusive_json(directory/('HORIZON_'+str(horizon)+'.json'),clean({'bounds':bound,'eligibility':eligible,'attempts':out['attempts'],'audits':out['audits'],'target_scores':out['target_scores'],'sector_support':out['sector_support'],'summary':summary,'paired':pair}))
                warnings_seen.extend({'horizon':horizon,'category':str(w.category.__name__),'message':str(w.message)} for w in recorded)
            attempts+=out['attempts'];audits+=out['audits'];signals+=out['signals'];targets+=out['target_scores'];support+=out['sector_support'];summaries+=summary;strata+=st;pairs+=pair;eligibility.extend({'horizon':horizon,**x} for x in eligible)
            print(json.dumps({'horizon':horizon,'status':'persisted','summary':summary}),flush=True)
            del panel,out
        diagnosis=frozen_development_diagnosis()
        manifest.update({'completed_at':datetime.now(timezone.utc).isoformat(),'date_bounds':bounds,'attempts':attempts,'sector_support':support,'warnings':warnings_seen,'status':'complete'})
        for name,payload in [('RESULTS.json',manifest),('ATTEMPTS.json',attempts),('FIT_AUDITS.json',audits),('ELIGIBILITY.json',eligibility),('SUMMARY.json',summaries),('PAIRED.json',pairs),('TARGETS.json',targets),('V5_DEVELOPMENT_DIAGNOSIS.json',diagnosis)]:exclusive_json(directory/name,clean(payload))
        for name,records in [('STRATA.csv.gz',strata),('SIGNALS.csv.gz',signals)]:write_csv(directory/name,pd.DataFrame(records))
        headline=reports(manifest,summaries,pairs,targets,signals,diagnosis)
        outputs={p.name:source_hash(p) for p in directory.iterdir() if p.is_file()}
        exclusive_json(directory/'OUTPUT_HASHES.json',outputs)
        print(json.dumps({'experiment_id':identifier,'directory':str(directory),'status':'complete','headline':headline,'attempts':len(attempts),'failed_attempts':sum(a['status']!='complete' for a in attempts)}),flush=True)
    except BaseException as exc:
        exclusive_json(directory/'FAILED_ATTEMPT.json',{'status':'stopped','type':type(exc).__name__,'reason':str(exc),'time':datetime.now(timezone.utc).isoformat(),'finished_horizons':[b['horizon'] for b in bounds],'final_data_used':False})
        raise
    return directory

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);args=parser.parse_args();run_study(args.run_id)
