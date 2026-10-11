"""Render documentation from the consumed final result; never trains or reevaluates."""
import json
from pathlib import Path
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from core.validation.metrics import forecast_metrics

def pct(value): return 'Unavailable' if value is None else f'{100*value:.2f}%'
def num(value): return 'Unavailable' if value is None else f'{value:.4f}'
def render():
    report=json.loads((ROOT/'reports/validation/FINAL_COMPARISON.json').read_text(encoding='utf-8'))
    rows=pd.read_csv(ROOT/'reports/validation/FINAL_COMPARISON.csv')
    results=report['results']
    lines=['# V4 versus V5: frozen research comparison','',
        '**Result: the frozen research performance gate FAILED. Production promotion remains BLOCKED.**','',
        'Candidate: '+report['model_version']+'. The final period began '+report['final_test_start']+' UTC and includes only labels matured by '+report['capture_cutoff']+'. The protocol was locked before evaluation and consumed once. No model decisions were changed after opening it.','',
        'This is a retrospective model experiment on revised Yahoo snapshots, six current equities and reconstructed availability/session cutoffs. It is not certified point-in-time evidence or a replay of historical production predictions. Models were created in 2026 with all fitting, blending, calibration and validation labels strictly before the final boundary. Overlapping horizons and correlated stocks make observation counts dependent.','',
        '## Frozen decision gates','',
        '| Gate | Observed groups | Required | Result |','|---|---:|---:|---|']
    names={'mae_improvements':('MAE better than zero-return',6),'rmse_improvements':('RMSE better than zero-return',6),
        'brier_below_coinflip':('Brier below 0.25',6),'coverage_80_within_10pp':('80% coverage between 70% and 90%',6),
        'calibrated_heads':('Calibration accepted on development validation',8)}
    for key,(name,required) in names.items():
        count=report['frozen_research_gate_counts'][key]
        lines.append(f'| {name} | {count}/8 | {required}/8 | {"PASS" if count>=required else "FAIL"} |')
    lines+=['','## Full-grid return errors','',
        'MAE/RMSE below are in percentage points of cumulative log return, not dollar/rupee price error. V4 refit uses the original 600-tree XGB/LGB architecture and pooled pre-boundary labels (5,580 observations), with training-fitted scaling and equal ensemble weights. Its one-day estimate is compounded as an architecture baseline. The originally shipped synthetic checkpoint is preserved and reported separately in V4_BASELINE.md; no historical training cutoff was available for that checkpoint.','',
        '| Family | Sessions | N | V5 MAE | Zero MAE | Drift MAE | V4 compound MAE | V5 RMSE | Zero RMSE | V4 compound RMSE |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for family in ['india_equity','us_equity']:
        for horizon in [1,5,10,20]:
            prefix=f'{family}_{horizon}_'
            v=results[prefix+'v5_direct']['aggregate']; z=results[prefix+'zero_return']['aggregate']
            d=results[prefix+'historical_drift']['aggregate']; b=results[prefix+'v4_refit_compounding']['aggregate']
            lines.append(f'| {family} | {horizon} | {v["n"]} | {pct(v["mae"])} | {pct(z["mae"])} | {pct(d["mae"])} | {pct(b["mae"])} | {pct(v["rmse"])} | {pct(z["rmse"])} | {pct(b["rmse"])} |')
    lines+=['','## Direction, calibration and uncertainty','',
        'Return direction tests the sign of the central regressor; classifier direction tests P(up) at 0.5. Brier/log loss/ECE use the accepted calibrated score where available, otherwise the explicitly raw classifier score. Acceptance was decided on development validation, not this period. Development acceptance does not establish calibration under later distribution shift. V4 does not emit calibrated classifier probabilities; its Brier/log loss are unavailable.','',
        '| Family | Sessions | Calibration | Return direction | Classifier direction | Brier | Log loss | ECE | 50% coverage | 80% coverage | Mean 80% width | Rank IC |','|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for key,count in report['sample_counts'].items():
        v=results[key+'_v5_direct']['aggregate']; family,horizon=key.rsplit('_',1)
        state='Accepted' if count['calibration_accepted'] else 'Rejected; raw'
        lines.append(f'| {family} | {horizon} | {state} | {pct(v["directional_accuracy"])} | {pct(v["classifier_directional_accuracy"])} | {num(v["brier"])} | {num(v["log_loss"])} | {num(v["expected_calibration_error"])} | {pct(v["coverage_50"])} | {pct(v["coverage_80"])} | {pct(v["width_80"])} | {num(v["spearman_ic"])} |')
    lines+=['','India 10/20-session direction and probability scores deteriorated substantially. Seven groups meet the broad 80% coverage tolerance, while several 50% intervals undercover; good coverage alone does not demonstrate useful forecast centers. Native five-quantile heads provide 50% and 80% bands; 90% coverage is unavailable. Pinball losses, Pearson IC, bias, class precision/recall and bucket sample counts are retained in the machine report.','',
        '## Matched recursive V4 comparison','',
        'The actual legacy recursive simulation is run on a fixed, evenly spaced sample of up to 12 origins per stock from the 20-session cohort, using 30 paths and recorded deterministic seeds. V5 below is restricted to exactly those origins. Simulation intervals describe this legacy generator, not calibrated probabilities. Small, dependent samples cannot establish stable superiority.','',
        '| Family | Sessions | Matched N | V4 recursive MAE | V5 matched MAE | V4 recursive RMSE | V5 matched RMSE | V4 direction | V5 direction | V4 simulation 80% coverage | V5 matched 80% coverage |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for family in ['india_equity','us_equity']:
        for horizon in [1,5,10,20]:
            b=rows[(rows.family==family)&(rows.horizon==horizon)&(rows.method=='v4_refit_recursive')]
            v=rows[(rows.family==family)&(rows.horizon==horizon)&(rows.method=='v5_direct')].merge(b[['ticker','origin_session']],on=['ticker','origin_session'],validate='one_to_one')
            vm=forecast_metrics(v.actual,v.predicted,v.probability,v[['q10','q25','q50','q75','q90']])
            bm=results[f'{family}_{horizon}_v4_refit_recursive']['aggregate']
            lines.append(f'| {family} | {horizon} | {len(v)} | {pct(bm["mae"])} | {pct(vm["mae"])} | {pct(bm["rmse"])} | {pct(vm["rmse"])} | {pct(bm["directional_accuracy"])} | {pct(vm["directional_accuracy"])} | {pct(bm["simulation_coverage_80"])} | {pct(vm["coverage_80"])} |')
    lines+=['','## Relative-return heads','',
        'Targets here are stock-minus-reference cumulative log returns. Sector availability is limited by the current mappings and source history; India Energy history was insufficient. Auxiliary calibration is independently governed. A high directional percentage may reflect an imbalanced label distribution; balanced accuracy is also reported.','',
        '| Family / sessions | Reference | N | MAE | RMSE | Return direction | Classifier direction | Classifier balanced accuracy | Brier | Rank IC |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for key,tasks in report['relative_metrics'].items():
        for task,v in tasks.items():
            lines.append(f'| {key} | {task} | {v["n"]} | {pct(v["mae"])} | {pct(v["rmse"])} | {pct(v["directional_accuracy"])} | {pct(v["classifier_directional_accuracy"])} | {pct(v["classifier_balanced_accuracy"])} | {num(v["brier"])} | {num(v["spearman_ic"])} |')
    for dimension,title in [('sector','Sector'),('market_regime','Market regime'),('year','Year'),('ticker','Stock')]:
        lines+=['','## '+title+' stability','',
            '| Family / sessions | Group | N | V5 MAE | Zero MAE | V4 compound MAE | V5 direction | V5 Brier | V5 80% coverage |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for key in report['sample_counts']:
            for group,v in results[key+'_v5_direct']['breakdowns'][dimension].items():
                z=results[key+'_zero_return']['breakdowns'][dimension][group]
                b=results[key+'_v4_refit_compounding']['breakdowns'][dimension][group]
                lines.append(f'| {key} | {group} | {v["n"]} | {pct(v["mae"])} | {pct(z["mae"])} | {pct(b["mae"])} | {pct(v["directional_accuracy"])} | {num(v["brier"])} | {pct(v["coverage_80"])} |')
    lines+=['','## Coverage, limitations and use','',
        'Model estimates are available for the eligible rows listed above. Actionable prediction coverage is 0%; research abstention rate is 100%. All research forecasts are LOW confidence. Twenty India row/horizon observations exceed the frozen OOD threshold; none do in the US cohort. That is a distance diagnostic, not proof of domain coverage. No live production track record exists.','',
        'The evidence supports software experimentation and auditing, not relying on these forecasts for trading or real-world financial decisions. The balanced improvement criterion is unmet. News/NLP expansion remains gated under spec §10: the core must first beat baselines. PIT certification, production promotion and browser visual/accessibility QA remain outstanding. Future experiments require a new untouched holdout; this period must not be reused for tuning.','',
        'Evidence: [machine comparison](validation/FINAL_COMPARISON.json), [row-level predictions](validation/FINAL_COMPARISON.csv), [protocol lock](validation/FINAL_PROTOCOL_LOCK.json), [consumed marker](validation/FINAL_TEST_CONSUMED.json), [original V4 diagnostic](baseline/V4_BASELINE.md).']
    (ROOT/'reports/V4_VS_V5.md').write_text(chr(10).join(lines)+chr(10),encoding='utf-8')
    print('Rendered reports/V4_VS_V5.md from consumed results; no fitting/evaluation performed')
if __name__=='__main__': render()
