"""Separate retrospective diagnostics from real prospective capture and shadow evidence."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from core.forecasting.ledger import PredictionLedger
from core.validation.metrics import forecast_metrics
from utils.ui_v5 import inject_theme,chart_layout
from research.post_v5.shadow import ROOT,DIRECTORY,STUDY
inject_theme()
st.title('Prospective Research')
st.caption('RESEARCH MODEL · Reconstructed historical studies and genuinely recorded prediction-time inputs are separate evidence streams')
st.warning('Production remains blocked. The completed V5 artifact and failed final result remain frozen. No successor final candidate has been trained.')
active_file=ROOT/'reports/post_v5/ACTIVE_DEVELOPMENT.json'
active=json.loads(active_file.read_text(encoding='utf-8')) if active_file.exists() else None
active_directory=(ROOT/active['directory']).resolve() if active else None
if active_directory and (ROOT/'reports/post_v5').resolve() not in active_directory.parents:raise ValueError('Invalid active study path')
protocol=json.loads((ROOT/'config/research'/STUDY/'PROTOCOL.json').read_text(encoding='utf-8'))
universe=json.loads((ROOT/'config/research'/STUDY/'UNIVERSE.json').read_text(encoding='utf-8'))
st.write('Fixed universe: '+str(universe['n_securities'])+' securities · '+str(len(universe['industries']))+' industries · Frozen '+universe['frozen_at'])
st.caption('Membership is retained through source failures, suspensions, delistings and index changes. This does not repair historical survivorship bias.')
tabs=st.tabs(['Live prospective study','Development comparisons','Targets & signal','Reports & protocol'])
with tabs[0]:
    captures=[]
    for file in sorted((ROOT/'reports/post_v5/captures').glob('*.json')):
        record=json.loads(file.read_text(encoding='utf-8'));captures.append(record)
    if captures:
        st.dataframe(pd.DataFrame([{'Run':r['run_id'],'Retrieved through':r['captured_at'],'Equities':r['captured_equities'],
            'Reference payloads':r['captured_references'],'Provider failures':len(r['failures'])} for r in captures]),hide_index=True,width='stretch')
        st.caption('A captured payload can still have invalid/missing prices or insufficient history. Payload capture does not establish forecast eligibility or historical publication certification.')
    else:st.info('No prospective inputs recorded yet.')
    ledger_file=DIRECTORY/'shadow_predictions.sqlite'
    all_records=PredictionLedger(ledger_file).rows() if ledger_file.exists() else []
    final_boundary=pd.Timestamp(protocol['final_origin_start'])
    from research.post_v5.reporting import visible_shadow_records
    visible,sealed=visible_shadow_records(all_records,final_boundary)
    st.caption('Future untouched origin window starts '+protocol['final_origin_start']+'. Its predictive outcomes stay sealed; only operational counts are shown. Sealed forecasts: '+str(sealed))
    members=[r for r in visible if r.get('frozen_universe_member')]
    primary={}
    for record in sorted(members,key=lambda r:r['generated_at']):
        key=(record['ticker'],record['origin_session'],record['horizon'],record['model_version'])
        primary.setdefault(key,record)
    open_records=list(primary.values())
    matured=[r for r in open_records if r['outcome'] and r['outcome']['status']=='matured']
    cols=st.columns(4)
    cols[0].metric('Recorded shadow forecasts',len(visible));cols[1].metric('Primary fixed-universe forecasts',len(open_records))
    cols[2].metric('Matured primary outcomes',len(matured));cols[3].metric('Abstention',f'{np.mean([r["abstain"] for r in open_records])*100:.0f}%' if open_records else 'No eligible forecasts')
    st.caption('Primary scoring uses the earliest issued forecast for each stock/origin/horizon/version; repeated requests remain recorded. Outcomes cannot be created until the horizon actually matures.')
    for file in sorted((ROOT/'reports/post_v5/shadow').glob('*.json')):
        batch=json.loads(file.read_text(encoding='utf-8'))
        with st.expander('Shadow batch '+batch['issued_at']+' · '+str(len(batch['successes']))+' success / '+str(len(batch['failures']))+' failed',expanded=not bool(visible)):
            st.dataframe(pd.DataFrame([{'Ticker':ticker,'Status':'forecast_failed','Reason':reason} for ticker,reason in batch['failures'].items()]),hide_index=True,width='stretch')
    if not visible:st.info('No eligible forecasts recorded. Capture attempts and source failures remain visible; no predictions or realized outcomes are fabricated.')
    else:
        history=pd.DataFrame([{'Ticker':r['ticker'],'Industry':r.get('industry_at_freeze'),'Cutoff':r['forecast_as_of'],'Origin':r['origin_session'],
            'Horizon':r['horizon'],'Model':r['model_version'],'Confidence':r['confidence'],'OOD':r['ood_score'],'Abstained':r['abstain'],
            'Status':r['outcome']['status'] if r['outcome'] else 'pending','Realized log return':r['outcome'].get('actual_log_return') if r['outcome'] else None,
            'Input run':r['input_run_id']} for r in visible])
        selected=st.multiselect('Live horizons',[1,5,10,20],default=[1,5,10,20])
        st.dataframe(history[history.Horizon.isin(selected)],hide_index=True,width='stretch')
        st.download_button('Export open warm-up shadow records',json.dumps(visible,indent=2),file_name='prospective_warmup_forecasts.json',mime='application/json')
        st.caption('Input existence at the prediction cutoff is archived. Frozen model training vintages and exchange calendars remain unverified. Outside-universe requests are listed but do not enter fixed-universe primary metrics.')
    if matured:
        observations=pd.DataFrame([{'ticker':r['ticker'],'industry':r['industry_at_freeze'],'horizon':r['horizon'],'regime':r['regime'],
            'confidence':r['confidence'],'asof':r['forecast_as_of'],'actual':r['outcome']['actual_log_return'],
            'predicted':r['central_log_return'],'probability':r['p_positive'],**dict(zip(['q10','q25','q50','q75','q90'],r['quantiles']))} for r in matured])
        selected_horizon=st.selectbox('Prospective metric horizon',[1,5,10,20])
        selected_rows=observations[observations.horizon==selected_horizon]
        if not selected_rows.empty:
            metrics=forecast_metrics(selected_rows.actual,selected_rows.predicted,selected_rows.probability,selected_rows[['q10','q25','q50','q75','q90']])
            st.json(metrics,expanded=False)
            for dimension in ['ticker','industry','regime','confidence']:
                summaries=[]
                for value,group in selected_rows.groupby(dimension):
                    scores=forecast_metrics(group.actual,group.predicted,group.probability,group[['q10','q25','q50','q75','q90']])
                    summaries.append({dimension:value,**{key:scores[key] for key in ['n','mae','rmse','brier','coverage_50','coverage_80']}})
                with st.expander('Prospective '+dimension+' performance'):st.dataframe(pd.DataFrame(summaries),width='stretch',hide_index=True)
            ordered=selected_rows.sort_values('asof').copy();ordered['Absolute log error']=(ordered.actual-ordered.predicted).abs()
            ordered['Rolling mean (63 recorded rows, overlapping/dependent)']=ordered['Absolute log error'].rolling(63,min_periods=10).mean()
            st.plotly_chart(chart_layout(px.line(ordered,x='asof',y='Rolling mean (63 recorded rows, overlapping/dependent)')),width='stretch')
with tabs[1]:
    studies=[p for p in (ROOT/'reports/post_v5').glob('development-*') if (p/'OUTPUT_HASHES.json').exists() and (p/'SUMMARY.json').exists() and not (p/'REVIEW_INVALIDATION.json').exists()]
    if not studies:st.info('Development experiments are in progress; no completed comparison is claimed.')
    else:
        study=st.selectbox('Completed development study',[p.name for p in sorted(studies)])
        directory=ROOT/'reports/post_v5'/study
        support=directory/'FOLD_SUPPORT.json'
        if support.exists():
            support_document=json.loads(support.read_text(encoding='utf-8'))
            support_rows=support_document.get('folds',support_document.get('rows',[])) if isinstance(support_document,dict) else support_document
            st.warning('Historical source gaps leave sparse stock/industry coverage in later quarters. No successor fit or new calibration is justified by pooled counts.')
            with st.expander('Actual fold and fixed-universe coverage',expanded=True):st.dataframe(pd.DataFrame(support_rows),hide_index=True,width='stretch')
        summary=pd.DataFrame(json.loads((directory/'SUMMARY.json').read_text(encoding='utf-8')))
        horizon=st.selectbox('Development horizon',[1,5,10,20])
        displayed=summary[summary.horizon==horizon].drop(columns=['date_rank_ic'],errors='ignore')
        st.dataframe(displayed,hide_index=True,width='stretch')
        audit=directory/'NUMERICAL_IC_AUDIT.json'
        if audit.exists():
            corrected=pd.DataFrame(json.loads(audit.read_text(encoding='utf-8'))['rows'])
            st.dataframe(corrected[corrected.horizon==horizon],hide_index=True,width='stretch')
            st.caption('Date-rank IC uses the additive tolerance audit; near-constant predictions and insufficient cross-sectional support are undefined, not signal. Original raw metrics remain archived.')
        else:st.info('Numerical IC audit pending. Raw date-rank summaries are withheld from this comparison.')
        st.caption('Reconstructed historical research only. Source/current-membership revisions and survivorship bias remain. Fold fits/selection exclude the consumed period; overlapping samples are dependent.')
        strata_file=directory/'STRATA.csv.gz'
        if strata_file.exists():
            strata=pd.read_csv(strata_file)
            dimension=st.selectbox('Development breakdown',['ticker','industry','fold','market_regime','trend_state'])
            eligible=strata[(strata.horizon==horizon)&(strata.stratum==dimension)]
            models=st.multiselect('Development models',sorted(eligible.model.unique()),default=[m for m in ['zero','ridge'] if m in eligible.model.unique()])
            st.dataframe(eligible[eligible.model.isin(models)],hide_index=True,width='stretch')
with tabs[2]:
    st.caption('Target-space and pooled IC values do not establish within-date stock ranking. The numerical audit flags near-constant ranks and sparse support; raw historical outputs remain immutable.')
    studies=[p for p in (ROOT/'reports/post_v5').glob('development-*') if (p/'OUTPUT_HASHES.json').exists() and not (p/'REVIEW_INVALIDATION.json').exists()]
    if studies:
        directory=sorted(studies)[-1]
        for name in ['TARGETS.json','PAIRED.json','ELIGIBILITY.json']:
            file=directory/name
            if file.exists():
                with st.expander(name.replace('.json','').replace('_',' ').title()):st.json(json.loads(file.read_text(encoding='utf-8')),expanded=False)
        audit=directory/'NUMERICAL_IC_AUDIT.json'
        if audit.exists():
            with st.expander('Numerical IC audit — raw values, corrected diagnostics and support',expanded=True):st.dataframe(pd.DataFrame(json.loads(audit.read_text(encoding='utf-8'))['rows']),hide_index=True,width='stretch')
        signals=directory/'SIGNALS.csv.gz'
        if signals.exists():
            frame=pd.read_csv(signals);st.dataframe(frame.head(300),hide_index=True,width='stretch')
            st.caption('Signal/conditional IC/stability diagnostics; view full file for complete results. Raw discrimination is assessed before any new calibration; no signal is created by calibration.')
    else:st.info('Target and feature diagnostics will appear after the registered development experiment completes.')
with tabs[3]:
    st.markdown((ROOT/'docs/NEXT_RESEARCH_TASK_BOARD.md').read_text(encoding='utf-8'))
    with st.expander('Registered protocol'):st.markdown((ROOT/'docs/NEXT_RESEARCH_PROTOCOL.md').read_text(encoding='utf-8'))
    if active_directory:
        st.write('Active source-audited development study: '+active['experiment_id'])
        for name in ['V5_FAILURE_ANALYSIS.md','BASELINE_COMPARISON.md','TARGET_FORMULATIONS.md','FEATURE_ABLATIONS.md','NEXT_CANDIDATE_ARCHITECTURE.md','CORRECTED_RESEARCH_DECISION.md','NUMERICAL_IC_ADDENDUM.md']:
            candidates=[active_directory/name,active_directory/'reports'/name]
            file=next((f for f in candidates if f.exists()),None)
            if file:
                with st.expander(name.replace('.md','').replace('_',' ').title()):st.markdown(file.read_text(encoding='utf-8'))
        st.caption('Any raw IC tables above must be interpreted with the displayed numerical audit; undefined near-constant ranks are not predictive evidence.')
    else:st.info('Corrected development study is not yet published. No invalidated study is presented as valid selection evidence.')
    for name,label in [('V5_NATIVE_TRAIN_ADDENDUM.md','Exact native V5 train/validation replay addendum'),('V5_CALIBRATION_ADDENDUM.md','Raw versus issued calibration-score addendum')]:
        file=ROOT/'reports/post_v5'/name
        if file.exists():
            with st.expander(label):st.markdown(file.read_text(encoding='utf-8'))
    invalidated=[f for f in (ROOT/'reports/post_v5').glob('development-*/REVIEW_INVALIDATION.json')]
    for file in invalidated:
        with st.expander('Invalidated study retained: '+file.parent.name):
            st.warning('INVALID FOR MODEL SELECTION: '+json.loads(file.read_text(encoding='utf-8'))['reason'])
            st.json(json.loads(file.read_text(encoding='utf-8')),expanded=False)
            for name in ['BASELINE_COMPARISON.md','TARGET_FORMULATIONS.md','FEATURE_ABLATIONS.md','NEXT_CANDIDATE_ARCHITECTURE.md','NUMERICAL_IC_ADDENDUM.md']:
                report=file.parent/name
                if report.exists():
                    with st.expander('Archived '+name):st.markdown(report.read_text(encoding='utf-8'))
    st.page_link('pages/validate.py',label='Preserved V5 evidence and failed final comparison')
