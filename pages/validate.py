"""Trust center: complete development/final evidence, quarantine and immutable history."""
import json
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from core.forecasting.service import ROOT,active_research,load_research_model
from core.forecasting.ledger import PredictionLedger
from utils.ui_v5 import inject_theme,percent,chart_layout

inject_theme()
st.title('Validate')
st.caption('Forecast-model evidence, calibration, interval coverage and the complete research record')
st.warning('Research evidence is not point-in-time certification. Production promotion remains blocked.')
try:
    pointer=active_research()
    report=json.loads((ROOT/pointer['report']/'DEVELOPMENT.json').read_text(encoding='utf-8'))
except Exception as error:
    st.error('Validation evidence unavailable: '+str(error)); st.stop()

rows=[]
for key,model in report['models'].items():
    metadata=model['metadata']; metrics=metadata['validation_metrics']
    rows.append({'Model':key,'Return MAE':metrics['mae'],'RMSE':metrics['rmse'],
        'Direction hit rate':percent(metrics['directional_accuracy'],False),'Brier':metrics['brier'],
        '80% coverage':percent(metrics['coverage_80'],False),'80% width':percent(metrics['width_80'],False),
        'Calibration':'Accepted' if metadata['calibration_accepted'] else 'Rejected; raw score',
        'N':metadata['validation_n'],'Groups':', '.join(metadata['ablation']['kept_groups'])})
st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
st.caption('Development selection/evidence metrics. Horizon labels overlap and assets are correlated; counts are not independent trials. Model selection and calibration rejection occurred before the final test.')

key=st.selectbox('Inspect model',list(report['models']))
model=report['models'][key]; metadata=model['metadata']; metrics=metadata['validation_metrics']
tabs=st.tabs(['Probability reliability','Intervals & regimes','Model card & ablations','Prediction ledger','Final comparison','Quarantined candidates'])
with tabs[0]:
    bins=[row for row in metrics['calibration'] if row['n']>0]
    if bins:
        figure=go.Figure()
        figure.add_trace(go.Scatter(x=[r['predicted'] for r in bins],y=[r['observed'] for r in bins],
            mode='lines+markers',name='Observed probability',text=[str(r['n'])+' observations' for r in bins]))
        figure.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',line_dash='dot',name='Ideal reference'))
        figure.update_xaxes(title='Predicted probability'); figure.update_yaxes(title='Observed positive-return rate')
        st.plotly_chart(chart_layout(figure),use_container_width=True)
        st.dataframe(pd.DataFrame(bins),hide_index=True,use_container_width=True)
    st.write('Calibration state: '+('Accepted on disjoint development validation' if metadata['calibration_accepted'] else 'Rejected; probabilities remain uncalibrated'))
    st.caption('Raw probabilities are shown explicitly when calibration worsens Brier/log loss. No calibration claim is attached to those heads.')
with tabs[1]:
    st.write('50% coverage: '+percent(metrics['coverage_50'],False)+' · 80% coverage: '+percent(metrics['coverage_80'],False))
    st.write('80% mean width: '+percent(metrics['width_80'],False))
    st.caption('q10/q90 define 80%, not 90%. No 90% interval is manufactured. OOD expansion at live inference has separate, unvalidated coverage.')
    for dimension in ['market_regime','sector','ticker','year']:
        with st.expander(dimension.replace('_',' ').title()+' breakdown'):
            st.dataframe(pd.DataFrame(model['breakdowns'].get(dimension,{})).T,use_container_width=True)
with tabs[2]:
    st.write('Model version: '+metadata['model_version'])
    st.write('Training universe: '+', '.join(metadata['training_universe']))
    st.write('Training label cutoff: '+metadata['training_cutoff'])
    st.write('Blending cutoff: '+metadata.get('blending_cutoff','Unavailable'))
    st.write('Calibration cutoff: '+metadata['calibration_cutoff'])
    st.write('Validation cutoff: '+metadata['validation_cutoff'])
    st.json(metadata['ablation'],expanded=True)
    st.json(metadata['relative_evidence'],expanded=False)
    st.json(metadata['source_provenance'],expanded=False)
with tabs[3]:
    ledger=PredictionLedger(ROOT/'data/v5/predictions.sqlite')
    records=ledger.rows()
    if not records: st.info('No recorded research forecasts yet. Explicit Forecast requests append immutable records here.')
    else:
        frame=pd.DataFrame([{'ID':r['prediction_id'],'Ticker':r['ticker'],'As of':r['forecast_as_of'],'Horizon':r['horizon'],
            'Central return':percent(__import__('numpy').expm1(r['central_log_return'])),'Confidence':r['confidence'],
            'Abstained':r['abstain'],'Model':r['model_version'],'Status':r['outcome']['status'] if r['outcome'] else 'pending',
            'Realized return':percent(r['outcome'].get('actual_return')) if r['outcome'] else 'Pending'} for r in records])
        st.dataframe(frame,hide_index=True,use_container_width=True)
        st.download_button('Export full prediction ledger',json.dumps(records,indent=2),file_name='research_predictions.json',mime='application/json')
    st.caption('Forecasts and outcomes are append-only. Failed and abstained predictions remain visible. Outcome resolution is an explicit job using an independent session calendar.')
with tabs[4]:
    file=ROOT/'reports/V4_VS_V5.md'
    if file.exists(): st.markdown(file.read_text(encoding='utf-8'))
    else: st.info('Final test remains sealed from 2025-10-01. A comparison will be published after all modeling decisions are frozen.')
with tabs[5]:
    file=ROOT/'models/v5/QUARANTINE.json'
    if file.exists(): st.dataframe(pd.DataFrame(json.loads(file.read_text(encoding='utf-8'))['versions']),hide_index=True,use_container_width=True)
    st.caption('Quarantined artifacts are retained for audit and cannot become the active research candidate silently.')
