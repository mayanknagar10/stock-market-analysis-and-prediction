"""V5 forecast workspace: research distributions, provenance, evidence and scenarios."""
from datetime import datetime,timezone
import pandas as pd
import streamlit as st
from core.data.providers import equity_family
from core.data.runtime import market_snapshot
from core.forecasting.service import active_research,load_research_model,load_reference_snapshots,research_forecast
from utils.ui_v5 import inject_theme,percent,price,distribution_chart,chart_layout
import plotly.graph_objects as go

inject_theme()
st.title('Forecast')
st.caption('Direct horizon-specific forecasts · India and US equities · Research only')
st.warning('Point-in-time certification and production promotion are blocked. Research forecasts abstain from actionable confidence.')

@st.cache_resource
def models(version,paths,family):
    pointer={'model_version':version,'models':paths}
    return {h:load_research_model(family,h,pointer) for h in [1,5,10,20]}

@st.cache_data(ttl=300,show_spinner=False)
def capture(ticker): return market_snapshot(ticker)

try:
    pointer=active_research()
except Exception as error:
    st.error(str(error)); st.code('python scripts/train_v5.py --manifest PATH_TO_SNAPSHOT_MANIFEST'); st.stop()

with st.form('v5_forecast'):
    left,right=st.columns([3,1])
    ticker=left.text_input('Equity ticker',value='AAPL',help='India: .NS or .BO · US: plain equity symbol').strip().upper()
    right.caption('Models load from saved research checkpoints. Ordinary requests never train a model.')
    run=st.form_submit_button('Generate research forecast',type='primary')

if run:
    try:
        family=equity_family(ticker)
        with st.spinner('Validating source snapshots and computing research evidence…'):
            references,reference_manifest=load_reference_snapshots()
            data=research_forecast(capture(ticker),models(pointer['model_version'],pointer['models'],family),references)
        st.session_state['v5_forecast']=data
    except Exception as error:
        st.error('Forecast unavailable: '+str(error)); st.stop()

data=st.session_state.get('v5_forecast')
if data is None:
    st.info('Choose an equity to inspect direct 1/5/10/20-session estimates, uncertainty and evidence. Each explicit request is recorded in the research ledger.')
    st.page_link('pages/validate.py',label='Inspect validation evidence before forecasting')
    st.stop()

symbol='₹' if data['ticker'].endswith(('.NS','.BO')) else '$'
st.subheader(data['ticker']+' · '+price(data['current_price'],symbol))
st.caption('Last completed session: '+data['origin_session']+' · Request cutoff: '+data['as_of']+' · Model: '+data['forecasts'][0]['record']['model_version'])
horizon=st.segmented_control('Evidence horizon',[1,5,10,20],default=5,format_func=lambda h:str(h)+'D') or 5
item=next(i for i in data['forecasts'] if i['forecast']['horizon']==horizon)
forecast=item['forecast']; metadata=item['metadata']; record=item['record']
calibration_state='Accepted on disjoint development validation' if forecast['p_positive_calibrated'] is not None else 'Rejected; uncalibrated raw score only'
st.warning(forecast['trust']['status']+' · '+', '.join(forecast['trust']['reasons']))
st.write('**Data & model health:** Price '+data['health']['price_data_status']+' · Historical source vintages UNVERIFIED · Calendar UNVERIFIED research proxy · OOD ratio '+f'{forecast["ood_score"]:.2f}')
st.write('**Calibration:** '+calibration_state+' · **Confidence:** '+forecast['trust']['confidence'])
stale=[key for key,status in data['context_health'].get('source_statuses',{}).items() if status=='STALE']
if stale: st.warning('Stale captured context sources: '+', '.join(stale)+'. Source freshness limits this research estimate.')
if forecast['ood_interval_expanded']: st.warning('Current features are outside the training domain. Intervals are conservatively expanded; that expansion has no validated coverage guarantee.')
summary=st.columns(3)
summary[0].metric('Central return',percent(forecast['central_return']))
summary[1].metric('Implied price',price(forecast['central_price'],symbol))
summary[2].metric('Calibrated P(positive)',percent(forecast['p_positive_calibrated'],False))
bands=st.columns(2)
bands[0].write('**50% interval:** '+price(forecast['quantile_prices'][1],symbol)+' – '+price(forecast['quantile_prices'][3],symbol))
bands[1].write('**80% interval:** '+price(forecast['quantile_prices'][0],symbol)+' – '+price(forecast['quantile_prices'][4],symbol))
if forecast['p_positive_calibrated'] is None: st.caption('Raw classifier score: '+percent(forecast['raw_p_positive'],False)+'; this is not presented as a calibrated probability.')
st.caption('Central prices convert log returns; future dividends/splits are unknown. Intervals use disjoint conformal adjustment and may expand under OOD. Development coverage is pre-expansion empirical evidence, not a guarantee.')
st.plotly_chart(distribution_chart(forecast,data['current_price']),use_container_width=True)
st.caption('Development 80% coverage: '+percent(metadata['validation_metrics']['coverage_80'],False)+' · '+str(metadata['validation_n'])+' observations · '+record['regime'])
with st.expander('All horizon estimates'):
    rows=[]
    for entry in data['forecasts']:
        value=entry['forecast']
        rows.append({'Horizon':str(value['horizon'])+' sessions','Central return':percent(value['central_return']),
            'Implied price':price(value['central_price'],symbol),
            '50% range':price(value['quantile_prices'][1],symbol)+' – '+price(value['quantile_prices'][3],symbol),
            '80% range':price(value['quantile_prices'][0],symbol)+' – '+price(value['quantile_prices'][4],symbol),
            'Calibrated P(positive)':percent(value['p_positive_calibrated'],False),
            'Calibration state':'Accepted' if value['p_positive_calibrated'] is not None else 'Rejected; raw only',
            'Confidence':value['trust']['confidence']})
    st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)

st.subheader('Evidence')
tabs=st.tabs(['Model contributions','Model & data health','Relative performance','Historical analogues','Scenario lab','Invalidation'])
with tabs[0]:
    contribution=record['drivers']
    frame=pd.DataFrame({'Feature':contribution['features'],'Log-return contribution':contribution['values']})
    frame['magnitude']=frame['Log-return contribution'].abs()
    frame=frame.nlargest(12,'magnitude').sort_values('Log-return contribution')
    figure=go.Figure(go.Bar(x=frame['Log-return contribution'],y=frame.Feature,orientation='h',
        marker_color=['#79C4A5' if v>=0 else '#EE9A9A' for v in frame['Log-return contribution']]))
    st.plotly_chart(chart_layout(figure,height=420),use_container_width=True)
    st.caption(contribution['label'])
with tabs[1]:
    st.dataframe(pd.DataFrame([{'Item':'Price structure','State':data['health']['price_data_status']},
        {'Item':'Historical source vintages','State':'UNVERIFIED'}, {'Item':'Calendar','State':'Independent research proxy'},
        {'Item':'Feature schema','State':record['feature_schema_hash'][:16]}, {'Item':'Training label cutoff','State':metadata['training_cutoff']},
        {'Item':'Blending cutoff','State':metadata.get('blending_cutoff','Unavailable')},
        {'Item':'Calibration cutoff','State':metadata['calibration_cutoff']},
        {'Item':'OOD domain ratio','State':f'{forecast["ood_score"]:.2f}'}, {'Item':'Production promotion','State':'BLOCKED'}]),hide_index=True,use_container_width=True)
    st.json(data['context_health'],expanded=False)
    st.json({'regressors':forecast['component_returns'],'classifier_components':forecast['probability_components']},expanded=False)
with tabs[2]:
    relative=forecast.get('relative_forecasts',{})
    if not relative: st.info('No auxiliary benchmark/sector model supports this configuration.')
    for name,result in relative.items():
        st.write(name.title()+' excess log-return estimate: '+percent(result['central_excess_log_return']))
        st.write('Calibrated probability outperform: '+percent(result['calibrated_p_outperform'],False))
        st.caption('Auxiliary model association; do not sum market and sector alpha as independent contributions.')
with tabs[3]:
    analogues=item['analogues']
    if analogues['n']==0: st.info('No matured historical analogues available.')
    else:
        st.write(str(analogues['n'])+' nearest matured states · median return '+percent(analogues['median_return'])+' · win rate '+percent(analogues['win_rate'],False))
        st.bar_chart(pd.DataFrame({'Realized return':analogues['distribution']}))
        st.caption(analogues['limitation'])
with tabs[4]:
    st.caption('Hypothetical source shocks rebuild features and rerun this exact trained horizon. Unsupported sources fail explicitly; no heuristic scenario returns.')
    with st.form('scenario'):
        source=st.selectbox('Source',['stock','market','sector','nasdaq','usd_inr','brent','vix'])
        shock=st.slider('Relative change (%)',-20.,20.,0.,1.)
        apply=st.form_submit_button('Recompute hypothetical forecast')
    if apply:
        try:
            result=item['scenario_engine'].run({source:shock/100})
            st.metric('Hypothetical central return',percent(result['central_return']))
            st.caption('Changed inputs: '+', '.join(result['changed_features'])+' · '+result['trust']['status'])
        except Exception as error: st.warning(str(error))
with tabs[5]:
    st.info('The base forecast already abstains; there is no actionable thesis to invalidate.')
    if st.button('Measure input sensitivity'):
        st.dataframe(pd.DataFrame(item['scenario_engine'].sensitivity()),hide_index=True,use_container_width=True)
with st.expander('Full immutable forecast provenance'):
    st.json(record)
