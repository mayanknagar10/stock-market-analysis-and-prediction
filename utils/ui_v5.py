"""Calm research UI tokens and honest numerical formatting."""
import math
import streamlit as st
import plotly.graph_objects as go

PALETTE={'background':'#0B1220','surface':'#121D30','border':'#2B3A50','text':'#E6EDF5','muted':'#AAB8CA',
         'accent':'#74A8EC','warning':'#E7BB65','positive':'#79C4A5','negative':'#EE9A9A'}

def inject_theme():
    st.markdown('''<style>
        .stApp{background:#0B1220;color:#E6EDF5;font-family:system-ui,-apple-system,'Segoe UI',sans-serif;}
        [data-testid="stSidebar"]{background:#121D30;border-right:1px solid #2B3A50;}
        .block-container{max-width:1440px;padding-top:2rem;padding-bottom:3rem;}
        h1{font-size:1.9rem!important;font-weight:650!important;letter-spacing:-.02em;}
        h2{font-size:1.35rem!important;margin-top:1.5rem!important;}
        h3{font-size:1.1rem!important;}
        p,label,.stMarkdown{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;}
        [data-testid="stMetricValue"]{font-size:1.65rem;font-variant-numeric:tabular-nums;}
        [data-testid="stMetricLabel"],.stCaption{color:#AAB8CA;}
        button,input,select{font-variant-numeric:tabular-nums;}
        button:focus-visible,input:focus-visible{outline:2px solid #74A8EC!important;outline-offset:3px;}
        ::selection{background:#31577D;color:#FFFFFF;}
        .kpi-card,.signal-card{background:#121D30!important;border:1px solid #2B3A50!important;box-shadow:none!important;}
        @media(max-width:700px){.block-container{padding:1rem;}h1{font-size:1.55rem!important;}}
    </style>''',unsafe_allow_html=True)

def percent(value,signed=True):
    if value is None: return 'Unavailable'
    try:
        if not math.isfinite(float(value)): return 'Unavailable'
        return f'{value*100:+.1f}%' if signed else f'{value*100:.1f}%'
    except (ValueError,TypeError): return 'Unavailable'

def price(value,symbol=''):
    if value is None or not math.isfinite(float(value)): return 'Unavailable'
    return f'{symbol}{value:,.2f}'

def chart_layout(figure,title=None,height=320):
    figure.update_layout(template='plotly_dark',paper_bgcolor=PALETTE['background'],plot_bgcolor=PALETTE['background'],
        font={'family':'system-ui, Segoe UI, sans-serif','color':PALETTE['text'],'size':13},
        margin={'l':24,'r':16,'t':40 if title else 16,'b':30},height=height,title=title,
        xaxis={'gridcolor':PALETTE['border']},yaxis={'gridcolor':PALETTE['border']},
        legend={'orientation':'h','y':1.12,'x':0})
    return figure

def distribution_chart(forecast,current_price):
    quantiles=[10,25,50,75,90]
    figure=go.Figure()
    figure.add_trace(go.Scatter(x=forecast['quantile_prices'],y=quantiles,mode='lines+markers',
        name='Conditional quantiles',line={'color':PALETTE['accent'],'width':2},
        hovertemplate='Implied price %{x:,.2f}<br>Quantile %{y}%<extra></extra>'))
    figure.add_vline(x=current_price,line_color=PALETTE['muted'],line_dash='dot',annotation_text='Current completed close')
    figure.update_yaxes(title='Return quantile (%)')
    figure.update_xaxes(title='Total-return-equivalent implied price')
    return chart_layout(figure,height=340)
