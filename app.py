from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.models import load_model
import yahoo_fin.stock_info as si
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import date, timedelta, datetime
from arch import arch_model
import yfinance as yf
import os
import logging
from io import StringIO
import requests
import functools

# ================================================
# Configuration
# ================================================
# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Configure logging for debugging
logging.basicConfig(level=logging.INFO)

# Custom LSTM to handle deprecated parameters
class CustomLSTM(LSTM):
    def __init__(self, *args, **kwargs):
        kwargs.pop('time_major', None)
        kwargs.pop('input_shape', None)
        super(CustomLSTM, self).__init__(*args, **kwargs)

# ================================================
# Technical Indicators Functions
# ================================================
def SMA(data, period=100, column='Close'):
    return data[column].rolling(window=period).mean()

def EMA(data, period=20, column='Close'):
    return data[column].ewm(span=period, adjust=False).mean()

def MACD(data, period_long=26, period_short=12, period_signal=9, column='Close'):
    shortEMA = EMA(data, period_short, column=column)
    longEMA = EMA(data, period_long, column=column)
    data['MACD'] = shortEMA - longEMA
    data['Signal_Line'] = EMA(data, period_signal, column='MACD')
    return data

def RSI(data, period=14, column='Close'):
    delta = data[column].diff(1)
    delta = delta[1:]
    up = delta.copy()
    down = delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    data['up'] = up
    data['down'] = down
    avg_gain = SMA(data, period, column='up')
    avg_loss = abs(SMA(data, period, column='down'))
    RS = avg_gain / avg_loss
    RSI = 100.0 - (100.0 / (1.0 + RS))
    data['RSI'] = RSI
    return data

def BB(data):
    data['TP'] = (data['Close'] + data['Low'] + data['High']) / 3
    data['std'] = data['TP'].rolling(20).std(ddof=0)
    data['MA-TP'] = data['TP'].rolling(20).mean()
    data['BOLU'] = data['MA-TP'] + 2 * data['std']
    data['BOLD'] = data['MA-TP'] - 2 * data['std']
    return data

# ================================================
# Plotting Functions
# ================================================
def get_stock_price_fig(df, v2, v3):
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_width=[0.1, 0.2, 0.1, 0.3],
        subplot_titles=("", "", v2, v3 + ' %')
    )
    
    fig.add_trace(
        go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            showlegend=False,
            name='Price'
        ),
        row=1,
        col=1
    )
    
    fig.add_trace(
        go.Bar(x=df['Date'], y=df['Volume'], opacity=0.5, showlegend=False, name='Volume'),
        row=2,
        col=1
    )
    
    # Indicators
    if v2 == 'RSI':
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['RSI'], mode="lines", name='RSI', marker=dict(color='rgb(31, 119, 180)'), showlegend=False),
            row=3,
            col=1
        )
    elif v2 == 'SMA':
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['SMA_50'], mode="lines", name='SMA_50', showlegend=False, marker=dict(color='rgb(31, 119, 180)')),
            row=3,
            col=1
        )
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['SMA_200'], mode="lines", name='SMA_200', showlegend=False, marker=dict(color='#ff3333')),
            row=3,
            col=1
        )
    elif v2 == 'EMA':
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['EMA'], mode="lines", name='EMA', showlegend=False, marker=dict(color='rgb(31, 119, 180)')),
            row=3,
            col=1
        )
    elif v2 == 'MACD':
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['MACD'], mode="lines", name='MACD', showlegend=False, marker=dict(color='rgb(31, 119, 180)')),
            row=3,
            col=1
        )
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['Signal_Line'], mode="lines", name='Signal_Line', showlegend=False, marker=dict(color='#ff3333')),
            row=3,
            col=1
        )
    elif v2 == 'Bollinger Bands':
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['Close'], mode="lines", line=dict(color='rgb(31, 119, 180)'), name='Close', showlegend=False),
            row=3,
            col=1
        )
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['BOLU'], mode="lines", line=dict(width=0.5), marker=dict(color="#89BCFD"), showlegend=False, name='Upper Band'),
            row=3,
            col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['BOLD'],
                mode="lines",
                line=dict(width=0.5),
                marker=dict(color="#89BCFD"),
                showlegend=False,
                fillcolor='rgba(56, 224, 56, 0.5)',
                fill='tonexty',
                name='Lower Band'
            ),
            row=3,
            col=1
        )
    
    # Returns
    if v3 == "Daily Returns":
        rets = df['Close'] / df['Close'].shift(1) - 1
        fig.add_trace(
            go.Scatter(x=df['Date'], y=rets, mode="lines", showlegend=False, name='Daily Return', line=dict(color='#FF4136')),
            row=4,
            col=1
        )
    elif v3 == "Cumulative Returns":
        rets = df['Close'] / df['Close'].shift(1) - 1
        cum_rets = (rets + 1).cumprod()
        fig.add_trace(
            go.Scatter(x=df['Date'], y=cum_rets, mode="lines", showlegend=False, name='Cumulative Returns', line=dict(color='#FF4136')),
            row=4,
            col=1
        )
    
    fig.update_layout(
        margin=dict(b=0, t=0, l=0, r=0),
        plot_bgcolor='#ebf3ff',
        width=500,
        height=600,
        xaxis=dict(
            showticklabels=True,
            showgrid=False,
            title=dict(text="Date", font=dict(family="Arial", size=12, color="black"))
        ),
        xaxis3=dict(
            showgrid=False,
            title=dict(text="Date", font=dict(family="Arial", size=12, color="black"))
        ),
        xaxis4=dict(
            showticklabels=False,
            showgrid=False,
            title=dict(text="Date", font=dict(family="Arial", size=12, color="black"))
        ),
        yaxis=dict(
            title=dict(text="Price", font=dict(family="Arial", size=12, color="black"))
        ),
        yaxis2=dict(
            title=dict(text="Volume", font=dict(family="Arial", size=12, color="black"))
        ),
        yaxis3=dict(
            title=dict(text=v2, font=dict(family="Arial", size=12, color="black"))
        ),
        yaxis4=dict(
            title=dict(text=v3 + " %", font=dict(family="Arial", size=12, color="black"))
        )
    )
    return fig

# ================================================
# Caching Functions
# ================================================
@st.cache_data
def fetch_stock_data(ticker, period, interval='1d'):
    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=False)
        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        return df
    except Exception as e:
        logging.error(f"Error in fetch_stock_data for {ticker}: {e}")
        return pd.DataFrame()

@st.cache_data
def get_ticker_info(ticker):
    try:
        return yf.Ticker(ticker).info or {}
    except Exception as e:
        logging.warning(f"Error fetching ticker info for {ticker}: {e}")
        return {}

@st.cache_data
def load_csv(url):
    try:
        return pd.read_csv(url)
    except Exception as e:
        logging.error(f"Error loading CSV from {url}: {e}")
        return pd.DataFrame()

# ================================================
# Main Application
# ================================================
st.set_page_config(page_title="Stock Market Analysis", layout="wide")

st.markdown('''
# Stock Market Analysis and Prediction
''')
st.write('---')

# ================================================
# Sidebar Configuration
# ================================================
st.sidebar.subheader('Query parameters')

# Load configuration files
period = load_csv('https://raw.githubusercontent.com/mayanknagar10/stock_market_analysis_and_prediction/main/period.csv')
if not period.empty:
    time = st.sidebar.selectbox("Time Period", period)
else:
    time = st.sidebar.selectbox("Time Period", ['1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'max'])

indicators_period = load_csv('https://raw.githubusercontent.com/mayanknagar10/stock_market_analysis_and_prediction/main/indicators_period.csv')
if not indicators_period.empty and 'indicators_time' in indicators_period.columns:
    time_period_input = st.sidebar.selectbox("Time Period for indicators", indicators_period.indicators_time)
else:
    time_period_input = st.sidebar.selectbox("Time Period for indicators", ['6m', '1y', '3y', '5y'])

# Map time period to days
time_period_map = {'6m': 126, '1y': 252, '3y': 756, '5y': 1800}
time_period = time_period_map.get(time_period_input, 252)

# Load ticker list
ticker_list = load_csv('https://raw.githubusercontent.com/mayanknagar10/stock_market_analysis_and_prediction/main/company_data.csv')
if not ticker_list.empty:
    user_input = st.sidebar.selectbox('Enter Stock Ticker', ticker_list)
else:
    user_input = st.sidebar.text_input('Enter Stock Ticker', 'AAPL')

st.sidebar.write('For other Ticker or Company refer to yahoo finance website: https://finance.yahoo.com/')

indicators = st.sidebar.radio("Indicators", ('SMA', 'EMA', 'MACD', 'RSI', 'Bollinger Bands'))
returns = st.sidebar.radio("Returns", ('Daily Returns', 'Cumulative Returns'))

# ================================================
# Display Company Information
# ================================================
st.header('INFORMATION')

information = get_ticker_info(user_input)

# Handle logo display with robust fallback
placeholder_logo = "https://via.placeholder.com/150?text=No+Logo"
logo_mapping = {
    'NVDA': 'logos/NVDA.png',
    'GOOGL': 'logos/GOOGL.png',
    'AAPL': 'logos/AAPL.png',
    'SUDARSCHEM.NS': 'logos/SUDARSCHEM.NS.png',
    'MSFT': 'logos/MSFT.png',
    'AMZN': 'logos/AMZN.png'
}

local_logo = logo_mapping.get(user_input, f"logos/{user_input}.png")
if os.path.exists(local_logo):
    st.image(local_logo, width=150, caption=f"{user_input} Logo")
    logging.info(f"Using local logo for {user_input}: {local_logo}")
else:
    if "logo_url" in information and information["logo_url"]:
        logo_url = information["logo_url"]
        logging.info(f"Logo URL for {user_input}: {logo_url}")
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(logo_url, headers=headers, timeout=5)
            if response.status_code == 200 and 'image' in response.headers.get('Content-Type', ''):
                st.image(logo_url, width=150, caption=f"{user_input} Logo")
            else:
                st.image(placeholder_logo, width=150, caption=f"No logo available for {user_input}")
                logging.warning(f"Invalid logo URL for {user_input}")
        except Exception as e:
            st.image(placeholder_logo, width=150, caption=f"No logo available for {user_input}")
            logging.warning(f"Error accessing logo URL for {user_input}: {str(e)}")
    else:
        st.image(placeholder_logo, width=150, caption=f"No logo available for {user_input}")
        logging.info(f"No logo_url found in info for {user_input}")

# ================================================
# Fetch and Process Stock Data
# ================================================
try:
    with st.spinner("Fetching data..."):
        df = fetch_stock_data(user_input, time)
    
    if df.empty:
        st.error(f"No data found for ticker {user_input}. Please check the ticker symbol.")
        st.stop()
    
    logging.info(f"Columns in df: {df.columns.tolist()}")
    
    # Display company name and summary
    string_name = information.get('longName', user_input)
    st.header(f'**{string_name}**')
    
    string_summary = information.get('longBusinessSummary', 'No summary available.')
    st.info(string_summary)
    
    # ================================================
    # Candlestick Chart
    # ================================================
    st.subheader('Candlestick Chart')
    candlestick = go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close']
    )
    fig = go.Figure(data=[candlestick])
    fig.update_layout(
        title=f"{user_input} Candlestick Chart",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Price"),
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # ================================================
    # Recent Data
    # ================================================
    st.subheader('Recent Data')
    df_display = df.copy()
    df_display['Date'] = df_display.index
    st.dataframe(df_display.tail(10))
    
    # ================================================
    # Fundamentals
    # ================================================
    st.subheader('Fundamentals')
    try:
        ticker = yf.Ticker(user_input)
        info = ticker.info or {}
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("52 Week Range", f"{info.get('fiftyTwoWeekLow', 'N/A')} - {info.get('fiftyTwoWeekHigh', 'N/A')}")
            st.metric("Day's Range", f"{info.get('dayLow', 'N/A')} - {info.get('dayHigh', 'N/A')}")
            st.metric("Average Volume", f"{info.get('averageVolume', 'N/A'):,}" if info.get('averageVolume') else 'N/A')
            st.metric("Volume", f"{info.get('volume', 'N/A'):,}" if info.get('volume') else 'N/A')
        
        with col2:
            st.metric("Market Cap", f"{info.get('marketCap', 'N/A'):,}" if info.get('marketCap') else 'N/A')
            st.metric("PE Ratio", f"{info.get('trailingPE', 'N/A'):.2f}" if info.get('trailingPE') else 'N/A')
            st.metric("EPS", f"{info.get('trailingEps', 'N/A'):.2f}" if info.get('trailingEps') else 'N/A')
            st.metric("Quote Price", f"{info.get('regularMarketPrice', info.get('previousClose', 'N/A')):.2f}" if info.get('regularMarketPrice') or info.get('previousClose') else 'N/A')
    
    except Exception as e:
        st.warning(f"Error fetching fundamentals: {str(e)}")
    
    # ================================================
    # Stock Price Chart
    # ================================================
    st.subheader("Stock Price Chart")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Open'], name='Open', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=df.index, y=df['High'], name='High', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Low'], name='Low', line=dict(color='purple')))
    fig.update_layout(
        title=f"{user_input} Stock Prices",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Stock Price"),
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # ================================================
    # Volume Chart
    # ================================================
    st.subheader("Volume Traded Chart")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', opacity=0.6, marker_color='lightblue'))
    fig.update_layout(
        title=f"{user_input} Volume Traded",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Volume")
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # ================================================
    # Technical Indicators
    # ================================================
    st.subheader('Technical Indicators and Returns')
    
    # Calculate all indicators
    df = MACD(df)
    df = RSI(df)
    df = BB(df)
    df['SMA_50'] = SMA(df, 50)
    df['SMA_200'] = SMA(df, 200)
    df['EMA'] = EMA(df)
    df['Date'] = df.index
    
    # Plot indicators
    fig = get_stock_price_fig(df.tail(time_period), indicators, returns)
    st.plotly_chart(fig, use_container_width=True)
    
    # ================================================
    # Bollinger Bands
    # ================================================
    st.subheader('Bollinger Bands')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='rgb(31, 119, 180)')))
    fig.add_trace(go.Scatter(x=df.index, y=df['BOLU'], name='Upper Band', line=dict(width=0.5, color='#89BCFD')))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['BOLD'],
        name='Lower Band',
        line=dict(width=0.5, color='#89BCFD'),
        fill='tonexty',
        fillcolor='rgba(56, 224, 56, 0.3)'
    ))
    fig.update_layout(
        title='Bollinger Bands',
        xaxis=dict(title="Date"),
        yaxis=dict(title="Price")
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # ================================================
    # Moving Averages
    # ================================================
    st.subheader('Moving Averages')
    
    st.write('Closing Price with 100MA')
    ma100 = df.Close.rolling(100).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=ma100, name='100MA', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='blue')))
    fig.update_layout(
        title=f"{user_input} Closing Price with 100MA",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Price")
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.write('Closing Price with 100MA & 200MA')
    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=ma100, name='100MA', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=df.index, y=ma200, name='200MA', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='blue')))
    fig.update_layout(
        title=f"{user_input} Closing Price with 100MA & 200MA",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Price")
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # ================================================
    # LSTM Prediction Section
    # ================================================
    st.subheader('LSTM Price Prediction')
    
    # Check if model file exists
    if not os.path.exists('keras_model.h5'):
        st.warning("LSTM model file 'keras_model.h5' not found. Skipping prediction section.")
        st.info("To use predictions, please ensure the keras_model.h5 file is in the same directory as this script.")
    else:
        try:
            # Prepare data for LSTM
            close = df[['Close']]
            ds = close.values
            normalizer = MinMaxScaler(feature_range=(0, 1))
            ds_scaled = normalizer.fit_transform(np.array(ds).reshape(-1, 1))
            
            # Split data
            train_size = int(len(ds_scaled) * 0.70)
            test_size = len(ds_scaled) - train_size
            ds_train, ds_test = ds_scaled[0:train_size, :], ds_scaled[train_size:len(ds_scaled), :]
            
            def create_ds(dataset, step):
                Xtrain, Ytrain = [], []
                for i in range(len(dataset) - step - 1):
                    a = dataset[i:(i + step), 0]
                    Xtrain.append(a)
                    Ytrain.append(dataset[i + step, 0])
                return np.array(Xtrain), np.array(Ytrain)
            
            time_stamp = 100
            X_train, y_train = create_ds(ds_train, time_stamp)
            X_test, y_test = create_ds(ds_test, time_stamp)
            
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
            
            # Load model
            model = load_model('keras_model.h5', custom_objects={'LSTM': CustomLSTM})
            logging.info(f"Model loaded successfully. Input shape: {model.input_shape}")
            
            # Make predictions
            train_predict = model.predict(X_train)
            test_predict = model.predict(X_test)
            
            train_predict = normalizer.inverse_transform(train_predict)
            test_predict = normalizer.inverse_transform(test_predict)
            
            # Future predictions (30 days)
            fut_inp = ds_test[-100:]
            fut_inp = fut_inp.reshape(1, -1)
            tmp_inp = fut_inp[0].tolist()
            lst_output = []
            n_steps = 100
            
            with st.spinner("Generating predictions..."):
                for i in range(30):
                    fut_inp_reshaped = np.array(tmp_inp[-n_steps:]).reshape(1, n_steps, 1)
                    yhat = model.predict(fut_inp_reshaped, verbose=0)
                    yhat_value = float(yhat[0, 0])
                    tmp_inp.append(yhat_value)
                    lst_output.append(yhat_value)
            
            lst_output = np.array(lst_output, dtype=float)
            
            # Plot prediction results
            st.write('Historical vs Predicted (Next 30 Days)')
            plot_new = np.arange(1, 101)
            plot_pred = np.arange(101, 131)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=plot_new, 
                y=normalizer.inverse_transform(ds_scaled[-100:]).flatten(), 
                name='Historical',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=plot_pred, 
                y=normalizer.inverse_transform(lst_output.reshape(-1, 1)).flatten(), 
                name='Predicted',
                line=dict(color='red', dash='dash')
            ))
            fig.update_layout(
                title=f"{user_input} - 30 Day Price Prediction",
                xaxis=dict(title="Time Period"),
                yaxis=dict(title="Price")
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Full prediction chart
            ds_new = ds_scaled.flatten().tolist()
            ds_new.extend(lst_output.tolist())
            ds_new = np.array(ds_new, dtype=float)
            
            final_graph = normalizer.inverse_transform(ds_new.reshape(-1, 1)).flatten().tolist()
            
            st.write('Complete Price Prediction Chart')
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=final_graph, name='Price', line=dict(color='blue')))
            predicted_price = round(final_graph[-1], 2)
            fig.add_hline(
                y=predicted_price, 
                line_dash="dot", 
                line_color="red", 
                annotation_text=f"Predicted 30D: ${predicted_price}"
            )
            fig.update_layout(
                title=f"{user_input} - Full Prediction",
                xaxis=dict(title="Time"),
                yaxis=dict(title="Price")
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display predicted price
            st.success(f"**Predicted price in 30 days: ${predicted_price}**")
            
            # Calculate price change
            current_price = df['Close'].iloc[-1]
            price_change = predicted_price - current_price
            price_change_pct = (price_change / current_price) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"${current_price:.2f}")
            with col2:
                st.metric("Predicted Price (30D)", f"${predicted_price:.2f}")
            with col3:
                st.metric("Expected Change", f"${price_change:.2f}", f"{price_change_pct:.2f}%")
        
        except Exception as e:
            st.error(f"Error in prediction model: {str(e)}")
            logging.error(f"LSTM prediction error: {str(e)}", exc_info=True)
            st.info("The prediction section encountered an error. This could be due to insufficient data or model compatibility issues.")

except Exception as e:
    st.error(f"Error processing data for ticker {user_input}: {str(e)}")
    logging.error(f"Main execution error: {str(e)}", exc_info=True)
    st.stop()

# ================================================
# Footer
# ================================================
st.write('---')
st.markdown("""
<div style='text-align: center'>
    <p>Made with ❤️ using Streamlit | Data source: Yahoo Finance</p>
</div>
""", unsafe_allow_html=True)
