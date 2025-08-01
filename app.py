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

# Functions for calculating SMA, EMA, MACD, RSI
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

# Function for plotting Stock Prices, Volume, Indicators & Returns
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

# Cache data fetching
@st.cache_data
def fetch_stock_data(ticker, period, interval='1d'):
    return yf.download(ticker, period=period, interval=interval, auto_adjust=False)

@st.cache_data
def get_ticker_info(ticker):
    return yf.Ticker(ticker).info

@st.cache_data
def load_csv(url):
    return pd.read_csv(url)

# Main app
st.markdown('''
# Stock Market Analysis and Prediction
''')
st.write('---')

st.header('INFORMATION')

st.sidebar.subheader('Query parameters')
period = load_csv('https://raw.githubusercontent.com/mayanknagar10/stock_market_analysis_and_prediction/main/period.csv')
time = st.sidebar.selectbox("Time Period", period)

indicators_period = load_csv('https://raw.githubusercontent.com/mayanknagar10/stock_market_analysis_and_prediction/main/indicators_period.csv')
time_period_input = st.sidebar.selectbox("Time Period for indicators", indicators_period.indicators_time)
if time_period_input == '6m':
    time_period = 126
elif time_period_input == '1y':
    time_period = 252
elif time_period_input == '3y':
    time_period = 756
elif time_period_input == '5y':
    time_period = 1800

ticker_list = load_csv('https://raw.githubusercontent.com/mayanknagar10/stock_market_analysis_and_prediction/main/company_data.csv')
user_input = st.sidebar.selectbox('Enter Stock Ticker', ticker_list)

st.sidebar.write('For other Ticker or Company refer a yahoo finance website: https://finance.yahoo.com/')

indicators = st.sidebar.radio("Indicators", ('SMA', 'EMA', 'MACD', 'RSI', 'Bollinger Bands'))
returns = st.sidebar.radio("Returns", ('Daily Returns', 'Cumulative Returns'))

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
    information = get_ticker_info(user_input)
    if "logo_url" in information and information["logo_url"]:
        logo_url = information["logo_url"]
        logging.info(f"Logo URL for {user_input}: {logo_url}")
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(logo_url, headers=headers, timeout=5)
            if response.status_code == 200 and 'image' in response.headers.get('Content-Type', ''):
                st.image(logo_url, width=150, caption=f"{user_input} Logo")
            else:
                st.image(placeholder_logo, width=150, caption=f"No logo available for {user_input} (invalid URL: {response.status_code})")
                logging.warning(f"Invalid logo URL for {user_input}: status={response.status_code}, content-type={response.headers.get('Content-Type')}")
        except Exception as e:
            st.image(placeholder_logo, width=150, caption=f"No logo available for {user_input} (error: {str(e)})")
            logging.warning(f"Error accessing logo URL for {user_input}: {str(e)}")
    else:
        st.image(placeholder_logo, width=150, caption=f"No logo available for {user_input}")
        logging.info(f"No logo_url found in info for {user_input}")

try:
    # Fetch stock data with auto_adjust=False to include Adj Close
    with st.spinner("Fetching data..."):
        df = fetch_stock_data(user_input, time)
    if df.empty:
        st.error(f"No data found for ticker {user_input}. Please check the ticker symbol.")
        st.stop()

    # Handle MultiIndex if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    logging.info(f"Columns in df: {df.columns.tolist()}")

    string_name = information.get('longName', 'Unknown Company')
    st.header('**%s**' % string_name)

    string_summary = information.get('longBusinessSummary', 'No summary available.')
    st.info(string_summary)

    # Process DataFrame
    df1 = df.reset_index()
    logging.info(f"Columns in df1: {df1.columns.tolist()}")
    columns_to_drop = [col for col in ['Date', 'Adj Close'] if col in df1.columns]
    if columns_to_drop:
        df1 = df1.drop(columns=columns_to_drop, axis=1)
    else:
        st.warning("Expected columns 'Date' or 'Adj Close' not found in data. Continuing with available columns.")

    st.subheader('Candlestick chart')
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
        xaxis=dict(title=dict(text="Date", font=dict(family="Arial", size=12, color="black"))),
        yaxis=dict(title=dict(text="Price", font=dict(family="Arial", size=12, color="black")))
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Data')
    df['Date'] = df.index
    st.write(df.tail(2))

    st.subheader('Fundamentals')
    try:
        ticker = yf.Ticker(user_input)
        info = ticker.info
        st.write("52 Week Range: ", f"{info.get('fiftyTwoWeekLow', 'N/A')} - {info.get('fiftyTwoWeekHigh', 'N/A')}")
        st.write("Day's Range: ", f"{info.get('dayLow', 'N/A')} - {info.get('dayHigh', 'N/A')}")
        st.write("Average Volume: ", info.get('averageVolume', 'N/A'))
        st.write("EPS: ", info.get('trailingEps', 'N/A'))
        st.write("Market Cap: ", info.get('marketCap', 'N/A'))
        st.write("PE Ratio: ", info.get('trailingPE', 'N/A'))
        st.write("Volume: ", info.get('volume', 'N/A'))
        st.write("Quote Price: ", info.get('regularMarketPrice', info.get('previousClose', 'N/A')))
    except Exception as e:
        st.warning(f"Error fetching fundamentals from yfinance: {str(e)}. Trying yahoo-fin as fallback.")
        try:
            data1 = si.get_quote_table(user_input)
            st.write("52 Week Range: ", data1.get("52 Week Range", "N/A"))
            st.write("Day's Range: ", data1.get("Day's Range", "N/A"))
            st.write("Average Volume: ", data1.get("Avg. Volume", "N/A"))
            st.write("EPS: ", data1.get("EPS (TTM)", "N/A"))
            st.write("Market Cap: ", data1.get("Market Cap", "N/A"))
            st.write("PE Ratio: ", data1.get("PE Ratio (TTM)", "N/A"))
            st.write("Volume: ", data1.get("Volume", "N/A"))
            st.write("Quote Price: ", data1.get("Quote Price", "N/A"))
        except Exception as e2:
            st.warning(f"Error fetching fundamentals from yahoo-fin: {str(e2)}. Continuing without fundamentals.")

    company = yf.Ticker(user_input)
    st.write("Major Holders: ", company.major_holders if company.major_holders is not None else "N/A")
    st.write("Institutional Holders: ", company.institutional_holders if company.institutional_holders is not None else "N/A")

    st.subheader("Stock Price Chart")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Open'], name='Open Price', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=df.index, y=df['High'], name='High Price', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Low'], name='Low Price', line=dict(color='purple')))
    fig.update_layout(
        title=f"{user_input} Stock Prices",
        xaxis=dict(title=dict(text="Date", font=dict(family="Arial", size=12, color="black"))),
        yaxis=dict(title=dict(text="Stock Price", font=dict(family="Arial", size=12, color="black")))
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Volume Traded Chart")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', opacity=0.5))
    fig.update_layout(
        title=f"{user_input} Volume Traded",
        xaxis=dict(title=dict(text="Date", font=dict(family="Arial", size=12, color="black"))),
        yaxis=dict(title=dict(text="Volume", font=dict(family="Arial", size=12, color="black")))
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Indicators and Returns')
    MACD(df)
    RSI(df)
    BB(df)
    df['SMA_50'] = SMA(df, 50)
    df['SMA_200'] = SMA(df, 200)
    df['EMA'] = EMA(df)
    df['Date'] = df.index
    fig = get_stock_price_fig(df.tail(time_period), indicators, returns)
    st.plotly_chart(fig, use_container_width=True)

    st.write("First Quant Figure")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='rgb(31, 119, 180)')))
    fig.add_trace(go.Scatter(x=df.index, y=df['BOLU'], name='Upper Bollinger Band', line=dict(width=0.5, color='#89BCFD')))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['BOLD'],
        name='Lower Bollinger Band',
        line=dict(width=0.5, color='#89BCFD'),
        fill='tonexty',
        fillcolor='rgba(56, 224, 56, 0.5)'
    ))
    fig.update_layout(
        title=dict(text='First Quant Figure', font=dict(family="Arial", size=12, color="black")),
        xaxis=dict(title=dict(text="Date", font=dict(family="Arial", size=12, color="black"))),
        yaxis=dict(title=dict(text="Price", font=dict(family="Arial", size=12, color="black")))
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Result and Prediction')
    close = df[['Close']]
    ds = close.values
    normalizer = MinMaxScaler(feature_range=(0,1))
    ds_scaled = normalizer.fit_transform(np.array(ds).reshape(-1,1))

    st.write('Closing Price vs Time chart with 100MA')
    ma100 = df.Close.rolling(100).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=ma100, name='100MA', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='blue')))
    fig.update_layout(
        title=f"{user_input} Closing Price with 100MA",
        xaxis=dict(title=dict(text="Date", font=dict(family="Arial", size=12, color="black"))),
        yaxis=dict(title=dict(text="Price", font=dict(family="Arial", size=12, color="black")))
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write('Closing Price vs Time chart with 100MA & 200MA')
    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=ma100, name='100MA', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=df.index, y=ma200, name='200MA', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='blue')))
    fig.update_layout(
        title=f"{user_input} Closing Price with 100MA & 200MA",
        xaxis=dict(title=dict(text="Date", font=dict(family="Arial", size=12, color="black"))),
        yaxis=dict(title=dict(text="Price", font=dict(family="Arial", size=12, color="black")))
    )
    st.plotly_chart(fig, use_container_width=True)

    # Model prediction with compatibility fix
    try:
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

        model = load_model('keras_model.h5', custom_objects={'LSTM': CustomLSTM})
        logging.info(f"Model loaded successfully. Input shape: {model.input_shape}")
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        train_predict = normalizer.inverse_transform(train_predict)
        test_predict = normalizer.inverse_transform(test_predict)

        test = np.vstack((train_predict, test_predict))

        fut_inp = ds_test[-100:]
        fut_inp = fut_inp.reshape(1, -1)
        tmp_inp = fut_inp[0].tolist()

        lst_output = []
        n_steps = 100
        i = 0
        while i < 30:
            fut_inp = np.array(tmp_inp).reshape(1, n_steps, 1)
            yhat = model.predict(fut_inp, verbose=0)
            logging.info(f"Iteration {i}: yhat shape: {yhat.shape}, yhat: {yhat.flatten()[:5]}")
            yhat = yhat.flatten()[0]  # Extract scalar
            tmp_inp.append(yhat)
            tmp_inp = tmp_inp[1:]
            lst_output.append(yhat)
            i += 1

        # Ensure lst_output is 1D
        lst_output = np.array(lst_output, dtype=float)
        logging.info(f"lst_output shape: {lst_output.shape}, values: {lst_output[:5]}")

        st.write('Result')
        plot_new = np.arange(1, 101)
        plot_pred = np.arange(101, 131)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=plot_new, y=normalizer.inverse_transform(ds_scaled[-100:]).flatten(), name='Historical'))
        fig.add_trace(go.Scatter(x=plot_pred, y=normalizer.inverse_transform(lst_output.reshape(-1, 1)).flatten(), name='Predicted'))
        fig.update_layout(
            title=f"{user_input} Prediction for Next 30 Days",
            xaxis=dict(title=dict(text="Time", font=dict(family="Arial", size=12, color="black"))),
            yaxis=dict(title=dict(text="Price", font=dict(family="Arial", size=12, color="black")))
        )
        st.plotly_chart(fig, use_container_width=True)

        ds_new = ds_scaled.flatten().tolist()
        ds_new.extend(lst_output.tolist())
        ds_new = np.array(ds_new, dtype=float)
        logging.info(f"ds_new shape: {ds_new.shape}, values: {ds_new[-5:]}")
        final_graph = normalizer.inverse_transform(ds_new.reshape(-1, 1)).flatten().tolist()

        st.write('Prediction')
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=final_graph, name='Price'))
        fig.add_hline(y=final_graph[-1], line_dash="dot", line_color="red", annotation_text=f"NEXT 30D: {round(final_graph[-1], 2)}")
        fig.update_layout(
            title=f"{user_input} Prediction of Next Month Close",
            xaxis=dict(title=dict(text="Time", font=dict(family="Arial", size=12, color="black"))),
            yaxis=dict(title=dict(text="Price", font=dict(family="Arial", size=12, color="black")))
        )
        st.plotly_chart(fig, use_container_width=True)

        price = round(float(final_graph[-1]), 2)
        st.write("Next 30D price: ", price)

    except Exception as e:
        st.error(f"Error loading or predicting with Keras model: {str(e)}. Skipping prediction section.")
        logging.error(f"Keras model error: {str(e)}")

except Exception as e:
    st.error(f"Error fetching or processing data for ticker {user_input}: {str(e)}")
    st.stop()
