from time import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf
import requests
import os
import logging
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
import tensorflow as tf

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
logging.basicConfig(level=logging.INFO)

# ==================== CONFIG ====================
st.set_page_config(page_title="Stock Analyzer", layout="wide")
st.markdown("# 📈 Stock Market Analysis & Prediction")
st.write("---")

# ==================== SAFE INFO & LOGO ====================
@st.cache_data(ttl=3600)
def get_ticker_info_safe(ticker: str) -> dict:
    ticker = ticker.upper().strip()
    try:
        url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
        params = {"modules": "summaryProfile,financialData,quoteType,defaultKeyStatistics,price"}
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()["quoteSummary"]["result"][0]

        profile = data.get("summaryProfile", {})
        price = data.get("price", {})
        financial = data.get("financialData", {})

        return {
            "longName": price.get("longName", ticker),
            "longBusinessSummary": profile.get("longBusinessSummary", "No summary available."),
            "logo_url": price.get("logo_url", ""),
            "regularMarketPrice": price.get("regularMarketPrice", {}).get("raw"),
            "previousClose": price.get("regularMarketPreviousClose", {}).get("raw"),
            "marketCap": price.get("marketCap", {}).get("raw"),
            "trailingPE": price.get("trailingPE", {}).get("raw"),
            "fiftyTwoWeekHigh": price.get("fiftyTwoWeekHigh", {}).get("raw"),
            "fiftyTwoWeekLow": price.get("fiftyTwoWeekLow", {}).get("raw"),
        }
    except Exception as e:
        logging.warning(f"Info fetch failed for {ticker}: {e}")
        return {
            "longName": ticker,
            "longBusinessSummary": "Information temporarily unavailable.",
            "logo_url": "",
            "regularMarketPrice": None,
            "previousClose": None,
        }

@st.cache_data(ttl=3600)
def get_logo_url(ticker: str) -> str:
    local = f"logos/{ticker}.png"
    if os.path.isfile(local):
        return local
    info = get_ticker_info_safe(ticker)
    logo = info.get("logo_url", "")
    return logo if logo else "https://via.placeholder.com/150?text=No+Logo"

# ==================== DATA FETCHING ====================
@st.cache_data(ttl=600)
def fetch_stock_data(ticker: str, period: str):
    df = yf.download(ticker, period=period, progress=False, auto_adjust=False)
    if df.empty:
        st.error(f"No data for {ticker}. Check the ticker symbol.")
        st.stop()
    return df

# ==================== INDICATORS ====================
def add_indicators(df):
    df = df.copy()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()

    # MACD
    short = df["Close"].ewm(span=12, adjust=False).mean()
    long = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = short - long
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # RSI
    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(14).mean()
    ma_down = down.rolling(14).mean()
    rs = ma_up / ma_down
    df["RSI"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df["TP"] = (df["Close"] + df["Low"] + df["High"]) / 3
    df["BB_Mid"] = df["TP"].rolling(20).mean()
    df["BB_Std"] = df["TP"].rolling(20).std()
    df["BB_Upper"] = df["BB_Mid"] + 2 * df["BB_Std"]
    df["BB_Lower"] = df["BB_Mid"] - 2 * df["BB_Std"]

    return df

# ==================== LSTM MODEL (with fallback training) ====================
MODEL_PATH = "keras_model.h5"

def build_and_train_model(X_train, y_train):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(60, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(80, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=32, epochs=15, verbose=0)
    model.save(MODEL_PATH)
    st.success("New LSTM model trained and saved!")
    return model

@st.cache_resource
def load_or_train_model(X_train, y_train):
    if os.path.exists(MODEL_PATH):
        try:
            return load_model(MODEL_PATH)
        except:
            pass
    st.warning("Pre-trained model not found or corrupted. Training a new one...")
    return build_and_train_model(X_train, y_train)

# ==================== SIDEBAR ====================
st.sidebar.header("Settings")
period_options = {"1 Month": "1mo", "6 Months": "6mo", "1 Year": "1y", "5 Years": "5y", "Max": "max"}
period = st.sidebar.selectbox("Time Period", options=list(period_options.keys()), index=2)
period_value = period_options[period]

default_tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "AMZN", "META", "NFLX"]
user_input = st.sidebar.selectbox("Stock Ticker", options=default_tickers, index=0).upper()
user_input = st.sidebar.text_input("Or enter custom ticker:", value=user_input).upper()

indicator = st.sidebar.radio("Indicator", ["SMA", "EMA", "MACD", "RSI", "Bollinger Bands"])
return_type = st.sidebar.radio("Returns", ["Daily Returns", "Cumulative Returns"])

# ==================== MAIN APP ====================
try:
    info = get_ticker_info_safe(user_input)
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image(get_logo_url(user_input), width=120)
    with col2:
        st.header(f"**{info['longName']} ({user_input})**")
        st.info(info["longBusinessSummary"])

    df = fetch_stock_data(user_input, period_value)
    df = add_indicators(df)
    df["Date"] = df.index

    # Charts
    st.subheader("Candlestick Chart")
    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                         low=df['Low'], close=df['Close'])])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Price & Volume + Indicator")
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05, subplot_titles=("Price", "Volume", indicator, return_type),
                        row_heights=[0.5, 0.2, 0.2, 0.2])

    fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="OHLC"), row=1, col=1)
    fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name="Volume"), row=2, col=1)

    # Indicator trace
    if indicator == "SMA":
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_50'], name="SMA 50"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_200'], name="SMA 200"), row=3, col=1)
    elif indicator == "EMA":
        fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA_20'], name="EMA 20"), row=3, col=1)
    elif indicator == "MACD":
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], name="MACD"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Signal_Line'], name="Signal"), row=3, col=1)
    elif indicator == "RSI":
        fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name="RSI"), row=3, col=1)
    elif indicator == "Bollinger Bands":
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Upper'], name="Upper"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Lower'], name="Lower", fill='tonexty'), row=3, col=1)

    # Returns
    rets = df['Close'].pct_change()
    if return_type == "Cumulative Returns":
        rets = (1 + rets).cumprod() - 1
    fig.add_trace(go.Scatter(x=df['Date'], y=rets, name=return_type), row=4, col=1)

    fig.update_layout(height=800, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # ==================== LSTM PREDICTION ====================
    st.subheader("🔮 30-Day Price Prediction (LSTM)")

    close_prices = df[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    # Prepare sequences
    def create_dataset(data, time_step=100):
        X, y = [], []
        for i in range(len(data) - time_step):
            X.append(data[i:i + time_step, 0])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)

    time_step = 100
    X, y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    if len(X) == 0:
        st.error("Not enough data to train LSTM (need >100 days)")
    else:
        model = load_or_train_model(X, y)

        # Predict next 30 days
        last_100 = scaled_data[-time_step:]
        predictions = []
        current = last_100.copy()

        for _ in range(30):
            pred = model.predict(current.reshape(1, time_step, 1), verbose=0)
            predictions.append(pred[0, 0])
            current = np.append(current[1:], pred, axis=0)

        pred_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

        # Plot prediction
        future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=30, freq='B')
        hist = df['Close'].tail(100)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index[-100:], y=hist, name="Historical"))
        fig.add_trace(go.Scatter(x=future_dates, y=pred_prices, name="Predicted", line=dict(dash="dot")))
        fig.update_layout(title=f"{user_input} – Next 30 Trading Days Forecast", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)

        st.success(f"**Predicted price in 30 days: ${pred_prices[-1]:.2f}** (+{(pred_prices[-1]/df['Close'].iloc[-1]-1)*100:.2f}%)")

except Exception as e:
    st.error(f"Unexpected error: {str(e)}")
    logging.error(f"App crash: {e}")
