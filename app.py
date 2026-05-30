"""Streamlit stock market analysis and prediction app.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import logging
import os
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler

# TensorFlow/Keras imports are optional at import time so the app can still run
# without prediction if the dependency is unavailable.
try:
    from keras.callbacks import Callback, EarlyStopping
    from keras.layers import Dense, Dropout, LSTM
    from keras.models import Sequential
    from keras.optimizers import Adam
except Exception:  # pragma: no cover - shown in UI when prediction is requested
    Callback = None
    EarlyStopping = None
    Dense = None
    Dropout = None
    LSTM = None
    Sequential = None
    Adam = None

# -----------------------------------------------------------------------------
# App configuration
# -----------------------------------------------------------------------------

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
warnings.filterwarnings("ignore")

APP_DIR = Path(__file__).resolve().parent

DEFAULT_PERIODS = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"]
DEFAULT_INDICATOR_PERIODS = ["6m", "1y", "3y", "5y"]
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "RELIANCE.NS"]
TIME_PERIOD_MAP = {"6m": 126, "1y": 252, "3y": 756, "5y": 1800}

# Local-first paths. These make the app work after cloning the repo. The URL
# fallback still helps if the app is deployed without the CSV files.
RAW_BASE_URL = "https://raw.githubusercontent.com/mayanknagar10/stock-market-analysis-and-prediction/main"
CSV_SOURCES = {
    "period": (APP_DIR / "period.csv", f"{RAW_BASE_URL}/period.csv"),
    "indicators_period": (APP_DIR / "indicators_period.csv", f"{RAW_BASE_URL}/indicators_period.csv"),
    "company_data": (APP_DIR / "company_data.csv", f"{RAW_BASE_URL}/company_data.csv"),
}


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


@st.cache_data(show_spinner=False)
def load_csv_data(name: str) -> pd.DataFrame:
    """Load config CSVs locally first, then from GitHub raw URL fallback."""
    local_path, remote_url = CSV_SOURCES[name]

    try:
        if local_path.exists():
            return pd.read_csv(local_path)
        return pd.read_csv(remote_url)
    except Exception as exc:
        logging.warning("Could not load %s CSV: %s", name, exc)
        return pd.DataFrame()


def dataframe_to_options(df: pd.DataFrame, preferred_columns: list[str], fallback: list[str]) -> list[str]:
    """Convert a config DataFrame into clean selectbox options."""
    if df.empty:
        return fallback

    column = first_existing_column(df, preferred_columns) or df.columns[0]
    options = df[column].dropna().astype(str).str.strip().tolist()
    options = [option for option in options if option]
    return options or fallback


def normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance multi-index columns and keep standard OHLCV names."""
    if df.empty:
        return df

    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    # Remove timezone so Streamlit/Plotly handle dates consistently.
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Downloaded data is missing columns: {', '.join(missing)}")

    return df.dropna(subset=["Open", "High", "Low", "Close"])


@st.cache_data(show_spinner=False, ttl=900)
def fetch_stock_data(ticker: str, period: str, interval: str = "1d") -> pd.DataFrame:
    """Fetch stock data using yfinance with retries.

    Uses Ticker.history first and yf.download second. Avoids deprecated/removed
    kwargs such as show_errors.
    """
    ticker = ticker.strip().upper()
    errors: list[str] = []

    for attempt in range(3):
        try:
            data = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
            if not data.empty:
                return normalize_ohlcv_columns(data)
        except Exception as exc:
            errors.append(f"Ticker.history attempt {attempt + 1}: {exc}")

        try:
            data = yf.download(
                ticker,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
                multi_level_index=False,
            )
            if data is not None and not data.empty:
                return normalize_ohlcv_columns(data)
        except Exception as exc:
            errors.append(f"yf.download attempt {attempt + 1}: {exc}")

        time.sleep(2**attempt)

    logging.warning("All data fetch attempts failed for %s: %s", ticker, " | ".join(errors))
    return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def get_ticker_info(ticker: str) -> dict[str, Any]:
    """Return ticker info with a fast_info fallback."""
    ticker = ticker.strip().upper()
    ticker_obj = yf.Ticker(ticker)

    try:
        info = ticker_obj.info
        if isinstance(info, dict) and info:
            return info
    except Exception as exc:
        logging.warning("ticker.info failed for %s: %s", ticker, exc)

    try:
        fast_info = ticker_obj.fast_info
        return {
            "longName": ticker,
            "previousClose": getattr(fast_info, "previous_close", None),
            "regularMarketPrice": getattr(fast_info, "last_price", None),
            "marketCap": getattr(fast_info, "market_cap", None),
            "fiftyTwoWeekHigh": getattr(fast_info, "year_high", None),
            "fiftyTwoWeekLow": getattr(fast_info, "year_low", None),
        }
    except Exception as exc:
        logging.warning("fast_info failed for %s: %s", ticker, exc)
        return {"longName": ticker}


def format_money(value: Any) -> str:
    if value is None or value == "N/A" or pd.isna(value):
        return "N/A"
    try:
        return f"${float(value):,.2f}"
    except Exception:
        return str(value)


def format_number(value: Any) -> str:
    if value is None or value == "N/A" or pd.isna(value):
        return "N/A"
    try:
        return f"{int(value):,}"
    except Exception:
        return str(value)


# -----------------------------------------------------------------------------
# Indicator functions
# -----------------------------------------------------------------------------


def sma(data: pd.DataFrame, period: int = 100, column: str = "Close") -> pd.Series:
    return data[column].rolling(window=period).mean()


def ema(data: pd.DataFrame, period: int = 20, column: str = "Close") -> pd.Series:
    return data[column].ewm(span=period, adjust=False).mean()


def add_macd(data: pd.DataFrame, column: str = "Close") -> pd.DataFrame:
    data = data.copy()
    short_ema = ema(data, 12, column)
    long_ema = ema(data, 26, column)
    data["MACD"] = short_ema - long_ema
    data["Signal_Line"] = ema(data, 9, "MACD")
    return data


def add_rsi(data: pd.DataFrame, period: int = 14, column: str = "Close") -> pd.DataFrame:
    data = data.copy()
    delta = data[column].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    data["RSI"] = 100 - (100 / (1 + rs))
    return data


def add_bollinger_bands(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    typical_price = (data["Close"] + data["Low"] + data["High"]) / 3
    rolling_mean = typical_price.rolling(20).mean()
    rolling_std = typical_price.rolling(20).std(ddof=0)
    data["BOLU"] = rolling_mean + 2 * rolling_std
    data["BOLD"] = rolling_mean - 2 * rolling_std
    return data


def add_all_indicators(data: pd.DataFrame) -> pd.DataFrame:
    data = add_macd(data)
    data = add_rsi(data)
    data = add_bollinger_bands(data)
    data["SMA_50"] = sma(data, 50)
    data["SMA_100"] = sma(data, 100)
    data["SMA_200"] = sma(data, 200)
    data["EMA"] = ema(data, 20)
    data["Date"] = data.index
    return data


# -----------------------------------------------------------------------------
# Plotting functions
# -----------------------------------------------------------------------------


def get_stock_price_fig(df: pd.DataFrame, indicator: str, returns_mode: str) -> go.Figure:
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_width=[0.1, 0.2, 0.1, 0.3],
        subplot_titles=("", "", indicator, f"{returns_mode} %"),
    )

    fig.add_trace(
        go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            showlegend=False,
            name="Price",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(go.Bar(x=df["Date"], y=df["Volume"], opacity=0.5, showlegend=False, name="Volume"), row=2, col=1)

    if indicator == "RSI":
        fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI"], mode="lines", name="RSI", showlegend=False), row=3, col=1)
    elif indicator == "SMA":
        fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA_50"], mode="lines", name="SMA 50", showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA_200"], mode="lines", name="SMA 200", showlegend=False), row=3, col=1)
    elif indicator == "EMA":
        fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA"], mode="lines", name="EMA", showlegend=False), row=3, col=1)
    elif indicator == "MACD":
        fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD"], mode="lines", name="MACD", showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Signal_Line"], mode="lines", name="Signal", showlegend=False), row=3, col=1)
    elif indicator == "Bollinger Bands":
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Close", showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=df["Date"], y=df["BOLU"], mode="lines", name="Upper Band", showlegend=False), row=3, col=1)
        fig.add_trace(
            go.Scatter(x=df["Date"], y=df["BOLD"], mode="lines", name="Lower Band", fill="tonexty", showlegend=False),
            row=3,
            col=1,
        )

    returns = df["Close"].pct_change()
    if returns_mode == "Cumulative Returns":
        returns = (returns + 1).cumprod() - 1

    fig.add_trace(go.Scatter(x=df["Date"], y=returns, mode="lines", showlegend=False, name=returns_mode), row=4, col=1)
    fig.update_layout(height=700, xaxis_rangeslider_visible=False, margin=dict(b=20, t=30, l=20, r=20))
    return fig


def line_chart(title: str, x: Any, series: dict[str, Any], y_title: str = "Price") -> go.Figure:
    fig = go.Figure()
    for name, y in series.items():
        fig.add_trace(go.Scatter(x=x, y=y, name=name, mode="lines"))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title=y_title, hovermode="x unified", height=450)
    return fig


# -----------------------------------------------------------------------------
# Ticker-specific LSTM prediction
# -----------------------------------------------------------------------------


def create_sequence_dataset(dataset: np.ndarray, step: int) -> tuple[np.ndarray, np.ndarray]:
    """Create supervised learning sequences from a scaled one-column array."""
    x_values, y_values = [], []
    for i in range(len(dataset) - step):
        x_values.append(dataset[i : i + step, 0])
        y_values.append(dataset[i + step, 0])
    return np.array(x_values), np.array(y_values)


def build_fast_lstm_model(sequence_length: int) -> Any:
    """Build a small LSTM that trains quickly inside Streamlit."""
    model = Sequential(
        [
            LSTM(32, return_sequences=True, input_shape=(sequence_length, 1)),
            Dropout(0.15),
            LSTM(16),
            Dense(8, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=0.001), loss="huber")
    return model


class StreamlitTrainingProgress(Callback if Callback is not None else object):
    """Keras callback that updates a Streamlit progress bar and ETA text."""

    def __init__(self, total_epochs: int, progress_bar: Any, status_box: Any) -> None:
        super().__init__()
        self.total_epochs = max(total_epochs, 1)
        self.progress_bar = progress_bar
        self.status_box = status_box
        self.start_time = time.time()
        self.epoch_start = self.start_time

    def on_epoch_begin(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        self.epoch_start = time.time()
        self.status_box.info(f"Training epoch {epoch + 1}/{self.total_epochs}...")

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        completed = epoch + 1
        elapsed = time.time() - self.start_time
        avg_epoch_time = elapsed / completed
        remaining_epochs = max(self.total_epochs - completed, 0)
        eta_seconds = int(avg_epoch_time * remaining_epochs)
        loss = (logs or {}).get("loss")
        val_loss = (logs or {}).get("val_loss")
        progress = min(completed / self.total_epochs, 1.0)
        self.progress_bar.progress(progress)
        loss_text = f"loss={loss:.5f}" if isinstance(loss, (int, float)) else "loss=N/A"
        val_text = f", val_loss={val_loss:.5f}" if isinstance(val_loss, (int, float)) else ""
        self.status_box.info(
            f"Training epoch {completed}/{self.total_epochs} complete — {loss_text}{val_text}. "
            f"Approx. time remaining: {eta_seconds}s"
        )


def forecast_with_recursive_lstm(
    model: Any,
    last_window: np.ndarray,
    scaler: MinMaxScaler,
    forecast_days: int,
) -> np.ndarray:
    """Generate a recursive forecast from the most recent scaled sequence."""
    window = last_window.astype(float).flatten().tolist()
    predictions: list[float] = []

    for _ in range(forecast_days):
        model_input = np.array(window[-len(last_window) :]).reshape(1, len(last_window), 1)
        next_scaled = float(model.predict(model_input, verbose=0)[0, 0])
        # Keep recursive values in a sane normalized range to avoid explosive output.
        next_scaled = float(np.clip(next_scaled, -0.25, 1.25))
        window.append(next_scaled)
        predictions.append(next_scaled)

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()


def render_lstm_prediction(df: pd.DataFrame, ticker: str) -> None:
    st.subheader("Ticker-Specific LSTM Forecast")
    st.caption(
        "This trains a small model inside the Streamlit app for the selected ticker. "
        "It is still an educational estimate, not a guaranteed market prediction."
    )

    if any(obj is None for obj in [Callback, EarlyStopping, Dense, Dropout, LSTM, Sequential, Adam]):
        st.warning("TensorFlow/Keras is not available, so prediction is disabled.")
        return

    close = df[["Close"]].dropna().copy()
    if len(close) < 260:
        st.warning("Not enough historical data for training. Select at least 2y or 5y for better results.")
        return

    with st.expander("Training settings", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            forecast_days = st.slider("Forecast days", min_value=7, max_value=60, value=30, step=1)
        with col2:
            epochs = st.slider("Training epochs", min_value=5, max_value=30, value=12, step=1)
        with col3:
            sequence_length = st.slider("Lookback days", min_value=30, max_value=120, value=60, step=10)
        st.caption("For faster training, keep epochs around 8–12 and lookback around 60 days.")

    # Fit scaler only on training data to avoid future leakage during backtesting.
    raw_prices = close.values.astype(float)
    train_cutoff = int(len(raw_prices) * 0.80)
    if train_cutoff <= sequence_length + 20:
        st.warning("Not enough training rows after split. Select a longer period.")
        return

    train_prices = raw_prices[:train_cutoff]
    test_prices = raw_prices[train_cutoff - sequence_length :]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train_prices)
    scaled_test = scaler.transform(test_prices)

    x_train, y_train = create_sequence_dataset(scaled_train, sequence_length)
    x_test, y_test = create_sequence_dataset(scaled_test, sequence_length)

    if x_train.size == 0 or x_test.size == 0:
        st.warning("Not enough sequence data for training/testing. Select a longer period or shorter lookback.")
        return

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    train_button = st.button("Train model and forecast", type="primary", use_container_width=True)
    if not train_button:
        st.info("Click **Train model and forecast** to train a fresh model for this ticker.")
        return

    progress_bar = st.progress(0)
    status_box = st.empty()

    try:
        model = build_fast_lstm_model(sequence_length)
        callbacks = [
            StreamlitTrainingProgress(epochs, progress_bar, status_box),
            EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ]

        start = time.time()
        history = model.fit(
            x_train,
            y_train,
            validation_split=0.15,
            epochs=epochs,
            batch_size=32,
            verbose=0,
            callbacks=callbacks,
            shuffle=False,
        )
        elapsed = time.time() - start
        progress_bar.progress(1.0)
        status_box.success(f"Training finished in {elapsed:.1f}s. Generating forecast...")

        # Backtest on the most recent 20% of selected data.
        predicted_test_scaled = model.predict(x_test, verbose=0)
        predicted_test = scaler.inverse_transform(predicted_test_scaled.reshape(-1, 1)).flatten()
        actual_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        errors = predicted_test - actual_test
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors**2)))
        mape = float(np.mean(np.abs(errors / np.where(actual_test == 0, np.nan, actual_test))) * 100)

        # Future forecast from the latest real prices, using the same training scaler.
        scaled_all = scaler.transform(raw_prices)
        last_window = scaled_all[-sequence_length:].flatten()
        future_prices = forecast_with_recursive_lstm(model, last_window, scaler, forecast_days)

        current_price = float(close["Close"].iloc[-1])
        predicted_price = float(future_prices[-1])
        price_change = predicted_price - current_price
        price_change_pct = (price_change / current_price) * 100 if current_price else 0.0

        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        metric_col1.metric("Current Price", format_money(current_price))
        metric_col2.metric(f"Forecast Price {forecast_days}D", format_money(predicted_price), f"{price_change:,.2f}")
        metric_col3.metric("Forecast Change", f"{price_change_pct:,.2f}%")
        metric_col4.metric("Backtest MAPE", f"{mape:,.2f}%")

        st.caption(
            f"Backtest on recent data: MAE {mae:,.2f}, RMSE {rmse:,.2f}, MAPE {mape:,.2f}%. "
            "Lower values mean the model matched recent historical data better."
        )

        backtest_index = close.index[-len(actual_test) :]
        future_index = pd.bdate_range(start=close.index[-1] + pd.Timedelta(days=1), periods=forecast_days)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=close.index[-180:],
                y=close["Close"].tail(180),
                name="Actual recent close",
                mode="lines",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=backtest_index,
                y=predicted_test,
                name="Backtest prediction",
                mode="lines",
                line=dict(dash="dot"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=future_index,
                y=future_prices,
                name=f"Forecast next {forecast_days} trading days",
                mode="lines",
                line=dict(dash="dash"),
            )
        )
        fig.update_layout(
            title=f"{ticker} ticker-specific LSTM forecast",
            xaxis_title="Date",
            yaxis_title="Price",
            hovermode="x unified",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

        forecast_table = pd.DataFrame({"Date": future_index, "Forecast Close": future_prices})
        st.dataframe(forecast_table, use_container_width=True)

        if mape > 8:
            st.warning(
                "This ticker's recent backtest error is high, so treat the forecast as weak. "
                "Try a longer period, fewer forecast days, or more epochs."
            )
        else:
            st.success("Recent backtest error is reasonably low for an educational model, but this is still not financial advice.")

    except Exception as exc:
        logging.exception("Ticker-specific LSTM training failed")
        st.error(f"Training or prediction failed: {exc}")
        st.info("Try selecting a longer period, reducing lookback days, or using a major ticker with enough price history.")


# -----------------------------------------------------------------------------
# Main app
# -----------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Stock Market Analysis", layout="wide")
    st.title("Stock Market Analysis and Prediction")
    st.caption("Educational project using Yahoo Finance data via yfinance. Not financial advice.")
    st.divider()

    st.sidebar.subheader("Query parameters")
    debug_mode = st.sidebar.checkbox("Debug mode", value=False)

    period_options = dataframe_to_options(load_csv_data("period"), ["period", "Period", "time", "Time"], DEFAULT_PERIODS)
    indicator_period_options = dataframe_to_options(
        load_csv_data("indicators_period"),
        ["indicators_time", "indicator_time", "period", "Period"],
        DEFAULT_INDICATOR_PERIODS,
    )
    ticker_options = dataframe_to_options(
        load_csv_data("company_data"),
        ["Symbol", "symbol", "Ticker", "ticker", "Company Symbol"],
        DEFAULT_TICKERS,
    )

    selected_period = st.sidebar.selectbox("Time Period", period_options, index=period_options.index("1y") if "1y" in period_options else 0)
    indicator_period_input = st.sidebar.selectbox("Time Period for indicators", indicator_period_options)
    selected_ticker = st.sidebar.selectbox("Enter Stock Ticker", ticker_options)
    custom_ticker = st.sidebar.text_input("Or type custom ticker", "")
    ticker = (custom_ticker or selected_ticker).strip().upper()

    st.sidebar.caption("For Indian NSE stocks, try the `.NS` suffix, e.g. `RELIANCE.NS`.")
    selected_indicator = st.sidebar.radio("Indicators", ("SMA", "EMA", "MACD", "RSI", "Bollinger Bands"))
    selected_returns = st.sidebar.radio("Returns", ("Daily Returns", "Cumulative Returns"))
    show_prediction = st.sidebar.checkbox("Show LSTM prediction", value=True)

    with st.spinner("Fetching stock data..."):
        df = fetch_stock_data(ticker, selected_period)

    if debug_mode:
        st.write("Debug data shape:", df.shape)
        st.write("Debug columns:", df.columns.tolist() if not df.empty else [])

    if df.empty:
        st.error(f"No data found for ticker `{ticker}`.")
        st.info("Check the symbol on Yahoo Finance, try a longer time period, or use suffixes like `.NS` for NSE stocks.")
        st.stop()

    info = get_ticker_info(ticker)
    company_name = info.get("longName") or info.get("shortName") or ticker

    st.header(company_name)
    summary = info.get("longBusinessSummary")
    if summary:
        st.info(summary)

    st.subheader("Candlestick Chart")
    candlestick_fig = go.Figure(
        data=[go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"])]
    )
    candlestick_fig.update_layout(title=f"{ticker} Candlestick Chart", xaxis_title="Date", yaxis_title="Price", xaxis_rangeslider_visible=False, height=500)
    st.plotly_chart(candlestick_fig, use_container_width=True)

    st.subheader("Recent Data")
    recent = df.copy()
    recent.insert(0, "Date", recent.index)
    st.dataframe(recent[["Date", "Open", "High", "Low", "Close", "Volume"]].tail(10), use_container_width=True)

    st.subheader("Fundamentals")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("52 Week Range", f"{format_money(info.get('fiftyTwoWeekLow'))} - {format_money(info.get('fiftyTwoWeekHigh'))}")
        st.metric("Day's Range", f"{format_money(info.get('dayLow'))} - {format_money(info.get('dayHigh'))}")
        st.metric("Average Volume", format_number(info.get("averageVolume")))
        st.metric("Volume", format_number(info.get("volume")))
    with col2:
        st.metric("Market Cap", format_money(info.get("marketCap")))
        pe = info.get("trailingPE")
        st.metric("PE Ratio", f"{pe:.2f}" if isinstance(pe, (int, float)) else "N/A")
        eps = info.get("trailingEps")
        st.metric("EPS", f"{eps:.2f}" if isinstance(eps, (int, float)) else "N/A")
        st.metric("Quote Price", format_money(info.get("regularMarketPrice") or info.get("previousClose") or df["Close"].iloc[-1]))

    st.subheader("Stock Price Chart")
    st.plotly_chart(
        line_chart(
            f"{ticker} Stock Prices",
            df.index,
            {"Open": df["Open"], "Close": df["Close"], "High": df["High"], "Low": df["Low"]},
        ),
        use_container_width=True,
    )

    st.subheader("Volume Traded Chart")
    volume_fig = go.Figure(data=[go.Bar(x=df.index, y=df["Volume"], name="Volume")])
    volume_fig.update_layout(title=f"{ticker} Volume Traded", xaxis_title="Date", yaxis_title="Volume", height=400)
    st.plotly_chart(volume_fig, use_container_width=True)

    df_indicators = add_all_indicators(df)
    indicator_window = TIME_PERIOD_MAP.get(indicator_period_input, 252)

    st.subheader("Technical Indicators and Returns")
    st.plotly_chart(get_stock_price_fig(df_indicators.tail(indicator_window), selected_indicator, selected_returns), use_container_width=True)

    st.subheader("Bollinger Bands")
    bollinger_fig = go.Figure()
    bollinger_fig.add_trace(go.Scatter(x=df_indicators.index, y=df_indicators["Close"], name="Close"))
    bollinger_fig.add_trace(go.Scatter(x=df_indicators.index, y=df_indicators["BOLU"], name="Upper Band"))
    bollinger_fig.add_trace(go.Scatter(x=df_indicators.index, y=df_indicators["BOLD"], name="Lower Band", fill="tonexty"))
    bollinger_fig.update_layout(title="Bollinger Bands", xaxis_title="Date", yaxis_title="Price", height=450)
    st.plotly_chart(bollinger_fig, use_container_width=True)

    st.subheader("Moving Averages")
    st.plotly_chart(
        line_chart(
            f"{ticker} - 100 & 200 Day Moving Averages",
            df_indicators.index,
            {"Close": df_indicators["Close"], "100MA": df_indicators["SMA_100"], "200MA": df_indicators["SMA_200"]},
        ),
        use_container_width=True,
    )

    if show_prediction:
        render_lstm_prediction(df, ticker)

    st.divider()
    st.markdown("Made with Streamlit | Data source: Yahoo Finance via yfinance")
    st.caption("Disclaimer: This tool is for educational purposes only. Not financial advice.")


if __name__ == "__main__":
    main()
