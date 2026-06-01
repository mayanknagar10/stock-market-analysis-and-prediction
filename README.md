# StockPro Analytics 📈
### Professional Stock Market Analysis & Prediction Platform

A Bloomberg Terminal-inspired stock analysis platform built entirely with Python and Streamlit.
No external AI APIs. No paid data subscriptions. Pure open-source stack.

---

## ✨ Features

### 📊 Overview Dashboard
- Real-time price header with intraday change
- 12-metric KPI row (market cap, P/E, beta, yield, volatility, YTD return)
- Interactive candlestick chart with overlay controls (SMA, EMA, Bollinger Bands)
- 8-indicator composite signal badge (STRONG BUY → STRONG SELL)
- Full fundamentals table (revenue, margins, ROE, debt/equity)
- Live news feed with timestamps

### 📈 Technical Analysis
- **25+ indicators** across 4 families: Trend, Momentum, Volatility, Volume
- Trend: SMA/EMA ribbon, VWAP, ADX (+DI / -DI), Parabolic SAR, Donchian Channels
- Momentum: RSI, MACD, Stochastic, Williams %R, CCI, ROC, Momentum
- Volatility: Bollinger Bands, Keltner Channels, ATR, Historical Volatility
- Volume: OBV, MFI, CMF, Volume Ratio
- Classic & Fibonacci **Pivot Point** levels with distance from current price
- Interactive indicator parameters (RSI period, MACD fast/slow, BB window/std)

### 🔮 Price Prediction
- **Ensemble model**: XGBoost + LightGBM with 60+ engineered features
- Quantile regression for 80% confidence intervals (q10/q90)
- **Walk-forward backtesting** — zero data leakage, expanding window
- Per-fold metrics: MAE, RMSE, MAPE, Directional Accuracy
- Feature importance chart (top 20 features)
- Day-by-day forecast table with % change vs current price

### ⚠️ Risk Analysis
- **VaR / CVaR** with 3 methods: Historical, Parametric (Gaussian), Cornish-Fisher
- 95% and 99% confidence levels + dollar-value impact
- **Monte Carlo** simulation: Geometric Brownian Motion, 100–1000 paths
- Outcome distribution with percentile statistics
- **CAPM**: Beta, Jensen's Alpha, R², Treynor Ratio, Information Ratio
- Rolling 63-day Beta chart
- Drawdown underwater chart + duration analysis
- Monthly returns heatmap
- Rolling Sharpe and rolling Return/Volatility charts
- Return distribution with Normal Q-Q plot

### 💼 Portfolio Tracker
- Multi-stock performance comparison (up to 10 positions)
- Custom weights + auto-normalisation
- Absolute P&L tracking with stacked contribution chart
- Portfolio-level Sharpe, Sortino, Max Drawdown, VaR
- Correlation heatmap + rolling pairwise correlation
- Risk/Return scatter map (colour = Sharpe ratio)
- Per-position risk-adjusted performance table with pandas Styler

### 🔍 Stock Screener
- Scan up to 50 S&P 500 stocks (or a custom watchlist)
- **Fundamental filters**: P/E range, max beta, min dividend yield
- **Technical filters**: RSI range, MA trend, composite signal
- Colour-coded results table with 1D / 1M / 3M returns
- Signal distribution, sector breakdown, RSI histogram charts
- Opportunity map: RSI vs Volatility scatter
- CSV export

### ⚖️ Stock Comparison
- Side-by-side chart (normalised or absolute)
- Return spread: cumulative and daily bar chart
- Tabbed indicator panels: RSI, MACD, Volatility, Volume
- Risk metrics head-to-head table (green = winner)
- Full fundamentals table side by side
- Signal summary per stock
- Drawdown overlay chart
- Rolling 30-day correlation with Pearson + Spearman r

---

## 🏗️ Architecture

```
stockpro/
├── app.py                          # Main overview page
├── requirements.txt
├── .streamlit/
│   └── config.toml                 # Dark theme
├── core/
│   ├── data_fetcher.py             # Yahoo Finance + caching
│   ├── indicators.py               # 25+ indicators, signal engine
│   ├── models.py                   # XGBoost + LightGBM ensemble
│   └── risk_metrics.py             # VaR, CVaR, CAPM, Monte Carlo
├── pages/
│   ├── 1_📈_Technical_Analysis.py
│   ├── 2_🔮_Price_Prediction.py
│   ├── 3_⚠️_Risk_Analysis.py
│   ├── 4_💼_Portfolio.py
│   ├── 5_🔍_Screener.py
│   └── 6_⚖️_Compare.py
└── utils/
    ├── charts.py                   # Plotly chart library (dark theme)
    └── helpers.py                  # CSS, formatters, HTML components
```

---

## 🚀 Quick Start

### 1. Clone / unzip the project
```bash
cd stockpro
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate.bat       # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run
```bash
streamlit run app.py
```
The app opens at **http://localhost:8501**

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | UI framework |
| `yfinance` | Market data (Yahoo Finance) |
| `pandas / numpy` | Data processing |
| `plotly` | Interactive charts |
| `scikit-learn` | Scaling, metrics, Ridge baseline |
| `xgboost` | Primary tree model |
| `lightgbm` | Secondary tree model |
| `scipy` | Statistical tests, VaR, QQ-plot |
| `statsmodels` | ARIMA baseline (optional) |

> **TensorFlow / Keras** is an optional dependency. If installed, a Bidirectional LSTM
> with MC Dropout is available. The platform degrades gracefully without it.

---

## ⚙️ Configuration

Edit `.streamlit/config.toml` to change theme colours, port, or upload size.
Edit `core/risk_metrics.py` → `RISK_FREE_RATE` to change the default risk-free rate.

---

## ⚠️ Disclaimer

This platform is for **informational and educational purposes only**.
It does not constitute financial advice. Always consult a qualified financial
professional before making investment decisions.

Data is sourced from Yahoo Finance and may be delayed, inaccurate, or incomplete.
Model forecasts are statistical estimates based on historical data and do not
guarantee future performance.
