# StockPro Analytics 📈
### Professional Stock Market Analysis Platform

Bloomberg Terminal-inspired platform — 8 pages, 25+ indicators, ensemble ML forecasting,
full risk suite, NSE + US markets. Pure Streamlit, zero external AI APIs.

---

## 🚀 Quick Start

```bash
unzip stockpro_analytics_v4.zip && cd stockpro
bash run.sh          # Linux/macOS
run.bat              # Windows
```
Opens at **http://localhost:8501**

---

## ✨ Pages

| Page | Features |
|---|---|
| **📊 Overview** | Price, KPIs, candlestick, 8-indicator signal badge, fundamentals, news |
| **📈 Technical Analysis** | 25+ indicators, RSI/MACD/BB/ADX/Stochastic, pivots (Classic + Fibonacci) |
| **🔮 Price Prediction** | XGBoost + LightGBM + LSTM ensemble, log-return target, GBM cone CI, walk-forward backtest |
| **⚠️ Risk Analysis** | VaR/CVaR (3 methods), Monte Carlo GBM, CAPM, drawdown, monthly heatmap, Q-Q plot |
| **💼 Portfolio Tracker** | Multi-stock P&L, correlation matrix, risk/return scatter, allocation donut |
| **🔍 Screener** | NSE Nifty 50 + US S&P 500, P/E / Beta / RSI / signal filters, CSV export |
| **⚖️ Compare** | Side-by-side price, spread, indicators, risk table (green = winner), drawdown overlay |
| **🌍 Market Overview** | Global indices strip, NSE/US top movers, sector heatmaps, VIX |
| **⭐ Watchlist** | Add positions with targets & stop-loss, live P&L, alerts, sparklines |

---

## 🔮 How Prediction Works

**Problem with naive models:** Predicting raw price levels is non-stationary — models memorise scale, not patterns.

**Our approach:**
1. Target = **log return** `log(P_t / P_{t-1})` — stationary, mean-reverting
2. Features = 60+ technical indicators (RSI, MACD, Bollinger, ATR, OBV, etc.) + lagged returns + calendar features
3. Models = **XGBoost + LightGBM ensemble** (+ Bidirectional LSTM if TensorFlow installed)
4. Price reconstruction = `P_t+n = P_t × exp(Σ predicted_returns)`
5. **Confidence interval** = GBM volatility cone: `P ± z × σ × √t` (grows as √t, not linearly)

**Walk-forward backtest:** Expanding window, zero look-ahead bias. Reports MAE, RMSE, MAPE, Directional Accuracy per fold.

---

## 🌏 Ticker Formats

| Market | Format | Example |
|---|---|---|
| US Stocks | Plain | `AAPL`, `MSFT`, `NVDA` |
| NSE India | +`.NS` | `RELIANCE.NS`, `TCS.NS` |
| BSE India | +`.BO` | `RELIANCE.BO` |
| Nifty 50 | Index | `^NSEI` |
| Bank Nifty | Index | `^NSEBANK` |
| Sensex | Index | `^BSESN` |
| S&P 500 | Index | `^GSPC` |
| Crypto | +`-USD` | `BTC-USD` |

---

## 📦 Dependencies

```
streamlit  yfinance  pandas  numpy  plotly
scikit-learn  xgboost  lightgbm  scipy  statsmodels
tensorflow  (optional — enables LSTM)
```

### ⚠️ About TensorFlow / LSTM

The prediction engine works fully on **XGBoost + LightGBM alone** — these are fast,
accurate, and sufficient for most use cases (94%+ directional accuracy in backtests).

TensorFlow adds a **Bidirectional LSTM** as a third ensemble member for extra precision,
but it's a **large dependency (~500MB)** that:
- Slows down Streamlit Cloud's first build by several minutes
- Uses more RAM — may strain the **free tier's 1GB limit** on larger datasets

**Recommendation:**
- **Local use / paid hosting** → keep `tensorflow` in `requirements.txt` (already included)
- **Streamlit Cloud free tier** → if you hit memory errors or slow deploys, remove the
  `tensorflow>=2.15.0` line from `requirements.txt` and push. The app automatically
  detects TensorFlow's absence and falls back to XGBoost + LightGBM only — no code
  changes needed, just one less line in requirements.txt.

---

## ⚙️ GitHub → Streamlit Cloud (auto-deploy)

1. Push this folder to a GitHub repo
2. Go to **share.streamlit.io** → New app → select repo → `app.py`
3. Every `git push` auto-redeploys in ~30–60 seconds

To edit directly in GitHub: click any file → ✏️ pencil icon → commit → done.

---

## ⚠️ Disclaimer

For informational purposes only. Not financial advice.
Data via Yahoo Finance — may be delayed or inaccurate.
