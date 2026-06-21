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
| **🔮 Price Prediction** | Universal pre-trained checkpoint (XGBoost+LightGBM), instant inference on any ticker, GBM cone CI |
| **⚠️ Risk Analysis** | VaR/CVaR (3 methods), Monte Carlo GBM, CAPM, drawdown, monthly heatmap, Q-Q plot |
| **💼 Portfolio Tracker** | Multi-stock P&L, correlation matrix, risk/return scatter, allocation donut |
| **🔍 Screener** | NSE Nifty 50 + US S&P 500, P/E / Beta / RSI / signal filters, CSV export |
| **⚖️ Compare** | Side-by-side price, spread, indicators, risk table (green = winner), drawdown overlay |
| **🌍 Market Overview** | Global indices strip, NSE/US top movers, sector heatmaps, VIX |
| **⭐ Watchlist** | Add positions with targets & stop-loss, live P&L, alerts, sparklines |

---

## 🔮 How Prediction Works — Universal Checkpoint Architecture

**The old approach (and its problems):** Training a fresh model from scratch on
every page load, fitted to only ~250–1500 rows of ONE stock's history. This
was slow (20–90s per request) and prone to overfitting — a model with 50+
features has far too little single-stock data to learn real patterns from.

**The new approach:** Train **one model, once**, on a pooled cross-section of
~40 diverse companies (different sectors, different price scales). Every
page load then just **loads that checkpoint** (instant) and runs inference
on whichever ticker you ask about — including tickers the model has never
seen before.

This works because every feature is **scale-free**:
- RSI, Stochastic, Williams %R, MFI, CCI — already bounded oscillators
- MACD — normalised by price (`MACD / Close`, not raw price units)
- Moving averages — expressed as **distance from price** (`Close/SMA - 1`),
  never as a raw price level
- Volatility — `ATR / Close`, annualised % — never raw dollar/rupee ATR
- OBV — rate-of-change %, never the raw cumulative level

A model trained this way sees **identical features** for a ₹10 stock and a
₹10,000 stock following the same relative price dynamics — verified in
testing: 56/56 features bit-for-bit identical across a 1000× price
difference. That's what makes one checkpoint genuinely apply to any company.

**Mechanics:**
1. Universal model predicts tomorrow's expected **log return** from 56
   scale-free features
2. Multi-day forecast = compound forward: `Price_t = Price_0 × exp(t × r)`
3. **Confidence interval** = GBM volatility cone computed from *this specific
   ticker's own* historical volatility: `P₀ × exp(±1.28σ√t)` — width grows
   as √t, consistent with random-walk theory
4. **Walk-forward evaluation** — with the checkpoint loaded, this is pure
   inference across rolling windows (no retraining), so backtesting is also
   near-instant

**Training the checkpoint:**
```bash
python scripts/train_universal_model.py
```
or use the **🔧 Train / Retrain Universal Model** panel directly in the
Price Prediction page. Takes ~2–5 minutes, requires real internet access to
Yahoo Finance. See `models/README.md` for details and how to make it
persist across Streamlit Cloud redeploys.

**Fallback mode:** If no checkpoint has been trained yet, the app
automatically falls back to a small, fast single-ticker model so it never
crashes — clearly labeled in the UI as fallback mode, since (like the old
approach) it's more overfitting-prone on small datasets.

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
scikit-learn  xgboost  lightgbm  scipy  statsmodels  matplotlib
```

All dependencies are lightweight — no TensorFlow/PyTorch required. The
prediction engine runs entirely on XGBoost + LightGBM, which handle
tabular, scale-free technical features extremely well and deploy fast on
Streamlit Cloud's free tier without memory concerns. `matplotlib` is needed
only for the colour-graded risk tables (`pandas.Styler.background_gradient`)
on the Portfolio and Watchlist pages.

---

## ⚙️ GitHub → Streamlit Cloud (auto-deploy)

1. Push this folder to a GitHub repo
2. Go to **share.streamlit.io** → New app → select repo → `app.py`
3. Every `git push` auto-redeploys in ~30–60 seconds

To edit directly in GitHub: click any file → ✏️ pencil icon → commit → done.

**First-time setup:** the `models/` folder ships empty (see `models/README.md`).
Train the universal prediction checkpoint once via the **🔧 Train Universal
Model** panel on the Price Prediction page, then commit the generated
`models/*.json` / `*.txt` files so it persists across redeploys. Until
trained, prediction still works via an automatic per-ticker fallback —
just slower and less accurate.

---

## 🐛 Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: matplotlib` | Risk tables use `pandas.Styler.background_gradient`, which needs matplotlib | Already in `requirements.txt` — redeploy after pulling latest |
| `TypeError: got multiple values for keyword argument 'xaxis'` (or `'margin'`, `'yaxis'`) | A chart passed the same Plotly layout key twice | Already fixed — all charts now use the `safe_layout()` helper in `utils/charts.py`, which deep-merges instead of colliding |
| Prediction page shows "⚠️ Fallback" badge | No universal checkpoint trained yet | Train it once via the in-app panel or `scripts/train_universal_model.py` |
| Training fails with a connection error | Sandbox/CI environment has no internet access to Yahoo Finance | Run training on Streamlit Cloud or your local machine instead |

---

## ⚠️ Disclaimer

For informational purposes only. Not financial advice.
Data via Yahoo Finance — may be delayed or inaccurate.
