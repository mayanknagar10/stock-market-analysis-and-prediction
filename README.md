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
| **📊 Strategy Backtester** | Vectorized backtesting (vectorbt) — MA crossover, RSI reversion, MACD, Bollinger, Donchian — full metrics + trade log |
| **🧮 Factor Analysis** | Fama-French factor exposures (free public data) + quant factor screening (Value/Momentum/Quality/Low-Vol) |
| **🤖 Insights** | News sentiment (offline NLP), SEC filing sentiment, rule-based Q&A assistant, personalized recommendations |

---

## 🧠 Mid-Term Features (Phase 3)

All built with **zero external accounts, zero API keys** — same philosophy as every prior phase.

**Strategy Backtester** — `core/strategy_backtest.py`, powered by `vectorbt`. Six built-in strategies (MA Crossover, RSI Mean Reversion, MACD Signal Cross, Bollinger Band Bounce, Donchian Breakout, Buy & Hold benchmark), each with tunable parameters and a brute-force grid-search optimizer. Reports CAGR, Sharpe, Sortino, Calmar, max drawdown, win rate, profit factor, and a full downloadable trade log — with realistic fees and slippage applied per trade.

**Factor Analysis** — `core/factor_models.py`. Two tools:
- *Factor exposures*: regresses a stock's monthly returns against Fama-French factors (Market, Size, Value, Profitability, Investment) using free public data from Kenneth French's Dartmouth data library (via `pandas_datareader`, no key needed). Reports annualised alpha, factor betas, t-stats, and R². Verified against synthetic data with known true betas — the regression recovers them within a few percentage points.
- *Quant factor screening*: ranks a universe of stocks by Value (inverse P/E), Momentum (12-1 month return), Quality (ROE + margins), and Low-Volatility — the standard building blocks of quantitative equity investing — using data already fetched elsewhere in the app.

**Insights** — sentiment + alternative data + assistant, three features in one page:
- *News sentiment*: `core/sentiment.py` uses VADER (lexicon-based, fully offline — no API, no rate limit) with a finance-specific vocabulary extension (upgrade/downgrade/beat/miss/etc.) layered on top of the general-purpose base dictionary.
- *SEC filing sentiment*: same VADER engine applied to EDGAR filing excerpts (US tickers only — NSE/BSE file with SEBI, not the SEC).
- *Rule-based assistant*: `core/assistant.py` — explicitly **not** an LLM. Pattern-matches question intent (price, RSI, MACD, signal, risk, forecast, sentiment) and answers using the exact same computations the rest of the app already trusts, so there's zero hallucination risk. Trades off open-ended flexibility for that guarantee. The `INTENT_HANDLERS` structure is designed to map directly onto LLM function-calling/tool-use patterns if you add a real LLM API key later.

**Personalization** — `core/personalization.py` extends the local auth system (`data/users.json`) with per-user view history. Logged-in users get sector-based recommendations ("you often view IT stocks — here are others you haven't seen") computed from their own behavior, no collaborative filtering or external ML service required.

**What's intentionally NOT built** (from the original mid-term roadmap): REST/GraphQL API server, broker integrations (Zerodha/IBKR), Twitter/Reddit sentiment (need developer accounts), and microservices/autoscaling infrastructure. These genuinely require either a paid account, a separate backend service, or an infrastructure decision that doesn't fit inside a Streamlit app's architecture — building a fake version would be misleading rather than useful.

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
