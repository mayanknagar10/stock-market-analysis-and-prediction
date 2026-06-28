"""
Prediction engine v4 — Universal Checkpoint Architecture.

KEY DESIGN CHANGE FROM v3:
  v3 trained a fresh XGBoost+LightGBM(+LSTM) model from scratch on every
  single page load, fitted to only ONE stock's ~250-1500 rows of history.
  This was slow (20-90s per request) AND inaccurate (a model with 60+
  features has far too little data to learn from on a single ticker —
  it mostly overfits noise).

  v4 trains ONE model ONCE on a cross-section of many different stocks
  (pooled together, scale-free features — see core/indicators.build_ml_features),
  saves it as a checkpoint to disk, and every page load just LOADS the
  checkpoint (instant) and runs inference on the current ticker's latest
  feature row. This is the standard approach used by real quant cross-
  sectional models: train once on a broad universe, predict for ANY stock
  without retraining, because the model learns general relationships
  between technical-indicator-patterns and forward returns rather than
  memorising one company's idiosyncratic price history.

  Training the checkpoint requires internet access to Yahoo Finance
  (see scripts/train_universal_model.py or the in-app "Train Universal
  Model" panel on the Price Prediction page). If no checkpoint exists
  yet, the app falls back to a small, fast, single-ticker XGBoost model
  so it never crashes — but accuracy and speed are both better once a
  real checkpoint is trained.
"""

import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    _XGB = True
except ImportError:
    _XGB = False

try:
    import lightgbm as lgb
    _LGB = True
except ImportError:
    _LGB = False

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
XGB_PATH   = os.path.join(MODELS_DIR, "universal_xgb.json")
LGB_PATH   = os.path.join(MODELS_DIR, "universal_lgb.txt")
META_PATH  = os.path.join(MODELS_DIR, "universal_meta.json")

DEFAULT_TRAIN_UNIVERSE = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS",
    "HINDUNILVR.NS","ITC.NS","SBIN.NS","BHARTIARTL.NS","KOTAKBANK.NS",
    "LT.NS","AXISBANK.NS","ASIANPAINT.NS","MARUTI.NS","HCLTECH.NS",
    "SUNPHARMA.NS","TITAN.NS","BAJFINANCE.NS","WIPRO.NS","TATAMOTORS.NS",
    "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","JPM","V","UNH",
    "XOM","JNJ","WMT","HD","BAC","NFLX","AMD","INTC","BA","GS",
]

TARGET_HORIZON = 1


def _build_training_row_set(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Build (X, y) for ONE ticker's history; y = next-day log return."""
    from core.indicators import build_ml_features
    feats = build_ml_features(df)
    log_c = np.log(df["Close"])
    next_ret = (log_c.shift(-1) - log_c)
    combined = pd.concat([feats, next_ret.rename("target")], axis=1)
    combined.replace([np.inf, -np.inf], np.nan, inplace=True)
    combined.dropna(inplace=True)
    X = combined.drop(columns=["target"])
    y = combined["target"]
    return X, y


def _latest_feature_row(df: pd.DataFrame, feature_names: List[str]) -> np.ndarray:
    """Most recent feature row for inference, aligned to feature_names."""
    from core.indicators import build_ml_features
    feats = build_ml_features(df)
    feats = feats.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    row = feats.iloc[[-1]]
    row = row.reindex(columns=feature_names, fill_value=0)
    return row.values.astype(np.float32)


# ─────────────────────────────────────────────────────────────────
# UNIVERSAL MODEL — TRAINING (pooled, cross-sectional)
# ─────────────────────────────────────────────────────────────────

def train_universal_model(universe: Optional[List[str]] = None,
                          period: str = "5y", progress_callback=None) -> Dict:
    """
    Train the universal checkpoint on a pooled cross-section of many tickers.
    Saves XGBoost + LightGBM boosters + metadata to MODELS_DIR.
    Requires internet access to Yahoo Finance.
    """
    from core.data_fetcher import fetch_ohlcv

    universe = universe or DEFAULT_TRAIN_UNIVERSE
    os.makedirs(MODELS_DIR, exist_ok=True)

    def _prog(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)

    all_X, all_y, all_tickers = [], [], []
    n = len(universe)
    for i, ticker in enumerate(universe):
        _prog((i / n) * 0.6, f"Fetching {ticker} ({i+1}/{n})…")
        try:
            df = fetch_ohlcv(ticker, period=period, interval="1d")
            if df.empty or len(df) < 100:
                continue
            X, y = _build_training_row_set(df)
            if len(X) < 50:
                continue
            all_X.append(X)
            all_y.append(y)
            all_tickers.extend([ticker] * len(X))
        except Exception:
            continue

    if not all_X:
        raise RuntimeError(
            "Could not fetch training data for any ticker. Check internet "
            "access to Yahoo Finance (works on Streamlit Cloud / local "
            "machines, fails in network-restricted sandboxes)."
        )

    _prog(0.62, "Pooling data across all tickers…")
    X_pool = pd.concat(all_X, axis=0, ignore_index=True)
    y_pool = pd.concat(all_y, axis=0, ignore_index=True)
    tickers_arr = np.array(all_tickers)
    feature_names = list(X_pool.columns)

    _prog(0.65, "Building train/test split…")
    train_mask = np.zeros(len(X_pool), dtype=bool)
    for t in np.unique(tickers_arr):
        idx = np.where(tickers_arr == t)[0]
        cut = int(len(idx) * 0.85)
        train_mask[idx[:cut]] = True
    test_mask = ~train_mask

    X_train, y_train = X_pool[train_mask], y_pool[train_mask]
    X_test, y_test = X_pool[test_mask], y_pool[test_mask]

    _prog(0.70, f"Training on {len(X_train):,} rows across {len(np.unique(tickers_arr))} tickers…")

    scaler = RobustScaler()
    Xtr_s = scaler.fit_transform(X_train)
    Xte_s = scaler.transform(X_test)

    models = {}
    if _XGB:
        xgb_model = xgb.XGBRegressor(
            n_estimators=600, learning_rate=0.03, max_depth=5,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=1.5,
            n_jobs=-1, random_state=42, verbosity=0,
            objective="reg:squarederror")
        xgb_model.fit(Xtr_s, y_train.values)
        models["xgb"] = xgb_model
    if _LGB:
        _prog(0.80, "Training LightGBM…")
        lgb_model = lgb.LGBMRegressor(
            n_estimators=600, learning_rate=0.03, max_depth=5, num_leaves=31,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=1.5,
            n_jobs=-1, random_state=42, verbose=-1,
            objective="regression")
        lgb_model.fit(Xtr_s, y_train.values)
        models["lgb"] = lgb_model

    if not models:
        raise RuntimeError("Neither XGBoost nor LightGBM is installed.")

    _prog(0.90, "Evaluating on held-out test set…")
    preds = [m.predict(Xte_s) for m in models.values()]
    pred_mean = np.mean(preds, axis=0)
    mae = float(mean_absolute_error(y_test, pred_mean))
    rmse = float(np.sqrt(mean_squared_error(y_test, pred_mean)))
    dir_acc = float(np.mean(np.sign(pred_mean) == np.sign(y_test.values)) * 100)

    per_ticker_dir = {}
    test_tickers = tickers_arr[test_mask]
    for t in np.unique(test_tickers):
        m = test_tickers == t
        if m.sum() < 5:
            continue
        p = pred_mean[m]; a = y_test.values[m]
        per_ticker_dir[t] = round(float(np.mean(np.sign(p) == np.sign(a)) * 100), 1)

    _prog(0.95, "Saving checkpoint…")
    if "xgb" in models:
        models["xgb"].get_booster().save_model(XGB_PATH)
    if "lgb" in models:
        models["lgb"].booster_.save_model(LGB_PATH)

    scaler_path = os.path.join(MODELS_DIR, "universal_scaler.json")
    with open(scaler_path, "w") as f:
        json.dump({"center": scaler.center_.tolist(), "scale": scaler.scale_.tolist()}, f)

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "universe": universe,
        "n_tickers_used": int(len(np.unique(tickers_arr))),
        "n_train_rows": int(len(X_train)),
        "n_test_rows": int(len(X_test)),
        "feature_names": feature_names,
        "models_trained": list(models.keys()),
        "test_mae": round(mae, 6),
        "test_rmse": round(rmse, 6),
        "test_directional_accuracy": round(dir_acc, 2),
        "per_ticker_directional_accuracy": per_ticker_dir,
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    _prog(1.0, "Done!")
    return meta


# ─────────────────────────────────────────────────────────────────
# UNIVERSAL MODEL — LOADING + INFERENCE
# ─────────────────────────────────────────────────────────────────

class UniversalPredictor:
    """Loads the pre-trained checkpoint once; reused across requests."""

    def __init__(self):
        self.xgb_model = None
        self.lgb_model = None
        self.scaler_center = None
        self.scaler_scale = None
        self.feature_names: List[str] = []
        self.meta: Dict = {}
        self.loaded = False

    def load(self) -> "UniversalPredictor":
        if not os.path.exists(META_PATH):
            return self
        try:
            with open(META_PATH) as f:
                self.meta = json.load(f)
            self.feature_names = self.meta.get("feature_names", [])

            if _XGB and os.path.exists(XGB_PATH):
                booster = xgb.Booster()
                booster.load_model(XGB_PATH)
                self.xgb_model = booster

            if _LGB and os.path.exists(LGB_PATH):
                self.lgb_model = lgb.Booster(model_file=LGB_PATH)

            scaler_path = os.path.join(MODELS_DIR, "universal_scaler.json")
            if os.path.exists(scaler_path):
                with open(scaler_path) as f:
                    sc = json.load(f)
                self.scaler_center = np.array(sc["center"], dtype=np.float32)
                self.scaler_scale = np.array(sc["scale"], dtype=np.float32)

            self.loaded = bool(
                (self.xgb_model is not None or self.lgb_model is not None)
                and self.scaler_center is not None)
        except Exception:
            self.loaded = False
        return self

    def _scale(self, X: np.ndarray) -> np.ndarray:
        scale = np.where(self.scaler_scale == 0, 1.0, self.scaler_scale)
        return (X - self.scaler_center) / scale

    def predict_next_return(self, X_row: np.ndarray) -> float:
        if not self.loaded:
            return 0.0
        Xs = self._scale(X_row)
        preds = []
        if self.xgb_model is not None:
            # NOTE: trained on a plain numpy array (no column names baked
            # into the Booster), so we deliberately do NOT pass
            # feature_names here — XGBoost validates feature names
            # strictly and a mismatch would raise an error. Column order
            # is already guaranteed correct via the reindex() at feature-
            # build time, so positional alignment is sufficient.
            d = xgb.DMatrix(Xs)
            preds.append(float(self.xgb_model.predict(d)[0]))
        if self.lgb_model is not None:
            preds.append(float(self.lgb_model.predict(Xs)[0]))
        return float(np.mean(preds)) if preds else 0.0

    def predict_next_return_batch(self, X_batch: np.ndarray) -> np.ndarray:
        """Batched version: X_batch shape (n_samples, n_features) -> (n_samples,).
        Used by Monte Carlo simulation to advance all paths in one call
        instead of looping — XGBoost/LightGBM both predict a batch in
        roughly the same time as a single row, so this is what makes
        simulating many paths fast."""
        if not self.loaded:
            return np.zeros(X_batch.shape[0])
        Xs = self._scale(X_batch)
        preds = []
        if self.xgb_model is not None:
            d = xgb.DMatrix(Xs)
            preds.append(self.xgb_model.predict(d))
        if self.lgb_model is not None:
            preds.append(self.lgb_model.predict(Xs))
        return np.mean(preds, axis=0) if preds else np.zeros(X_batch.shape[0])


_predictor_singleton: Optional[UniversalPredictor] = None


def get_universal_predictor() -> UniversalPredictor:
    """Module-level cache so the checkpoint loads from disk only once."""
    global _predictor_singleton
    if _predictor_singleton is None:
        _predictor_singleton = UniversalPredictor().load()
    return _predictor_singleton


def reload_universal_predictor() -> UniversalPredictor:
    """Force-reload after (re)training so the new checkpoint takes effect."""
    global _predictor_singleton
    _predictor_singleton = UniversalPredictor().load()
    return _predictor_singleton


def universal_model_available() -> bool:
    return get_universal_predictor().loaded


def universal_model_metadata() -> Dict:
    return get_universal_predictor().meta


# ─────────────────────────────────────────────────────────────────
# FALLBACK MODEL — fast, single-ticker, used only if no checkpoint exists
# ─────────────────────────────────────────────────────────────────

class _FallbackPredictor:
    """Small, fast XGBoost fitted on just this one ticker. Only used when
    no universal checkpoint has been trained yet."""

    def __init__(self):
        self.model = None
        self.scaler = RobustScaler()
        self.feature_names: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_FallbackPredictor":
        self.feature_names = list(X.columns)
        Xs = self.scaler.fit_transform(X)
        if _XGB:
            self.model = xgb.XGBRegressor(
                n_estimators=150, learning_rate=0.08, max_depth=4,
                subsample=0.85, colsample_bytree=0.8,
                n_jobs=-1, random_state=42, verbosity=0,
                objective="reg:squarederror")
            self.model.fit(Xs, y.values)
        elif _LGB:
            self.model = lgb.LGBMRegressor(
                n_estimators=150, learning_rate=0.08, max_depth=4,
                n_jobs=-1, random_state=42, verbose=-1)
            self.model.fit(Xs, y.values)
        return self

    def predict_next_return(self, X_row: np.ndarray) -> float:
        if self.model is None:
            return 0.0
        Xs = self.scaler.transform(X_row)
        return float(self.model.predict(Xs)[0])

    def predict_next_return_batch(self, X_batch: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.zeros(X_batch.shape[0])
        Xs = self.scaler.transform(X_batch)
        return self.model.predict(Xs)


def _vol_cone(last_price: float, daily_vol: float, horizon: int, z: float = 1.28
             ) -> Tuple[np.ndarray, np.ndarray]:
    """GBM confidence cone: width grows as sqrt(t), not linearly."""
    t = np.arange(1, horizon + 1)
    lo = last_price * np.exp(-z * daily_vol * np.sqrt(t))
    hi = last_price * np.exp(+z * daily_vol * np.sqrt(t))
    return lo, hi


def _simulate_paths(df: pd.DataFrame, predictor, daily_vol: float, horizon: int,
                    n_paths: int = 30, seed: Optional[int] = None) -> np.ndarray:
    """
    Run an ensemble of n_paths recursive Monte Carlo simulations.

    Critically, this re-computes technical features and re-queries the
    model at EVERY future day using each path's own simulated trajectory
    so far — not just once. If a simulated run-up pushes RSI into
    overbought territory, the model can genuinely predict a smaller (or
    negative) next-day return in response, exactly like it would for a
    real stock in that state. A random shock drawn from the ticker's own
    historical volatility is added on top of the model's drift estimate,
    since the model predicts an *expected* return, not a noise-free one.

    Speed: ALL n_paths advance together at each day-step — one batched
    feature computation (build_ml_features_batch) and one batched model
    prediction per day, not per path. This means total cost scales with
    `horizon`, not `horizon × n_paths`, which is what makes simulating
    30 paths over 30 days complete in roughly the same time as computing
    features once would for a single path, instead of 30x longer.

    Returns shape (n_paths, horizon).
    """
    from core.indicators import build_ml_features_batch

    last_date = df.index[-1]
    last_price = float(df["Close"].iloc[-1])
    if seed is None:
        # Deterministic seed so re-running the same query gives the same
        # answer (not a different random forecast every page refresh),
        # while different tickers/dates naturally get different paths.
        seed = abs(hash((str(last_date), round(last_price, 4)))) % (2**31)
    rng = np.random.default_rng(seed)

    buf = df.iloc[-230:].copy() if len(df) > 230 else df.copy()
    recent_vol_mean = (float(buf["Volume"].iloc[-20:].mean())
                       if len(buf) >= 20 else float(buf["Volume"].mean()))

    path_cols = [f"p{i}" for i in range(n_paths)]
    close_w = pd.DataFrame({col: buf["Close"] for col in path_cols})
    open_w  = pd.DataFrame({col: buf["Open"]  for col in path_cols})
    high_w  = pd.DataFrame({col: buf["High"]  for col in path_cols})
    low_w   = pd.DataFrame({col: buf["Low"]   for col in path_cols})
    vol_w   = pd.DataFrame({col: buf["Volume"] for col in path_cols})

    feature_names = getattr(predictor, "feature_names", None)
    last_closes = np.full(n_paths, last_price)
    paths = np.zeros((n_paths, horizon))

    for step in range(horizon):
        feat = build_ml_features_batch(close_w, open_w, high_w, low_w, vol_w)

        if feature_names:
            X_batch = np.column_stack([
                feat[name].iloc[-1].values if name in feat else np.zeros(n_paths)
                for name in feature_names
            ])
        else:
            X_batch = np.column_stack([feat[name].iloc[-1].values for name in feat])
        X_batch = np.nan_to_num(X_batch, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        drifts = predictor.predict_next_return_batch(X_batch)
        shocks = rng.normal(0.0, daily_vol, size=n_paths)
        new_closes = last_closes * np.exp(drifts + shocks)

        new_opens = last_closes * (1 + rng.normal(0, 0.001, size=n_paths))
        new_highs = np.maximum(new_opens, new_closes) * (1 + np.abs(rng.normal(0, 0.003, size=n_paths)))
        new_lows  = np.minimum(new_opens, new_closes) * (1 - np.abs(rng.normal(0, 0.003, size=n_paths)))
        new_vols  = np.maximum(recent_vol_mean * (1 + rng.normal(0, 0.15, size=n_paths)), 1.0)

        next_date = close_w.index[-1] + pd.tseries.offsets.BDay(1)
        close_w.loc[next_date] = new_closes
        open_w.loc[next_date]  = new_opens
        high_w.loc[next_date]  = new_highs
        low_w.loc[next_date]   = new_lows
        vol_w.loc[next_date]   = new_vols

        if len(close_w) > 250:
            close_w = close_w.iloc[-230:]
            open_w  = open_w.iloc[-230:]
            high_w  = high_w.iloc[-230:]
            low_w   = low_w.iloc[-230:]
            vol_w   = vol_w.iloc[-230:]

        paths[:, step] = new_closes
        last_closes = new_closes

    return paths


def forecast_future(df: pd.DataFrame, horizon: int = 10, n_paths: int = 30) -> Dict:
    """
    Forecast next `horizon` business days.

    Runs an ensemble of `n_paths` recursive Monte Carlo simulations (see
    _simulate_one_path) — each one re-queries the model at every future
    day using that simulation's own evolving trajectory, so the forecast
    can genuinely show down days, not just a monotonic ramp. The "Forecast"
    column is the per-day MEDIAN across all simulated paths; Lower_80/
    Upper_80 are the empirical 10th/90th percentiles across the same
    ensemble, so the central estimate and the confidence band are built
    from the same consistent simulation rather than two different formulas.

    Loads the universal checkpoint if available (fast inference, no
    training); falls back to a small single-ticker model otherwise.
    """
    t0 = time.time()
    predictor = get_universal_predictor()
    mode = "universal" if predictor.loaded else "fallback"

    log_c = np.log(df["Close"])
    hist_log_ret = (log_c - log_c.shift(1)).dropna()
    daily_vol = float(hist_log_ret.std())
    last_price = float(df["Close"].iloc[-1])

    if mode == "universal":
        active_predictor = predictor
        n_features = len(predictor.feature_names)
        meta = predictor.meta
        in_sample_mae = meta.get("test_mae")
        in_sample_dir = meta.get("test_directional_accuracy")
        n_train = meta.get("n_train_rows")
        trained_at = meta.get("trained_at")
        universe_size = meta.get("n_tickers_used")
    else:
        X_train, y_train = _build_training_row_set(df)
        if len(X_train) < 40:
            raise RuntimeError("Not enough history. Use a longer training period.")
        fb = _FallbackPredictor().fit(X_train, y_train)
        active_predictor = fb
        n_features = len(fb.feature_names)

        Xs_all = fb.scaler.transform(X_train.values)
        fitted = fb.model.predict(Xs_all)
        in_sample_mae = round(float(mean_absolute_error(y_train, fitted)), 6)
        in_sample_dir = round(float(np.mean(np.sign(fitted) == np.sign(y_train.values)) * 100), 1)
        n_train = len(X_train)
        trained_at = None
        universe_size = 1

    future_dates = pd.bdate_range(df.index[-1], periods=horizon + 1)[1:]
    paths = _simulate_paths(df, active_predictor, daily_vol, horizon, n_paths=n_paths)

    median_path = np.median(paths, axis=0)
    lo_path = np.percentile(paths, 10, axis=0)
    hi_path = np.percentile(paths, 90, axis=0)

    rows = []
    prev_price = last_price
    for i, date in enumerate(future_dates):
        day_ret = np.log(median_path[i] / prev_price) if prev_price > 0 else 0.0
        rows.append({"Date": date, "Forecast": round(float(median_path[i]), 4),
                     "Lower_80": round(float(lo_path[i]), 4),
                     "Upper_80": round(float(hi_path[i]), 4),
                     "Log_Return": round(float(day_ret), 6)})
        prev_price = median_path[i]

    forecast_df = pd.DataFrame(rows).set_index("Date")

    elapsed = round(time.time() - t0, 2)

    return {
        "forecast": forecast_df, "mode": mode,
        "in_sample_mae": in_sample_mae, "in_sample_mape": None,
        "in_sample_dir_acc": in_sample_dir,
        "daily_volatility": round(daily_vol * 100, 4),
        "n_features": n_features, "n_train": n_train,
        "universe_size": universe_size, "trained_at": trained_at,
        "elapsed_seconds": elapsed, "n_paths": n_paths,
        "models_available": {"xgboost": _XGB, "lightgbm": _LGB},
    }


def walk_forward_backtest(df: pd.DataFrame, horizon: int = 1, n_folds: int = 5,
                          train_frac: float = 0.65) -> Dict:
    """
    Evaluate prediction quality on this ticker's own past data.
    With the universal checkpoint loaded this is pure inference across
    rolling windows (no retraining) — fast regardless of n_folds.
    """
    predictor = get_universal_predictor()
    use_universal = predictor.loaded

    X_all, y_all = _build_training_row_set(df)
    if len(X_all) < 60:
        return {"fold_metrics": pd.DataFrame(), "aggregate": {}, "predictions": pd.DataFrame()}

    prices_aligned = df["Close"].reindex(X_all.index)
    dates = X_all.index
    total = len(X_all)
    min_tr = int(total * train_frac)
    fold_sz = max((total - min_tr) // n_folds, 10)

    fold_records = []
    all_preds = []

    for fold in range(n_folds):
        ts = min_tr + fold * fold_sz
        te = min(ts + fold_sz, total)
        if te <= ts:
            break

        X_te = X_all.iloc[ts:te]
        y_te = y_all.iloc[ts:te]
        p_te = prices_aligned.iloc[ts:te].values
        d_te = dates[ts:te]

        if use_universal:
            X_te_aligned = X_te.reindex(columns=predictor.feature_names, fill_value=0)
            Xs = predictor._scale(X_te_aligned.values.astype(np.float32))
            preds = []
            if predictor.xgb_model is not None:
                d = xgb.DMatrix(Xs)   # no feature_names — see note in predict_next_return
                preds.append(predictor.xgb_model.predict(d))
            if predictor.lgb_model is not None:
                preds.append(predictor.lgb_model.predict(Xs))
            pred_ret = np.mean(preds, axis=0) if preds else np.zeros(len(X_te))
        else:
            X_tr = X_all.iloc[:ts]
            y_tr = y_all.iloc[:ts]
            if len(X_tr) < 40:
                continue
            fb = _FallbackPredictor().fit(X_tr, y_tr)
            Xs = fb.scaler.transform(X_te.values)
            pred_ret = fb.model.predict(Xs)

        actual_p = p_te * np.exp(y_te.values)
        pred_p = p_te * np.exp(pred_ret)

        mae = float(mean_absolute_error(actual_p, pred_p))
        rmse = float(np.sqrt(mean_squared_error(actual_p, pred_p)))
        mape = float(np.mean(np.abs((actual_p - pred_p) / (np.abs(actual_p) + 1e-9))) * 100)
        dir_a = float(np.mean(np.sign(pred_ret) == np.sign(y_te.values)) * 100)

        fold_records.append({"Fold": fold + 1, "N": len(X_te),
            "MAE": round(mae, 2), "RMSE": round(rmse, 2),
            "MAPE (%)": round(mape, 2), "Dir. Accuracy (%)": round(dir_a, 2)})
        for i, d in enumerate(d_te):
            all_preds.append({"Date": d, "Actual": float(actual_p[i]),
                              "Predicted": float(pred_p[i]), "Fold": fold + 1})

    fold_df = pd.DataFrame(fold_records)
    pred_df = pd.DataFrame(all_preds).set_index("Date") if all_preds else pd.DataFrame()

    agg = {}
    for col in ["MAE", "RMSE", "MAPE (%)", "Dir. Accuracy (%)"]:
        if col in fold_df.columns and not fold_df.empty:
            agg[f"{col}_mean"] = round(float(fold_df[col].mean()), 3)
            agg[f"{col}_std"] = round(float(fold_df[col].std()), 3)

    return {"fold_metrics": fold_df, "aggregate": agg, "predictions": pred_df,
            "mode": "universal" if use_universal else "fallback"}
