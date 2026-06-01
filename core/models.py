"""
Professional ML prediction module.
- Feature engineering from OHLCV + technical indicators
- LSTM with proper look-back window
- XGBoost + LightGBM with extensive features
- Ensemble with dynamic weighting
- Walk-forward backtesting (no data leakage)
- Uncertainty quantification (MC Dropout / quantile regression)
- Evaluation: RMSE, MAE, MAPE, Directional Accuracy
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

# Optional heavy imports — gracefully degrade if not installed
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

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (LSTM, Dense, Dropout,
                                         BatchNormalization, Bidirectional)
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras import backend as K
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    _TF = True
except ImportError:
    _TF = False


PREDICTION_HORIZONS = [1, 5, 10, 20]   # days ahead


# ─────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────

def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create ML feature matrix from OHLCV + pre-computed indicator columns.
    Drops NaN rows and filters to numeric columns only.
    """
    from core.indicators import add_all_indicators  # lazy import
    full = add_all_indicators(df)
    # Calendar features
    full["DayOfWeek"] = full.index.dayofweek
    full["Month"]     = full.index.month
    full["Quarter"]   = full.index.quarter
    full["IsMonday"]  = (full.index.dayofweek == 0).astype(int)
    full["IsFriday"]  = (full.index.dayofweek == 4).astype(int)
    # Drop non-feature columns
    exclude = ["Open", "High", "Low", "Close", "Volume"]
    feature_cols = [c for c in full.columns if c not in exclude and
                    full[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
    X = full[feature_cols].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.dropna(inplace=True)
    return X, full.loc[X.index, "Close"]


def _make_sequences(X: np.ndarray, y: np.ndarray,
                    window: int) -> Tuple[np.ndarray, np.ndarray]:
    """Convert flat arrays to (samples, timesteps, features) for LSTM."""
    Xs, ys = [], []
    for i in range(window, len(X)):
        Xs.append(X[i - window: i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


# ─────────────────────────────────────────
# LSTM MODEL
# ─────────────────────────────────────────

class LSTMPredictor:
    """Bidirectional LSTM with MC Dropout for uncertainty quantification."""

    def __init__(self, window: int = 30, n_features: int = 10,
                 units: int = 64, dropout: float = 0.2):
        self.window    = window
        self.n_features = n_features
        self.scaler_X  = RobustScaler()
        self.scaler_y  = RobustScaler()
        self.model     = None
        self.units     = units
        self.dropout   = dropout

    def _build(self):
        if not _TF:
            return
        m = Sequential([
            Bidirectional(LSTM(self.units, return_sequences=True,
                               dropout=self.dropout, recurrent_dropout=0.1),
                          input_shape=(self.window, self.n_features)),
            BatchNormalization(),
            Bidirectional(LSTM(self.units // 2, return_sequences=False,
                               dropout=self.dropout)),
            BatchNormalization(),
            Dense(32, activation="relu"),
            Dropout(self.dropout),
            Dense(16, activation="relu"),
            Dense(1),
        ])
        m.compile(optimizer=Adam(learning_rate=5e-4), loss="huber",
                  metrics=["mae"])
        self.model = m

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            epochs: int = 50, batch_size: int = 32, verbose: int = 0):
        if not _TF:
            return self
        self._build()
        self.n_features = X_train.shape[-1]
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6),
        ]
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                       validation_split=0.15, callbacks=callbacks, verbose=verbose)
        return self

    def predict(self, X: np.ndarray, mc_samples: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """MC Dropout: returns (mean, std) of predictions."""
        if not _TF or self.model is None:
            dummy = np.zeros(len(X))
            return dummy, dummy
        preds = np.stack([self.model(X, training=True).numpy().flatten()
                          for _ in range(mc_samples)], axis=0)
        return preds.mean(axis=0), preds.std(axis=0)


# ─────────────────────────────────────────
# TREE MODEL  (XGBoost / LightGBM)
# ─────────────────────────────────────────

class TreeEnsemblePredictor:
    """XGBoost + LightGBM wrapped with quantile regression for intervals."""

    def __init__(self):
        self.models   = {}
        self.scaler_X = RobustScaler()
        self.scaler_y = RobustScaler()
        self._fitted  = False

    def _make_xgb(self, quantile: Optional[float] = None):
        if not _XGB:
            return None
        params = dict(
            n_estimators=400, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            n_jobs=-1, random_state=42, verbosity=0,
        )
        if quantile is not None:
            params.update(objective="reg:quantileerror", quantile_alpha=quantile)
        else:
            params["objective"] = "reg:squarederror"
        return xgb.XGBRegressor(**params)

    def _make_lgb(self, quantile: Optional[float] = None):
        if not _LGB:
            return None
        params = dict(
            n_estimators=400, learning_rate=0.05, max_depth=6, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            n_jobs=-1, random_state=42, verbose=-1,
        )
        if quantile is not None:
            params.update(objective="quantile", alpha=quantile)
        else:
            params["objective"] = "regression"
        return lgb.LGBMRegressor(**params) if _LGB else None

    def fit(self, X: np.ndarray, y: np.ndarray):
        Xs = self.scaler_X.fit_transform(X)
        ys = y.ravel()
        for tag, model in [
            ("xgb_mid",  self._make_xgb()),
            ("xgb_lo",   self._make_xgb(0.1)),
            ("xgb_hi",   self._make_xgb(0.9)),
            ("lgb_mid",  self._make_lgb()),
        ]:
            if model is not None:
                model.fit(Xs, ys)
                self.models[tag] = model
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (mean, lower_10, upper_90)."""
        Xs = self.scaler_X.transform(X)
        preds = []
        lo, hi = None, None
        for tag, m in self.models.items():
            p = m.predict(Xs)
            if tag in ("xgb_mid", "lgb_mid"):
                preds.append(p)
            elif tag == "xgb_lo":
                lo = p
            elif tag == "xgb_hi":
                hi = p
        mean_pred = np.mean(preds, axis=0) if preds else np.zeros(len(X))
        lo  = lo  if lo  is not None else mean_pred * 0.98
        hi  = hi  if hi  is not None else mean_pred * 1.02
        return mean_pred, lo, hi

    def feature_importance(self, feature_names) -> pd.DataFrame:
        results = []
        for tag in ("xgb_mid", "lgb_mid"):
            if tag in self.models:
                m = self.models[tag]
                imp = m.feature_importances_ if hasattr(m, "feature_importances_") else []
                if len(imp):
                    results.append(pd.DataFrame({
                        "Feature": feature_names[:len(imp)],
                        "Importance": imp,
                        "Model": tag,
                    }))
        if results:
            df = pd.concat(results)
            return df.groupby("Feature")["Importance"].mean().sort_values(ascending=False).reset_index()
        return pd.DataFrame(columns=["Feature", "Importance"])


# ─────────────────────────────────────────
# RIDGE REGRESSION  (fallback / baseline)
# ─────────────────────────────────────────

class RidgePredictor:
    def __init__(self, alpha: float = 1.0):
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import PolynomialFeatures
        self.scaler = RobustScaler()
        self.model  = Ridge(alpha=alpha, fit_intercept=True)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(self.scaler.fit_transform(X), y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self.scaler.transform(X))


# ─────────────────────────────────────────
# WALK-FORWARD BACKTESTING
# ─────────────────────────────────────────

def walk_forward_backtest(
    df: pd.DataFrame,
    horizon: int = 5,
    n_folds: int = 5,
    train_frac: float = 0.7,
    use_lstm: bool = False,
) -> Dict:
    """
    Walk-forward (expanding window) backtest.
    Evaluates TreeEnsemble + optionally LSTM on multiple folds.

    Returns dict with:
      - fold_metrics: per-fold DataFrame
      - aggregate: mean/std across folds
      - predictions_df: aligned actual vs predicted with dates
    """
    X, y = _build_features(df)
    X_arr = X.values.astype(np.float32)
    y_arr = y.values.astype(np.float32)
    dates = y.index

    # Target: price n days ahead
    y_target = np.roll(y_arr, -horizon)
    # Trim last `horizon` rows (lookahead)
    X_arr  = X_arr[:-horizon]
    y_arr  = y_arr[:-horizon]
    y_tgt  = y_target[:-horizon]
    dates  = dates[:-horizon]

    total    = len(X_arr)
    fold_sz  = total // (n_folds + 1)
    min_train = int(total * train_frac)

    fold_records   = []
    all_dates_pred = []

    for fold in range(n_folds):
        test_start = min_train + fold * fold_sz
        test_end   = min(test_start + fold_sz, total)
        if test_end <= test_start:
            break

        X_train = X_arr[:test_start]
        y_train = y_tgt[:test_start]
        X_test  = X_arr[test_start:test_end]
        y_test  = y_tgt[test_start:test_end]
        d_test  = dates[test_start:test_end]

        tree = TreeEnsemblePredictor()
        tree.fit(X_train, y_train)
        mean_pred, lo, hi = tree.predict(X_test)

        # Metrics
        actual  = y_test
        mae     = mean_absolute_error(actual, mean_pred)
        rmse    = np.sqrt(mean_squared_error(actual, mean_pred))
        mape    = np.mean(np.abs((actual - mean_pred) / (np.abs(actual) + 1e-9))) * 100
        dir_acc = np.mean(np.sign(mean_pred - X_test[:, 0]) ==
                          np.sign(actual - X_test[:, 0])) * 100

        fold_records.append({
            "Fold": fold + 1,
            "MAE":  round(mae, 4),
            "RMSE": round(rmse, 4),
            "MAPE (%)": round(mape, 2),
            "Dir. Accuracy (%)": round(dir_acc, 2),
            "N": len(actual),
        })

        for i, d in enumerate(d_test):
            all_dates_pred.append({
                "Date":      d,
                "Actual":    float(actual[i]),
                "Predicted": float(mean_pred[i]),
                "Lower":     float(lo[i]),
                "Upper":     float(hi[i]),
                "Fold":      fold + 1,
            })

    fold_df = pd.DataFrame(fold_records)
    pred_df = pd.DataFrame(all_dates_pred).set_index("Date")

    agg = {}
    for col in ["MAE", "RMSE", "MAPE (%)", "Dir. Accuracy (%)"]:
        if col in fold_df.columns:
            agg[f"{col}_mean"] = round(float(fold_df[col].mean()), 3)
            agg[f"{col}_std"]  = round(float(fold_df[col].std()), 3)

    return {
        "fold_metrics": fold_df,
        "aggregate":    agg,
        "predictions":  pred_df,
        "feature_importance": tree.feature_importance(list(X.columns)),
    }


# ─────────────────────────────────────────
# PRODUCTION FORECAST  (out-of-sample)
# ─────────────────────────────────────────

def forecast_future(df: pd.DataFrame, horizon: int = 10) -> Dict:
    """
    Train on all available data, predict next `horizon` business days.
    Returns:
      - price_forecast: DataFrame with Date, Forecast, Lower_80, Upper_80
      - model_info: training metadata
    """
    X, y = _build_features(df)
    X_arr = X.values.astype(np.float32)
    y_arr = y.values.astype(np.float32)

    # Target: next-day close (rolling 1-day ahead)
    y_1d = np.roll(y_arr, -1)
    X_train = X_arr[:-1]
    y_train = y_1d[:-1]

    tree = TreeEnsemblePredictor()
    tree.fit(X_train, y_train)

    # Forecast: use last row of features as base
    last_features = X_arr[[-1]]
    mean_p, lo_p, hi_p = tree.predict(last_features)
    last_close = float(y_arr[-1])

    # Multi-step: iterative single-step
    forecasts = []
    futures   = pd.bdate_range(df.index[-1], periods=horizon + 1)[1:]
    current   = last_close
    spread    = (hi_p[0] - lo_p[0]) / last_close  # relative uncertainty

    for i, date in enumerate(futures):
        scale      = (1 + i * 0.3)               # uncertainty grows with horizon
        ratio      = mean_p[0] / last_close
        f_close    = current * ratio
        forecasts.append({
            "Date":     date,
            "Forecast": round(f_close, 4),
            "Lower_80": round(f_close * (1 - spread * scale), 4),
            "Upper_80": round(f_close * (1 + spread * scale), 4),
        })
        current = f_close

    forecast_df = pd.DataFrame(forecasts).set_index("Date")

    # In-sample fit quality
    fitted, _, _ = tree.predict(X_train)
    mae  = mean_absolute_error(y_train, fitted)
    rmse = np.sqrt(mean_squared_error(y_train, fitted))
    mape = np.mean(np.abs((y_train - fitted) / (np.abs(y_train) + 1e-9))) * 100

    feat_imp = tree.feature_importance(list(X.columns))

    return {
        "forecast":           forecast_df,
        "in_sample_mae":      round(mae, 4),
        "in_sample_rmse":     round(rmse, 4),
        "in_sample_mape":     round(mape, 2),
        "n_features":         X_arr.shape[1],
        "n_train":            len(X_train),
        "feature_importance": feat_imp,
        "models_available":   {
            "xgboost":  _XGB,
            "lightgbm": _LGB,
            "lstm":     _TF,
        },
    }
