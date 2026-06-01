"""
Professional technical indicators library.
All calculations are vectorised with pandas/numpy — no external TA library dependency
so there are no version conflicts. Covers trend, momentum, volatility, and volume families.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


# ─────────────────────────────────────────
# TREND INDICATORS
# ─────────────────────────────────────────

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def wma(series: pd.Series, window: int) -> pd.Series:
    weights = np.arange(1, window + 1)
    return series.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP — resets daily (grouped by date)."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    cum_tp_vol = (tp * df["Volume"]).cumsum()
    cum_vol = df["Volume"].cumsum()
    return cum_tp_vol / cum_vol


def adx(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Average Directional Index with +DI and -DI."""
    high, low, close = df["High"], df["Low"], df["Close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_ = tr.ewm(span=window, adjust=False).mean()

    dm_plus  = np.where((high - high.shift(1)) > (low.shift(1) - low),
                        np.maximum(high - high.shift(1), 0), 0)
    dm_minus = np.where((low.shift(1) - low) > (high - high.shift(1)),
                        np.maximum(low.shift(1) - low, 0), 0)

    sm_plus  = pd.Series(dm_plus,  index=df.index).ewm(span=window, adjust=False).mean()
    sm_minus = pd.Series(dm_minus, index=df.index).ewm(span=window, adjust=False).mean()

    di_plus  = 100 * sm_plus  / atr_.replace(0, np.nan)
    di_minus = 100 * sm_minus / atr_.replace(0, np.nan)
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    adx_val  = dx.ewm(span=window, adjust=False).mean()

    return pd.DataFrame({"ADX": adx_val, "DI+": di_plus, "DI-": di_minus}, index=df.index)


def parabolic_sar(df: pd.DataFrame, af_start: float = 0.02, af_max: float = 0.2) -> pd.Series:
    """Parabolic SAR."""
    high, low = df["High"].values, df["Low"].values
    n = len(high)
    sar = np.zeros(n)
    ep  = np.zeros(n)
    af  = np.full(n, af_start)
    bull = True

    sar[0] = low[0]
    ep[0]  = high[0]

    for i in range(1, n):
        prev_sar = sar[i - 1]
        prev_ep  = ep[i - 1]
        prev_af  = af[i - 1]

        if bull:
            sar[i] = prev_sar + prev_af * (prev_ep - prev_sar)
            sar[i] = min(sar[i], low[i - 1], low[i - 2] if i > 1 else low[i - 1])
            if high[i] > prev_ep:
                ep[i] = high[i]
                af[i] = min(prev_af + af_start, af_max)
            else:
                ep[i] = prev_ep
                af[i] = prev_af
            if low[i] < sar[i]:
                bull = False
                sar[i] = prev_ep
                ep[i]  = low[i]
                af[i]  = af_start
        else:
            sar[i] = prev_sar + prev_af * (prev_ep - prev_sar)
            sar[i] = max(sar[i], high[i - 1], high[i - 2] if i > 1 else high[i - 1])
            if low[i] < prev_ep:
                ep[i] = low[i]
                af[i] = min(prev_af + af_start, af_max)
            else:
                ep[i] = prev_ep
                af[i] = prev_af
            if high[i] > sar[i]:
                bull = True
                sar[i] = prev_ep
                ep[i]  = high[i]
                af[i]  = af_start

    return pd.Series(sar, index=df.index, name="PSAR")


# ─────────────────────────────────────────
# MOMENTUM INDICATORS
# ─────────────────────────────────────────

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(com=window - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=window - 1, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
         ) -> pd.DataFrame:
    ema_fast   = ema(series, fast)
    ema_slow   = ema(series, slow)
    macd_line  = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram   = macd_line - signal_line
    return pd.DataFrame({"MACD": macd_line, "Signal": signal_line, "Hist": histogram},
                        index=series.index)


def stochastic(df: pd.DataFrame, k_window: int = 14, d_window: int = 3) -> pd.DataFrame:
    low_min  = df["Low"].rolling(k_window).min()
    high_max = df["High"].rolling(k_window).max()
    k = 100 * (df["Close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    d = k.rolling(d_window).mean()
    return pd.DataFrame({"%K": k, "%D": d}, index=df.index)


def williams_r(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high_max = df["High"].rolling(window).max()
    low_min  = df["Low"].rolling(window).min()
    return -100 * (high_max - df["Close"]) / (high_max - low_min).replace(0, np.nan)


def cci(df: pd.DataFrame, window: int = 20) -> pd.Series:
    tp     = (df["High"] + df["Low"] + df["Close"]) / 3
    sma_tp = tp.rolling(window).mean()
    mad    = tp.rolling(window).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))


def roc(series: pd.Series, window: int = 10) -> pd.Series:
    return 100 * (series - series.shift(window)) / series.shift(window)


def momentum(series: pd.Series, window: int = 10) -> pd.Series:
    return series - series.shift(window)


# ─────────────────────────────────────────
# VOLATILITY INDICATORS
# ─────────────────────────────────────────

def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0
                    ) -> pd.DataFrame:
    mid = sma(series, window)
    std = series.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = (upper - lower) / mid
    pct_b = (series - lower) / (upper - lower).replace(0, np.nan)
    return pd.DataFrame({
        "BB_Upper": upper, "BB_Mid": mid, "BB_Lower": lower,
        "BB_Width": width, "BB_%B": pct_b
    }, index=series.index)


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - df["Close"].shift(1)).abs()
    tr3 = (df["Low"]  - df["Close"].shift(1)).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=window, adjust=False).mean()


def keltner_channels(df: pd.DataFrame, ema_span: int = 20, atr_mult: float = 2.0
                     ) -> pd.DataFrame:
    mid   = ema(df["Close"], ema_span)
    atr_v = atr(df, ema_span)
    return pd.DataFrame({
        "KC_Upper": mid + atr_mult * atr_v,
        "KC_Mid":   mid,
        "KC_Lower": mid - atr_mult * atr_v,
    }, index=df.index)


def donchian_channels(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    upper = df["High"].rolling(window).max()
    lower = df["Low"].rolling(window).min()
    mid   = (upper + lower) / 2
    return pd.DataFrame({"DC_Upper": upper, "DC_Mid": mid, "DC_Lower": lower},
                        index=df.index)


def historical_volatility(series: pd.Series, window: int = 20) -> pd.Series:
    """Annualised historical volatility (log-return std * √252)."""
    log_ret = np.log(series / series.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252)


# ─────────────────────────────────────────
# VOLUME INDICATORS
# ─────────────────────────────────────────

def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["Close"].diff()).fillna(0)
    return (direction * df["Volume"]).cumsum()


def money_flow_index(df: pd.DataFrame, window: int = 14) -> pd.Series:
    tp  = (df["High"] + df["Low"] + df["Close"]) / 3
    rmf = tp * df["Volume"]
    pos_mf = rmf.where(tp > tp.shift(1), 0)
    neg_mf = rmf.where(tp < tp.shift(1), 0)
    pos_sum = pos_mf.rolling(window).sum()
    neg_sum = neg_mf.rolling(window).sum()
    mfr = pos_sum / neg_sum.replace(0, np.nan)
    return 100 - (100 / (1 + mfr))


def chaikin_money_flow(df: pd.DataFrame, window: int = 20) -> pd.Series:
    clv = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / \
          (df["High"] - df["Low"]).replace(0, np.nan)
    mfv = clv * df["Volume"]
    return mfv.rolling(window).sum() / df["Volume"].rolling(window).sum()


def volume_ratio(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Volume relative to rolling mean — useful for spotting breakouts."""
    return df["Volume"] / df["Volume"].rolling(window).mean()


# ─────────────────────────────────────────
# SIGNAL GENERATION
# ─────────────────────────────────────────

def generate_signals(df: pd.DataFrame) -> Dict[str, dict]:
    """
    Compute all indicators and generate BUY / SELL / NEUTRAL signals.
    Returns a dict keyed by indicator name with value, signal, and strength.
    """
    c = df["Close"]

    # --- RSI ---
    rsi_val = rsi(c).iloc[-1]
    rsi_sig = "BUY" if rsi_val < 30 else ("SELL" if rsi_val > 70 else "NEUTRAL")

    # --- MACD ---
    macd_df  = macd(c)
    macd_val = macd_df["Hist"].iloc[-1]
    prev_hist = macd_df["Hist"].iloc[-2]
    macd_sig = "BUY" if (macd_val > 0 and prev_hist < 0) else (
               "SELL" if (macd_val < 0 and prev_hist > 0) else (
               "BUY" if macd_val > 0 else "SELL"))

    # --- Bollinger Bands ---
    bb   = bollinger_bands(c)
    pctb = bb["BB_%B"].iloc[-1]
    bb_sig = "BUY" if pctb < 0 else ("SELL" if pctb > 1 else "NEUTRAL")

    # --- Moving average cross ---
    ema20 = ema(c, 20).iloc[-1]
    ema50 = ema(c, 50).iloc[-1]
    ma_sig = "BUY" if ema20 > ema50 else "SELL"

    # --- Stochastic ---
    stoch = stochastic(df)
    k_val = stoch["%K"].iloc[-1]
    d_val = stoch["%D"].iloc[-1]
    stoch_sig = "BUY" if (k_val < 20 and k_val > d_val) else (
                "SELL" if (k_val > 80 and k_val < d_val) else "NEUTRAL")

    # --- ADX Trend strength ---
    adx_df = adx(df)
    adx_val = adx_df["ADX"].iloc[-1]
    di_plus  = adx_df["DI+"].iloc[-1]
    di_minus = adx_df["DI-"].iloc[-1]
    adx_sig  = "BUY" if (adx_val > 25 and di_plus > di_minus) else (
               "SELL" if (adx_val > 25 and di_minus > di_plus) else "NEUTRAL")

    # --- Volume trend ---
    vr = volume_ratio(df).iloc[-1]
    vol_sig = "BUY" if (vr > 1.5 and c.iloc[-1] > c.iloc[-2]) else (
              "SELL" if (vr > 1.5 and c.iloc[-1] < c.iloc[-2]) else "NEUTRAL")

    # --- CCI ---
    cci_val = cci(df).iloc[-1]
    cci_sig = "BUY" if cci_val < -100 else ("SELL" if cci_val > 100 else "NEUTRAL")

    def strength(sig, mag):
        if sig == "NEUTRAL":
            return "—"
        return "Strong" if mag else "Moderate"

    signals = {
        "RSI (14)":          {"value": f"{rsi_val:.1f}",  "signal": rsi_sig,
                              "note": "Oversold<30 / Overbought>70"},
        "MACD (12,26,9)":    {"value": f"{macd_val:.3f}", "signal": macd_sig,
                              "note": "Histogram crossover"},
        "Bollinger %B":      {"value": f"{pctb:.2f}",     "signal": bb_sig,
                              "note": "<0 oversold / >1 overbought"},
        "MA Cross (20/50)":  {"value": f"{ema20:.2f}/{ema50:.2f}", "signal": ma_sig,
                              "note": "EMA 20 vs EMA 50"},
        "Stochastic":        {"value": f"K={k_val:.1f}",  "signal": stoch_sig,
                              "note": "K<20 oversold / K>80 overbought"},
        "ADX (14)":          {"value": f"{adx_val:.1f}",  "signal": adx_sig,
                              "note": ">25 = strong trend"},
        "Volume Ratio":      {"value": f"{vr:.2f}x",      "signal": vol_sig,
                              "note": ">1.5x avg = high activity"},
        "CCI (20)":          {"value": f"{cci_val:.1f}",  "signal": cci_sig,
                              "note": "<-100 oversold / >100 overbought"},
    }

    # Composite signal
    buy_count  = sum(1 for v in signals.values() if v["signal"] == "BUY")
    sell_count = sum(1 for v in signals.values() if v["signal"] == "SELL")
    total = buy_count + sell_count
    if total == 0:
        composite = "NEUTRAL"
    elif buy_count / total >= 0.65:
        composite = "STRONG BUY"
    elif buy_count / total >= 0.5:
        composite = "BUY"
    elif sell_count / total >= 0.65:
        composite = "STRONG SELL"
    else:
        composite = "SELL"

    return {"indicators": signals, "composite": composite,
            "buy_count": buy_count, "sell_count": sell_count}


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all indicator columns to the DataFrame for ML feature engineering."""
    out = df.copy()
    c = out["Close"]

    # Trend
    for w in [9, 20, 50, 200]:
        out[f"SMA_{w}"]  = sma(c, w)
        out[f"EMA_{w}"]  = ema(c, w)
    out["VWAP"] = vwap(out)

    # Momentum
    out["RSI_14"]  = rsi(c)
    out["RSI_7"]   = rsi(c, 7)
    for w in [5, 10, 20]:
        out[f"ROC_{w}"] = roc(c, w)
        out[f"MOM_{w}"] = momentum(c, w)
    macd_df = macd(c)
    out["MACD"]        = macd_df["MACD"]
    out["MACD_Signal"] = macd_df["Signal"]
    out["MACD_Hist"]   = macd_df["Hist"]
    stoch_df = stochastic(out)
    out["Stoch_K"] = stoch_df["%K"]
    out["Stoch_D"] = stoch_df["%D"]
    out["Williams_R"] = williams_r(out)
    out["CCI_20"]     = cci(out)

    # Volatility
    bb_df = bollinger_bands(c)
    out["BB_Upper"] = bb_df["BB_Upper"]
    out["BB_Lower"] = bb_df["BB_Lower"]
    out["BB_Width"] = bb_df["BB_Width"]
    out["BB_PctB"]  = bb_df["BB_%B"]
    out["ATR_14"]   = atr(out)
    out["HV_20"]    = historical_volatility(c, 20)

    # Volume
    out["OBV"]      = obv(out)
    out["MFI_14"]   = money_flow_index(out)
    out["CMF_20"]   = chaikin_money_flow(out)
    out["VolRatio"] = volume_ratio(out)

    # Price-based features
    out["DailyReturn"]  = c.pct_change()
    out["LogReturn"]    = np.log(c / c.shift(1))
    out["HL_Spread"]    = (out["High"] - out["Low"]) / c
    out["OC_Spread"]    = (out["Close"] - out["Open"]) / out["Open"]
    out["GapUp"]        = (out["Open"] - out["Close"].shift(1)) / out["Close"].shift(1)

    # Lagged closes
    for lag in [1, 2, 3, 5, 10, 20]:
        out[f"Close_Lag{lag}"] = c.shift(lag)

    # Rolling stats
    for w in [5, 10, 20]:
        out[f"Rolling_Mean_{w}"] = c.rolling(w).mean()
        out[f"Rolling_Std_{w}"]  = c.rolling(w).std()
        out[f"Rolling_Min_{w}"]  = c.rolling(w).min()
        out[f"Rolling_Max_{w}"]  = c.rolling(w).max()

    return out
