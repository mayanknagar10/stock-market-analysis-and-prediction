"""25+ technical indicators — all pure numpy/pandas, zero TA-lib dependency."""
import pandas as pd
import numpy as np
from typing import Dict

# ── Trend ──────────────────────────────────────────────────────────────────
def sma(s, w): return s.rolling(w, min_periods=1).mean()
def ema(s, span): return s.ewm(span=span, adjust=False).mean()
def wma(s, w):
    wts = np.arange(1, w+1)
    return s.rolling(w).apply(lambda x: np.dot(x, wts)/wts.sum(), raw=True)
def vwap(df):
    tp = (df["High"]+df["Low"]+df["Close"])/3
    return (tp*df["Volume"]).cumsum()/df["Volume"].cumsum()

def adx(df, w=14):
    H,L,C = df["High"],df["Low"],df["Close"]
    tr = pd.concat([H-L,(H-C.shift(1)).abs(),(L-C.shift(1)).abs()],axis=1).max(axis=1)
    atr_ = tr.ewm(span=w,adjust=False).mean()
    dmp = np.where((H-H.shift(1))>(L.shift(1)-L), np.maximum(H-H.shift(1),0),0)
    dmm = np.where((L.shift(1)-L)>(H-H.shift(1)), np.maximum(L.shift(1)-L,0),0)
    dip = 100*pd.Series(dmp,index=df.index).ewm(span=w,adjust=False).mean()/atr_.replace(0,np.nan)
    dim = 100*pd.Series(dmm,index=df.index).ewm(span=w,adjust=False).mean()/atr_.replace(0,np.nan)
    dx  = 100*(dip-dim).abs()/(dip+dim).replace(0,np.nan)
    return pd.DataFrame({"ADX":dx.ewm(span=w,adjust=False).mean(),"DI+":dip,"DI-":dim},index=df.index)

def parabolic_sar(df, af0=0.02, af_max=0.2):
    H,L = df["High"].values, df["Low"].values
    n   = len(H); sar=np.zeros(n); ep=np.zeros(n); af=np.full(n,af0); bull=True
    sar[0]=L[0]; ep[0]=H[0]
    for i in range(1,n):
        ps,pe,pa = sar[i-1],ep[i-1],af[i-1]
        if bull:
            sar[i]=min(ps+pa*(pe-ps), L[i-1], L[i-2] if i>1 else L[i-1])
            if H[i]>pe: ep[i]=H[i]; af[i]=min(pa+af0,af_max)
            else:        ep[i]=pe;   af[i]=pa
            if L[i]<sar[i]: bull=False; sar[i]=pe; ep[i]=L[i]; af[i]=af0
        else:
            sar[i]=max(ps+pa*(pe-ps), H[i-1], H[i-2] if i>1 else H[i-1])
            if L[i]<pe: ep[i]=L[i]; af[i]=min(pa+af0,af_max)
            else:        ep[i]=pe;   af[i]=pa
            if H[i]>sar[i]: bull=True; sar[i]=pe; ep[i]=H[i]; af[i]=af0
    return pd.Series(sar,index=df.index,name="PSAR")

# ── Momentum ───────────────────────────────────────────────────────────────
def rsi(s, w=14):
    d=s.diff(); g=d.clip(lower=0).ewm(com=w-1,adjust=False).mean()
    ls=(-d.clip(upper=0)).ewm(com=w-1,adjust=False).mean()
    return 100-100/(1+g/ls.replace(0,np.nan))

def macd(s, fast=12, slow=26, sig=9):
    m=ema(s,fast)-ema(s,slow); signal=ema(m,sig)
    return pd.DataFrame({"MACD":m,"Signal":signal,"Hist":m-signal},index=s.index)

def stochastic(df, k=14, d=3):
    lo=df["Low"].rolling(k).min(); hi=df["High"].rolling(k).max()
    K=100*(df["Close"]-lo)/(hi-lo).replace(0,np.nan)
    return pd.DataFrame({"%K":K,"%D":K.rolling(d).mean()},index=df.index)

def williams_r(df, w=14):
    hi=df["High"].rolling(w).max(); lo=df["Low"].rolling(w).min()
    return -100*(hi-df["Close"])/(hi-lo).replace(0,np.nan)

def _rolling_mad(values: np.ndarray, w: int) -> np.ndarray:
    """Fast vectorized rolling mean absolute deviation — no Python-level
    per-window callback, unlike pandas' rolling().apply(lambda...)."""
    n = len(values)
    out = np.full(n, np.nan)
    if n < w:
        return out
    windows = np.lib.stride_tricks.sliding_window_view(values, w)
    means = windows.mean(axis=1, keepdims=True)
    out[w - 1:] = np.abs(windows - means).mean(axis=1)
    return out


def _rolling_mad_2d(values: np.ndarray, w: int) -> np.ndarray:
    """2D version of _rolling_mad: values shape (T, n_paths), rolls along
    axis 0 (time), independently per column (path)."""
    T, P = values.shape
    out = np.full((T, P), np.nan)
    if T < w:
        return out
    windows = np.lib.stride_tricks.sliding_window_view(values, w, axis=0)  # (T-w+1, P, w)
    means = windows.mean(axis=2, keepdims=True)
    out[w - 1:, :] = np.abs(windows - means).mean(axis=2)
    return out


def build_ml_features_batch(close: pd.DataFrame, open_: pd.DataFrame,
                            high: pd.DataFrame, low: pd.DataFrame,
                            vol: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Batched version of build_ml_features for Monte Carlo simulation.

    Inputs are WIDE DataFrames of shape (T timesteps, n_paths columns) —
    one column per simulated path, all sharing the same DatetimeIndex.
    Returns a dict of {feature_name: wide_DataFrame}, same shape as the
    inputs, so feat[name].iloc[-1] gives every path's latest value at once.

    This exists purely for speed: pandas rolling/ewm operations are
    natively vectorized across columns, so computing features for 30
    simulated paths this way costs barely more than computing them for
    1 path — replacing 30 separate calls to build_ml_features with a
    single call here. Every formula mirrors build_ml_features exactly;
    see test coverage that verifies numerical equivalence on a single
    path before this is trusted for production use.
    """
    c, o, h, l, v = close, open_, high, low, vol
    feat: Dict[str, pd.DataFrame] = {}

    log_c = np.log(c)
    for w in [1, 2, 3, 5, 10, 20]:
        feat[f"ret_{w}d"] = log_c - log_c.shift(w)

    daily_ret = c.pct_change()
    for w in [5, 10, 20]:
        feat[f"ret_mean_{w}"] = daily_ret.rolling(w).mean()
        feat[f"ret_std_{w}"]  = daily_ret.rolling(w).std()
        feat[f"ret_skew_{w}"] = daily_ret.rolling(w).skew()

    def _rsi_wide(s, w):
        d = s.diff()
        g = d.clip(lower=0).ewm(com=w - 1, adjust=False).mean()
        ls = (-d.clip(upper=0)).ewm(com=w - 1, adjust=False).mean()
        return 100 - 100 / (1 + g / ls.replace(0, np.nan))
    feat["rsi_7"]  = _rsi_wide(c, 7)
    feat["rsi_14"] = _rsi_wide(c, 14)

    lo_k = l.rolling(14).min(); hi_k = h.rolling(14).max()
    stoch_k = 100 * (c - lo_k) / (hi_k - lo_k).replace(0, np.nan)
    feat["stoch_k"] = stoch_k
    feat["stoch_d"] = stoch_k.rolling(3).mean()

    hi_w = h.rolling(14).max(); lo_w = l.rolling(14).min()
    feat["williams_r"] = -100 * (hi_w - c) / (hi_w - lo_w).replace(0, np.nan)

    tp = (h + l + c) / 3
    mad_vals = _rolling_mad_2d(tp.values, 20)
    mad = pd.DataFrame(mad_vals, index=tp.index, columns=tp.columns)
    feat["cci_20"] = (tp - tp.rolling(20).mean()) / (0.015 * mad.replace(0, np.nan))

    for w in [5, 10, 20]:
        feat[f"roc_{w}"] = 100 * (c - c.shift(w)) / c.shift(w)

    def _ema_wide(s, span):
        return s.ewm(span=span, adjust=False).mean()
    macd_line = _ema_wide(c, 12) - _ema_wide(c, 26)
    macd_sig  = _ema_wide(macd_line, 9)
    feat["macd_norm"]      = macd_line / c
    feat["macd_sig_norm"]  = macd_sig / c
    feat["macd_hist_norm"] = (macd_line - macd_sig) / c

    mid20 = c.rolling(20, min_periods=1).mean()
    std20 = c.rolling(20).std()
    up = mid20 + 2.0 * std20
    lo_bb = mid20 - 2.0 * std20
    feat["bb_pctb"]  = (c - lo_bb) / (up - lo_bb).replace(0, np.nan)
    feat["bb_width"] = (up - lo_bb) / mid20

    for w in [9, 20, 50, 200]:
        feat[f"dist_sma_{w}"] = c / c.rolling(w, min_periods=1).mean() - 1
        feat[f"dist_ema_{w}"] = c / _ema_wide(c, w) - 1

    tr1 = h - l
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    # NaN-aware elementwise max (matches pandas .max(axis=1, skipna=True)
    # behaviour from the single-path atr() — np.maximum would propagate
    # NaN instead of skipping it, which breaks day-1 ATR/ADX values)
    tr_stack = np.stack([tr1.values, tr2.values, tr3.values], axis=-1)
    tr = pd.DataFrame(np.nanmax(tr_stack, axis=-1), index=c.index, columns=c.columns)
    atr_s = tr.ewm(span=14, adjust=False).mean()
    feat["atr_pct"] = atr_s / c

    for w in [10, 20, 50]:
        feat[f"hv_{w}"] = np.log(c / c.shift(1)).rolling(w).std() * np.sqrt(252)

    tp_mfi = (h + l + c) / 3
    rmf = tp_mfi * v
    pos = rmf.where(tp_mfi > tp_mfi.shift(1), 0)
    neg = rmf.where(tp_mfi < tp_mfi.shift(1), 0)
    feat["mfi_14"] = 100 - 100 / (1 + pos.rolling(14).sum() / neg.rolling(14).sum().replace(0, np.nan))

    rng_hl = (h - l).replace(0, np.nan)
    clv = ((c - l) - (h - c)) / rng_hl
    feat["cmf_20"] = (clv * v).rolling(20).sum() / v.rolling(20).sum()

    feat["vol_ratio"] = v / v.rolling(14).mean()

    obv_s = (np.sign(c.diff()).fillna(0) * v).cumsum()
    for w in [5, 10, 20]:
        feat[f"obv_roc_{w}"] = obv_s.pct_change(w).replace([np.inf, -np.inf], 0)

    h_diff = h - h.shift(1)
    l_diff = l.shift(1) - l
    dmp_vals = np.where(h_diff.values > l_diff.values, np.maximum(h_diff.values, 0), 0)
    dmm_vals = np.where(l_diff.values > h_diff.values, np.maximum(l_diff.values, 0), 0)
    dmp = pd.DataFrame(dmp_vals, index=c.index, columns=c.columns)
    dmm = pd.DataFrame(dmm_vals, index=c.index, columns=c.columns)
    dip = 100 * dmp.ewm(span=14, adjust=False).mean() / atr_s.replace(0, np.nan)
    dim = 100 * dmm.ewm(span=14, adjust=False).mean() / atr_s.replace(0, np.nan)
    dx  = 100 * (dip - dim).abs() / (dip + dim).replace(0, np.nan)
    feat["adx"]     = dx.ewm(span=14, adjust=False).mean()
    feat["di_diff"] = dip - dim

    feat["hl_spread"] = (h - l) / c
    feat["oc_spread"] = (c - o) / o
    feat["gap"]       = (o - c.shift(1)) / c.shift(1)

    n_paths = c.shape[1]
    feat["day_of_week"] = pd.DataFrame(
        np.tile(c.index.dayofweek.values.reshape(-1, 1), (1, n_paths)),
        index=c.index, columns=c.columns)
    feat["month"] = pd.DataFrame(
        np.tile(c.index.month.values.reshape(-1, 1), (1, n_paths)),
        index=c.index, columns=c.columns)
    feat["is_monday"] = pd.DataFrame(
        np.tile((c.index.dayofweek == 0).astype(int).reshape(-1, 1), (1, n_paths)),
        index=c.index, columns=c.columns)
    feat["is_friday"] = pd.DataFrame(
        np.tile((c.index.dayofweek == 4).astype(int).reshape(-1, 1), (1, n_paths)),
        index=c.index, columns=c.columns)

    return feat


def cci(df, w=20):
    tp = (df["High"]+df["Low"]+df["Close"])/3
    mad = pd.Series(_rolling_mad(tp.values, w), index=tp.index)
    return (tp-tp.rolling(w).mean())/(0.015*mad.replace(0,np.nan))


def roc(s, w=10): return 100*(s-s.shift(w))/s.shift(w)
def momentum(s, w=10): return s-s.shift(w)

# ── Volatility ─────────────────────────────────────────────────────────────
def bollinger_bands(s, w=20, n=2.0):
    mid=sma(s,w); std=s.rolling(w).std()
    up=mid+n*std; lo=mid-n*std
    return pd.DataFrame({"BB_Upper":up,"BB_Mid":mid,"BB_Lower":lo,
                          "BB_Width":(up-lo)/mid,"BB_%B":(s-lo)/(up-lo).replace(0,np.nan)},
                         index=s.index)

def atr(df, w=14):
    tr=pd.concat([df["High"]-df["Low"],
                  (df["High"]-df["Close"].shift(1)).abs(),
                  (df["Low"] -df["Close"].shift(1)).abs()],axis=1).max(axis=1)
    return tr.ewm(span=w,adjust=False).mean()

def keltner_channels(df, span=20, mult=2.0):
    mid=ema(df["Close"],span); a=atr(df,span)
    return pd.DataFrame({"KC_Upper":mid+mult*a,"KC_Mid":mid,"KC_Lower":mid-mult*a},index=df.index)

def donchian_channels(df, w=20):
    up=df["High"].rolling(w).max(); lo=df["Low"].rolling(w).min()
    return pd.DataFrame({"DC_Upper":up,"DC_Mid":(up+lo)/2,"DC_Lower":lo},index=df.index)

def historical_volatility(s, w=20): return np.log(s/s.shift(1)).rolling(w).std()*np.sqrt(252)

# ── Volume ─────────────────────────────────────────────────────────────────
def obv(df): return (np.sign(df["Close"].diff()).fillna(0)*df["Volume"]).cumsum()

def money_flow_index(df, w=14):
    tp=( df["High"]+df["Low"]+df["Close"])/3; rmf=tp*df["Volume"]
    pos=rmf.where(tp>tp.shift(1),0); neg=rmf.where(tp<tp.shift(1),0)
    return 100-100/(1+pos.rolling(w).sum()/neg.rolling(w).sum().replace(0,np.nan))

def chaikin_money_flow(df, w=20):
    rng=(df["High"]-df["Low"]).replace(0,np.nan)
    clv=((df["Close"]-df["Low"])-(df["High"]-df["Close"]))/rng
    return (clv*df["Volume"]).rolling(w).sum()/df["Volume"].rolling(w).sum()

def volume_ratio(df, w=14): return df["Volume"]/df["Volume"].rolling(w).mean()

# ── Signal engine ──────────────────────────────────────────────────────────
def generate_signals(df: pd.DataFrame) -> Dict:
    c = df["Close"]
    rsi_v  = float(rsi(c).iloc[-1])
    macd_h = float(macd(c)["Hist"].iloc[-1])
    macd_h_prev = float(macd(c)["Hist"].iloc[-2])
    bb     = bollinger_bands(c); pctb=float(bb["BB_%B"].iloc[-1])
    e20=float(ema(c,20).iloc[-1]); e50=float(ema(c,50).iloc[-1])
    last=float(c.iloc[-1])
    stoch  = stochastic(df); k=float(stoch["%K"].iloc[-1]); d_=float(stoch["%D"].iloc[-1])
    adx_df = adx(df); adxv=float(adx_df["ADX"].iloc[-1])
    dip=float(adx_df["DI+"].iloc[-1]); dim=float(adx_df["DI-"].iloc[-1])
    vr=float(volume_ratio(df).iloc[-1])
    cci_v=float(cci(df).iloc[-1])

    def _s(cond_buy, cond_sell):
        return "BUY" if cond_buy else ("SELL" if cond_sell else "NEUTRAL")

    signals = {
        "RSI (14)":        {"value":f"{rsi_v:.1f}",  "signal":_s(rsi_v<30,rsi_v>70),
                            "note":"Oversold<30 / Overbought>70"},
        "MACD (12,26,9)":  {"value":f"{macd_h:+.4f}","signal":_s(macd_h>0 and macd_h_prev<=0,macd_h<0 and macd_h_prev>=0),
                            "note":"Histogram crossover"},
        "Bollinger %B":    {"value":f"{pctb:.2f}",   "signal":_s(pctb<0,pctb>1),
                            "note":"<0 oversold / >1 overbought"},
        "MA Cross (20/50)":{"value":f"{e20:.1f}/{e50:.1f}","signal":_s(e20>e50,e20<e50),
                            "note":"EMA20 vs EMA50"},
        "Stochastic":      {"value":f"K={k:.1f}",    "signal":_s(k<20 and k>d_,k>80 and k<d_),
                            "note":"K<20 oversold / K>80 overbought"},
        "ADX (14)":        {"value":f"{adxv:.1f}",   "signal":_s(adxv>25 and dip>dim,adxv>25 and dim>dip),
                            "note":">25 = strong trend"},
        "Volume Ratio":    {"value":f"{vr:.2f}x",    "signal":_s(vr>1.5 and last>float(c.iloc[-2]),
                                                                  vr>1.5 and last<float(c.iloc[-2])),
                            "note":">1.5x avg = high activity"},
        "CCI (20)":        {"value":f"{cci_v:.0f}",  "signal":_s(cci_v<-100,cci_v>100),
                            "note":"<-100 oversold / >100 overbought"},
    }
    buy_n  = sum(1 for v in signals.values() if v["signal"]=="BUY")
    sell_n = sum(1 for v in signals.values() if v["signal"]=="SELL")
    total  = buy_n+sell_n
    if total == 0:    composite = "NEUTRAL"
    elif buy_n/total>=0.65:  composite = "STRONG BUY"
    elif buy_n/total>=0.50:  composite = "BUY"
    elif sell_n/total>=0.65: composite = "STRONG SELL"
    else:                    composite = "SELL"
    return {"indicators":signals,"composite":composite,
            "buy_count":buy_n,"sell_count":sell_n}

def build_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a SCALE-FREE feature matrix suitable for a cross-sectional model
    trained across many different stocks (different price levels, different
    volume levels). Every feature here is a ratio, percentage, or bounded
    oscillator — never a raw price or raw volume level — so the same
    trained model generalises from a ₹50 stock to a ₹50,000 stock without
    re-training.

    Returns a DataFrame indexed the same as df, with all-NaN warmup rows
    still present (caller should .dropna()).

    Performance note: all 56 feature Series are collected into a plain
    dict and the DataFrame is constructed ONCE at the end via
    pd.DataFrame(feat). Assigning 56 columns one-at-a-time onto a growing
    DataFrame (out["x"] = ...) is significantly slower in pandas due to
    per-insertion block-manager overhead — this matters here because
    build_ml_features is called recursively inside Monte Carlo forecast
    simulation (core.models._simulate_one_path), often hundreds of times
    per request.
    """
    c, o, h, l, v = df["Close"], df["Open"], df["High"], df["Low"], df["Volume"]
    feat = {}

    # ── Multi-horizon historical log returns (what already happened) ──────
    log_c = np.log(c)
    for w in [1, 2, 3, 5, 10, 20]:
        feat[f"ret_{w}d"] = log_c - log_c.shift(w)

    # ── Rolling return statistics (scale-free: based on returns) ──────────
    daily_ret = c.pct_change()
    for w in [5, 10, 20]:
        feat[f"ret_mean_{w}"] = daily_ret.rolling(w).mean()
        feat[f"ret_std_{w}"]  = daily_ret.rolling(w).std()
        feat[f"ret_skew_{w}"] = daily_ret.rolling(w).skew()

    # ── Momentum oscillators (already bounded / scale-free) ───────────────
    feat["rsi_7"]  = rsi(c, 7)
    feat["rsi_14"] = rsi(c, 14)
    st_ = stochastic(df)
    feat["stoch_k"] = st_["%K"]
    feat["stoch_d"] = st_["%D"]
    feat["williams_r"] = williams_r(df)
    feat["cci_20"] = cci(df)
    for w in [5, 10, 20]:
        feat[f"roc_{w}"] = roc(c, w)

    # ── MACD — normalised by price (was raw price units before) ───────────
    md = macd(c)
    feat["macd_norm"]     = md["MACD"]   / c
    feat["macd_sig_norm"] = md["Signal"] / c
    feat["macd_hist_norm"] = md["Hist"]  / c

    # ── Bollinger — already scale-free ─────────────────────────────────────
    bb = bollinger_bands(c)
    feat["bb_pctb"]  = bb["BB_%B"]
    feat["bb_width"] = bb["BB_Width"]

    # ── Price distance from moving averages (% above/below, not raw MA) ───
    for w in [9, 20, 50, 200]:
        feat[f"dist_sma_{w}"] = c / sma(c, w) - 1
        feat[f"dist_ema_{w}"] = c / ema(c, w) - 1

    # ── Volatility (already annualised %, scale-free) ──────────────────────
    feat["atr_pct"] = atr(df) / c              # ATR normalised by price
    for w in [10, 20, 50]:
        feat[f"hv_{w}"] = historical_volatility(c, w)

    # ── Volume-based (already ratios) ───────────────────────────────────────
    feat["mfi_14"] = money_flow_index(df)
    feat["cmf_20"] = chaikin_money_flow(df)
    feat["vol_ratio"] = volume_ratio(df)
    obv_s = obv(df)
    for w in [5, 10, 20]:
        feat[f"obv_roc_{w}"] = obv_s.pct_change(w).replace([np.inf, -np.inf], 0)

    # ── Trend strength (already bounded 0-100) ──────────────────────────────
    adx_df = adx(df)
    feat["adx"] = adx_df["ADX"]
    feat["di_diff"] = adx_df["DI+"] - adx_df["DI-"]   # bounded, scale-free

    # ── Intraday structure (already ratios) ──────────────────────────────────
    feat["hl_spread"] = (h - l) / c
    feat["oc_spread"] = (c - o) / o
    feat["gap"]       = (o - c.shift(1)) / c.shift(1)

    # ── Calendar (categorical, ticker-agnostic) ─────────────────────────────
    feat["day_of_week"] = df.index.dayofweek
    feat["month"]       = df.index.month
    feat["is_monday"]   = (df.index.dayofweek == 0).astype(int)
    feat["is_friday"]   = (df.index.dayofweek == 4).astype(int)

    return pd.DataFrame(feat, index=df.index)


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add raw-scale indicator columns (SMA/EMA/MACD/Bollinger at actual price
    level, not normalised). Used for chart overlays where the real price
    level is exactly what you want to plot on top of a candlestick chart —
    a different use case from build_ml_features(), which is deliberately
    scale-free for cross-sectional model training.
    """
    out = df.copy()
    c = out["Close"]
    for w in [9, 20, 50, 200]:
        out[f"SMA_{w}"] = sma(c, w)
        out[f"EMA_{w}"] = ema(c, w)
    out["VWAP"] = vwap(out)
    out["RSI_14"] = rsi(c)
    out["RSI_7"] = rsi(c, 7)
    for w in [5, 10, 20]:
        out[f"ROC_{w}"] = roc(c, w)
        out[f"MOM_{w}"] = momentum(c, w)
    md = macd(c)
    out["MACD"] = md["MACD"]
    out["MACD_Sig"] = md["Signal"]
    out["MACD_H"] = md["Hist"]
    st_ = stochastic(out)
    out["Stoch_K"] = st_["%K"]
    out["Stoch_D"] = st_["%D"]
    out["WilliamsR"] = williams_r(out)
    out["CCI_20"] = cci(out)
    bb = bollinger_bands(c)
    out["BB_Up"] = bb["BB_Upper"]
    out["BB_Lo"] = bb["BB_Lower"]
    out["BB_W"] = bb["BB_Width"]
    out["BB_B"] = bb["BB_%B"]
    out["ATR_14"] = atr(out)
    out["HV_20"] = historical_volatility(c, 20)
    out["OBV"] = obv(out)
    out["MFI_14"] = money_flow_index(out)
    out["CMF_20"] = chaikin_money_flow(out)
    out["VolRatio"] = volume_ratio(out)
    out["DayRet"] = c.pct_change()
    out["LogRet"] = np.log(c / c.shift(1))
    out["HL"] = (out["High"] - out["Low"]) / c
    out["OC"] = (c - out["Open"]) / out["Open"]
    out["Gap"] = (out["Open"] - c.shift(1)) / c.shift(1)
    out["DayOfWeek"] = out.index.dayofweek
    out["Month"] = out.index.month
    out["IsMonday"] = (out.index.dayofweek == 0).astype(int)
    out["IsFriday"] = (out.index.dayofweek == 4).astype(int)
    for lag in [1, 2, 3, 5, 10, 20]:
        out[f"CLag{lag}"] = c.shift(lag)
    for w in [5, 10, 20]:
        out[f"RM{w}"] = c.rolling(w).mean()
        out[f"RS{w}"] = c.rolling(w).std()
        out[f"RMn{w}"] = c.rolling(w).min()
        out[f"RMx{w}"] = c.rolling(w).max()
    return out
