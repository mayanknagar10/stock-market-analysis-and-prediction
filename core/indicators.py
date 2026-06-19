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

def cci(df, w=20):
    tp=( df["High"]+df["Low"]+df["Close"])/3
    mad=tp.rolling(w).apply(lambda x: np.mean(np.abs(x-np.mean(x))),raw=True)
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

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all indicator columns — used as ML features."""
    out = df.copy(); c = out["Close"]
    for w in [9,20,50,200]: out[f"SMA_{w}"]=sma(c,w); out[f"EMA_{w}"]=ema(c,w)
    out["VWAP"] = vwap(out)
    out["RSI_14"]=rsi(c); out["RSI_7"]=rsi(c,7)
    for w in [5,10,20]: out[f"ROC_{w}"]=roc(c,w); out[f"MOM_{w}"]=momentum(c,w)
    md=macd(c); out["MACD"]=md["MACD"]; out["MACD_Sig"]=md["Signal"]; out["MACD_H"]=md["Hist"]
    st_=stochastic(out); out["Stoch_K"]=st_["%K"]; out["Stoch_D"]=st_["%D"]
    out["WilliamsR"]=williams_r(out); out["CCI_20"]=cci(out)
    bb=bollinger_bands(c)
    out["BB_Up"]=bb["BB_Upper"]; out["BB_Lo"]=bb["BB_Lower"]
    out["BB_W"]=bb["BB_Width"];  out["BB_B"]=bb["BB_%B"]
    out["ATR_14"]=atr(out); out["HV_20"]=historical_volatility(c,20)
    out["OBV"]=obv(out); out["MFI_14"]=money_flow_index(out)
    out["CMF_20"]=chaikin_money_flow(out); out["VolRatio"]=volume_ratio(out)
    out["DayRet"]=c.pct_change(); out["LogRet"]=np.log(c/c.shift(1))
    out["HL"]=( out["High"]-out["Low"])/c; out["OC"]=(c-out["Open"])/out["Open"]
    out["Gap"]=(out["Open"]-c.shift(1))/c.shift(1)
    out["DayOfWeek"]=out.index.dayofweek; out["Month"]=out.index.month
    out["IsMonday"]=(out.index.dayofweek==0).astype(int)
    out["IsFriday"]=(out.index.dayofweek==4).astype(int)
    for lag in [1,2,3,5,10,20]: out[f"CLag{lag}"]=c.shift(lag)
    for w in [5,10,20]:
        out[f"RM{w}"]=c.rolling(w).mean(); out[f"RS{w}"]=c.rolling(w).std()
        out[f"RMn{w}"]=c.rolling(w).min();  out[f"RMx{w}"]=c.rolling(w).max()
    return out
