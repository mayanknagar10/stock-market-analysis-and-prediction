"""Professional risk metrics: VaR/CVaR (3 methods), CAPM, drawdown, Monte Carlo."""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Optional, Tuple

TRADING_DAYS = 252
RISK_FREE     = 0.045   # annualised; update as needed

def compute_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change().dropna()

def annualised_return(r: pd.Series) -> float:
    if len(r)==0: return 0.0
    n = len(r)/TRADING_DAYS
    return float((1+r).prod()**(1/n)-1) if n>0 else 0.0

def annualised_volatility(r: pd.Series) -> float:
    return float(r.std()*np.sqrt(TRADING_DAYS))

def cumulative_return(prices: pd.Series) -> float:
    return float(prices.iloc[-1]/prices.iloc[0]-1) if len(prices)>=2 else 0.0

# VaR
def var_historical(r, c=0.95): return float(-np.percentile(r.dropna(),(1-c)*100))
def var_parametric(r, c=0.95):
    mu,s=r.mean(),r.std(); return float(-(mu+stats.norm.ppf(1-c)*s))
def var_cornish_fisher(r, c=0.95):
    r=r.dropna(); mu,s=r.mean(),r.std()
    sk=float(r.skew()); ku=float(r.kurtosis()); z=stats.norm.ppf(1-c)
    zcf=z+(z**2-1)*sk/6+(z**3-3*z)*ku/24-(2*z**3-5*z)*sk**2/36
    return float(-(mu+zcf*s))
def cvar(r, c=0.95):
    r=r.dropna(); cut=np.percentile(r,(1-c)*100); tail=r[r<=cut]
    return float(-tail.mean()) if len(tail)>0 else 0.0

# Ratios
def sharpe_ratio(r: pd.Series, rf=RISK_FREE) -> float:
    daily_rf=(1+rf)**(1/TRADING_DAYS)-1; ex=r-daily_rf
    return float(ex.mean()/ex.std()*np.sqrt(TRADING_DAYS)) if ex.std()>0 else 0.0

def sortino_ratio(r: pd.Series, rf=RISK_FREE) -> float:
    daily_rf=(1+rf)**(1/TRADING_DAYS)-1; ex=r-daily_rf
    dd=ex[ex<0].std()*np.sqrt(TRADING_DAYS)
    return float((annualised_return(r)-rf)/dd) if dd>0 else 0.0

def calmar_ratio(prices: pd.Series, r: pd.Series) -> float:
    _,mdd,_=drawdown_analysis(prices)
    return float(annualised_return(r)/abs(mdd)) if mdd!=0 else 0.0

def information_ratio(r: pd.Series, bench: pd.Series) -> float:
    active=r-bench.reindex(r.index).fillna(0)
    te=active.std()*np.sqrt(TRADING_DAYS)
    return float(active.mean()*TRADING_DAYS/te) if te>0 else 0.0

# Drawdown
def drawdown_series(prices: pd.Series) -> pd.Series:
    return (prices-prices.cummax())/prices.cummax()

def drawdown_analysis(prices: pd.Series) -> Tuple[pd.Series,float,int]:
    dd=drawdown_series(prices); mdd=float(dd.min())
    in_dd=dd<0; grp=(in_dd!=in_dd.shift()).cumsum()
    dur=in_dd.groupby(grp).cumsum(); max_dur=int(dur.max()) if in_dd.any() else 0
    return dd,mdd,max_dur

# CAPM
def capm_stats(sr: pd.Series, br: pd.Series, rf=RISK_FREE) -> Dict:
    aligned=pd.concat([sr,br],axis=1,join="inner").dropna()
    if len(aligned)<10: return {"beta":1.0,"alpha":0.0,"r_squared":0.0,"treynor":0.0}
    y,x=aligned.iloc[:,0].values, aligned.iloc[:,1].values
    slope,intercept,r,_,_=stats.linregress(x,y)
    daily_rf=(1+rf)**(1/TRADING_DAYS)-1
    return {"beta":round(float(slope),4),"alpha":round(float(intercept*TRADING_DAYS),4),
            "r_squared":round(float(r**2),4),
            "treynor":round(float((y.mean()-daily_rf)*TRADING_DAYS/slope),4) if slope!=0 else 0.0}

# Monte Carlo
def monte_carlo(prices: pd.Series, n_simulations=500, n_days=30, seed=42) -> pd.DataFrame:
    rng=np.random.default_rng(seed); r=np.log(prices/prices.shift(1)).dropna()
    mu,sigma=r.mean(),r.std(); last=float(prices.iloc[-1])
    shock=rng.normal((mu-0.5*sigma**2),sigma,(n_days,n_simulations))
    paths=last*np.exp(np.vstack([np.zeros(n_simulations),shock.cumsum(axis=0)]))
    dates=pd.bdate_range(prices.index[-1],periods=n_days+1)[1:]
    return pd.DataFrame(paths[1:],index=dates)

# Full report
def full_risk_report(prices: pd.Series, bench_prices: Optional[pd.Series]=None) -> Dict:
    r=compute_returns(prices); dd,mdd,mdd_dur=drawdown_analysis(prices)
    rep={
        "cumulative_return":    cumulative_return(prices),
        "annualised_return":    annualised_return(r),
        "annualised_volatility":annualised_volatility(r),
        "daily_return_mean":    float(r.mean()),
        "daily_return_std":     float(r.std()),
        "skewness":             float(r.skew()),
        "kurtosis":             float(r.kurtosis()),
        "var_95_historical":    var_historical(r),
        "var_95_parametric":    var_parametric(r),
        "var_95_cf":            var_cornish_fisher(r),
        "cvar_95":              cvar(r),
        "var_99_historical":    var_historical(r,0.99),
        "var_99_parametric":    var_parametric(r,0.99),
        "cvar_99":              cvar(r,0.99),
        "sharpe_ratio":         sharpe_ratio(r),
        "sortino_ratio":        sortino_ratio(r),
        "max_drawdown":         mdd,
        "max_drawdown_duration":mdd_dur,
        "drawdown_series":      dd,
    }
    if bench_prices is not None:
        br=compute_returns(bench_prices)
        rep.update(capm_stats(r,br))
        rep["calmar_ratio"]      = calmar_ratio(prices,r)
        rep["information_ratio"] = information_ratio(r,br)
    return rep
