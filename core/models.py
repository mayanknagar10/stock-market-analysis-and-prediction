"""
Prediction engine v3.
Target = log return (stationary). CI = GBM volatility cone sqrt(t).
Ensemble: XGBoost + LightGBM + optional Bidirectional LSTM.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

try: import xgboost as xgb; _XGB=True
except: _XGB=False
try: import lightgbm as lgb; _LGB=True
except: _LGB=False
try:
    import tensorflow as tf; tf.get_logger().setLevel("ERROR")
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (Bidirectional, LSTM, Dense,
                                         Dropout, BatchNormalization, Input)
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    _TF=True
except: _TF=False

TRADING_DAYS=252


def _build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Returns (X_features, log_returns) — X aligned to log_returns index."""
    from core.indicators import add_all_indicators
    full = add_all_indicators(df)
    full["LogReturn"] = np.log(full["Close"]/full["Close"].shift(1))
    drop = ["Open","High","Low","Close","Volume","LogReturn"]
    feat_cols = [c for c in full.columns if c not in drop
                 and full[c].dtype in [np.float64,np.float32,np.int64,np.int32]]
    combined = pd.concat([full[feat_cols], full["LogReturn"]], axis=1).dropna()
    X = combined[feat_cols].copy()
    y = combined["LogReturn"]
    X.replace([np.inf,-np.inf], np.nan, inplace=True)
    X.ffill(inplace=True); X.fillna(0, inplace=True)
    return X, y


def _vol_cone(last_price, daily_vol, horizon, z=1.28):
    """GBM confidence cone: width grows as sqrt(t)."""
    t = np.arange(1, horizon+1)
    lo = last_price * np.exp(-z * daily_vol * np.sqrt(t))
    hi = last_price * np.exp(+z * daily_vol * np.sqrt(t))
    return lo, hi


def _make_sequences(X, y, window=30):
    Xs,ys=[],[]
    for i in range(window,len(X)):
        Xs.append(X[i-window:i]); ys.append(y[i])
    return np.array(Xs,dtype=np.float32), np.array(ys,dtype=np.float32)


class LSTMModel:
    WINDOW=20
    def __init__(self, n_feat, units=48, drop=0.2):
        self.n_feat=n_feat; self.units=units; self.drop=drop
        self.sx=RobustScaler(); self.sy=RobustScaler()
        self.model=None; self._fitted=False

    def _build(self):
        m=Sequential([Input(shape=(self.WINDOW,self.n_feat)),
            Bidirectional(LSTM(self.units,return_sequences=True,
                               dropout=self.drop,recurrent_dropout=0.0)),
            BatchNormalization(),
            Bidirectional(LSTM(max(self.units//2,16),return_sequences=False,dropout=self.drop)),
            BatchNormalization(), Dense(24,activation="relu"),
            Dropout(self.drop), Dense(1)])
        m.compile(optimizer=Adam(2e-3), loss="huber", metrics=["mae"])
        self.model=m

    def fit(self, X, y, epochs=35, batch=64):
        if not _TF: return self
        Xs=self.sx.fit_transform(X); ys=self.sy.fit_transform(y.reshape(-1,1)).ravel()
        Xseq,yseq=_make_sequences(Xs,ys,self.WINDOW)
        if len(Xseq)<40: return self
        self._build()
        self.model.fit(Xseq,yseq,epochs=epochs,batch_size=batch,
            validation_split=0.15,verbose=0,
            callbacks=[EarlyStopping(patience=6,restore_best_weights=True),
                       ReduceLROnPlateau(patience=3,factor=0.5,min_lr=1e-6)])
        self._fitted=True; return self

    def predict(self, X, mc=30):
        if not _TF or not self._fitted: return np.zeros(1),np.zeros(1)
        Xs=self.sx.transform(X); seq=Xs[-self.WINDOW:].reshape(1,self.WINDOW,self.n_feat)
        preds=np.array([self.model(seq,training=True).numpy().flatten()[0] for _ in range(mc)])
        mr=float(self.sy.inverse_transform([[preds.mean()]])[0,0])
        sr=float(preds.std()*self.sy.scale_[0])
        return np.array([mr]), np.array([sr])


class TreeEnsemble:
    def __init__(self):
        self.models={}; self.sx=RobustScaler()
        self._fitted=False; self.feat_names:List[str]=[]

    def _xgb(self, q=None):
        if not _XGB: return None
        p=dict(n_estimators=500,learning_rate=0.04,max_depth=5,subsample=0.8,
               colsample_bytree=0.7,reg_alpha=0.05,reg_lambda=1.0,
               n_jobs=-1,random_state=42,verbosity=0)
        if q is not None: p.update(objective="reg:quantileerror",quantile_alpha=q)
        else: p["objective"]="reg:squarederror"
        return xgb.XGBRegressor(**p)

    def _lgb(self, q=None):
        if not _LGB: return None
        p=dict(n_estimators=500,learning_rate=0.04,max_depth=5,num_leaves=31,
               subsample=0.8,colsample_bytree=0.7,reg_alpha=0.05,reg_lambda=1.0,
               n_jobs=-1,random_state=42,verbose=-1)
        if q is not None: p.update(objective="quantile",alpha=q)
        else: p["objective"]="regression"
        return lgb.LGBMRegressor(**p)

    def fit(self, X, y, feat_names=None):
        Xs=self.sx.fit_transform(X); yr=y.ravel()
        if feat_names: self.feat_names=feat_names
        for tag,m in [("xgb",self._xgb()),("xgb_lo",self._xgb(0.10)),
                       ("xgb_hi",self._xgb(0.90)),("lgb",self._lgb())]:
            if m is not None: m.fit(Xs,yr); self.models[tag]=m
        self._fitted=True; return self

    def predict(self, X):
        Xs=self.sx.transform(X)
        preds=[self.models[k].predict(Xs) for k in ("xgb","lgb") if k in self.models]
        mean_=np.mean(preds,axis=0) if preds else np.zeros(len(X))
        lo_=self.models["xgb_lo"].predict(Xs) if "xgb_lo" in self.models else mean_*0.98
        hi_=self.models["xgb_hi"].predict(Xs) if "xgb_hi" in self.models else mean_*1.02
        return mean_,lo_,hi_

    def feature_importance(self) -> pd.DataFrame:
        rows=[]
        for tag in ("xgb","lgb"):
            if tag in self.models:
                imp=getattr(self.models[tag],"feature_importances_",np.array([]))
                if len(imp) and self.feat_names:
                    n=min(len(imp),len(self.feat_names))
                    rows.append(pd.DataFrame({"Feature":self.feat_names[:n],"Importance":imp[:n]}))
        if rows:
            return (pd.concat(rows).groupby("Feature")["Importance"]
                    .mean().sort_values(ascending=False).reset_index())
        return pd.DataFrame(columns=["Feature","Importance"])


def forecast_future(df: pd.DataFrame, horizon: int=10) -> Dict:
    """
    Train on full history; forecast next `horizon` business days.
    Target = log return (stationary). Reconstructs price via P·exp(Σr).
    CI = GBM volatility cone (±1.28σ√t).
    """
    X,y_ret = _build_features(df)
    Xa=X.values.astype(np.float32); ya=y_ret.values.astype(np.float32)
    prices  = df["Close"].reindex(y_ret.index)

    # target = NEXT-day log return
    y_next=np.roll(ya,-1)
    X_tr,y_tr=Xa[:-1],y_next[:-1]

    # Tree ensemble
    tree=TreeEnsemble()
    tree.fit(X_tr,y_tr,feat_names=list(X.columns))

    # In-sample metrics
    fitted_ret,_,_=tree.predict(X_tr)
    p_actual=prices.values[:-1]*np.exp(np.roll(ya,-1)[:-1])
    p_pred  =prices.values[:-1]*np.exp(fitted_ret)
    valid=p_actual[:-1]; pred_v=p_pred[:-1]
    mae   = float(mean_absolute_error(valid,pred_v))
    rmse  = float(np.sqrt(mean_squared_error(valid,pred_v)))
    mape  = float(np.mean(np.abs((valid-pred_v)/(np.abs(valid)+1e-9)))*100)
    dir_a = float(np.mean(np.sign(fitted_ret[:-1])==np.sign(y_tr[:-1]))*100)

    # Optional LSTM
    lstm=None; lstm_w=0.0
    if _TF and len(Xa)>=80:
        try:
            lstm=LSTMModel(n_feat=Xa.shape[1])
            lstm.fit(Xa[:-1],ya[:-1],epochs=35,batch=64)
            if lstm._fitted: lstm_w=0.35
        except: lstm=None

    # Predict 1-day log return from latest features
    mean_ret_tree,_,_=tree.predict(Xa[[-1]])
    mean_ret=float(mean_ret_tree[0])
    if lstm is not None and lstm._fitted:
        mr_lstm,_=lstm.predict(Xa)
        mean_ret=(1-lstm_w)*mean_ret+lstm_w*float(mr_lstm[0])

    # Historical vol for CI cone
    daily_vol=float(ya.std()); last_price=float(prices.iloc[-1])
    lo_cone,hi_cone=_vol_cone(last_price,daily_vol,horizon)

    # Multi-step: compound 1-day return
    future_dates=pd.bdate_range(prices.index[-1],periods=horizon+1)[1:]
    cum=0.0; rows=[]
    for i,dt in enumerate(future_dates):
        cum+=mean_ret
        p=last_price*np.exp(cum)
        rows.append({"Date":dt,"Forecast":round(p,4),
                     "Lower_80":round(float(lo_cone[i]),4),
                     "Upper_80":round(float(hi_cone[i]),4),
                     "Log_Return":round(mean_ret,6)})
    fc=pd.DataFrame(rows).set_index("Date")

    return {
        "forecast":fc, "in_sample_mae":round(mae,4),
        "in_sample_rmse":round(rmse,4), "in_sample_mape":round(mape,2),
        "in_sample_dir_acc":round(dir_a,1),
        "daily_volatility":round(daily_vol*100,4),
        "n_features":Xa.shape[1], "n_train":len(X_tr),
        "feature_importance":tree.feature_importance(),
        "models_available":{"xgboost":_XGB,"lightgbm":_LGB,"lstm":_TF},
        "lstm_used":lstm is not None and lstm._fitted,
        "lstm_weight":round(lstm_w,2),
    }


def walk_forward_backtest(df: pd.DataFrame, horizon:int=1,
                          n_folds:int=5, train_frac:float=0.65) -> Dict:
    X,y_ret=_build_features(df)
    Xa=X.values.astype(np.float32); ya=y_ret.values.astype(np.float32)
    dates=y_ret.index; prices=df["Close"].reindex(dates).values.astype(np.float32)

    y_tgt=np.roll(ya,-horizon); Xa=Xa[:-horizon]; y_tgt=y_tgt[:-horizon]
    prices=prices[:-horizon]; dates=dates[:-horizon]

    total=len(Xa); min_tr=int(total*train_frac)
    fold_sz=max((total-min_tr)//n_folds,10)
    fold_recs=[]; all_preds=[]; last_tree=None

    for fold in range(n_folds):
        ts=min_tr+fold*fold_sz; te=min(ts+fold_sz,total)
        if te<=ts: break
        Xtr,ytr=Xa[:ts],y_tgt[:ts]; Xte,yte=Xa[ts:te],y_tgt[ts:te]
        pte=prices[ts:te]; dte=dates[ts:te]

        tree=TreeEnsemble()
        tree.fit(Xtr,ytr,feat_names=list(X.columns))
        pr,lo,hi=tree.predict(Xte); last_tree=tree

        ap=pte*np.exp(yte); pp=pte*np.exp(pr)
        mae=float(mean_absolute_error(ap,pp))
        rmse=float(np.sqrt(mean_squared_error(ap,pp)))
        mape=float(np.mean(np.abs((ap-pp)/(np.abs(ap)+1e-9)))*100)
        dir_a=float(np.mean(np.sign(pr)==np.sign(yte))*100)
        fold_recs.append({"Fold":fold+1,"Train":ts,"Test":te-ts,
            "MAE":round(mae,2),"RMSE":round(rmse,2),
            "MAPE (%)":round(mape,2),"Dir. Accuracy (%)":round(dir_a,2)})
        for i,d in enumerate(dte):
            all_preds.append({"Date":d,"Actual":float(ap[i]),"Predicted":float(pp[i]),
                "Lower":float(pte[i]*np.exp(lo[i])),"Upper":float(pte[i]*np.exp(hi[i])),"Fold":fold+1})

    fold_df=pd.DataFrame(fold_recs)
    pred_df=(pd.DataFrame(all_preds).set_index("Date") if all_preds else pd.DataFrame())
    agg={}
    for col in ["MAE","RMSE","MAPE (%)","Dir. Accuracy (%)"]:
        if col in fold_df.columns:
            agg[f"{col}_mean"]=round(float(fold_df[col].mean()),3)
            agg[f"{col}_std"] =round(float(fold_df[col].std()),3)
    fi=last_tree.feature_importance() if last_tree else pd.DataFrame()
    return {"fold_metrics":fold_df,"aggregate":agg,"predictions":pred_df,"feature_importance":fi}
