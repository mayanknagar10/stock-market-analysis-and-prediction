"""Direct market/horizon regressors, retaining both XGBoost and LightGBM."""
from __future__ import annotations
import numpy as np
import pandas as pd
from core.validation.preprocessing import fit_training_scaler
from core.validation.contracts import schema_hash

def price_from_log_return(price,log_return):
    if not np.isfinite(price) or price<=0 or not np.isfinite(log_return):
        raise ValueError('Price and log return must be finite; price positive')
    try:
        with np.errstate(over='raise',under='raise'): result=float(price*np.exp(log_return))
    except FloatingPointError as exc: raise ValueError('Forecast price overflow') from exc
    if not np.isfinite(result) or result<=0: raise ValueError('Invalid forecast price')
    return result

class DirectRegressor:
    def __init__(self,family,horizon,n_estimators=120):
        if family not in {'india_equity','us_equity'} or type(horizon) is not int or horizon not in {1,5,10,20}:
            raise ValueError('Unsupported model family/horizon')
        self.family=family
        self.horizon=horizon
        self.n_estimators=n_estimators
        self.models={}
        self.feature_names=()
        self.weights=np.array([.5,.5])

    def fit(self,X,y,feature_times,label_ends,fit_before):
        import xgboost as xgb
        import lightgbm as lgb
        if not isinstance(X,pd.DataFrame): raise ValueError('Named feature frame required')
        self.feature_names=tuple(X.columns)
        self.schema_hash=schema_hash(self.feature_names)
        y=np.asarray(y,dtype=float)
        if y.shape!=(len(X),) or not np.isfinite(y).all(): raise ValueError('Invalid direct targets')
        self.scaler=fit_training_scaler(X,feature_times,label_ends,fit_before)
        scaled=self.scaler.transform(X)
        common=dict(n_estimators=self.n_estimators,learning_rate=.04,max_depth=3,
                    subsample=.85,colsample_bytree=.8,reg_lambda=2.,random_state=42,n_jobs=2)
        self.models={'xgb':xgb.XGBRegressor(**common,objective='reg:squarederror',tree_method='hist',verbosity=0),
                     'lgb':lgb.LGBMRegressor(**common,objective='regression',num_leaves=15,
                                            min_child_samples=40,verbose=-1,deterministic=True,force_col_wise=True)}
        for model in self.models.values(): model.fit(scaled,y)
        self.training_cutoff=str(pd.DatetimeIndex(label_ends).max())
        return self

    def scaled(self,X):
        if not self.models or not isinstance(X,pd.DataFrame) or tuple(X.columns)!=self.feature_names:
            raise ValueError('Missing fitted model or feature schema/order mismatch')
        if not np.isfinite(X.to_numpy(dtype=float)).all(): raise ValueError('Missing/nonfinite feature values')
        return self.scaler.transform(X)

    def predict(self,X):
        scaled=self.scaled(X)
        components={name:np.asarray(model.predict(scaled),dtype=float) for name,model in self.models.items()}
        central=sum(weight*values for weight,values in zip(self.weights,components.values()))
        if not np.isfinite(central).all(): raise ValueError('Invalid model return output')
        return {'central_log_return':central,'components':components}

    def learn_weights(self,X_oof,y_oof):
        predictions=self.predict(X_oof)['components']
        errors=np.array([np.mean((p-np.asarray(y_oof))**2) for p in predictions.values()])
        inverse=1/np.maximum(errors,1e-12)
        self.weights=inverse/inverse.sum()
        return self.weights
