"""Direction and conditional quantiles per horizon, from both retained tree families."""
import numpy as np

LEVELS=(.1,.25,.5,.75,.9)

def ordered_quantiles(values):
    q=np.asarray(values,dtype=float)
    if q.ndim!=2 or q.shape[1]!=5 or not np.isfinite(q).all(): raise ValueError('Invalid quantile matrix')
    crossed=np.any(np.diff(q,axis=1)<0,axis=1)
    return np.sort(q,axis=1),crossed

class DistributionHeads:
    def __init__(self,n_estimators=120):
        self.n_estimators=n_estimators
        self.classifiers={}
        self.quantile_models={}
        self.weights=np.array([.5,.5])
        self.class_weights=np.array([.5,.5])

    def fit(self,scaled,y):
        import xgboost as xgb
        import lightgbm as lgb
        X=np.asarray(scaled,dtype=float)
        y=np.asarray(y,dtype=float)
        if X.ndim!=2 or y.shape!=(len(X),) or not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError('Invalid distribution fitting data')
        labels=(y>0).astype(int)
        if len(np.unique(labels))!=2: raise ValueError('Both direction classes required')
        common=dict(n_estimators=self.n_estimators,learning_rate=.04,max_depth=3,subsample=.85,
                    colsample_bytree=.8,reg_lambda=2.,random_state=42,n_jobs=2)
        self.classifiers={'xgb':xgb.XGBClassifier(**common,objective='binary:logistic',tree_method='hist',verbosity=0),
                          'lgb':lgb.LGBMClassifier(**common,objective='binary',num_leaves=15,min_child_samples=40,verbose=-1)}
        for model in self.classifiers.values(): model.fit(X,labels)
        for level in LEVELS:
            self.quantile_models[str(level)]={
                'xgb':xgb.XGBRegressor(**common,objective='reg:quantileerror',quantile_alpha=level,tree_method='hist',verbosity=0),
                'lgb':lgb.LGBMRegressor(**common,objective='quantile',alpha=level,num_leaves=15,min_child_samples=40,verbose=-1)}
            for model in self.quantile_models[str(level)].values(): model.fit(X,y)
        return self

    def predict(self,scaled):
        X=np.asarray(scaled,dtype=float)
        if not self.classifiers or not np.isfinite(X).all(): raise ValueError('Missing distribution model/nonfinite features')
        components={name:(model.predict_proba(X)[:,1] if hasattr(model,'predict_proba') else model.predict(X))
                    for name,model in self.classifiers.items()}
        probability=sum(w*p for w,p in zip(self.class_weights,components.values()))
        columns=[]
        for models in self.quantile_models.values():
            columns.append(sum(w*m.predict(X) for w,m in zip(self.weights,models.values())))
        quantiles,crossed=ordered_quantiles(np.column_stack(columns))
        return {'probability':np.clip(probability,0,1),'probability_components':components,
                'quantiles':quantiles,'quantile_crossing_corrected':crossed}
