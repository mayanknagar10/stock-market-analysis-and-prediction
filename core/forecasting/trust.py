"""Disjoint calibration, training-domain distance and conservative evidence policy."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from core.validation.alignment import aware_index

def _logit(probability):
    p=np.asarray(probability,dtype=float)
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)): raise ValueError('Invalid probability')
    p=np.clip(p,1e-6,1-1e-6)
    return np.log(p/(1-p)).reshape(-1,1)

class PlattCalibrator:
    def fit(self,probability,labels,times,base_label_cutoff):
        dates=aware_index(times,'calibration timestamps')
        cutoff=aware_index([base_label_cutoff],'base label cutoff')[0]
        values=_logit(probability)
        labels=np.asarray(labels,dtype=int)
        if len(dates)!=len(values) or labels.shape!=(len(values),) or len(values)<30 or len(np.unique(labels))!=2:
            raise ValueError('Insufficient/misaligned independent calibration data')
        if dates.min()<=cutoff: raise ValueError('Calibration overlaps base-model training labels')
        model=LogisticRegression(C=1.,random_state=42).fit(values,labels)
        self.coefficient=float(model.coef_[0,0])
        self.intercept=float(model.intercept_[0])
        self.n=len(values)
        self.start=dates.min().isoformat()
        self.end=dates.max().isoformat()
        return self

    def predict(self,probability):
        value=np.clip(self.coefficient*_logit(probability).ravel()+self.intercept,-40,40)
        return 1/(1+np.exp(-value))

class OODDetector:
    def fit(self,scaled):
        X=np.asarray(scaled,dtype=float)
        if X.ndim!=2 or not len(X) or not np.isfinite(X).all(): raise ValueError('Invalid OOD reference domain')
        self.threshold=float(max(np.quantile(np.sqrt(np.mean(X**2,axis=1)),.99),1e-6))
        self.n_features=X.shape[1]
        return self
    def score(self,scaled):
        X=np.asarray(scaled,dtype=float)
        if X.ndim!=2 or X.shape[1]!=self.n_features or not np.isfinite(X).all(): raise ValueError('OOD schema mismatch')
        return np.sqrt(np.mean(X**2,axis=1))/self.threshold

def learn_edge_threshold(actual,probability,predicted,cost=.0015):
    y=np.asarray(actual,dtype=float)
    p=np.asarray(probability,dtype=float)
    direction=np.where(p>=.5,1.,-1.)
    strength=np.maximum(p,1-p)
    evidence=[]
    selected=None
    for threshold in [.5,.55,.6,.65,.7,.75,.8,.85,.9]:
        mask=strength>=threshold
        count=int(mask.sum())
        if count<50: continue
        rate=float(((y[mask]>0)==(direction[mask]>0)).mean())
        z=1.645
        lower=(rate+z*z/(2*count)-z*np.sqrt(rate*(1-rate)/count+z*z/(4*count*count)))/(1+z*z/count)
        net=float(np.mean(direction[mask]*np.expm1(y[mask]))-cost)
        row={'threshold':threshold,'n':count,'observed_accuracy':rate,'wilson_lower_90':float(lower),'indicative_net_return':net}
        evidence.append(row)
        if selected is None and lower>.5 and net>0: selected=threshold
    return {'probability_edge_min':selected,'validation_bins':evidence,'cost_assumption':cost,
            'limitation':'Thresholds estimated on development validation only; overlapping observations reduce effective support'}

def evidence_policy(probability,mean_return,quantiles,component_returns,data_eligible,ood_score,
                    validation_support,calibrated,point_in_time_verified,final_test_passed,learned_threshold=None,cost=.0015):
    for value in [data_eligible,calibrated,point_in_time_verified,final_test_passed]:
        if type(value) is not bool: raise ValueError('Evidence flags require literal booleans')
    q=np.asarray(quantiles,dtype=float)
    components=np.asarray(component_returns,dtype=float)
    if q.shape!=(5,) or not np.isfinite(q).all() or np.any(np.diff(q)<0) or not np.isfinite(components).all():
        raise ValueError('Invalid distribution evidence')
    if not np.isfinite([probability,mean_return,ood_score]).all() or not 0<=probability<=1:
        raise ValueError('Invalid probability/return/OOD evidence')
    sign=1 if probability>=.5 else -1
    agreement=float(np.mean(np.sign(components)==sign)) if len(components) else 0.
    reasons=[]
    if not data_eligible: reasons.append('DATA_UNHEALTHY')
    if not point_in_time_verified: reasons.append('RESEARCH_ONLY')
    if not final_test_passed: reasons.append('FINAL_TEST_NOT_PASSED')
    if not calibrated: reasons.append('CALIBRATION_UNVERIFIED')
    if ood_score>1: reasons.append('OUT_OF_DISTRIBUTION')
    if agreement<1 or np.sign(mean_return)!=sign: reasons.append('MODEL_DISAGREEMENT')
    if learned_threshold is None or max(probability,1-probability)<learned_threshold:
        reasons.append('NO_VALIDATED_PROBABILITY_EDGE')
    if abs(np.expm1(mean_return))<=cost: reasons.append('RETURN_BELOW_COST')
    if q[0]<=0<=q[4]: reasons.append('INTERVAL_SPANS_ZERO')
    if validation_support<50: reasons.append('INSUFFICIENT_SUPPORT')
    return {'confidence':'LOW' if reasons else 'MEDIUM','abstain':bool(reasons),
            'status':'NO STRONG STATISTICAL EDGE' if reasons else 'VALIDATED RESEARCH EDGE',
            'agreement':agreement,'reasons':reasons,'evidence':{'calibrated_probability':probability if calibrated else None,'raw_probability':None if calibrated else probability,
                'interval_width_80':float(q[4]-q[0]),'ood_ratio':ood_score,'validation_support':validation_support,
                'data_eligible':data_eligible,'point_in_time_verified':point_in_time_verified}}
