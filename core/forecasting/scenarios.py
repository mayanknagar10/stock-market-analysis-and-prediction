"""Counterfactual source perturbations rerun the identical research feature/inference pipeline."""
import numpy as np
import pandas as pd
from core.data.contracts import DataSnapshot
from core.forecasting.context import research_feature_frame
from core.forecasting.inference import infer_bundle
from core.validation.alignment import session_cutoffs

class ScenarioEngine:
    def __init__(self,bundle,stock,ticker,family,references,settings,current_price,as_of):
        self.bundle=bundle; self.stock=stock.copy(); self.ticker=ticker; self.family=family
        self.references=references; self.settings=settings; self.current_price=current_price; self.as_of=as_of
    def run(self,shocks):
        stock=self.stock.copy()
        references=dict(self.references)
        price=self.current_price
        cutoff=(session_cutoffs(stock.index,self.family)+pd.Timedelta('20min'))[-1]
        for key,shock in shocks.items():
            if not np.isfinite(shock) or not -.4<=shock<=.4: raise ValueError('Scenario shocks must be finite changes between -40% and +40%')
            if key=='stock':
                stock.loc[stock.index[-1],['Open','High','Low','Close']]*=1+shock
                price*=1+shock
                continue
            source_key=('market_india' if self.family=='india_equity' else 'market_us') if key=='market' else self.settings.get('sector_map',{}).get(self.ticker) if key=='sector' else key
            if source_key not in references: raise ValueError('Unsupported or missing scenario source: '+str(key))
            source=references[source_key]
            frame=source.frame if hasattr(source,'metadata') else source.copy()
            zone=source.metadata.exchange_timezone if hasattr(source,'metadata') else None
            local=frame.index.tz_convert(zone).tz_localize(None).normalize() if zone else frame.index.tz_localize(None).normalize() if frame.index.tz is not None else frame.index.normalize()
            available=pd.DatetimeIndex(local).tz_localize('UTC')+pd.Timedelta('1D')
            visible=np.flatnonzero(available<=cutoff)
            if not len(visible): raise ValueError('No reference observation available for scenario cutoff')
            position=int(visible[-1])
            columns=[c for c in ['Open','High','Low','Close','Adj Close'] if c in frame]
            frame.loc[frame.index[position],columns]*=1+shock
            references[source_key]=DataSnapshot(frame,source.metadata) if hasattr(source,'metadata') else frame
        base,_=research_feature_frame(self.stock,self.ticker,self.family,self.references,self.settings,research_only=True)
        rebuilt,_=research_feature_frame(stock,self.ticker,self.family,references,self.settings,research_only=True)
        names=list(self.bundle['regression'].feature_names)
        if not set(names).issubset(rebuilt.columns): raise ValueError('Scenario feature schema unsupported')
        row=rebuilt.iloc[[-1]][names]
        old=base.iloc[[-1]][names]
        changed=[name for name in names if not np.isclose(row[name].iloc[0],old[name].iloc[0],equal_nan=True)]
        if any(value!=0 for value in shocks.values()) and not changed: raise ValueError('These sources are not active features of this horizon model')
        result=infer_bundle(self.bundle,row,price,self.as_of,data_eligible=True,research_only=True)
        result.update(scope='hypothetical_research',scenario_shocks=dict(shocks),changed_features=changed)
        return result
    def sensitivity(self):
        results=[]
        for source in ['stock','market','sector','nasdaq','usd_inr','brent','vix']:
            for change in [-.1,-.05,.05,.1]:
                try:
                    forecast=self.run({source:change})
                    results.append({'source':source,'change':change,'return':forecast['central_return'],
                                    'p_positive':forecast['p_positive'],'abstain':forecast['trust']['abstain']})
                except ValueError: continue
        return results
    def invalidation(self,base_forecast):
        if base_forecast['trust']['abstain']:
            return {'status':'NO_BASE_EDGE','explanation':'The base forecast already abstains; no actionable thesis to invalidate',
                    'sensitivity':self.sensitivity()}
        return {'status':'SENSITIVITY_ONLY','sensitivity':self.sensitivity(),
                'explanation':'Controlled source perturbations are model associations, not causal stress predictions'}
