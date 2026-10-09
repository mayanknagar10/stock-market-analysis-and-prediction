"""Registered, bounded, retrospective development experiments; no promotion API.

Only clipped inputs enter panel construction. Reference future returns are labels,
never predictors; excess reconstruction uses independently predicted components.
Fits are ephemeral. Every attempt and train-only preprocessing audit is retained.
"""
import hashlib
import json
import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, brier_score_loss
from threadpoolctl import threadpool_limits
from research.post_v5.development import clip_frame, design_frame, fixed_folds, BOUNDARY

BLOCKED={'actual','market_future','sector_future','rank_target','feature_time','label_end','origin_session','maturity_session','ticker','industry','market_regime','trend_state'}
BASELINES=['zero','historical_mean','momentum20','reversion5','ridge','elastic_net','shallow_boost','market_only','market_sector']
GROUPS=['technical','market','sector','global','fx','commodity']


def exclusive_json(path, payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(payload,f,allow_nan=False,default=str)


def predictor_columns(panel,groups):
    return list(dict.fromkeys(c for values in groups.values() for c in values if c in panel and c not in BLOCKED and not c.endswith('_future')))


def rank_target(panel):
    return panel.groupby('origin_session',sort=False).actual.rank(pct=True,method='average').to_numpy()


def reconstruct(target,pred,market_pred,sector_excess_pred,vol):
    if target=='market_excess':return pred+market_pred
    if target=='sector_excess':return pred+market_pred+sector_excess_pred
    if target=='vol_normalized':return pred*vol
    return pred


def reference_close(value):
    frame=clip_frame(value.frame if hasattr(value,'frame') else value)
    idx=pd.DatetimeIndex(frame.index)
    if idx.tz is not None:
        idx=idx.tz_convert(value.metadata.exchange_timezone).tz_localize(None).normalize() if hasattr(value,'metadata') else idx.tz_localize(None).normalize()
    close=frame['Adj Close'] if 'Adj Close' in frame else frame.Close
    close=pd.Series(close.to_numpy(),index=idx)
    return close.loc[~close.index.duplicated(keep=False)].sort_index()


def build_panel(frames,references,settings,universe,horizon):
    frames={k:clip_frame(v) for k,v in frames.items()}
    membership={m['ticker']:m['industry'] for m in universe['members']}
    market=reference_close(references['market_india']) if 'market_india' in references else None
    calendar=market.index if market is not None else pd.DatetimeIndex(sorted(set(d for v in frames.values() for d in v.index)))
    calendar=calendar[calendar<pd.Timestamp('2025-10-01')]
    closes=pd.DataFrame({k:v.Close.reindex(calendar) for k,v in frames.items()},index=calendar)
    daily=np.log(closes/closes.shift(1)).replace([np.inf,-np.inf],np.nan)
    panels=[];eligibility=[];groups={g:[] for g in GROUPS}
    for ticker,industry in membership.items():
        if ticker not in frames or frames[ticker].empty:
            eligibility.append({'ticker':ticker,'industry':industry,'status':'NO_PRE_BOUNDARY_HISTORY','rows':0});continue
        try:
            p=design_frame(frames[ticker],horizon,references or None,ticker,settings,calendar)
            if p.empty:
                eligibility.append({'ticker':ticker,'industry':industry,'status':'NO_VALID_FEATURE_LABEL_ROWS','rows':0});continue
            native=p.attrs.get('feature_groups',{})
            for g,cols in native.items():
                for c in cols:
                    dest='technical' if g in {'stock','technical'} else g
                    if c.startswith(('usd_inr_','dxy_')):dest='fx'
                    elif c.startswith(('brent_','wti_','gold_')):dest='commodity'
                    if dest in groups and c not in groups[dest]:groups[dest].append(c)
            origins=pd.DatetimeIndex(pd.to_datetime(p.origin_session));ends=pd.DatetimeIndex(pd.to_datetime(p.maturity_session))
            p['market_future']=np.log(market.reindex(ends).to_numpy()/market.reindex(origins).to_numpy()) if market is not None else np.nan
            sector_key=settings.get('sector_map',{}).get(ticker)
            sector=reference_close(references[sector_key]) if sector_key in references else None
            if sector is not None and np.isfinite(sector.to_numpy()).sum()>=252:
                p['sector_future']=np.log(sector.reindex(ends).to_numpy()/sector.reindex(origins).to_numpy())
                p['sector_source']='official_reference_reconstructed'
            else:
                peers=[t for t,i in membership.items() if i==industry and t!=ticker and t in daily]
                peer=daily[peers].mean(axis=1).where(daily[peers].notna().sum(axis=1)>=2) if len(peers)>=2 else pd.Series(np.nan,index=calendar)
                p['sector_future']=peer.rolling(horizon,min_periods=horizon).sum().shift(-horizon).reindex(origins).to_numpy()
                for n in [1,5,20]:
                    c='sector_ret_'+str(n)+'d';p[c]=peer.rolling(n,min_periods=n).sum().shift(1).reindex(origins).to_numpy()
                    if c not in groups['sector']:groups['sector'].append(c)
                for c,v in {'sector_vol_20':peer.rolling(20).std(),'sector_trend_50':peer.rolling(50).sum(),'sector_minus_market_5d':None}.items():
                    p[c]=v.shift(1).reindex(origins).to_numpy() if v is not None else p.sector_ret_5d-p.get('market_ret_5d',np.nan)
                    if c not in groups['sector']:groups['sector'].append(c)
                p['sector_source']='fixed_universe_equal_weight_ex_stock_min2_peer_proxy'
            p['ticker']=ticker;p['industry']=industry
            p['trend_state']=np.where(np.abs(p.get('ret_20d',0))>.02,'trend','range')
            p.attrs={};panels.append(p)
            eligibility.append({'ticker':ticker,'industry':industry,'status':'ELIGIBLE','rows':len(p),'sector_labeled_rows':int(p.sector_future.notna().sum()),'sector_source':p.sector_source.iloc[0]})
        except (ValueError,KeyError,TypeError) as exc:
            eligibility.append({'ticker':ticker,'industry':industry,'status':'INVALID_SOURCE_OR_SCHEMA','rows':0,'reason':str(exc)})
            raise ValueError('Stop experiment on corrupt/source schema: '+ticker+': '+str(exc)) from exc
    if not panels:raise ValueError('No eligible development panel')
    panel=pd.concat(panels,ignore_index=True).sort_values(['feature_time','ticker'],kind='stable').reset_index(drop=True)
    panel['rank_target']=rank_target(panel)
    panel.attrs={'feature_groups':groups,'feature_columns':predictor_columns(panel,groups),'historical_point_in_time_certified':False}
    if pd.to_datetime(panel.feature_time,utc=True).max()>=BOUNDARY or pd.to_datetime(panel.label_end,utc=True).max()>=BOUNDARY:raise ValueError('Consumed period entered experiment')
    return panel,eligibility


def fit_transform(train,test,prune=False,target=None):
    """Fit median, variance, correlation and temporal stability using train only."""
    tr=train.replace([np.inf,-np.inf],np.nan)
    med=tr.median().fillna(0.)
    a=tr.fillna(med);b=test.replace([np.inf,-np.inf],np.nan).fillna(med)
    std=a.std(ddof=0);cols=[c for c in a if std[c]>1e-12]
    if not cols:raise ValueError('No nonconstant training features')
    if prune:
        corr=a[cols].corr().abs();keep=[]
        for c in cols:
            if not any(corr.loc[c,k]>.95 for k in keep):keep.append(c)
        cols=keep
        if target is not None and len(a)>100:
            middle=len(a)//2
            first=a[cols].iloc[:middle].corrwith(pd.Series(np.asarray(target)[:middle],index=a.index[:middle]),method='spearman')
            last=a[cols].iloc[middle:].corrwith(pd.Series(np.asarray(target)[middle:],index=a.index[middle:]),method='spearman')
            stable=[c for c in cols if first[c]*last[c]>0 and min(abs(first[c]),abs(last[c]))>=.01]
            strength=(first.abs()+last.abs())/2
            cols=sorted(stable,key=lambda c:-strength[c])[:24] or cols[:5]
    mean=a[cols].mean();scale=a[cols].std(ddof=0)
    return {'train':((a[cols]-mean)/scale).to_numpy(dtype=np.float64),'test':((b[cols]-mean)/scale).to_numpy(dtype=np.float64),'columns':cols,'fit_rows':len(train),'medians':med.to_dict(),'means':mean.to_dict(),'scales':scale.to_dict()}


def correlation(x,y):
    ok=np.isfinite(x)&np.isfinite(y)
    if ok.sum()<8 or np.std(x[ok])==0 or np.std(y[ok])==0:return None
    value=float(spearmanr(x[ok],y[ok]).statistic)
    return value if np.isfinite(value) else None


def regression_metrics(actual,pred):
    ok=np.isfinite(actual)&np.isfinite(pred)
    if not ok.any():return {'n':0,'mae':None,'rmse':None,'direction_accuracy':None,'ic':None}
    actual=actual[ok];pred=pred[ok];error=pred-actual
    return {'n':int(len(actual)),'mae':float(np.mean(np.abs(error))),'rmse':float(np.sqrt(np.mean(error**2))),'direction_accuracy':float(np.mean((pred>0)==(actual>0))),'ic':correlation(actual,pred)}


def date_block_summary(frame,horizon):
    """Nonoverlapping date blocks; stocks within one date are not independent."""
    dates=sorted(frame.origin_session.unique());mapping={d:i//horizon for i,d in enumerate(dates)}
    f=frame.copy();f['block']=f.origin_session.map(mapping)
    blocks=f.groupby('block').apply(lambda g:pd.Series({'mae':np.mean(np.abs(g.prediction-g.actual)),'rank_ic':np.nanmean([v for v in [correlation(x.actual.to_numpy(),x.prediction.to_numpy()) for _,x in g.groupby('origin_session')] if v is not None]) if g.prediction.nunique()>1 else np.nan}),include_groups=False)
    return {'dates':len(dates),'nonoverlap_blocks':len(blocks),'block_mae_mean':float(blocks.mae.mean()),'block_mae_se':float(blocks.mae.std(ddof=1)/np.sqrt(len(blocks))) if len(blocks)>1 else None,'date_rank_ic':float(blocks.rank_ic.mean()) if blocks.rank_ic.notna().any() else None}


def make_interactions(train,test,columns):
    """Identity vocabulary comes exclusively from the training set."""
    a=train[columns].copy();b=test[columns].copy();industries=sorted(train.industry.unique())
    base=[c for c in ['ret_1d','ret_5d','ret_20d','market_ret_5d','sector_ret_5d','volatility_20'] if c in columns]
    for i,industry in enumerate(industries):
        name='industry_'+str(i);a[name]=(train.industry==industry).astype(float);b[name]=(test.industry==industry).astype(float)
        for c in base:a[name+'_'+c]=a[name]*train[c];b[name+'_'+c]=b[name]*test[c]
    return a,b,industries


def evaluate_panel(panel,folds,horizon,models=None,progress_callback=None):
    groups=panel.attrs['feature_groups'];columns=panel.attrs['feature_columns']
    models=models or BASELINES+[g+'_only' for g in GROUPS if g!='market']+['without_'+g for g in GROUPS]+['reduced_stable','vol_normalized','market_excess','sector_excess','rank','positive_direction','sector_specific','sector_interactions','decomposition','sector_heldout']
    predictions=[];attempts=[];audits=[];signals=[];target_scores=[];support=[]
    metadata=panel[['feature_time','label_end']]
    with threadpool_limits(limits=2):
      for fold,(tridx,teidx) in enumerate(fixed_folds(metadata,folds),1):
        tr=panel.iloc[tridx].copy();te=panel.iloc[teidx].copy();y=tr.actual.to_numpy();yt=te.actual.to_numpy()
        full=fit_transform(tr[columns],te[columns]);trainmax=str(pd.to_datetime(tr.feature_time,utc=True).max());labelmax=str(pd.to_datetime(tr.label_end,utc=True).max())
        cache={};head_cache={}
        def fit_predict(a,b,target,kind='ridge',prune=False,label=''):
            prep=fit_transform(a,b,prune,target)
            model=Ridge(alpha=10.) if kind=='ridge' else ElasticNet(alpha=.001,l1_ratio=.5,max_iter=2000,selection='cyclic') if kind=='elastic_net' else HistGradientBoostingRegressor(max_iter=50,max_depth=2,min_samples_leaf=100,learning_rate=.03,random_state=42,early_stopping=False)
            model.fit(prep['train'],target)
            pred=model.predict(prep['test']);trainpred=model.predict(prep['train'])
            audits.append({'horizon':horizon,'fold':fold,'model':label,'fit_rows':len(a),'fit_feature_max':trainmax,'fit_label_max':labelmax,'columns':prep['columns'],'medians':prep['medians'],'means':prep['means'],'scales':prep['scales'],'industry_vocabulary':None})
            return pred,trainpred,prep['columns']
        def heads():
            if head_cache:return head_cache['market'],head_cache['sector']
            mcols=[c for c in groups['market'] if c in columns and not c.startswith(('stock_minus','rolling_market'))]
            mtr=tr.drop_duplicates('origin_session');valid=mtr.market_future.notna()
            if len(mcols) and valid.sum()>=100:
                mp,_,_=fit_predict(mtr.loc[valid,mcols],te[mcols],mtr.loc[valid,'market_future'].to_numpy(),label='independent_market_head')
            else:mp=np.full(len(te),np.nan)
            scols=list(dict.fromkeys(groups['market']+groups['sector']));scols=[c for c in scols if c in columns]
            valid=tr.sector_future.notna()&tr.market_future.notna()
            if len(scols) and valid.sum()>=100:
                a,b,vocab=make_interactions(tr.loc[valid],te,scols)
                sp,_,_=fit_predict(a,b,(tr.sector_future-tr.market_future).loc[valid].to_numpy(),label='independent_sector_excess_head')
                audits[-1]['industry_vocabulary']=vocab
            else:sp=np.full(len(te),np.nan)
            head_cache['market']=mp;head_cache['sector']=sp
            return mp,sp
        for name in models:
            attempt={'horizon':horizon,'fold':fold,'model':name,'train_n':len(tr),'test_n':len(te),'train_dates':tr.origin_session.nunique(),'test_dates':te.origin_session.nunique(),'status':'started'}
            if progress_callback:progress_callback(dict(attempt))
            try:
                trainmetric=None;used=columns;target='cumulative_log';pred=None
                if name=='zero':pred=np.zeros(len(te));used=[]
                elif name=='historical_mean':pred=np.full(len(te),float(np.mean(y)));used=[]
                elif name=='momentum20':pred=te.ret_20d.to_numpy()*horizon/20.;used=['ret_20d']
                elif name=='reversion5':pred=-te.ret_5d.to_numpy()*horizon/5.;used=['ret_5d']
                elif name in {'ridge','elastic_net','shallow_boost'}:
                    pred,tp,used=fit_predict(tr[columns],te[columns],y,name,label=name);trainmetric=regression_metrics(y,tp)
                elif name in {'market_only','market_sector'} or name.endswith('_only') or name.startswith('without_') or name=='reduced_stable':
                    if name=='market_sector':used=groups['market']+groups['sector']
                    elif name.endswith('_only'):used=groups[name.removesuffix('_only')]
                    elif name.startswith('without_'):used=[c for c in columns if c not in groups[name.removeprefix('without_')]]
                    else:used=columns
                    used=[c for c in used if c in columns]
                    if name in {'market_only','market_sector','sector_only'}:
                        used=[c for c in used if not c.startswith(('stock_minus','rolling_market'))]
                    if not used:raise ValueError('Feature group absent')
                    pred,tp,used=fit_predict(tr[used],te[used],y,prune=name=='reduced_stable',label=name);trainmetric=regression_metrics(y,tp)
                elif name in {'vol_normalized','market_excess','sector_excess','rank'}:
                    target=name
                    voltr=tr.volatility_20.to_numpy()*np.sqrt(horizon);volte=te.volatility_20.to_numpy()*np.sqrt(horizon)
                    labels=y/voltr if name=='vol_normalized' else tr.rank_target.to_numpy() if name=='rank' else y-tr.market_future.to_numpy() if name=='market_excess' else y-tr.sector_future.to_numpy()
                    valid=np.isfinite(labels)
                    raw,tp,used=fit_predict(tr.loc[valid,columns],te[columns],labels[valid],label=name)
                    actualtarget=yt/volte if name=='vol_normalized' else te.rank_target.to_numpy() if name=='rank' else yt-te.market_future.to_numpy() if name=='market_excess' else yt-te.sector_future.to_numpy()
                    score={'horizon':horizon,'fold':fold,'target':name,**regression_metrics(actualtarget,raw),'train_target_mae':float(np.mean(abs(tp-labels[valid])))}
                    if name=='rank':
                        score['metric_space']='same_origin_percentile';target_scores.append(score)
                        pred=raw;target='rank'
                    else:
                        score['metric_space']='ex_ante_vol_units' if name=='vol_normalized' else 'excess_log_return';target_scores.append(score)
                        mp,sp=heads() if name!='vol_normalized' else (np.zeros(len(te)),np.zeros(len(te)))
                        pred=reconstruct(name,raw,mp,sp,volte)
                elif name=='positive_direction':
                    labels=(np.expm1(y)>.003).astype(int)
                    model=HistGradientBoostingClassifier(max_iter=50,max_depth=2,min_samples_leaf=100,learning_rate=.03,random_state=42,early_stopping=False)
                    model.fit(full['train'],labels);raw=model.predict_proba(full['test'])[:,1];truth=(np.expm1(yt)>.003).astype(int)
                    target_scores.append({'horizon':horizon,'fold':fold,'target':name,'metric_space':'raw_uncalibrated_classifier','n':len(te),'roc_auc':float(roc_auc_score(truth,raw)) if len(np.unique(truth))==2 else None,'balanced_accuracy':float(balanced_accuracy_score(truth,raw>=.5)),'brier':float(brier_score_loss(truth,raw)),'train_rate_brier':float(brier_score_loss(truth,np.full(len(truth),labels.mean()))),'coin_flip_brier':.25,'train_roc_auc':float(roc_auc_score(labels,model.predict_proba(full['train'])[:,1])), 'date_rank_ic':float(np.nanmean([correlation(raw[te.origin_session.to_numpy()==d],truth[te.origin_session.to_numpy()==d]) or np.nan for d in te.origin_session.unique()]))})
                    audits.append({'horizon':horizon,'fold':fold,'model':name,'fit_rows':len(tr),'fit_feature_max':trainmax,'fit_label_max':labelmax,'columns':columns,'medians':full['medians'],'means':full['means'],'scales':full['scales']})
                    pred=raw;target='positive_direction'
                elif name=='sector_specific':
                    pred=np.full(len(te),np.nan)
                    for industry in sorted(te.industry.unique()):
                        a=tr.loc[tr.industry==industry];mask=te.industry==industry;b=te.loc[mask]
                        enough=len(a)>=2000 and a.origin_session.nunique()>=252 and a.ticker.nunique()>=2
                        support.append({'horizon':horizon,'fold':fold,'industry':industry,'train_n':len(a),'train_dates':a.origin_session.nunique(),'train_stocks':a.ticker.nunique(),'test_n':len(b),'supported':enough})
                        if enough:pred[mask],_,_=fit_predict(a[columns],b[columns],a.actual.to_numpy(),label=name+':'+industry)
                elif name=='sector_interactions':
                    a,b,vocab=make_interactions(tr,te,columns);pred,tp,used=fit_predict(a,b,y,label=name);audits[-1]['industry_vocabulary']=vocab;trainmetric=regression_metrics(y,tp)
                elif name=='decomposition':
                    mp,sp=heads();valid=tr.sector_future.notna()&tr.market_future.notna()
                    alpha,_,used=fit_predict(tr.loc[valid,columns],te[columns],(tr.actual-tr.sector_future).loc[valid].to_numpy(),label='independent_stock_alpha_head')
                    pred=mp+sp+alpha
                elif name=='sector_heldout':
                    pred=np.full(len(te),np.nan)
                    # One fit per industry, held out in its entirety. Training-only identities.
                    for industry in sorted(te.industry.unique()):
                        a=tr.loc[tr.industry!=industry];mask=te.industry==industry;b=te.loc[mask]
                        pred[mask],_,_=fit_predict(a[columns],b[columns],a.actual.to_numpy(),label=name+':'+industry)
                else:raise ValueError('Unregistered model')
                if pred is not None:
                    if name in {'sector_excess','decomposition'}:
                        pred=np.where(te.sector_future.notna()&te.market_future.notna(),pred,np.nan)
                    elif name=='market_excess':
                        pred=np.where(te.market_future.notna(),pred,np.nan)
                    cols=['ticker','industry','origin_session','feature_time','label_end','market_regime','trend_state','sector_source','volatility_20']
                    result=te[cols].copy();result['actual']=yt;result['prediction']=pred;result['fold']=fold;result['horizon']=horizon;result['model']=name;result['target']=target
                    predictions.append(result)
                    actualscore=te.rank_target.to_numpy() if target=='rank' else (np.expm1(yt)>.003).astype(float) if target=='positive_direction' else yt
                    attempt.update({'status':'complete','feature_count':len(used),'train_metrics':trainmetric,'validation_metrics':regression_metrics(actualscore,pred),'target':target})
            except ValueError as exc:
                attempt.update({'status':'unsupported','reason':str(exc)})
            attempts.append(attempt)
            if progress_callback:progress_callback(dict(attempt))
        # Univariate and market-conditional signal; train and validation residualizers fit train only.
        if 'market_ret_5d' in columns:
            condcols=[c for c in groups['market'] if c in columns]
            cond=fit_transform(tr[condcols],te[condcols]);ym=Ridge(alpha=10).fit(cond['train'],y)
            yr=y-ym.predict(cond['train']);yvr=yt-ym.predict(cond['test'])
            xm=Ridge(alpha=10).fit(cond['train'],full['train']);xr=full['train']-xm.predict(cond['train']);xvr=full['test']-xm.predict(cond['test'])
        else:yr=y;yvr=yt;xr=full['train'];xvr=full['test']
        for i,c in enumerate(full['columns']):
            signals.append({'horizon':horizon,'fold':fold,'feature':c,'train_ic':correlation(full['train'][:,i],y),'validation_ic':correlation(full['test'][:,i],yt),'train_residual_ic':correlation(xr[:,i],yr),'validation_residual_ic':correlation(xvr[:,i],yvr),'train_rows':len(tr),'train_dates':tr.origin_session.nunique()})
        print(json.dumps({'horizon':horizon,'fold':fold,'train_rows':len(tr),'test_rows':len(te),'models_complete':sum(a['status']=='complete' for a in attempts if a['fold']==fold),'message':'fold complete'}),flush=True)
    return {'predictions':pd.concat(predictions,ignore_index=True),'attempts':attempts,'audits':audits,'signals':signals,'target_scores':target_scores,'sector_support':support}


def summarize(predictions):
    records=[];strata=[];paired=[]
    returns=predictions.loc[~predictions.target.isin(['rank','positive_direction'])].copy()
    for (horizon,model),f in returns.groupby(['horizon','model']):
        record={'horizon':int(horizon),'model':model,**regression_metrics(f.actual.to_numpy(),f.prediction.to_numpy())}
        finite=f.loc[np.isfinite(f.prediction)]
        if len(finite):record.update(date_block_summary(finite,int(horizon)))
        records.append(record)
        for col in ['ticker','industry','fold','market_regime','trend_state']:
            for value,g in f.groupby(col):
                strata.append({'horizon':int(horizon),'model':model,'stratum':col,'value':str(value),'dates':g.origin_session.nunique(),**regression_metrics(g.actual.to_numpy(),g.prediction.to_numpy())})
    # Pair every model with every simple baseline on identical finite rows.
    keys=['horizon','fold','ticker','origin_session']
    for model,f in returns.groupby('model'):
        for baseline in ['zero','historical_mean','momentum20','reversion5']:
            b=returns.loc[returns.model==baseline,keys+['prediction']].rename(columns={'prediction':'baseline'})
            joined=f.merge(b,on=keys,validate='one_to_one');joined=joined.loc[np.isfinite(joined.prediction)&np.isfinite(joined.baseline)]
            for horizon,g in joined.groupby('horizon'):
                err=abs(g.prediction-g.actual);berr=abs(g.baseline-g.actual)
                stock=(err-berr).groupby(g.ticker).mean();fold=(err-berr).groupby(g.fold).mean();date=(err-berr).groupby(g.origin_session).mean()
                blocks=pd.DataFrame({'date':date.index,'delta':date.to_numpy()});blocks['block']=np.arange(len(blocks))//int(horizon);block=blocks.groupby('block').delta.mean()
                paired.append({'horizon':int(horizon),'model':model,'baseline':baseline,'n':len(g),'stocks':len(stock),'mae_improvement':float(1-err.mean()/berr.mean()),'stocks_improved_fraction':float((stock<0).mean()),'folds_improved':int((fold<0).sum()),'nonoverlap_blocks':len(block),'block_delta_mean':float(block.mean()),'block_delta_se':float(block.std(ddof=1)/np.sqrt(len(block))) if len(block)>1 else None})
    # Explicit sector-specific vs pooled pair; unsupported industry rows never enter comparison.
    for horizon,f in returns.loc[returns.model=='sector_specific'].groupby('horizon'):
        b=returns.loc[(returns.model=='ridge')&(returns.horizon==horizon),keys+['prediction']].rename(columns={'prediction':'baseline'})
        j=f.merge(b,on=keys,validate='one_to_one');j=j.loc[np.isfinite(j.prediction)&np.isfinite(j.baseline)]
        paired.append({'horizon':int(horizon),'model':'sector_specific','baseline':'ridge_paired_supported_industries','n':len(j),'mae_improvement':float(1-abs(j.prediction-j.actual).mean()/abs(j.baseline-j.actual).mean()) if len(j) else None})
    return records,strata,paired


def source_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
