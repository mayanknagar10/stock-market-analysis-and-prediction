"""Repeatable frozen-checkpoint V4 diagnostic; never mislabels retrospective inference OOS."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from core.validation.metrics import forecast_metrics, grouped_metrics

def read_snapshot(manifest_path):
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding='utf-8'))
    frames = {}
    for ticker,meta in manifest['assets'].items():
        file = path.parent/meta['file']
        payload = file.read_bytes()
        if hashlib.sha256(payload).hexdigest()!=meta['sha256']: raise ValueError('Corrupted snapshot: '+ticker)
        bars = pd.read_csv(file,index_col=0)
        # Preserve exchange local session date for legacy V4 features.
        bars.index = pd.DatetimeIndex(pd.to_datetime(bars.index,utc=True)).tz_convert(meta['exchange_timezone']).tz_localize(None)
        factor = bars['Adj Close']/bars['Close']
        frame = bars[['Open','High','Low','Close','Volume']].copy()
        frame[['Open','High','Low','Close']] = frame[['Open','High','Low','Close']].mul(factor,axis=0)
        frames[ticker] = frame
    return manifest,frames

def evaluate(manifest_path, origins=12, paths=30, normalize_text=False):
    from core.models import _simulate_paths
    from core.validation.v4_adapter import load_frozen_v4
    protocol = json.loads((ROOT/'config/v5_validation.json').read_text(encoding='utf-8'))
    boundary = pd.Timestamp(protocol['final_test_start'])
    manifest,frames = read_snapshot(manifest_path)
    model = load_frozen_v4(normalize_text=normalize_text)
    if not model.loaded: raise RuntimeError('Frozen V4 checkpoint could not load; do not train/overwrite it')
    rows = []
    for ticker,full in frames.items():
        df = full.loc[full.index<boundary]
        if len(df)<300: raise ValueError('Insufficient pre-final history: '+ticker)
        choices = np.unique(np.linspace(max(252,len(df)-252),len(df)-21,origins,dtype=int))
        for i in choices:
            prefix = df.iloc[:i+1]
            seed = int(hashlib.sha256((manifest['assets'][ticker]['sha256']+str(df.index[i])).encode()).hexdigest()[:8],16)
            sigma = float(np.log(prefix['Close']).diff().dropna().std())
            simulations = _simulate_paths(prefix,model,sigma,20,n_paths=paths,seed=seed)
            central = np.median(simulations,axis=0)
            lo = np.percentile(simulations,10,axis=0)
            hi = np.percentile(simulations,90,axis=0)
            close = float(df['Close'].iloc[i])
            one_day = model.predict_next_return(__import__('core.models',fromlist=['_latest_feature_row'])._latest_feature_row(prefix,model.feature_names))
            ret = np.log(prefix['Close']).diff()
            # Definitions frozen before evaluating models. Regimes are descriptive only.
            drift = float(ret.tail(252).mean())
            volatility = 'high' if float(ret.tail(20).std())>float(ret.dropna().median()+ret.dropna().std()) else 'low'
            trend = float(prefix['Close'].iloc[-1]/prefix['Close'].tail(50).mean()-1)
            regime = ('bullish' if trend>.02 else 'bearish' if trend<-.02 else 'sideways')+'_'+volatility
            meta = manifest['assets'][ticker]
            for horizon in protocol['horizons']:
                actual = float(np.log(df['Close'].iloc[i+horizon]/close))
                base = {'ticker':ticker,'sector':meta.get('sector','unknown'),'family':meta['family'],
                        'year':int(df.index[i].year),'date':df.index[i].isoformat(),'horizon':horizon,
                        'volatility_regime':volatility,'market_regime':regime,'actual':actual,
                        'actual_price':float(df['Close'].iloc[i+horizon]),'origin_price':close,
                        'probability':None,'oos_verified':False,'seed':seed}
                for name,prediction in [('v4_recursive',float(np.log(central[horizon-1]/close))),
                                        ('v4_readme_compounding',one_day*horizon),
                                        ('zero_return',0.),('historical_drift',drift*horizon)]:
                    item = {**base,'method':name,'predicted':float(prediction),'predicted_price':close*np.exp(prediction)}
                    if name=='v4_recursive':
                        item.update(lower_80=float(np.log(lo[horizon-1]/close)),upper_80=float(np.log(hi[horizon-1]/close)))
                    rows.append(item)
        print(ticker, len(choices),'origins measured',flush=True)
    frame = pd.DataFrame(rows)
    report = {'kind':'retrospective_checkpoint_diagnostic', 'oos_verified':False,
              'snapshot_manifest':Path(manifest_path).name,'snapshot_sha256':hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
              'final_test_start':protocol['final_test_start'],'final_test_evaluated':False,
              'v4_metadata_unchanged':model.meta,'paths':paths,'origins_per_ticker':origins,'v4_text_normalized':normalize_text,
              'limitations':['Shipped checkpoint was trained only on synthetic tickers',
                  'Training start/end and original training data unavailable',
                  'Inference from loaded checkpoint is not walk-forward refitting',
                  'Retrospective Yahoo adjusted-price vintage is not verified point-in-time',
                  'Sparse sample origins are diagnostics, not full-universe performance',
                  'V4 has no classification probabilities or model-based quantiles',
                  'Market/sector beta baseline unavailable in this six-stock snapshot',
                  'No forecast strategy portfolio exists; Sharpe/Sortino/turnover are unavailable, not manufactured'],
              'results':{}}
    for (method,horizon),group in frame.groupby(['method','horizon']):
        metrics = forecast_metrics(group['actual'],group['predicted'])
        metrics['price_mae'] = float(np.abs(group['predicted_price']-group['actual_price']).mean())
        metrics['price_rmse'] = float(np.sqrt(np.mean((group['predicted_price']-group['actual_price'])**2)))
        if method=='v4_recursive':
            metrics['simulation_coverage_80'] = float(((group.actual>=group.lower_80)&(group.actual<=group.upper_80)).mean())
            metrics['simulation_width_80'] = float((group.upper_80-group.lower_80).mean())
        report['results'][method+'_'+str(horizon)+'d'] = {'aggregate':metrics,'breakdowns':grouped_metrics(group,
            ['ticker','sector','family','year','volatility_regime','market_regime'])}
    return report,frame

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True,type=Path)
    parser.add_argument('--origins',type=int,default=12)
    parser.add_argument('--paths',type=int,default=30)
    parser.add_argument('--normalize-v4-text',action='store_true',help='Read-only LF normalization for Windows CRLF checkpoint offsets; numeric trees unchanged')
    parser.add_argument('--output',type=Path,default=ROOT/'reports'/'baseline'/'V4_DIAGNOSTIC.json')
    args = parser.parse_args()
    if args.origins<2 or args.paths<30: parser.error('At least 2 origins and 30 paths are required')
    start=time.perf_counter()
    report,rows = evaluate(args.manifest,args.origins,args.paths,args.normalize_v4_text)
    report['elapsed_seconds']=time.perf_counter()-start
    payload=json.dumps(report,indent=2,allow_nan=False).encode()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as stream: stream.write(payload)
    with args.output.with_suffix('.csv').open('x',encoding='utf-8',newline='') as stream: rows.to_csv(stream,index=False)
    print('Persisted',args.output,flush=True)

if __name__=='__main__': main()
