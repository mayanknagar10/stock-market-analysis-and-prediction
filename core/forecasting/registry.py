"""Immutable native checkpoints; checked schema and hashes, no arbitrary pickle loading."""
from pathlib import Path
import hashlib
import json
import numpy as np
from sklearn.preprocessing import RobustScaler
from core.forecasting.heads import DirectRegressor
from core.forecasting.distribution import DistributionHeads,LEVELS
from core.forecasting.trust import PlattCalibrator,OODDetector
from core.validation.contracts import schema_hash

class NativeModel:
    def __init__(self,kind,booster): self.kind=kind; self.booster=booster
    def predict(self,X):
        if self.kind=='xgb':
            import xgboost as xgb
            return self.booster.predict(xgb.DMatrix(np.asarray(X)))
        return self.booster.predict(np.asarray(X))

def _save_model(model,kind,path):
    if hasattr(model,'get_booster'): model.get_booster().save_model(path)
    elif hasattr(model,'booster_'): model.booster_.save_model(str(path))
    elif isinstance(model,NativeModel): model.booster.save_model(str(path))
    else: raise ValueError('Unsupported checkpoint model')

def save_bundle(bundle,directory):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=False)
    regression=bundle['regression']
    distribution=bundle['distribution']
    document={'format_version':1,'family':regression.family,'horizon':regression.horizon,
        'features':list(regression.feature_names),'schema_hash':regression.schema_hash,
        'scaler_center':regression.scaler.center_.tolist(),'scaler_scale':regression.scaler.scale_.tolist(),
        'regression_weights':regression.weights.tolist(),'distribution_weights':distribution.weights.tolist(),
        'distribution_class_weights':distribution.class_weights.tolist(),
        'metadata':bundle['metadata'],'conformal_margin':float(bundle['conformal_margin']),
        'policy':bundle['policy'],'ood_threshold':bundle['ood'].threshold,'files':{},'calibrator':None,'auxiliary':{}}
    if bundle['calibrator'] is not None:
        calibrator=bundle['calibrator']
        document['calibrator']={name:getattr(calibrator,name) for name in ['coefficient','intercept','n','start','end']}
    families={'reg':regression.models,'class':distribution.classifiers}
    for level,models in distribution.quantile_models.items(): families['quantile_'+level.replace('.','_')]=models
    for task,heads in bundle.get('auxiliary',{}).items():
        families['alpha_'+task+'_reg']=heads['regression']
        families['alpha_'+task+'_class']=heads['classifiers']
        document['auxiliary'][task]={k:v for k,v in heads.items() if k not in {'regression','classifiers','calibrator'}}
        document['auxiliary'][task]['calibrator']=None
        if heads['calibrator'] is not None:
            document['auxiliary'][task]['calibrator']={name:getattr(heads['calibrator'],name) for name in ['coefficient','intercept','n','start','end']}
    for group,models in families.items():
        for kind,model in models.items():
            name=group+'_'+kind+('.json' if kind=='xgb' else '.txt')
            _save_model(model,kind,directory/name)
            document['files'][name]=hashlib.sha256((directory/name).read_bytes()).hexdigest()
    if bundle.get('analogues') is not None:
        name='analogues.npz'
        np.savez_compressed(directory/name,**bundle['analogues'])
        document['files'][name]=hashlib.sha256((directory/name).read_bytes()).hexdigest()
    payload=json.dumps(document,indent=2,sort_keys=True,allow_nan=False).encode('utf-8')
    (directory/'manifest.json').write_bytes(payload)
    (directory/'manifest.sha256').write_text(hashlib.sha256(payload).hexdigest(),encoding='ascii')
    return document

def load_bundle(directory):
    import xgboost as xgb
    import lightgbm as lgb
    directory=Path(directory)
    payload=(directory/'manifest.json').read_bytes()
    if hashlib.sha256(payload).hexdigest()!=(directory/'manifest.sha256').read_text(encoding='ascii').strip():
        raise ValueError('Manifest integrity failure')
    document=json.loads(payload.decode('utf-8'))
    if document['format_version']!=1 or schema_hash(document['features'])!=document['schema_hash']:
        raise ValueError('Model schema/version mismatch')
    for flag in ['point_in_time_verified','final_test_passed']:
        if type(document['metadata'].get(flag)) is not bool: raise ValueError('Invalid model evidence boolean')
    expected={group+'_'+kind+('.json' if kind=='xgb' else '.txt')
        for group in ['reg','class']+['quantile_'+str(q).replace('.','_') for q in LEVELS]
        for kind in ['xgb','lgb']}
    for task in document.get('auxiliary',{}):
        if task not in {'market','sector'}: raise ValueError('Unknown auxiliary task')
        expected.update('alpha_'+task+'_'+group+'_'+kind+('.json' if kind=='xgb' else '.txt')
            for group in ['reg','class'] for kind in ['xgb','lgb'])
    if not expected.issubset(document['files']): raise ValueError('Missing required artifact integrity entries')
    for name,digest in document['files'].items():
        if Path(name).name!=name or name in {'.','..'}: raise ValueError('Invalid artifact path')
        if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=digest: raise ValueError('Artifact integrity failure: '+name)
    def models(group):
        output={}
        for kind in ['xgb','lgb']:
            file=directory/(group+'_'+kind+('.json' if kind=='xgb' else '.txt'))
            if kind=='xgb':
                booster=xgb.Booster(); booster.load_model(file)
            else:
                booster=lgb.Booster(model_str=file.read_text(encoding='utf-8'))
            output[kind]=NativeModel(kind,booster)
        return output
    regression=DirectRegressor(document['family'],document['horizon'])
    regression.feature_names=tuple(document['features'])
    regression.schema_hash=document['schema_hash']
    regression.models=models('reg')
    regression.weights=np.asarray(document['regression_weights'])
    scaler=RobustScaler()
    scaler.center_=np.asarray(document['scaler_center'])
    scaler.scale_=np.asarray(document['scaler_scale'])
    scaler.n_features_in_=len(regression.feature_names)
    scaler.feature_names_in_=np.asarray(regression.feature_names,dtype=object)
    regression.scaler=scaler
    regression.training_cutoff=document['metadata'].get('training_cutoff')
    distribution=DistributionHeads()
    distribution.classifiers=models('class')
    distribution.quantile_models={str(level):models('quantile_'+str(level).replace('.','_')) for level in LEVELS}
    distribution.weights=np.asarray(document['distribution_weights'])
    distribution.class_weights=np.asarray(document.get('distribution_class_weights',document['distribution_weights']))
    calibrator=None
    if document['calibrator'] is not None:
        calibrator=PlattCalibrator()
        for name,value in document['calibrator'].items(): setattr(calibrator,name,value)
    detector=OODDetector()
    detector.threshold=document['ood_threshold']
    detector.n_features=len(regression.feature_names)
    analogues=None
    if 'analogues.npz' in document['files']:
        with np.load(directory/'analogues.npz',allow_pickle=False) as archive:
            analogues={name:archive[name] for name in archive.files}
    auxiliary={}
    for task,evidence in document.get('auxiliary',{}).items():
        head_calibrator=None
        if evidence['calibrator'] is not None:
            head_calibrator=PlattCalibrator()
            for key,value in evidence['calibrator'].items(): setattr(head_calibrator,key,value)
        auxiliary[task]={**evidence,'regression':models('alpha_'+task+'_reg'),
            'classifiers':models('alpha_'+task+'_class'),'calibrator':head_calibrator}
    return {'auxiliary':auxiliary,'regression':regression,'distribution':distribution,'calibrator':calibrator,
        'ood':detector,'metadata':document['metadata'],'conformal_margin':document['conformal_margin'],
        'policy':document['policy'],'analogues':analogues,'manifest':document}
