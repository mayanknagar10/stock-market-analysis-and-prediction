"""Read-only diagnostic adapter for the frozen V4 artifact; never production fallback."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

def load_frozen_v4(normalize_text=False):
    from core.models import UniversalPredictor
    import xgboost as xgb
    import lightgbm as lgb
    freeze = json.loads((ROOT/'reports/baseline/V4_FREEZE.json').read_text(encoding='utf-8'))
    for name in ['models/universal_meta.json','models/universal_scaler.json',
                 'models/universal_xgb.json','models/universal_lgb.txt','core/models.py','core/indicators.py']:
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=freeze['files'][name]['sha256']:
            raise ValueError('Frozen V4 source/artifact changed: '+name)
    model = UniversalPredictor()
    model.meta = json.loads((ROOT/'models/universal_meta.json').read_text(encoding='utf-8'))
    model.feature_names = model.meta['feature_names']
    scaler = json.loads((ROOT/'models/universal_scaler.json').read_text(encoding='utf-8'))
    model.scaler_center = np.asarray(scaler['center'],dtype=np.float32)
    model.scaler_scale = np.asarray(scaler['scale'],dtype=np.float32)
    payload = (ROOT/'models/universal_lgb.txt').read_bytes()
    if b'\r\n' in payload and not normalize_text:
        raise ValueError('V4 LightGBM text contains CRLF; byte offsets require LF. Use explicit --normalize-v4-text for read-only diagnostics.')
    # Universal newline normalization leaves every numeric tree value unchanged.
    model.lgb_model = lgb.Booster(model_str=payload.decode('utf-8').replace('\r\n','\n'))
    model.xgb_model = xgb.Booster()
    model.xgb_model.load_model(ROOT/'models/universal_xgb.json')
    model.loaded = True
    return model
