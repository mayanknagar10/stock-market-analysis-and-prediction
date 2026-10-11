import numpy as np
import pandas as pd
import pytest
from core.forecasting.registry import save_bundle,load_bundle
from core.forecasting.heads import DirectRegressor
from core.forecasting.distribution import DistributionHeads
from core.forecasting.trust import OODDetector

def test_native_artifact_roundtrip_and_checksum_rejection(tmp_path):
    rng=np.random.default_rng(21)
    times=pd.date_range('2020-01-01',periods=100,tz='UTC')
    X=pd.DataFrame(rng.normal(size=(100,2)),columns=['a','b'],index=times)
    y=.01*X.a+.001*rng.normal(size=100)
    model=DirectRegressor('india_equity',5,n_estimators=8).fit(X,y,times,times+pd.Timedelta('5D'),pd.Timestamp('2021-01-01',tz='UTC'))
    distribution=DistributionHeads(n_estimators=8).fit(model.scaled(X),y)
    bundle={'regression':model,'distribution':distribution,'calibrator':None,'ood':OODDetector().fit(model.scaled(X)),
            'metadata':{'model_version':'test','point_in_time_verified':False,'final_test_passed':False},
            'conformal_margin':0.,'policy':{},'analogues':None}
    directory=tmp_path/'model'
    save_bundle(bundle,directory)
    restored=load_bundle(directory)
    np.testing.assert_allclose(model.predict(X.iloc[:3])['central_log_return'],restored['regression'].predict(X.iloc[:3])['central_log_return'],rtol=1e-6)
    import json,hashlib
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    original=(directory/'manifest.json').read_bytes()
    manifest['files'].pop('reg_xgb.json')
    modified=json.dumps(manifest).encode('utf-8')
    (directory/'manifest.json').write_bytes(modified)
    (directory/'manifest.sha256').write_text(hashlib.sha256(modified).hexdigest(),encoding='ascii')
    with pytest.raises(ValueError,match='required artifact'): load_bundle(directory)
    (directory/'manifest.json').write_bytes(original)
    (directory/'manifest.sha256').write_text(hashlib.sha256(original).hexdigest(),encoding='ascii')
    file=directory/'reg_xgb.json'
    file.write_bytes(file.read_bytes()+b' ')
    with pytest.raises(ValueError,match='integrity'): load_bundle(directory)

def test_model_version_cannot_overwrite_existing_artifacts(tmp_path):
    directory=tmp_path/'exists'
    directory.mkdir()
    with pytest.raises(FileExistsError): save_bundle({},directory)
