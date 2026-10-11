import hashlib
from pathlib import Path
import numpy as np
import pytest
from core.validation.v4_adapter import load_frozen_v4
from core.models import _latest_feature_row

def test_v4_loader_leaves_frozen_artifacts_unchanged(candles):
    path = Path(__file__).resolve().parents[1]/'models'/'universal_lgb.txt'
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    predictor = load_frozen_v4(normalize_text=True)
    value = predictor.predict_next_return(_latest_feature_row(candles,predictor.feature_names))
    assert np.isfinite(value)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before

def test_unacknowledged_crlf_is_rejected_before_native_loading():
    path = Path(__file__).resolve().parents[1]/'models'/'universal_lgb.txt'
    if b'\r\n' not in path.read_bytes(): pytest.skip('Native LF checkout needs no CRLF guard')
    with pytest.raises(ValueError,match='CRLF'):
        load_frozen_v4()
