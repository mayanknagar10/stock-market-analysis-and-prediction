import pandas as pd
import pytest
from scripts.evaluate_final import final_rows,check_cutoffs

def test_final_rows_exclude_development_and_unmatured_labels():
    meta=pd.DataFrame({'feature_time':pd.to_datetime(['2025-09-30T10:00Z','2025-10-01T10:00Z','2025-10-02T10:00Z'],utc=True),
        'label_end':pd.to_datetime(['2025-10-01T10:00Z','2025-10-02T10:00Z','2026-10-09T10:00Z'],utc=True)})
    assert final_rows(meta,'2025-10-01T00:00Z','2026-10-07T00:00Z').tolist()==[1]

def test_candidate_decisions_cannot_reach_final_boundary():
    metadata={name:'2025-09-30T00:00Z' for name in ['training_cutoff','blending_cutoff','calibration_cutoff','validation_cutoff']}
    metadata.update(scope='research_only',point_in_time_verified=False)
    check_cutoffs(metadata,'2025-10-01T00:00Z')
    metadata['validation_cutoff']='2025-10-01T00:00Z'
    with pytest.raises(ValueError): check_cutoffs(metadata,'2025-10-01T00:00Z')
