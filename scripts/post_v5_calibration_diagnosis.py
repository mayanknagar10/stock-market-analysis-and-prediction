"""Read-only released calibration diagnostic on exact pre-boundary fit arrays."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import uuid
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from core.forecasting.service import active_research,load_research_model
from research.post_v5.preservation import verify
from research.post_v5.development import BOUNDARY
verify(ROOT,ROOT/'reports/post_v5/V5_PRESERVATION.json')
pointer=active_research();rows=[]
for family in ['india_equity','us_equity']:
    for horizon in [1,5,10,20]:
        bundle=load_research_model(family,horizon,pointer);pool=bundle['analogues']
        assert pd.to_datetime(pool['label_ends'],utc=True).max()<BOUNDARY
        distribution=bundle['distribution'].predict(pool['features']);raw=distribution['probability'];labels=np.asarray(pool['returns'])>0
        calibrated=bundle['calibrator'].predict(raw) if bundle['calibrator'] is not None else raw
        rows.append({'family':family,'horizon':horizon,'n_fit':len(labels),'raw_fit_roc_auc':float(roc_auc_score(labels,raw)),
            'issued_fit_roc_auc':float(roc_auc_score(labels,calibrated)),
            'raw_fit_brier':float(np.mean((raw-labels)**2)),'issued_fit_brier':float(np.mean((calibrated-labels)**2)),
            'released_calibration_accepted':bundle['calibrator'] is not None,
            'platt_coefficient':bundle['calibrator'].coefficient if bundle['calibrator'] is not None else None,
            'ranking_inverted_by_released_calibration':bool(bundle['calibrator'] is not None and bundle['calibrator'].coefficient<0)})
record={'diagnosis_id':'calibration-diagnostic-'+uuid.uuid4().hex,'created_at':datetime.now(timezone.utc).isoformat(),
    'model_version':pointer['model_version'],'rows':rows,'new_fit_or_calibration':False,'final_period_read':False,
    'interpretation':'These are in-sample base-row diagnostics under released later-development weights/calibration. Negative Platt slope reverses ranking; it may reflect unstable calibration-period ranking and is not proof of a causal mechanism. No calibrator, probability or threshold is changed.'}
folder=ROOT/'reports/post_v5'/record['diagnosis_id'];folder.mkdir(parents=True,exist_ok=False)
with (folder/'CALIBRATION_DIAGNOSIS.json').open('x',encoding='utf-8') as stream:json.dump(record,stream,indent=2)
lines=['# Released V5 calibration diagnostic addendum','',record['interpretation'],'',
    '| Family | Sessions | Fit N | Raw fit ROC-AUC | Issued fit ROC-AUC | Platt coefficient | Ranking reversed |',
    '|---|---:|---:|---:|---:|---:|---|']
for row in rows:
    coefficient='Unavailable' if row['platt_coefficient'] is None else f'{row["platt_coefficient"]:.4f}'
    lines.append(f'| {row["family"]} | {row["horizon"]} | {row["n_fit"]} | {row["raw_fit_roc_auc"]:.3f} | {row["issued_fit_roc_auc"]:.3f} | {coefficient} | {row["ranking_inverted_by_released_calibration"]} |')
lines+=['','Raw base-fit ROC-AUC measures fitted discrimination, not generalization. The native train/validation addendum reports issued scores, so accepted negative-slope calibrations can show fit AUC below 0.5 despite strong raw in-sample ranking. The new research protocol requires stable held-out discrimination before calibration. No consumed-period rows or outcomes enter this diagnostic or any model choice.']
text=chr(10).join(lines)+chr(10)
with (folder/'CALIBRATION_DIAGNOSIS.md').open('x',encoding='utf-8') as stream:stream.write(text)
with (ROOT/'reports/post_v5/V5_CALIBRATION_ADDENDUM.md').open('x',encoding='utf-8') as stream:stream.write(text)
print(json.dumps(rows,indent=2))
