"""Verify prospective input/ledger contracts and original V5 preservation without tuning."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import uuid
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.post_v5.archive import RunArchive
from research.post_v5.preservation import verify
from research.post_v5.shadow import DIRECTORY,STUDY,FROZEN
from core.forecasting.ledger import PredictionLedger

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-id',required=True);args=parser.parse_args()
    preservation=verify(ROOT,ROOT/'reports/post_v5/V5_PRESERVATION.json')
    archive=RunArchive(DIRECTORY);run=archive.load_run(args.run_id)
    cutoff=pd.Timestamp(run['cutoff']);assert run['study_id']==STUDY
    assert pd.Timestamp(run['archived_at'])>=cutoff
    universe_file=ROOT/'config/research'/STUDY/'UNIVERSE.json'
    universe=json.loads(universe_file.read_text(encoding='utf-8'))
    raw=ROOT/'config/research'/STUDY/universe['raw_file']
    assert hashlib.sha256(raw.read_bytes()).hexdigest()==universe['raw_membership_sha256']
    assert run['universe_hash']==hashlib.sha256(universe_file.read_bytes()).hexdigest()
    assert len({m['isin'] for m in universe['members']})==len(universe['members'])
    quality=[]
    for key,identity in run['snapshots'].items():
        snapshot=archive.load_snapshot(identity);frame=snapshot.frame
        assert pd.Timestamp(snapshot.metadata.fetched_at)<=cutoff
        assert snapshot.metadata.provider_id and snapshot.metadata.provider_version
        assert frame.index.tz is not None
        if snapshot.metadata.family in {'india_equity','us_equity'}:
            assert {'Dividends','Stock Splits','Close','Adj Close'}.issubset(frame)
        required=[c for c in ['Open','High','Low','Close','Adj Close','Volume'] if c in frame]
        latest_missing=frame[required].iloc[-1].isna().any()
        quality.append({'source':key,'rows':len(frame),'retrieved_at':snapshot.metadata.fetched_at,
            'observation_timestamp':snapshot.metadata.source_timestamp,'latest_missing_critical_values':bool(latest_missing),
            'any_missing_rows':int(frame[required].isna().any(axis=1).sum()),'snapshot_id':identity,
            'historical_availability_verified':False})
    ledger_file=DIRECTORY/'shadow_predictions.sqlite'
    records=PredictionLedger(ledger_file).rows() if ledger_file.exists() else []
    for record in records:
        assert record['model_version']==FROZEN and record['confidence']=='LOW' and record['abstain'] is True
        assert record['point_in_time_verified'] is False and record['input_archive_observed_at_prediction'] is True
        source_run=archive.load_run(record['input_run_id'])
        assert pd.Timestamp(source_run['archived_at'])<=pd.Timestamp(record['forecast_as_of'])
        assert record['feature_values'] and record['feature_schema_hash'] and record['training_cutoff']
        for identity in record['snapshot_ids']:archive.load_snapshot(identity)
    evidence={'verification_id':'verification-'+uuid.uuid4().hex,'verified_at':datetime.now(timezone.utc).isoformat(),
        'study_id':STUDY,'run_id':args.run_id,'preservation':preservation,'n_fixed_members':len(universe['members']),
        'n_captured_sources':len(quality),'n_captured_equities':sum(row['source'].endswith('.NS') for row in quality),
        'latest_missing_equity_sources':sum(row['source'].endswith('.NS') and row['latest_missing_critical_values'] for row in quality),
        'prospective_shadow_predictions':len(records),'prospective_matured_outcomes':sum(r['outcome'] is not None and r['outcome']['status']=='matured' for r in records),
        'input_archive_hashes_and_retrieval_bounds':'PASS','universe_source_hash':'PASS','ledger_lineage':'PASS',
        'historical_training_point_in_time_certified':False,'production_promotion':'BLOCKED','source_quality':quality,
        'new_final_candidate_trained':False}
    directory=ROOT/'reports/post_v5/verification';directory.mkdir(parents=True,exist_ok=True)
    with (directory/(evidence['verification_id']+'.json')).open('x',encoding='utf-8') as stream:json.dump(evidence,stream,indent=2)
    print({k:v for k,v in evidence.items() if k!='source_quality'})
if __name__=='__main__':main()
