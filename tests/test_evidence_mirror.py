import hashlib
import json
import shutil
import sqlite3

import pytest

from scripts.evidence_mirror import build, sha256, verify


def _database(path, forecasts=0, matured=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    identities = {}
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE forecasts(prediction_id TEXT PRIMARY KEY,payload TEXT,payload_hash TEXT)")
        connection.execute("CREATE TABLE outcomes(prediction_id TEXT PRIMARY KEY,payload TEXT)")
        for index in range(forecasts):
            identity = f"original-{index:03d}"
            payload = json.dumps({"prediction_id": identity, "abstain": True}, sort_keys=True)
            digest = hashlib.sha256(payload.encode()).hexdigest()
            connection.execute("INSERT INTO forecasts VALUES(?,?,?)", (identity, payload, digest))
            identities[identity] = digest
            if index < matured:
                connection.execute("INSERT INTO outcomes VALUES(?,?)", (identity, json.dumps({"status": "matured"})))
    return identities


def test_mirror_online_backup_restore_and_tamper_detection(tmp_path):
    root = tmp_path / "source"
    canonical = root / "data/prospective_automation/collection.sqlite"
    legacy = root / "data/research/prospective-india-20261008-v1/shadow_predictions.sqlite"
    v5 = root / "data/v5/predictions.sqlite"
    _database(canonical)
    expected = _database(legacy, 388, 97)
    _database(v5)
    for name in ("data/prospective_automation/archive/snapshot.json",
                 "data/research/prospective-india-20261008-v1/provenance.json",
                 "data/v5/REFERENCE_CAPTURE.json", "data/v5/snapshots/reference.json"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}', encoding='utf-8')
    seal = root / "reports/automation/RESEARCH_PRESERVATION.json"
    seal.parent.mkdir(parents=True)
    seal.write_text(json.dumps({"legacy_forecast_hashes": expected}), encoding='utf-8')
    pointer = root / "models/v5/ACTIVE_RESEARCH.json"
    pointer.parent.mkdir(parents=True)
    pointer.write_text(json.dumps({"model_version": "research-20261008T190859"}), encoding='utf-8')
    before = {name: sha256(path) for name, path in (("canonical", canonical), ("legacy", legacy), ("v5", v5))}
    package, manifest = build(tmp_path / "packages", root=root)
    assert manifest["counts"]["total_forecasts"] == 388
    assert manifest["counts"]["matured"] == 97
    assert {name: sha256(path) for name, path in (("canonical", canonical), ("legacy", legacy), ("v5", v5))} == before
    restored = tmp_path / "restored"
    shutil.copytree(package, restored)
    assert verify(restored, expected)["counts"]["pending"] == 291
    (restored / "data/prospective_automation/archive/snapshot.json").write_text('tampered', encoding='utf-8')
    with pytest.raises(ValueError, match="hash mismatch"):
        verify(restored, expected)


def test_forged_secret_file_is_rejected_even_with_matching_manifest(tmp_path):
    from scripts.evidence_mirror import allowed_mirror_name
    assert not allowed_mirror_name("data/research/prospective-india-20261008-v1/users.json")
    assert not allowed_mirror_name("data/v5/../../users.json")
