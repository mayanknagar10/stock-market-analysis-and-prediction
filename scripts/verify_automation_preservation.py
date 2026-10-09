"""Seal and verify existing research before additive automation (never rewrite a seal)."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.post_v5.preservation import verify as verify_v5
SEAL = ROOT / "reports/automation/RESEARCH_PRESERVATION.json"
LEGACY = ROOT / "data/research/prospective-india-20261008-v1/shadow_predictions.sqlite"
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def forecasts():
    with sqlite3.connect(LEGACY.as_uri() + "?mode=ro", uri=True) as con:
        rows = con.execute("SELECT prediction_id,payload,payload_hash FROM forecasts").fetchall()
    result = {}
    for identity, payload, stored in rows:
        if hashlib.sha256(payload.encode()).hexdigest() != stored:
            raise ValueError("Legacy prediction integrity failure")
        result[identity] = stored
    return result
def create_seal():
    verify_v5(ROOT, ROOT / "reports/post_v5/V5_PRESERVATION.json")
    baseline = forecasts()
    if len(baseline) != 388:
        raise ValueError("Expected exactly 388 original prospective forecasts")
    files = {}
    for relative in ["reports/post_v5", "config/research", "data/research"]:
        for file in (ROOT / relative).rglob("*"):
            if not file.is_file() or "__pycache__" in file.parts or file.suffix in {".sqlite",".log"} or file.name.endswith(("-wal","-shm")):
                continue
            # Existing regression tests write new attempt reports; these are retained
            # but not represented as completed research from commit b3a9edc.
            if relative == "reports/post_v5" and file.parent.name == "outcomes":
                continue
            files[file.relative_to(ROOT).as_posix()] = digest(file)
    record = {"sealed_at":datetime.now(timezone.utc).isoformat(),
              "base_commit":"b3a9edc","branch":"research/prospective-automation",
              "model_version":"research-20261008T190859","files":files,
              "legacy_forecast_hashes":baseline,
              "ledger_rule":"Original forecast payloads immutable; separate outcomes may append"}
    SEAL.parent.mkdir(parents=True, exist_ok=True)
    with SEAL.open("x", encoding="utf-8") as out:
        json.dump(record, out, indent=2, sort_keys=True)
    return check_seal()
def check_seal():
    original = verify_v5(ROOT, ROOT / "reports/post_v5/V5_PRESERVATION.json")
    record = json.loads(SEAL.read_text(encoding="utf-8"))
    for relative, expected in record["files"].items():
        file = ROOT / relative
        if not file.is_file() or digest(file) != expected:
            raise ValueError("Preserved research file changed: " + relative)
    current = forecasts()
    for identity, expected in record["legacy_forecast_hashes"].items():
        if current.get(identity) != expected:
            raise ValueError("Original prospective prediction changed")
    return {"v5_files_verified":original["verified_files"],
            "additional_files_verified":len(record["files"]),
            "original_forecasts_verified":len(record["legacy_forecast_hashes"]),
            "production_promotion":"BLOCKED"}
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    print(json.dumps(create_seal() if args.seal else check_seal(), indent=2))
