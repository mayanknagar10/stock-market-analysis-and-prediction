"""Environment-owned runtime locations; immutable model/universe paths stay pinned."""
import hashlib
import json
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PROTOCOL="india-delayed-eod-0630-v1"
MODEL="research-20261008T190859"
UNIVERSE=ROOT/"config/research/prospective-india-20261008-v1/UNIVERSE.json"
UNIVERSE_HASH="a6fed59fe4f94cbae67c996e8c3ae93596c217cfe310ec573ee8fd2648aa42f0"
def universe():
    payload=UNIVERSE.read_bytes()
    document=json.loads(payload)
    if hashlib.sha256(payload).hexdigest()!=UNIVERSE_HASH:raise ValueError("Frozen universe bytes changed")
    if document["raw_membership_sha256"]!="0a53ad14c15e9ec258dc72117a8c0699546ec25f2d283c47d53412e2de8a06fc" or len(document["members"])!=100:
        raise ValueError("Frozen universe mismatch")
    return document,hashlib.sha256(payload).hexdigest()
def data_root():
    return Path(os.environ.get("STOCKPRO_DATA_ROOT",str(ROOT/"data/prospective_automation"))).resolve()
