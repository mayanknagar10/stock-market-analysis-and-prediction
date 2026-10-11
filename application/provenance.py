"""Issuance code and native parameter/artifact fingerprints; no model mutation."""
import hashlib
import json
from application.config import ROOT,MODEL
def digest_document(document):
    return hashlib.sha256(json.dumps(document,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def runtime_identity():
    issuing_files=["application/config.py","application/provenance.py","application/collection/calendar.py",
        "application/collection/store.py","application/collection/collector.py","application/collection/engine.py","config/context_sources.json","research/post_v5/shadow.py","research/post_v5/archive.py"]
    training_seal=json.loads((ROOT/"reports/post_v5/V5_PRESERVATION.json").read_text(encoding="utf-8"))
    # Model inference/feature/data logic was preserved; include its actual issuing bytes.
    issuing_files.extend(k for k in training_seal["files"] if k.startswith("core/") and k.endswith(".py"))
    code_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in issuing_files}
    for name,actual in code_hashes.items():
        if name in training_seal["files"] and actual!=training_seal["files"][name]:raise ValueError("FROZEN_CORE_CHANGED")
    artifacts={}
    for name,expected in training_seal["files"].items():
        if name.startswith("models/v5/"+MODEL+"/india_equity/") or name=="models/v5/ACTIVE_RESEARCH.json":
            actual=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            if actual!=expected:raise ValueError("MODEL_ARTIFACT_CHANGED")
            artifacts[name]=actual
    manifest_hashes={str(h):artifacts[f"models/v5/{MODEL}/india_equity/{h}/manifest.json"] for h in (1,5,10,20)}
    parameters={}
    for h in (1,5,10,20):
        document=json.loads((ROOT/f"models/v5/{MODEL}/india_equity/{h}/manifest.json").read_text(encoding="utf-8"))
        # The entire serialized manifest captures feature/scaler/blend/conformal/
        # policy/calibrator/OOD state and links every native head parameter file.
        parameters[str(h)]=digest_document(document)
    return {"issuing_code_identity":digest_document(code_hashes),"issuing_code_hashes":code_hashes,
        "model_artifact_identity":digest_document(artifacts),"model_artifact_hashes":artifacts,
        "model_parameters_identity":digest_document(parameters),"model_parameter_hashes":parameters,"model_manifest_hashes":manifest_hashes}
def verify_runtime(batch):
    current=runtime_identity()
    for key in ("issuing_code_identity","model_artifact_identity","model_parameters_identity"):
        if current[key]!=batch[key]:raise ValueError("RUNTIME_IDENTITY_CHANGED")
