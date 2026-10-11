"""Build and verify a private, versioned, read-only StockPro evidence copy.

Only allowlisted research paths are exported. SQLite is copied with its online
backup API, so active WAL pages are included without changing the source.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MODEL = "research-20261008T190859"
SOURCE_DIRS = (
    "data/prospective_automation",
    "data/research/prospective-india-20261008-v1",
    "data/v5",
)
REQUIRED_DATABASES = (
    "data/prospective_automation/collection.sqlite",
    "data/research/prospective-india-20261008-v1/shadow_predictions.sqlite",
    "data/v5/predictions.sqlite",
)
MANIFEST = "data/EVIDENCE_MIRROR_MANIFEST.json"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def backup_sqlite(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(source.as_uri() + "?mode=ro", uri=True) as original:
        with sqlite3.connect(target) as copy:
            original.backup(copy)
            if copy.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("SQLite backup failed integrity check")


def identity_rows(path):
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("SQLite integrity failure")
        rows = connection.execute("SELECT prediction_id,payload,payload_hash FROM forecasts").fetchall()
        outcomes = dict(connection.execute("SELECT prediction_id,payload FROM outcomes"))
    identities = {}
    for identity, payload, digest in rows:
        if hashlib.sha256(payload.encode()).hexdigest() != digest or identity in identities:
            raise ValueError("Forecast payload integrity failure")
        identities[identity] = digest
    statuses = {identity: json.loads(payload).get("status") for identity, payload in outcomes.items()}
    if set(statuses) - set(identities):
        raise ValueError("Outcome without forecast")
    return identities, statuses


def evidence_counts(data, expected_hashes):
    canonical, canonical_outcomes = identity_rows(data / "prospective_automation/collection.sqlite")
    legacy, legacy_outcomes = identity_rows(data / "research/prospective-india-20261008-v1/shadow_predictions.sqlite")
    identity_rows(data / "v5/predictions.sqlite")
    if len(expected_hashes) != 388 or any(legacy.get(key) != value for key, value in expected_hashes.items()):
        raise ValueError("Original 388 forecast identities changed")
    if set(canonical) & set(legacy):
        raise ValueError("Duplicate cross-ledger forecast identity")
    outcomes = {**canonical_outcomes, **legacy_outcomes}
    total = len(canonical) + len(legacy)
    matured = sum(status == "matured" for status in outcomes.values())
    pending = total - len(outcomes)
    other = len(outcomes) - matured
    if matured + pending + other != total:
        raise ValueError("Forecast count inconsistency")
    return {"canonical_forecasts": len(canonical), "legacy_forecasts": len(legacy),
            "total_forecasts": total, "matured": matured, "pending": pending,
            "other_outcomes": other, "original_forecasts_verified": len(expected_hashes)}


def source_files(root):
    files = []
    for relative in SOURCE_DIRS:
        directory = root / relative
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError("Required source directory missing or linked: " + relative)
        for source in sorted(directory.rglob("*")):
            if source.is_symlink():
                raise ValueError("Linked evidence entry")
            if not source.is_file():
                continue
            if source.suffix not in {".json", ".sqlite"}:
                continue  # Local locks, logs, WAL and caches are deliberately excluded.
            if any(part.lower().split(".")[0] in {"users", "auth", "sessions", "secrets"}
                   for part in source.relative_to(root).parts):
                raise ValueError("Sensitive evidence path")
            files.append(source)
    expected = {root / path for path in REQUIRED_DATABASES}
    if not expected.issubset(set(files)):
        raise ValueError("Required SQLite evidence missing")
    return files


def allowed_mirror_name(name):
    path = Path(name)
    return (not path.is_absolute() and ".." not in path.parts
            and any(name.startswith(prefix + "/") for prefix in SOURCE_DIRS)
            and path.suffix in {".json", ".sqlite"}
            and not any(part.lower().split(".")[0] in {"users", "auth", "sessions", "secrets"}
                        for part in path.parts))


def build(output_parent, root=ROOT, expected_hashes=None):
    root = Path(root).resolve()
    output_parent = Path(output_parent).resolve()
    if any(output_parent == root / item or root / item in output_parent.parents for item in SOURCE_DIRS):
        raise ValueError("Mirror output must not be inside an evidence source directory")
    seal = json.loads((root / "reports/automation/RESEARCH_PRESERVATION.json").read_text(encoding="utf-8"))
    expected_hashes = expected_hashes if expected_hashes is not None else seal["legacy_forecast_hashes"]
    if json.loads((root / "models/v5/ACTIVE_RESEARCH.json").read_text(encoding="utf-8"))["model_version"] != MODEL:
        raise ValueError("Unexpected model version")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    package = output_parent / ("stockpro-evidence-mirror-" + stamp)
    if package.exists():
        raise FileExistsError(package)
    files = source_files(root)
    package.mkdir(parents=True)
    try:
        records = {}
        for source in files:
            relative = source.relative_to(root)
            target = package / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.suffix == ".sqlite":
                backup_sqlite(source, target)
            else:
                before = sha256(source)
                shutil.copyfile(source, target)
                if sha256(source) != before or sha256(target) != before:
                    raise ValueError("Source changed during mirror creation")
            records[relative.as_posix()] = {"bytes": target.stat().st_size, "sha256": sha256(target)}
        counts = evidence_counts(package / "data", expected_hashes)
        if not any((package / "data/prospective_automation/archive").rglob("*.json")):
            raise ValueError("Canonical source snapshots missing")
        if not any((package / "data/research/prospective-india-20261008-v1").rglob("*.json")):
            raise ValueError("Legacy provenance missing")
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        manifest = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
                    "source_commit": commit, "model_version": MODEL, "production_promotion": "BLOCKED",
                    "counts": counts, "files": records}
        destination = package / MANIFEST
        destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        verify(package, expected_hashes)
        return package, manifest
    except Exception:
        shutil.rmtree(package)
        raise


def verify(package, expected_hashes=None):
    package = Path(package).resolve()
    manifest = json.loads((package / MANIFEST).read_text(encoding="utf-8"))
    if expected_hashes is None:
        seal = json.loads((ROOT / "reports/automation/RESEARCH_PRESERVATION.json").read_text(encoding="utf-8"))
        expected_hashes = seal["legacy_forecast_hashes"]
    if manifest["schema_version"] != 1 or manifest["model_version"] != MODEL or manifest["production_promotion"] != "BLOCKED":
        raise ValueError("Mirror identity invalid")
    allowed = set(manifest["files"])
    if not all(allowed_mirror_name(name) for name in allowed):
        raise ValueError("Mirror contains a forbidden path")
    if not set(REQUIRED_DATABASES).issubset(allowed):
        raise ValueError("Required mirror databases missing")
    if ("data/v5/REFERENCE_CAPTURE.json" not in allowed
            or not any(name.startswith("data/v5/snapshots/") for name in allowed)
            or not any(name.startswith("data/prospective_automation/archive/") for name in allowed)
            or not any(name.startswith("data/research/prospective-india-20261008-v1/") and name.endswith(".json") for name in allowed)):
        raise ValueError("Required snapshots or provenance missing")
    actual = {path.relative_to(package).as_posix() for path in (package / "data").rglob("*") if path.is_file()}
    if actual != allowed | {MANIFEST}:
        raise ValueError("Unexpected or missing mirror file")
    for name, entry in manifest["files"].items():
        original = package / name
        path = original.resolve()
        if package not in path.parents or original.is_symlink() or path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError("Mirror file hash mismatch: " + name)
    counts = evidence_counts(package / "data", expected_hashes)
    if counts != manifest["counts"]:
        raise ValueError("Mirror count mismatch")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify"))
    parser.add_argument("path", type=Path, help="Output parent for build, package directory for verify")
    args = parser.parse_args()
    result = build(args.path)[1] if args.command == "build" else verify(args.path)
    print(json.dumps({"model_version": result["model_version"], "counts": result["counts"],
                      "files_verified": len(result["files"]), "created_at": result["created_at"]}, indent=2))
