"""Opt-in public read-only boundary; local research behavior is unchanged."""
from collections import deque
from functools import lru_cache
from pathlib import Path
from threading import Lock
from time import monotonic
import json
import logging
import os
import sqlite3

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel
from application.config import MODEL, ROOT
from api.operations_routes import operator_bearer, local_cookie


def hosted_mode():
    return os.environ.get("STOCKPRO_HOSTED_READ_ONLY") == "1"


class ReadyResponse(BaseModel):
    state: str
    model_version: str
    production_promotion: str = "BLOCKED"
    evidence: str


class PublicRateLimit:
    """One-worker, global rolling limits protect a finite free VM/provider quota."""
    def __init__(self, maximum=30, window=60):
        self.maximum, self.window = maximum, window
        self.events = deque()
        self.lock = Lock()

    def allow(self):
        now = monotonic()
        with self.lock:
            while self.events and self.events[0] <= now - self.window:
                self.events.popleft()
            if len(self.events) >= self.maximum:
                return False
            self.events.append(now)
            return True


def ready(prospective):
    """Check local, non-provider dependencies without creating missing files."""
    try:
        pointer = json.loads((ROOT / "models/v5/ACTIVE_RESEARCH.json").read_text(encoding="utf-8"))
        if pointer["model_version"] != MODEL:
            raise ValueError("Model identity")
        for relative in pointer["models"].values():
            directory = (ROOT / relative).resolve()
            if ROOT.resolve() not in directory.parents or not (directory / "manifest.json").is_file():
                raise ValueError("Model artifact")
        canonical = prospective.path
        legacy = ROOT / "data/research/prospective-india-20261008-v1/shadow_predictions.sqlite"
        v5 = ROOT / "data/v5/predictions.sqlite"
        for path in (canonical, legacy, v5):
            if not path.is_file():
                raise ValueError("Evidence unavailable")
            with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
                if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                    raise ValueError("Evidence integrity")
                connection.execute("SELECT prediction_id FROM forecasts LIMIT 1").fetchone()
        seal = json.loads((ROOT / "reports/automation/RESEARCH_PRESERVATION.json").read_text(encoding="utf-8"))
        expected = seal["legacy_forecast_hashes"]
        if len(expected) != 388:
            raise ValueError("Original forecast seal")
        with sqlite3.connect(legacy.as_uri() + "?mode=ro", uri=True) as connection:
            actual = dict(connection.execute("SELECT prediction_id,payload_hash FROM forecasts"))
        if any(actual.get(identity) != digest for identity, digest in expected.items()):
            raise ValueError("Original forecasts")
        if hosted_mode():
            manifest = os.environ.get("STOCKPRO_EVIDENCE_MANIFEST")
            if not manifest or not Path(manifest).is_file():
                raise ValueError("Evidence manifest")
            if Path(manifest).resolve() != (ROOT / "data/EVIDENCE_MIRROR_MANIFEST.json").resolve():
                raise ValueError("Unexpected evidence manifest")
            _verify_hosted_mirror()
            _verify_model_bundles()
        return ReadyResponse(state="READY", model_version=MODEL, evidence="VERIFIED_LOCAL")
    except Exception:
        logging.getLogger(__name__).exception("Hosted readiness check failed")
        return ReadyResponse(state="NOT_READY", model_version=MODEL, evidence="UNAVAILABLE_OR_INVALID")


@lru_cache(maxsize=1)
def _verify_hosted_mirror():
    # The deployment bind mount is read-only; a refresh replaces it and restarts
    # the single worker. Hash the full package once per process before readiness.
    from scripts.evidence_mirror import verify
    verify(ROOT)


@lru_cache(maxsize=1)
def _verify_model_bundles():
    from core.forecasting.service import active_research, load_research_model
    pointer = active_research()
    if pointer["model_version"] != MODEL:
        raise ValueError("Model pointer changed")
    for family in ("india_equity", "us_equity"):
        for horizon in (1, 5, 10, 20):
            load_research_model(family, horizon, pointer)


def require_public_analysis(request: Request,
                            bearer: HTTPAuthorizationCredentials | None = Depends(operator_bearer),
                            cookie: str | None = Depends(local_cookie)):
    if hosted_mode():
        return {"mode": "STATELESS_PUBLIC"}
    return request.app.state.operator_auth.require(request)
