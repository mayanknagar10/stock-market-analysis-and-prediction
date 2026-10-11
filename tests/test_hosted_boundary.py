from fastapi.testclient import TestClient


def test_tunnel_read_only_mode_accepts_verified_local_evidence_without_mirror(tmp_path, monkeypatch):
    import json
    import sqlite3
    from types import SimpleNamespace
    import api.hosted as hosted

    monkeypatch.setenv("STOCKPRO_HOSTED_READ_ONLY", "1")
    monkeypatch.setenv("STOCKPRO_TUNNEL_LOCAL_EVIDENCE", "1")
    monkeypatch.setenv("STOCKPRO_ENV", "development")
    monkeypatch.setattr(hosted, "ROOT", tmp_path)
    monkeypatch.setattr(hosted, "_verify_model_bundles", lambda: None)
    monkeypatch.setattr(hosted, "_verify_hosted_mirror", lambda: (_ for _ in ()).throw(AssertionError("mirror must not be read")))

    model_dir = tmp_path / "models" / "v5" / "bundle"
    model_dir.mkdir(parents=True)
    (model_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "models" / "v5" / "ACTIVE_RESEARCH.json").write_text(
        json.dumps({"model_version": hosted.MODEL, "models": {"india_1d": "models/v5/bundle"}}), encoding="utf-8"
    )
    expected = {f"original-{i}": f"hash-{i}" for i in range(388)}
    seal = tmp_path / "reports" / "automation" / "RESEARCH_PRESERVATION.json"
    seal.parent.mkdir(parents=True)
    seal.write_text(json.dumps({"legacy_forecast_hashes": expected}), encoding="utf-8")
    canonical = tmp_path / "data" / "prospective_automation" / "collection.sqlite"
    legacy = tmp_path / "data" / "research" / "prospective-india-20261008-v1" / "shadow_predictions.sqlite"
    v5 = tmp_path / "data" / "v5" / "predictions.sqlite"
    for path in (canonical, legacy, v5):
        path.parent.mkdir(parents=True)
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE forecasts (prediction_id TEXT, payload_hash TEXT)")
            if path == legacy:
                connection.executemany("INSERT INTO forecasts VALUES (?, ?)", expected.items())

    result = hosted.ready(SimpleNamespace(path=canonical))
    assert result.state == "READY"
    assert result.evidence == "VERIFIED_LOCAL"

    monkeypatch.delenv("STOCKPRO_TUNNEL_LOCAL_EVIDENCE")
    assert hosted.ready(SimpleNamespace(path=canonical)).state == "NOT_READY"


def test_hosted_mode_blocks_operator_and_allows_bounded_stateless_analysis(tmp_path, monkeypatch):
    from api.main import create_app
    import api.main as api_main
    from api.hosted import ReadyResponse
    from application.services.operator_jobs import OperatorJobs
    from application.services.prospective import ProspectiveService
    from test_web_analytics import service

    monkeypatch.setenv("STOCKPRO_HOSTED_READ_ONLY", "1")
    monkeypatch.setenv("STOCKPRO_ENV", "production")
    monkeypatch.setenv("STOCKPRO_OPERATOR_TOKEN", "private")
    monkeypatch.setattr(api_main, "ready", lambda _: ReadyResponse(state="READY", model_version="research-20261008T190859", evidence="VERIFIED_LOCAL"))
    svc = service()
    client = TestClient(create_app(ProspectiveService(tmp_path), svc.analytics,
                                   web_analytics=svc, operator_jobs=OperatorJobs(tmp_path / "jobs")))
    headers = {"Authorization": "Bearer private"}
    assert client.post("/api/v1/operations/jobs", headers=headers,
                       json={"job_type": "verify"}).status_code == 403
    assert client.get("/api/v1/operations/status", headers=headers).status_code == 403
    assert client.get("/api/v1/system/performance", headers=headers).status_code == 403
    body = {"holdings": [{"ticker": "TCS.NS", "weight": 1}]}
    assert client.post("/api/v1/analysis/portfolio", json=body).status_code == 200
    large = b"{" + b" " * 17000 + b"}"
    response = client.post("/api/v1/analysis/portfolio", content=large,
                           headers={"Content-Type": "application/json"})
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "REQUEST_TOO_LARGE"


def test_hosted_readiness_fails_closed_without_evidence(tmp_path, monkeypatch):
    from api.main import create_app
    from application.services.operator_jobs import OperatorJobs
    from application.services.prospective import ProspectiveService

    monkeypatch.setenv("STOCKPRO_HOSTED_READ_ONLY", "1")
    monkeypatch.setenv("STOCKPRO_ENV", "production")
    client = TestClient(create_app(ProspectiveService(tmp_path), operator_jobs=OperatorJobs(tmp_path / "jobs")))
    assert client.get("/api/v1/health").status_code == 200
    response = client.get("/api/v1/ready")
    assert response.status_code == 503
    assert response.json()["state"] == "NOT_READY"
    assert not (tmp_path / "collection.sqlite").exists()
    assert client.get("/api/v1/validation/prospective").status_code == 503


def test_hosted_rate_limit_and_exact_cors(tmp_path, monkeypatch):
    from api.main import create_app
    import api.main as api_main
    from api.hosted import ReadyResponse
    from api.hosted import PublicRateLimit
    from application.services.operator_jobs import OperatorJobs
    from test_web_analytics import service

    monkeypatch.setenv("STOCKPRO_HOSTED_READ_ONLY", "1")
    monkeypatch.setenv("STOCKPRO_CORS_ORIGINS", "https://stockpro.example")
    monkeypatch.setattr(api_main, "ready", lambda _: ReadyResponse(state="READY", model_version="research-20261008T190859", evidence="VERIFIED_LOCAL"))
    svc = service()
    app = create_app(analytics=svc.analytics, web_analytics=svc, operator_jobs=OperatorJobs(tmp_path / "jobs"))
    app.state.public_rate_limit = PublicRateLimit(maximum=1)
    client = TestClient(app)
    response = client.get("/api/v1/stocks/TCS.NS/overview", headers={"Origin": "https://stockpro.example"})
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://stockpro.example"
    response = client.get("/api/v1/stocks/TCS.NS/overview", headers={"Origin": "https://stockpro.example"})
    assert response.status_code == 429
    assert response.headers["access-control-allow-origin"] == "https://stockpro.example"
    response = client.get("/api/v1/stocks/TCS.NS/overview", headers={"Origin": "https://unapproved.example"})
    assert response.status_code == 429
    assert "access-control-allow-origin" not in response.headers
    response = client.get("/api/v1/operations/status", headers={"Origin": "https://stockpro.example"})
    assert response.status_code == 403
    assert response.headers["access-control-allow-origin"] == "https://stockpro.example"
    assert client.get("/api/v1/stocks/search?q=TCS").status_code == 200


def test_hosted_provider_concurrency_is_bounded(tmp_path, monkeypatch):
    import asyncio
    from api.main import create_app
    import api.main as api_main
    from api.hosted import ReadyResponse
    from application.services.operator_jobs import OperatorJobs
    from test_web_analytics import service

    monkeypatch.setenv("STOCKPRO_HOSTED_READ_ONLY", "1")
    monkeypatch.setattr(api_main, "ready", lambda _: ReadyResponse(state="READY", model_version="research-20261008T190859", evidence="VERIFIED_LOCAL"))
    svc = service()
    app = create_app(analytics=svc.analytics, web_analytics=svc, operator_jobs=OperatorJobs(tmp_path / "jobs"))
    app.state.provider_slots = asyncio.Semaphore(0)
    app.state.provider_wait_seconds = 0.01
    response = TestClient(app).get("/api/v1/stocks/TCS.NS/overview")
    assert response.status_code == 429
    assert response.json()["error"]["code"] == "RATE_LIMITED"


def test_hosted_provider_requests_wait_briefly_for_capacity(tmp_path, monkeypatch):
    import asyncio
    import time
    from httpx import ASGITransport, AsyncClient
    from api.main import create_app
    import api.main as api_main
    from api.hosted import ReadyResponse
    from test_web_analytics import service

    monkeypatch.setenv("STOCKPRO_HOSTED_READ_ONLY", "1")
    monkeypatch.setattr(api_main, "ready", lambda _: ReadyResponse(state="READY", model_version="research-20261008T190859", evidence="VERIFIED_LOCAL"))
    svc = service()
    original = svc.analytics.overview

    def slow_overview(ticker):
        time.sleep(0.2)
        return original(ticker)

    monkeypatch.setattr(svc.analytics, "overview", slow_overview)
    app = create_app(analytics=svc.analytics, web_analytics=svc)

    async def concurrent_reads():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            return await asyncio.gather(*(
                client.get("/api/v1/stocks/TCS.NS/overview") for _ in range(3)
            ))

    assert [response.status_code for response in asyncio.run(concurrent_reads())] == [200, 200, 200]
