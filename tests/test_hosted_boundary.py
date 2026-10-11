from fastapi.testclient import TestClient


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
    response = client.get("/api/v1/stocks/TCS.NS/overview", headers={"Origin": "https://unapproved.example"})
    assert response.status_code == 429
    assert "access-control-allow-origin" not in response.headers
    assert client.get("/api/v1/stocks/search?q=TCS").status_code == 200


def test_hosted_provider_concurrency_is_bounded(tmp_path, monkeypatch):
    from api.main import create_app
    import api.main as api_main
    from api.hosted import ReadyResponse
    from application.services.operator_jobs import OperatorJobs
    from test_web_analytics import service

    monkeypatch.setenv("STOCKPRO_HOSTED_READ_ONLY", "1")
    monkeypatch.setattr(api_main, "ready", lambda _: ReadyResponse(state="READY", model_version="research-20261008T190859", evidence="VERIFIED_LOCAL"))
    svc = service()
    app = create_app(analytics=svc.analytics, web_analytics=svc, operator_jobs=OperatorJobs(tmp_path / "jobs"))
    assert app.state.provider_slots.acquire(blocking=False)
    assert app.state.provider_slots.acquire(blocking=False)
    try:
        response = TestClient(app).get("/api/v1/stocks/TCS.NS/overview")
        assert response.status_code == 429
        assert response.json()["error"]["code"] == "RATE_LIMITED"
    finally:
        app.state.provider_slots.release()
        app.state.provider_slots.release()
