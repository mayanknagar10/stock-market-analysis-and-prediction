from fastapi.testclient import TestClient

def test_web_schema_routes_and_protected_post_requests(tmp_path,monkeypatch):
    from api.main import create_app
    from test_web_analytics import service
    from application.services.operator_jobs import OperatorJobs
    from application.services.prospective import ProspectiveService
    svc=service();jobs=OperatorJobs(tmp_path/'jobs',dispatch=lambda task:None)
    monkeypatch.setenv('STOCKPRO_OPERATOR_TOKEN','private')
    client=TestClient(create_app(ProspectiveService(tmp_path),svc.analytics,web_analytics=svc,operator_jobs=jobs))
    for route in ('fundamentals','news','history/series','technicals/series','risk/report','factors/report'):
        response=client.get('/api/v1/stocks/TCS.NS/'+route)
        assert response.status_code==200,response.text
    assert client.get('/api/v1/markets/context').status_code==200
    assert client.get('/api/v1/validation/historical/v5/final').json()['final_gate']=='FAILED'
    assert client.get('/api/v1/research/forecasts?group=v5').status_code==200
    body={'holdings':[{'ticker':'TCS.NS','weight':1}]}
    assert client.post('/api/v1/analysis/portfolio',json=body).status_code==401
    assert client.post('/api/v1/analysis/portfolio',headers={'Authorization':'Bearer private'},json=body).status_code==200
    for invalid in ({'holdings':[{'ticker':'TCS.NS','weight':-1}]},{'holdings':[{'ticker':'TCS.NS','weight':0}]},{'holdings':[{'ticker':'TCS.NS','weight':float('inf')}] }):
        # JSON cannot carry infinity: raw body tests the parser/finite schema.
        import json
        assert client.post('/api/v1/analysis/portfolio',headers={'Authorization':'Bearer private','Content-Type':'application/json'},content=json.dumps(invalid)).status_code==422
    assert not list((tmp_path/'jobs').glob('*.jsonl'))

def test_operator_cors_preflight_with_global_read_auth(tmp_path,monkeypatch):
    from api.main import create_app
    from application.services.operator_jobs import OperatorJobs
    monkeypatch.setenv('STOCKPRO_API_TOKEN','public')
    monkeypatch.setenv('STOCKPRO_CORS_ORIGINS','http://localhost:3000')
    client=TestClient(create_app(operator_jobs=OperatorJobs(tmp_path)))
    response=client.options('/api/v1/operations/jobs',headers={'Origin':'http://localhost:3000','Access-Control-Request-Method':'POST','Access-Control-Request-Headers':'content-type'})
    assert response.status_code==200 and response.headers['access-control-allow-credentials']=='true'
    assert response.headers['access-control-allow-origin']=='http://localhost:3000'

def test_openapi_declares_operator_security_and_typed_finance(tmp_path):
    from api.main import create_app
    from application.services.operator_jobs import OperatorJobs
    schema=create_app(operator_jobs=OperatorJobs(tmp_path)).openapi()
    security=schema['paths']['/api/v1/operations/jobs']['post']['security']
    assert {'OperatorBearer':[]} in security and {'LocalOperatorCookie':[]} in security
    assert schema['components']['schemas']['FactorResponse']['properties']['exposures']['anyOf'][0]['$ref'].endswith('/FactorExposures')

def test_actual_generation_timestamp_is_distinct_from_fixed_cutoff(tmp_path):
    from test_automation_api import setup
    client,ticker,prospective=setup(tmp_path)
    row=prospective.rows()[0]
    item=client.get('/api/v1/stocks/'+ticker+'/forecast/history').json()['items'][0]
    from datetime import datetime
    assert datetime.fromisoformat(item['generated_at'])==datetime.fromisoformat(row['generated_at'])
    assert item['generated_at']!=item['information_cutoff']
    item=client.get('/api/v1/research/forecasts').json()['items'][0]
    assert item['generated_at'].startswith('2026-10-10T01:01:00')


def test_published_openapi_matches_live_contract(tmp_path):
    import json
    from application.config import ROOT
    from api.main import create_app
    from application.services.operator_jobs import OperatorJobs
    published=json.loads((ROOT/"docs/api/openapi.json").read_text(encoding="utf-8"))
    current=create_app(operator_jobs=OperatorJobs(tmp_path)).openapi()
    assert published==current, "Export OpenAPI and regenerate frontend types after contract changes"
