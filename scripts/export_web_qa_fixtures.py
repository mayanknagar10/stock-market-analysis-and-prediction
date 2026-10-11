"""Offline browser-only contracts from existing Python fixture services; no live capture."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
sys.path.insert(0,str(ROOT))
from fastapi.testclient import TestClient
from api.main import create_app
from test_web_analytics import service
from application.services.prospective import ProspectiveService
from application.services.operator_jobs import OperatorJobs

def main():
    with TemporaryDirectory() as temp:
        web=service()
        client=TestClient(create_app(ProspectiveService(Path(temp)),web.analytics,web_analytics=web,operator_jobs=OperatorJobs(Path(temp)/'jobs',dispatch=lambda task:None)))
        routes=['/health','/model/status','/model/health','/system/status','/stocks/search?q=TCS&limit=10',
            '/stocks/search?limit=100','/stocks/TCS.NS/overview','/stocks/TCS.NS/fundamentals',
            '/stocks/TCS.NS/news','/stocks/TCS.NS/history/series?latest=true&limit=1000',
            '/stocks/TCS.NS/technicals','/stocks/TCS.NS/risk/report','/stocks/TCS.NS/factors/report',
            '/markets/context','/data/health','/collection/status']
        routes += ['/stocks/TCS.NS/technicals/series?group='+g for g in ['trend','momentum','volatility','volume']]
        results={}
        for route in routes:
            result=client.get('/api/v1'+route)
            if result.status_code!=200:raise RuntimeError(route+' failed: '+str(result.status_code))
            results[route]=result.json()
        # Real preserved public research payloads are read-only; fake market bars
        # remain exclusively test data. No fixture is served by production code.
        evidence=json.loads((ROOT/'.superpowers/sdd/web-platform/read-api-checks.json').read_text(encoding='utf-8'))
        for item in evidence:
            if item['status']==200 and item['path'].startswith(('/validation/','/research/','/stocks/TCS.NS/forecast')):
                results[item['path']]=item['body']
        output=ROOT/'frontend/e2e/fixtures/contracts.json'
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps({'source':'OFFLINE_TEST_FIXTURES_NOT_PRODUCT_DATA','responses':results},indent=2),encoding='utf-8')
        print(json.dumps({'responses':len(results),'fixture_only':True,'models_trained':False,'live_collection':False}))
if __name__=='__main__':main()
