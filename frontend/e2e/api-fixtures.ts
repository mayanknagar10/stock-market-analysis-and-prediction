import type { Page } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

type Envelope = { source: string; responses: Record<string, unknown> };
export async function mockResearchApi(page: Page, operator = false) {
 const fixture = JSON.parse(await readFile(resolve('e2e/fixtures/contracts.json'), 'utf8')) as Envelope;
 if (fixture.source !== 'OFFLINE_TEST_FIXTURES_NOT_PRODUCT_DATA') throw new Error('Fixture provenance missing');
 await page.route('**/api/v1/**', async route => {
  const url = new URL(route.request().url()); const path = url.pathname.replace('/api/v1', ''); const group = url.searchParams.get('group') || 'canonical';
  if (decodeURIComponent(url.href).includes('INVALID$')) { await route.fulfill({status:422,contentType:'application/json',body:JSON.stringify({error:{code:'INVALID_TICKER',message:'Unsupported equity ticker.',request_id:'invalid-ticker-fixture'}})});return; }
  let key = path;
  if (path === '/stocks/search') key = url.searchParams.get('q') ? '/stocks/search?q=TCS&limit=10' : '/stocks/search?limit=100';
  if (path === '/validation/summary') key = '/validation/prospective?group=canonical';
  if (path === '/validation/prospective') key = path + '?group=' + group;
  if (path === '/stocks/TCS.NS/history/series') key = path + '?latest=true&limit=1000';
  if (path === '/stocks/TCS.NS/technicals/series') key = path + '?group=' + (url.searchParams.get('group') || 'trend');
  if (path === '/stocks/TCS.NS/forecast' && group === 'legacy') key = path + '?group=legacy';
  if (path === '/research/forecasts') key = group === 'legacy' ? '/research/forecasts?group=legacy&ticker=TCS.NS&limit=8' : '';
  if (/^\/research\/forecasts\/[^/]+$/.test(path)) key = path + '?group=' + group;
  let body = fixture.responses[key]; let status = 200;
  if(body===undefined && /^\/stocks\/RELIANCE\.NS\/(overview|fundamentals|news|history\/series|technicals(?:\/series)?|risk\/report|factors\/report)/.test(path)){
    const alias=key.replace('/stocks/RELIANCE.NS/','/stocks/TCS.NS/');
    const candidate=fixture.responses[alias] ?? (path.endsWith('/history/series')?fixture.responses['/stocks/TCS.NS/history/series?latest=true&limit=1000']:undefined);
    if(candidate && typeof candidate==='object')body='ticker' in candidate?{...candidate,ticker:'RELIANCE.NS'}:candidate;
  }
  if (operator && path === '/operations/session') body = {authenticated:true,mode:'LOCAL_LOOPBACK',expires_at:'2026-10-10T23:00:00Z'};
  if (operator && path === '/operations/status') body = {backend_available:true,execution_mode:'LOCAL_PC_MANUAL',cloud_independence:'DEFERRED',running_job:null,timeout_seconds:7200};
  if (operator && path === '/operations/jobs') body = {items:[],total:0,offset:0,limit:100};
  if (path === '/research/forecasts' && group === 'canonical') body = { items: [], total: 0, offset: 0, limit: 20, order: 'FORECAST_AS_OF_DESC_PREDICTION_ID_ASC' };
  if (body === undefined) { status = path.startsWith('/operations/') ? 401 : 404; body = { error: { code: path.startsWith('/operations/') ? 'OPERATOR_UNAUTHORIZED' : 'MODEL_UNAVAILABLE', message: path.startsWith('/operations/') ? 'Operator access required.' : 'No eligible saved research data are available.', request_id: 'offline-browser-fixture' } }; }
  await route.fulfill({status, contentType: 'application/json', body: JSON.stringify(body)});
 });
}
