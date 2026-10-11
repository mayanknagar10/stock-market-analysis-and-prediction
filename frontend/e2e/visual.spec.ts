import {test,expect} from '@playwright/test';
import {mkdir,writeFile} from 'node:fs/promises';
import {mockResearchApi} from './api-fixtures';

test('batched desktop/mobile dark/light visual and overflow audit', async({page})=>{
 test.setTimeout(300_000); await mockResearchApi(page);
 await page.addInitScript(()=>{const requested=new URL(location.href).searchParams.get('__qa_theme');if(requested==='dark'||requested==='light')localStorage.setItem('stockpro.theme',requested);});
 const pages:[string,string][]=[['dashboard','/'],['forecast','/forecast/TCS.NS?group=legacy'],['validate','/validate?group=legacy'],['overview','/stocks/TCS.NS'],['stock-chart','/stocks/TCS.NS?section=chart'],['stock-forecast','/stocks/TCS.NS?section=forecast&group=legacy'],['news','/stocks/TCS.NS?section=news'],['stock-history','/stocks/TCS.NS?section=history&group=legacy'],['technicals','/stocks/TCS.NS?section=technicals'],['fundamentals','/stocks/TCS.NS?section=fundamentals'],['factors','/stocks/TCS.NS?section=factors'],['risk','/risk/TCS.NS'],['markets','/markets'],['portfolio','/portfolio'],['watchlist','/watchlist'],['screener','/screener'],['compare','/compare'],['admin','/admin/research-status']];
 const rows:Record<string,unknown>[]=[]; const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
 await mkdir('test-results/visual',{recursive:true});
 for(const [name,route] of pages){
  for(const [width,theme] of [[1440,'dark'],[390,'light']] as const){
   await page.setViewportSize({width,height:1000});
   const start=Date.now();await page.goto(route+(route.includes('?')?'&':'?')+'__qa_theme='+theme);await expect(page.locator('main h1')).toBeVisible();
   await expect(page.locator('html')).toHaveAttribute('data-theme',theme);
   await page.waitForLoadState('networkidle');
   const metrics=await page.evaluate(()=>({width:innerWidth,scrollWidth:document.documentElement.scrollWidth,theme:document.documentElement.dataset.theme,heading:document.querySelector('main h1')?.textContent,canvas:document.querySelectorAll('main canvas').length,mainText:document.querySelector('main')?.textContent?.slice(0,400)}));
   rows.push({name,route,...metrics,elapsed_ms:Date.now()-start,fixture_based:true});
   expect.soft(metrics.scrollWidth,name+' '+width+' page overflow').toBeLessThanOrEqual(width);
   expect.soft(metrics.theme).toBe(theme);
   if(metrics.scrollWidth>width){const overflow=await page.evaluate(()=>Array.from(document.querySelectorAll('main *')).filter(e=>{const r=e.getBoundingClientRect();return r.right>innerWidth+1&&!e.closest('.table-scroll,.tabs');}).slice(0,25).map(e=>{const r=e.getBoundingClientRect(),s=getComputedStyle(e);return {tag:e.tagName,class:e.className,right:r.right,width:r.width,parent:e.parentElement?.className,display:s.display,minWidth:s.minWidth,grid:s.gridTemplateColumns};}));rows.push({name,width,overflow});}
   await page.evaluate(()=>{const badge=document.createElement('div');badge.textContent='AUTOMATED QA · FIXTURE DATA';badge.style.cssText='position:fixed;bottom:6px;right:6px;z-index:9999;background:#fff;color:#18283b;border:1px solid #536579;padding:3px 6px;font:11px sans-serif;pointer-events:none';document.body.appendChild(badge);});
   await page.screenshot({path:'test-results/visual/'+name+'-'+width+'-'+theme+'.png',animations:'disabled'});
  }
 }
 const core:[string,string][]=[['dashboard','/'],['forecast','/forecast/TCS.NS?group=legacy'],['validate','/validate?group=legacy'],['overview','/stocks/TCS.NS'],['markets','/markets'],['portfolio','/portfolio']];
 for(const width of [1280,1024,768]){
  for(const [name,route] of core){
   await page.setViewportSize({width,height:1000});
   const start=Date.now();await page.goto(route+(route.includes('?')?'&':'?')+'__qa_theme=dark');
   await expect(page.locator('main h1')).toBeVisible();await expect(page.locator('html')).toHaveAttribute('data-theme','dark');await page.waitForLoadState('networkidle');
   const metrics=await page.evaluate(()=>({width:innerWidth,scrollWidth:document.documentElement.scrollWidth,theme:document.documentElement.dataset.theme,heading:document.querySelector('main h1')?.textContent,canvas:document.querySelectorAll('main canvas').length}));
   rows.push({name,route,...metrics,elapsed_ms:Date.now()-start,fixture_based:true});
   expect.soft(metrics.scrollWidth,name+' '+width+' page overflow').toBeLessThanOrEqual(width);
   await page.evaluate(()=>{const badge=document.createElement('div');badge.textContent='AUTOMATED QA · FIXTURE DATA';badge.style.cssText='position:fixed;bottom:6px;right:6px;z-index:9999;background:#fff;color:#18283b;border:1px solid #536579;padding:3px 6px;font:11px sans-serif;pointer-events:none';document.body.appendChild(badge);});
   await page.screenshot({path:'test-results/visual/'+name+'-'+width+'-dark.png',animations:'disabled'});
  }
 }
 await writeFile('test-results/visual/audit.json',JSON.stringify({rows,errors,mode:'OFFLINE_CONTRACT_FIXTURES'},null,2));
 expect(errors).toEqual([]);
});
