import {test} from '@playwright/test';
import {mkdir,writeFile} from 'node:fs/promises';
test('inspect public Tradytics chart workspace without account actions',async({page})=>{
 test.skip(process.env.RUN_PUBLIC_BENCHMARK!=='1','Public benchmark already inspected once; enable explicitly for a fresh external inspection.');
 try{
  const response=await page.goto('https://tradytics.com/charts',{waitUntil:'domcontentloaded',timeout:30000});
  await mkdir('test-results/benchmark',{recursive:true});
  await page.screenshot({path:'test-results/benchmark/tradytics-public-charts.png'});
  await writeFile('test-results/benchmark/inspection.json',JSON.stringify({status:response?.status(),title:await page.title(),public_only:true,text:(await page.locator('body').innerText()).slice(0,2000)},null,2));
 }catch{test.skip(true,'Public benchmark browser unavailable; primary web-tool pages were reviewed separately.');}
});
