import {test,expect} from '@playwright/test';
import {mockResearchApi} from './api-fixtures';
import {mkdir} from 'node:fs/promises';
test('internal backtest blueprint requires local operator and never runs a strategy',async({page})=>{
 await mockResearchApi(page,true);await mkdir('test-results/visual',{recursive:true});
 for(const width of [1440,390]){
  await page.setViewportSize({width,height:1000});await page.goto('/admin/research-tools');
  await expect(page.getByRole('heading',{name:'Strategy backtesting',exact:true})).toBeVisible();
  await expect(page.getByRole('button',{name:'Run backtest · awaiting API'})).toBeDisabled();
  await expect(page.getByText('Execution unavailable · API integration required')).toBeVisible();
  expect(await page.evaluate(()=>document.documentElement.scrollWidth)).toBeLessThanOrEqual(width);
  await page.screenshot({path:'test-results/visual/backtesting-'+width+'.png'});
 }
});
