import { test, expect } from '@playwright/test';
import { mockResearchApi } from './api-fixtures';

test('search → forecast → chart → drivers → outcomes → validation → dashboard', async ({page}) => {
 await mockResearchApi(page); await page.goto('/');
 const search=page.getByRole('combobox',{name:'Search companies or tickers'});
 await search.fill('TCS'); await expect(page.getByRole('option')).toBeVisible();
 await search.press('ArrowDown'); await search.press('Enter');
 await expect(page).toHaveURL(/\/stocks\/TCS\.NS/);
 await page.locator('.sidebar').getByRole('link',{name:'Forecast',exact:true}).click();
 await expect(page.getByText(/No canonical forecasts recorded for/)).toBeVisible();
 await page.getByRole('link',{name:'View preserved manual forecasts'}).click();
 await expect(page.locator('.horizon')).toHaveCount(4);
 await expect(page.getByText('No validated statistical edge').first()).toBeVisible();
 await expect(page.locator('.price-chart canvas').first()).toBeVisible();
 expect(await page.locator('main').innerText()).not.toMatch(/\b(BUY|SELL|STRONG BUY)\b/);
 await page.getByRole('link',{name:'Drivers',exact:true}).click();
 await expect(page.getByRole('heading',{name:'Saved model contributions'})).toBeVisible();
 await page.getByRole('link',{name:'Prediction history',exact:true}).click();
 await expect(page.getByRole('columnheader',{name:'Realized return'})).toBeVisible();
 await expect(page.getByText('Pending / unavailable').first()).toBeVisible();
 await expect(page.getByText('matured',{exact:true}).first()).toBeVisible();
 await page.locator('.sidebar').getByRole('link',{name:'Validate',exact:true}).click();
 await expect(page.getByText(/FAILED/).first()).toBeVisible();
 await page.locator('.sidebar').getByRole('link',{name:'Dashboard',exact:true}).click();
 await expect(page.getByRole('heading',{name:'Dashboard',exact:true})).toBeVisible();
});

test('mobile navigation keeps keyboard and responsive workflows usable', async ({page}) => {
 await page.setViewportSize({width:390,height:844}); await mockResearchApi(page); await page.goto('/');
 await page.getByRole('button',{name:'Open navigation'}).click();
 const drawer=page.getByRole('dialog',{name:'Navigation'}); await expect(drawer).toBeVisible();
 await drawer.getByRole('link',{name:'Validate',exact:true}).click();
 await expect(drawer).not.toBeVisible();
 await expect(page.locator('main h1')).toBeVisible();
 expect(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth)).toBe(true);
 await page.getByRole('button',{name:'Use light theme'}).click();
 await expect(page.locator('html')).toHaveAttribute('data-theme','light');
});

test('provider failure and unavailable saved forecasts offer honest recovery', async ({page}) => {
 await mockResearchApi(page);
 await page.route('**/api/v1/**',route=>route.fulfill({status:503,contentType:'application/json',body:JSON.stringify({error:{code:'PROVIDER_FAILURE',message:'Market provider is unavailable.',request_id:'browser-failure-fixture'}})}));
 await page.goto('/forecast/TCS.NS');
 await expect(page.getByRole('button',{name:'Retry'}).first()).toBeVisible();
 await expect(page.getByText('Market provider is unavailable.').first()).toBeVisible();
 expect(await page.locator('main').innerText()).not.toMatch(/Traceback|\bBUY\b|\bSELL\b/);
});

test('invalid ticker never fabricates a forecast', async ({page}) => {
 await mockResearchApi(page); await page.goto('/forecast/INVALID%24%24');
 await expect(page.getByRole('button',{name:'Retry'}).first()).toBeVisible();
 expect(await page.locator('main').innerText()).not.toMatch(/\bBUY\b|\bSELL\b/);
});
