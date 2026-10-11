import { expect, test } from '@playwright/test';

test.skip(!process.env.STOCKPRO_LOCAL_FORECAST_QA, 'Run against the local Next.js build and real FastAPI backend.');

test('four saved Indian forecasts keep math, abstention and ticker context honest', async ({ page, request }) => {
  test.setTimeout(180_000);
  const failures: string[] = [];
  page.on('response', response => { if (response.url().includes('/api/v1/') && response.status() >= 500) failures.push(`${response.status()} ${response.url()}`); });
  for (const ticker of ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS']) {
    const apiResponse = await request.get(`http://127.0.0.1:8000/api/v1/stocks/${ticker}/forecast?group=legacy`);
    expect(apiResponse.ok()).toBe(true);
    const saved = await apiResponse.json();
    expect(saved.horizons.map((h: { horizon: number }) => h.horizon)).toEqual([1, 5, 10, 20]);
    for (const h of saved.horizons) {
      expect(h.predicted_price).toBeCloseTo(saved.current_price * Math.exp(h.predicted_log_return), 5);
      expect(h.quantile_prices).toEqual([...h.quantile_prices].sort((a: number, b: number) => a - b));
      expect(h.probability).toBeNull();
      expect(h.abstain).toBe(true);
    }
    await page.goto(`/forecast/${ticker}?group=legacy`, { waitUntil: 'domcontentloaded' });
    await expect(page.getByRole('heading', { name: 'Final overall forecast' })).toBeVisible({ timeout: 45_000 });
    await expect(page.getByText('No strong statistical edge')).toBeVisible();
    await expect(page.getByText('Implied total-return-equivalent price')).toBeVisible();
    await expect(page.locator('.endpoint-chart')).toBeVisible();
    await expect(page.getByText('Can I trust this forecast?')).toBeVisible();
    await expect(page.locator('main').getByText(ticker).first()).toBeVisible();
    expect(await page.locator('main').innerText()).not.toMatch(/\b(BUY|SELL|STRONG BUY)\b/);
    if (ticker === 'TCS.NS') {
      await expect(page.locator('.price-chart canvas').first()).toBeVisible({ timeout: 45_000 });
      await page.screenshot({ path: 'test-results/live-forecast-tcs.png', fullPage: true });
    }
  }
  expect(failures).toEqual([]);
  await page.goto('/forecast/TCS.NS?group=canonical');
  await expect(page.getByText('No canonical forecasts recorded for TCS.NS')).toBeVisible({ timeout: 30_000 });
});

test('slow current context does not block the saved forecast', async ({ page }) => {
  test.setTimeout(90_000);
  await page.route('**/api/v1/markets/context', async route => { await new Promise(resolve => setTimeout(resolve, 12_000)); await route.continue(); });
  await page.goto('/forecast/TCS.NS?group=legacy', { waitUntil: 'domcontentloaded' });
  await expect(page.getByRole('heading', { name: 'Final overall forecast' })).toBeVisible({ timeout: 30_000 });
  await expect(page.getByText('No strong statistical edge')).toBeVisible();
  await expect(page.getByText('Current context')).toBeVisible();
});

test('ticker switching keeps forecast evidence tied to the selected company', async ({ page }) => {
  test.setTimeout(90_000);
  await page.goto('/forecast/TCS.NS?group=legacy');
  await expect(page.getByRole('heading', { name: 'Final overall forecast' })).toBeVisible({ timeout: 30_000 });
  await page.getByRole('combobox', { name: 'Search companies or tickers' }).fill('INFY');
  await page.getByRole('option').filter({ hasText: 'INFY.NS' }).click();
  await expect(page).toHaveURL(/\/stocks\/INFY\.NS/);
  await page.getByRole('navigation', { name: 'Primary navigation' }).getByRole('link', { name: 'Forecast' }).click();
  await page.getByRole('link', { name: 'Legacy / manual' }).click();
  await expect(page.getByRole('heading', { name: 'Final overall forecast' })).toBeVisible({ timeout: 30_000 });
  await expect(page.locator('main').getByText('INFY.NS').first()).toBeVisible();
  expect(await page.locator('main').innerText()).not.toContain('TCS.NS');
});
