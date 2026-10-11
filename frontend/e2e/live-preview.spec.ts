import { test, expect } from '@playwright/test';

const preview = process.env.STOCKPRO_LIVE_PREVIEW_URL;
test.skip(!preview, 'Set STOCKPRO_LIVE_PREVIEW_URL to audit a deployed preview with its real backend.');

test('deployed stock workspace retrieves real TCS data without backend overload', async ({ page }) => {
  const apiFailures: string[] = [];
  page.on('response', response => {
    if (response.url().includes('/api/v1/') && response.status() >= 400)
      apiFailures.push(`${response.status()} ${new URL(response.url()).pathname}`);
  });
  await page.goto(`${preview}/stocks/TCS.NS`, { waitUntil: 'domcontentloaded' });
  await expect(page.getByRole('heading', { name: /TCS/ }).first()).toBeVisible();
  await expect(page.locator('.price-chart canvas').first()).toBeVisible({ timeout: 45_000 });
  if (process.env.STOCKPRO_CAPTURE_QA) await page.screenshot({ path: 'test-results/live-stock.png', fullPage: true });
  await page.waitForTimeout(20_000);
  expect(apiFailures).toEqual([]);
  expect(await page.locator('main').innerText()).not.toMatch(/\b(BUY|SELL|STRONG BUY)\b/);
});

test('deployed market, forecast and validation flow preserves research semantics', async ({ page }) => {
  test.setTimeout(120_000);
  await page.goto(`${preview}/`, { waitUntil: 'domcontentloaded' });
  await expect(page.getByRole('heading', { name: 'Dashboard' })).toBeVisible();
  await expect(page.getByText('Nifty 50').first()).toBeVisible({ timeout: 45_000 });
  if (process.env.STOCKPRO_CAPTURE_QA) await page.screenshot({ path: 'test-results/live-dashboard.png', fullPage: true });
  await page.goto(`${preview}/markets`, { waitUntil: 'domcontentloaded' });
  await expect(page.getByRole('heading', { name: 'Markets', exact: true })).toBeVisible();
  await expect(page.getByText('Nifty 50').first()).toBeVisible({ timeout: 45_000 });
  await page.goto(`${preview}/forecast/TCS.NS?group=legacy`, { waitUntil: 'domcontentloaded' });
  await expect(page.locator('.horizon')).toHaveCount(4, { timeout: 45_000 });
  expect(await page.locator('main').innerText()).toMatch(/Research only/i);
  expect(await page.locator('main').innerText()).not.toMatch(/\b(BUY|SELL|STRONG BUY)\b/);
  await page.goto(`${preview}/validate`, { waitUntil: 'domcontentloaded' });
  await expect(page.getByText(/FAILED/).first()).toBeVisible({ timeout: 45_000 });
  await expect(page.getByText(/Prospective/i).first()).toBeVisible();
  if (process.env.STOCKPRO_CAPTURE_QA) await page.screenshot({ path: 'test-results/live-validate.png', fullPage: true });
});

test('deployed stock sections and ticker switching retain the selected security', async ({ page }) => {
  test.setTimeout(120_000);
  await page.goto(`${preview}/stocks/TCS.NS?section=fundamentals`, { waitUntil: 'domcontentloaded' });
  await expect(page.getByRole('heading', { name: 'Valuation' })).toBeVisible({ timeout: 45_000 });
  const tabs = page.getByRole('navigation', { name: 'Company research sections' });
  await tabs.getByRole('link', { name: 'Technicals' }).click();
  await expect(page.getByRole('heading', { name: 'Latest selected indicator' })).toBeVisible({ timeout: 45_000 });
  await tabs.getByRole('link', { name: 'Factors' }).click();
  await expect(page.getByRole('heading', { name: 'Factor signals' })).toBeVisible({ timeout: 45_000 });
  await tabs.getByRole('link', { name: 'News' }).click();
  await expect(page.getByText(/News unavailable|No recent news|No news/i).first()).toBeVisible({ timeout: 45_000 });
  await page.goto(`${preview}/stocks/INFY.NS`, { waitUntil: 'domcontentloaded' });
  await expect(page.locator('main').getByText('INFY.NS').first()).toBeVisible();
  await expect(page.locator('.price-chart canvas').first()).toBeVisible({ timeout: 45_000 });
});

test('deployed public research tools return backend calculations', async ({ page }) => {
  test.setTimeout(180_000);
  await page.goto(`${preview}/screener`, { waitUntil: 'domcontentloaded' });
  await page.getByRole('combobox', { name: 'Ticker to add' }).fill('TCS.NS');
  await page.getByRole('button', { name: 'Add', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Run screen' })).toBeEnabled({ timeout: 30_000 });
  await page.getByRole('button', { name: 'Run screen' }).click();
  await expect(page.getByRole('heading', { name: 'Screen results' })).toBeVisible({ timeout: 45_000 });

  await page.goto(`${preview}/compare?ticker=TCS.NS`, { waitUntil: 'domcontentloaded' });
  await page.getByRole('combobox', { name: 'Ticker to add' }).fill('INFY.NS');
  await page.getByRole('button', { name: 'Add', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Run comparison' })).toBeEnabled({ timeout: 30_000 });
  await page.getByRole('button', { name: 'Run comparison' }).click();
  await expect(page.getByRole('heading', { name: 'Normalized price history' })).toBeVisible({ timeout: 60_000 });

  await page.goto(`${preview}/portfolio`, { waitUntil: 'domcontentloaded' });
  await page.getByRole('combobox', { name: 'Ticker to add' }).fill('TCS.NS');
  await page.getByRole('button', { name: 'Add', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Analyze portfolio' })).toBeEnabled({ timeout: 30_000 });
  await page.getByRole('button', { name: 'Analyze portfolio' }).click();
  await expect(page.getByRole('heading', { name: 'Hypothetical allocation value' })).toBeVisible({ timeout: 60_000 });
});

test('deployed watchlist, risk and scenarios use the public read-only API', async ({ page }) => {
  test.setTimeout(120_000);
  await page.goto(`${preview}/watchlist`, { waitUntil: 'domcontentloaded' });
  await page.locator('.watchlist-entry input[aria-label="Ticker"]').fill('TCS.NS');
  await page.locator('.watchlist-entry input[aria-label="Quantity"]').fill('2');
  await page.locator('.watchlist-entry input[aria-label="Average Buy"]').fill('2000');
  await page.getByRole('button', { name: 'Add position' }).click();
  await expect(page.getByRole('heading', { name: 'Saved guest positions' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Refresh backend valuation' })).toBeEnabled({ timeout: 30_000 });
  await page.getByRole('button', { name: 'Refresh backend valuation' }).click();
  await expect(page.getByRole('heading', { name: 'Current tracking valuation' })).toBeVisible();
  await expect(page.getByRole('columnheader', { name: 'Last close' })).toBeVisible({ timeout: 45_000 });

  await page.goto(`${preview}/risk/TCS.NS`, { waitUntil: 'domcontentloaded' });
  await expect(page.getByRole('heading', { name: 'Risk measurements' })).toBeVisible({ timeout: 45_000 });
  await page.goto(`${preview}/forecast/TCS.NS?group=legacy&tab=scenarios`, { waitUntil: 'domcontentloaded' });
  await expect(page.getByRole('button', { name: 'Run conditional scenario' })).toBeEnabled({ timeout: 45_000 });
  await page.getByRole('button', { name: 'Run conditional scenario' }).click();
  await expect(page.getByText('Conditional return')).toBeVisible({ timeout: 45_000 });
  await page.goto(`${preview}/admin/research-tools`, { waitUntil: 'domcontentloaded' });
  await expect(page.getByRole('heading', { name: 'Operator connection required' })).toBeVisible();
});

test('deployed preview uses one page notice and quiet sections when the tunnel is offline', async ({ page }) => {
  await page.route('**/api/v1/**', route => route.abort());
  await page.goto(`${preview}/stocks/TCS.NS`, { waitUntil: 'domcontentloaded' });
  await expect(page.getByText('Market data is unavailable while the local backend is disconnected.')).toBeVisible({ timeout: 30_000 });
  await expect(page.locator('.backend-page-notice')).toHaveCount(1);
  await expect(page.getByText('Unavailable while the backend is offline.').first()).toBeVisible();
  expect(await page.locator('main .notice.danger').count()).toBe(0);
});

test('deployed preview can switch temporary backend URLs locally without operator controls', async ({ page }) => {
  const backend = process.env.STOCKPRO_LIVE_BACKEND_URL;
  test.skip(!backend, 'Set STOCKPRO_LIVE_BACKEND_URL to exercise the temporary connection setting.');
  await page.goto(`${preview}/admin/research-status`, { waitUntil: 'domcontentloaded' });
  await expect(page.getByRole('heading', { name: 'Preview backend connection' })).toBeVisible();
  await expect(page.getByText('Manual research jobs are available only in the local internal console.')).toBeVisible({ timeout: 30_000 });
  expect(await page.getByRole('button', { name: 'Submit Verify job' }).count()).toBe(0);
  await page.getByLabel('Temporary HTTPS tunnel URL').fill(backend!);
  await page.getByRole('button', { name: 'Save backend URL' }).click();
  await page.getByRole('link', { name: 'Reload to connect' }).click();
  await expect(page.getByText('API connected')).toBeVisible({ timeout: 30_000 });
});
