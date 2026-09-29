import { expect, test } from '@playwright/test';

test('tokenizes, replays merges by keyboard, and fits the supported viewport', async ({
  page,
}, testInfo) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto('/tokenizer');
  await expect(page.getByRole('heading', { name: 'Tokenize any text' })).toBeVisible();

  const input = page.getByLabel('Input text');
  await input.fill('Rare bird: 🐦 <|bos|>');
  await expect(page.locator('.token-stats')).toContainText('Exact round-trip');

  await page.getByRole('button', { name: '256 byte*', exact: true }).click();
  await expect(page.locator('.token-stats')).toContainText('259 vocabulary entries');
  await page.getByRole('button', { name: '1,024 BPE' }).click();
  await expect(page.locator('.token-stats')).toContainText('1,024 vocabulary entries');

  const next = page.getByRole('button', { name: 'Next merge' });
  await next.focus();
  await page.keyboard.press('Enter');
  await expect(page.locator('.merge-rank')).toContainText('Learned rank');
  const previous = page.getByRole('button', { name: 'Previous' });
  await previous.focus();
  await page.keyboard.press('Space');
  await expect(page.locator('.merge-rank')).toContainText('Starting state');

  await page.getByLabel('Filter by id, text, or decimal byte').fill('259');
  await expect(page.locator('tbody tr')).toHaveCount(1);

  const overflow = await page.evaluate(() => ({
    clientWidth: document.documentElement.clientWidth,
    scrollWidth: document.documentElement.scrollWidth,
  }));
  expect(overflow.scrollWidth).toBeLessThanOrEqual(overflow.clientWidth);
  await page.evaluate(() => window.scrollTo({ top: 0, behavior: 'instant' }));
  await page.screenshot({ path: testInfo.outputPath('tokenizer-lab.png') });
});
