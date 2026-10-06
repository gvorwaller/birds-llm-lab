import { expect, test } from '@playwright/test';

test('runs a real worker while the page stays responsive and supports control and checkpoint', async ({
  page,
}) => {
  await page.goto('/training');
  await page.getByLabel('Preset').selectOption('protocol');
  await expect(page.getByText('Three five-token synthetic sequences')).toBeVisible();
  const heartbeatBefore = Number(await page.getByTestId('ui-heartbeat').textContent());
  await page.getByRole('button', { name: 'Start fixture training' }).click();
  await expect(page.getByTestId('worker-state')).toHaveText('running');
  await expect
    .poll(async () => Number(await page.getByTestId('ui-heartbeat').textContent()))
    .toBeGreaterThan(heartbeatBefore);
  await page.getByRole('button', { name: 'Pause', exact: true }).click();
  await expect(page.getByTestId('worker-state')).toHaveText('paused');
  const pausedStep = Number((await page.getByTestId('worker-step').textContent())?.split(' / ')[0]);
  await page.getByRole('button', { name: 'One batch' }).click();
  await expect(page.getByTestId('worker-step')).toHaveText(`${pausedStep + 1} / 1000`);
  await page.getByRole('button', { name: 'Checkpoint', exact: true }).click();
  await expect(page.getByTestId('checkpoint-summary')).toContainText(
    `Checkpoint at batch ${pausedStep + 1}`,
  );
  await page.getByRole('button', { name: 'Restore saved checkpoint' }).click();
  await expect(page.getByTestId('worker-state')).toHaveText('paused');
  await expect(page.getByTestId('worker-step')).toHaveText(`${pausedStep + 1} / 1000`);
  await page.getByRole('button', { name: 'Resume', exact: true }).click();
  await expect(page.getByTestId('worker-state')).toHaveText('running');
  await page.getByRole('button', { name: 'Cancel', exact: true }).click();
  await expect(page.getByTestId('worker-state')).toHaveText('cancelled');
  const cancelledStep = await page.getByTestId('worker-step').textContent();
  await page.waitForTimeout(100);
  await expect(page.getByTestId('worker-step')).toHaveText(cancelledStep ?? '');
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow).toBe(0);
});

test('plots small-preset train and validation metrics, samples, histogram, and matrix', async ({
  page,
}) => {
  await page.goto('/training');
  await expect(page.getByLabel('Preset')).toHaveValue('small');
  await expect(page.getByText('12 train lines · 4 validation lines')).toBeVisible();
  await page.getByRole('button', { name: 'Start small preset' }).click();
  await expect(page.getByTestId('worker-state')).toHaveText('running');
  await expect(page.getByTestId('validation-loss')).not.toHaveText('Not configured');
  await expect(
    page.getByRole('img', { name: 'Train and validation loss by training step' }),
  ).toBeVisible();
  await expect(page.getByRole('img', { name: 'Learning rate by training step' })).toBeVisible();
  await expect(
    page.getByRole('img', { name: 'Gradient norm before clipping by training step' }),
  ).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Seeded sample generations' })).toBeVisible();
  await expect(page.locator('.samples li').first()).toContainText('Step');
  await expect(
    page.getByRole('img', { name: /Histogram of blocks.0.attn.q.weight/ }),
  ).toBeVisible();
  await expect(
    page.getByRole('grid', { name: 'blocks.0.attn.q.weight values' }).getByRole('gridcell'),
  ).toHaveCount(144);
  await page.getByRole('button', { name: 'Cancel', exact: true }).click();
  await expect(page.getByTestId('worker-state')).toHaveText('cancelled');
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow).toBe(0);
});
