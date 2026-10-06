import { expect, test } from '@playwright/test';

test('runs a real worker while the page stays responsive and supports control and checkpoint', async ({
  page,
}) => {
  await page.goto('/training');
  await expect(page.getByText('synthetic token set')).toBeVisible();
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
