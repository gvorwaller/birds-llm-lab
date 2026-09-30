import { expect, test, type Page } from '@playwright/test';

const checkpoint = {
  id: 'ready-v1',
  trainingStep: 2_000,
  totalSteps: 2_000,
  corpusSha256: 'a'.repeat(64),
  tokenizerSha256: 'b'.repeat(64),
  sourceGitRevision: 'c'.repeat(40),
  finalTrainLoss: 4.1295,
  finalValidationLoss: 4.1563,
  modifiedAt: '2026-09-29T19:00:00.000Z',
};

function trainingJob(status: 'running' | 'cancelling') {
  return {
    id: 'job-1',
    kind: 'model-training',
    preset: 'quick',
    status,
    createdAt: '2026-09-29T20:00:00.000Z',
    updatedAt: '2026-09-29T20:00:05.000Z',
    progress: {
      step: 25,
      totalSteps: 100,
      trainLoss: 5.25,
      validationLoss: 5.4,
      learningRate: 0.002,
      elapsedMs: 5_000,
      estimatedRemainingMs: 15_000,
      latestSample: 'Common name: Test bird\nOrder: Passeriformes',
    },
    logs: [],
    checkpointId: null,
    error: null,
  };
}

async function mockApi(page: Page): Promise<void> {
  await page.route('**/api/**', async (route) => {
    const request = route.request();
    const pathname = new URL(request.url()).pathname;
    if (pathname === '/api/status') {
      await route.fulfill({
        json: {
          status: 'ok',
          version: 1,
          endpoint: '127.0.0.1:5301',
          corpusAvailable: false,
          activeExportJobId: null,
          latestExportJobId: null,
          activeTrainingJobId: null,
          latestTrainingJobId: null,
          recoveryWarning: null,
        },
      });
    } else if (pathname === '/api/corpus/manifest') {
      await route.fulfill({ status: 404, json: { error: 'No corpus.' } });
    } else if (pathname === '/api/checkpoints') {
      await route.fulfill({
        json: { checkpoints: [checkpoint], activeCheckpointId: null, recoveryWarning: null },
      });
    } else if (pathname === '/api/checkpoints/ready-v1/select') {
      await route.fulfill({
        json: {
          checkpoints: [checkpoint],
          activeCheckpointId: 'ready-v1',
          recoveryWarning: null,
        },
      });
    } else if (pathname === '/api/training/jobs') {
      await route.fulfill({ status: 202, json: trainingJob('running') });
    } else if (pathname === '/api/jobs/job-1/cancel') {
      await route.fulfill({ json: trainingJob('cancelling') });
    } else if (pathname === '/api/jobs/job-1') {
      await route.fulfill({ json: trainingJob('running') });
    } else {
      await route.fulfill({ status: 404, json: { error: 'Unexpected test route.' } });
    }
  });
}

test('starts, monitors, cancels, and globally selects through the Data workbench', async ({
  page,
}, testInfo) => {
  await mockApi(page);
  await page.goto('/data');
  await page.evaluate(() => {
    const meta = document.querySelector<HTMLMetaElement>('meta[name="birds-llm-lab-csrf"]');
    if (meta) meta.content = 'browser-test-token';
  });

  await expect(page.getByRole('heading', { name: 'Training & checkpoints' })).toBeVisible();
  await expect(page.getByText('ready-v1', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Use checkpoint' }).click();
  await expect(page.getByLabel('Active checkpoint')).toHaveValue('ready-v1');

  await page.getByRole('button', { name: 'Train new checkpoint' }).click();
  await expect(page.getByText('25 / 100')).toBeVisible();
  await expect(page.getByText('5.4000')).toBeVisible();
  await expect(page.getByText('Common name: Test bird')).toBeVisible();
  await page.getByRole('button', { name: 'Cancel safely' }).click();
  await expect(page.getByRole('button', { name: 'Cancelling safely…' })).toBeDisabled();

  const overflow = await page.evaluate(() => ({
    clientWidth: document.documentElement.clientWidth,
    scrollWidth: document.documentElement.scrollWidth,
  }));
  expect(overflow.scrollWidth).toBeLessThanOrEqual(overflow.clientWidth);
  await page.screenshot({ path: testInfo.outputPath('data-training.png'), fullPage: true });
});
