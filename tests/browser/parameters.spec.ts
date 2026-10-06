import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { encodeCheckpointWeights } from '../../src/lib/model/checkpoint';
import { initializeParameters } from '../../src/lib/model/parameters';
import type { CheckpointConfig, TokenizerArtifact } from '../../src/lib/data/schemas';

test('browses every stored tensor in bounded tiles and follows a trace link', async ({ page }) => {
  const model: CheckpointConfig['model'] = {
    formatVersion: 1,
    vocabSize: 1024,
    contextLength: 64,
    dModel: 4,
    nLayers: 2,
    nHeads: 2,
    dHead: 2,
    dMlp: 6,
    tiedEmbeddings: true,
    useBias: true,
    layerNormEpsilon: 1e-5,
    gelu: 'tanh-approximation',
  };
  const config: CheckpointConfig = {
    formatVersion: 1,
    model,
    optimizer: {
      name: 'adamw',
      beta1: 0.9,
      beta2: 0.95,
      epsilon: 1e-8,
      weightDecay: 0.1,
      gradientClipNorm: 1,
      learningRate: 0.003,
      warmupSteps: 1,
      totalSteps: 2,
    },
    seed: 123,
    corpusSha256: 'a'.repeat(64),
    tokenizerSha256: 'b'.repeat(64),
    trainingStep: 2,
    sourceGitRevision: null,
  };
  const tokenizer = JSON.parse(
    await readFile(join(process.cwd(), 'src/assets/tokenizer-1024.json'), 'utf8'),
  ) as TokenizerArtifact;
  const registry = initializeParameters(model, 123);
  const encoded = encodeCheckpointWeights(model, registry);
  const bundle = {
    checkpointId: 'tiny-fixture',
    config,
    tokenizer,
    weightIndex: encoded.index,
    weightsBase64: Buffer.from(encoded.bytes).toString('base64'),
  };
  await page.route('**/api/**', async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === '/api/checkpoints')
      return route.fulfill({
        json: {
          checkpoints: [
            {
              id: 'tiny-fixture',
              trainingStep: 2,
              totalSteps: 2,
              corpusSha256: config.corpusSha256,
              tokenizerSha256: config.tokenizerSha256,
              sourceGitRevision: null,
              finalTrainLoss: 5,
              finalValidationLoss: 5,
              modifiedAt: '2026-10-06T00:00:00.000Z',
            },
          ],
          activeCheckpointId: 'tiny-fixture',
          recoveryWarning: null,
        },
      });
    if (path === '/api/checkpoints/active/inspect') return route.fulfill({ json: bundle });
    return route.fulfill({ status: 404, json: { error: 'Unexpected API route.' } });
  });
  await page.goto('/parameters');
  await expect(page.getByTestId('parameter-reconciliation')).toContainText(
    registry.elementCount.toLocaleString(),
  );
  const tree = page.getByRole('navigation', { name: 'Checkpoint parameters' });
  await expect(tree.getByRole('button')).toHaveCount(registry.size);
  await page.getByRole('button', { name: 'Next rows' }).click();
  await expect(page.getByTestId('parameter-cell-value')).toContainText('[32, 0]');
  await tree.getByRole('button', { name: /blocks\.0\.attn\.q\.weight/ }).click();
  await expect(page.getByRole('region', { name: 'Selected parameter' })).toContainText(
    'blocks.0.attn.q.weight',
  );
  await expect(
    page.getByRole('button', { name: /Heatmap window for blocks\.0\.attn\.q\.weight/ }),
  ).toBeVisible();
  await expect(page.getByTestId('parameter-cell-value')).toContainText('[0, 0]');
  await page.getByRole('spinbutton', { name: 'Column' }).fill('2');
  await page.getByRole('spinbutton', { name: 'Column' }).press('Tab');
  await expect(page.getByTestId('parameter-cell-value')).toContainText('[0, 2]');
  await page
    .getByRole('link', { name: /See this parameter in the blocks\.0\.attn\.q forward trace/ })
    .click();
  await expect(page).toHaveURL(/\/forward-pass\?stage=blocks\.0\.attn\.q/);
  await expect(page.getByRole('heading', { name: 'Layer 0 · query projection' })).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
    ),
  ).toBe(0);
});
