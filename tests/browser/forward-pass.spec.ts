import { expect, test, type Page } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import type { CheckpointInspectionBundle } from '../../src/lib/data/api-types';
import type { CheckpointConfig, TokenizerArtifact } from '../../src/lib/data/schemas';
import { encodeCheckpointWeights } from '../../src/lib/model/checkpoint';
import { initializeParameters } from '../../src/lib/model/parameters';
import { encodeText } from '../../src/lib/tokenizer/bpe';
import { generateReplay } from '../../src/lib/generation/replay';

const config: CheckpointConfig = {
  formatVersion: 1,
  model: {
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
  },
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

async function mockInspection(page: Page): Promise<TokenizerArtifact> {
  const tokenizer = JSON.parse(
    await readFile(join(process.cwd(), 'src/assets/tokenizer-1024.json'), 'utf8'),
  ) as TokenizerArtifact;
  const encoded = encodeCheckpointWeights(config.model, initializeParameters(config.model, 123));
  const bundle: CheckpointInspectionBundle = {
    checkpointId: 'tiny-fixture',
    config,
    tokenizer,
    weightIndex: encoded.index,
    weightsBase64: Buffer.from(encoded.bytes).toString('base64'),
  };
  const summary = {
    id: 'tiny-fixture',
    trainingStep: 2,
    totalSteps: 2,
    corpusSha256: config.corpusSha256,
    tokenizerSha256: config.tokenizerSha256,
    sourceGitRevision: null,
    finalTrainLoss: 5,
    finalValidationLoss: 5,
    modifiedAt: '2026-10-05T00:00:00.000Z',
  };
  await page.route('**/api/**', async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === '/api/checkpoints') {
      await route.fulfill({
        json: { checkpoints: [summary], activeCheckpointId: 'tiny-fixture', recoveryWarning: null },
      });
    } else if (path === '/api/checkpoints/active/inspect') {
      await route.fulfill({ json: bundle });
    } else {
      await route.fulfill({ status: 404, json: { error: 'Unexpected API route.' } });
    }
  });
  return tokenizer;
}

test('steps reversibly through real selected cells with keyboard at laptop widths', async ({
  page,
}) => {
  await mockInspection(page);

  await page.goto('/forward-pass');
  await expect(page.getByRole('heading', { name: 'Token embeddings' })).toBeVisible();
  const grid = page.getByRole('region', { name: 'embedding.token values' });
  await expect(grid).toBeVisible();
  const cell = grid.getByRole('button', { name: /Position 0, column 0, value/ });
  await cell.focus();
  await page.keyboard.press('Enter');
  await expect(cell).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByText('Full Float32 value:')).toBeVisible();
  const before = await page.getByRole('heading', { name: /Selected cell/ }).textContent();
  await page.getByRole('button', { name: 'Next →' }).click();
  await expect(page.getByRole('heading', { name: 'Position embeddings' })).toBeVisible();
  await page.getByRole('button', { name: '← Previous' }).click();
  await expect(page.getByRole('heading', { name: /Selected cell/ })).toHaveText(before ?? '');
  await page.getByLabel('Jump to operation').selectOption('19');
  await expect(page.getByRole('heading', { name: 'Layer 0 · MLP residual' })).toBeVisible();
  const lastStage = await page
    .getByLabel('Jump to operation')
    .locator('option')
    .last()
    .getAttribute('value');
  if (!lastStage) throw new Error('Missing final forward stage.');
  await page.getByLabel('Jump to operation').selectOption(lastStage);
  await expect(page.getByRole('heading', { name: 'Top five next-token logits' })).toBeVisible();
  await expect(
    page.getByRole('group', { name: 'Top five logits' }).getByRole('button'),
  ).toHaveCount(5);
  const overflow = await page.evaluate(() => ({
    clientWidth: document.documentElement.clientWidth,
    scrollWidth: document.documentElement.scrollWidth,
  }));
  expect(overflow.scrollWidth).toBeLessThanOrEqual(overflow.clientWidth);
});

test('compares each attention head with the average of probabilities', async ({ page }) => {
  await mockInspection(page);
  await page.goto('/attention');
  await expect(page.getByRole('heading', { name: 'Layer 0, head 0' })).toBeVisible();
  const grid = page.getByRole('region', { name: 'blocks.0.attn.probabilities values' });
  await expect(grid).toBeVisible();
  await page.getByLabel('View').selectOption('average');
  await expect(page.getByRole('heading', { name: 'Average of head probabilities' })).toBeVisible();
  await expect(page.getByText('sum(head probability) / head count')).toBeVisible();
  await expect(page.getByText('head 0', { exact: true })).toBeVisible();
  await expect(page.getByText('head 1', { exact: true })).toBeVisible();
  await page.getByLabel('Layer').selectOption('1');
  await expect(
    page.getByRole('region', { name: 'blocks.1.attn.averageProbabilities values' }),
  ).toBeVisible();
  const cell = page
    .getByRole('region', { name: 'blocks.1.attn.averageProbabilities values' })
    .getByRole('button', { name: /Position 1, column 0, value/ });
  await cell.focus();
  await page.keyboard.press('Enter');
  await expect(cell).toHaveAttribute('aria-pressed', 'true');
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow).toBe(0);
});

test('inspects embedding cells, deterministic PCA maps, and cosine neighbours', async ({
  page,
}) => {
  await mockInspection(page);
  await page.goto('/embeddings');
  await expect(page.getByRole('heading', { name: 'Choose a token' })).toBeVisible();
  await expect(page.getByText('Checkpoint tiny-fixture · step 2', { exact: true })).toBeVisible();
  await page.getByLabel('Search tokens').fill('97');
  await expect(page.getByText(/matching tokens/)).toBeVisible();
  await page.getByLabel('Token list').selectOption('97');
  const channel = page
    .getByRole('group', { name: 'Trained embedding channels' })
    .getByRole('button', { name: /Channel 1, trained value/ });
  await channel.focus();
  await page.keyboard.press('Enter');
  await expect(channel).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByText('The actual math · channel 1')).toBeVisible();
  await expect(page.getByRole('img', { name: /Initial · reconstructed PCA map/ })).toBeVisible();
  await expect(page.getByRole('img', { name: /Trained · stored PCA map/ })).toBeVisible();
  await expect(
    page.getByRole('heading', { name: 'Nearest rows by cosine similarity' }),
  ).toBeVisible();
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow).toBe(0);
});

test('matches Node sampling numbers and IDs in the browser and replays a sliding window', async ({
  page,
}) => {
  const tokenizer = await mockInspection(page);
  const prompt = Array.from({ length: 90 }, (_, index) =>
    String.fromCharCode(33 + (index % 90)),
  ).join('');
  const promptIds = encodeText(prompt, tokenizer, { bos: true });
  expect(promptIds.length).toBeGreaterThan(config.model.contextLength);
  const expected = generateReplay(
    promptIds,
    42,
    { temperature: 1, topK: 0, topP: 1 },
    4,
    initializeParameters(config.model, 123),
    config.model,
  );

  await page.goto('/generation');
  await expect(page.getByRole('heading', { name: 'Inspect a next-token draw' })).toBeVisible();
  await page.getByLabel('Prompt').fill(prompt);
  await page.getByLabel('Tokens to generate').fill('4');
  await page.getByRole('button', { name: 'Generate and save replay' }).click();
  await expect(page.getByTestId('generated-ids')).toHaveText(expected.generatedIds.join(', '));
  await expect(page.getByTestId('retained-mass')).toHaveText(
    expected.steps[0].distribution.retainedMass.toPrecision(8),
  );
  await expect(page.getByTestId('random-draw')).toHaveText(
    expected.steps[0].draw.randomNumber.toPrecision(8),
  );
  await expect(page.getByTestId('selected-interval')).toHaveText(
    `[${expected.steps[0].draw.intervalStart.toPrecision(8)}, ${expected.steps[0].draw.intervalEnd.toPrecision(8)})`,
  );
  const firstId = expected.steps[0].distribution.rankOrder[0];
  const firstRow = page
    .getByRole('table', { name: 'Vocabulary probabilities' })
    .locator(`tr[data-token-id="${firstId}"]`);
  await expect(firstRow).toHaveAttribute(
    'data-before',
    String(expected.steps[0].distribution.beforeFilters[firstId]),
  );
  await expect(firstRow).toHaveAttribute(
    'data-after',
    String(expected.steps[0].distribution.probabilities[firstId]),
  );
  await expect(firstRow).toHaveAttribute(
    'data-interval-end',
    String(expected.steps[0].distribution.cumulative[firstId]),
  );
  const firstPageIds = expected.steps[0].distribution.rankOrder.slice(0, 25);
  const expectedRemainder =
    1 - firstPageIds.reduce((sum, id) => sum + expected.steps[0].distribution.probabilities[id], 0);
  await expect(page.getByTestId('remainder-after')).toHaveText(expectedRemainder.toPrecision(8));
  await page.getByRole('button', { name: 'Next page →' }).click();
  await expect(page.getByText('page 2 of 41')).toBeVisible();
  await page.getByLabel('Search vocabulary').fill(String(expected.generatedIds[0]));
  await expect(page.getByText(/matching tokens/)).toBeVisible();
  await page.getByRole('button', { name: 'Next decision →' }).click();
  await expect(page.getByRole('heading', { name: 'Decision 2 of 4' })).toBeVisible();
  await expect(
    page.getByText(`Dropped leading tokens (${expected.steps[1].dropped.length})`),
  ).toBeVisible();
  await page.getByRole('button', { name: '← Previous decision' }).click();
  await expect(page.getByTestId('random-draw')).toHaveText(
    expected.steps[0].draw.randomNumber.toPrecision(8),
  );
  await page.getByLabel('Temperature').fill('0.7');
  await page.getByLabel('Top-k').fill('3');
  await page.getByLabel('Top-p').fill('0.5');
  await page.getByRole('button', { name: 'Inspect next token' }).click();
  const filtered = generateReplay(
    promptIds,
    42,
    { temperature: 0.7, topK: 3, topP: 0.5 },
    1,
    initializeParameters(config.model, 123),
    config.model,
  );
  await expect(page.getByTestId('generated-ids')).toHaveText(String(filtered.generatedIds[0]));
  await expect(page.getByTestId('retained-mass')).toHaveText(
    filtered.steps[0].distribution.retainedMass.toPrecision(8),
  );
  await expect(page.getByTestId('removed-mass')).toHaveText(
    filtered.steps[0].distribution.removedMass.toPrecision(8),
  );
  await expect(page.getByTestId('selected-interval')).toHaveText(
    `[${filtered.steps[0].draw.intervalStart.toPrecision(8)}, ${filtered.steps[0].draw.intervalEnd.toPrecision(8)})`,
  );
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow).toBe(0);
});
