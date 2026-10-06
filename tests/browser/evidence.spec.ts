import { expect, test } from '@playwright/test';

const manifest = {
  formatVersion: 1,
  exportedAt: '2026-10-06T12:00:00.000Z',
  database: { host: '127.0.0.1', port: 15436, name: 'birds_test' },
  sourceRows: 3,
  emittedRows: 3,
  skippedRows: 0,
  skippedReasons: {},
  malformedSectionsSkipped: 0,
  splits: {
    train: { documents: 1, bytes: 100 },
    validation: { documents: 1, bytes: 100 },
    test: { documents: 1, bytes: 100 },
  },
  sourceBytes: { wikipediaExtracts: 100, wikipediaSections: 100, fieldCraft: 0 },
  corpusBytes: 300,
  corpusSha256: 'a'.repeat(64),
  splitAlgorithm: 'sha256-bucket',
  splitVersion: 'split-v1',
  splitOverrides: [],
  templateVersion: 'corpus-v1',
  sources: {
    wikipediaExtracts: true,
    wikipediaSections: true,
    fieldCraftExported: true,
    fieldCraftIncludedInTraining: false,
  },
  gitRevision: null,
  warnings: [],
};

test('shows separate source-labeled search modes and careful no-hit language', async ({ page }) => {
  await page.route('**/api/corpus/manifest', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(manifest) }),
  );
  await page.route('**/api/checkpoints', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ checkpoints: [], activeCheckpointId: null, recoveryWarning: null }),
    }),
  );
  await page.route(/\/api\/evidence\/search/, (route) => {
    const query = new URL(route.request().url()).searchParams.get('q');
    const hit = {
      code: 'osprey',
      name: 'Osprey',
      sci: 'Pandion haliaetus',
      split: 'test',
      field: 'Wikipedia extract',
      snippet: 'The Osprey hunts over open ocean.',
    };
    const found = query === 'open ocean';
    const body = {
      query,
      page: 0,
      pageSize: 20,
      corpusSha256: manifest.corpusSha256,
      exportedAt: manifest.exportedAt,
      indexedDocuments: 3,
      exactPhrase: { total: found ? 1 : 0, hits: found ? [hit] : [] },
      normalizedTerms: { total: found ? 1 : 0, hits: found ? [hit] : [] },
      speciesDocuments: { total: 0, hits: [] },
    };
    return route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(body),
    });
  });
  await page.goto('/evidence');
  await expect(page.getByText('3 exported documents')).toBeVisible();
  await expect(page.getByText('only the train split updated the model')).toBeVisible();
  await page.getByLabel('Words or exact phrase').fill('open ocean');
  await page.getByRole('button', { name: 'Search evidence' }).click();
  await expect(page.getByTestId('exact-phrase-results')).toContainText('Wikipedia extract');
  await expect(page.getByTestId('exact-phrase-results').locator('.split')).toHaveText('test');
  await expect(page.getByTestId('normalized-term-results')).toContainText(
    'The Osprey hunts over open ocean.',
  );
  await expect(page.getByTestId('species-document-results')).toContainText('0 matches');
  await page.getByLabel('Words or exact phrase').fill('no such bird');
  await page.getByRole('button', { name: 'Search evidence' }).click();
  await expect(page.getByTestId('evidence-no-hits')).toContainText(
    'No matching evidence found with these searches',
  );
  await expect(page.getByTestId('evidence-no-hits')).toContainText(
    'does not establish that the claim is false',
  );
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow).toBe(0);
});
