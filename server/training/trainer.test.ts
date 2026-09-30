import { createHash } from 'node:crypto';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import {
  serializeCorpusRow,
  serializeStableJson,
  type CorpusManifest,
  type CorpusRow,
  type ModelConfig,
} from '../../src/lib/data/schemas';
import { encodeCheckpointWeights } from '../../src/lib/model/checkpoint';
import { trainingDocumentText } from '../../src/lib/tokenizer/corpus';
import { trainBpe } from '../../src/lib/tokenizer/bpe';
import { loadCheckpoint } from '../checkpoint-store';
import { DeterministicBatchStream, StreamedTokenDataset } from './token-dataset';
import { trainReadyCheckpoint, TrainingCancelledError } from './trainer';

const temporaryDirectories: string[] = [];

afterEach(async () => {
  await Promise.all(temporaryDirectories.splice(0).map((path) => rm(path, { recursive: true })));
});

function row(code: string, split: CorpusRow['split'], extract: string): CorpusRow {
  return {
    code,
    name: `${code} bird`,
    sci: `${code} scientific`,
    order: 'Testiformes',
    family: 'Testidae',
    extract,
    sections: [],
    field_craft: null,
    tags: [],
    split,
  };
}

async function fixture(): Promise<{
  root: string;
  corpusPath: string;
  manifestPath: string;
  tokenizerPath: string;
  cacheRoot: string;
}> {
  const root = await mkdtemp(join(tmpdir(), 'birds-llm-trainer-'));
  temporaryDirectories.push(root);
  const rows = [
    row('train-a', 'train', 'red bird red bird red bird'),
    row('train-b', 'train', 'blue bird blue bird blue bird'),
    row('valid-a', 'validation', 'red bird blue bird'),
    row('test-a', 'test', 'secret test-only wording'),
  ];
  const corpusText = `${rows.map(serializeCorpusRow).join('\n')}\n`;
  const corpusSha256 = createHash('sha256').update(corpusText).digest('hex');
  const corpusPath = join(root, 'corpus.jsonl');
  const manifestPath = join(root, 'manifest.json');
  const tokenizerPath = join(root, 'tokenizer.json');
  const cacheRoot = join(root, 'cache');
  await writeFile(corpusPath, corpusText);
  const manifest: CorpusManifest = {
    formatVersion: 1,
    exportedAt: '2026-09-29T12:00:00.000Z',
    database: { host: '127.0.0.1', port: 15436, name: 'birds_test' },
    sourceRows: rows.length,
    emittedRows: rows.length,
    skippedRows: 0,
    skippedReasons: {},
    malformedSectionsSkipped: 0,
    splits: {
      train: { documents: 2, bytes: 1 },
      validation: { documents: 1, bytes: 1 },
      test: { documents: 1, bytes: 1 },
    },
    sourceBytes: { wikipediaExtracts: 1, wikipediaSections: 0, fieldCraft: 0 },
    corpusBytes: Buffer.byteLength(corpusText),
    corpusSha256,
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
  await writeFile(manifestPath, serializeStableJson(manifest));
  const tokenizer = trainBpe(
    rows.filter(({ split }) => split === 'train').map(trainingDocumentText),
    { targetSize: 259, corpusSha256 },
  );
  await writeFile(tokenizerPath, serializeStableJson(tokenizer));
  return { root, corpusPath, manifestPath, tokenizerPath, cacheRoot };
}

const tinyModel: ModelConfig = {
  formatVersion: 1,
  vocabSize: 259,
  contextLength: 4,
  dModel: 8,
  nLayers: 1,
  nHeads: 2,
  dHead: 4,
  dMlp: 16,
  tiedEmbeddings: true,
  useBias: true,
  layerNormEpsilon: 1e-5,
  gelu: 'tanh-approximation',
};

describe('streamed token dataset and Node trainer', () => {
  it('indexes train/validation tokens without copying test tokens and resumes data order', async () => {
    const paths = await fixture();
    const dataset = await StreamedTokenDataset.open(paths);
    try {
      expect(dataset.index.sourceDocumentCounts).toEqual({ train: 2, validation: 1, test: 1 });
      expect(dataset.index.documents.map(({ split }) => split)).toEqual([
        'train',
        'train',
        'validation',
      ]);
      const windows = dataset.windows('train', 4);
      const first = new DeterministicBatchStream(dataset, windows, 2, 4, 123);
      await first.next();
      const state = first.snapshot();
      const expectedNext = await first.next();
      const resumed = new DeterministicBatchStream(dataset, windows, 2, 4, 123, state);
      expect(await resumed.next()).toEqual(expectedNext);
    } finally {
      await dataset.close();
    }
  });

  it('resumes to exactly the same weights and trainer state as an uninterrupted run', async () => {
    const paths = await fixture();
    const shared = {
      ...paths,
      totalSteps: 6,
      checkpointEvery: 3,
      logEvery: 3,
      validateEvery: 6,
      validationBatches: 1,
      batchSize: 2,
      model: tinyModel,
      sourceGitRevision: 'c'.repeat(40),
    } as const;
    const uninterrupted = await trainReadyCheckpoint({
      ...shared,
      outputRoot: join(paths.root, 'uninterrupted'),
    });
    const partial = await trainReadyCheckpoint({
      ...shared,
      outputRoot: join(paths.root, 'partial'),
      stopAfterStep: 3,
    });
    const resumed = await trainReadyCheckpoint({
      ...shared,
      outputRoot: join(paths.root, 'resumed'),
      resumeFrom: partial.checkpointPath,
    });
    const [uninterruptedLoaded, resumedLoaded] = await Promise.all([
      loadCheckpoint(uninterrupted.checkpointPath),
      loadCheckpoint(resumed.checkpointPath),
    ]);
    expect(encodeCheckpointWeights(tinyModel, resumedLoaded.parameters).bytes).toEqual(
      encodeCheckpointWeights(tinyModel, uninterruptedLoaded.parameters).bytes,
    );
    expect(resumedLoaded.resumeState).toEqual(uninterruptedLoaded.resumeState);
    expect(
      resumedLoaded.trainingLog.entries.map(({ elapsedMs: _elapsed, ...entry }) => entry),
    ).toEqual(
      uninterruptedLoaded.trainingLog.entries.map(({ elapsedMs: _elapsed, ...entry }) => entry),
    );
    expect(JSON.parse(await readFile(join(resumed.checkpointPath, 'config.json'), 'utf8'))).toEqual(
      JSON.parse(await readFile(join(uninterrupted.checkpointPath, 'config.json'), 'utf8')),
    );
  });

  it('cooperatively cancels without publishing a checkpoint', async () => {
    const paths = await fixture();
    const controller = new AbortController();
    const outputRoot = join(paths.root, 'cancelled');
    await expect(
      trainReadyCheckpoint({
        ...paths,
        outputRoot,
        totalSteps: 6,
        checkpointEvery: 3,
        logEvery: 3,
        validateEvery: 3,
        validationBatches: 1,
        batchSize: 2,
        model: tinyModel,
        sourceGitRevision: 'c'.repeat(40),
        signal: controller.signal,
        onProgress() {
          controller.abort();
        },
      }),
    ).rejects.toBeInstanceOf(TrainingCancelledError);
    await expect(readFile(join(outputRoot, 'ready-v1', 'config.json'))).rejects.toMatchObject({
      code: 'ENOENT',
    });
  });
});
