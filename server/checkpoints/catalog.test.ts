import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import type { CheckpointConfig, TrainingLogArtifact } from '../../src/lib/data/schemas';
import { DEFAULT_MODEL_CONFIG } from '../../src/lib/model/config';
import { initializeParameters } from '../../src/lib/model/parameters';
import { writeCheckpoint } from '../checkpoint-store';
import { CheckpointCatalog } from './catalog';

const temporaryDirectories: string[] = [];

afterEach(async () => {
  await Promise.all(
    temporaryDirectories.splice(0).map((path) => rm(path, { recursive: true, force: true })),
  );
});

async function fixture(): Promise<{ root: string; checkpoints: string; selection: string }> {
  const root = await mkdtemp(join(tmpdir(), 'birds-llm-catalog-'));
  temporaryDirectories.push(root);
  return {
    root,
    checkpoints: join(root, 'checkpoints'),
    selection: join(root, 'data', 'checkpoint-selection.json'),
  };
}

const config: CheckpointConfig = {
  formatVersion: 1,
  model: DEFAULT_MODEL_CONFIG,
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
  seed: 1,
  corpusSha256: 'a'.repeat(64),
  tokenizerSha256: 'b'.repeat(64),
  trainingStep: 2,
  sourceGitRevision: null,
};

const log: TrainingLogArtifact = {
  formatVersion: 1,
  entries: [
    {
      step: 2,
      split: 'train',
      predictionCount: 512,
      meanLoss: 4.2,
      perplexity: Math.exp(4.2),
      learningRate: 0,
      gradientNorm: 1,
      elapsedMs: 100,
      sample: null,
    },
    {
      step: 2,
      split: 'validation',
      predictionCount: 256,
      meanLoss: 4.3,
      perplexity: Math.exp(4.3),
      learningRate: 0,
      gradientNorm: 0,
      elapsedMs: 120,
      sample: 'Common name: Test bird',
    },
  ],
};

describe('checkpoint catalog', () => {
  it('lists only validated checkpoints and persists selection across restart', async () => {
    const paths = await fixture();
    await writeCheckpoint(
      join(paths.checkpoints, 'valid-one'),
      config,
      initializeParameters(DEFAULT_MODEL_CONFIG, 1),
      log,
    );
    await mkdir(join(paths.checkpoints, 'corrupt-one'), { recursive: true });
    await writeFile(join(paths.checkpoints, 'corrupt-one', 'config.json'), '{}');
    await mkdir(join(paths.checkpoints, '.training-hidden'), { recursive: true });

    const catalog = new CheckpointCatalog(paths.checkpoints, paths.selection);
    await catalog.initialize();
    const listed = await catalog.list();
    expect(listed.checkpoints).toHaveLength(1);
    expect(listed.checkpoints[0]).toMatchObject({
      id: 'valid-one',
      finalTrainLoss: 4.2,
      finalValidationLoss: 4.3,
    });
    expect(listed.recoveryWarning).toContain('1 invalid checkpoint');

    await catalog.select('valid-one');
    const restarted = new CheckpointCatalog(paths.checkpoints, paths.selection);
    await restarted.initialize();
    expect((await restarted.list()).activeCheckpointId).toBe('valid-one');
    expect(JSON.parse(await readFile(paths.selection, 'utf8'))).toEqual({
      formatVersion: 1,
      activeCheckpointId: 'valid-one',
    });
  });

  it('rejects invalid, missing, and corrupt checkpoint selections', async () => {
    const paths = await fixture();
    await mkdir(paths.checkpoints, { recursive: true });
    const catalog = new CheckpointCatalog(paths.checkpoints, paths.selection);
    await catalog.initialize();
    await expect(catalog.select('../outside')).rejects.toThrow('Invalid checkpoint id');
    await expect(catalog.select('missing')).rejects.toThrow('not found or failed validation');
  });
});
