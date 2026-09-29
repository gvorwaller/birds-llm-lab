import { mkdtemp, mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import type { CheckpointConfig, TrainingLogArtifact } from '../src/lib/data/schemas';
import { DEFAULT_MODEL_CONFIG } from '../src/lib/model/config';
import { encodeCheckpointWeights } from '../src/lib/model/checkpoint';
import { SeededRandom } from '../src/lib/math/rng';
import { AdamWOptimizer } from '../src/lib/model/optimizer';
import { initializeParameters } from '../src/lib/model/parameters';
import { loadCheckpoint, writeCheckpoint } from './checkpoint-store';
import type { TrainerResumeState } from './trainer-state';

const temporaryDirectories: string[] = [];

async function temporaryDirectory(): Promise<string> {
  const directory = await mkdtemp(join(tmpdir(), 'birds-llm-checkpoint-'));
  temporaryDirectories.push(directory);
  return directory;
}

afterEach(async () => {
  await Promise.all(temporaryDirectories.splice(0).map((path) => rm(path, { recursive: true })));
});

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
    learningRate: 3e-4,
    warmupSteps: 2,
    totalSteps: 10,
  },
  seed: 0x5eed,
  corpusSha256: 'a'.repeat(64),
  tokenizerSha256: 'b'.repeat(64),
  trainingStep: 3,
  sourceGitRevision: 'c'.repeat(40),
};

const trainingLog: TrainingLogArtifact = {
  formatVersion: 1,
  entries: [
    {
      step: 3,
      split: 'train',
      predictionCount: 512,
      meanLoss: 5.5,
      perplexity: Math.exp(5.5),
      learningRate: 3e-4,
      gradientNorm: 0.8,
      elapsedMs: 1_234,
      sample: 'Birds are',
    },
  ],
};

function resumeState(): TrainerResumeState {
  const parameters = initializeParameters(DEFAULT_MODEL_CONFIG, config.seed);
  const optimizer = new AdamWOptimizer(config.optimizer);
  for (let step = 0; step < config.trainingStep; step += 1) {
    for (const parameter of parameters) parameter.gradient.data.fill((step + 1) * 1e-4);
    optimizer.step(parameters, config.optimizer.learningRate);
  }
  return {
    optimizer: optimizer.snapshot(),
    samplingRandom: new SeededRandom(123).snapshot(),
    dataOrder: { epoch: 2, cursor: 17 },
  };
}

describe('checkpoint directory store', () => {
  it('writes to a validated temporary sibling and atomically installs a round-trippable checkpoint', async () => {
    const parent = await temporaryDirectory();
    const destination = join(parent, 'step-000003');
    const parameters = initializeParameters(DEFAULT_MODEL_CONFIG, config.seed);
    const written = await writeCheckpoint(destination, config, parameters, trainingLog);
    expect(await readdir(parent)).toEqual(['step-000003']);
    expect((await readdir(destination)).sort()).toEqual([
      'config.json',
      'training-log.json',
      'weights.bin',
      'weights.index.json',
    ]);
    const loaded = await loadCheckpoint(destination);
    const expectedBytes = encodeCheckpointWeights(DEFAULT_MODEL_CONFIG, parameters).bytes;
    expect(encodeCheckpointWeights(DEFAULT_MODEL_CONFIG, written.parameters).bytes).toEqual(
      expectedBytes,
    );
    expect(encodeCheckpointWeights(DEFAULT_MODEL_CONFIG, loaded.parameters).bytes).toEqual(
      expectedBytes,
    );
    expect(loaded.config).toEqual(config);
    expect(loaded.trainingLog).toEqual(trainingLog);
  });

  it('refuses to overwrite an existing path', async () => {
    const parent = await temporaryDirectory();
    const destination = join(parent, 'existing');
    await mkdir(destination);
    await writeFile(join(destination, 'keep.txt'), 'keep');
    await expect(
      writeCheckpoint(
        destination,
        config,
        initializeParameters(DEFAULT_MODEL_CONFIG, config.seed),
        trainingLog,
      ),
    ).rejects.toThrow('Refusing to replace');
    expect(await readFile(join(destination, 'keep.txt'), 'utf8')).toBe('keep');
  });

  it('round-trips resumable optimizer, RNG, and data-order state', async () => {
    const parent = await temporaryDirectory();
    const destination = join(parent, 'resumable');
    const state = resumeState();
    const loaded = await writeCheckpoint(
      destination,
      config,
      initializeParameters(DEFAULT_MODEL_CONFIG, config.seed),
      trainingLog,
      state,
    );
    expect((await readdir(destination)).sort()).toEqual([
      'config.json',
      'trainer-state.json',
      'training-log.json',
      'weights.bin',
      'weights.index.json',
    ]);
    expect(loaded.resumeState).toEqual(state);
  });

  it('rejects resumable state whose optimizer step disagrees with the checkpoint', async () => {
    const parent = await temporaryDirectory();
    const destination = join(parent, 'bad-resume-step');
    await writeCheckpoint(
      destination,
      config,
      initializeParameters(DEFAULT_MODEL_CONFIG, config.seed),
      trainingLog,
      resumeState(),
    );
    const statePath = join(destination, 'trainer-state.json');
    const state = JSON.parse(await readFile(statePath, 'utf8')) as {
      optimizer: { step: number };
    };
    state.optimizer.step += 1;
    await writeFile(statePath, JSON.stringify(state));
    await expect(loadCheckpoint(destination)).rejects.toThrow('does not match checkpoint step');
  });

  it('leaves no destination or temporary directory when validation fails', async () => {
    const parent = await temporaryDirectory();
    const destination = join(parent, 'invalid');
    const invalid = { ...config, formatVersion: 2 } as unknown as CheckpointConfig;
    await expect(
      writeCheckpoint(
        destination,
        invalid,
        initializeParameters(DEFAULT_MODEL_CONFIG, config.seed),
        trainingLog,
      ),
    ).rejects.toThrow('expected 1');
    expect(await readdir(parent)).toEqual([]);
  });

  it('rejects filesystem corruption and a log from after the saved training step', async () => {
    const parent = await temporaryDirectory();
    const destination = join(parent, 'corrupt');
    await writeCheckpoint(
      destination,
      config,
      initializeParameters(DEFAULT_MODEL_CONFIG, config.seed),
      trainingLog,
    );
    const weightsPath = join(destination, 'weights.bin');
    const weights = await readFile(weightsPath);
    await writeFile(weightsPath, Buffer.concat([weights, Buffer.alloc(4)]));
    await expect(loadCheckpoint(destination)).rejects.toThrow('extra bytes');

    const laterLog = {
      ...trainingLog,
      entries: [{ ...trainingLog.entries[0], step: config.trainingStep + 1 }],
    };
    await writeFile(join(destination, 'training-log.json'), JSON.stringify(laterLog));
    await expect(loadCheckpoint(destination)).rejects.toThrow('after checkpoint step');
  });
});
