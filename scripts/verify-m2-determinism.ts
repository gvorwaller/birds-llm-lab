import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { loadCheckpoint } from '../server/checkpoint-store';
import { trainReadyCheckpoint } from '../server/training/trainer';
import { encodeCheckpointWeights } from '../src/lib/model/checkpoint';

const root = await mkdtemp(join(tmpdir(), 'birds-llm-m2-determinism-'));
try {
  const run = (name: string) =>
    trainReadyCheckpoint({
      outputRoot: join(root, name),
      totalSteps: 100,
      checkpointEvery: 100,
      logEvery: 25,
      validateEvery: 25,
      validationBatches: 4,
      sourceGitRevision: null,
    });
  const first = await run('first');
  const second = await run('second');
  const [firstLoaded, secondLoaded] = await Promise.all([
    loadCheckpoint(first.checkpointPath),
    loadCheckpoint(second.checkpointPath),
  ]);
  const firstWeights = encodeCheckpointWeights(
    firstLoaded.config.model,
    firstLoaded.parameters,
  ).bytes;
  const secondWeights = encodeCheckpointWeights(
    secondLoaded.config.model,
    secondLoaded.parameters,
  ).bytes;
  if (!Buffer.from(firstWeights).equals(Buffer.from(secondWeights))) {
    throw new Error('Fixed-seed 100-step runs produced different weights.');
  }
  const deterministicEntries = (loaded: typeof firstLoaded) =>
    loaded.trainingLog.entries.map(({ elapsedMs: _elapsedMs, ...entry }) => entry);
  if (
    JSON.stringify(deterministicEntries(firstLoaded)) !==
    JSON.stringify(deterministicEntries(secondLoaded))
  ) {
    throw new Error('Fixed-seed 100-step runs produced different model metrics.');
  }
  if (JSON.stringify(firstLoaded.resumeState) !== JSON.stringify(secondLoaded.resumeState)) {
    throw new Error('Fixed-seed 100-step runs produced different resumable state.');
  }
  console.log(
    JSON.stringify(
      {
        status: 'passed',
        steps: 100,
        weightsByteIdentical: true,
        metricsIdentical: true,
        resumeStateIdentical: true,
        loggedEntries: firstLoaded.trainingLog.entries.length,
      },
      null,
      2,
    ),
  );
} finally {
  await rm(root, { recursive: true, force: true });
}
