import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import type { TrainingJobState } from '../../src/lib/data/api-types';
import { TrainingCancelledError } from '../training/trainer';
import {
  DuplicateTrainingJobError,
  TrainingJobCoordinator,
  type TrainingRunner,
} from './training-jobs';

const temporaryDirectories: string[] = [];
const coordinators: TrainingJobCoordinator[] = [];

afterEach(async () => {
  await Promise.all(coordinators.splice(0).map((coordinator) => coordinator.shutdown()));
  await Promise.all(
    temporaryDirectories.splice(0).map((path) => rm(path, { recursive: true, force: true })),
  );
});

async function fixture(runTraining: TrainingRunner): Promise<TrainingJobCoordinator> {
  const directory = await mkdtemp(join(tmpdir(), 'birds-llm-training-jobs-'));
  temporaryDirectories.push(directory);
  const coordinator = new TrainingJobCoordinator({
    statePath: join(directory, 'jobs.json'),
    runTraining,
    createId: () => '00000000-0000-4000-8000-000000000001',
  });
  await coordinator.initialize();
  coordinators.push(coordinator);
  return coordinator;
}

async function waitFor(
  coordinator: TrainingJobCoordinator,
  id: string,
  status: TrainingJobState['status'],
): Promise<TrainingJobState> {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    const job = coordinator.get(id);
    if (job?.status === status) return job;
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
  throw new Error(`Training job did not reach ${status}.`);
}

describe('training job coordinator', () => {
  it('rejects duplicate starts and cooperatively cancels one active job', async () => {
    const coordinator = await fixture(
      ({ signal }) =>
        new Promise((_, reject) => {
          signal.addEventListener('abort', () => reject(new TrainingCancelledError()), {
            once: true,
          });
        }),
    );
    const job = await coordinator.start('quick');
    await expect(coordinator.start('ready')).rejects.toBeInstanceOf(DuplicateTrainingJobError);
    expect(['cancelling', 'cancelled']).toContain((await coordinator.cancel(job.id))?.status);
    const cancelled = await waitFor(coordinator, job.id, 'cancelled');
    expect(cancelled.error).toBeNull();
    expect(cancelled.checkpointId).toBeNull();
  });

  it('reports bounded metrics and the completed checkpoint', async () => {
    const coordinator = await fixture(async ({ onProgress }) => {
      onProgress({
        step: 25,
        split: 'train',
        predictionCount: 1_000,
        meanLoss: 5.25,
        perplexity: Math.exp(5.25),
        learningRate: 0.002,
        gradientNorm: 0.8,
        elapsedMs: 5_000,
        sample: null,
      });
      onProgress({
        step: 25,
        split: 'validation',
        predictionCount: 500,
        meanLoss: 5.4,
        perplexity: Math.exp(5.4),
        learningRate: 0.002,
        gradientNorm: 0,
        elapsedMs: 5_500,
        sample: 'Common name: Test bird',
      });
      return { checkpointId: 'trained-test' };
    });
    const started = await coordinator.start('quick');
    const completed = await waitFor(coordinator, started.id, 'succeeded');
    expect(completed).toMatchObject({
      checkpointId: 'trained-test',
      progress: {
        step: 100,
        totalSteps: 100,
        trainLoss: 5.25,
        validationLoss: 5.4,
        estimatedRemainingMs: 0,
        latestSample: 'Common name: Test bird',
      },
    });
  });

  it('recovers an active persisted job as interrupted without a selectable checkpoint', async () => {
    const directory = await mkdtemp(join(tmpdir(), 'birds-llm-training-recovery-'));
    temporaryDirectories.push(directory);
    const statePath = join(directory, 'jobs.json');
    await mkdir(directory, { recursive: true });
    await writeFile(
      statePath,
      JSON.stringify({
        formatVersion: 1,
        jobs: [
          {
            id: 'job-1',
            kind: 'model-training',
            preset: 'ready',
            status: 'running',
            createdAt: '2026-09-29T00:00:00.000Z',
            updatedAt: '2026-09-29T00:01:00.000Z',
            progress: {
              step: 25,
              totalSteps: 2_000,
              trainLoss: 6,
              validationLoss: null,
              learningRate: 0.001,
              elapsedMs: 10_000,
              estimatedRemainingMs: 790_000,
              latestSample: null,
            },
            logs: [],
            checkpointId: null,
            error: null,
          },
        ],
      }),
    );
    const recovered = new TrainingJobCoordinator({
      statePath,
      runTraining: async () => ({ checkpointId: 'unused' }),
    });
    await recovered.initialize();
    coordinators.push(recovered);
    expect(recovered.latest()).toMatchObject({
      id: 'job-1',
      status: 'interrupted',
      checkpointId: null,
    });
    expect(recovered.latest()?.error).toContain('service restarted');
  });
});
