import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import type { CorpusManifest } from '../../src/lib/data/schemas';
import { ExportCancelledError } from '../export/corpus-export';
import { DuplicateExportJobError, ExportJobCoordinator, type ExportRunner } from './export-jobs';

const temporaryDirectories: string[] = [];

afterEach(async () => {
  await Promise.all(
    temporaryDirectories.splice(0).map((path) => rm(path, { recursive: true, force: true })),
  );
});

function manifest(): CorpusManifest {
  return {
    formatVersion: 1,
    exportedAt: '2026-09-28T12:00:00.000Z',
    database: { host: '127.0.0.1', port: 15436, name: 'birds_test' },
    sourceRows: 1,
    emittedRows: 1,
    skippedRows: 0,
    skippedReasons: {},
    malformedSectionsSkipped: 0,
    splits: {
      train: { documents: 1, bytes: 10 },
      validation: { documents: 0, bytes: 0 },
      test: { documents: 0, bytes: 0 },
    },
    sourceBytes: { wikipediaExtracts: 10, wikipediaSections: 0, fieldCraft: 0 },
    corpusBytes: 10,
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
}

async function coordinator(runExport: ExportRunner): Promise<ExportJobCoordinator> {
  const directory = await mkdtemp(join(tmpdir(), 'birds-llm-jobs-'));
  temporaryDirectories.push(directory);
  const result = new ExportJobCoordinator({
    statePath: join(directory, 'jobs.json'),
    runExport,
    createId: () => '00000000-0000-4000-8000-000000000001',
  });
  await result.initialize();
  return result;
}

async function waitForStatus(
  jobs: ExportJobCoordinator,
  id: string,
  status: 'cancelled' | 'succeeded',
): Promise<void> {
  for (let attempt = 0; attempt < 50; attempt += 1) {
    if (jobs.get(id)?.status === status) return;
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
  throw new Error(`Job did not reach ${status}.`);
}

describe('export job coordinator', () => {
  it('rejects a duplicate and cancels the active job', async () => {
    const jobs = await coordinator(
      ({ signal }) =>
        new Promise((_, reject) => {
          signal.addEventListener('abort', () => reject(new ExportCancelledError()), {
            once: true,
          });
        }),
    );
    const job = await jobs.start();
    await expect(jobs.start()).rejects.toBeInstanceOf(DuplicateExportJobError);
    expect(['cancelling', 'cancelled']).toContain((await jobs.cancel(job.id))?.status);
    await waitForStatus(jobs, job.id, 'cancelled');
    await jobs.flush();
    expect(jobs.get(job.id)?.error).toBeNull();
  });

  it('persists a successful terminal job for refresh recovery', async () => {
    const directory = await mkdtemp(join(tmpdir(), 'birds-llm-jobs-'));
    temporaryDirectories.push(directory);
    const statePath = join(directory, 'jobs.json');
    const runner: ExportRunner = async ({ onProgress }) => {
      onProgress({ phase: 'writing', processedRows: 1, totalRows: 1 });
      return {
        corpusPath: '/data/corpus.jsonl',
        manifestPath: '/data/manifest.json',
        manifest: manifest(),
      };
    };
    const first = new ExportJobCoordinator({
      statePath,
      runExport: runner,
      createId: () => 'job-1',
    });
    await first.initialize();
    const job = await first.start();
    await waitForStatus(first, job.id, 'succeeded');
    await first.flush();

    const recovered = new ExportJobCoordinator({ statePath, runExport: runner });
    await recovered.initialize();
    expect(recovered.latest()).toMatchObject({ id: 'job-1', status: 'succeeded' });
    expect(recovered.latest()?.manifest?.emittedRows).toBe(1);
  });
});
