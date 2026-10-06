import { createHash } from 'node:crypto';
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises';
import { request } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import { serializeCorpusRow, type CorpusManifest, type CorpusRow } from '../src/lib/data/schemas';
import { ExportCancelledError, type ExportCorpusResult } from './export/corpus-export';
import { TrainingCancelledError } from './training/trainer';
import { startLabServer, type LabServerOptions, type RunningLabServer } from './index';

const temporaryDirectories: string[] = [];
const runningServers: RunningLabServer[] = [];

afterEach(async () => {
  await Promise.all(runningServers.splice(0).map((server) => server.close()));
  await Promise.all(
    temporaryDirectories.splice(0).map((path) => rm(path, { recursive: true, force: true })),
  );
});

async function fixtureServer(
  runExport: LabServerOptions['runExport'],
  runTraining: NonNullable<LabServerOptions['runTraining']> = delayedTraining(),
  seedData?: (directory: string) => Promise<void>,
): Promise<RunningLabServer> {
  const directory = await mkdtemp(join(tmpdir(), 'birds-llm-server-'));
  temporaryDirectories.push(directory);
  const dist = join(directory, 'dist');
  const data = join(directory, 'data');
  await mkdir(dist, { recursive: true });
  await mkdir(data, { recursive: true });
  await seedData?.(data);
  await writeFile(
    join(dist, 'index.html'),
    '<!doctype html><meta name="birds-llm-lab-csrf" content="__BIRDS_LLM_LAB_CSRF__"><main>Lab</main>',
  );
  await writeFile(join(dist, 'asset.js'), 'export const local = true;');
  const server = await startLabServer({
    projectDirectory: directory,
    distDirectory: dist,
    dataDirectory: data,
    port: 0,
    csrfToken: 'test-csrf-token',
    runExport,
    runTraining,
  });
  runningServers.push(server);
  return server;
}

async function seedEvidenceCorpus(directory: string): Promise<void> {
  const row: CorpusRow = {
    code: 'osprey',
    name: 'Osprey',
    sci: 'Pandion haliaetus',
    order: 'Accipitriformes',
    family: 'Pandionidae',
    extract: 'The Osprey hunts above coastal water.',
    sections: [{ title: 'Range', text: 'Found near many shorelines.' }],
    field_craft: 'Private field craft text.',
    tags: ['private-tag'],
    split: 'test',
  };
  const corpus = Buffer.from(`${serializeCorpusRow(row)}\n`);
  const manifest: CorpusManifest = {
    formatVersion: 1,
    exportedAt: '2026-10-06T12:00:00.000Z',
    database: { host: '127.0.0.1', port: 15436, name: 'birds_test' },
    sourceRows: 1,
    emittedRows: 1,
    skippedRows: 0,
    skippedReasons: {},
    malformedSectionsSkipped: 0,
    splits: {
      train: { documents: 0, bytes: 0 },
      validation: { documents: 0, bytes: 0 },
      test: { documents: 1, bytes: 0 },
    },
    sourceBytes: { wikipediaExtracts: 0, wikipediaSections: 0, fieldCraft: 0 },
    corpusBytes: corpus.length,
    corpusSha256: createHash('sha256').update(corpus).digest('hex'),
    splitAlgorithm: 'sha256-bucket',
    splitVersion: 'split-v1',
    splitOverrides: ['osprey'],
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
  await writeFile(join(directory, 'corpus.jsonl'), corpus);
  await writeFile(join(directory, 'manifest.json'), JSON.stringify(manifest));
}

function delayedTraining(): NonNullable<LabServerOptions['runTraining']> {
  return ({ signal }) =>
    new Promise((_, reject) => {
      signal.addEventListener('abort', () => reject(new TrainingCancelledError()), { once: true });
    });
}

function delayedExport(): NonNullable<LabServerOptions['runExport']> {
  return ({ signal }) =>
    new Promise<ExportCorpusResult>((_, reject) => {
      signal.addEventListener('abort', () => reject(new ExportCancelledError()), { once: true });
    });
}

async function requestWithHost(server: RunningLabServer, host: string): Promise<number> {
  const url = new URL(server.origin);
  return new Promise((resolve, reject) => {
    const outgoing = request(
      {
        hostname: url.hostname,
        port: url.port,
        path: '/api/status',
        headers: { Host: host },
      },
      (response) => {
        response.resume();
        response.once('end', () => resolve(response.statusCode ?? 0));
      },
    );
    outgoing.once('error', reject);
    outgoing.end();
  });
}

async function post(
  server: RunningLabServer,
  path: string,
  overrides: HeadersInit = {},
): Promise<Response> {
  return fetch(`${server.origin}${path}`, {
    method: 'POST',
    headers: {
      Origin: server.origin,
      'X-Birds-LLM-Lab-CSRF': server.csrfToken,
      ...overrides,
    },
  });
}

async function postJson(server: RunningLabServer, path: string, body: unknown): Promise<Response> {
  return fetch(`${server.origin}${path}`, {
    method: 'POST',
    headers: {
      Origin: server.origin,
      'X-Birds-LLM-Lab-CSRF': server.csrfToken,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });
}

describe('loopback service', () => {
  it('serves bounded evidence results from exported training fields through the fixed API', async () => {
    const server = await fixtureServer(delayedExport(), delayedTraining(), seedEvidenceCorpus);
    const response = await fetch(`${server.origin}/api/evidence/search?q=coastal&page=0`);
    expect(response.status).toBe(200);
    const result = await response.json();
    expect(result).toMatchObject({
      indexedDocuments: 1,
      pageSize: 20,
      exactPhrase: {
        total: 1,
        hits: [{ code: 'osprey', split: 'test', field: 'Wikipedia extract' }],
      },
    });
    expect((await fetch(`${server.origin}/api/evidence/search?q=private`)).status).toBe(200);
    const hidden = await fetch(`${server.origin}/api/evidence/search?q=private`).then((item) =>
      item.json(),
    );
    expect(hidden.normalizedTerms.total).toBe(0);
    expect((await fetch(`${server.origin}/api/evidence/search?q=coastal&page=nope`)).status).toBe(
      400,
    );
  });

  it('reports evidence search unavailable before export', async () => {
    const server = await fixtureServer(delayedExport());
    expect((await fetch(`${server.origin}/api/evidence/search?q=osprey`)).status).toBe(409);
  });

  it('requires an active checkpoint for inspection', async () => {
    const server = await fixtureServer(delayedExport());
    const response = await fetch(`${server.origin}/api/checkpoints/active/inspect`);
    expect(response.status).toBe(404);
  });

  it('serves the SPA fallback with an embedded per-process token and restrictive CSP', async () => {
    const server = await fixtureServer(delayedExport());
    const response = await fetch(`${server.origin}/data`);
    expect(response.status).toBe(200);
    expect(response.headers.get('content-security-policy')).toContain("default-src 'self'");
    expect(response.headers.get('access-control-allow-origin')).toBeNull();
    expect(await response.text()).toContain('content="test-csrf-token"');
  });

  it('rejects foreign Host, foreign Origin, and missing CSRF', async () => {
    const server = await fixtureServer(delayedExport());
    expect(await requestWithHost(server, 'attacker.invalid')).toBe(403);
    const foreignOrigin = await post(server, '/api/corpus/export', {
      Origin: ['https:/', '/attacker', 'invalid'].join('.').replace('/.', '/'),
    });
    expect(foreignOrigin.status).toBe(403);
    const missingCsrf = await fetch(`${server.origin}/api/corpus/export`, {
      method: 'POST',
      headers: { Origin: server.origin },
    });
    expect(missingCsrf.status).toBe(403);
  });

  it('exposes the fixed API, rejects duplicate export, and cancels safely', async () => {
    const server = await fixtureServer(delayedExport());
    const status = await fetch(`${server.origin}/api/status`).then((response) => response.json());
    expect(status).toMatchObject({
      status: 'ok',
      endpoint: '127.0.0.1:5301',
      corpusAvailable: false,
    });
    const first = await post(server, '/api/corpus/export');
    expect(first.status).toBe(202);
    const job = (await first.json()) as { id: string };
    const duplicate = await post(server, '/api/corpus/export');
    expect(duplicate.status).toBe(409);
    const cancelled = await post(server, `/api/jobs/${job.id}/cancel`);
    expect(cancelled.status).toBe(200);
    for (let attempt = 0; attempt < 50; attempt += 1) {
      const current = await fetch(`${server.origin}/api/jobs/${job.id}`).then((response) =>
        response.json(),
      );
      if ((current as { status: string }).status === 'cancelled') {
        expect(current).toMatchObject({ error: null });
        return;
      }
      await new Promise((resolve) => setTimeout(resolve, 5));
    }
    throw new Error('API job did not cancel.');
  });

  it('starts only safe training presets, rejects duplicates, and cancels through the fixed API', async () => {
    const server = await fixtureServer(delayedExport());
    const checkpoints = await fetch(`${server.origin}/api/checkpoints`).then((response) =>
      response.json(),
    );
    expect(checkpoints).toEqual({
      checkpoints: [],
      activeCheckpointId: null,
      recoveryWarning: null,
    });
    expect((await postJson(server, '/api/training/jobs', { preset: 'custom' })).status).toBe(400);
    const started = await postJson(server, '/api/training/jobs', { preset: 'quick' });
    expect(started.status).toBe(202);
    const job = (await started.json()) as { id: string };
    expect((await postJson(server, '/api/training/jobs', { preset: 'ready' })).status).toBe(409);
    expect((await post(server, `/api/jobs/${job.id}/cancel`)).status).toBe(200);
    let finalStatus = '';
    for (let attempt = 0; attempt < 50; attempt += 1) {
      const current = (await fetch(`${server.origin}/api/jobs/${job.id}`).then((response) =>
        response.json(),
      )) as { status: string };
      finalStatus = current.status;
      if (current.status === 'cancelled') break;
      await new Promise((resolve) => setTimeout(resolve, 5));
    }
    expect(finalStatus).toBe('cancelled');
    expect((await post(server, '/api/checkpoints/example/select')).status).toBe(404);
  });
});
