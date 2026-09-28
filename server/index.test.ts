import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises';
import { request } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import { ExportCancelledError, type ExportCorpusResult } from './export/corpus-export';
import { startLabServer, type LabServerOptions, type RunningLabServer } from './index';

const temporaryDirectories: string[] = [];
const runningServers: RunningLabServer[] = [];

afterEach(async () => {
  await Promise.all(runningServers.splice(0).map((server) => server.close()));
  await Promise.all(
    temporaryDirectories.splice(0).map((path) => rm(path, { recursive: true, force: true })),
  );
});

async function fixtureServer(runExport: LabServerOptions['runExport']): Promise<RunningLabServer> {
  const directory = await mkdtemp(join(tmpdir(), 'birds-llm-server-'));
  temporaryDirectories.push(directory);
  const dist = join(directory, 'dist');
  const data = join(directory, 'data');
  await mkdir(dist, { recursive: true });
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
  });
  runningServers.push(server);
  return server;
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

describe('loopback service', () => {
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

  it('keeps unimplemented training and checkpoint actions explicit', async () => {
    const server = await fixtureServer(delayedExport());
    const checkpoints = await fetch(`${server.origin}/api/checkpoints`).then((response) =>
      response.json(),
    );
    expect(checkpoints).toEqual({
      checkpoints: [],
      activeCheckpointId: null,
      availableInMilestone: 2,
    });
    expect((await post(server, '/api/training/jobs')).status).toBe(501);
    expect((await post(server, '/api/checkpoints/example/select')).status).toBe(501);
  });
});
