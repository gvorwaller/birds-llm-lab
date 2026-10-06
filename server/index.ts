import { randomBytes } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { mkdir, readFile, rename, rm, stat } from 'node:fs/promises';
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { extname, join, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { ServiceStatus, TrainingPresetId } from '../src/lib/data/api-types.js';
import { parseCorpusManifest } from '../src/lib/data/schemas.js';
import { exportCorpus } from './export/corpus-export.js';
import { CheckpointCatalog } from './checkpoints/catalog.js';
import {
  DuplicateExportJobError,
  ExportJobCoordinator,
  type ExportRunner,
} from './jobs/export-jobs.js';
import {
  DuplicateTrainingJobError,
  TrainingJobCoordinator,
  type TrainingRunner,
} from './jobs/training-jobs.js';
import { trainReadyCheckpoint } from './training/trainer.js';

export const LOOPBACK_HOST = '127.0.0.1';
export const SERVICE_PORT = 5301;
const CSRF_PLACEHOLDER = '__BIRDS_LLM_LAB_CSRF__';
const CONTENT_SECURITY_POLICY = [
  "default-src 'self'",
  "base-uri 'none'",
  "connect-src 'self'",
  "font-src 'self'",
  "form-action 'self'",
  "frame-ancestors 'none'",
  "img-src 'self' data:",
  "object-src 'none'",
  "script-src 'self'",
  "style-src 'self'",
].join('; ');

const MIME_TYPES: Record<string, string> = {
  '.css': 'text/css; charset=utf-8',
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.map': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.woff2': 'font/woff2',
};

export interface LabServerOptions {
  projectDirectory?: string;
  distDirectory?: string;
  dataDirectory?: string;
  port?: number;
  csrfToken?: string;
  runExport?: ExportRunner;
  runTraining?: TrainingRunner;
  checkpointDirectory?: string;
}

export interface RunningLabServer {
  server: Server;
  origin: string;
  csrfToken: string;
  coordinator: ExportJobCoordinator;
  trainingCoordinator: TrainingJobCoordinator;
  checkpointCatalog: CheckpointCatalog;
  close(): Promise<void>;
}

async function readJsonBody(request: IncomingMessage): Promise<unknown> {
  const chunks: Buffer[] = [];
  let length = 0;
  for await (const chunk of request) {
    const bytes = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk as Uint8Array);
    length += bytes.byteLength;
    if (length > 4_096) throw new Error('Request body is too large.');
    chunks.push(bytes);
  }
  try {
    return JSON.parse(Buffer.concat(chunks).toString('utf8')) as unknown;
  } catch {
    throw new Error('Request body must be valid JSON.');
  }
}

function trainingPreset(value: unknown): TrainingPresetId {
  if (
    typeof value !== 'object' ||
    value === null ||
    Array.isArray(value) ||
    Object.keys(value).length !== 1 ||
    !('preset' in value) ||
    (value.preset !== 'quick' && value.preset !== 'ready')
  ) {
    throw new Error('Training request must contain only preset quick or ready.');
  }
  return value.preset;
}

function securityHeaders(response: ServerResponse): void {
  response.setHeader('Content-Security-Policy', CONTENT_SECURITY_POLICY);
  response.setHeader('Referrer-Policy', 'no-referrer');
  response.setHeader('X-Content-Type-Options', 'nosniff');
  response.setHeader('X-Frame-Options', 'DENY');
}

function json(response: ServerResponse, status: number, body: unknown): void {
  securityHeaders(response);
  response.statusCode = status;
  response.setHeader('Cache-Control', 'no-store');
  response.setHeader('Content-Type', 'application/json; charset=utf-8');
  response.end(`${JSON.stringify(body)}\n`);
}

function reject(response: ServerResponse, status: number, error: string): void {
  json(response, status, { error });
}

function isPost(request: IncomingMessage): boolean {
  return request.method === 'POST';
}

function validateRequestBoundary(
  request: IncomingMessage,
  response: ServerResponse,
  allowedHosts: Set<string>,
  allowedOrigins: Set<string>,
  csrfToken: string,
): boolean {
  const host = request.headers.host;
  if (!host || !allowedHosts.has(host)) {
    reject(response, 403, 'Unexpected Host header.');
    return false;
  }
  const origin = request.headers.origin;
  if (origin !== undefined && !allowedOrigins.has(origin)) {
    reject(response, 403, 'Unexpected Origin header.');
    return false;
  }
  if (isPost(request)) {
    if (!origin || !allowedOrigins.has(origin)) {
      reject(response, 403, 'A same-origin request is required.');
      return false;
    }
    if (request.headers['x-birds-llm-lab-csrf'] !== csrfToken) {
      reject(response, 403, 'The local request token was missing or invalid.');
      return false;
    }
  }
  return true;
}

async function readManifest(
  manifestPath: string,
): Promise<ReturnType<typeof parseCorpusManifest> | null> {
  try {
    return parseCorpusManifest(JSON.parse(await readFile(manifestPath, 'utf8')) as unknown);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return null;
    throw error;
  }
}

async function serveStatic(
  request: IncomingMessage,
  response: ServerResponse,
  pathname: string,
  distDirectory: string,
  csrfToken: string,
): Promise<void> {
  if (request.method !== 'GET' && request.method !== 'HEAD') {
    reject(response, 405, 'Method not allowed.');
    return;
  }

  let decodedPath: string;
  try {
    decodedPath = decodeURIComponent(pathname);
  } catch {
    reject(response, 400, 'Invalid URL encoding.');
    return;
  }
  const requestedRelativePath =
    decodedPath === '/' ? 'index.html' : decodedPath.replace(/^\/+/, '');
  let filePath = resolve(distDirectory, requestedRelativePath);
  const distPrefix = `${resolve(distDirectory)}${sep}`;
  if (!filePath.startsWith(distPrefix)) {
    reject(response, 404, 'Not found.');
    return;
  }
  try {
    const details = await stat(filePath);
    if (!details.isFile()) throw Object.assign(new Error('not a file'), { code: 'ENOENT' });
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error;
    if (extname(requestedRelativePath)) {
      reject(response, 404, 'Not found.');
      return;
    }
    filePath = join(distDirectory, 'index.html');
  }

  securityHeaders(response);
  const extension = extname(filePath);
  response.statusCode = 200;
  response.setHeader('Content-Type', MIME_TYPES[extension] ?? 'application/octet-stream');
  if (extension === '.html') {
    response.setHeader('Cache-Control', 'no-store');
    const html = (await readFile(filePath, 'utf8')).replaceAll(CSRF_PLACEHOLDER, csrfToken);
    response.setHeader('Content-Length', Buffer.byteLength(html));
    response.end(request.method === 'HEAD' ? undefined : html);
    return;
  }
  response.setHeader('Cache-Control', 'public, max-age=31536000, immutable');
  if (request.method === 'HEAD') {
    response.end();
    return;
  }
  createReadStream(filePath).pipe(response);
}

async function routeApi(
  request: IncomingMessage,
  response: ServerResponse,
  pathname: string,
  coordinator: ExportJobCoordinator,
  trainingCoordinator: TrainingJobCoordinator,
  checkpointCatalog: CheckpointCatalog,
  manifestPath: string,
  tokenizerPath: string,
): Promise<void> {
  if (request.method === 'GET' && pathname === '/api/status') {
    const manifest = await readManifest(manifestPath);
    const status: ServiceStatus = {
      status: 'ok',
      version: 1,
      endpoint: '127.0.0.1:5301',
      corpusAvailable: manifest !== null,
      activeExportJobId: coordinator.active()?.id ?? null,
      latestExportJobId: coordinator.latest()?.id ?? null,
      activeTrainingJobId: trainingCoordinator.active()?.id ?? null,
      latestTrainingJobId: trainingCoordinator.latest()?.id ?? null,
      recoveryWarning:
        [coordinator.recoveryWarning, trainingCoordinator.recoveryWarning]
          .filter((value) => value !== null)
          .join(' ') || null,
    };
    json(response, 200, status);
    return;
  }

  if (request.method === 'GET' && pathname === '/api/corpus/manifest') {
    const manifest = await readManifest(manifestPath);
    if (!manifest) reject(response, 404, 'No exported corpus is available yet.');
    else json(response, 200, manifest);
    return;
  }

  if (request.method === 'POST' && pathname === '/api/corpus/export') {
    try {
      json(response, 202, await coordinator.start());
    } catch (error) {
      if (error instanceof DuplicateExportJobError) {
        json(response, 409, { error: error.message, job: error.job });
        return;
      }
      throw error;
    }
    return;
  }

  const jobMatch = pathname.match(/^\/api\/jobs\/([a-zA-Z0-9-]+)$/);
  if (request.method === 'GET' && jobMatch) {
    const job = coordinator.get(jobMatch[1]) ?? trainingCoordinator.get(jobMatch[1]);
    if (!job) reject(response, 404, 'Job not found.');
    else json(response, 200, job);
    return;
  }

  const cancelMatch = pathname.match(/^\/api\/jobs\/([a-zA-Z0-9-]+)\/cancel$/);
  if (request.method === 'POST' && cancelMatch) {
    const job =
      (await coordinator.cancel(cancelMatch[1])) ??
      (await trainingCoordinator.cancel(cancelMatch[1]));
    if (!job) reject(response, 404, 'Job not found.');
    else json(response, 200, job);
    return;
  }

  if (request.method === 'GET' && pathname === '/api/checkpoints') {
    json(response, 200, await checkpointCatalog.list());
    return;
  }

  if (request.method === 'GET' && pathname === '/api/checkpoints/active/inspect') {
    try {
      const bundle = await checkpointCatalog.inspectionBundle(tokenizerPath);
      if (bundle === null) reject(response, 404, 'Select a validated checkpoint first.');
      else json(response, 200, bundle);
    } catch (error) {
      reject(
        response,
        409,
        error instanceof Error ? error.message : 'Checkpoint inspection unavailable.',
      );
    }
    return;
  }

  if (request.method === 'POST' && pathname === '/api/training/jobs') {
    try {
      const preset = trainingPreset(await readJsonBody(request));
      json(response, 202, await trainingCoordinator.start(preset));
    } catch (error) {
      if (error instanceof DuplicateTrainingJobError) {
        json(response, 409, { error: error.message, job: error.job });
      } else {
        reject(response, 400, error instanceof Error ? error.message : 'Invalid training request.');
      }
    }
    return;
  }

  const selectionMatch = pathname.match(/^\/api\/checkpoints\/([^/]+)\/select$/);
  if (request.method === 'POST' && selectionMatch) {
    try {
      json(response, 200, await checkpointCatalog.select(decodeURIComponent(selectionMatch[1])));
    } catch (error) {
      reject(response, 404, error instanceof Error ? error.message : 'Checkpoint not found.');
    }
    return;
  }

  reject(response, 404, 'API operation not found.');
}

export async function startLabServer(options: LabServerOptions = {}): Promise<RunningLabServer> {
  const projectDirectory = resolve(
    options.projectDirectory ?? join(fileURLToPath(new URL('.', import.meta.url)), '..'),
  );
  const distDirectory = resolve(options.distDirectory ?? join(projectDirectory, 'dist'));
  const dataDirectory = resolve(options.dataDirectory ?? join(projectDirectory, 'data'));
  const checkpointDirectory = resolve(
    options.checkpointDirectory ?? join(projectDirectory, 'checkpoints'),
  );
  const requestedPort = options.port ?? SERVICE_PORT;
  const csrfToken = options.csrfToken ?? randomBytes(32).toString('hex');
  const runExport =
    options.runExport ??
    ((jobOptions) =>
      exportCorpus({
        ...jobOptions,
        outputDirectory: dataDirectory,
        repositoryDirectory: projectDirectory,
      }));
  const coordinator = new ExportJobCoordinator({
    statePath: join(dataDirectory, 'job-state.json'),
    runExport,
  });
  await coordinator.initialize();
  const checkpointCatalog = new CheckpointCatalog(
    checkpointDirectory,
    join(dataDirectory, 'checkpoint-selection.json'),
  );
  await checkpointCatalog.initialize();
  const runTraining =
    options.runTraining ??
    (async ({ jobId, preset, signal, onProgress }) => {
      const staging = join(checkpointDirectory, `.training-${jobId}`);
      const checkpointId = `trained-${jobId}`;
      const destination = join(checkpointDirectory, checkpointId);
      await rm(staging, { recursive: true, force: true });
      await mkdir(staging, { recursive: true, mode: 0o700 });
      try {
        const totalSteps = TrainingJobCoordinator.totalSteps(preset);
        const result = await trainReadyCheckpoint({
          outputRoot: staging,
          totalSteps,
          checkpointEvery: preset === 'quick' ? totalSteps : 500,
          logEvery: preset === 'quick' ? 10 : 25,
          validateEvery: preset === 'quick' ? 25 : 100,
          validationBatches: preset === 'quick' ? 4 : 8,
          signal,
          onProgress,
        });
        await rename(result.checkpointPath, destination);
        return { checkpointId };
      } finally {
        await rm(staging, { recursive: true, force: true });
      }
    });
  const trainingCoordinator = new TrainingJobCoordinator({
    statePath: join(dataDirectory, 'training-job-state.json'),
    runTraining,
  });
  await trainingCoordinator.initialize();

  const server = createServer((request, response) => {
    void (async () => {
      const actualPort = (server.address() as { port: number } | null)?.port ?? requestedPort;
      const allowedHosts = new Set([`127.0.0.1:${actualPort}`, `localhost:${actualPort}`]);
      const allowedOrigins = new Set([
        `http://127.0.0.1:${actualPort}`,
        `http://localhost:${actualPort}`,
      ]);
      if (!validateRequestBoundary(request, response, allowedHosts, allowedOrigins, csrfToken))
        return;
      const url = new URL(request.url ?? '/', `http://${request.headers.host}`);
      if (url.pathname.startsWith('/api/')) {
        await routeApi(
          request,
          response,
          url.pathname,
          coordinator,
          trainingCoordinator,
          checkpointCatalog,
          join(dataDirectory, 'manifest.json'),
          join(projectDirectory, 'src/assets/tokenizer-1024.json'),
        );
      } else {
        await serveStatic(request, response, url.pathname, distDirectory, csrfToken);
      }
    })().catch((error: unknown) => {
      console.error('Request failed:', error instanceof Error ? error.message : 'unknown error');
      if (!response.headersSent)
        reject(response, 500, 'The local service could not complete this request.');
      else response.destroy();
    });
  });

  await new Promise<void>((resolveListen, rejectListen) => {
    server.once('error', rejectListen);
    server.listen(requestedPort, LOOPBACK_HOST, () => {
      server.off('error', rejectListen);
      resolveListen();
    });
  });
  const actualPort = (server.address() as { port: number }).port;
  return {
    server,
    origin: `http://${LOOPBACK_HOST}:${actualPort}`,
    csrfToken,
    coordinator,
    trainingCoordinator,
    checkpointCatalog,
    close: async () => {
      await new Promise<void>((resolveClose, rejectClose) => {
        server.close((error) => (error ? rejectClose(error) : resolveClose()));
      });
      await coordinator.flush();
      await trainingCoordinator.shutdown();
    },
  };
}

async function main(): Promise<void> {
  const running = await startLabServer();
  console.log(`Birds LLM Lab listening at ${running.origin}`);
  const stop = () => {
    void running.close().finally(() => process.exit(0));
  };
  process.once('SIGINT', stop);
  process.once('SIGTERM', stop);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((error: unknown) => {
    console.error('Service failed:', error instanceof Error ? error.message : 'unknown error');
    process.exitCode = 1;
  });
}
