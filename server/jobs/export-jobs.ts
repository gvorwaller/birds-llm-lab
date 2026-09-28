import { randomUUID } from 'node:crypto';
import { mkdir, readFile, rename, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import type { ExportJobState } from '../../src/lib/data/api-types.js';
import type { ExportCorpusResult, ExportProgress } from '../export/corpus-export.js';
import { ExportCancelledError } from '../export/corpus-export.js';

const ACTIVE_STATUSES = new Set<ExportJobState['status']>(['queued', 'running', 'cancelling']);
const MAX_RETAINED_JOBS = 10;
const MAX_LOG_LINES = 50;

export type ExportRunner = (options: {
  signal: AbortSignal;
  onProgress: (progress: ExportProgress) => void;
}) => Promise<ExportCorpusResult>;

interface PersistedJobState {
  formatVersion: 1;
  jobs: ExportJobState[];
}

export interface ExportJobCoordinatorOptions {
  statePath: string;
  runExport: ExportRunner;
  now?: () => Date;
  createId?: () => string;
}

export class DuplicateExportJobError extends Error {
  constructor(readonly job: ExportJobState) {
    super('A corpus export is already running.');
    this.name = 'DuplicateExportJobError';
  }
}

function copyJob(job: ExportJobState): ExportJobState {
  return structuredClone(job);
}

function isActive(job: ExportJobState): boolean {
  return ACTIVE_STATUSES.has(job.status);
}

function sanitizedError(error: unknown): string {
  if (error instanceof ExportCancelledError) return error.message;
  if (!(error instanceof Error)) return 'The export failed for an unknown reason.';
  const singleLine = error.message.replace(/[\r\n]+/g, ' ').slice(0, 500);
  return singleLine || 'The export failed without an error message.';
}

export class ExportJobCoordinator {
  private readonly jobs = new Map<string, ExportJobState>();
  private readonly controllers = new Map<string, AbortController>();
  private persistQueue = Promise.resolve();
  private recoveryWarningValue: string | null = null;
  private readonly now: () => Date;
  private readonly createId: () => string;

  constructor(private readonly options: ExportJobCoordinatorOptions) {
    this.now = options.now ?? (() => new Date());
    this.createId = options.createId ?? randomUUID;
  }

  get recoveryWarning(): string | null {
    return this.recoveryWarningValue;
  }

  async initialize(): Promise<void> {
    try {
      const persisted = JSON.parse(
        await readFile(this.options.statePath, 'utf8'),
      ) as PersistedJobState;
      if (persisted.formatVersion !== 1 || !Array.isArray(persisted.jobs)) {
        throw new Error('unsupported job-state format');
      }
      for (const job of persisted.jobs.slice(-MAX_RETAINED_JOBS)) {
        const recovered = copyJob(job);
        if (isActive(recovered)) {
          recovered.status = 'interrupted';
          recovered.error =
            'The service restarted before this export finished. Start a new export when ready.';
          recovered.updatedAt = this.now().toISOString();
          this.appendLog(recovered, 'Marked interrupted after service restart.');
        }
        this.jobs.set(recovered.id, recovered);
      }
      await this.persist();
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') return;
      this.recoveryWarningValue =
        'Previous job history could not be read; corpus artifacts were left untouched.';
    }
  }

  list(): ExportJobState[] {
    return [...this.jobs.values()].map(copyJob);
  }

  get(id: string): ExportJobState | null {
    const job = this.jobs.get(id);
    return job ? copyJob(job) : null;
  }

  latest(): ExportJobState | null {
    return this.list().at(-1) ?? null;
  }

  active(): ExportJobState | null {
    const job = [...this.jobs.values()].find(isActive);
    return job ? copyJob(job) : null;
  }

  async start(): Promise<ExportJobState> {
    const active = this.active();
    if (active) throw new DuplicateExportJobError(active);

    const timestamp = this.now().toISOString();
    const job: ExportJobState = {
      id: this.createId(),
      kind: 'corpus-export',
      status: 'queued',
      createdAt: timestamp,
      updatedAt: timestamp,
      progress: { phase: 'queued', processedRows: 0, totalRows: null },
      logs: ['Export queued.'],
      manifest: null,
      error: null,
    };
    this.jobs.set(job.id, job);
    this.trimJobs();
    await this.persist();
    void this.execute(job.id);
    return copyJob(job);
  }

  async cancel(id: string): Promise<ExportJobState | null> {
    const job = this.jobs.get(id);
    if (!job) return null;
    if (!isActive(job)) return copyJob(job);
    job.status = 'cancelling';
    job.updatedAt = this.now().toISOString();
    this.appendLog(job, 'Cancellation requested.');
    this.controllers.get(id)?.abort();
    await this.persist();
    return copyJob(job);
  }

  async flush(): Promise<void> {
    await this.persistQueue;
  }

  private async execute(id: string): Promise<void> {
    const job = this.jobs.get(id);
    if (!job) return;
    const controller = new AbortController();
    this.controllers.set(id, controller);
    job.status = 'running';
    job.updatedAt = this.now().toISOString();
    this.appendLog(job, 'Export started.');
    await this.persist();

    try {
      if (controller.signal.aborted) throw new ExportCancelledError();
      const result = await this.options.runExport({
        signal: controller.signal,
        onProgress: (progress) => {
          const current = this.jobs.get(id);
          if (!current || !isActive(current)) return;
          current.progress = progress;
          current.updatedAt = this.now().toISOString();
          void this.persist();
        },
      });
      job.status = 'succeeded';
      job.progress = {
        phase: 'complete',
        processedRows: result.manifest.emittedRows,
        totalRows: result.manifest.sourceRows,
      };
      job.manifest = result.manifest;
      job.error = null;
      this.appendLog(job, `Exported ${result.manifest.emittedRows} corpus rows.`);
    } catch (error) {
      if (controller.signal.aborted || error instanceof ExportCancelledError) {
        job.status = 'cancelled';
        job.error = null;
        this.appendLog(job, 'Export cancelled safely.');
      } else {
        job.status = 'failed';
        job.error = sanitizedError(error);
        this.appendLog(job, 'Export failed.');
      }
    } finally {
      job.updatedAt = this.now().toISOString();
      this.controllers.delete(id);
      await this.persist();
    }
  }

  private appendLog(job: ExportJobState, message: string): void {
    job.logs.push(`${this.now().toISOString()} ${message}`);
    if (job.logs.length > MAX_LOG_LINES) job.logs.splice(0, job.logs.length - MAX_LOG_LINES);
  }

  private trimJobs(): void {
    while (this.jobs.size > MAX_RETAINED_JOBS) {
      const oldest = this.jobs.keys().next().value as string | undefined;
      if (oldest === undefined) return;
      this.jobs.delete(oldest);
    }
  }

  private async persist(): Promise<void> {
    const state: PersistedJobState = { formatVersion: 1, jobs: this.list() };
    const contents = `${JSON.stringify(state, null, 2)}\n`;
    this.persistQueue = this.persistQueue.then(async () => {
      await mkdir(dirname(this.options.statePath), { recursive: true });
      const temporaryPath = `${this.options.statePath}.${process.pid}.tmp`;
      await writeFile(temporaryPath, contents, { mode: 0o600 });
      await rename(temporaryPath, this.options.statePath);
    });
    await this.persistQueue;
  }
}
