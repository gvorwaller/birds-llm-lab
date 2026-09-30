import { randomUUID } from 'node:crypto';
import { mkdir, readFile, rename, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import type { TrainingJobState, TrainingPresetId } from '../../src/lib/data/api-types';
import type { TrainingLogEntry } from '../../src/lib/data/schemas';
import { TrainingCancelledError } from '../training/trainer';

const ACTIVE_STATUSES = new Set<TrainingJobState['status']>(['queued', 'running', 'cancelling']);
const PRESET_STEPS: Record<TrainingPresetId, number> = { quick: 100, ready: 2_000 };
const MAX_RETAINED_JOBS = 10;
const MAX_LOG_LINES = 50;

export type TrainingRunner = (options: {
  jobId: string;
  preset: TrainingPresetId;
  signal: AbortSignal;
  onProgress: (entry: TrainingLogEntry) => void;
}) => Promise<{ checkpointId: string }>;

interface PersistedTrainingJobs {
  readonly formatVersion: 1;
  readonly jobs: TrainingJobState[];
}

export interface TrainingJobCoordinatorOptions {
  readonly statePath: string;
  readonly runTraining: TrainingRunner;
  readonly now?: () => Date;
  readonly createId?: () => string;
}

export class DuplicateTrainingJobError extends Error {
  constructor(readonly job: TrainingJobState) {
    super('A model training job is already running.');
    this.name = 'DuplicateTrainingJobError';
  }
}

function isPreset(value: string): value is TrainingPresetId {
  return value === 'quick' || value === 'ready';
}

function isActive(job: TrainingJobState): boolean {
  return ACTIVE_STATUSES.has(job.status);
}

function copy(job: TrainingJobState): TrainingJobState {
  return structuredClone(job);
}

function sanitizedError(error: unknown): string {
  if (error instanceof TrainingCancelledError) return error.message;
  if (!(error instanceof Error)) return 'Training failed for an unknown reason.';
  return error.message.replace(/[\r\n]+/g, ' ').slice(0, 500) || 'Training failed.';
}

export class TrainingJobCoordinator {
  private readonly jobs = new Map<string, TrainingJobState>();
  private readonly controllers = new Map<string, AbortController>();
  private readonly executions = new Map<string, Promise<void>>();
  private persistQueue = Promise.resolve();
  private recoveryWarningValue: string | null = null;
  private readonly now: () => Date;
  private readonly createId: () => string;

  constructor(private readonly options: TrainingJobCoordinatorOptions) {
    this.now = options.now ?? (() => new Date());
    this.createId = options.createId ?? randomUUID;
  }

  get recoveryWarning(): string | null {
    return this.recoveryWarningValue;
  }

  static totalSteps(preset: TrainingPresetId): number {
    return PRESET_STEPS[preset];
  }

  async initialize(): Promise<void> {
    try {
      const persisted = JSON.parse(
        await readFile(this.options.statePath, 'utf8'),
      ) as PersistedTrainingJobs;
      if (persisted.formatVersion !== 1 || !Array.isArray(persisted.jobs)) {
        throw new Error('unsupported training job-state format');
      }
      for (const value of persisted.jobs.slice(-MAX_RETAINED_JOBS)) {
        if (!isPreset(value.preset)) throw new Error('unsupported training preset');
        const job = copy(value);
        if (isActive(job)) {
          job.status = 'interrupted';
          job.error =
            'The service restarted before training finished. No partial checkpoint was made selectable.';
          job.updatedAt = this.now().toISOString();
          this.appendLog(job, 'Marked interrupted after service restart.');
        }
        this.jobs.set(job.id, job);
      }
      await this.persist();
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') return;
      this.recoveryWarningValue =
        'Previous training history could not be read; checkpoint artifacts were left untouched.';
    }
  }

  get(id: string): TrainingJobState | null {
    const job = this.jobs.get(id);
    return job ? copy(job) : null;
  }

  latest(): TrainingJobState | null {
    return [...this.jobs.values()].map(copy).at(-1) ?? null;
  }

  active(): TrainingJobState | null {
    const job = [...this.jobs.values()].find(isActive);
    return job ? copy(job) : null;
  }

  async start(presetInput: string): Promise<TrainingJobState> {
    if (!isPreset(presetInput)) throw new Error('Training preset must be quick or ready.');
    const active = this.active();
    if (active) throw new DuplicateTrainingJobError(active);
    const timestamp = this.now().toISOString();
    const job: TrainingJobState = {
      id: this.createId(),
      kind: 'model-training',
      preset: presetInput,
      status: 'queued',
      createdAt: timestamp,
      updatedAt: timestamp,
      progress: {
        step: 0,
        totalSteps: PRESET_STEPS[presetInput],
        trainLoss: null,
        validationLoss: null,
        learningRate: 0,
        elapsedMs: 0,
        estimatedRemainingMs: null,
        latestSample: null,
      },
      logs: ['Training queued.'],
      checkpointId: null,
      error: null,
    };
    this.jobs.set(job.id, job);
    this.trim();
    await this.persist();
    const execution = this.execute(job.id);
    this.executions.set(job.id, execution);
    void execution.then(
      () => this.executions.delete(job.id),
      () => this.executions.delete(job.id),
    );
    return copy(job);
  }

  async cancel(id: string): Promise<TrainingJobState | null> {
    const job = this.jobs.get(id);
    if (!job) return null;
    if (!isActive(job)) return copy(job);
    job.status = 'cancelling';
    job.updatedAt = this.now().toISOString();
    this.appendLog(job, 'Cancellation requested.');
    this.controllers.get(id)?.abort();
    await this.persist();
    return copy(job);
  }

  async shutdown(): Promise<void> {
    const activeIds = [...this.jobs.values()].filter(isActive).map(({ id }) => id);
    await Promise.all(activeIds.map((id) => this.cancel(id)));
    await Promise.all([...this.executions.values()]);
    await this.flush();
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
    this.appendLog(job, `${job.preset} preset started.`);
    await this.persist();
    try {
      if (controller.signal.aborted) throw new TrainingCancelledError();
      const result = await this.options.runTraining({
        jobId: job.id,
        preset: job.preset,
        signal: controller.signal,
        onProgress: (entry) => {
          const current = this.jobs.get(id);
          if (!current || !isActive(current)) return;
          current.progress.step = Math.max(current.progress.step, entry.step);
          current.progress.learningRate = entry.learningRate;
          current.progress.elapsedMs = entry.elapsedMs;
          if (entry.split === 'train') current.progress.trainLoss = entry.meanLoss;
          else {
            current.progress.validationLoss = entry.meanLoss;
            current.progress.latestSample = entry.sample;
          }
          current.progress.estimatedRemainingMs =
            entry.step > 0
              ? Math.max(
                  0,
                  Math.round(
                    (entry.elapsedMs / entry.step) * (current.progress.totalSteps - entry.step),
                  ),
                )
              : null;
          current.updatedAt = this.now().toISOString();
          if (entry.split === 'validation') {
            this.appendLog(
              current,
              `Validation loss ${entry.meanLoss.toFixed(6)} at step ${entry.step}.`,
            );
          }
          void this.persist();
        },
      });
      job.status = 'succeeded';
      job.checkpointId = result.checkpointId;
      job.error = null;
      job.progress.step = job.progress.totalSteps;
      job.progress.estimatedRemainingMs = 0;
      this.appendLog(job, `Checkpoint ${result.checkpointId} completed and validated.`);
    } catch (error) {
      if (controller.signal.aborted || error instanceof TrainingCancelledError) {
        job.status = 'cancelled';
        job.error = null;
        this.appendLog(job, 'Training cancelled safely; partial artifacts remain unavailable.');
      } else {
        job.status = 'failed';
        job.error = sanitizedError(error);
        this.appendLog(job, 'Training failed.');
      }
    } finally {
      job.updatedAt = this.now().toISOString();
      this.controllers.delete(id);
      await this.persist();
    }
  }

  private appendLog(job: TrainingJobState, message: string): void {
    job.logs.push(`${this.now().toISOString()} ${message}`);
    if (job.logs.length > MAX_LOG_LINES) job.logs.splice(0, job.logs.length - MAX_LOG_LINES);
  }

  private trim(): void {
    while (this.jobs.size > MAX_RETAINED_JOBS) {
      const oldest = this.jobs.keys().next().value as string | undefined;
      if (oldest === undefined) return;
      this.jobs.delete(oldest);
    }
  }

  private async persist(): Promise<void> {
    const value: PersistedTrainingJobs = {
      formatVersion: 1,
      jobs: [...this.jobs.values()].map(copy),
    };
    const contents = `${JSON.stringify(value, null, 2)}\n`;
    this.persistQueue = this.persistQueue.then(async () => {
      await mkdir(dirname(this.options.statePath), { recursive: true, mode: 0o700 });
      const temporary = `${this.options.statePath}.${process.pid}.tmp`;
      await writeFile(temporary, contents, { mode: 0o600 });
      await rename(temporary, this.options.statePath);
    });
    await this.persistQueue;
  }
}
