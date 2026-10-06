import { createHash } from 'node:crypto';
import { lstat, mkdir, readFile, readdir, rename, writeFile } from 'node:fs/promises';
import { basename, dirname, join, resolve } from 'node:path';
import type {
  CheckpointInspectionBundle,
  CheckpointList,
  CheckpointSummary,
} from '../../src/lib/data/api-types';
import { parseTokenizerArtifact } from '../../src/lib/data/schemas';
import { encodeCheckpointWeights } from '../../src/lib/model/checkpoint';
import { loadCheckpoint } from '../checkpoint-store';

const CHECKPOINT_ID = /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,99}$/;

interface SelectionFile {
  readonly formatVersion: 1;
  readonly activeCheckpointId: string | null;
}

function finalLoss(
  entries: Awaited<ReturnType<typeof loadCheckpoint>>['trainingLog']['entries'],
  split: 'train' | 'validation',
): number | null {
  return entries.filter((entry) => entry.split === split).at(-1)?.meanLoss ?? null;
}

export class CheckpointCatalog {
  private activeId: string | null = null;
  private recoveryWarningValue: string | null = null;

  constructor(
    private readonly checkpointDirectory: string,
    private readonly selectionPath: string,
  ) {}

  async initialize(): Promise<void> {
    let hadSelectionFile = false;
    try {
      const value = JSON.parse(
        await readFile(this.selectionPath, 'utf8'),
      ) as Partial<SelectionFile>;
      if (
        value.formatVersion !== 1 ||
        (value.activeCheckpointId !== null &&
          (typeof value.activeCheckpointId !== 'string' ||
            !CHECKPOINT_ID.test(value.activeCheckpointId)))
      ) {
        throw new Error('unsupported checkpoint selection format');
      }
      hadSelectionFile = true;
      this.activeId = value.activeCheckpointId;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'ENOENT') {
        this.recoveryWarningValue =
          'The saved checkpoint selection could not be read. No checkpoint was selected.';
      }
    }

    const list = await this.list();
    if (this.activeId !== null && !list.checkpoints.some(({ id }) => id === this.activeId)) {
      this.activeId = null;
      this.recoveryWarningValue =
        'The previously selected checkpoint is missing or invalid. Select another checkpoint.';
      await this.persist();
    } else if (!hadSelectionFile && this.activeId === null) {
      const ready = list.checkpoints.find(({ id }) => id === 'ready-v1');
      if (ready) {
        this.activeId = ready.id;
        await this.persist();
      }
    }
  }

  async list(): Promise<CheckpointList> {
    await mkdir(this.checkpointDirectory, { recursive: true, mode: 0o700 });
    const checkpoints: CheckpointSummary[] = [];
    let invalidCount = 0;
    for (const entry of await readdir(this.checkpointDirectory, { withFileTypes: true })) {
      if (!entry.isDirectory() || entry.name.startsWith('.') || !CHECKPOINT_ID.test(entry.name)) {
        continue;
      }
      const path = join(this.checkpointDirectory, entry.name);
      try {
        const [loaded, details] = await Promise.all([loadCheckpoint(path), lstat(path)]);
        checkpoints.push({
          id: entry.name,
          trainingStep: loaded.config.trainingStep,
          totalSteps: loaded.config.optimizer.totalSteps,
          corpusSha256: loaded.config.corpusSha256,
          tokenizerSha256: loaded.config.tokenizerSha256,
          sourceGitRevision: loaded.config.sourceGitRevision,
          finalTrainLoss: finalLoss(loaded.trainingLog.entries, 'train'),
          finalValidationLoss: finalLoss(loaded.trainingLog.entries, 'validation'),
          modifiedAt: details.mtime.toISOString(),
        });
      } catch {
        invalidCount += 1;
      }
    }
    checkpoints.sort(
      (left, right) =>
        right.modifiedAt.localeCompare(left.modifiedAt) || left.id.localeCompare(right.id),
    );
    const invalidWarning =
      invalidCount > 0
        ? `${invalidCount} invalid checkpoint ${invalidCount === 1 ? 'directory was' : 'directories were'} hidden.`
        : null;
    return {
      checkpoints,
      activeCheckpointId: this.activeId,
      recoveryWarning: this.recoveryWarningValue ?? invalidWarning,
    };
  }

  async select(idInput: string): Promise<CheckpointList> {
    const id = basename(idInput);
    if (id !== idInput || !CHECKPOINT_ID.test(id)) throw new Error('Invalid checkpoint id.');
    const list = await this.list();
    if (!list.checkpoints.some((checkpoint) => checkpoint.id === id)) {
      throw new Error('Checkpoint not found or failed validation.');
    }
    this.activeId = id;
    this.recoveryWarningValue = null;
    await this.persist();
    return this.list();
  }

  async inspectionBundle(tokenizerPath: string): Promise<CheckpointInspectionBundle | null> {
    const id = this.activeId;
    if (id === null) return null;
    const list = await this.list();
    if (!list.checkpoints.some((checkpoint) => checkpoint.id === id)) return null;
    const checkpoint = await loadCheckpoint(join(this.checkpointDirectory, id));
    const tokenizerBytes = await readFile(tokenizerPath);
    const tokenizerHash = createHash('sha256').update(tokenizerBytes).digest('hex');
    if (tokenizerHash !== checkpoint.config.tokenizerSha256) {
      throw new Error(
        'The active checkpoint tokenizer hash does not match the installed tokenizer.',
      );
    }
    const tokenizer = parseTokenizerArtifact(
      JSON.parse(tokenizerBytes.toString('utf8')) as unknown,
    );
    if (tokenizer.tokens.length !== checkpoint.config.model.vocabSize) {
      throw new Error(
        'The active checkpoint vocabulary size does not match the installed tokenizer.',
      );
    }
    const weights = encodeCheckpointWeights(checkpoint.config.model, checkpoint.parameters);
    return {
      checkpointId: id,
      config: checkpoint.config,
      weightIndex: weights.index,
      weightsBase64: Buffer.from(weights.bytes).toString('base64'),
      tokenizer,
    };
  }

  private async persist(): Promise<void> {
    await mkdir(dirname(this.selectionPath), { recursive: true, mode: 0o700 });
    const temporary = `${this.selectionPath}.${process.pid}.tmp`;
    const value: SelectionFile = { formatVersion: 1, activeCheckpointId: this.activeId };
    await writeFile(temporary, `${JSON.stringify(value, null, 2)}\n`, { mode: 0o600 });
    await rename(temporary, resolve(this.selectionPath));
  }
}
