import { parseModelConfig } from '../data/schemas';
import { SeededRandom } from '../math/rng';
import { decodeCheckpointWeights, encodeCheckpointWeights } from '../model/checkpoint';
import { languageModelBackward, languageModelForward } from '../model/model';
import { AdamWOptimizer, learningRateAtStep } from '../model/optimizer';
import { initializeParameters, type ParameterRegistry } from '../model/parameters';
import { deterministicLanguageModelBatches, type LanguageModelBatch } from './batches';
import {
  LIVE_TRAINING_PROTOCOL_VERSION as VERSION,
  type LiveTrainingCheckpoint,
  type LiveTrainingCommand,
  type LiveTrainingConfig,
  type LiveTrainingReply,
  type LiveTrainingState,
} from './worker-protocol';

interface Run {
  readonly id: string;
  readonly config: LiveTrainingConfig;
  readonly trainingIdentity: string;
  readonly registry: ParameterRegistry;
  readonly optimizer: AdamWOptimizer;
  state: LiveTrainingState;
  step: number;
  epoch: number;
  batchIndex: number;
  batches: LanguageModelBatch[];
  lastCheckpointStep: number | null;
}

type PostReply = (reply: LiveTrainingReply, transfer?: Transferable[]) => void;

function positiveInteger(value: number, name: string): void {
  if (!Number.isSafeInteger(value) || value < 1)
    throw new Error(`${name} must be a positive integer.`);
}

function datasetHash(sequences: readonly (readonly number[])[]): string {
  // Stable 64-bit FNV-1a over the exact tokenized sequences; the checkpoint also
  // stores the full training configuration. This detects accidental data changes.
  let hash = 0xcbf2_9ce4_8422_2325n;
  for (const character of JSON.stringify(sequences)) {
    hash ^= BigInt(character.charCodeAt(0));
    hash = BigInt.asUintN(64, hash * 0x100_0000_01b3n);
  }
  return hash.toString(16).padStart(16, '0');
}

function validatedConfig(input: LiveTrainingConfig): {
  config: LiveTrainingConfig;
  identity: string;
} {
  if (!input || typeof input !== 'object') throw new Error('Training config is required.');
  const model = parseModelConfig(input.model);
  new SeededRandom(input.seed);
  positiveInteger(input.batchSize, 'Batch size');
  positiveInteger(input.totalSteps, 'Total steps');
  positiveInteger(input.displayEvery, 'Display cadence');
  if (!Number.isInteger(input.warmupSteps) || input.warmupSteps < 0) {
    throw new Error('Warmup steps must be a non-negative integer.');
  }
  learningRateAtStep(0, {
    baseLearningRate: input.baseLearningRate,
    warmupSteps: input.warmupSteps,
    totalSteps: input.totalSteps,
  });
  const optimizer = new AdamWOptimizer(input.optimizer);
  if (!Array.isArray(input.sequences) || input.sequences.length === 0) {
    throw new Error('Training requires tokenized sequences.');
  }
  const sequences = input.sequences.map((sequence, sequenceIndex) => {
    if (!Array.isArray(sequence)) throw new Error(`Sequence ${sequenceIndex} must be an array.`);
    return sequence.map((id, index) => {
      if (!Number.isInteger(id) || id < 0 || id >= model.vocabSize) {
        throw new Error(`Sequence ${sequenceIndex} has invalid token ID at ${index}.`);
      }
      return id;
    });
  });
  const batches = deterministicLanguageModelBatches(
    sequences,
    input.batchSize,
    model.contextLength,
    input.seed,
    0,
  );
  if (batches.length === 0) throw new Error('Training sequences contain no next-token examples.');
  const config: LiveTrainingConfig = {
    model,
    sequences,
    seed: input.seed,
    batchSize: input.batchSize,
    totalSteps: input.totalSteps,
    displayEvery: input.displayEvery,
    baseLearningRate: input.baseLearningRate,
    warmupSteps: input.warmupSteps,
    optimizer: { ...optimizer.hyperparameters },
  };
  const identity = JSON.stringify({
    model,
    seed: config.seed,
    batchSize: config.batchSize,
    totalSteps: config.totalSteps,
    baseLearningRate: config.baseLearningRate,
    warmupSteps: config.warmupSteps,
    optimizer: config.optimizer,
    datasetHash: datasetHash(sequences),
  });
  return { config, identity };
}

function batchesAt(config: LiveTrainingConfig, epoch: number): LanguageModelBatch[] {
  return deterministicLanguageModelBatches(
    config.sequences,
    config.batchSize,
    config.model.contextLength,
    config.seed,
    epoch,
  );
}

function restoreRun(
  checkpoint: LiveTrainingCheckpoint,
  config: LiveTrainingConfig,
  identity: string,
): Pick<Run, 'registry' | 'optimizer' | 'step' | 'epoch' | 'batchIndex' | 'batches'> {
  if (checkpoint.formatVersion !== 1 || checkpoint.trainingIdentity !== identity) {
    throw new Error('Checkpoint training configuration or token data does not match.');
  }
  if (
    !Number.isInteger(checkpoint.step) ||
    checkpoint.step < 0 ||
    checkpoint.step >= config.totalSteps
  ) {
    throw new Error('Checkpoint step is outside the training run.');
  }
  if (!Number.isInteger(checkpoint.epoch) || checkpoint.epoch < 0) {
    throw new Error('Checkpoint epoch is invalid.');
  }
  const batches = batchesAt(config, checkpoint.epoch);
  if (
    !Number.isInteger(checkpoint.batchIndex) ||
    checkpoint.batchIndex < 0 ||
    checkpoint.batchIndex >= batches.length
  ) {
    throw new Error('Checkpoint batch cursor is invalid.');
  }
  if (
    checkpoint.epoch !== Math.floor(checkpoint.step / batches.length) ||
    checkpoint.batchIndex !== checkpoint.step % batches.length
  ) {
    throw new Error('Checkpoint batch cursor does not match its training step.');
  }
  const registry = decodeCheckpointWeights(
    config.model,
    checkpoint.weightIndex,
    checkpoint.weightBytes,
  );
  const optimizer = new AdamWOptimizer(config.optimizer);
  optimizer.restore(checkpoint.optimizer, registry);
  if (optimizer.stepCount !== checkpoint.step) {
    throw new Error('Checkpoint optimizer step does not match training step.');
  }
  return {
    registry,
    optimizer,
    step: checkpoint.step,
    epoch: checkpoint.epoch,
    batchIndex: checkpoint.batchIndex,
    batches,
  };
}

/** Owns one worker's training state. Every batch is a separate event-loop task. */
export class LiveTrainingWorkerRuntime {
  private run: Run | null = null;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private singleStepPending = false;

  constructor(private readonly post: PostReply) {}

  handle(input: unknown): void {
    const candidate = input as Partial<LiveTrainingCommand> | null;
    const runId = typeof candidate?.runId === 'string' ? candidate.runId : 'unknown';
    if (!candidate || typeof candidate !== 'object' || runId.length === 0) {
      this.error(runId, 'invalid-message', 'A command with a run ID is required.', true);
      return;
    }
    if (candidate.version !== VERSION) {
      this.error(runId, 'unsupported-version', `Expected protocol version ${VERSION}.`, true);
      return;
    }
    if (candidate.type === 'start') {
      this.start(candidate as Extract<LiveTrainingCommand, { type: 'start' }>);
      return;
    }
    if (!this.run || this.run.id !== runId) {
      this.error(runId, 'invalid-state', 'No matching active training run.', true);
      return;
    }
    switch (candidate.type) {
      case 'pause':
        if (this.run.state !== 'running') return this.invalidState('Pause requires a running run.');
        this.clearTimer();
        this.run.state = 'paused';
        this.status();
        break;
      case 'resume':
        if (this.run.state !== 'paused' || this.singleStepPending) {
          return this.invalidState('Resume requires a paused run without a pending step.');
        }
        this.run.state = 'running';
        this.status();
        this.schedule();
        break;
      case 'step':
        if (this.run.state !== 'paused' || this.singleStepPending) {
          return this.invalidState('Step requires a paused run without a pending step.');
        }
        this.singleStepPending = true;
        this.schedule();
        break;
      case 'cancel':
        if (this.run.state !== 'running' && this.run.state !== 'paused') {
          return this.invalidState('Cancel requires a running or paused run.');
        }
        this.clearTimer();
        this.singleStepPending = false;
        this.run.state = 'cancelled';
        this.status();
        break;
      case 'checkpoint':
        this.checkpoint();
        break;
      default:
        this.error(runId, 'invalid-message', 'Unknown training command.', true);
    }
  }

  dispose(): void {
    this.clearTimer();
    this.run = null;
  }

  private start(command: Extract<LiveTrainingCommand, { type: 'start' }>): void {
    if (this.run?.state === 'running' || this.singleStepPending) {
      this.error(
        command.runId,
        'invalid-state',
        'Cancel the current run before starting another.',
        true,
      );
      return;
    }
    try {
      const { config, identity } = validatedConfig(command.config);
      const restored = command.resumeFrom
        ? restoreRun(command.resumeFrom, config, identity)
        : {
            registry: initializeParameters(config.model, config.seed),
            optimizer: new AdamWOptimizer(config.optimizer),
            step: 0,
            epoch: 0,
            batchIndex: 0,
            batches: batchesAt(config, 0),
          };
      this.clearTimer();
      this.run = {
        id: command.runId,
        config,
        trainingIdentity: identity,
        ...restored,
        state: command.startPaused ? 'paused' : 'running',
        lastCheckpointStep: command.resumeFrom?.step ?? null,
      };
      this.singleStepPending = false;
      this.status();
      if (this.run.state === 'running') this.schedule();
    } catch (cause) {
      this.error(
        command.runId,
        'invalid-config',
        cause instanceof Error ? cause.message : 'Invalid training configuration.',
        true,
      );
    }
  }

  private schedule(): void {
    if (this.timer !== null) return;
    this.timer = setTimeout(() => {
      this.timer = null;
      this.trainOneBatch();
    }, 0);
  }

  private clearTimer(): void {
    if (this.timer !== null) clearTimeout(this.timer);
    this.timer = null;
  }

  private trainOneBatch(): void {
    const run = this.run;
    if (!run || (run.state !== 'running' && !this.singleStepPending)) return;
    try {
      const batch = run.batches[run.batchIndex];
      const result = languageModelForward(
        batch.inputIds,
        [batch.batchSize, batch.sequenceLength],
        run.registry,
        run.config.model,
        batch.targetIds,
      );
      if (result.loss === null) throw new Error('Training loss was not computed.');
      languageModelBackward(result.cache);
      const learningRate = learningRateAtStep(run.step, {
        baseLearningRate: run.config.baseLearningRate,
        warmupSteps: run.config.warmupSteps,
        totalSteps: run.config.totalSteps,
      });
      const update = run.optimizer.step(run.registry, learningRate);
      run.step = update.step;
      run.batchIndex += 1;
      if (run.batchIndex === run.batches.length) {
        run.epoch += 1;
        run.batchIndex = 0;
        run.batches = batchesAt(run.config, run.epoch);
      }
      if (run.step % run.config.displayEvery === 0 || run.step === run.config.totalSteps) {
        this.post({
          version: VERSION,
          runId: run.id,
          type: 'progress',
          step: run.step,
          totalSteps: run.config.totalSteps,
          trainLoss: result.loss,
          predictionCount: batch.predictionCount,
          learningRate,
          gradientNorm: update.gradientNorm,
        });
      }
      if (run.step === run.config.totalSteps) {
        this.singleStepPending = false;
        run.state = 'completed';
        this.status();
      } else if (this.singleStepPending) {
        this.singleStepPending = false;
        this.status();
      } else if (run.state === 'running') {
        this.schedule();
      }
    } catch (cause) {
      this.singleStepPending = false;
      run.state = 'error';
      this.error(
        run.id,
        'training-failed',
        cause instanceof Error ? cause.message : 'Training failed.',
        false,
      );
      this.status();
    }
  }

  private checkpoint(): void {
    const run = this.run;
    if (!run || run.state === 'error') return this.invalidState('No checkpointable run exists.');
    if (
      run.lastCheckpointStep !== null &&
      run.step - run.lastCheckpointStep < run.config.displayEvery
    ) {
      this.error(
        run.id,
        'checkpoint-cadence',
        'Wait for the next display cadence before another checkpoint.',
        true,
      );
      return;
    }
    try {
      const weights = encodeCheckpointWeights(run.config.model, run.registry);
      const optimizer = run.optimizer.snapshot();
      const checkpoint: LiveTrainingCheckpoint = {
        formatVersion: 1,
        trainingIdentity: run.trainingIdentity,
        step: run.step,
        epoch: run.epoch,
        batchIndex: run.batchIndex,
        weightIndex: weights.index,
        weightBytes: weights.bytes,
        optimizer,
      };
      run.lastCheckpointStep = run.step;
      const transfer: Transferable[] = [weights.bytes.buffer as ArrayBuffer];
      for (const moment of optimizer.moments) {
        transfer.push(moment.first.buffer as ArrayBuffer, moment.second.buffer as ArrayBuffer);
      }
      this.post({ version: VERSION, runId: run.id, type: 'checkpoint', checkpoint }, transfer);
    } catch (cause) {
      this.error(
        run.id,
        'training-failed',
        cause instanceof Error ? cause.message : 'Checkpoint failed.',
        true,
      );
    }
  }

  private status(): void {
    const run = this.run;
    if (!run) return;
    this.post({
      version: VERSION,
      runId: run.id,
      type: 'status',
      state: run.state,
      step: run.step,
      totalSteps: run.config.totalSteps,
    });
  }

  private invalidState(message: string): void {
    this.error(this.run?.id ?? 'unknown', 'invalid-state', message, true);
  }

  private error(
    runId: string,
    code: Extract<LiveTrainingReply, { type: 'error' }>['code'],
    message: string,
    recoverable: boolean,
  ): void {
    this.post({ version: VERSION, runId, type: 'error', code, message, recoverable });
  }
}
