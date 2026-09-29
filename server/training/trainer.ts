import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { performance } from 'node:perf_hooks';
import { execFileSync } from 'node:child_process';
import { join, resolve } from 'node:path';
import {
  parseTokenizerArtifact,
  type CheckpointConfig,
  type ModelConfig,
  type TokenizerArtifact,
  type TrainingLogArtifact,
  type TrainingLogEntry,
} from '../../src/lib/data/schemas';
import { sampleCategorical } from '../../src/lib/generation/sampling';
import { SeededRandom } from '../../src/lib/math/rng';
import { DEFAULT_MODEL_CONFIG } from '../../src/lib/model/config';
import { languageModelBackward, languageModelForward } from '../../src/lib/model/model';
import { AdamWOptimizer, learningRateAtStep } from '../../src/lib/model/optimizer';
import { initializeParameters, type ParameterRegistry } from '../../src/lib/model/parameters';
import { decodeText, encodeText } from '../../src/lib/tokenizer/bpe';
import { BOS_ID, EOS_ID } from '../../src/lib/tokenizer/constants';
import { DEFAULT_TRAINING_PRESET } from '../../src/lib/training/config';
import { loadCheckpoint, writeCheckpoint } from '../checkpoint-store';
import type { TrainerResumeState } from '../trainer-state';
import { DeterministicBatchStream, StreamedTokenDataset } from './token-dataset';

export interface TrainerOptions {
  readonly corpusPath?: string;
  readonly manifestPath?: string;
  readonly tokenizerPath?: string;
  readonly cacheRoot?: string;
  readonly outputRoot?: string;
  readonly resumeFrom?: string;
  readonly totalSteps?: number;
  readonly stopAfterStep?: number;
  readonly checkpointEvery?: number;
  readonly logEvery?: number;
  readonly validateEvery?: number;
  readonly validationBatches?: number;
  readonly model?: ModelConfig;
  readonly seed?: number;
  readonly batchSize?: number;
  readonly sourceGitRevision?: string | null;
  readonly onProgress?: (entry: TrainingLogEntry) => void;
}

export interface TrainerResult {
  readonly checkpointPath: string;
  readonly config: CheckpointConfig;
  readonly trainingLog: TrainingLogArtifact;
}

interface Evaluation {
  readonly meanLoss: number;
  readonly predictionCount: number;
}

function positiveInteger(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value <= 0) throw new Error(`${name} must be positive.`);
  return value;
}

function currentGitRevision(): string | null {
  try {
    const dirty = execFileSync('git', ['status', '--porcelain', '--untracked-files=no'], {
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
    });
    if (dirty.trim().length > 0) return null;
    const revision = execFileSync('git', ['rev-parse', 'HEAD'], {
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
    }).trim();
    return /^[a-f0-9]{40}$/.test(revision) ? revision : null;
  } catch {
    return null;
  }
}

async function loadTokenizer(pathInput: string): Promise<{
  artifact: TokenizerArtifact;
  sha256: string;
}> {
  const bytes = await readFile(resolve(pathInput));
  return {
    artifact: parseTokenizerArtifact(JSON.parse(bytes.toString('utf8')) as unknown),
    sha256: createHash('sha256').update(bytes).digest('hex'),
  };
}

async function evaluate(
  dataset: StreamedTokenDataset,
  registry: ParameterRegistry,
  config: ModelConfig,
  split: 'train' | 'validation',
  batchSize: number,
  batches: number,
  seed: number,
): Promise<Evaluation> {
  const stream = new DeterministicBatchStream(
    dataset,
    dataset.windows(split, config.contextLength),
    batchSize,
    config.contextLength,
    seed,
  );
  let weightedLoss = 0;
  let predictionCount = 0;
  for (let index = 0; index < batches; index += 1) {
    const batch = await stream.next();
    const result = languageModelForward(
      batch.inputIds,
      [batch.batchSize, batch.sequenceLength],
      registry,
      config,
      batch.targetIds,
    );
    if (result.loss === null) throw new Error('Evaluation loss was not computed.');
    weightedLoss += result.loss * batch.predictionCount;
    predictionCount += batch.predictionCount;
  }
  return { meanLoss: weightedLoss / predictionCount, predictionCount };
}

export function generateTextSample(
  registry: ParameterRegistry,
  config: ModelConfig,
  tokenizer: TokenizerArtifact,
  random: SeededRandom,
  maximumNewTokens = 48,
): string {
  const ids = [BOS_ID, ...encodeText('Common name:', tokenizer)];
  for (let generated = 0; generated < maximumNewTokens; generated += 1) {
    const context = ids.slice(-config.contextLength);
    const result = languageModelForward(context, [1, context.length], registry, config);
    const start = (context.length - 1) * config.vocabSize;
    const selected = sampleCategorical(
      result.logits.data.subarray(start, start + config.vocabSize),
      random,
      0.8,
    );
    ids.push(selected);
    if (selected === EOS_ID) break;
  }
  return decodeText(ids, tokenizer);
}

function checkpointConfig(
  model: ModelConfig,
  seed: number,
  totalSteps: number,
  trainingStep: number,
  corpusSha256: string,
  tokenizerSha256: string,
  sourceGitRevision: string | null,
): CheckpointConfig {
  return {
    formatVersion: 1,
    model,
    optimizer: {
      name: 'adamw',
      beta1: DEFAULT_TRAINING_PRESET.beta1,
      beta2: DEFAULT_TRAINING_PRESET.beta2,
      epsilon: DEFAULT_TRAINING_PRESET.epsilon,
      weightDecay: DEFAULT_TRAINING_PRESET.weightDecay,
      gradientClipNorm: DEFAULT_TRAINING_PRESET.gradientClipNorm,
      learningRate: DEFAULT_TRAINING_PRESET.baseLearningRate,
      warmupSteps: Math.min(DEFAULT_TRAINING_PRESET.warmupSteps, totalSteps - 1),
      totalSteps,
    },
    seed,
    corpusSha256,
    tokenizerSha256,
    trainingStep,
    sourceGitRevision,
  };
}

export async function trainReadyCheckpoint(options: TrainerOptions = {}): Promise<TrainerResult> {
  const tokenizerPath = options.tokenizerPath ?? 'src/assets/tokenizer-1024.json';
  const outputRoot = resolve(options.outputRoot ?? 'checkpoints');
  const totalSteps = positiveInteger(
    options.totalSteps ?? DEFAULT_TRAINING_PRESET.totalSteps,
    'Total steps',
  );
  const stopAfterStep = positiveInteger(options.stopAfterStep ?? totalSteps, 'Stop-after step');
  if (stopAfterStep > totalSteps) throw new Error('Stop-after step cannot exceed total steps.');
  const checkpointEvery = positiveInteger(options.checkpointEvery ?? 500, 'Checkpoint interval');
  const logEvery = positiveInteger(options.logEvery ?? 25, 'Log interval');
  const validateEvery = positiveInteger(options.validateEvery ?? 100, 'Validation interval');
  const validationBatches = positiveInteger(options.validationBatches ?? 8, 'Validation batches');
  const batchSize = positiveInteger(
    options.batchSize ?? DEFAULT_TRAINING_PRESET.batchSize,
    'Batch size',
  );
  const seed = options.seed ?? DEFAULT_TRAINING_PRESET.seed;
  const model = options.model ?? DEFAULT_MODEL_CONFIG;
  const dataset = await StreamedTokenDataset.open({
    corpusPath: options.corpusPath,
    manifestPath: options.manifestPath,
    tokenizerPath,
    cacheRoot: options.cacheRoot,
  });
  try {
    const { artifact: tokenizer, sha256: tokenizerSha256 } = await loadTokenizer(tokenizerPath);
    if (model.vocabSize !== tokenizer.targetSize) {
      throw new Error(
        `Model vocabulary ${model.vocabSize} does not match tokenizer size ${tokenizer.targetSize}.`,
      );
    }
    let registry: ParameterRegistry;
    let optimizer: AdamWOptimizer;
    let samplingRandom: SeededRandom;
    let trainingLog: TrainingLogArtifact;
    let startStep: number;
    let resumeState: TrainerResumeState['dataOrder'] = { epoch: 0, cursor: 0 };
    let sourceGitRevision =
      options.sourceGitRevision === undefined ? currentGitRevision() : options.sourceGitRevision;

    if (options.resumeFrom) {
      const loaded = await loadCheckpoint(options.resumeFrom);
      if (loaded.resumeState === null)
        throw new Error('Checkpoint has no resumable trainer state.');
      if (
        loaded.config.corpusSha256 !== dataset.index.corpusSha256 ||
        loaded.config.tokenizerSha256 !== tokenizerSha256
      ) {
        throw new Error('Resume checkpoint corpus or tokenizer identity does not match.');
      }
      if (JSON.stringify(loaded.config.model) !== JSON.stringify(model)) {
        throw new Error('Resume checkpoint model configuration does not match.');
      }
      if (loaded.config.optimizer.totalSteps !== totalSteps || loaded.config.seed !== seed) {
        throw new Error('Resume checkpoint schedule or seed does not match.');
      }
      registry = loaded.parameters;
      optimizer = new AdamWOptimizer(loaded.config.optimizer);
      optimizer.restore(loaded.resumeState.optimizer, registry);
      samplingRandom = new SeededRandom(loaded.resumeState.samplingRandom);
      resumeState = loaded.resumeState.dataOrder;
      trainingLog = loaded.trainingLog;
      startStep = loaded.config.trainingStep;
      sourceGitRevision = loaded.config.sourceGitRevision;
    } else {
      registry = initializeParameters(model, seed);
      optimizer = new AdamWOptimizer({
        beta1: DEFAULT_TRAINING_PRESET.beta1,
        beta2: DEFAULT_TRAINING_PRESET.beta2,
        epsilon: DEFAULT_TRAINING_PRESET.epsilon,
        weightDecay: DEFAULT_TRAINING_PRESET.weightDecay,
        gradientClipNorm: DEFAULT_TRAINING_PRESET.gradientClipNorm,
      });
      samplingRandom = new SeededRandom(seed ^ 0xa511_e9b3);
      trainingLog = { formatVersion: 1, entries: [] };
      startStep = 0;
    }
    if (stopAfterStep <= startStep) {
      throw new Error(`Stop-after step ${stopAfterStep} must be after resume step ${startStep}.`);
    }

    const trainingStream = new DeterministicBatchStream(
      dataset,
      dataset.windows('train', model.contextLength),
      batchSize,
      model.contextLength,
      seed,
      resumeState,
    );
    const startedAt = performance.now();
    const append = (entry: TrainingLogEntry) => {
      trainingLog.entries.push(entry);
      options.onProgress?.(entry);
    };
    if (startStep === 0) {
      for (const split of ['train', 'validation'] as const) {
        const result = await evaluate(
          dataset,
          registry,
          model,
          split,
          batchSize,
          validationBatches,
          seed ^ (split === 'train' ? 0x1234 : 0x5678),
        );
        append({
          step: 0,
          split,
          predictionCount: result.predictionCount,
          meanLoss: result.meanLoss,
          perplexity: Math.exp(result.meanLoss),
          learningRate: 0,
          gradientNorm: 0,
          elapsedMs: Math.round(performance.now() - startedAt),
          sample:
            split === 'validation'
              ? generateTextSample(registry, model, tokenizer, samplingRandom)
              : null,
        });
      }
    }

    let accumulatedLoss = 0;
    let accumulatedPredictions = 0;
    let lastGradientNorm = 0;
    let lastLearningRate = 0;
    for (let step = startStep; step < stopAfterStep; step += 1) {
      const batch = await trainingStream.next();
      const forward = languageModelForward(
        batch.inputIds,
        [batch.batchSize, batch.sequenceLength],
        registry,
        model,
        batch.targetIds,
      );
      if (forward.loss === null) throw new Error('Training loss was not computed.');
      languageModelBackward(forward.cache);
      lastLearningRate = learningRateAtStep(step, {
        baseLearningRate: DEFAULT_TRAINING_PRESET.baseLearningRate,
        warmupSteps: Math.min(DEFAULT_TRAINING_PRESET.warmupSteps, totalSteps - 1),
        totalSteps,
      });
      const update = optimizer.step(registry, lastLearningRate);
      lastGradientNorm = update.gradientNorm;
      accumulatedLoss += forward.loss * batch.predictionCount;
      accumulatedPredictions += batch.predictionCount;
      const completedStep = step + 1;

      if (completedStep % logEvery === 0 || completedStep === stopAfterStep) {
        const meanLoss = accumulatedLoss / accumulatedPredictions;
        append({
          step: completedStep,
          split: 'train',
          predictionCount: accumulatedPredictions,
          meanLoss,
          perplexity: Math.exp(meanLoss),
          learningRate: lastLearningRate,
          gradientNorm: lastGradientNorm,
          elapsedMs: Math.round(performance.now() - startedAt),
          sample: null,
        });
        accumulatedLoss = 0;
        accumulatedPredictions = 0;
      }
      if (completedStep % validateEvery === 0 || completedStep === totalSteps) {
        const result = await evaluate(
          dataset,
          registry,
          model,
          'validation',
          batchSize,
          validationBatches,
          seed ^ 0x5678,
        );
        append({
          step: completedStep,
          split: 'validation',
          predictionCount: result.predictionCount,
          meanLoss: result.meanLoss,
          perplexity: Math.exp(result.meanLoss),
          learningRate: lastLearningRate,
          gradientNorm: 0,
          elapsedMs: Math.round(performance.now() - startedAt),
          sample: generateTextSample(registry, model, tokenizer, samplingRandom),
        });
      }

      if (
        (completedStep % checkpointEvery === 0 && completedStep < stopAfterStep) ||
        completedStep === stopAfterStep
      ) {
        const config = checkpointConfig(
          model,
          seed,
          totalSteps,
          completedStep,
          dataset.index.corpusSha256,
          tokenizerSha256,
          sourceGitRevision,
        );
        const state: TrainerResumeState = {
          optimizer: optimizer.snapshot(),
          samplingRandom: samplingRandom.snapshot(),
          dataOrder: trainingStream.snapshot(),
        };
        const name =
          completedStep === totalSteps
            ? 'ready-v1'
            : `step-${completedStep.toString().padStart(6, '0')}`;
        const checkpointPath = join(outputRoot, name);
        await writeCheckpoint(checkpointPath, config, registry, trainingLog, state);
        if (completedStep === stopAfterStep) {
          return { checkpointPath, config, trainingLog };
        }
      }
    }
    throw new Error('Training ended without writing a checkpoint.');
  } finally {
    await dataset.close();
  }
}
