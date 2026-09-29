import { performance } from 'node:perf_hooks';
import { arch, cpus, platform } from 'node:os';
import { DEFAULT_MODEL_CONFIG } from '../src/lib/model/config';
import { languageModelBackward, languageModelForward } from '../src/lib/model/model';
import { AdamWOptimizer } from '../src/lib/model/optimizer';
import { initializeParameters } from '../src/lib/model/parameters';
import { BOS_ID, EOS_ID } from '../src/lib/tokenizer/constants';
import { DEFAULT_TRAINING_PRESET } from '../src/lib/training/config';

const BATCH_SIZE = DEFAULT_TRAINING_PRESET.batchSize;
const CONTEXT_LENGTH = DEFAULT_MODEL_CONFIG.contextLength;
const WARMUP_STEPS = 2;
const MEASURED_STEPS = 7;
const READY_TRAINING_STEPS = DEFAULT_TRAINING_PRESET.totalSteps;
const MAX_READY_MILLISECONDS = 15 * 60 * 1_000;

const inputIds = Array.from({ length: BATCH_SIZE * CONTEXT_LENGTH }, (_, index) =>
  index % CONTEXT_LENGTH === 0 ? BOS_ID : 97 + (index % 23),
);
const targetIds = inputIds.map((_, index) =>
  index % CONTEXT_LENGTH === CONTEXT_LENGTH - 1 ? EOS_ID : inputIds[index + 1],
);
const registry = initializeParameters(DEFAULT_MODEL_CONFIG, 0x5eed);
const optimizer = new AdamWOptimizer({
  beta1: DEFAULT_TRAINING_PRESET.beta1,
  beta2: DEFAULT_TRAINING_PRESET.beta2,
  epsilon: DEFAULT_TRAINING_PRESET.epsilon,
  weightDecay: DEFAULT_TRAINING_PRESET.weightDecay,
  gradientClipNorm: DEFAULT_TRAINING_PRESET.gradientClipNorm,
});

interface Timing {
  forwardMs: number;
  backwardMs: number;
  updateMs: number;
  totalMs: number;
}

function measuredStep(): Timing {
  const start = performance.now();
  const forward = languageModelForward(
    inputIds,
    [BATCH_SIZE, CONTEXT_LENGTH],
    registry,
    DEFAULT_MODEL_CONFIG,
    targetIds,
  );
  const afterForward = performance.now();
  languageModelBackward(forward.cache);
  const afterBackward = performance.now();
  optimizer.step(registry, DEFAULT_TRAINING_PRESET.baseLearningRate);
  const afterUpdate = performance.now();
  return {
    forwardMs: afterForward - start,
    backwardMs: afterBackward - afterForward,
    updateMs: afterUpdate - afterBackward,
    totalMs: afterUpdate - start,
  };
}

for (let step = 0; step < WARMUP_STEPS; step += 1) measuredStep();
const timings = Array.from({ length: MEASURED_STEPS }, measuredStep);
const mean = (key: keyof Timing) =>
  timings.reduce((total, timing) => total + timing[key], 0) / timings.length;
const meanStepMs = mean('totalMs');
const estimatedReadyMs = meanStepMs * READY_TRAINING_STEPS;
const maximumStepMs = Math.max(...timings.map(({ totalMs }) => totalMs));
const conservativeEstimatedReadyMs = maximumStepMs * READY_TRAINING_STEPS;
const result = {
  status: conservativeEstimatedReadyMs <= MAX_READY_MILLISECONDS ? 'passed' : 'profiling-required',
  runtime: process.version,
  platform: platform(),
  architecture: arch(),
  cpuModel: cpus()[0]?.model ?? 'unknown',
  parameterCount: registry.elementCount,
  batchSize: BATCH_SIZE,
  contextLength: CONTEXT_LENGTH,
  tokensPerStep: BATCH_SIZE * CONTEXT_LENGTH,
  warmupSteps: WARMUP_STEPS,
  measuredSteps: MEASURED_STEPS,
  readyTrainingSteps: READY_TRAINING_STEPS,
  readyTrainingTokens: READY_TRAINING_STEPS * BATCH_SIZE * CONTEXT_LENGTH,
  meanForwardMs: mean('forwardMs'),
  meanBackwardMs: mean('backwardMs'),
  meanUpdateMs: mean('updateMs'),
  meanStepMs,
  maximumStepMs,
  meanTokensPerSecond: (BATCH_SIZE * CONTEXT_LENGTH * 1_000) / meanStepMs,
  estimatedReadySeconds: estimatedReadyMs / 1_000,
  conservativeEstimatedReadySeconds: conservativeEstimatedReadyMs / 1_000,
  limitSeconds: MAX_READY_MILLISECONDS / 1_000,
};
console.log(JSON.stringify(result, null, 2));
if (conservativeEstimatedReadyMs > MAX_READY_MILLISECONDS) process.exitCode = 2;
