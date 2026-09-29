import type { ModelConfig } from '../data/schemas';
import { languageModelBackward, languageModelForward } from '../model/model';
import { AdamWOptimizer } from '../model/optimizer';
import { initializeParameters } from '../model/parameters';
import { DEFAULT_TRAINING_PRESET } from './config';

export const TINY_OVERFIT_LEARNING_RATE = DEFAULT_TRAINING_PRESET.baseLearningRate;
export const TINY_OVERFIT_STEPS = 250;

export const TINY_OVERFIT_CONFIG: ModelConfig = {
  formatVersion: 1,
  vocabSize: 259,
  contextLength: 4,
  dModel: 8,
  nLayers: 1,
  nHeads: 2,
  dHead: 4,
  dMlp: 16,
  tiedEmbeddings: true,
  useBias: true,
  layerNormEpsilon: 1e-5,
  gelu: 'tanh-approximation',
};

const TINY_INPUT = [256, 97, 98, 99] as const;
const TINY_TARGET = [97, 98, 99, 257] as const;

export interface TinyOverfitResult {
  readonly initialLoss: number;
  readonly finalLoss: number;
  readonly lossReduction: number;
  readonly predictions: number[];
  readonly targets: number[];
  readonly loggedLosses: Array<{ step: number; loss: number }>;
}

function greedyPredictions(logits: Float32Array, rows: number, vocabularySize: number): number[] {
  const predictions: number[] = [];
  for (let row = 0; row < rows; row += 1) {
    let selected = 0;
    for (let token = 1; token < vocabularySize; token += 1) {
      if (logits[row * vocabularySize + token] > logits[row * vocabularySize + selected]) {
        selected = token;
      }
    }
    predictions.push(selected);
  }
  return predictions;
}

export function runTinyOverfit(
  seed = 0x5eed,
  learningRate = TINY_OVERFIT_LEARNING_RATE,
  steps = TINY_OVERFIT_STEPS,
): TinyOverfitResult {
  const registry = initializeParameters(TINY_OVERFIT_CONFIG, seed);
  const optimizer = new AdamWOptimizer({
    beta1: 0.9,
    beta2: 0.95,
    epsilon: 1e-8,
    weightDecay: 0.1,
    gradientClipNorm: 1,
  });
  const initial = languageModelForward(
    TINY_INPUT,
    [1, TINY_OVERFIT_CONFIG.contextLength],
    registry,
    TINY_OVERFIT_CONFIG,
    TINY_TARGET,
  );
  if (initial.loss === null) throw new Error('Tiny overfit initial loss was not computed.');
  const loggedLosses = [{ step: 0, loss: initial.loss }];
  for (let step = 0; step < steps; step += 1) {
    const forward = languageModelForward(
      TINY_INPUT,
      [1, TINY_OVERFIT_CONFIG.contextLength],
      registry,
      TINY_OVERFIT_CONFIG,
      TINY_TARGET,
    );
    languageModelBackward(forward.cache);
    optimizer.step(registry, learningRate);
    if ((step + 1) % 50 === 0 && forward.loss !== null) {
      loggedLosses.push({ step: step + 1, loss: forward.loss });
    }
  }
  const final = languageModelForward(
    TINY_INPUT,
    [1, TINY_OVERFIT_CONFIG.contextLength],
    registry,
    TINY_OVERFIT_CONFIG,
    TINY_TARGET,
  );
  if (final.loss === null) throw new Error('Tiny overfit final loss was not computed.');
  return {
    initialLoss: initial.loss,
    finalLoss: final.loss,
    lossReduction: (initial.loss - final.loss) / initial.loss,
    predictions: greedyPredictions(
      final.logits.data,
      TINY_OVERFIT_CONFIG.contextLength,
      TINY_OVERFIT_CONFIG.vocabSize,
    ),
    targets: [...TINY_TARGET],
    loggedLosses,
  };
}
