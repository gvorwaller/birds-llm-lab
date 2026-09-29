export const DEFAULT_TRAINING_PRESET = Object.freeze({
  seed: 0x5eed,
  batchSize: 8,
  baseLearningRate: 0.003,
  warmupSteps: 100,
  totalSteps: 2_000,
  beta1: 0.9,
  beta2: 0.95,
  epsilon: 1e-8,
  weightDecay: 0.1,
  gradientClipNorm: 1,
});
