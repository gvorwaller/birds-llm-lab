import { describe, expect, it } from 'vitest';
import { DEFAULT_TRAINING_PRESET } from './config';
import { runTinyOverfit, TINY_OVERFIT_LEARNING_RATE, TINY_OVERFIT_STEPS } from './learning-gates';

describe('M2 learning gate', () => {
  it('uses the selected default rate to overfit the fixed batch by at least 80 percent', () => {
    const result = runTinyOverfit();
    expect(TINY_OVERFIT_LEARNING_RATE).toBe(DEFAULT_TRAINING_PRESET.baseLearningRate);
    expect(TINY_OVERFIT_STEPS).toBe(250);
    expect(result.lossReduction).toBeGreaterThanOrEqual(0.8);
    expect(result.predictions).toEqual(result.targets);
    for (let index = 1; index < result.loggedLosses.length; index += 1) {
      expect(result.loggedLosses[index].loss).toBeLessThan(result.loggedLosses[index - 1].loss);
    }
  });
});
