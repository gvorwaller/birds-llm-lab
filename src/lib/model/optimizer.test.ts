import { describe, expect, it } from 'vitest';
import { AdamWOptimizer, learningRateAtStep } from './optimizer';
import { ParameterRegistry, type ParameterSpec } from './parameters';

const specs: ParameterSpec[] = [
  { name: 'linear.weight', shape: [2], kind: 'weight', decay: true },
  { name: 'linear.bias', shape: [1], kind: 'bias', decay: false },
];

describe('AdamW optimizer', () => {
  it('matches a hand-computed clipped first update and excludes bias from decay', () => {
    const registry = new ParameterRegistry(specs, null);
    registry.setValues('linear.weight', [1, -2]);
    registry.setValues('linear.bias', [0.5]);
    registry.get('linear.weight').gradient.data.set([3, 4]);
    registry.get('linear.bias').gradient.data.set([0]);
    const optimizer = new AdamWOptimizer({
      beta1: 0.9,
      beta2: 0.95,
      epsilon: 1e-8,
      weightDecay: 0.1,
      gradientClipNorm: 1,
    });
    const result = optimizer.step(registry, 0.01);
    expect(result).toMatchObject({ step: 1, learningRate: 0.01, gradientNorm: 5, clipScale: 0.2 });
    expect(registry.get('linear.weight').value.data[0]).toBeCloseTo(0.989, 6);
    expect(registry.get('linear.weight').value.data[1]).toBeCloseTo(-2.008, 6);
    expect(registry.get('linear.bias').value.data[0]).toBe(0.5);
    const state = optimizer.snapshot();
    expect(state.step).toBe(1);
    expect(state.moments[0].first[0]).toBeCloseTo(0.06, 6);
    expect(state.moments[0].first[1]).toBeCloseTo(0.08, 6);
    expect(state.moments[0].second[0]).toBeCloseTo(0.018, 6);
    expect(state.moments[0].second[1]).toBeCloseTo(0.032, 6);
  });

  it('defines zero-based warmup, cosine boundary, and final-step behavior', () => {
    const schedule = { baseLearningRate: 1, warmupSteps: 2, totalSteps: 10 };
    expect(learningRateAtStep(0, schedule)).toBe(0.5);
    expect(learningRateAtStep(1, schedule)).toBe(1);
    expect(learningRateAtStep(2, schedule)).toBe(1);
    expect(learningRateAtStep(9, schedule)).toBe(0);
    expect(learningRateAtStep(5, schedule)).toBeCloseTo(
      0.5 * (1 + Math.cos((Math.PI * 3) / 7)),
      12,
    );
    expect(() => learningRateAtStep(10, schedule)).toThrow('from 0 to 9');
  });

  it('produces identical state and values for identical updates', () => {
    const makeRun = () => {
      const registry = new ParameterRegistry(specs, null);
      registry.setValues('linear.weight', [0.25, -0.75]);
      registry.setValues('linear.bias', [0.1]);
      const optimizer = new AdamWOptimizer({
        beta1: 0.9,
        beta2: 0.95,
        epsilon: 1e-8,
        weightDecay: 0.1,
        gradientClipNorm: 1,
      });
      for (let step = 0; step < 3; step += 1) {
        registry.get('linear.weight').gradient.data.set([0.2 + step, -0.4]);
        registry.get('linear.bias').gradient.data.set([0.05]);
        optimizer.step(registry, 1e-3);
      }
      return {
        values: registry.entries().map(({ value }) => Array.from(value.data)),
        state: optimizer.snapshot(),
      };
    };
    expect(makeRun()).toEqual(makeRun());
  });
});
