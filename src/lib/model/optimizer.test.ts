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

  it('recomputes every displayed scalar stage independently on a resumed moment update', () => {
    const hyperparameters = {
      beta1: 0.9,
      beta2: 0.95,
      epsilon: 1e-8,
      weightDecay: 0.1,
      gradientClipNorm: 1,
    };
    const registry = new ParameterRegistry(specs, null);
    registry.setValues('linear.weight', [1, -2]);
    registry.setValues('linear.bias', [0.5]);
    const optimizer = new AdamWOptimizer(hyperparameters);
    registry.get('linear.weight').gradient.data.set([0.2, -0.1]);
    registry.get('linear.bias').gradient.data.set([0.05]);
    optimizer.step(registry, 0.01);

    const before = registry.get('linear.weight').value.data[0];
    const prior = optimizer.snapshot().moments.find((moment) => moment.name === 'linear.weight');
    if (!prior) throw new Error('Missing prior weight moments.');
    const firstBefore = prior.first[0];
    const secondBefore = prior.second[0];
    registry.get('linear.weight').gradient.data.set([3, -4]);
    registry.get('linear.bias').gradient.data.set([0.5]);
    const raw = 3;
    const clipScale = 1 / Math.sqrt(3 ** 2 + (-4) ** 2 + 0.5 ** 2);
    const clipped = raw * clipScale;
    const first = Math.fround(0.9 * firstBefore + 0.1 * clipped);
    const second = Math.fround(0.95 * secondBefore + 0.05 * clipped * clipped);
    const firstCorrection = 1 - 0.9 ** 2;
    const secondCorrection = 1 - 0.95 ** 2;
    const firstEstimate = first / firstCorrection;
    const secondEstimate = second / secondCorrection;
    const adaptive = firstEstimate / (Math.sqrt(secondEstimate) + 1e-8);
    const decay = 0.1 * before;
    const expected = Math.fround(before - 0.01 * (adaptive + decay));

    const trace = optimizer.step(registry, 0.01, { name: 'linear.weight', index: 0 }).scalarTrace;
    if (!trace) throw new Error('Missing selected scalar trace.');
    expect(trace.name).toBe('linear.weight');
    expect(trace.index).toBe(0);
    expect(trace.step).toBe(2);
    expect(trace.valueBefore).toBe(before);
    expect(trace.rawGradient).toBe(raw);
    expect(trace.clipScale).toBeCloseTo(clipScale, 12);
    expect(trace.clippedGradient).toBeCloseTo(clipped, 12);
    expect(trace.firstMomentBefore).toBe(firstBefore);
    expect(trace.secondMomentBefore).toBe(secondBefore);
    expect(trace.firstMoment).toBe(first);
    expect(trace.secondMoment).toBe(second);
    expect(trace.firstBiasCorrection).toBe(firstCorrection);
    expect(trace.secondBiasCorrection).toBe(secondCorrection);
    expect(trace.firstEstimate).toBeCloseTo(firstEstimate, 12);
    expect(trace.secondEstimate).toBeCloseTo(secondEstimate, 12);
    expect(trace.adaptiveTerm).toBeCloseTo(adaptive, 12);
    expect(trace.decayTerm).toBe(decay);
    expect(trace.learningRate).toBe(0.01);
    expect(trace.valueAfter).toBe(expected);
    expect(registry.get('linear.weight').value.data[0]).toBe(expected);
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

  it('restores moments and step count for an identical resumed update', () => {
    const createRegistry = () => {
      const registry = new ParameterRegistry(specs, null);
      registry.setValues('linear.weight', [0.25, -0.75]);
      registry.setValues('linear.bias', [0.1]);
      return registry;
    };
    const hyperparameters = {
      beta1: 0.9,
      beta2: 0.95,
      epsilon: 1e-8,
      weightDecay: 0.1,
      gradientClipNorm: 1,
    };
    const uninterruptedRegistry = createRegistry();
    const uninterrupted = new AdamWOptimizer(hyperparameters);
    uninterruptedRegistry.get('linear.weight').gradient.data.set([0.2, -0.4]);
    uninterruptedRegistry.get('linear.bias').gradient.data.set([0.05]);
    uninterrupted.step(uninterruptedRegistry, 1e-3);

    const resumedRegistry = createRegistry();
    for (const parameter of uninterruptedRegistry) {
      resumedRegistry.setValues(parameter.name, parameter.value.data);
    }
    const resumed = new AdamWOptimizer(hyperparameters);
    resumed.restore(uninterrupted.snapshot(), resumedRegistry);

    for (const registry of [uninterruptedRegistry, resumedRegistry]) {
      registry.get('linear.weight').gradient.data.set([1.2, 0.3]);
      registry.get('linear.bias').gradient.data.set([-0.1]);
    }
    uninterrupted.step(uninterruptedRegistry, 8e-4);
    resumed.step(resumedRegistry, 8e-4);
    expect(resumed.snapshot()).toEqual(uninterrupted.snapshot());
    expect(resumedRegistry.entries().map(({ value }) => Array.from(value.data))).toEqual(
      uninterruptedRegistry.entries().map(({ value }) => Array.from(value.data)),
    );
  });
});
