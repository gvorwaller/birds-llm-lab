import { describe, expect, it } from 'vitest';
import { drawDistribution, transformDistribution } from './distribution';

const noFilters = { temperature: 1, topK: 0, topP: 1 };

describe('distribution transforms', () => {
  it('applies temperature, softmax, top-k, top-p, then renormalizes', () => {
    const result = transformDistribution([0, Math.log(2), Math.log(4), Math.log(8)], {
      temperature: 1,
      topK: 3,
      topP: 0.7,
    });
    for (const [id, expected] of [1 / 15, 2 / 15, 4 / 15, 8 / 15].entries()) {
      expect(result.beforeFilters[id]).toBeCloseTo(expected);
    }
    for (const [id, expected] of [0, 2 / 15, 4 / 15, 8 / 15].entries()) {
      expect(result.afterTopK[id]).toBeCloseTo(expected);
    }
    for (const [id, expected] of [0, 0, 4 / 15, 8 / 15].entries()) {
      expect(result.afterTopP[id]).toBeCloseTo(expected);
    }
    for (const [id, expected] of [0, 0, 1 / 3, 2 / 3].entries()) {
      expect(result.probabilities[id]).toBeCloseTo(expected);
    }
    for (const [id, expected] of [0, 0, 1 / 3, 1].entries()) {
      expect(result.cumulative[id]).toBeCloseTo(expected);
    }
    expect(result.retainedMass).toBeCloseTo(12 / 15);
    expect(result.removedMass).toBeCloseTo(3 / 15);
    expect(result.probabilities.reduce((sum, probability) => sum + probability, 0)).toBeCloseTo(1);
  });

  it('uses token id for ties and retains one token for tiny top-p', () => {
    const result = transformDistribution([1, 1, 1], {
      temperature: 1,
      topK: 2,
      topP: Number.MIN_VALUE,
    });
    expect(result.rankOrder).toEqual([0, 1, 2]);
    expect(result.probabilities).toEqual([1, 0, 0]);
  });

  it('keeps stable mass for extreme finite logits and temperature', () => {
    const result = transformDistribution([1e300, -1e300], {
      temperature: Number.MIN_VALUE,
      topK: 0,
      topP: 1,
    });
    expect(result.probabilities).toEqual([1, 0]);
    expect(result.cumulative).toEqual([1, 1]);
    expect(transformDistribution([0, 0], noFilters).probabilities).toEqual([0.5, 0.5]);
  });

  it('rejects invalid logits and controls', () => {
    expect(() => transformDistribution([], noFilters)).toThrow('at least one');
    expect(() => transformDistribution([Number.NaN], noFilters)).toThrow('finite');
    expect(() => transformDistribution([0], { ...noFilters, temperature: 0 })).toThrow('positive');
    expect(() => transformDistribution([0], { ...noFilters, topK: 1.5 })).toThrow('integer');
    expect(() => transformDistribution([0], { ...noFilters, topP: 0 })).toThrow('(0, 1]');
    expect(() => transformDistribution([0], { ...noFilters, topP: 1.1 })).toThrow('(0, 1]');
  });
});

describe('seeded draw trace', () => {
  it('returns a reproducible random number, interval, token, and next state', () => {
    const distribution = transformDistribution([0, 0, 0, 0], noFilters);
    const first = drawDistribution(distribution, 42);
    expect(first.randomNumber).toBeCloseTo(0.2523451747838408, 15);
    expect(first.tokenId).toBe(1);
    expect(first.intervalStart).toBe(0.25);
    expect(first.intervalEnd).toBe(0.5);
    const second = drawDistribution(distribution, first.nextRandomState);
    expect(second.randomNumber).toBeCloseTo(0.08812504541128874, 15);
    expect(second.tokenId).toBe(0);
    expect(drawDistribution(distribution, 42)).toEqual(first);
  });
});
