import { describe, expect, it } from 'vitest';
import { SeededRandom } from '../math/rng';
import { sampleCategorical } from './sampling';

describe('seeded categorical sampling', () => {
  it('produces identical token sequences for identical seeds', () => {
    const sample = (seed: number) => {
      const random = new SeededRandom(seed);
      return Array.from({ length: 30 }, () =>
        sampleCategorical(new Float32Array([0.5, -1, 2, 0.25]), random),
      );
    };
    expect(sample(123)).toEqual(sample(123));
    expect(sample(124)).not.toEqual(sample(123));
  });

  it('remains stable for extreme logits and validates inputs', () => {
    expect(sampleCategorical([10_000, -10_000], new SeededRandom(1))).toBe(0);
    expect(() => sampleCategorical([], new SeededRandom(1))).toThrow('at least one logit');
    expect(() => sampleCategorical([0, Number.NaN], new SeededRandom(1))).toThrow('finite');
    expect(() => sampleCategorical([0, 1], new SeededRandom(1), 0)).toThrow('positive');
  });
});
