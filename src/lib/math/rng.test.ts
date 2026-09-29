import { describe, expect, it } from 'vitest';
import { SeededRandom } from './rng';

describe('SeededRandom', () => {
  it('produces an exact, repeatable unsigned-integer sequence', () => {
    const first = new SeededRandom(42);
    const second = new SeededRandom(42);
    const firstValues = Array.from({ length: 8 }, () => first.nextUint32());
    const secondValues = Array.from({ length: 8 }, () => second.nextUint32());
    expect(firstValues).toEqual(secondValues);
    expect(firstValues).toEqual([
      1083814273, 378494188, 2479403867, 955863294, 1613448261, 110225632, 1921058495, 508781842,
    ]);
  });

  it('restores both uniform state and a pending Box-Muller variate', () => {
    const original = new SeededRandom(0x5eed);
    original.normal();
    const restored = new SeededRandom(original.snapshot());
    expect(restored.normal()).toBe(original.normal());
    expect(restored.uniform()).toBe(original.uniform());
  });

  it('generates finite normal values with plausible aggregate moments', () => {
    const random = new SeededRandom(1234);
    const values = Array.from({ length: 20_000 }, () => random.normal());
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
    const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
    expect(values.every(Number.isFinite)).toBe(true);
    expect(mean).toBeCloseTo(0, 1);
    expect(variance).toBeCloseTo(1, 1);
  });

  it('rejects invalid seeds and saved states', () => {
    expect(() => new SeededRandom(1.5)).toThrow('safe integer');
    expect(() => new SeededRandom({ state: -1, spareNormal: null })).toThrow('unsigned 32-bit');
    expect(() => new SeededRandom({ state: 1, spareNormal: Infinity })).toThrow('finite or null');
  });
});
