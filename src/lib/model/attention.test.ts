import { describe, expect, it } from 'vitest';
import { tensor } from '../math/tensor';
import {
  causalPaddingMask,
  mergeHeads,
  multiHeadAttentionBackward,
  multiHeadAttentionForward,
  scaledDotProductAttentionBackward,
  scaledDotProductAttentionForward,
  splitHeads,
  type MultiHeadAttentionParameters,
} from './attention';

function expectClose(actual: Float32Array, expected: readonly number[], digits = 5): void {
  expect(actual).toHaveLength(expected.length);
  expected.forEach((value, index) => expect(actual[index]).toBeCloseTo(value, digits));
}

function identity(width: number) {
  const values = new Float32Array(width * width);
  for (let index = 0; index < width; index += 1) values[index * width + index] = 1;
  return tensor(values, [width, width]);
}

describe('scaled dot-product attention', () => {
  it('matches a hand-computed causal single-head case', () => {
    const query = tensor([1, 2], [1, 2, 1]);
    const key = tensor([1, 2], [1, 2, 1]);
    const value = tensor([10, 20], [1, 2, 1]);
    const result = scaledDotProductAttentionForward(
      query,
      key,
      value,
      new Uint8Array([1, 0, 1, 1]),
    );
    const laterFirstProbability = Math.exp(2) / (Math.exp(2) + Math.exp(4));
    expectClose(result.probabilities.data, [
      1,
      0,
      laterFirstProbability,
      1 - laterFirstProbability,
    ]);
    expectClose(result.output.data, [10, 20 - 10 * laterFirstProbability]);
  });

  it('returns finite gradients and exactly zero gradient through masked scores', () => {
    const query = tensor([0.5, -0.25, 1, 0.75], [1, 2, 2]);
    const key = tensor([0.2, 0.4, -0.5, 1], [1, 2, 2]);
    const value = tensor([1, 2, 3, 4], [1, 2, 2]);
    const result = scaledDotProductAttentionForward(
      query,
      key,
      value,
      new Uint8Array([1, 0, 1, 1]),
    );
    const gradients = scaledDotProductAttentionBackward(
      result.cache,
      tensor([1, -1, 0.5, 2], [1, 2, 2]),
    );
    expect(
      [...gradients.query.data, ...gradients.key.data, ...gradients.value.data].every(
        Number.isFinite,
      ),
    ).toBe(true);
    // The first query has one allowed key, so its softmax is constant and its query gradient is zero.
    expectClose(gradients.query.data.slice(0, 2), [0, 0]);
  });
});

describe('multi-head attention', () => {
  it('splits and merges heads without changing channel order', () => {
    const input = tensor(
      Array.from({ length: 24 }, (_, index) => index),
      [2, 3, 4],
    );
    const split = splitHeads(input, 2, 2);
    expect(split.shape).toEqual([4, 3, 2]);
    expect(mergeHeads(split, 2, 2, 2)).toEqual(input);
  });

  it('applies causal and padding masks independently in every head', () => {
    const width = 4;
    const zeroBias = tensor(new Float32Array(width), [width]);
    const parameters: MultiHeadAttentionParameters = {
      queryWeight: identity(width),
      queryBias: zeroBias,
      keyWeight: identity(width),
      keyBias: zeroBias,
      valueWeight: identity(width),
      valueBias: zeroBias,
      outputWeight: identity(width),
      outputBias: zeroBias,
    };
    const input = tensor([1, 0, 0, 1, 0.5, 1, -1, 0.5, 2, 2, 2, 2], [1, 3, 4]);
    const padding = new Uint8Array([1, 1, 0]);
    const result = multiHeadAttentionForward(input, parameters, 2, 2, padding);
    expect(result.output.shape).toEqual([1, 3, 4]);
    expect(result.probabilities.shape).toEqual([2, 3, 3]);
    for (let head = 0; head < 2; head += 1) {
      const offset = head * 9;
      expectClose(result.probabilities.data.slice(offset, offset + 3), [1, 0, 0]);
      expect(result.probabilities.data[offset + 5]).toBe(0);
      expectClose(result.probabilities.data.slice(offset + 6, offset + 9), [0, 0, 0]);
    }
    const backward = multiHeadAttentionBackward(
      result.cache,
      tensor(new Float32Array(12).fill(1), [1, 3, 4]),
    );
    expect(backward.input.shape).toEqual(input.shape);
    expect(backward.queryWeight.shape).toEqual([4, 4]);
  });

  it('builds the same causal-padding pattern for each requested head', () => {
    const mask = causalPaddingMask(1, 2, 3, new Uint8Array([1, 1, 0]));
    expect(Array.from(mask.slice(0, 9))).toEqual([1, 0, 0, 1, 1, 0, 0, 0, 0]);
    expect(Array.from(mask.slice(9))).toEqual(Array.from(mask.slice(0, 9)));
  });
});
