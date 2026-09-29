import { describe, expect, it } from 'vitest';
import {
  addBackward,
  addForward,
  batchedMatmulBackward,
  batchedMatmulForward,
  crossEntropyBackward,
  crossEntropyForward,
  embeddingBackward,
  embeddingForward,
  geluBackward,
  geluForward,
  layerNormBackward,
  layerNormForward,
  maskedSoftmaxBackward,
  maskedSoftmaxForward,
  matmulBackward,
  matmulForward,
  meanForward,
  reductionBackward,
  softmaxBackward,
  softmaxForward,
  sumForward,
  tensor,
  tensorsEqualShape,
} from './index';

function expectClose(
  actual: Float32Array | readonly number[],
  expected: readonly number[],
  digits = 5,
) {
  expect(actual).toHaveLength(expected.length);
  expected.forEach((value, index) => expect(actual[index]).toBeCloseTo(value, digits));
}

function finiteDifference(
  values: readonly number[],
  evaluate: (candidate: readonly number[]) => number,
  epsilon = 1e-3,
): number[] {
  return values.map((_, index) => {
    const positive = [...values];
    const negative = [...values];
    positive[index] += epsilon;
    negative[index] -= epsilon;
    return (evaluate(positive) - evaluate(negative)) / (2 * epsilon);
  });
}

describe('tensor validation and elementwise kernels', () => {
  it('rejects invalid shapes, lengths, and non-finite values', () => {
    expect(() => tensor([1, 2], [3])).toThrow('requires 3 values');
    expect(() => tensor([1], [-1])).toThrow('non-negative integer');
    expect(() => tensor([Number.NaN], [1])).toThrow('non-finite');
    expect(() => tensor([Infinity], [1])).toThrow('non-finite');
  });

  it('adds equal tensors and returns the upstream gradient to both inputs', () => {
    const result = addForward(tensor([1, 2, 3], [3]), tensor([4, -2, 0.5], [3]));
    expectClose(result.output.data, [5, 0, 3.5]);
    const [leftGradient, rightGradient] = addBackward(result.cache, tensor([2, 3, 4], [3]));
    expectClose(leftGradient.data, [2, 3, 4]);
    expectClose(rightGradient.data, [2, 3, 4]);
    expect(() => addForward(tensor([1, 2], [2]), tensor([1, 2], [1, 2]))).toThrow('equal shapes');
  });

  it('reduces by sum and mean with matching backward scale', () => {
    const input = tensor([1, 2, 3, 4], [2, 2]);
    const sum = sumForward(input);
    const mean = meanForward(input);
    expect(sum.output).toBe(10);
    expect(mean.output).toBe(2.5);
    expectClose(reductionBackward(sum.cache, 2).data, [2, 2, 2, 2]);
    expectClose(reductionBackward(mean.cache, 2).data, [0.5, 0.5, 0.5, 0.5]);
  });
});

describe('matrix multiplication', () => {
  it('matches a hand-computed forward and backward case', () => {
    const result = matmulForward(
      tensor([1, 2, 3, 4, 5, 6], [2, 3]),
      tensor([7, 8, 9, 10, 11, 12], [3, 2]),
    );
    expectClose(result.output.data, [58, 64, 139, 154]);
    const [leftGradient, rightGradient] = matmulBackward(
      result.cache,
      tensor([1, 2, 3, 4], [2, 2]),
    );
    expectClose(leftGradient.data, [23, 29, 35, 53, 67, 81]);
    expectClose(rightGradient.data, [13, 18, 17, 24, 21, 30]);
  });

  it('matches independent Float64 loop references for batched forward and backward', () => {
    const leftValues = [1, -2, 3, 0.5, 4, -1, -3, 2, 1, 5, -2, 0.25];
    const rightValues = [2, 1, -1, 3, 0.5, -2, 4, -0.5, 2, 1, -3, 2];
    const left = tensor(leftValues, [2, 2, 3]);
    const right = tensor(rightValues, [2, 3, 2]);
    const result = batchedMatmulForward(left, right);
    const expected: number[] = [];
    for (let batch = 0; batch < 2; batch += 1) {
      for (let row = 0; row < 2; row += 1) {
        for (let column = 0; column < 2; column += 1) {
          let sum = 0;
          for (let inner = 0; inner < 3; inner += 1) {
            sum +=
              leftValues[(batch * 2 + row) * 3 + inner] *
              rightValues[(batch * 3 + inner) * 2 + column];
          }
          expected.push(sum);
        }
      }
    }
    expectClose(result.output.data, expected);

    const upstream = tensor([1, -1, 0.5, 2, -2, 3, 1.5, -0.25], [2, 2, 2]);
    const [leftGradient, rightGradient] = batchedMatmulBackward(result.cache, upstream);
    const leftNumeric = finiteDifference(
      leftValues,
      (candidate) => {
        const output = batchedMatmulForward(tensor(candidate, [2, 2, 3]), right).output.data;
        return Array.from(output).reduce(
          (sum, value, index) => sum + value * upstream.data[index],
          0,
        );
      },
      1e-2,
    );
    const rightNumeric = finiteDifference(
      rightValues,
      (candidate) => {
        const output = batchedMatmulForward(left, tensor(candidate, [2, 3, 2])).output.data;
        return Array.from(output).reduce(
          (sum, value, index) => sum + value * upstream.data[index],
          0,
        );
      },
      1e-2,
    );
    expectClose(leftGradient.data, leftNumeric, 3);
    expectClose(rightGradient.data, rightNumeric, 3);
  });

  it('rejects incompatible matrix shapes', () => {
    expect(() => matmulForward(tensor([1, 2], [1, 2]), tensor([1, 2, 3], [3, 1]))).toThrow(
      'inner dimensions',
    );
  });
});

describe('softmax and cross-entropy', () => {
  it('is stable for large logits and normalizes each final-axis row', () => {
    const result = softmaxForward(
      tensor([10_000, 10_001, 10_002, -10_000, -10_000, -10_000], [2, 3]),
    );
    const rowOne = Array.from(result.output.data.slice(0, 3)).reduce(
      (sum, value) => sum + value,
      0,
    );
    expect(rowOne).toBeCloseTo(1, 6);
    expect(result.output.data.every(Number.isFinite)).toBe(true);
  });

  it('matches a Float64 Jacobian-vector reference', () => {
    const input = tensor([0.25, -0.5, 1.5], [1, 3]);
    const result = softmaxForward(input);
    const upstream = tensor([0.5, 2, -1], [1, 3]);
    const gradient = softmaxBackward(result.cache, upstream);
    const probabilities = Array.from(result.output.data);
    const expected = probabilities.map((probability, row) =>
      probabilities.reduce(
        (sum, otherProbability, column) =>
          sum + upstream.data[column] * otherProbability * ((row === column ? 1 : 0) - probability),
        0,
      ),
    );
    expectClose(gradient.data, expected);
  });

  it('treats masked logits as negative infinity and supports fully masked rows', () => {
    const result = maskedSoftmaxForward(
      tensor([1, 100, 3, 4, 5, 6], [2, 3]),
      new Uint8Array([1, 0, 1, 0, 0, 0]),
    );
    expect(result.output.data[1]).toBe(0);
    expect(result.output.data[0] + result.output.data[2]).toBeCloseTo(1, 6);
    expectClose(result.output.data.slice(3), [0, 0, 0]);
    const gradient = maskedSoftmaxBackward(result.cache, tensor([1, 2, 3, 4, 5, 6], [2, 3]));
    expect(gradient.data[1]).toBe(0);
    expectClose(gradient.data.slice(3), [0, 0, 0]);
  });

  it('computes mean cross-entropy and its fused gradient with ignored rows', () => {
    const logits = tensor([2, 1, 0, -1, 0, 1], [2, 3]);
    const result = crossEntropyForward(logits, [0, -1], -1);
    expect(result.output).toBeCloseTo(-Math.log(Math.exp(2) / (Math.exp(2) + Math.exp(1) + 1)), 6);
    const gradient = crossEntropyBackward(result.cache);
    expectClose(gradient.data.slice(3), [0, 0, 0]);
    expect(
      Array.from(gradient.data.slice(0, 3)).reduce((sum, value) => sum + value, 0),
    ).toBeCloseTo(0, 6);
  });

  it('keeps cross-entropy finite when the target probability underflows in Float32', () => {
    const result = crossEntropyForward(tensor([10_000, -10_000], [1, 2]), [1]);
    expect(result.output).toBe(20_000);
    expectClose(crossEntropyBackward(result.cache).data, [1, -1]);
  });

  it('matches finite differences for cross-entropy logits', () => {
    const values = [0.2, -0.4, 1.2, 2, 0.5, -1];
    const result = crossEntropyForward(tensor(values, [2, 3]), [2, 1]);
    const analytic = crossEntropyBackward(result.cache).data;
    const numeric = finiteDifference(
      values,
      (candidate) => crossEntropyForward(tensor(candidate, [2, 3]), [2, 1]).output,
    );
    expectClose(analytic, numeric, 3);
  });
});

describe('LayerNorm and GELU', () => {
  it('normalizes each row and applies learned scale and bias', () => {
    const result = layerNormForward(
      tensor([1, 2, 3, 4, 4, 4], [2, 3]),
      tensor([1, 2, 0.5], [3]),
      tensor([0, 1, -1], [3]),
    );
    expectClose(result.output.data.slice(0, 3), [-1.2247357, 1, -0.3876321], 4);
    expectClose(result.output.data.slice(3), [0, 1, -1]);
  });

  it('matches finite differences for LayerNorm input, gamma, and beta', () => {
    const inputValues = [1.2, -0.4, 2.1, 0.3, -1.5, 0.8];
    const gammaValues = [0.9, -1.2, 0.4];
    const betaValues = [0.1, 0.2, -0.3];
    const upstream = tensor([0.5, -1, 2, -0.25, 1.5, 0.75], [2, 3]);
    const result = layerNormForward(
      tensor(inputValues, [2, 3]),
      tensor(gammaValues, [3]),
      tensor(betaValues, [3]),
    );
    const analytic = layerNormBackward(result.cache, upstream);
    const objective = (
      input: readonly number[],
      gamma: readonly number[],
      beta: readonly number[],
    ) => {
      const output = layerNormForward(tensor(input, [2, 3]), tensor(gamma, [3]), tensor(beta, [3]))
        .output.data;
      return Array.from(output).reduce(
        (sum, value, index) => sum + value * upstream.data[index],
        0,
      );
    };
    expectClose(
      analytic.input.data,
      finiteDifference(inputValues, (value) => objective(value, gammaValues, betaValues)),
      3,
    );
    expectClose(
      analytic.gamma.data,
      finiteDifference(gammaValues, (value) => objective(inputValues, value, betaValues)),
      3,
    );
    expectClose(
      analytic.beta.data,
      finiteDifference(betaValues, (value) => objective(inputValues, gammaValues, value)),
      3,
    );
  });

  it('matches finite differences for the tanh GELU approximation', () => {
    const values = [-2, -0.25, 0, 0.7, 2.5];
    const upstream = tensor([1, -2, 0.5, 3, -1], [5]);
    const gradient = geluBackward(tensor(values, [5]), upstream);
    const numeric = finiteDifference(values, (candidate) => {
      const output = geluForward(tensor(candidate, [5])).output.data;
      return Array.from(output).reduce(
        (sum, value, index) => sum + value * upstream.data[index],
        0,
      );
    });
    expectClose(gradient.data, numeric, 3);
  });
});

describe('embedding lookup and scatter-add', () => {
  it('preserves index shape and accumulates repeated token gradients', () => {
    const weights = tensor([1, 2, 3, 4, 5, 6, 7, 8], [4, 2]);
    const result = embeddingForward(weights, [2, 0, 2, 3], [2, 2]);
    expect(result.output.shape).toEqual([2, 2, 2]);
    expectClose(result.output.data, [5, 6, 1, 2, 5, 6, 7, 8]);
    const gradient = embeddingBackward(result.cache, tensor([1, 1, 2, 2, 3, 3, 4, 4], [2, 2, 2]));
    expectClose(gradient.data, [2, 2, 0, 0, 4, 4, 4, 4]);
    expect(tensorsEqualShape(gradient, weights)).toBe(true);
  });

  it('rejects invalid token IDs', () => {
    expect(() => embeddingForward(tensor([1, 2], [1, 2]), [1])).toThrow('out of range');
  });
});
