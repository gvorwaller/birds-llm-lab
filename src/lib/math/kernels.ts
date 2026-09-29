import {
  assertFinite,
  assertSameShape,
  assertShape,
  cloneTensor,
  elementCount,
  sameShape,
  tensor,
  type Tensor,
} from './tensor';

function requireRank(value: Tensor, rank: number, label: string): void {
  assertShape(value, value.shape, label);
  if (value.shape.length !== rank) {
    throw new Error(`${label} must have rank ${rank}; received rank ${value.shape.length}.`);
  }
}

export interface BinaryCache {
  readonly left: Tensor;
  readonly right: Tensor;
}

export function addForward(left: Tensor, right: Tensor): { output: Tensor; cache: BinaryCache } {
  assertSameShape(left, right, 'Add');
  const output = new Float32Array(left.data.length);
  for (let index = 0; index < output.length; index += 1) {
    output[index] = left.data[index] + right.data[index];
  }
  return { output: tensor(output, left.shape, 'Add output'), cache: { left, right } };
}

export function addBackward(cache: BinaryCache, gradient: Tensor): [Tensor, Tensor] {
  assertShape(gradient, cache.left.shape, 'Add output gradient');
  return [cloneTensor(gradient), cloneTensor(gradient)];
}

export interface MatmulCache extends BinaryCache {}

export function matmulForward(left: Tensor, right: Tensor): { output: Tensor; cache: MatmulCache } {
  requireRank(left, 2, 'Matmul left input');
  requireRank(right, 2, 'Matmul right input');
  const [rows, inner] = left.shape;
  const [rightInner, columns] = right.shape;
  if (inner !== rightInner) {
    throw new Error(`Matmul inner dimensions must match; received ${inner} and ${rightInner}.`);
  }
  const output = new Float32Array(rows * columns);
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      let sum = 0;
      for (let index = 0; index < inner; index += 1) {
        sum += left.data[row * inner + index] * right.data[index * columns + column];
      }
      output[row * columns + column] = sum;
    }
  }
  return { output: tensor(output, [rows, columns], 'Matmul output'), cache: { left, right } };
}

export function matmulBackward(cache: MatmulCache, gradient: Tensor): [Tensor, Tensor] {
  const { left, right } = cache;
  const [rows, inner] = left.shape;
  const columns = right.shape[1];
  assertShape(gradient, [rows, columns], 'Matmul output gradient');
  const leftGradient = new Float32Array(left.data.length);
  const rightGradient = new Float32Array(right.data.length);
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const upstream = gradient.data[row * columns + column];
      for (let index = 0; index < inner; index += 1) {
        leftGradient[row * inner + index] += upstream * right.data[index * columns + column];
        rightGradient[index * columns + column] += left.data[row * inner + index] * upstream;
      }
    }
  }
  return [tensor(leftGradient, left.shape), tensor(rightGradient, right.shape)];
}

export function batchedMatmulForward(
  left: Tensor,
  right: Tensor,
): { output: Tensor; cache: MatmulCache } {
  requireRank(left, 3, 'Batched matmul left input');
  requireRank(right, 3, 'Batched matmul right input');
  const [batches, rows, inner] = left.shape;
  const [rightBatches, rightInner, columns] = right.shape;
  if (batches !== rightBatches || inner !== rightInner) {
    throw new Error(
      `Batched matmul requires [B, M, K] x [B, K, N]; received [${left.shape.join(', ')}] and [${right.shape.join(', ')}].`,
    );
  }
  const output = new Float32Array(batches * rows * columns);
  for (let batch = 0; batch < batches; batch += 1) {
    for (let row = 0; row < rows; row += 1) {
      for (let column = 0; column < columns; column += 1) {
        let sum = 0;
        for (let index = 0; index < inner; index += 1) {
          sum +=
            left.data[(batch * rows + row) * inner + index] *
            right.data[(batch * inner + index) * columns + column];
        }
        output[(batch * rows + row) * columns + column] = sum;
      }
    }
  }
  return {
    output: tensor(output, [batches, rows, columns], 'Batched matmul output'),
    cache: { left, right },
  };
}

export function batchedMatmulBackward(cache: MatmulCache, gradient: Tensor): [Tensor, Tensor] {
  const { left, right } = cache;
  requireRank(left, 3, 'Batched matmul cached left input');
  requireRank(right, 3, 'Batched matmul cached right input');
  const [batches, rows, inner] = left.shape;
  const columns = right.shape[2];
  assertShape(gradient, [batches, rows, columns], 'Batched matmul output gradient');
  const leftGradient = new Float32Array(left.data.length);
  const rightGradient = new Float32Array(right.data.length);
  for (let batch = 0; batch < batches; batch += 1) {
    for (let row = 0; row < rows; row += 1) {
      for (let column = 0; column < columns; column += 1) {
        const upstream = gradient.data[(batch * rows + row) * columns + column];
        for (let index = 0; index < inner; index += 1) {
          const leftIndex = (batch * rows + row) * inner + index;
          const rightIndex = (batch * inner + index) * columns + column;
          leftGradient[leftIndex] += upstream * right.data[rightIndex];
          rightGradient[rightIndex] += left.data[leftIndex] * upstream;
        }
      }
    }
  }
  return [tensor(leftGradient, left.shape), tensor(rightGradient, right.shape)];
}

export interface ReductionCache {
  readonly inputShape: readonly number[];
  readonly divisor: number;
}

export function sumForward(input: Tensor): { output: number; cache: ReductionCache } {
  assertShape(input, input.shape, 'Sum input');
  let output = 0;
  for (const value of input.data) output += value;
  if (!Number.isFinite(output)) throw new Error('Sum output is non-finite.');
  return { output, cache: { inputShape: input.shape, divisor: 1 } };
}

export function meanForward(input: Tensor): { output: number; cache: ReductionCache } {
  if (input.data.length === 0) throw new Error('Mean requires at least one value.');
  const result = sumForward(input);
  return {
    output: result.output / input.data.length,
    cache: { inputShape: input.shape, divisor: input.data.length },
  };
}

export function reductionBackward(cache: ReductionCache, gradient = 1): Tensor {
  if (!Number.isFinite(gradient)) throw new Error('Reduction gradient must be finite.');
  return tensor(
    new Float32Array(elementCount(cache.inputShape)).fill(gradient / cache.divisor),
    cache.inputShape,
  );
}

export interface SoftmaxCache {
  readonly output: Tensor;
  readonly axisSize: number;
}

export function softmaxForward(input: Tensor): { output: Tensor; cache: SoftmaxCache } {
  assertShape(input, input.shape, 'Softmax input');
  const axisSize = input.shape.at(-1);
  if (axisSize === undefined || axisSize <= 0) {
    throw new Error('Softmax requires a non-empty final axis.');
  }
  const output = new Float32Array(input.data.length);
  for (let offset = 0; offset < input.data.length; offset += axisSize) {
    let maximum = -Infinity;
    for (let index = 0; index < axisSize; index += 1) {
      maximum = Math.max(maximum, input.data[offset + index]);
    }
    let total = 0;
    for (let index = 0; index < axisSize; index += 1) {
      const value = Math.exp(input.data[offset + index] - maximum);
      output[offset + index] = value;
      total += value;
    }
    for (let index = 0; index < axisSize; index += 1) output[offset + index] /= total;
  }
  const outputTensor = tensor(output, input.shape, 'Softmax output');
  return { output: outputTensor, cache: { output: outputTensor, axisSize } };
}

export function softmaxBackward(cache: SoftmaxCache, gradient: Tensor): Tensor {
  assertShape(gradient, cache.output.shape, 'Softmax output gradient');
  const output = new Float32Array(gradient.data.length);
  for (let offset = 0; offset < output.length; offset += cache.axisSize) {
    let dot = 0;
    for (let index = 0; index < cache.axisSize; index += 1) {
      dot += gradient.data[offset + index] * cache.output.data[offset + index];
    }
    for (let index = 0; index < cache.axisSize; index += 1) {
      output[offset + index] =
        cache.output.data[offset + index] * (gradient.data[offset + index] - dot);
    }
  }
  return tensor(output, gradient.shape, 'Softmax input gradient');
}

export interface MaskedSoftmaxCache extends SoftmaxCache {
  readonly allowed: Uint8Array;
}

export function maskedSoftmaxForward(
  input: Tensor,
  allowed: Uint8Array,
): { output: Tensor; cache: MaskedSoftmaxCache } {
  assertShape(input, input.shape, 'Masked softmax input');
  const axisSize = input.shape.at(-1);
  if (axisSize === undefined || axisSize <= 0) {
    throw new Error('Masked softmax requires a non-empty final axis.');
  }
  if (allowed.length !== input.data.length) {
    throw new Error(
      `Masked softmax mask requires ${input.data.length} entries; received ${allowed.length}.`,
    );
  }
  const output = new Float32Array(input.data.length);
  for (let offset = 0; offset < input.data.length; offset += axisSize) {
    let maximum = -Infinity;
    let allowedCount = 0;
    for (let index = 0; index < axisSize; index += 1) {
      if (allowed[offset + index] === 0) continue;
      maximum = Math.max(maximum, input.data[offset + index]);
      allowedCount += 1;
    }
    if (allowedCount === 0) continue;
    let total = 0;
    for (let index = 0; index < axisSize; index += 1) {
      if (allowed[offset + index] === 0) continue;
      const value = Math.exp(input.data[offset + index] - maximum);
      output[offset + index] = value;
      total += value;
    }
    for (let index = 0; index < axisSize; index += 1) {
      if (allowed[offset + index] !== 0) output[offset + index] /= total;
    }
  }
  const outputTensor = tensor(output, input.shape, 'Masked softmax output');
  return {
    output: outputTensor,
    cache: { output: outputTensor, axisSize, allowed: allowed.slice() },
  };
}

export function maskedSoftmaxBackward(cache: MaskedSoftmaxCache, gradient: Tensor): Tensor {
  assertShape(gradient, cache.output.shape, 'Masked softmax output gradient');
  const output = softmaxBackward(cache, gradient).data;
  for (let index = 0; index < output.length; index += 1) {
    if (cache.allowed[index] === 0) output[index] = 0;
  }
  return tensor(output, gradient.shape, 'Masked softmax input gradient');
}

export interface LayerNormCache {
  readonly normalized: Tensor;
  readonly inverseStandardDeviation: Float32Array;
  readonly gamma: Tensor;
  readonly axisSize: number;
}

export interface LayerNormGradients {
  readonly input: Tensor;
  readonly gamma: Tensor;
  readonly beta: Tensor;
}

export function layerNormForward(
  input: Tensor,
  gamma: Tensor,
  beta: Tensor,
  epsilon = 1e-5,
): { output: Tensor; cache: LayerNormCache } {
  assertShape(input, input.shape, 'LayerNorm input');
  const axisSize = input.shape.at(-1);
  if (axisSize === undefined || axisSize <= 0) {
    throw new Error('LayerNorm requires a non-empty final axis.');
  }
  assertShape(gamma, [axisSize], 'LayerNorm gamma');
  assertShape(beta, [axisSize], 'LayerNorm beta');
  if (!(epsilon > 0) || !Number.isFinite(epsilon)) {
    throw new Error('LayerNorm epsilon must be a positive finite number.');
  }
  const normalized = new Float32Array(input.data.length);
  const output = new Float32Array(input.data.length);
  const inverseStandardDeviation = new Float32Array(input.data.length / axisSize);
  for (let row = 0; row < inverseStandardDeviation.length; row += 1) {
    const offset = row * axisSize;
    let mean = 0;
    for (let index = 0; index < axisSize; index += 1) mean += input.data[offset + index];
    mean /= axisSize;
    let variance = 0;
    for (let index = 0; index < axisSize; index += 1) {
      const centered = input.data[offset + index] - mean;
      variance += centered * centered;
    }
    const inverse = 1 / Math.sqrt(variance / axisSize + epsilon);
    inverseStandardDeviation[row] = inverse;
    for (let index = 0; index < axisSize; index += 1) {
      const value = (input.data[offset + index] - mean) * inverse;
      normalized[offset + index] = value;
      output[offset + index] = value * gamma.data[index] + beta.data[index];
    }
  }
  const normalizedTensor = tensor(normalized, input.shape, 'LayerNorm normalized values');
  return {
    output: tensor(output, input.shape, 'LayerNorm output'),
    cache: { normalized: normalizedTensor, inverseStandardDeviation, gamma, axisSize },
  };
}

export function layerNormBackward(cache: LayerNormCache, gradient: Tensor): LayerNormGradients {
  assertShape(gradient, cache.normalized.shape, 'LayerNorm output gradient');
  const inputGradient = new Float32Array(gradient.data.length);
  const gammaGradient = new Float32Array(cache.axisSize);
  const betaGradient = new Float32Array(cache.axisSize);
  const rows = gradient.data.length / cache.axisSize;
  for (let row = 0; row < rows; row += 1) {
    const offset = row * cache.axisSize;
    let sumScaledGradient = 0;
    let sumScaledGradientTimesNormalized = 0;
    for (let index = 0; index < cache.axisSize; index += 1) {
      const position = offset + index;
      const upstream = gradient.data[position];
      const scaled = upstream * cache.gamma.data[index];
      sumScaledGradient += scaled;
      sumScaledGradientTimesNormalized += scaled * cache.normalized.data[position];
      gammaGradient[index] += upstream * cache.normalized.data[position];
      betaGradient[index] += upstream;
    }
    for (let index = 0; index < cache.axisSize; index += 1) {
      const position = offset + index;
      const scaled = gradient.data[position] * cache.gamma.data[index];
      inputGradient[position] =
        (cache.inverseStandardDeviation[row] / cache.axisSize) *
        (cache.axisSize * scaled -
          sumScaledGradient -
          cache.normalized.data[position] * sumScaledGradientTimesNormalized);
    }
  }
  return {
    input: tensor(inputGradient, gradient.shape, 'LayerNorm input gradient'),
    gamma: tensor(gammaGradient, [cache.axisSize], 'LayerNorm gamma gradient'),
    beta: tensor(betaGradient, [cache.axisSize], 'LayerNorm beta gradient'),
  };
}

const GELU_FACTOR = Math.sqrt(2 / Math.PI);
const GELU_CUBIC = 0.044715;

export function geluForward(input: Tensor): { output: Tensor; cache: Tensor } {
  assertShape(input, input.shape, 'GELU input');
  const output = new Float32Array(input.data.length);
  for (let index = 0; index < output.length; index += 1) {
    const value = input.data[index];
    output[index] = 0.5 * value * (1 + Math.tanh(GELU_FACTOR * (value + GELU_CUBIC * value ** 3)));
  }
  return { output: tensor(output, input.shape, 'GELU output'), cache: input };
}

export function geluBackward(input: Tensor, gradient: Tensor): Tensor {
  assertSameShape(input, gradient, 'GELU backward');
  const output = new Float32Array(input.data.length);
  for (let index = 0; index < output.length; index += 1) {
    const value = input.data[index];
    const argument = GELU_FACTOR * (value + GELU_CUBIC * value ** 3);
    const tanh = Math.tanh(argument);
    const derivative =
      0.5 * (1 + tanh) +
      0.5 * value * (1 - tanh * tanh) * GELU_FACTOR * (1 + 3 * GELU_CUBIC * value * value);
    output[index] = gradient.data[index] * derivative;
  }
  return tensor(output, input.shape, 'GELU input gradient');
}

export interface EmbeddingCache {
  readonly indices: readonly number[];
  readonly weightShape: readonly [number, number];
  readonly outputShape: readonly number[];
}

export function embeddingForward(
  weights: Tensor,
  indices: readonly number[],
  indexShape: readonly number[] = [indices.length],
): { output: Tensor; cache: EmbeddingCache } {
  requireRank(weights, 2, 'Embedding weights');
  if (elementCount(indexShape) !== indices.length) {
    throw new Error('Embedding index shape does not match the number of indices.');
  }
  const [vocabularySize, width] = weights.shape;
  const output = new Float32Array(indices.length * width);
  indices.forEach((tokenId, row) => {
    if (!Number.isInteger(tokenId) || tokenId < 0 || tokenId >= vocabularySize) {
      throw new Error(`Embedding token ID ${tokenId} at index ${row} is out of range.`);
    }
    output.set(weights.data.subarray(tokenId * width, (tokenId + 1) * width), row * width);
  });
  const outputShape = [...indexShape, width];
  return {
    output: tensor(output, outputShape, 'Embedding output'),
    cache: {
      indices: [...indices],
      weightShape: [vocabularySize, width],
      outputShape,
    },
  };
}

export function embeddingBackward(cache: EmbeddingCache, gradient: Tensor): Tensor {
  assertShape(gradient, cache.outputShape, 'Embedding output gradient');
  const [vocabularySize, width] = cache.weightShape;
  const weightGradient = new Float32Array(vocabularySize * width);
  cache.indices.forEach((tokenId, row) => {
    for (let column = 0; column < width; column += 1) {
      weightGradient[tokenId * width + column] += gradient.data[row * width + column];
    }
  });
  return tensor(weightGradient, cache.weightShape, 'Embedding weight gradient');
}

export interface CrossEntropyCache {
  readonly probabilities: Tensor;
  readonly targets: readonly number[];
  readonly ignoreIndex: number | null;
  readonly validCount: number;
}

export function crossEntropyForward(
  logits: Tensor,
  targets: readonly number[],
  ignoreIndex: number | null = null,
): { output: number; cache: CrossEntropyCache } {
  requireRank(logits, 2, 'Cross-entropy logits');
  const [rows, classes] = logits.shape;
  if (targets.length !== rows) {
    throw new Error(`Cross-entropy requires ${rows} targets; received ${targets.length}.`);
  }
  const probabilities = softmaxForward(logits).output;
  let loss = 0;
  let validCount = 0;
  targets.forEach((target, row) => {
    if (target === ignoreIndex) return;
    if (!Number.isInteger(target) || target < 0 || target >= classes) {
      throw new Error(`Cross-entropy target ${target} at row ${row} is out of range.`);
    }
    const offset = row * classes;
    let maximum = -Infinity;
    for (let column = 0; column < classes; column += 1) {
      maximum = Math.max(maximum, logits.data[offset + column]);
    }
    let exponentialSum = 0;
    for (let column = 0; column < classes; column += 1) {
      exponentialSum += Math.exp(logits.data[offset + column] - maximum);
    }
    loss += maximum + Math.log(exponentialSum) - logits.data[offset + target];
    validCount += 1;
  });
  if (validCount === 0) throw new Error('Cross-entropy requires at least one non-ignored target.');
  const output = loss / validCount;
  if (!Number.isFinite(output)) throw new Error('Cross-entropy loss is non-finite.');
  return {
    output,
    cache: { probabilities, targets: [...targets], ignoreIndex, validCount },
  };
}

export function crossEntropyBackward(cache: CrossEntropyCache, gradient = 1): Tensor {
  if (!Number.isFinite(gradient)) throw new Error('Cross-entropy gradient must be finite.');
  const [rows, classes] = cache.probabilities.shape;
  const output = cache.probabilities.data.slice();
  for (let row = 0; row < rows; row += 1) {
    const target = cache.targets[row];
    if (target === cache.ignoreIndex) {
      output.fill(0, row * classes, (row + 1) * classes);
    } else {
      output[row * classes + target] -= 1;
    }
  }
  const scale = gradient / cache.validCount;
  for (let index = 0; index < output.length; index += 1) output[index] *= scale;
  return tensor(output, cache.probabilities.shape, 'Cross-entropy logits gradient');
}

export function tensorsEqualShape(left: Tensor, right: Tensor): boolean {
  return sameShape(left, right);
}

export function validateGradient(gradient: Tensor, label: string): void {
  assertShape(gradient, gradient.shape, label);
  assertFinite(gradient.data, label);
}
