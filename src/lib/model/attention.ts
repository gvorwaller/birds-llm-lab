import {
  batchedMatmulBackward,
  batchedMatmulForward,
  maskedSoftmaxBackward,
  maskedSoftmaxForward,
  type MaskedSoftmaxCache,
  type MatmulCache,
} from '../math/kernels';
import { assertShape, tensor, type Tensor } from '../math/tensor';
import { linearBackward, linearForward, sumTensorGradients, type LinearCache } from './layers';

export interface ScaledAttentionCache {
  readonly query: Tensor;
  readonly key: Tensor;
  readonly value: Tensor;
  readonly keyTranspose: Tensor;
  readonly scoreMatmul: MatmulCache;
  readonly contextMatmul: MatmulCache;
  readonly softmax: MaskedSoftmaxCache;
  readonly scale: number;
}

export interface ScaledAttentionGradients {
  readonly query: Tensor;
  readonly key: Tensor;
  readonly value: Tensor;
}

function transposeLastTwo(input: Tensor): Tensor {
  if (input.shape.length !== 3) throw new Error('Attention transpose requires a rank-3 tensor.');
  const [groups, rows, columns] = input.shape;
  assertShape(input, [groups, rows, columns], 'Attention transpose input');
  const output = new Float32Array(input.data.length);
  for (let group = 0; group < groups; group += 1) {
    for (let row = 0; row < rows; row += 1) {
      for (let column = 0; column < columns; column += 1) {
        output[(group * columns + column) * rows + row] =
          input.data[(group * rows + row) * columns + column];
      }
    }
  }
  return tensor(output, [groups, columns, rows], 'Attention transposed tensor');
}

export function scaledDotProductAttentionForward(
  query: Tensor,
  key: Tensor,
  value: Tensor,
  allowed: Uint8Array,
): { output: Tensor; probabilities: Tensor; cache: ScaledAttentionCache } {
  if (query.shape.length !== 3) throw new Error('Attention query must have rank 3.');
  const [groups, sequenceLength, width] = query.shape;
  assertShape(query, [groups, sequenceLength, width], 'Attention query');
  assertShape(key, [groups, sequenceLength, width], 'Attention key');
  assertShape(value, [groups, sequenceLength, width], 'Attention value');
  if (allowed.length !== groups * sequenceLength * sequenceLength) {
    throw new Error('Attention mask shape does not match query groups and sequence length.');
  }
  const keyTranspose = transposeLastTwo(key);
  const scores = batchedMatmulForward(query, keyTranspose);
  const scale = 1 / Math.sqrt(width);
  const scaledScores = tensor(
    Float32Array.from(scores.output.data, (score) => score * scale),
    scores.output.shape,
    'Scaled attention scores',
  );
  const softmax = maskedSoftmaxForward(scaledScores, allowed);
  const context = batchedMatmulForward(softmax.output, value);
  return {
    output: context.output,
    probabilities: softmax.output,
    cache: {
      query,
      key,
      value,
      keyTranspose,
      scoreMatmul: scores.cache,
      contextMatmul: context.cache,
      softmax: softmax.cache,
      scale,
    },
  };
}

export function scaledDotProductAttentionBackward(
  cache: ScaledAttentionCache,
  gradient: Tensor,
): ScaledAttentionGradients {
  const [probabilityGradient, valueGradient] = batchedMatmulBackward(cache.contextMatmul, gradient);
  const scaledScoreGradient = maskedSoftmaxBackward(cache.softmax, probabilityGradient);
  const scoreGradient = tensor(
    Float32Array.from(scaledScoreGradient.data, (value) => value * cache.scale),
    scaledScoreGradient.shape,
    'Attention score gradient',
  );
  const [queryGradient, keyTransposeGradient] = batchedMatmulBackward(
    cache.scoreMatmul,
    scoreGradient,
  );
  return {
    query: queryGradient,
    key: transposeLastTwo(keyTransposeGradient),
    value: valueGradient,
  };
}

export function splitHeads(input: Tensor, headCount: number, headWidth: number): Tensor {
  if (input.shape.length !== 3) throw new Error('Head splitting requires a rank-3 tensor.');
  const [batchSize, sequenceLength, modelWidth] = input.shape;
  assertShape(input, [batchSize, sequenceLength, modelWidth], 'Head split input');
  if (headCount * headWidth !== modelWidth) {
    throw new Error('Head count times head width must equal model width.');
  }
  const output = new Float32Array(input.data.length);
  for (let batch = 0; batch < batchSize; batch += 1) {
    for (let head = 0; head < headCount; head += 1) {
      for (let token = 0; token < sequenceLength; token += 1) {
        for (let channel = 0; channel < headWidth; channel += 1) {
          output[((batch * headCount + head) * sequenceLength + token) * headWidth + channel] =
            input.data[(batch * sequenceLength + token) * modelWidth + head * headWidth + channel];
        }
      }
    }
  }
  return tensor(output, [batchSize * headCount, sequenceLength, headWidth], 'Split heads');
}

export function mergeHeads(
  input: Tensor,
  batchSize: number,
  headCount: number,
  headWidth: number,
): Tensor {
  if (input.shape.length !== 3) throw new Error('Head merging requires a rank-3 tensor.');
  const sequenceLength = input.shape[1];
  assertShape(input, [batchSize * headCount, sequenceLength, headWidth], 'Head merge input');
  const modelWidth = headCount * headWidth;
  const output = new Float32Array(input.data.length);
  for (let batch = 0; batch < batchSize; batch += 1) {
    for (let head = 0; head < headCount; head += 1) {
      for (let token = 0; token < sequenceLength; token += 1) {
        for (let channel = 0; channel < headWidth; channel += 1) {
          output[(batch * sequenceLength + token) * modelWidth + head * headWidth + channel] =
            input.data[((batch * headCount + head) * sequenceLength + token) * headWidth + channel];
        }
      }
    }
  }
  return tensor(output, [batchSize, sequenceLength, modelWidth], 'Merged heads');
}

export function causalPaddingMask(
  batchSize: number,
  headCount: number,
  sequenceLength: number,
  paddingMask?: Uint8Array,
): Uint8Array {
  const valid = paddingMask ?? new Uint8Array(batchSize * sequenceLength).fill(1);
  if (valid.length !== batchSize * sequenceLength) {
    throw new Error('Padding mask shape does not match batch and sequence dimensions.');
  }
  const allowed = new Uint8Array(batchSize * headCount * sequenceLength * sequenceLength);
  for (let batch = 0; batch < batchSize; batch += 1) {
    for (let head = 0; head < headCount; head += 1) {
      const group = batch * headCount + head;
      for (let query = 0; query < sequenceLength; query += 1) {
        for (let key = 0; key < sequenceLength; key += 1) {
          const isAllowed =
            valid[batch * sequenceLength + query] !== 0 &&
            valid[batch * sequenceLength + key] !== 0 &&
            key <= query;
          allowed[(group * sequenceLength + query) * sequenceLength + key] = isAllowed ? 1 : 0;
        }
      }
    }
  }
  return allowed;
}

export interface MultiHeadAttentionParameters {
  readonly queryWeight: Tensor;
  readonly queryBias: Tensor | null;
  readonly keyWeight: Tensor;
  readonly keyBias: Tensor | null;
  readonly valueWeight: Tensor;
  readonly valueBias: Tensor | null;
  readonly outputWeight: Tensor;
  readonly outputBias: Tensor | null;
}

export interface MultiHeadAttentionCache {
  readonly queryLinear: LinearCache;
  readonly keyLinear: LinearCache;
  readonly valueLinear: LinearCache;
  readonly outputLinear: LinearCache;
  readonly attention: ScaledAttentionCache;
  readonly batchSize: number;
  readonly headCount: number;
  readonly headWidth: number;
}

export interface MultiHeadAttentionGradients {
  readonly input: Tensor;
  readonly queryWeight: Tensor;
  readonly queryBias: Tensor | null;
  readonly keyWeight: Tensor;
  readonly keyBias: Tensor | null;
  readonly valueWeight: Tensor;
  readonly valueBias: Tensor | null;
  readonly outputWeight: Tensor;
  readonly outputBias: Tensor | null;
}

export function multiHeadAttentionForward(
  input: Tensor,
  parameters: MultiHeadAttentionParameters,
  headCount: number,
  headWidth: number,
  paddingMask?: Uint8Array,
): { output: Tensor; probabilities: Tensor; cache: MultiHeadAttentionCache } {
  if (input.shape.length !== 3) throw new Error('Multi-head attention input must have rank 3.');
  const [batchSize, sequenceLength, modelWidth] = input.shape;
  if (headCount * headWidth !== modelWidth) {
    throw new Error('Multi-head dimensions do not match the model width.');
  }
  const query = linearForward(input, parameters.queryWeight, parameters.queryBias);
  const key = linearForward(input, parameters.keyWeight, parameters.keyBias);
  const value = linearForward(input, parameters.valueWeight, parameters.valueBias);
  const queryHeads = splitHeads(query.output, headCount, headWidth);
  const keyHeads = splitHeads(key.output, headCount, headWidth);
  const valueHeads = splitHeads(value.output, headCount, headWidth);
  const mask = causalPaddingMask(batchSize, headCount, sequenceLength, paddingMask);
  const attention = scaledDotProductAttentionForward(queryHeads, keyHeads, valueHeads, mask);
  const merged = mergeHeads(attention.output, batchSize, headCount, headWidth);
  const output = linearForward(merged, parameters.outputWeight, parameters.outputBias);
  return {
    output: output.output,
    probabilities: attention.probabilities,
    cache: {
      queryLinear: query.cache,
      keyLinear: key.cache,
      valueLinear: value.cache,
      outputLinear: output.cache,
      attention: attention.cache,
      batchSize,
      headCount,
      headWidth,
    },
  };
}

export function multiHeadAttentionBackward(
  cache: MultiHeadAttentionCache,
  gradient: Tensor,
): MultiHeadAttentionGradients {
  const output = linearBackward(cache.outputLinear, gradient);
  const mergedGradient = splitHeads(output.input, cache.headCount, cache.headWidth);
  const attention = scaledDotProductAttentionBackward(cache.attention, mergedGradient);
  const queryGradient = mergeHeads(
    attention.query,
    cache.batchSize,
    cache.headCount,
    cache.headWidth,
  );
  const keyGradient = mergeHeads(attention.key, cache.batchSize, cache.headCount, cache.headWidth);
  const valueGradient = mergeHeads(
    attention.value,
    cache.batchSize,
    cache.headCount,
    cache.headWidth,
  );
  const query = linearBackward(cache.queryLinear, queryGradient);
  const key = linearBackward(cache.keyLinear, keyGradient);
  const value = linearBackward(cache.valueLinear, valueGradient);
  return {
    input: sumTensorGradients(query.input.shape, query.input, key.input, value.input),
    queryWeight: query.weight,
    queryBias: query.bias,
    keyWeight: key.weight,
    keyBias: key.bias,
    valueWeight: value.weight,
    valueBias: value.bias,
    outputWeight: output.weight,
    outputBias: output.bias,
  };
}
