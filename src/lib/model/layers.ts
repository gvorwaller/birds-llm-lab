import { assertShape, elementCount, tensor, type Tensor } from '../math/tensor';

export interface LinearCache {
  readonly input: Tensor;
  readonly weight: Tensor;
  readonly bias: Tensor | null;
  readonly rows: number;
  readonly inputWidth: number;
  readonly outputWidth: number;
}

export interface LinearGradients {
  readonly input: Tensor;
  readonly weight: Tensor;
  readonly bias: Tensor | null;
}

export function linearForward(
  input: Tensor,
  weight: Tensor,
  bias: Tensor | null,
): { output: Tensor; cache: LinearCache } {
  assertShape(input, input.shape, 'Linear input');
  if (input.shape.length === 0) throw new Error('Linear input requires at least one axis.');
  if (weight.shape.length !== 2) {
    throw new Error(`Linear weight must have rank 2; received rank ${weight.shape.length}.`);
  }
  const inputWidth = input.shape.at(-1) as number;
  const [weightInputWidth, outputWidth] = weight.shape;
  assertShape(weight, [weightInputWidth, outputWidth], 'Linear weight');
  if (inputWidth !== weightInputWidth) {
    throw new Error(
      `Linear input width ${inputWidth} does not match weight input width ${weightInputWidth}.`,
    );
  }
  if (bias !== null) assertShape(bias, [outputWidth], 'Linear bias');
  const rows = input.data.length / inputWidth;
  const output = new Float32Array(rows * outputWidth);
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < outputWidth; column += 1) {
      let sum = bias?.data[column] ?? 0;
      for (let inner = 0; inner < inputWidth; inner += 1) {
        sum += input.data[row * inputWidth + inner] * weight.data[inner * outputWidth + column];
      }
      output[row * outputWidth + column] = sum;
    }
  }
  const outputShape = [...input.shape.slice(0, -1), outputWidth];
  return {
    output: tensor(output, outputShape, 'Linear output'),
    cache: { input, weight, bias, rows, inputWidth, outputWidth },
  };
}

export function linearBackward(cache: LinearCache, gradient: Tensor): LinearGradients {
  const outputShape = [...cache.input.shape.slice(0, -1), cache.outputWidth];
  assertShape(gradient, outputShape, 'Linear output gradient');
  const inputGradient = new Float32Array(cache.input.data.length);
  const weightGradient = new Float32Array(cache.weight.data.length);
  const biasGradient = cache.bias === null ? null : new Float32Array(cache.outputWidth);
  for (let row = 0; row < cache.rows; row += 1) {
    for (let column = 0; column < cache.outputWidth; column += 1) {
      const upstream = gradient.data[row * cache.outputWidth + column];
      if (biasGradient !== null) biasGradient[column] += upstream;
      for (let inner = 0; inner < cache.inputWidth; inner += 1) {
        inputGradient[row * cache.inputWidth + inner] +=
          upstream * cache.weight.data[inner * cache.outputWidth + column];
        weightGradient[inner * cache.outputWidth + column] +=
          cache.input.data[row * cache.inputWidth + inner] * upstream;
      }
    }
  }
  return {
    input: tensor(inputGradient, cache.input.shape, 'Linear input gradient'),
    weight: tensor(weightGradient, cache.weight.shape, 'Linear weight gradient'),
    bias:
      biasGradient === null
        ? null
        : tensor(biasGradient, [cache.outputWidth], 'Linear bias gradient'),
  };
}

export function sumTensorGradients(shape: readonly number[], ...values: readonly Tensor[]): Tensor {
  const count = elementCount(shape);
  const output = new Float32Array(count);
  for (const [valueIndex, value] of values.entries()) {
    assertShape(value, shape, `Gradient summand ${valueIndex}`);
    for (let index = 0; index < count; index += 1) output[index] += value.data[index];
  }
  return tensor(output, shape, 'Summed gradient');
}

export interface UnembeddingCache {
  readonly input: Tensor;
  readonly weight: Tensor;
  readonly bias: Tensor | null;
  readonly rows: number;
  readonly modelWidth: number;
  readonly vocabularySize: number;
}

export interface UnembeddingGradients {
  readonly input: Tensor;
  readonly weight: Tensor;
  readonly bias: Tensor | null;
}

export function unembeddingForward(
  input: Tensor,
  weight: Tensor,
  bias: Tensor | null,
): { output: Tensor; cache: UnembeddingCache } {
  if (input.shape.length === 0) throw new Error('Unembedding input requires at least one axis.');
  if (weight.shape.length !== 2) throw new Error('Unembedding weight must have rank 2.');
  const modelWidth = input.shape.at(-1) as number;
  const [vocabularySize, weightWidth] = weight.shape;
  assertShape(input, input.shape, 'Unembedding input');
  assertShape(weight, [vocabularySize, weightWidth], 'Unembedding weight');
  if (weightWidth !== modelWidth) {
    throw new Error('Unembedding weight width does not match the model width.');
  }
  if (bias !== null) assertShape(bias, [vocabularySize], 'Unembedding bias');
  const rows = input.data.length / modelWidth;
  const output = new Float32Array(rows * vocabularySize);
  for (let row = 0; row < rows; row += 1) {
    for (let token = 0; token < vocabularySize; token += 1) {
      let sum = bias?.data[token] ?? 0;
      for (let channel = 0; channel < modelWidth; channel += 1) {
        sum += input.data[row * modelWidth + channel] * weight.data[token * modelWidth + channel];
      }
      output[row * vocabularySize + token] = sum;
    }
  }
  return {
    output: tensor(output, [...input.shape.slice(0, -1), vocabularySize], 'Unembedding output'),
    cache: { input, weight, bias, rows, modelWidth, vocabularySize },
  };
}

export function unembeddingBackward(
  cache: UnembeddingCache,
  gradient: Tensor,
): UnembeddingGradients {
  assertShape(
    gradient,
    [...cache.input.shape.slice(0, -1), cache.vocabularySize],
    'Unembedding output gradient',
  );
  const inputGradient = new Float32Array(cache.input.data.length);
  const weightGradient = new Float32Array(cache.weight.data.length);
  const biasGradient = cache.bias === null ? null : new Float32Array(cache.vocabularySize);
  for (let row = 0; row < cache.rows; row += 1) {
    for (let token = 0; token < cache.vocabularySize; token += 1) {
      const upstream = gradient.data[row * cache.vocabularySize + token];
      if (biasGradient !== null) biasGradient[token] += upstream;
      for (let channel = 0; channel < cache.modelWidth; channel += 1) {
        inputGradient[row * cache.modelWidth + channel] +=
          upstream * cache.weight.data[token * cache.modelWidth + channel];
        weightGradient[token * cache.modelWidth + channel] +=
          cache.input.data[row * cache.modelWidth + channel] * upstream;
      }
    }
  }
  return {
    input: tensor(inputGradient, cache.input.shape, 'Unembedding input gradient'),
    weight: tensor(weightGradient, cache.weight.shape, 'Unembedding weight gradient'),
    bias:
      biasGradient === null
        ? null
        : tensor(biasGradient, [cache.vocabularySize], 'Unembedding bias gradient'),
  };
}
