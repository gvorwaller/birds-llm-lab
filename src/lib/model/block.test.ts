import { describe, expect, it } from 'vitest';
import type { ModelConfig } from '../data/schemas';
import { tensor } from '../math/tensor';
import { transformerBlockBackward, transformerBlockForward } from './block';
import { initializeParameters } from './parameters';

const tinyConfig: ModelConfig = {
  formatVersion: 1,
  vocabSize: 259,
  contextLength: 3,
  dModel: 4,
  nLayers: 1,
  nHeads: 2,
  dHead: 2,
  dMlp: 5,
  tiedEmbeddings: true,
  useBias: true,
  layerNormEpsilon: 1e-5,
  gelu: 'tanh-approximation',
};

type ReferenceParameters = Map<string, number[]>;

function referenceLinear(
  input: readonly number[],
  rows: number,
  inputWidth: number,
  outputWidth: number,
  weight: readonly number[],
  bias: readonly number[],
): number[] {
  const output = Array<number>(rows * outputWidth).fill(0);
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < outputWidth; column += 1) {
      let sum = bias[column];
      for (let inner = 0; inner < inputWidth; inner += 1) {
        sum += input[row * inputWidth + inner] * weight[inner * outputWidth + column];
      }
      output[row * outputWidth + column] = sum;
    }
  }
  return output;
}

function referenceLayerNorm(
  input: readonly number[],
  rows: number,
  width: number,
  scale: readonly number[],
  offset: readonly number[],
): number[] {
  const output = Array<number>(input.length).fill(0);
  for (let row = 0; row < rows; row += 1) {
    let mean = 0;
    for (let channel = 0; channel < width; channel += 1) {
      mean += input[row * width + channel];
    }
    mean /= width;
    let variance = 0;
    for (let channel = 0; channel < width; channel += 1) {
      variance += (input[row * width + channel] - mean) ** 2;
    }
    const inverse = 1 / Math.sqrt(variance / width + tinyConfig.layerNormEpsilon);
    for (let channel = 0; channel < width; channel += 1) {
      output[row * width + channel] =
        (input[row * width + channel] - mean) * inverse * scale[channel] + offset[channel];
    }
  }
  return output;
}

function parameter(parameters: ReferenceParameters, name: string): number[] {
  const value = parameters.get(name);
  if (!value) throw new Error(`Missing reference parameter ${name}.`);
  return value;
}

function referenceAttention(
  input: readonly number[],
  parameters: ReferenceParameters,
  padding: Uint8Array,
): number[] {
  const tokenCount = 3;
  const width = tinyConfig.dModel;
  const prefix = 'blocks.0.attn';
  const query = referenceLinear(
    input,
    tokenCount,
    width,
    width,
    parameter(parameters, `${prefix}.q.weight`),
    parameter(parameters, `${prefix}.q.bias`),
  );
  const key = referenceLinear(
    input,
    tokenCount,
    width,
    width,
    parameter(parameters, `${prefix}.k.weight`),
    parameter(parameters, `${prefix}.k.bias`),
  );
  const value = referenceLinear(
    input,
    tokenCount,
    width,
    width,
    parameter(parameters, `${prefix}.v.weight`),
    parameter(parameters, `${prefix}.v.bias`),
  );
  const mergedContext = Array<number>(tokenCount * width).fill(0);
  for (let head = 0; head < tinyConfig.nHeads; head += 1) {
    for (let queryToken = 0; queryToken < tokenCount; queryToken += 1) {
      if (padding[queryToken] === 0) continue;
      const scores = Array<number>(tokenCount).fill(Number.NEGATIVE_INFINITY);
      for (let keyToken = 0; keyToken <= queryToken; keyToken += 1) {
        if (padding[keyToken] === 0) continue;
        let score = 0;
        for (let channel = 0; channel < tinyConfig.dHead; channel += 1) {
          const modelChannel = head * tinyConfig.dHead + channel;
          score += query[queryToken * width + modelChannel] * key[keyToken * width + modelChannel];
        }
        scores[keyToken] = score / Math.sqrt(tinyConfig.dHead);
      }
      const maximum = Math.max(...scores);
      const exponentials = scores.map((score) =>
        Number.isFinite(score) ? Math.exp(score - maximum) : 0,
      );
      const total = exponentials.reduce((sum, item) => sum + item, 0);
      for (let channel = 0; channel < tinyConfig.dHead; channel += 1) {
        let context = 0;
        for (let keyToken = 0; keyToken < tokenCount; keyToken += 1) {
          const modelChannel = head * tinyConfig.dHead + channel;
          context += (exponentials[keyToken] / total) * value[keyToken * width + modelChannel];
        }
        mergedContext[queryToken * width + head * tinyConfig.dHead + channel] = context;
      }
    }
  }
  return referenceLinear(
    mergedContext,
    tokenCount,
    width,
    width,
    parameter(parameters, `${prefix}.out.weight`),
    parameter(parameters, `${prefix}.out.bias`),
  );
}

function referenceBlock(
  input: readonly number[],
  parameters: ReferenceParameters,
  padding: Uint8Array,
): number[] {
  const rows = 3;
  const width = tinyConfig.dModel;
  const prefix = 'blocks.0';
  const ln1 = referenceLayerNorm(
    input,
    rows,
    width,
    parameter(parameters, `${prefix}.ln1.scale`),
    parameter(parameters, `${prefix}.ln1.offset`),
  );
  const attention = referenceAttention(ln1, parameters, padding);
  const residual = input.map((value, index) => value + attention[index]);
  const ln2 = referenceLayerNorm(
    residual,
    rows,
    width,
    parameter(parameters, `${prefix}.ln2.scale`),
    parameter(parameters, `${prefix}.ln2.offset`),
  );
  const hidden = referenceLinear(
    ln2,
    rows,
    width,
    tinyConfig.dMlp,
    parameter(parameters, `${prefix}.mlp.in.weight`),
    parameter(parameters, `${prefix}.mlp.in.bias`),
  );
  const factor = Math.sqrt(2 / Math.PI);
  const activated = hidden.map(
    (value) => 0.5 * value * (1 + Math.tanh(factor * (value + 0.044715 * value * value * value))),
  );
  const projected = referenceLinear(
    activated,
    rows,
    tinyConfig.dMlp,
    width,
    parameter(parameters, `${prefix}.mlp.out.weight`),
    parameter(parameters, `${prefix}.mlp.out.bias`),
  );
  return residual.map((value, index) => value + projected[index]);
}

function objective(
  input: readonly number[],
  parameters: ReferenceParameters,
  padding: Uint8Array,
  upstream: readonly number[],
): number {
  return referenceBlock(input, parameters, padding).reduce(
    (sum, value, index) => sum + value * upstream[index],
    0,
  );
}

function assertGradient(
  label: string,
  analytic: readonly number[] | Float32Array,
  numeric: readonly number[],
): void {
  expect(analytic, `${label} gradient length`).toHaveLength(numeric.length);
  for (let index = 0; index < numeric.length; index += 1) {
    const absoluteError = Math.abs(analytic[index] - numeric[index]);
    const denominator = Math.max(1e-6, Math.abs(analytic[index]) + Math.abs(numeric[index]));
    const relativeError = absoluteError / denominator;
    const nearZero = Math.abs(analytic[index]) + Math.abs(numeric[index]) < 1e-3;
    const passes = nearZero ? absoluteError < 2e-5 : relativeError < 1e-3;
    expect(
      passes,
      `${label}[${index}] analytic=${analytic[index]} numeric=${numeric[index]} absolute=${absoluteError} relative=${relativeError}`,
    ).toBe(true);
  }
}

describe('pre-LayerNorm transformer block', () => {
  it('matches an independent Float64 reference and central differences for every group', () => {
    const registry = initializeParameters(tinyConfig, 0x5eed);
    const inputValues = [-0.4, 0.2, 0.7, -0.1, 0.3, -0.8, 0.5, 1.1, -0.2, 0.9, -0.6, 0.4];
    const upstreamValues = [0.7, -0.3, 0.2, 1.1, -0.4, 0.8, -0.9, 0.5, 0.6, -0.2, 1.3, -0.7];
    const padding = new Uint8Array([1, 1, 0]);
    const forward = transformerBlockForward(
      tensor(inputValues, [1, 3, 4]),
      registry,
      tinyConfig,
      0,
      padding,
    );
    const backward = transformerBlockBackward(forward.cache, tensor(upstreamValues, [1, 3, 4]));
    const parameters: ReferenceParameters = new Map(
      registry
        .entries()
        .filter(({ name }) => name.startsWith('blocks.0.'))
        .map(({ name, value }) => [name, Array.from(value.data)]),
    );
    const referenceOutput = referenceBlock(inputValues, parameters, padding);
    referenceOutput.forEach((value, index) =>
      expect(forward.output.data[index]).toBeCloseTo(value, 5),
    );

    const epsilon = 1e-4;
    const numericInput = inputValues.map((original, index) => {
      const positive = [...inputValues];
      const negative = [...inputValues];
      positive[index] = original + epsilon;
      negative[index] = original - epsilon;
      return (
        (objective(positive, parameters, padding, upstreamValues) -
          objective(negative, parameters, padding, upstreamValues)) /
        (2 * epsilon)
      );
    });
    assertGradient('input', backward.input.data, numericInput);

    expect([...backward.parameters.keys()].sort()).toEqual([...parameters.keys()].sort());
    for (const [name, values] of parameters) {
      const analytic = backward.parameters.get(name);
      if (!analytic) throw new Error(`Missing analytic gradient for ${name}.`);
      const numeric = values.map((original, index) => {
        values[index] = original + epsilon;
        const positive = objective(inputValues, parameters, padding, upstreamValues);
        values[index] = original - epsilon;
        const negative = objective(inputValues, parameters, padding, upstreamValues);
        values[index] = original;
        return (positive - negative) / (2 * epsilon);
      });
      assertGradient(name, analytic.data, numeric);
    }
  });

  it('rejects invalid layer, width, context, and padding-mask dimensions', () => {
    const registry = initializeParameters(tinyConfig, 1);
    expect(() =>
      transformerBlockForward(tensor(new Float32Array(12), [1, 3, 4]), registry, tinyConfig, 1),
    ).toThrow('outside the configured model');
    expect(() =>
      transformerBlockForward(tensor(new Float32Array(9), [1, 3, 3]), registry, tinyConfig, 0),
    ).toThrow('must have shape');
    expect(() =>
      transformerBlockForward(tensor(new Float32Array(16), [1, 4, 4]), registry, tinyConfig, 0),
    ).toThrow('exceeds the configured context');
    expect(() =>
      transformerBlockForward(
        tensor(new Float32Array(12), [1, 3, 4]),
        registry,
        tinyConfig,
        0,
        new Uint8Array(2),
      ),
    ).toThrow('Padding mask shape');
  });
});
