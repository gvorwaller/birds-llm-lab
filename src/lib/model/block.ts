import {
  addForward,
  geluBackward,
  geluForward,
  layerNormBackward,
  layerNormForward,
  type LayerNormCache,
} from '../math/kernels';
import { assertShape, type Tensor } from '../math/tensor';
import {
  multiHeadAttentionBackward,
  multiHeadAttentionForward,
  type MultiHeadAttentionCache,
  type MultiHeadAttentionParameters,
} from './attention';
import type { ModelConfig } from '../data/schemas';
import { linearBackward, linearForward, sumTensorGradients, type LinearCache } from './layers';
import type { ParameterRegistry } from './parameters';

export interface TransformerBlockCache {
  readonly input: Tensor;
  readonly residualAfterAttention: Tensor;
  readonly ln1: LayerNormCache;
  readonly attention: MultiHeadAttentionCache;
  readonly ln2: LayerNormCache;
  readonly mlpInput: LinearCache;
  readonly geluInput: Tensor;
  readonly mlpOutput: LinearCache;
  readonly prefix: string;
}

export interface TransformerBlockGradients {
  readonly input: Tensor;
  readonly parameters: ReadonlyMap<string, Tensor>;
}

function optionalParameter(
  registry: ParameterRegistry,
  name: string,
  enabled: boolean,
): Tensor | null {
  return enabled ? registry.get(name).value : null;
}

function attentionParameters(
  registry: ParameterRegistry,
  prefix: string,
  useBias: boolean,
): MultiHeadAttentionParameters {
  return {
    queryWeight: registry.get(`${prefix}.attn.q.weight`).value,
    queryBias: optionalParameter(registry, `${prefix}.attn.q.bias`, useBias),
    keyWeight: registry.get(`${prefix}.attn.k.weight`).value,
    keyBias: optionalParameter(registry, `${prefix}.attn.k.bias`, useBias),
    valueWeight: registry.get(`${prefix}.attn.v.weight`).value,
    valueBias: optionalParameter(registry, `${prefix}.attn.v.bias`, useBias),
    outputWeight: registry.get(`${prefix}.attn.out.weight`).value,
    outputBias: optionalParameter(registry, `${prefix}.attn.out.bias`, useBias),
  };
}

export function transformerBlockForward(
  input: Tensor,
  registry: ParameterRegistry,
  config: ModelConfig,
  layer: number,
  paddingMask?: Uint8Array,
): { output: Tensor; attentionProbabilities: Tensor; cache: TransformerBlockCache } {
  if (!Number.isInteger(layer) || layer < 0 || layer >= config.nLayers) {
    throw new Error(`Transformer layer ${layer} is outside the configured model.`);
  }
  if (input.shape.length !== 3) throw new Error('Transformer block input must have rank 3.');
  const [batchSize, sequenceLength, modelWidth] = input.shape;
  assertShape(input, [batchSize, sequenceLength, config.dModel], 'Transformer block input');
  if (modelWidth !== config.dModel) throw new Error('Transformer block model width mismatch.');
  if (sequenceLength > config.contextLength) {
    throw new Error('Transformer block sequence exceeds the configured context length.');
  }
  const prefix = `blocks.${layer}`;
  const ln1 = layerNormForward(
    input,
    registry.get(`${prefix}.ln1.scale`).value,
    registry.get(`${prefix}.ln1.offset`).value,
    config.layerNormEpsilon,
  );
  const attention = multiHeadAttentionForward(
    ln1.output,
    attentionParameters(registry, prefix, config.useBias),
    config.nHeads,
    config.dHead,
    paddingMask,
  );
  const residualAfterAttention = addForward(input, attention.output).output;
  const ln2 = layerNormForward(
    residualAfterAttention,
    registry.get(`${prefix}.ln2.scale`).value,
    registry.get(`${prefix}.ln2.offset`).value,
    config.layerNormEpsilon,
  );
  const mlpInput = linearForward(
    ln2.output,
    registry.get(`${prefix}.mlp.in.weight`).value,
    optionalParameter(registry, `${prefix}.mlp.in.bias`, config.useBias),
  );
  const activated = geluForward(mlpInput.output);
  const mlpOutput = linearForward(
    activated.output,
    registry.get(`${prefix}.mlp.out.weight`).value,
    optionalParameter(registry, `${prefix}.mlp.out.bias`, config.useBias),
  );
  const output = addForward(residualAfterAttention, mlpOutput.output).output;
  return {
    output,
    attentionProbabilities: attention.probabilities,
    cache: {
      input,
      residualAfterAttention,
      ln1: ln1.cache,
      attention: attention.cache,
      ln2: ln2.cache,
      mlpInput: mlpInput.cache,
      geluInput: activated.cache,
      mlpOutput: mlpOutput.cache,
      prefix,
    },
  };
}

function put(gradients: Map<string, Tensor>, name: string, value: Tensor | null): void {
  if (value !== null) gradients.set(name, value);
}

export function transformerBlockBackward(
  cache: TransformerBlockCache,
  gradient: Tensor,
): TransformerBlockGradients {
  assertShape(gradient, cache.input.shape, 'Transformer block output gradient');
  const gradients = new Map<string, Tensor>();

  const mlpOutput = linearBackward(cache.mlpOutput, gradient);
  put(gradients, `${cache.prefix}.mlp.out.weight`, mlpOutput.weight);
  put(gradients, `${cache.prefix}.mlp.out.bias`, mlpOutput.bias);
  const activationGradient = geluBackward(cache.geluInput, mlpOutput.input);
  const mlpInput = linearBackward(cache.mlpInput, activationGradient);
  put(gradients, `${cache.prefix}.mlp.in.weight`, mlpInput.weight);
  put(gradients, `${cache.prefix}.mlp.in.bias`, mlpInput.bias);
  const ln2 = layerNormBackward(cache.ln2, mlpInput.input);
  put(gradients, `${cache.prefix}.ln2.scale`, ln2.gamma);
  put(gradients, `${cache.prefix}.ln2.offset`, ln2.beta);
  const residualGradient = sumTensorGradients(gradient.shape, gradient, ln2.input);

  const attention = multiHeadAttentionBackward(cache.attention, residualGradient);
  put(gradients, `${cache.prefix}.attn.q.weight`, attention.queryWeight);
  put(gradients, `${cache.prefix}.attn.q.bias`, attention.queryBias);
  put(gradients, `${cache.prefix}.attn.k.weight`, attention.keyWeight);
  put(gradients, `${cache.prefix}.attn.k.bias`, attention.keyBias);
  put(gradients, `${cache.prefix}.attn.v.weight`, attention.valueWeight);
  put(gradients, `${cache.prefix}.attn.v.bias`, attention.valueBias);
  put(gradients, `${cache.prefix}.attn.out.weight`, attention.outputWeight);
  put(gradients, `${cache.prefix}.attn.out.bias`, attention.outputBias);
  const ln1 = layerNormBackward(cache.ln1, attention.input);
  put(gradients, `${cache.prefix}.ln1.scale`, ln1.gamma);
  put(gradients, `${cache.prefix}.ln1.offset`, ln1.beta);

  return {
    input: sumTensorGradients(residualGradient.shape, residualGradient, ln1.input),
    parameters: gradients,
  };
}
