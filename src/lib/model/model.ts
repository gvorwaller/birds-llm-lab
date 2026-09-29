import type { ModelConfig } from '../data/schemas';
import {
  addForward,
  crossEntropyBackward,
  crossEntropyForward,
  embeddingBackward,
  embeddingForward,
  layerNormBackward,
  layerNormForward,
  type CrossEntropyCache,
  type EmbeddingCache,
  type LayerNormCache,
} from '../math/kernels';
import { assertShape, tensor, type Tensor } from '../math/tensor';
import { PAD_ID } from '../tokenizer/constants';
import {
  transformerBlockBackward,
  transformerBlockForward,
  type TransformerBlockCache,
} from './block';
import { unembeddingBackward, unembeddingForward, type UnembeddingCache } from './layers';
import type { ParameterRegistry } from './parameters';

export interface LanguageModelCache {
  readonly config: ModelConfig;
  readonly registry: ParameterRegistry;
  readonly batchSize: number;
  readonly sequenceLength: number;
  readonly tokenEmbedding: EmbeddingCache;
  readonly positionEmbedding: EmbeddingCache;
  readonly blocks: readonly TransformerBlockCache[];
  readonly finalLayerNorm: LayerNormCache;
  readonly unembedding: UnembeddingCache;
  readonly crossEntropy: CrossEntropyCache | null;
}

export interface LanguageModelForwardResult {
  readonly logits: Tensor;
  readonly loss: number | null;
  readonly cache: LanguageModelCache;
}

export interface TiedEmbeddingGradientContributions {
  readonly lookup: Tensor;
  readonly unembedding: Tensor;
}

export interface LanguageModelBackwardResult {
  readonly parameters: ReadonlyMap<string, Tensor>;
  readonly tiedEmbeddingContributions: TiedEmbeddingGradientContributions | null;
}

function addGradient(gradients: Map<string, Tensor>, name: string, value: Tensor): void {
  const existing = gradients.get(name);
  if (!existing) {
    gradients.set(name, tensor(value.data.slice(), value.shape, `${name} accumulated gradient`));
    return;
  }
  assertShape(value, existing.shape, `${name} gradient contribution`);
  for (let index = 0; index < existing.data.length; index += 1) {
    existing.data[index] += value.data[index];
  }
}

function validateTokenIds(
  tokenIds: readonly number[],
  expected: number,
  vocabularySize: number,
): void {
  if (tokenIds.length !== expected) {
    throw new Error(`Expected ${expected} token IDs; received ${tokenIds.length}.`);
  }
  tokenIds.forEach((tokenId, index) => {
    if (!Number.isInteger(tokenId) || tokenId < 0 || tokenId >= vocabularySize) {
      throw new Error(`Token ID ${tokenId} at flat index ${index} is outside the vocabulary.`);
    }
  });
}

export function languageModelForward(
  tokenIds: readonly number[],
  shape: readonly [number, number],
  registry: ParameterRegistry,
  config: ModelConfig,
  targets?: readonly number[],
): LanguageModelForwardResult {
  const [batchSize, sequenceLength] = shape;
  if (!Number.isInteger(batchSize) || batchSize <= 0) {
    throw new Error('Language-model batch size must be a positive integer.');
  }
  if (
    !Number.isInteger(sequenceLength) ||
    sequenceLength <= 0 ||
    sequenceLength > config.contextLength
  ) {
    throw new Error('Language-model sequence length must fit the configured context.');
  }
  const tokenCount = batchSize * sequenceLength;
  validateTokenIds(tokenIds, tokenCount, config.vocabSize);
  if (targets !== undefined) validateTokenIds(targets, tokenCount, config.vocabSize);

  const tokenEmbedding = embeddingForward(
    registry.get('token_embedding.weight').value,
    tokenIds,
    shape,
  );
  const positionIds = Array.from(
    { length: tokenCount },
    (_, flatIndex) => flatIndex % sequenceLength,
  );
  const positionEmbedding = embeddingForward(
    registry.get('position_embedding.weight').value,
    positionIds,
    shape,
  );
  let hidden = addForward(tokenEmbedding.output, positionEmbedding.output).output;
  const paddingMask = Uint8Array.from(tokenIds, (tokenId) => (tokenId === PAD_ID ? 0 : 1));
  const blocks: TransformerBlockCache[] = [];
  for (let layer = 0; layer < config.nLayers; layer += 1) {
    const block = transformerBlockForward(hidden, registry, config, layer, paddingMask);
    hidden = block.output;
    blocks.push(block.cache);
  }
  const finalLayerNorm = layerNormForward(
    hidden,
    registry.get('final_ln.scale').value,
    registry.get('final_ln.offset').value,
    config.layerNormEpsilon,
  );
  const unembeddingName = config.tiedEmbeddings ? 'token_embedding.weight' : 'unembedding.weight';
  const unembeddingBias =
    !config.tiedEmbeddings && config.useBias ? registry.get('unembedding.bias').value : null;
  const unembedding = unembeddingForward(
    finalLayerNorm.output,
    registry.get(unembeddingName).value,
    unembeddingBias,
  );
  let loss: number | null = null;
  let crossEntropy: CrossEntropyCache | null = null;
  if (targets !== undefined) {
    const lossResult = crossEntropyForward(
      tensor(unembedding.output.data, [tokenCount, config.vocabSize], 'Flattened logits'),
      targets,
      PAD_ID,
    );
    loss = lossResult.output;
    crossEntropy = lossResult.cache;
  }
  return {
    logits: unembedding.output,
    loss,
    cache: {
      config,
      registry,
      batchSize,
      sequenceLength,
      tokenEmbedding: tokenEmbedding.cache,
      positionEmbedding: positionEmbedding.cache,
      blocks,
      finalLayerNorm: finalLayerNorm.cache,
      unembedding: unembedding.cache,
      crossEntropy,
    },
  };
}

export function languageModelBackward(
  cache: LanguageModelCache,
  lossGradient = 1,
): LanguageModelBackwardResult {
  if (cache.crossEntropy === null) {
    throw new Error('Language-model backward requires targets from the forward pass.');
  }
  const gradients = new Map<string, Tensor>();
  const flatLogitsGradient = crossEntropyBackward(cache.crossEntropy, lossGradient);
  const logitsGradient = tensor(
    flatLogitsGradient.data,
    [cache.batchSize, cache.sequenceLength, cache.config.vocabSize],
    'Logits gradient',
  );
  const unembedding = unembeddingBackward(cache.unembedding, logitsGradient);
  const unembeddingName = cache.config.tiedEmbeddings
    ? 'token_embedding.weight'
    : 'unembedding.weight';
  addGradient(gradients, unembeddingName, unembedding.weight);
  if (unembedding.bias !== null) addGradient(gradients, 'unembedding.bias', unembedding.bias);

  const finalLayerNorm = layerNormBackward(cache.finalLayerNorm, unembedding.input);
  addGradient(gradients, 'final_ln.scale', finalLayerNorm.gamma);
  addGradient(gradients, 'final_ln.offset', finalLayerNorm.beta);
  let hiddenGradient = finalLayerNorm.input;
  for (let layer = cache.blocks.length - 1; layer >= 0; layer -= 1) {
    const block = transformerBlockBackward(cache.blocks[layer], hiddenGradient);
    for (const [name, gradient] of block.parameters) addGradient(gradients, name, gradient);
    hiddenGradient = block.input;
  }
  const tokenLookupGradient = embeddingBackward(cache.tokenEmbedding, hiddenGradient);
  const positionGradient = embeddingBackward(cache.positionEmbedding, hiddenGradient);
  addGradient(gradients, 'token_embedding.weight', tokenLookupGradient);
  addGradient(gradients, 'position_embedding.weight', positionGradient);

  cache.registry.zeroGradients();
  for (const parameter of cache.registry) {
    const gradient = gradients.get(parameter.name);
    if (!gradient) throw new Error(`Full backward did not produce ${parameter.name}.`);
    parameter.gradient.data.set(gradient.data);
  }
  return {
    parameters: gradients,
    tiedEmbeddingContributions: cache.config.tiedEmbeddings
      ? { lookup: tokenLookupGradient, unembedding: unembedding.weight }
      : null,
  };
}
