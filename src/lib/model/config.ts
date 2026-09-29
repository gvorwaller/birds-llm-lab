import { ARTIFACT_FORMAT_VERSION, parseModelConfig, type ModelConfig } from '../data/schemas';

export const DEFAULT_MODEL_CONFIG: ModelConfig = Object.freeze({
  formatVersion: ARTIFACT_FORMAT_VERSION,
  vocabSize: 1_024,
  contextLength: 64,
  dModel: 64,
  nLayers: 2,
  nHeads: 4,
  dHead: 16,
  dMlp: 256,
  tiedEmbeddings: true,
  useBias: true,
  layerNormEpsilon: 1e-5,
  gelu: 'tanh-approximation',
});

export function validatedModelConfig(config: ModelConfig): ModelConfig {
  return parseModelConfig(config);
}
