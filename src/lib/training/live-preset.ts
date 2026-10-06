import { encodeByteTokens } from '../tokenizer/codec';
import type { LiveTrainingConfig } from './worker-protocol';

const trainingNames = [
  'robin',
  'osprey',
  'sparrow',
  'wren',
  'swallow',
  'finch',
  'warbler',
  'oriole',
  'tern',
  'gull',
  'owl',
  'falcon',
] as const;

const validationNames = ['heron', 'egret', 'puffin', 'kingfisher'] as const;

function teachingLine(name: string): number[] {
  return encodeByteTokens(`Bird name: ${name}.\n`, { bos: true, eos: true });
}

/** A deliberately small, bundled text preset; no database or saved checkpoint is read. */
export function smallBirdNamePreset(displayEvery = 10): LiveTrainingConfig {
  return {
    model: {
      formatVersion: 1,
      vocabSize: 259,
      contextLength: 16,
      dModel: 12,
      nLayers: 1,
      nHeads: 3,
      dHead: 4,
      dMlp: 24,
      tiedEmbeddings: true,
      useBias: true,
      layerNormEpsilon: 1e-5,
      gelu: 'tanh-approximation',
    },
    sequences: trainingNames.map(teachingLine),
    validationSequences: validationNames.map(teachingLine),
    samplePromptIds: encodeByteTokens('Bird name: ', { bos: true }),
    sampleNewTokens: 18,
    selectedWeightName: 'blocks.0.attn.q.weight',
    seed: 0x5eed,
    batchSize: 4,
    totalSteps: 300,
    displayEvery,
    baseLearningRate: 0.003,
    warmupSteps: 12,
    optimizer: {
      beta1: 0.9,
      beta2: 0.95,
      epsilon: 1e-8,
      weightDecay: 0.1,
      gradientClipNorm: 1,
    },
  };
}

export const SMALL_BIRD_NAME_PRESET_SUMMARY = Object.freeze({
  trainingNames: [...trainingNames],
  validationNames: [...validationNames],
  template: 'Bird name: {name}.',
});
