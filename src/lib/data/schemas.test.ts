import malformedSection from '../../../tests/fixtures/corpus-malformed-section.json';
import validCorpusRow from '../../../tests/fixtures/corpus-valid.json';
import { describe, expect, it } from 'vitest';
import {
  ArtifactValidationError,
  parseCheckpointConfig,
  parseCorpusManifest,
  parseCorpusRow,
  parseModelConfig,
  parseTokenizerArtifact,
  parseTrainingLog,
  parseWeightIndex,
  serializeCorpusRow,
  serializeStableJson,
} from './schemas';

const sha256 = 'a'.repeat(64);

describe('corpus row schema', () => {
  it('accepts Unicode, empty optionals, and literal special-token text', () => {
    const parsed = parseCorpusRow(validCorpusRow);
    expect(parsed.code).toBe('spécial');
    expect(parsed.family).toBe('');
    expect(parsed.extract).toContain('<|bos|>');
    expect(parseCorpusRow(JSON.parse(serializeCorpusRow(parsed)))).toEqual(parsed);
  });

  it('rejects malformed section input with an exact path', () => {
    expect(() => parseCorpusRow(malformedSection)).toThrowError(
      new ArtifactValidationError('$.sections[0].text', 'expected a non-empty string'),
    );
  });

  it('rejects unexpected properties rather than silently dropping them', () => {
    expect(() => parseCorpusRow({ ...validCorpusRow, password: 'not allowed' })).toThrow(
      '$.password: unexpected property',
    );
  });
});

describe('artifact schemas', () => {
  it('validates a complete manifest and its fixed local database identity', () => {
    const manifest = parseCorpusManifest({
      formatVersion: 1,
      exportedAt: '2026-09-28T12:00:00.000Z',
      database: { host: '127.0.0.1', port: 15436, name: 'birds_test' },
      sourceRows: 1,
      emittedRows: 1,
      skippedRows: 0,
      skippedReasons: {},
      malformedSectionsSkipped: 0,
      splits: {
        train: { documents: 1, bytes: 10 },
        validation: { documents: 0, bytes: 0 },
        test: { documents: 0, bytes: 0 },
      },
      sourceBytes: { wikipediaExtracts: 10, wikipediaSections: 0, fieldCraft: 0 },
      corpusBytes: 10,
      corpusSha256: sha256,
      splitAlgorithm: 'sha256-bucket',
      splitVersion: 'split-v1',
      splitOverrides: [],
      templateVersion: 'corpus-v1',
      sources: {
        wikipediaExtracts: true,
        wikipediaSections: true,
        fieldCraftExported: true,
        fieldCraftIncludedInTraining: false,
      },
      gitRevision: null,
      warnings: [],
    });
    expect(manifest.database.name).toBe('birds_test');
    expect(() =>
      parseCorpusManifest({
        ...manifest,
        database: { ...manifest.database, host: 'db.example.test' },
      }),
    ).toThrow('$.database.host: expected "127.0.0.1"');
  });

  it('validates tokenizer artifacts', () => {
    const artifact = parseTokenizerArtifact({
      formatVersion: 1,
      specialIds: { bos: 256, eos: 257, pad: 258 },
      targetSize: 260,
      merges: [{ left: 65, right: 66, id: 259, trainingCount: 3 }],
      tokens: [
        ...Array.from({ length: 256 }, (_, id) => ({ id, bytes: [id], display: '' })),
        { id: 256, bytes: [], display: '<|bos|>' },
        { id: 257, bytes: [], display: '<|eos|>' },
        { id: 258, bytes: [], display: '<|pad|>' },
        { id: 259, bytes: [65, 66], display: 'AB' },
      ],
      corpusSha256: sha256,
      trainer: {
        algorithmVersion: 'bpe-v1',
        split: 'train',
        templateVersion: 'corpus-v1',
        documentCount: 2,
        byteCount: 12,
      },
    });
    expect(artifact.merges[0].id).toBe(259);
    expect(() =>
      parseTokenizerArtifact({
        ...artifact,
        tokens: artifact.tokens.slice(0, -1),
      }),
    ).toThrow('length must equal targetSize');
  });

  it('validates model dimensions', () => {
    const config = parseModelConfig({
      formatVersion: 1,
      vocabSize: 1024,
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
    expect(config.dModel).toBe(64);
    expect(() => parseModelConfig({ ...config, dHead: 15 })).toThrow(
      'nHeads * dHead must equal dModel',
    );
  });

  it('validates the complete checkpoint configuration and optimizer bounds', () => {
    const model = parseModelConfig({
      formatVersion: 1,
      vocabSize: 1024,
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
    const checkpoint = {
      formatVersion: 1,
      model,
      optimizer: {
        name: 'adamw',
        beta1: 0.9,
        beta2: 0.95,
        epsilon: 1e-8,
        weightDecay: 0.1,
        gradientClipNorm: 1,
        learningRate: 3e-4,
        warmupSteps: 10,
        totalSteps: 100,
      },
      seed: 42,
      corpusSha256: sha256,
      tokenizerSha256: 'b'.repeat(64),
      trainingStep: 25,
      sourceGitRevision: 'c'.repeat(40),
    };
    expect(parseCheckpointConfig(checkpoint).trainingStep).toBe(25);
    expect(() =>
      parseCheckpointConfig({
        ...checkpoint,
        optimizer: { ...checkpoint.optimizer, warmupSteps: 100 },
      }),
    ).toThrow('warmupSteps must be smaller than totalSteps');
    expect(() =>
      parseCheckpointConfig({
        ...checkpoint,
        optimizer: { ...checkpoint.optimizer, beta1: 1 },
      }),
    ).toThrow('strictly between 0 and 1');
    expect(() => parseCheckpointConfig({ ...checkpoint, tokenizerSha256: 'not-a-hash' })).toThrow(
      'SHA-256',
    );
  });

  it('validates weight indexes and training logs', () => {
    const index = parseWeightIndex({
      formatVersion: 1,
      byteLength: 16,
      entries: [
        {
          name: 'embedding.weight',
          shape: [2, 2],
          elementOffset: 0,
          elementCount: 4,
          byteOffset: 0,
          byteLength: 16,
        },
      ],
    });
    const log = parseTrainingLog({
      formatVersion: 1,
      entries: [
        {
          step: 0,
          split: 'train',
          meanLoss: 6.9,
          perplexity: 992,
          learningRate: 0,
          gradientNorm: 1.2,
          elapsedMs: 10,
          sample: null,
        },
      ],
    });
    expect(index.entries[0].shape).toEqual([2, 2]);
    expect(log.entries[0].sample).toBeNull();
  });
});

describe('stable serialization', () => {
  it('sorts object keys recursively and preserves array order', () => {
    const left = serializeStableJson({ z: 1, nested: { b: 2, a: 1 }, values: [3, 2, 1] });
    const right = serializeStableJson({ values: [3, 2, 1], nested: { a: 1, b: 2 }, z: 1 });
    expect(left).toBe(right);
    expect(left).toBe(
      '{\n  "nested": {\n    "a": 1,\n    "b": 2\n  },\n  "values": [\n    3,\n    2,\n    1\n  ],\n  "z": 1\n}\n',
    );
  });
});
