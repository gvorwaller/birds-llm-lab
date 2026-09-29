import { createHash } from 'node:crypto';
import { describe, expect, it } from 'vitest';
import { serializeStableJson, type CorpusManifest } from '../data/schemas';
import { BASE_VOCAB_SIZE } from './constants';
import {
  assertArtifactCorpusHash,
  decodeText,
  encodeText,
  encodeTextReference,
  replayMerges,
  trainBpe,
  trainBpeReference,
} from './bpe';

const corpusSha256 = createHash('sha256').update('fixture').digest('hex');

function train(documents: readonly string[], targetSize = BASE_VOCAB_SIZE + 8) {
  return trainBpe(documents, { targetSize, corpusSha256 });
}

describe('deterministic BPE training', () => {
  it('keeps pair counts inside documents and outside special tokens', () => {
    const artifact = train(['ab', 'cd'], BASE_VOCAB_SIZE + 1);
    expect(artifact.merges[0]).toMatchObject({ left: 97, right: 98, trainingCount: 1 });
    expect(artifact.merges[0]).not.toMatchObject({ left: 98, right: 99 });
  });

  it('uses lexicographic token-id tie breaking', () => {
    const artifact = train(['ba', 'ab'], BASE_VOCAB_SIZE + 1);
    expect(artifact.merges[0]).toMatchObject({ left: 97, right: 98, trainingCount: 1 });
  });

  it('counts overlapping repeated pairs and replaces them left-to-right', () => {
    const artifact = train(['aaaa'], BASE_VOCAB_SIZE + 2);
    expect(artifact.merges[0]).toMatchObject({ left: 97, right: 97, trainingCount: 3 });
    expect(decodeText(encodeText('aaaa', artifact), artifact)).toBe('aaaa');
  });

  it('handles Unicode, empty documents, and minimum target size', () => {
    const artifact = train(['', '🐦 café'], BASE_VOCAB_SIZE);
    expect(artifact.merges).toHaveLength(0);
    expect(artifact.tokens).toHaveLength(BASE_VOCAB_SIZE);
    expect(decodeText(encodeText('🐦 café', artifact), artifact)).toBe('🐦 café');
  });

  it('matches the deliberately simple reference trainer', () => {
    const documents = ['banana bandana', 'bananas', '🐦 banana'];
    const options = { targetSize: BASE_VOCAB_SIZE + 12, corpusSha256 };
    expect(trainBpe(documents, options)).toEqual(trainBpeReference(documents, options));
  });

  it('matches the reference across varied deterministic small corpora', () => {
    let state = 0x51f15e;
    const next = () => {
      state = (Math.imul(state, 1_664_525) + 1_013_904_223) >>> 0;
      return state;
    };
    for (let run = 0; run < 20; run += 1) {
      const documents = Array.from({ length: 1 + (next() % 5) }, () =>
        Array.from({ length: next() % 40 }, () => String.fromCharCode(97 + (next() % 5))).join(''),
      );
      const options = { targetSize: BASE_VOCAB_SIZE + 16, corpusSha256 };
      expect(trainBpe(documents, options)).toEqual(trainBpeReference(documents, options));
    }
  });

  it('produces byte-identical stable artifacts for identical input', () => {
    const documents = ['warbler warbler', 'osprey'];
    expect(serializeStableJson(train(documents))).toBe(serializeStableJson(train(documents)));
  });
});

describe('rank-based encoding and replay', () => {
  const artifact = train(['warbler warbler', 'warble', 'bird'], BASE_VOCAB_SIZE + 14);

  it.each(['', 'warbler', 'rare 🐦', '<|bos|>', 'warbler\nwarble'])(
    'matches the reference encoder for %j',
    (text) => {
      expect(encodeText(text, artifact)).toEqual(encodeTextReference(text, artifact));
      expect(decodeText(encodeText(text, artifact), artifact)).toBe(text);
    },
  );

  it('derives reversible replay snapshots without mutating earlier steps', () => {
    const steps = replayMerges('warbler', artifact);
    expect(steps.length).toBeGreaterThan(0);
    for (const [index, step] of steps.entries()) {
      expect(step.applications).toBeGreaterThan(0);
      if (index > 0) expect(step.before).toEqual(steps[index - 1].after);
    }
    expect(decodeText(steps.at(-1)?.after ?? [], artifact)).toBe('warbler');
  });

  it('refuses a tokenizer from a different corpus', () => {
    const manifest = {
      corpusSha256: '0'.repeat(64),
    } as CorpusManifest;
    expect(() => assertArtifactCorpusHash(artifact, manifest)).toThrow('hash mismatch');
  });
});
