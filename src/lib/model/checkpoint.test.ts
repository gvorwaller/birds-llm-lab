import { describe, expect, it } from 'vitest';
import { decodeCheckpointWeights, encodeCheckpointWeights } from './checkpoint';
import { DEFAULT_MODEL_CONFIG } from './config';
import { emptyParameters, initializeParameters } from './parameters';

function encodedFixture() {
  const registry = initializeParameters(DEFAULT_MODEL_CONFIG, 123);
  return { registry, ...encodeCheckpointWeights(DEFAULT_MODEL_CONFIG, registry) };
}

describe('checkpoint weight encoding', () => {
  it('writes a contiguous index and explicit little-endian Float32 bytes', () => {
    const registry = emptyParameters(DEFAULT_MODEL_CONFIG);
    registry.get('token_embedding.weight').value.data[0] = 1;
    const encoded = encodeCheckpointWeights(DEFAULT_MODEL_CONFIG, registry);
    expect(Array.from(encoded.bytes.slice(0, 4))).toEqual([0, 0, 128, 63]);
    expect(encoded.index.entries[0]).toMatchObject({
      name: 'token_embedding.weight',
      elementOffset: 0,
      byteOffset: 0,
      byteLength: 1_024 * 64 * 4,
    });
    expect(encoded.index.entries.at(-1)?.byteOffset).toBe(encoded.index.byteLength - 64 * 4);
  });

  it('round-trips every initialized parameter exactly', () => {
    const fixture = encodedFixture();
    const loaded = decodeCheckpointWeights(DEFAULT_MODEL_CONFIG, fixture.index, fixture.bytes);
    expect(encodeCheckpointWeights(DEFAULT_MODEL_CONFIG, loaded).bytes).toEqual(fixture.bytes);
  });

  it('rejects duplicate, missing, unexpected, and shape/count-corrupt entries', () => {
    const fixture = encodedFixture();
    const duplicate = structuredClone(fixture.index);
    duplicate.entries[1].name = duplicate.entries[0].name;
    expect(() => decodeCheckpointWeights(DEFAULT_MODEL_CONFIG, duplicate, fixture.bytes)).toThrow(
      'Duplicate weight index name',
    );

    const missing = structuredClone(fixture.index);
    missing.entries.pop();
    expect(() => decodeCheckpointWeights(DEFAULT_MODEL_CONFIG, missing, fixture.bytes)).toThrow(
      'Missing parameter',
    );

    const unexpected = structuredClone(fixture.index);
    unexpected.entries[0].name = 'surprise.weight';
    expect(() => decodeCheckpointWeights(DEFAULT_MODEL_CONFIG, unexpected, fixture.bytes)).toThrow(
      'Unexpected parameter',
    );

    const shape = structuredClone(fixture.index);
    shape.entries[0].shape = [1, shape.entries[0].elementCount];
    expect(() => decodeCheckpointWeights(DEFAULT_MODEL_CONFIG, shape, fixture.bytes)).toThrow(
      'has shape',
    );

    const count = structuredClone(fixture.index);
    count.entries[0].elementCount -= 1;
    expect(() => decodeCheckpointWeights(DEFAULT_MODEL_CONFIG, count, fixture.bytes)).toThrow(
      'does not match shape count',
    );
  });

  it('rejects overlaps, unindexed gaps, and inconsistent offset units', () => {
    const fixture = encodedFixture();
    const overlap = structuredClone(fixture.index);
    overlap.entries[1].elementOffset = 0;
    overlap.entries[1].byteOffset = 0;
    expect(() => decodeCheckpointWeights(DEFAULT_MODEL_CONFIG, overlap, fixture.bytes)).toThrow(
      'overlaps',
    );

    const gap = structuredClone(fixture.index);
    gap.entries[1].elementOffset += 1;
    gap.entries[1].byteOffset += 4;
    expect(() => decodeCheckpointWeights(DEFAULT_MODEL_CONFIG, gap, fixture.bytes)).toThrow(
      'unindexed bytes',
    );

    const units = structuredClone(fixture.index);
    units.entries[1].elementOffset += 1;
    expect(() => decodeCheckpointWeights(DEFAULT_MODEL_CONFIG, units, fixture.bytes)).toThrow(
      'offsets disagree',
    );
  });

  it('rejects truncated, extra, unsupported-version, and non-finite data', () => {
    const fixture = encodedFixture();
    expect(() =>
      decodeCheckpointWeights(DEFAULT_MODEL_CONFIG, fixture.index, fixture.bytes.slice(0, -4)),
    ).toThrow('truncated');

    const extra = new Uint8Array(fixture.bytes.length + 4);
    extra.set(fixture.bytes);
    expect(() => decodeCheckpointWeights(DEFAULT_MODEL_CONFIG, fixture.index, extra)).toThrow(
      'extra bytes',
    );

    expect(() =>
      decodeCheckpointWeights(
        DEFAULT_MODEL_CONFIG,
        { ...fixture.index, formatVersion: 2 },
        fixture.bytes,
      ),
    ).toThrow('expected 1');

    const nonFinite = fixture.bytes.slice();
    new DataView(nonFinite.buffer).setFloat32(0, Number.POSITIVE_INFINITY, true);
    expect(() => decodeCheckpointWeights(DEFAULT_MODEL_CONFIG, fixture.index, nonFinite)).toThrow(
      'non-finite',
    );
  });
});
