import {
  ARTIFACT_FORMAT_VERSION,
  parseWeightIndex,
  type ModelConfig,
  type WeightIndexArtifact,
} from '../data/schemas';
import { assertFinite, elementCount } from '../math/tensor';
import { emptyParameters, parameterSpecs, type ParameterRegistry } from './parameters';

export class CheckpointValidationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'CheckpointValidationError';
  }
}

function sameShape(left: readonly number[], right: readonly number[]): boolean {
  return left.length === right.length && left.every((dimension, axis) => dimension === right[axis]);
}

function validateRegistry(config: ModelConfig, registry: ParameterRegistry): void {
  const expected = parameterSpecs(config);
  const actual = registry.entries();
  if (actual.length !== expected.length) {
    throw new CheckpointValidationError(
      `Parameter registry has ${actual.length} entries; model config requires ${expected.length}.`,
    );
  }
  expected.forEach((specification, index) => {
    const parameter = actual[index];
    if (parameter.name !== specification.name || !sameShape(parameter.shape, specification.shape)) {
      throw new CheckpointValidationError(
        `Parameter registry entry ${index} must be ${specification.name} [${specification.shape.join(', ')}].`,
      );
    }
    assertFinite(parameter.value.data, `Parameter ${parameter.name}`);
  });
}

export function encodeCheckpointWeights(
  config: ModelConfig,
  registry: ParameterRegistry,
): { index: WeightIndexArtifact; bytes: Uint8Array } {
  validateRegistry(config, registry);
  const byteLength = registry.elementCount * Float32Array.BYTES_PER_ELEMENT;
  const bytes = new Uint8Array(byteLength);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let elementOffset = 0;
  const entries = registry.entries().map((parameter) => {
    const count = parameter.value.data.length;
    const byteOffset = elementOffset * Float32Array.BYTES_PER_ELEMENT;
    for (let index = 0; index < count; index += 1) {
      view.setFloat32(
        byteOffset + index * Float32Array.BYTES_PER_ELEMENT,
        parameter.value.data[index],
        true,
      );
    }
    const entry = {
      name: parameter.name,
      shape: [...parameter.shape],
      elementOffset,
      elementCount: count,
      byteOffset,
      byteLength: count * Float32Array.BYTES_PER_ELEMENT,
    };
    elementOffset += count;
    return entry;
  });
  return {
    index: { formatVersion: ARTIFACT_FORMAT_VERSION, byteLength, entries },
    bytes,
  };
}

interface ValidatedRange {
  readonly name: string;
  readonly start: number;
  readonly end: number;
}

export function decodeCheckpointWeights(
  config: ModelConfig,
  indexInput: unknown,
  bytes: Uint8Array,
): ParameterRegistry {
  const index = parseWeightIndex(indexInput);
  if (bytes.byteLength !== index.byteLength) {
    const kind = bytes.byteLength < index.byteLength ? 'truncated' : 'contains extra bytes';
    throw new CheckpointValidationError(
      `weights.bin is ${kind}: index declares ${index.byteLength} bytes, file has ${bytes.byteLength}.`,
    );
  }
  const expectedSpecs = parameterSpecs(config);
  const expectedByName = new Map(
    expectedSpecs.map((specification) => [specification.name, specification]),
  );
  const seen = new Set<string>();
  const ranges: ValidatedRange[] = [];
  for (const entry of index.entries) {
    if (seen.has(entry.name)) {
      throw new CheckpointValidationError(`Duplicate weight index name: ${entry.name}.`);
    }
    seen.add(entry.name);
    const expected = expectedByName.get(entry.name);
    if (!expected)
      throw new CheckpointValidationError(`Unexpected parameter in weight index: ${entry.name}.`);
    if (!sameShape(entry.shape, expected.shape)) {
      throw new CheckpointValidationError(
        `${entry.name} has shape [${entry.shape.join(', ')}]; expected [${expected.shape.join(', ')}].`,
      );
    }
    const expectedCount = elementCount(entry.shape);
    if (entry.elementCount !== expectedCount) {
      throw new CheckpointValidationError(
        `${entry.name} elementCount ${entry.elementCount} does not match shape count ${expectedCount}.`,
      );
    }
    const expectedByteLength = expectedCount * Float32Array.BYTES_PER_ELEMENT;
    if (entry.byteLength !== expectedByteLength) {
      throw new CheckpointValidationError(
        `${entry.name} byteLength ${entry.byteLength} does not match ${expectedByteLength}.`,
      );
    }
    if (entry.byteOffset !== entry.elementOffset * Float32Array.BYTES_PER_ELEMENT) {
      throw new CheckpointValidationError(`${entry.name} byte and element offsets disagree.`);
    }
    const end = entry.byteOffset + entry.byteLength;
    if (!Number.isSafeInteger(end) || end > index.byteLength) {
      throw new CheckpointValidationError(`${entry.name} range extends beyond weights.bin.`);
    }
    ranges.push({ name: entry.name, start: entry.byteOffset, end });
  }
  for (const specification of expectedSpecs) {
    if (!seen.has(specification.name)) {
      throw new CheckpointValidationError(
        `Missing parameter from weight index: ${specification.name}.`,
      );
    }
  }

  ranges.sort((left, right) => left.start - right.start || left.end - right.end);
  let coveredUntil = 0;
  for (const range of ranges) {
    if (range.start < coveredUntil) {
      throw new CheckpointValidationError(
        `Weight range for ${range.name} overlaps an earlier range.`,
      );
    }
    if (range.start > coveredUntil) {
      throw new CheckpointValidationError(
        `weights.bin contains ${range.start - coveredUntil} unindexed bytes before ${range.name}.`,
      );
    }
    coveredUntil = range.end;
  }
  if (coveredUntil !== index.byteLength) {
    throw new CheckpointValidationError(
      `weights.bin contains ${index.byteLength - coveredUntil} unindexed trailing bytes.`,
    );
  }

  const registry = emptyParameters(config);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (const entry of index.entries) {
    const values = new Float32Array(entry.elementCount);
    for (let valueIndex = 0; valueIndex < values.length; valueIndex += 1) {
      const value = view.getFloat32(
        entry.byteOffset + valueIndex * Float32Array.BYTES_PER_ELEMENT,
        true,
      );
      if (!Number.isFinite(value)) {
        throw new CheckpointValidationError(
          `${entry.name} contains a non-finite value at element ${valueIndex}.`,
        );
      }
      values[valueIndex] = value;
    }
    registry.setValues(entry.name, values);
  }
  return registry;
}
