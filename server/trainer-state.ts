import type { RandomState } from '../src/lib/math/rng';
import type { AdamWSnapshot } from '../src/lib/model/optimizer';
import type { ParameterRegistry } from '../src/lib/model/parameters';

export interface DataOrderState {
  readonly epoch: number;
  readonly cursor: number;
}

export interface TrainerResumeState {
  readonly optimizer: AdamWSnapshot;
  readonly samplingRandom: RandomState;
  readonly dataOrder: DataOrderState;
}

interface SerializedMoment {
  readonly name: string;
  readonly elementCount: number;
  readonly firstBase64: string;
  readonly secondBase64: string;
}

interface SerializedTrainerState {
  readonly formatVersion: 1;
  readonly optimizer: {
    readonly step: number;
    readonly moments: SerializedMoment[];
  };
  readonly samplingRandom: RandomState;
  readonly dataOrder: DataOrderState;
}

function encodeFloat32(values: Float32Array): string {
  const bytes = Buffer.allocUnsafe(values.length * Float32Array.BYTES_PER_ELEMENT);
  for (let index = 0; index < values.length; index += 1) {
    bytes.writeFloatLE(values[index], index * Float32Array.BYTES_PER_ELEMENT);
  }
  return bytes.toString('base64');
}

function objectAt(value: unknown, path: string): Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new Error(`${path} must be an object.`);
  }
  return value as Record<string, unknown>;
}

function exactKeys(
  value: Record<string, unknown>,
  expected: readonly string[],
  path: string,
): void {
  const expectedSet = new Set(expected);
  for (const key of Object.keys(value)) {
    if (!expectedSet.has(key)) throw new Error(`${path}.${key} is unexpected.`);
  }
  for (const key of expected) {
    if (!(key in value)) throw new Error(`${path}.${key} is required.`);
  }
}

function integerAt(value: unknown, path: string): number {
  if (!Number.isSafeInteger(value) || (value as number) < 0) {
    throw new Error(`${path} must be a non-negative safe integer.`);
  }
  return value as number;
}

function randomStateAt(value: unknown, path: string): RandomState {
  const object = objectAt(value, path);
  exactKeys(object, ['state', 'spareNormal'], path);
  const state = integerAt(object.state, `${path}.state`);
  if (state > 0xffff_ffff) throw new Error(`${path}.state must fit in uint32.`);
  const spareNormal = object.spareNormal;
  if (spareNormal !== null && (typeof spareNormal !== 'number' || !Number.isFinite(spareNormal))) {
    throw new Error(`${path}.spareNormal must be finite or null.`);
  }
  return { state, spareNormal: spareNormal as number | null };
}

function decodeFloat32(value: unknown, elementCount: number, path: string): Float32Array {
  if (typeof value !== 'string') throw new Error(`${path} must be base64 text.`);
  const bytes = Buffer.from(value, 'base64');
  if (bytes.toString('base64') !== value) throw new Error(`${path} is not canonical base64.`);
  const expectedBytes = elementCount * Float32Array.BYTES_PER_ELEMENT;
  if (bytes.byteLength !== expectedBytes) {
    throw new Error(`${path} has ${bytes.byteLength} bytes; expected ${expectedBytes}.`);
  }
  const values = new Float32Array(elementCount);
  for (let index = 0; index < elementCount; index += 1) {
    const item = bytes.readFloatLE(index * Float32Array.BYTES_PER_ELEMENT);
    if (!Number.isFinite(item)) throw new Error(`${path} contains a non-finite value.`);
    values[index] = item;
  }
  return values;
}

export function serializeTrainerState(state: TrainerResumeState): SerializedTrainerState {
  return {
    formatVersion: 1,
    optimizer: {
      step: state.optimizer.step,
      moments: state.optimizer.moments.map((moment) => ({
        name: moment.name,
        elementCount: moment.first.length,
        firstBase64: encodeFloat32(moment.first),
        secondBase64: encodeFloat32(moment.second),
      })),
    },
    samplingRandom: state.samplingRandom,
    dataOrder: state.dataOrder,
  };
}

export function parseTrainerState(value: unknown, registry: ParameterRegistry): TrainerResumeState {
  const root = objectAt(value, 'trainer-state.json');
  exactKeys(
    root,
    ['formatVersion', 'optimizer', 'samplingRandom', 'dataOrder'],
    'trainer-state.json',
  );
  if (root.formatVersion !== 1) throw new Error('trainer-state.json.formatVersion must be 1.');

  const optimizer = objectAt(root.optimizer, 'trainer-state.json.optimizer');
  exactKeys(optimizer, ['step', 'moments'], 'trainer-state.json.optimizer');
  const step = integerAt(optimizer.step, 'trainer-state.json.optimizer.step');
  if (!Array.isArray(optimizer.moments)) {
    throw new Error('trainer-state.json.optimizer.moments must be an array.');
  }
  const expected = new Map(registry.entries().map((parameter) => [parameter.name, parameter]));
  const seen = new Set<string>();
  const moments = optimizer.moments.map((value, index) => {
    const path = `trainer-state.json.optimizer.moments[${index}]`;
    const object = objectAt(value, path);
    exactKeys(object, ['name', 'elementCount', 'firstBase64', 'secondBase64'], path);
    if (typeof object.name !== 'string') throw new Error(`${path}.name must be text.`);
    const parameter = expected.get(object.name);
    if (!parameter) throw new Error(`${path}.name is not a checkpoint parameter.`);
    if (seen.has(object.name)) throw new Error(`${path}.name is duplicated.`);
    seen.add(object.name);
    const elementCount = integerAt(object.elementCount, `${path}.elementCount`);
    if (elementCount !== parameter.value.data.length) {
      throw new Error(`${path}.elementCount does not match ${object.name}.`);
    }
    return {
      name: object.name,
      first: decodeFloat32(object.firstBase64, elementCount, `${path}.firstBase64`),
      second: decodeFloat32(object.secondBase64, elementCount, `${path}.secondBase64`),
    };
  });
  if (step > 0 && seen.size !== expected.size) {
    throw new Error('trainer-state.json is missing optimizer moments.');
  }
  if (step === 0 && moments.length > 0) {
    throw new Error('trainer-state.json has moments at optimizer step zero.');
  }

  const dataOrder = objectAt(root.dataOrder, 'trainer-state.json.dataOrder');
  exactKeys(dataOrder, ['epoch', 'cursor'], 'trainer-state.json.dataOrder');
  return {
    optimizer: { step, moments },
    samplingRandom: randomStateAt(root.samplingRandom, 'trainer-state.json.samplingRandom'),
    dataOrder: {
      epoch: integerAt(dataOrder.epoch, 'trainer-state.json.dataOrder.epoch'),
      cursor: integerAt(dataOrder.cursor, 'trainer-state.json.dataOrder.cursor'),
    },
  };
}
