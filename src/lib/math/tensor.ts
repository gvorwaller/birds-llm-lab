export interface Tensor {
  readonly data: Float32Array;
  readonly shape: readonly number[];
}

export function elementCount(shape: readonly number[]): number {
  if (shape.length === 0) return 1;
  return shape.reduce((count, dimension, axis) => {
    if (!Number.isInteger(dimension) || dimension < 0) {
      throw new Error(
        `Tensor dimension ${axis} must be a non-negative integer; received ${dimension}.`,
      );
    }
    return count * dimension;
  }, 1);
}

export function assertFinite(values: Float32Array, label = 'Tensor'): void {
  for (let index = 0; index < values.length; index += 1) {
    if (!Number.isFinite(values[index])) {
      throw new Error(`${label} contains a non-finite value at flat index ${index}.`);
    }
  }
}

export function tensor(
  values: Float32Array | readonly number[],
  shape: readonly number[],
  label = 'Tensor',
): Tensor {
  const data = values instanceof Float32Array ? values : Float32Array.from(values);
  const normalizedShape = [...shape];
  const expected = elementCount(normalizedShape);
  if (data.length !== expected) {
    throw new Error(
      `${label} shape [${normalizedShape.join(', ')}] requires ${expected} values; received ${data.length}.`,
    );
  }
  assertFinite(data, label);
  return { data, shape: normalizedShape };
}

export function zeros(shape: readonly number[]): Tensor {
  return tensor(new Float32Array(elementCount(shape)), shape);
}

export function sameShape(left: Tensor, right: Tensor): boolean {
  return (
    left.shape.length === right.shape.length &&
    left.shape.every((dimension, axis) => dimension === right.shape[axis])
  );
}

export function assertShape(value: Tensor, expected: readonly number[], label: string): void {
  if (
    value.shape.length !== expected.length ||
    value.shape.some((dimension, axis) => dimension !== expected[axis])
  ) {
    throw new Error(
      `${label} must have shape [${expected.join(', ')}]; received [${value.shape.join(', ')}].`,
    );
  }
  if (value.data.length !== elementCount(value.shape)) {
    throw new Error(`${label} data length does not match its declared shape.`);
  }
  assertFinite(value.data, label);
}

export function assertSameShape(left: Tensor, right: Tensor, operation: string): void {
  if (!sameShape(left, right)) {
    throw new Error(
      `${operation} requires equal shapes; received [${left.shape.join(', ')}] and [${right.shape.join(', ')}].`,
    );
  }
  assertShape(left, left.shape, `${operation} left input`);
  assertShape(right, right.shape, `${operation} right input`);
}

export function cloneTensor(value: Tensor): Tensor {
  return tensor(value.data.slice(), value.shape);
}
