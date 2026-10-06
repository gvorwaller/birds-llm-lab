import type { Tensor } from '../math/tensor';

export interface PcaBasis {
  readonly mean: Float32Array;
  readonly components: readonly [Float32Array, Float32Array];
  readonly eigenvalues: readonly [number, number];
  readonly sweeps: number;
  readonly tolerance: number;
}

function dimensions(weights: Tensor): readonly [number, number] {
  if (
    weights.shape.length !== 2 ||
    weights.shape[0] < 2 ||
    weights.shape[1] < 2 ||
    weights.data.length !== weights.shape[0] * weights.shape[1]
  ) {
    throw new Error('Embedding analysis requires a [tokens >= 2, channels >= 2] matrix.');
  }
  return [weights.shape[0], weights.shape[1]];
}

function stabilized(column: Float32Array): Float32Array {
  let pivot = 0;
  for (let index = 1; index < column.length; index += 1) {
    if (Math.abs(column[index]) > Math.abs(column[pivot])) pivot = index;
  }
  if (column[pivot] < 0) for (let index = 0; index < column.length; index += 1) column[index] *= -1;
  return column;
}

/** PCA uses population-centred rows, sample covariance (n-1), and symmetric Jacobi rotations. */
export function fitEmbeddingPca(weights: Tensor, tolerance = 1e-6, maxSweeps = 64): PcaBasis {
  const [tokens, width] = dimensions(weights);
  if (
    !(tolerance > 0) ||
    !Number.isFinite(tolerance) ||
    !Number.isInteger(maxSweeps) ||
    maxSweeps < 1
  ) {
    throw new Error('PCA requires a positive finite tolerance and positive sweep limit.');
  }
  const mean = new Float32Array(width);
  for (let channel = 0; channel < width; channel += 1) {
    let sum = 0;
    for (let token = 0; token < tokens; token += 1) sum += weights.data[token * width + channel];
    mean[channel] = sum / tokens;
  }
  const covariance = new Float32Array(width * width);
  for (let left = 0; left < width; left += 1) {
    for (let right = left; right < width; right += 1) {
      let sum = 0;
      for (let token = 0; token < tokens; token += 1) {
        sum +=
          (weights.data[token * width + left] - mean[left]) *
          (weights.data[token * width + right] - mean[right]);
      }
      const value = sum / (tokens - 1);
      covariance[left * width + right] = value;
      covariance[right * width + left] = value;
    }
  }
  const matrix = covariance.slice();
  const vectors = new Float32Array(width * width);
  for (let channel = 0; channel < width; channel += 1) vectors[channel * width + channel] = 1;
  const maxDiagonal = Math.max(
    ...Array.from({ length: width }, (_, channel) => Math.abs(matrix[channel * width + channel])),
  );
  const threshold = tolerance * Math.max(maxDiagonal, 1e-12);
  let sweeps = 0;
  for (; sweeps < maxSweeps; sweeps += 1) {
    let rotated = false;
    for (let left = 0; left < width; left += 1) {
      for (let right = left + 1; right < width; right += 1) {
        const offDiagonal = matrix[left * width + right];
        if (Math.abs(offDiagonal) <= threshold) continue;
        rotated = true;
        const tau =
          (matrix[right * width + right] - matrix[left * width + left]) / (2 * offDiagonal);
        const rotation =
          tau === 0 ? 1 : Math.sign(tau) / (Math.abs(tau) + Math.sqrt(1 + tau * tau));
        const cosine = 1 / Math.sqrt(1 + rotation * rotation);
        const sine = rotation * cosine;
        const oldLeft = matrix[left * width + left];
        const oldRight = matrix[right * width + right];
        matrix[left * width + left] = oldLeft - rotation * offDiagonal;
        matrix[right * width + right] = oldRight + rotation * offDiagonal;
        matrix[left * width + right] = 0;
        matrix[right * width + left] = 0;
        for (let channel = 0; channel < width; channel += 1) {
          if (channel !== left && channel !== right) {
            const leftValue = matrix[channel * width + left];
            const rightValue = matrix[channel * width + right];
            const nextLeft = cosine * leftValue - sine * rightValue;
            const nextRight = sine * leftValue + cosine * rightValue;
            matrix[channel * width + left] = nextLeft;
            matrix[left * width + channel] = nextLeft;
            matrix[channel * width + right] = nextRight;
            matrix[right * width + channel] = nextRight;
          }
          const oldVectorLeft = vectors[channel * width + left];
          const oldVectorRight = vectors[channel * width + right];
          vectors[channel * width + left] = cosine * oldVectorLeft - sine * oldVectorRight;
          vectors[channel * width + right] = sine * oldVectorLeft + cosine * oldVectorRight;
        }
      }
    }
    if (!rotated) break;
  }
  if (sweeps === maxSweeps)
    throw new Error(`PCA did not converge within ${maxSweeps} Jacobi sweeps.`);
  const order = Array.from({ length: width }, (_, index) => index).sort(
    (left, right) => matrix[right * width + right] - matrix[left * width + left] || left - right,
  );
  const component = (column: number): Float32Array =>
    stabilized(Float32Array.from({ length: width }, (_, row) => vectors[row * width + column]));
  return {
    mean,
    components: [component(order[0]), component(order[1])],
    eigenvalues: [matrix[order[0] * width + order[0]], matrix[order[1] * width + order[1]]],
    sweeps: sweeps + 1,
    tolerance,
  };
}

/** Project onto one shared basis so before/after coordinates are directly comparable. */
export function projectEmbeddings(weights: Tensor, basis: PcaBasis): Float32Array {
  const [tokens, width] = dimensions(weights);
  if (
    basis.mean.length !== width ||
    basis.components.some((component) => component.length !== width)
  ) {
    throw new Error('PCA basis width does not match embedding width.');
  }
  const points = new Float32Array(tokens * 2);
  for (let token = 0; token < tokens; token += 1) {
    for (let axis = 0; axis < 2; axis += 1) {
      let sum = 0;
      for (let channel = 0; channel < width; channel += 1) {
        sum +=
          (weights.data[token * width + channel] - basis.mean[channel]) *
          basis.components[axis][channel];
      }
      points[token * 2 + axis] = sum;
    }
  }
  return points;
}

export interface CosineNeighbour {
  readonly id: number;
  readonly similarity: number;
}

export function nearestCosine(weights: Tensor, tokenId: number, count = 8): CosineNeighbour[] {
  const [tokens, width] = dimensions(weights);
  if (
    !Number.isInteger(tokenId) ||
    tokenId < 0 ||
    tokenId >= tokens ||
    !Number.isInteger(count) ||
    count < 1
  ) {
    throw new Error('Cosine search requires a valid token id and positive result count.');
  }
  let targetNormSquared = 0;
  for (let channel = 0; channel < width; channel += 1) {
    const value = weights.data[tokenId * width + channel];
    targetNormSquared += value * value;
  }
  if (targetNormSquared === 0) return [];
  const neighbours: CosineNeighbour[] = [];
  for (let token = 0; token < tokens; token += 1) {
    if (token === tokenId) continue;
    let dot = 0;
    let normSquared = 0;
    for (let channel = 0; channel < width; channel += 1) {
      const candidate = weights.data[token * width + channel];
      dot += weights.data[tokenId * width + channel] * candidate;
      normSquared += candidate * candidate;
    }
    if (normSquared > 0)
      neighbours.push({ id: token, similarity: dot / Math.sqrt(targetNormSquared * normSquared) });
  }
  return neighbours
    .sort((left, right) => right.similarity - left.similarity || left.id - right.id)
    .slice(0, count);
}
