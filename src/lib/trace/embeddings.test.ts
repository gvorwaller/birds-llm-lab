import { describe, expect, it } from 'vitest';
import { tensor } from '../math/tensor';
import { DEFAULT_MODEL_CONFIG } from '../model/config';
import { initializeParameters } from '../model/parameters';
import { fitEmbeddingPca, nearestCosine, projectEmbeddings } from './embeddings';

describe('embedding analysis', () => {
  it('centres sample covariance, sorts axes, and fixes eigenvector signs', () => {
    const weights = tensor([1, 0, -1, 0, 0, 2, 0, -2], [4, 2]);
    const basis = fitEmbeddingPca(weights);
    expect(Array.from(basis.mean)).toEqual([0, 0]);
    expect(basis.eigenvalues[0]).toBeCloseTo(8 / 3, 6);
    expect(basis.eigenvalues[1]).toBeCloseTo(2 / 3, 6);
    expect(Array.from(basis.components[0])).toEqual([0, 1]);
    expect(Array.from(basis.components[1])).toEqual([1, 0]);
    expect(Array.from(projectEmbeddings(weights, basis))).toEqual([0, 1, 0, -1, 2, 0, -2, 0]);
    expect(fitEmbeddingPca(weights)).toEqual(basis);
  });

  it('reconstructs seeded initial weights and gives deterministic cosine neighbours', () => {
    const initial = initializeParameters(DEFAULT_MODEL_CONFIG, 123).get(
      'token_embedding.weight',
    ).value;
    const reconstructed = initializeParameters(DEFAULT_MODEL_CONFIG, 123).get(
      'token_embedding.weight',
    ).value;
    expect(reconstructed.data).toEqual(initial.data);
    const basis = fitEmbeddingPca(initial);
    expect(basis.sweeps).toBeLessThanOrEqual(64);
    const projected = projectEmbeddings(initial, basis);
    expect(projected.length).toBe(DEFAULT_MODEL_CONFIG.vocabSize * 2);
    expect(projectEmbeddings(reconstructed, basis)).toEqual(projected);
    const neighbours = nearestCosine(initial, 97, 8);
    expect(neighbours).toHaveLength(8);
    expect(neighbours.every((item) => item.id !== 97 && item.similarity <= 1)).toBe(true);
    expect(nearestCosine(reconstructed, 97, 8)).toEqual(neighbours);
  });

  it('rejects invalid PCA dimensions and returns no neighbours for a zero vector', () => {
    expect(() => fitEmbeddingPca(tensor([1, 2], [1, 2]))).toThrow('Embedding analysis');
    const weights = tensor([0, 0, 1, 0], [2, 2]);
    expect(nearestCosine(weights, 0)).toEqual([]);
    expect(() => nearestCosine(weights, 2)).toThrow('valid token id');
  });
});
