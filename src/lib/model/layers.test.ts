import { describe, expect, it } from 'vitest';
import { tensor } from '../math/tensor';
import { unembeddingBackward, unembeddingForward } from './layers';

describe('unembedding', () => {
  it('uses the vocabulary-by-model matrix as a transpose and returns hand-computed gradients', () => {
    const result = unembeddingForward(
      tensor([1, 2], [1, 1, 2]),
      tensor([3, 4, 5, 6], [2, 2]),
      tensor([0.5, -0.5], [2]),
    );
    expect(Array.from(result.output.data)).toEqual([11.5, 16.5]);
    const gradients = unembeddingBackward(result.cache, tensor([2, -1], [1, 1, 2]));
    expect(Array.from(gradients.input.data)).toEqual([1, 2]);
    expect(Array.from(gradients.weight.data)).toEqual([2, 4, -1, -2]);
    expect(Array.from(gradients.bias?.data ?? [])).toEqual([2, -1]);
  });
});
