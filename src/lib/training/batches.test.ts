import { describe, expect, it } from 'vitest';
import { deterministicLanguageModelBatches } from './batches';

describe('deterministic next-token batches', () => {
  it('covers every transition exactly once without dropping a partial batch', () => {
    const batches = deterministicLanguageModelBatches(
      [
        [1, 2, 3, 4, 5],
        [6, 7],
      ],
      2,
      2,
      42,
      0,
    );
    expect(batches).toHaveLength(2);
    expect(batches.map(({ batchSize }) => batchSize)).toEqual([2, 1]);
    expect(batches.reduce((sum, batch) => sum + batch.predictionCount, 0)).toBe(5);
  });

  it('repeats exactly for a seed and changes deterministic order by epoch', () => {
    const sequences = Array.from({ length: 10 }, (_, index) => [
      index * 3 + 1,
      index * 3 + 2,
      index * 3 + 3,
    ]);
    const first = deterministicLanguageModelBatches(sequences, 2, 2, 123, 0);
    expect(deterministicLanguageModelBatches(sequences, 2, 2, 123, 0)).toEqual(first);
    expect(deterministicLanguageModelBatches(sequences, 2, 2, 123, 1)).not.toEqual(first);
  });

  it('validates dimensions and token IDs', () => {
    expect(() => deterministicLanguageModelBatches([[1, -1]], 1, 2, 1, 0)).toThrow(
      'invalid token ID',
    );
    expect(() => deterministicLanguageModelBatches([[1, 2]], 0, 2, 1, 0)).toThrow(
      'positive integer',
    );
  });
});
