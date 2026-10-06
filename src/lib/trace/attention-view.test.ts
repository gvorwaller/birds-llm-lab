import { describe, expect, it } from 'vitest';
import { initializeParameters } from '../model/parameters';
import { languageModelForward } from '../model/model';
import type { ModelConfig } from '../data/schemas';
import { averageAttentionEntry } from './attention-view';
import { forwardStages } from './forward-stages';

const config: ModelConfig = {
  formatVersion: 1,
  vocabSize: 259,
  contextLength: 4,
  dModel: 4,
  nLayers: 2,
  nHeads: 2,
  dHead: 2,
  dMlp: 6,
  tiedEmbeddings: true,
  useBias: true,
  layerNormEpsilon: 1e-5,
  gelu: 'tanh-approximation',
};

describe('complete inspection views', () => {
  it('includes every block operation in a reversible sequence', () => {
    const stages = forwardStages(2);
    const names = stages.map((stage) => stage.name);
    expect(names[0]).toBe('embedding.token');
    expect(names.at(-1)).toBe('logits');
    expect(names).toContain('blocks.1.mlp.gelu');
    expect(names).toContain('blocks.1.attn.mask');
    expect(new Set(names).size).toBe(names.length);
  });

  it('averages probabilities, preserving causal zeros and exact selected operands', () => {
    const registry = initializeParameters(config, 123);
    const trace = languageModelForward([256, 97, 98], [1, 3], registry, config, undefined, {
      trace: true,
    }).trace;
    if (!trace) throw new Error('Expected trace.');
    for (const stage of forwardStages(2)) {
      expect(trace.entries.has(stage.name), stage.name).toBe(true);
    }
    for (let layer = 0; layer < 2; layer += 1) {
      const source = trace.get(`blocks.${layer}.attn.probabilities`).output;
      const average = averageAttentionEntry(trace, layer, 2);
      expect(average.output.shape).toEqual([1, 3, 3]);
      for (let row = 0; row < 3; row += 1) {
        let rowTotal = 0;
        for (let column = 0; column < 3; column += 1) {
          const expected = Math.fround(
            (source.data[row * 3 + column] + source.data[9 + row * 3 + column]) / 2,
          );
          const selected = average.explainCell([0, row, column]);
          expect(selected.output).toBe(expected);
          expect(selected.terms).toHaveLength(2);
          if (column > row) expect(selected.output).toBe(0);
          rowTotal += selected.output;
        }
        expect(rowTotal).toBeCloseTo(1, 6);
      }
    }
  });
});
