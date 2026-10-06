import { describe, expect, it } from 'vitest';
import type { ModelConfig } from '../data/schemas';
import { languageModelForward } from '../model/model';
import { initializeParameters } from '../model/parameters';
import { batchedMatmulForward } from '../math/kernels';
import { mergeHeads } from '../model/attention';
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

describe('opt-in forward trace', () => {
  it('preserves bit-identical logits and explains every selected operation', () => {
    const registry = initializeParameters(config, 123);
    const ids = [256, 97, 98];
    const plain = languageModelForward(ids, [1, 3], registry, config);
    const inspected = languageModelForward(ids, [1, 3], registry, config, undefined, {
      trace: true,
    });
    expect(plain.trace).toBeUndefined();
    expect(inspected.logits.data).toEqual(plain.logits.data);
    const trace = inspected.trace;
    if (!trace) throw new Error('Expected trace.');
    const selected = [
      ['embedding.token', [0, 0, 0]],
      ['embedding.position', [0, 0, 0]],
      ['embedding.sum', [0, 0, 0]],
      ['blocks.0.ln1', [0, 0, 0]],
      ['blocks.0.attn.q', [0, 0, 0]],
      ['blocks.0.attn.scores', [0, 0, 0]],
      ['blocks.0.attn.probabilities', [0, 0, 0]],
      ['blocks.0.attn.weightedValue', [0, 0, 0]],
      ['blocks.0.attn.projection', [0, 0, 0]],
      ['blocks.0.residual.attention', [0, 0, 0]],
      ['blocks.0.mlp.preactivation', [0, 0, 0]],
      ['blocks.0.mlp.gelu', [0, 0, 0]],
      ['blocks.0.mlp.output', [0, 0, 0]],
      ['blocks.0.residual.mlp', [0, 0, 0]],
      ['final_ln', [0, 0, 0]],
      ['logits', [0, 2, 97]],
      ['probabilities', [0, 2, 97]],
    ] as const;
    for (const [name, indices] of selected) {
      const entry = trace.get(name);
      const explanation = entry.explainCell(indices);
      let flat = 0;
      for (let axis = 0; axis < indices.length; axis += 1) {
        flat = flat * entry.output.shape[axis] + indices[axis];
      }
      expect(explanation.output, name).toBe(entry.output.data[flat]);
      expect(explanation.formula.length, name).toBeGreaterThan(0);
    }
    expect(trace.get('blocks.0.attn.scores').explainCell([0, 0, 0]).terms).toHaveLength(2);
    expect(trace.get('blocks.0.attn.probabilities').explainCell([0, 0, 2]).output).toBe(0);
    expect(trace.get('blocks.1.attn.probabilities').output.shape).toEqual([2, 3, 3]);
    expect(trace.get('embedding.sum').explainCell([0, 0, 0]).output).toBe(-0.005797200836241245);
    expect(trace.get('blocks.0.ln1').explainCell([0, 0, 0]).output).toBe(1.0070322751998901);
    expect(trace.get('blocks.0.attn.scores').explainCell([0, 0, 0]).output).toBe(
      0.0012210430577397346,
    );
    expect(trace.get('blocks.0.attn.probabilities').explainCell([0, 0, 0]).output).toBe(1);
    expect(trace.get('blocks.0.attn.weightedValue').explainCell([0, 0, 0]).output).toBe(
      0.003617535112425685,
    );
    expect(trace.get('logits').explainCell([0, 2, 97]).output).toBe(-0.0666971281170845);
    expect(trace.get('probabilities').explainCell([0, 2, 97]).output).toBe(0.0036140463780611753);
    expect(() => trace.get('logits').explainCell([0, 3, 0])).toThrow('outside axis');
    expect(() => trace.get('missing')).toThrow('Unknown trace entry');
  });

  it('rejects a training batch trace while preserving normal batch execution', () => {
    const registry = initializeParameters(config, 1);
    const ids = [256, 97, 98, 256, 97, 98];
    expect(languageModelForward(ids, [2, 3], registry, config).logits.shape).toEqual([2, 3, 259]);
    expect(() =>
      languageModelForward(ids, [2, 3], registry, config, undefined, { trace: true }),
    ).toThrow('one inspected example');
  });

  it('keeps UI-facing cells aligned with independent forward-cache values across both layers', () => {
    const registry = initializeParameters(config, 456);
    const result = languageModelForward([256, 97, 98], [1, 3], registry, config, undefined, {
      trace: true,
    });
    const trace = result.trace;
    if (!trace) throw new Error('Expected trace.');
    expect(trace.get('embedding.sum').output.data).toEqual(result.cache.blocks[0].input.data);
    for (let layer = 0; layer < config.nLayers; layer += 1) {
      const block = result.cache.blocks[layer];
      const prefix = `blocks.${layer}`;
      expect(trace.get(`${prefix}.ln1`).output.data).toEqual(
        block.attention.queryLinear.input.data,
      );
      expect(trace.get(`${prefix}.attn.probabilities`).output.data).toEqual(
        block.attention.attention.softmax.output.data,
      );
      const rawScores = batchedMatmulForward(
        block.attention.attention.scoreMatmul.left,
        block.attention.attention.scoreMatmul.right,
      ).output;
      expect(trace.get(`${prefix}.attn.rawScores`).output.data).toEqual(rawScores.data);
      const weighted = batchedMatmulForward(
        block.attention.attention.contextMatmul.left,
        block.attention.attention.contextMatmul.right,
      ).output;
      expect(trace.get(`${prefix}.attn.weightedValue`).output.data).toEqual(weighted.data);
      expect(trace.get(`${prefix}.attn.merged`).output.data).toEqual(
        mergeHeads(weighted, 1, config.nHeads, config.dHead).data,
      );
      expect(trace.get(`${prefix}.residual.attention`).output.data).toEqual(
        block.residualAfterAttention.data,
      );
      expect(trace.get(`${prefix}.ln2`).output.data).toEqual(block.mlpInput.input.data);
      expect(trace.get(`${prefix}.mlp.preactivation`).output.data).toEqual(block.geluInput.data);
      expect(trace.get(`${prefix}.mlp.gelu`).output.data).toEqual(block.mlpOutput.input.data);
      for (const stage of forwardStages(config.nLayers).filter((stage) =>
        stage.name.startsWith(prefix),
      )) {
        const entry = trace.get(stage.name);
        const selected = [0, 0, 0];
        expect(entry.explainCell(selected).output, stage.name).toBe(entry.output.data[0]);
      }
    }
    expect(trace.get('final_ln').output.data).toEqual(result.cache.unembedding.input.data);
    expect(trace.get('logits').output.data).toEqual(result.logits.data);
  });
});
