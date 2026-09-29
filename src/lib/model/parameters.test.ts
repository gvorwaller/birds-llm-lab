import { describe, expect, it } from 'vitest';
import { encodeCheckpointWeights } from './checkpoint';
import { DEFAULT_MODEL_CONFIG } from './config';
import { emptyParameters, initializeParameters, parameterSpecs } from './parameters';

describe('parameter registry', () => {
  it('defines the exact default names, shapes, and 169,728-element total', () => {
    const registry = emptyParameters(DEFAULT_MODEL_CONFIG);
    expect(registry.size).toBe(36);
    expect(registry.elementCount).toBe(169_728);
    expect(registry.entries().map(({ name, shape }) => ({ name, shape }))).toEqual([
      { name: 'token_embedding.weight', shape: [1_024, 64] },
      { name: 'position_embedding.weight', shape: [64, 64] },
      { name: 'blocks.0.ln1.scale', shape: [64] },
      { name: 'blocks.0.ln1.offset', shape: [64] },
      { name: 'blocks.0.attn.q.weight', shape: [64, 64] },
      { name: 'blocks.0.attn.q.bias', shape: [64] },
      { name: 'blocks.0.attn.k.weight', shape: [64, 64] },
      { name: 'blocks.0.attn.k.bias', shape: [64] },
      { name: 'blocks.0.attn.v.weight', shape: [64, 64] },
      { name: 'blocks.0.attn.v.bias', shape: [64] },
      { name: 'blocks.0.attn.out.weight', shape: [64, 64] },
      { name: 'blocks.0.attn.out.bias', shape: [64] },
      { name: 'blocks.0.ln2.scale', shape: [64] },
      { name: 'blocks.0.ln2.offset', shape: [64] },
      { name: 'blocks.0.mlp.in.weight', shape: [64, 256] },
      { name: 'blocks.0.mlp.in.bias', shape: [256] },
      { name: 'blocks.0.mlp.out.weight', shape: [256, 64] },
      { name: 'blocks.0.mlp.out.bias', shape: [64] },
      { name: 'blocks.1.ln1.scale', shape: [64] },
      { name: 'blocks.1.ln1.offset', shape: [64] },
      { name: 'blocks.1.attn.q.weight', shape: [64, 64] },
      { name: 'blocks.1.attn.q.bias', shape: [64] },
      { name: 'blocks.1.attn.k.weight', shape: [64, 64] },
      { name: 'blocks.1.attn.k.bias', shape: [64] },
      { name: 'blocks.1.attn.v.weight', shape: [64, 64] },
      { name: 'blocks.1.attn.v.bias', shape: [64] },
      { name: 'blocks.1.attn.out.weight', shape: [64, 64] },
      { name: 'blocks.1.attn.out.bias', shape: [64] },
      { name: 'blocks.1.ln2.scale', shape: [64] },
      { name: 'blocks.1.ln2.offset', shape: [64] },
      { name: 'blocks.1.mlp.in.weight', shape: [64, 256] },
      { name: 'blocks.1.mlp.in.bias', shape: [256] },
      { name: 'blocks.1.mlp.out.weight', shape: [256, 64] },
      { name: 'blocks.1.mlp.out.bias', shape: [64] },
      { name: 'final_ln.scale', shape: [64] },
      { name: 'final_ln.offset', shape: [64] },
    ]);
  });

  it('initializes deterministically with the documented normal, zero, and one rules', () => {
    const first = initializeParameters(DEFAULT_MODEL_CONFIG, 0x5eed);
    const second = initializeParameters(DEFAULT_MODEL_CONFIG, 0x5eed);
    const different = initializeParameters(DEFAULT_MODEL_CONFIG, 0x5eee);
    expect(encodeCheckpointWeights(DEFAULT_MODEL_CONFIG, first).bytes).toEqual(
      encodeCheckpointWeights(DEFAULT_MODEL_CONFIG, second).bytes,
    );
    expect(first.get('token_embedding.weight').value.data).not.toEqual(
      different.get('token_embedding.weight').value.data,
    );
    expect(Array.from(first.get('blocks.0.attn.q.bias').value.data)).toEqual(Array(64).fill(0));
    expect(Array.from(first.get('blocks.0.ln1.scale').value.data)).toEqual(Array(64).fill(1));
    expect(Array.from(first.get('blocks.0.ln1.offset').value.data)).toEqual(Array(64).fill(0));
  });

  it('marks only matrices and embeddings for weight decay', () => {
    const specs = parameterSpecs(DEFAULT_MODEL_CONFIG);
    expect(specs.filter(({ decay }) => decay).every(({ name }) => name.endsWith('.weight'))).toBe(
      true,
    );
    expect(specs.find(({ name }) => name === 'blocks.0.attn.q.bias')?.decay).toBe(false);
    expect(specs.find(({ name }) => name === 'blocks.0.ln1.scale')?.decay).toBe(false);
  });

  it('supports the documented untied and bias-free variants without hidden parameters', () => {
    const untied = emptyParameters({ ...DEFAULT_MODEL_CONFIG, tiedEmbeddings: false });
    expect(untied.get('unembedding.weight').shape).toEqual([1_024, 64]);
    expect(untied.get('unembedding.bias').shape).toEqual([1_024]);
    const biasFree = emptyParameters({
      ...DEFAULT_MODEL_CONFIG,
      tiedEmbeddings: false,
      useBias: false,
    });
    expect(biasFree.entries().some(({ name }) => name.endsWith('.bias'))).toBe(false);
    expect(() => biasFree.get('unembedding.bias')).toThrow('Unknown parameter');
  });
});
