import { describe, expect, it } from 'vitest';
import type { ModelConfig } from '../data/schemas';
import { PAD_ID } from '../tokenizer/constants';
import { languageModelBackward, languageModelForward } from './model';
import { AdamWOptimizer } from './optimizer';
import { initializeParameters } from './parameters';

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

describe('full language model', () => {
  it('stacks every block and produces finite next-token loss and all parameter gradients', () => {
    const registry = initializeParameters(config, 123);
    const forward = languageModelForward(
      [256, 97, 98, PAD_ID, 256, 99, 100, 101],
      [2, 4],
      registry,
      config,
      [97, 98, 257, PAD_ID, 99, 100, 101, 257],
    );
    expect(forward.logits.shape).toEqual([2, 4, 259]);
    expect(forward.loss).not.toBeNull();
    expect(Number.isFinite(forward.loss)).toBe(true);
    const backward = languageModelBackward(forward.cache);
    expect(backward.parameters.size).toBe(registry.size);
    for (const parameter of registry) {
      const gradient = backward.parameters.get(parameter.name);
      expect(gradient, parameter.name).toBeDefined();
      expect(gradient?.shape).toEqual(parameter.shape);
      expect(parameter.gradient.data.every(Number.isFinite)).toBe(true);
      expect(parameter.gradient.data).toEqual(gradient?.data);
    }
  });

  it('accumulates tied lookup and unembedding gradients exactly once', () => {
    const registry = initializeParameters(config, 456);
    const forward = languageModelForward([256, 97, 98], [1, 3], registry, config, [97, 98, 257]);
    const backward = languageModelBackward(forward.cache);
    const contributions = backward.tiedEmbeddingContributions;
    if (!contributions) throw new Error('Expected tied embedding contributions.');
    const combined = backward.parameters.get('token_embedding.weight');
    if (!combined) throw new Error('Expected combined token embedding gradient.');
    let observedBoth = false;
    for (let index = 0; index < combined.data.length; index += 1) {
      const expected = new Float32Array([
        contributions.lookup.data[index] + contributions.unembedding.data[index],
      ])[0];
      expect(combined.data[index]).toBe(expected);
      if (contributions.lookup.data[index] !== 0 && contributions.unembedding.data[index] !== 0) {
        observedBoth = true;
      }
    }
    expect(observedBoth).toBe(true);
  });

  it('supports untied unembedding and refuses backward without targets', () => {
    const untiedConfig = { ...config, tiedEmbeddings: false };
    const registry = initializeParameters(untiedConfig, 789);
    const inference = languageModelForward([256, 97], [1, 2], registry, untiedConfig);
    expect(inference.loss).toBeNull();
    expect(() => languageModelBackward(inference.cache)).toThrow('requires targets');
    const training = languageModelForward([256, 97], [1, 2], registry, untiedConfig, [97, 257]);
    const backward = languageModelBackward(training.cache);
    expect(backward.tiedEmbeddingContributions).toBeNull();
    expect(backward.parameters.has('unembedding.weight')).toBe(true);
    expect(backward.parameters.has('unembedding.bias')).toBe(true);
  });

  it('runs a complete deterministic forward, backward, clipped optimizer update', () => {
    const run = () => {
      const registry = initializeParameters(config, 1_234);
      const before = registry.get('token_embedding.weight').value.data.slice();
      const forward = languageModelForward([256, 97, 98], [1, 3], registry, config, [97, 98, 257]);
      languageModelBackward(forward.cache);
      const optimizer = new AdamWOptimizer({
        beta1: 0.9,
        beta2: 0.95,
        epsilon: 1e-8,
        weightDecay: 0.1,
        gradientClipNorm: 1,
      });
      const update = optimizer.step(registry, 1e-3);
      return {
        loss: forward.loss,
        update,
        changed: registry
          .get('token_embedding.weight')
          .value.data.some((value, index) => value !== before[index]),
        weights: Array.from(registry.get('token_embedding.weight').value.data),
        state: optimizer.snapshot(),
      };
    };
    const first = run();
    expect(first.changed).toBe(true);
    expect(first.update.gradientNorm).toBeGreaterThan(0);
    expect(first).toEqual(run());
  });
});
