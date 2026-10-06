import { describe, expect, it } from 'vitest';
import type { ModelConfig } from '../data/schemas';
import { languageModelForward } from '../model/model';
import { initializeParameters } from '../model/parameters';
import { drawDistribution, transformDistribution } from './distribution';
import { generateReplay } from './replay';

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
const controls = { temperature: 1, topK: 0, topP: 1 };

describe('generation replay', () => {
  it('recomputes each sliding context and preserves absolute positions and exact dropped tokens', () => {
    const registry = initializeParameters(config, 123);
    const replay = generateReplay([256, 97, 98, 99, 100], 42, controls, 6, registry, config);
    expect(replay.generatedIds).toEqual([65, 22, 149, 57, 97, 6]);
    expect(replay.promptIds).toEqual([256, 97, 98, 99, 100]);
    expect(replay.seed).toBe(42);
    expect(replay.steps[0].dropped).toEqual([
      { absolutePosition: 0, modelPosition: null, id: 256 },
    ]);
    expect(replay.steps[0].context).toEqual([
      { absolutePosition: 1, modelPosition: 0, id: 97 },
      { absolutePosition: 2, modelPosition: 1, id: 98 },
      { absolutePosition: 3, modelPosition: 2, id: 99 },
      { absolutePosition: 4, modelPosition: 3, id: 100 },
    ]);
    expect(replay.steps[1].dropped.map((token) => token.id)).toEqual([256, 97]);
    expect(replay.steps[1].context.map((token) => token.id)).toEqual([98, 99, 100, 65]);
    expect(replay.steps[1].nextAbsolutePosition).toBe(6);
    expect(replay.steps[5].inputIds).toEqual([256, 97, 98, 99, 100, 65, 22, 149, 57, 97]);

    for (const step of replay.steps) {
      const contextIds = step.context.map((token) => token.id);
      const forward = languageModelForward(contextIds, [1, contextIds.length], registry, config);
      const logits = forward.logits.data.slice(-config.vocabSize);
      expect(step.distribution).toEqual(transformDistribution(logits, controls));
      const state =
        step.index === 0 ? replay.seed : replay.steps[step.index - 1].draw.nextRandomState;
      expect(step.draw).toEqual(drawDistribution(step.distribution, state));
    }
    expect(generateReplay([256, 97, 98, 99, 100], 42, controls, 6, registry, config)).toEqual(
      replay,
    );
  });

  it('validates prompt and count instead of silently truncating the requested replay', () => {
    const registry = initializeParameters(config, 123);
    expect(() => generateReplay([], 42, controls, 1, registry, config)).toThrow('prompt token');
    expect(() => generateReplay([256], 42, controls, 0, registry, config)).toThrow('1 to 32');
    expect(() => generateReplay([256], 42, controls, 33, registry, config)).toThrow('1 to 32');
  });
});
