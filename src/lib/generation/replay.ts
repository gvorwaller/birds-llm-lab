import type { ModelConfig } from '../data/schemas';
import { languageModelForward } from '../model/model';
import type { ParameterRegistry } from '../model/parameters';
import {
  drawDistribution,
  transformDistribution,
  type DistributionControls,
  type DrawTrace,
  type TokenDistribution,
} from './distribution';
import type { RandomState } from '../math/rng';

export interface PositionedToken {
  readonly absolutePosition: number;
  readonly modelPosition: number | null;
  readonly id: number;
}

export interface GenerationStep {
  readonly index: number;
  /** Every token before this draw, including all preceding generated tokens. */
  readonly inputIds: readonly number[];
  /** Exact leading tokens omitted from this model call. */
  readonly dropped: readonly PositionedToken[];
  /** Tokens passed to the model; model positions restart at zero after sliding. */
  readonly context: readonly PositionedToken[];
  readonly nextAbsolutePosition: number;
  readonly distribution: TokenDistribution;
  readonly draw: DrawTrace;
}

export interface GenerationReplay {
  readonly promptIds: readonly number[];
  readonly seed: number;
  readonly controls: DistributionControls;
  readonly generatedIds: readonly number[];
  readonly steps: readonly GenerationStep[];
}

/** Recompute the same model for every appended token and store each decision. */
export function generateReplay(
  promptIds: readonly number[],
  seed: number,
  controls: DistributionControls,
  count: number,
  registry: ParameterRegistry,
  config: ModelConfig,
): GenerationReplay {
  if (promptIds.length === 0) throw new Error('Generation requires at least one prompt token.');
  if (!Number.isInteger(count) || count < 1 || count > 32) {
    throw new Error('Generation count must be an integer from 1 to 32.');
  }
  const inputIds = [...promptIds];
  const steps: GenerationStep[] = [];
  const generatedIds: number[] = [];
  let randomState: number | RandomState = seed;
  for (let index = 0; index < count; index += 1) {
    const contextStart = Math.max(0, inputIds.length - config.contextLength);
    const contextIds = inputIds.slice(contextStart);
    const result = languageModelForward(contextIds, [1, contextIds.length], registry, config);
    const offset = (contextIds.length - 1) * config.vocabSize;
    const distribution = transformDistribution(
      result.logits.data.slice(offset, offset + config.vocabSize),
      controls,
    );
    const draw = drawDistribution(distribution, randomState);
    steps.push({
      index,
      inputIds: [...inputIds],
      dropped: inputIds.slice(0, contextStart).map((id, absolutePosition) => ({
        absolutePosition,
        modelPosition: null,
        id,
      })),
      context: contextIds.map((id, modelPosition) => ({
        absolutePosition: contextStart + modelPosition,
        modelPosition,
        id,
      })),
      nextAbsolutePosition: inputIds.length,
      distribution,
      draw,
    });
    generatedIds.push(draw.tokenId);
    inputIds.push(draw.tokenId);
    randomState = draw.nextRandomState;
  }
  return { promptIds: [...promptIds], seed, controls: { ...controls }, generatedIds, steps };
}
