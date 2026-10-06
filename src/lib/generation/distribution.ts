import { SeededRandom, type RandomState } from '../math/rng';

export interface DistributionControls {
  readonly temperature: number;
  readonly topK: number;
  readonly topP: number;
}

export interface TokenDistribution {
  readonly controls: DistributionControls;
  readonly logits: readonly number[];
  /** Stable softmax after temperature, before top-k or top-p. */
  readonly beforeFilters: readonly number[];
  /** Unnormalized probabilities after top-k. */
  readonly afterTopK: readonly number[];
  /** Unnormalized probabilities after top-p. */
  readonly afterTopP: readonly number[];
  /** Final normalized probabilities, indexed by token id. */
  readonly probabilities: readonly number[];
  /** Cumulative upper bounds in token-id order. */
  readonly cumulative: readonly number[];
  /** Descending pre-filter probability; ties use smaller token id. */
  readonly rankOrder: readonly number[];
  readonly massAfterTopK: number;
  readonly retainedMass: number;
  readonly removedMass: number;
}

export interface DrawTrace {
  readonly randomNumber: number;
  readonly tokenId: number;
  readonly intervalStart: number;
  readonly intervalEnd: number;
  readonly nextRandomState: RandomState;
}

function validate(logits: Float32Array | readonly number[], controls: DistributionControls): void {
  if (logits.length === 0) throw new Error('Distribution requires at least one logit.');
  for (const logit of logits) {
    if (!Number.isFinite(logit)) throw new Error('Distribution logits must all be finite.');
  }
  if (!(controls.temperature > 0) || !Number.isFinite(controls.temperature)) {
    throw new Error('Temperature must be positive and finite.');
  }
  if (!Number.isInteger(controls.topK) || controls.topK < 0) {
    throw new Error('Top-k must be a non-negative integer.');
  }
  if (!(controls.topP > 0 && controls.topP <= 1) || !Number.isFinite(controls.topP)) {
    throw new Error('Top-p must be in (0, 1].');
  }
}

function mass(probabilities: readonly number[]): number {
  let result = 0;
  for (const probability of probabilities) result += probability;
  return result;
}

export function transformDistribution(
  logitsInput: Float32Array | readonly number[],
  controls: DistributionControls,
): TokenDistribution {
  validate(logitsInput, controls);
  const logits = Array.from(logitsInput);
  // Subtract before dividing: dividing large finite logits by a tiny temperature
  // can overflow even though their softmax is well defined.
  const maximum = Math.max(...logits);
  const exponentials = logits.map((logit) => Math.exp((logit - maximum) / controls.temperature));
  const denominator = mass(exponentials);
  const beforeFilters = exponentials.map((value) => value / denominator);
  const rankOrder = Array.from({ length: logits.length }, (_, id) => id).sort(
    (left, right) => beforeFilters[right] - beforeFilters[left] || left - right,
  );

  const topKCount = controls.topK === 0 ? logits.length : Math.min(controls.topK, logits.length);
  const afterTopK = new Array<number>(logits.length).fill(0);
  for (let rank = 0; rank < topKCount; rank += 1) {
    const id = rankOrder[rank];
    afterTopK[id] = beforeFilters[id];
  }
  const massAfterTopK = mass(afterTopK);

  const afterTopP = new Array<number>(logits.length).fill(0);
  if (controls.topP === 1) {
    for (let id = 0; id < logits.length; id += 1) afterTopP[id] = afterTopK[id];
  } else {
    let running = 0;
    const threshold = controls.topP * massAfterTopK;
    for (let rank = 0; rank < topKCount; rank += 1) {
      const id = rankOrder[rank];
      afterTopP[id] = afterTopK[id];
      running += afterTopK[id];
      if (running >= threshold) break;
    }
  }
  const retainedMass = mass(afterTopP);
  // At least the highest-ranked token always survives, including extreme logits.
  if (!(retainedMass > 0)) throw new Error('Distribution filters removed every token.');
  const probabilities = afterTopP.map((value) => value / retainedMass);
  const cumulative = new Array<number>(logits.length);
  let running = 0;
  let lastRetained = 0;
  for (let id = 0; id < logits.length; id += 1) {
    running += probabilities[id];
    cumulative[id] = running;
    if (probabilities[id] > 0) lastRetained = id;
  }
  // Close the final interval despite unavoidable binary floating-point summation error.
  for (let id = lastRetained; id < logits.length; id += 1) cumulative[id] = 1;
  return {
    controls: { ...controls },
    logits,
    beforeFilters,
    afterTopK,
    afterTopP,
    probabilities,
    cumulative,
    rankOrder,
    massAfterTopK,
    retainedMass,
    removedMass: Math.max(0, 1 - retainedMass),
  };
}

/** Pure seeded draw: the caller supplies a seed/state and receives the next state. */
export function drawDistribution(
  distribution: TokenDistribution,
  seedOrState: number | RandomState,
): DrawTrace {
  const random = new SeededRandom(seedOrState);
  const randomNumber = random.uniform();
  let tokenId = distribution.probabilities.length - 1;
  for (let id = 0; id < distribution.cumulative.length; id += 1) {
    if (distribution.probabilities[id] > 0 && randomNumber < distribution.cumulative[id]) {
      tokenId = id;
      break;
    }
  }
  return {
    randomNumber,
    tokenId,
    intervalStart: tokenId === 0 ? 0 : distribution.cumulative[tokenId - 1],
    intervalEnd: distribution.cumulative[tokenId],
    nextRandomState: random.snapshot(),
  };
}
