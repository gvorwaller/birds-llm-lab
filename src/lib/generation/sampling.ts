import { SeededRandom } from '../math/rng';

export function sampleCategorical(
  logits: Float32Array | readonly number[],
  random: SeededRandom,
  temperature = 1,
): number {
  if (logits.length === 0) throw new Error('Sampling requires at least one logit.');
  if (!(temperature > 0) || !Number.isFinite(temperature)) {
    throw new Error('Sampling temperature must be positive and finite.');
  }
  let maximum = -Infinity;
  for (const logit of logits) {
    if (!Number.isFinite(logit)) throw new Error('Sampling logits must all be finite.');
    maximum = Math.max(maximum, logit / temperature);
  }
  const weights = new Float64Array(logits.length);
  let total = 0;
  for (let index = 0; index < logits.length; index += 1) {
    const weight = Math.exp(logits[index] / temperature - maximum);
    weights[index] = weight;
    total += weight;
  }
  const threshold = random.uniform() * total;
  let cumulative = 0;
  for (let index = 0; index < weights.length; index += 1) {
    cumulative += weights[index];
    if (threshold < cumulative) return index;
  }
  return weights.length - 1;
}
