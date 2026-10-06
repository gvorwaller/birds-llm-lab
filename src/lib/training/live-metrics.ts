import type { ModelConfig } from '../data/schemas';
import { sampleCategorical } from '../generation/sampling';
import { SeededRandom } from '../math/rng';
import { languageModelForward } from '../model/model';
import type { ParameterRegistry } from '../model/parameters';
import { EOS_ID } from '../tokenizer/constants';
import { deterministicLanguageModelBatches } from './batches';
import type { SelectedWeightSnapshot } from './worker-protocol';

export function validationLoss(
  sequences: readonly (readonly number[])[],
  registry: ParameterRegistry,
  model: ModelConfig,
  batchSize: number,
  seed: number,
): number {
  const batches = deterministicLanguageModelBatches(
    sequences,
    batchSize,
    model.contextLength,
    seed ^ 0x5a17_1d5e,
    0,
  );
  let weightedLoss = 0;
  let predictions = 0;
  for (const batch of batches) {
    const result = languageModelForward(
      batch.inputIds,
      [batch.batchSize, batch.sequenceLength],
      registry,
      model,
      batch.targetIds,
    );
    if (result.loss === null) throw new Error('Validation loss was not computed.');
    weightedLoss += result.loss * batch.predictionCount;
    predictions += batch.predictionCount;
  }
  if (predictions === 0) throw new Error('Validation sequences contain no predictions.');
  return weightedLoss / predictions;
}

export function trainingSample(
  promptIds: readonly number[],
  newTokens: number,
  registry: ParameterRegistry,
  model: ModelConfig,
  seed: number,
  step: number,
): number[] {
  const random = new SeededRandom((seed ^ Math.imul(step, 0x9e37_79b1) ^ 0xa511_e9b3) >>> 0);
  const ids = [...promptIds];
  for (let generated = 0; generated < newTokens; generated += 1) {
    const context = ids.slice(-model.contextLength);
    const result = languageModelForward(context, [1, context.length], registry, model);
    const offset = (context.length - 1) * model.vocabSize;
    const selected = sampleCategorical(
      result.logits.data.subarray(offset, offset + model.vocabSize),
      random,
      0.8,
    );
    ids.push(selected);
    if (selected === EOS_ID) break;
  }
  return ids;
}

export function selectedWeightSnapshot(
  registry: ParameterRegistry,
  name: string,
  binCount = 12,
): SelectedWeightSnapshot {
  const parameter = registry.get(name);
  if (parameter.shape.length !== 2) throw new Error('Selected weight must be a matrix.');
  const values = Array.from(parameter.value.data);
  let minimum = Infinity;
  let maximum = -Infinity;
  for (const value of values) {
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
  }
  const counts = Array<number>(binCount).fill(0);
  for (const value of values) {
    const index =
      maximum === minimum
        ? 0
        : Math.min(binCount - 1, Math.floor(((value - minimum) / (maximum - minimum)) * binCount));
    counts[index] += 1;
  }
  return {
    name,
    shape: [parameter.shape[0], parameter.shape[1]],
    values,
    histogramMinimum: minimum,
    histogramMaximum: maximum,
    histogramCounts: counts,
  };
}
