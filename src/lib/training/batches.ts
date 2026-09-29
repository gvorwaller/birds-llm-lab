import { SeededRandom } from '../math/rng';
import { PAD_ID } from '../tokenizer/constants';

export interface LanguageModelBatch {
  readonly inputIds: number[];
  readonly targetIds: number[];
  readonly batchSize: number;
  readonly sequenceLength: number;
  readonly predictionCount: number;
}

interface LanguageModelExample {
  readonly inputIds: number[];
  readonly targetIds: number[];
  readonly predictionCount: number;
}

function examplesFromSequences(
  sequences: readonly (readonly number[])[],
  contextLength: number,
): LanguageModelExample[] {
  if (!Number.isInteger(contextLength) || contextLength <= 0) {
    throw new Error('Batch context length must be a positive integer.');
  }
  const examples: LanguageModelExample[] = [];
  for (const [sequenceIndex, sequence] of sequences.entries()) {
    for (const [tokenIndex, tokenId] of sequence.entries()) {
      if (!Number.isInteger(tokenId) || tokenId < 0) {
        throw new Error(
          `Sequence ${sequenceIndex} has invalid token ID ${tokenId} at ${tokenIndex}.`,
        );
      }
    }
    for (let start = 0; start < sequence.length - 1; start += contextLength) {
      const predictionCount = Math.min(contextLength, sequence.length - 1 - start);
      const inputIds = Array<number>(contextLength).fill(PAD_ID);
      const targetIds = Array<number>(contextLength).fill(PAD_ID);
      for (let offset = 0; offset < predictionCount; offset += 1) {
        inputIds[offset] = sequence[start + offset];
        targetIds[offset] = sequence[start + offset + 1];
      }
      examples.push({ inputIds, targetIds, predictionCount });
    }
  }
  return examples;
}

function shuffledIndices(length: number, seed: number, epoch: number): number[] {
  if (!Number.isInteger(epoch) || epoch < 0) throw new Error('Batch epoch must be non-negative.');
  const random = new SeededRandom((seed ^ Math.imul(epoch + 1, 0x9e37_79b1)) >>> 0);
  const indices = Array.from({ length }, (_, index) => index);
  for (let index = indices.length - 1; index > 0; index -= 1) {
    const selected = Math.floor(random.uniform() * (index + 1));
    [indices[index], indices[selected]] = [indices[selected], indices[index]];
  }
  return indices;
}

export function deterministicLanguageModelBatches(
  sequences: readonly (readonly number[])[],
  batchSize: number,
  contextLength: number,
  seed: number,
  epoch: number,
): LanguageModelBatch[] {
  if (!Number.isInteger(batchSize) || batchSize <= 0) {
    throw new Error('Batch size must be a positive integer.');
  }
  const examples = examplesFromSequences(sequences, contextLength);
  const order = shuffledIndices(examples.length, seed, epoch);
  const batches: LanguageModelBatch[] = [];
  for (let start = 0; start < order.length; start += batchSize) {
    const selected = order.slice(start, start + batchSize).map((index) => examples[index]);
    batches.push({
      inputIds: selected.flatMap((example) => example.inputIds),
      targetIds: selected.flatMap((example) => example.targetIds),
      batchSize: selected.length,
      sequenceLength: contextLength,
      predictionCount: selected.reduce((total, example) => total + example.predictionCount, 0),
    });
  }
  return batches;
}
