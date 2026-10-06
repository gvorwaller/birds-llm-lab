import { tensor } from '../math/tensor';
import type { ForwardTrace, TraceEntry, TraceTerm } from './forward-trace';

export function averageAttentionEntry(
  trace: ForwardTrace,
  layer: number,
  headCount: number,
): TraceEntry {
  if (!Number.isInteger(layer) || layer < 0 || !Number.isInteger(headCount) || headCount < 1) {
    throw new Error('Attention average requires a valid layer and positive head count.');
  }
  const source = trace.get(`blocks.${layer}.attn.probabilities`).output;
  const [groups, rows, columns] = source.shape;
  if (groups !== headCount || rows !== columns) {
    throw new Error('Attention probabilities do not match the selected head count.');
  }
  const planeSize = rows * columns;
  const values = new Float32Array(planeSize);
  for (let flat = 0; flat < planeSize; flat += 1) {
    let sum = 0;
    for (let head = 0; head < headCount; head += 1) {
      sum += source.data[head * planeSize + flat];
    }
    values[flat] = sum / headCount;
  }
  const output = tensor(values, [1, rows, columns], `blocks.${layer}.attn.averageProbabilities`);
  return {
    name: `blocks.${layer}.attn.averageProbabilities`,
    output,
    explainCell(indices) {
      if (
        indices.length !== 3 ||
        indices[0] !== 0 ||
        !Number.isInteger(indices[1]) ||
        indices[1] < 0 ||
        indices[1] >= rows ||
        !Number.isInteger(indices[2]) ||
        indices[2] < 0 ||
        indices[2] >= columns
      ) {
        throw new Error('Average attention cell indices are outside the matrix.');
      }
      const flat = indices[1] * columns + indices[2];
      const terms: TraceTerm[] = [];
      for (let head = 0; head < headCount; head += 1) {
        const left = source.data[head * planeSize + flat];
        terms.push({
          label: `head ${head}`,
          left,
          right: 1 / headCount,
          product: left / headCount,
        });
      }
      return {
        formula: 'sum(head probability) / head count',
        output: output.data[flat],
        terms,
        scalars: { headCount },
        note: 'Displayed decimal terms may not sum exactly after Float32 rounding.',
      };
    },
  };
}
