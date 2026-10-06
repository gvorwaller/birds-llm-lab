import { addForward, geluForward, layerNormForward, softmaxForward } from '../math/kernels';
import { tensor, type Tensor } from '../math/tensor';
import { mergeHeads } from '../model/attention';
import { linearForward, type LinearCache } from '../model/layers';
import type { LanguageModelCache } from '../model/model';
import type { TransformerBlockCache } from '../model/block';

export interface TraceTerm {
  readonly label: string;
  readonly left: number;
  readonly right: number;
  readonly product: number;
}

export interface CellExplanation {
  readonly formula: string;
  readonly output: number;
  readonly terms: readonly TraceTerm[];
  readonly scalars: Readonly<Record<string, number>>;
  readonly note?: string;
}

export interface TraceEntry {
  readonly name: string;
  readonly output: Tensor;
  explainCell(indices: readonly number[]): CellExplanation;
}

export interface ForwardTrace {
  readonly tokenIds: readonly number[];
  readonly entries: ReadonlyMap<string, TraceEntry>;
  get(name: string): TraceEntry;
}

function flatIndex(shape: readonly number[], indices: readonly number[]): number {
  if (indices.length !== shape.length) {
    throw new Error(`Expected ${shape.length} trace indices; received ${indices.length}.`);
  }
  let result = 0;
  for (let axis = 0; axis < shape.length; axis += 1) {
    const index = indices[axis];
    if (!Number.isInteger(index) || index < 0 || index >= shape[axis]) {
      throw new Error(`Trace index ${index} is outside axis ${axis} of size ${shape[axis]}.`);
    }
    result = result * shape[axis] + index;
  }
  return result;
}

const roundingNote = 'Displayed decimal terms may not sum exactly after Float32 rounding.';

class TraceBuilder {
  private readonly values = new Map<string, TraceEntry>();

  add(
    name: string,
    output: Tensor,
    explain: (indices: readonly number[], offset: number) => CellExplanation,
  ): Tensor {
    if (this.values.has(name)) throw new Error(`Duplicate trace entry ${name}.`);
    this.values.set(name, {
      name,
      output,
      explainCell: (indices) => explain(indices, flatIndex(output.shape, indices)),
    });
    return output;
  }

  lookup(name: string, output: Tensor, ids: readonly number[], weights: Tensor): Tensor {
    const width = weights.shape[1];
    return this.add(name, output, (_indices, offset) => {
      const row = Math.floor(offset / width);
      const id = ids[row];
      return {
        formula: 'embedding[id, channel]',
        output: output.data[offset],
        terms: [],
        scalars: {
          id,
          channel: offset % width,
          storedWeight: weights.data[id * width + (offset % width)],
        },
      };
    });
  }

  addValues(
    name: string,
    left: Tensor,
    right: Tensor,
    output = addForward(left, right).output,
  ): Tensor {
    return this.add(name, output, (_indices, offset) => ({
      formula: 'left + right',
      output: output.data[offset],
      terms: [
        { label: 'left', left: left.data[offset], right: 1, product: left.data[offset] },
        { label: 'right', left: right.data[offset], right: 1, product: right.data[offset] },
      ],
      scalars: {},
      note: roundingNote,
    }));
  }

  norm(name: string, input: Tensor, scale: Tensor, offset: Tensor, epsilon: number): Tensor {
    const result = layerNormForward(input, scale, offset, epsilon);
    const width = input.shape.at(-1) as number;
    return this.add(name, result.output, (_indices, flat) => {
      const row = Math.floor(flat / width);
      const channel = flat % width;
      let mean = 0;
      for (let i = 0; i < width; i += 1) mean += input.data[row * width + i];
      mean /= width;
      let variance = 0;
      const terms: TraceTerm[] = [];
      for (let i = 0; i < width; i += 1) {
        const centered = input.data[row * width + i] - mean;
        terms.push({
          label: `centered[${i}]²`,
          left: centered,
          right: centered,
          product: centered * centered,
        });
        variance += centered * centered;
      }
      variance /= width;
      return {
        formula: '(input - mean) / sqrt(variance + epsilon) * scale + offset',
        output: result.output.data[flat],
        terms,
        scalars: {
          input: input.data[flat],
          mean,
          variance,
          epsilon,
          normalized: result.cache.normalized.data[flat],
          scale: scale.data[channel],
          offset: offset.data[channel],
        },
        note: roundingNote,
      };
    });
  }

  linear(name: string, cache: LinearCache): Tensor {
    const output = linearForward(cache.input, cache.weight, cache.bias).output;
    return this.add(name, output, (_indices, flat) => {
      const row = Math.floor(flat / cache.outputWidth);
      const column = flat % cache.outputWidth;
      const terms: TraceTerm[] = [];
      for (let inner = 0; inner < cache.inputWidth; inner += 1) {
        const left = cache.input.data[row * cache.inputWidth + inner];
        const right = cache.weight.data[inner * cache.outputWidth + column];
        terms.push({ label: `channel ${inner}`, left, right, product: left * right });
      }
      return {
        formula: 'bias[column] + sum(input[channel] * weight[channel, column])',
        output: output.data[flat],
        terms,
        scalars: { bias: cache.bias?.data[column] ?? 0 },
        note: roundingNote,
      };
    });
  }

  gelu(name: string, input: Tensor): Tensor {
    const output = geluForward(input).output;
    return this.add(name, output, (_indices, flat) => ({
      formula: '0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x³)))',
      output: output.data[flat],
      terms: [],
      scalars: { x: input.data[flat] },
      note: roundingNote,
    }));
  }

  scores(name: string, block: TransformerBlockCache): Tensor {
    const attention = block.attention.attention;
    const query = attention.query;
    const key = attention.key;
    const [groups, sequenceLength, width] = query.shape;
    const rawValues = new Float32Array(groups * sequenceLength * sequenceLength);
    const values = new Float32Array(rawValues.length);
    for (let group = 0; group < groups; group += 1) {
      for (let row = 0; row < sequenceLength; row += 1) {
        for (let column = 0; column < sequenceLength; column += 1) {
          let dot = 0;
          for (let channel = 0; channel < width; channel += 1) {
            dot +=
              query.data[(group * sequenceLength + row) * width + channel] *
              key.data[(group * sequenceLength + column) * width + channel];
          }
          const flat = (group * sequenceLength + row) * sequenceLength + column;
          rawValues[flat] = dot;
          values[flat] = dot * attention.scale;
        }
      }
    }
    const shape = [groups, sequenceLength, sequenceLength];
    const raw = tensor(rawValues, shape, `${name}.raw`);
    const mask = tensor(Float32Array.from(attention.softmax.allowed), shape, `${name}.mask`);
    this.add(name.replace(/\.scores$/, '.rawScores'), raw, (indices, flat) => ({
      formula: 'sum(query[channel] * key[channel])',
      output: raw.data[flat],
      terms: Array.from({ length: width }, (_, channel) => {
        const [group, row, column] = indices;
        const left = query.data[(group * sequenceLength + row) * width + channel];
        const right = key.data[(group * sequenceLength + column) * width + channel];
        return { label: `channel ${channel}`, left, right, product: left * right };
      }),
      scalars: {},
      note: roundingNote,
    }));
    this.add(name.replace(/\.scores$/, '.mask'), mask, (_indices, flat) => ({
      formula: 'query and key are unpadded, and key position <= query position',
      output: mask.data[flat],
      terms: [],
      scalars: {},
    }));
    const output = tensor(values, shape, name);
    return this.add(name, output, (indices, flat) => {
      const [group, row, column] = indices;
      const terms: TraceTerm[] = [];
      for (let channel = 0; channel < width; channel += 1) {
        const left = query.data[(group * sequenceLength + row) * width + channel];
        const right = key.data[(group * sequenceLength + column) * width + channel];
        terms.push({ label: `channel ${channel}`, left, right, product: left * right });
      }
      return {
        formula: 'sum(query[channel] * key[channel]) / sqrt(headWidth)',
        output: output.data[flat],
        terms,
        scalars: { scale: attention.scale, allowed: attention.softmax.allowed[flat] },
        note: roundingNote,
      };
    });
  }

  probabilities(name: string, scores: Tensor, output: Tensor, allowed: Uint8Array): Tensor {
    const width = scores.shape.at(-1) as number;
    return this.add(name, output, (_indices, flat) => {
      const rowStart = Math.floor(flat / width) * width;
      let maximum = -Infinity;
      for (let i = 0; i < width; i += 1) {
        if (allowed[rowStart + i]) maximum = Math.max(maximum, scores.data[rowStart + i]);
      }
      const terms: TraceTerm[] = [];
      let denominator = 0;
      if (maximum !== -Infinity) {
        for (let i = 0; i < width; i += 1) {
          const exp = allowed[rowStart + i] ? Math.exp(scores.data[rowStart + i] - maximum) : 0;
          terms.push({
            label: `key ${i}`,
            left: scores.data[rowStart + i],
            right: allowed[rowStart + i],
            product: exp,
          });
          denominator += exp;
        }
      }
      return {
        formula: 'allowed ? exp(score - rowMax) / sum(exp(allowed scores - rowMax)) : 0',
        output: output.data[flat],
        terms,
        scalars: { rowMax: maximum, denominator, allowed: allowed[flat] },
        note: roundingNote,
      };
    });
  }

  weighted(name: string, block: TransformerBlockCache): Tensor {
    const attention = block.attention.attention;
    const probabilities = attention.softmax.output;
    const values = attention.value;
    const [groups, sequenceLength, width] = values.shape;
    const outputData = new Float32Array(values.data.length);
    for (let group = 0; group < groups; group += 1) {
      for (let row = 0; row < sequenceLength; row += 1) {
        for (let channel = 0; channel < width; channel += 1) {
          let sum = 0;
          for (let key = 0; key < sequenceLength; key += 1) {
            sum +=
              probabilities.data[(group * sequenceLength + row) * sequenceLength + key] *
              values.data[(group * sequenceLength + key) * width + channel];
          }
          outputData[(group * sequenceLength + row) * width + channel] = sum;
        }
      }
    }
    const output = tensor(outputData, values.shape, name);
    return this.add(name, output, (indices, flat) => {
      const [group, row, channel] = indices;
      const terms: TraceTerm[] = [];
      for (let key = 0; key < sequenceLength; key += 1) {
        const left = probabilities.data[(group * sequenceLength + row) * sequenceLength + key];
        const right = values.data[(group * sequenceLength + key) * width + channel];
        terms.push({ label: `key ${key}`, left, right, product: left * right });
      }
      return {
        formula: 'sum(probability[key] * value[key, channel])',
        output: output.data[flat],
        terms,
        scalars: {},
        note: roundingNote,
      };
    });
  }

  merged(name: string, weighted: Tensor, headCount: number, headWidth: number): Tensor {
    const output = mergeHeads(weighted, 1, headCount, headWidth);
    const sequenceLength = output.shape[1];
    return this.add(name, output, (indices, flat) => {
      const [, token, channel] = indices;
      const head = Math.floor(channel / headWidth);
      const headChannel = channel % headWidth;
      const source = weighted.data[(head * sequenceLength + token) * headWidth + headChannel];
      return {
        formula: 'merged[token, channel] = head[head, token, headChannel]',
        output: output.data[flat],
        terms: [],
        scalars: { head, headChannel, source },
      };
    });
  }

  logits(name: string, cache: LanguageModelCache, output: Tensor): Tensor {
    const { input, weight, bias, modelWidth, vocabularySize } = cache.unembedding;
    return this.add(name, output, (_indices, flat) => {
      const row = Math.floor(flat / vocabularySize);
      const token = flat % vocabularySize;
      const terms: TraceTerm[] = [];
      for (let channel = 0; channel < modelWidth; channel += 1) {
        const left = input.data[row * modelWidth + channel];
        const right = weight.data[token * modelWidth + channel];
        terms.push({ label: `channel ${channel}`, left, right, product: left * right });
      }
      return {
        formula: 'bias[token] + sum(hidden[channel] * tokenWeight[channel])',
        output: output.data[flat],
        terms,
        scalars: { bias: bias?.data[token] ?? 0 },
        note: roundingNote,
      };
    });
  }

  finish(tokenIds: readonly number[]): ForwardTrace {
    const entries = this.values;
    return {
      tokenIds: [...tokenIds],
      entries,
      get(name) {
        const entry = entries.get(name);
        if (!entry) throw new Error(`Unknown trace entry ${name}.`);
        return entry;
      },
    };
  }
}

export function buildForwardTrace(
  tokenIds: readonly number[],
  cache: LanguageModelCache,
  logits: Tensor,
): ForwardTrace {
  if (cache.batchSize !== 1) throw new Error('ForwardTrace supports one inspected example only.');
  const builder = new TraceBuilder();
  const { registry, config, sequenceLength } = cache;
  const tokenWeights = registry.get('token_embedding.weight').value;
  const positionWeights = registry.get('position_embedding.weight').value;
  const shape: readonly [number, number, number] = [1, sequenceLength, config.dModel];
  const tokenData = new Float32Array(sequenceLength * config.dModel);
  const positionData = new Float32Array(tokenData.length);
  const positions = Array.from({ length: sequenceLength }, (_, index) => index);
  for (let row = 0; row < sequenceLength; row += 1) {
    tokenData.set(
      tokenWeights.data.subarray(
        tokenIds[row] * config.dModel,
        (tokenIds[row] + 1) * config.dModel,
      ),
      row * config.dModel,
    );
    positionData.set(
      positionWeights.data.subarray(row * config.dModel, (row + 1) * config.dModel),
      row * config.dModel,
    );
  }
  const tokenEmbedding = builder.lookup(
    'embedding.token',
    tensor(tokenData, shape),
    tokenIds,
    tokenWeights,
  );
  const positionEmbedding = builder.lookup(
    'embedding.position',
    tensor(positionData, shape),
    positions,
    positionWeights,
  );
  let hidden = builder.addValues('embedding.sum', tokenEmbedding, positionEmbedding);

  for (let layer = 0; layer < cache.blocks.length; layer += 1) {
    const block = cache.blocks[layer];
    const prefix = `blocks.${layer}`;
    const ln1 = builder.norm(
      `${prefix}.ln1`,
      hidden,
      registry.get(`${prefix}.ln1.scale`).value,
      registry.get(`${prefix}.ln1.offset`).value,
      config.layerNormEpsilon,
    );
    for (const projection of ['q', 'k', 'v'] as const) {
      builder.linear(
        `${prefix}.attn.${projection}`,
        block.attention[
          `${projection === 'q' ? 'query' : projection === 'k' ? 'key' : 'value'}Linear`
        ],
      );
    }
    const scores = builder.scores(`${prefix}.attn.scores`, block);
    const attn = block.attention.attention;
    builder.probabilities(
      `${prefix}.attn.probabilities`,
      scores,
      attn.softmax.output,
      attn.softmax.allowed,
    );
    const weighted = builder.weighted(`${prefix}.attn.weightedValue`, block);
    const merged = builder.merged(`${prefix}.attn.merged`, weighted, config.nHeads, config.dHead);
    const projected = builder.linear(`${prefix}.attn.projection`, block.attention.outputLinear);
    // The projection cache retains the exact merged input. It must agree with the inspected heads.
    if (
      merged.data.some((value, index) => value !== block.attention.outputLinear.input.data[index])
    ) {
      throw new Error(`Trace attention merge differs from engine at ${prefix}.`);
    }
    const residualAttention = builder.addValues(`${prefix}.residual.attention`, hidden, projected);
    if (
      residualAttention.data.some(
        (value, index) => value !== block.residualAfterAttention.data[index],
      )
    ) {
      throw new Error(`Trace attention residual differs from engine at ${prefix}.`);
    }
    const ln2 = builder.norm(
      `${prefix}.ln2`,
      residualAttention,
      registry.get(`${prefix}.ln2.scale`).value,
      registry.get(`${prefix}.ln2.offset`).value,
      config.layerNormEpsilon,
    );
    builder.linear(`${prefix}.mlp.preactivation`, block.mlpInput);
    builder.gelu(`${prefix}.mlp.gelu`, block.geluInput);
    const mlpOutput = builder.linear(`${prefix}.mlp.output`, block.mlpOutput);
    hidden = builder.addValues(`${prefix}.residual.mlp`, residualAttention, mlpOutput);
    // These checks protect the UI boundary if the numerical forward path changes.
    if (
      ln1.data.some((value, index) => value !== block.attention.queryLinear.input.data[index]) ||
      ln2.data.some((value, index) => value !== block.mlpInput.input.data[index])
    ) {
      throw new Error(`Trace normalization differs from engine at ${prefix}.`);
    }
  }
  const finalNormalized = builder.norm(
    'final_ln',
    hidden,
    registry.get('final_ln.scale').value,
    registry.get('final_ln.offset').value,
    config.layerNormEpsilon,
  );
  if (finalNormalized.data.some((value, index) => value !== cache.unembedding.input.data[index])) {
    throw new Error('Trace final normalization differs from engine.');
  }
  builder.logits('logits', cache, logits);
  const probabilities = softmaxForward(logits).output;
  builder.probabilities(
    'probabilities',
    logits,
    probabilities,
    new Uint8Array(logits.data.length).fill(1),
  );
  return builder.finish(tokenIds);
}
