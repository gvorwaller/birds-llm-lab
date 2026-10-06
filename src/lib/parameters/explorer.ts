import type { ModelConfig, WeightIndexArtifact } from '../data/schemas';
import type { Parameter, ParameterRegistry } from '../model/parameters';
import { parameterSpecs } from '../model/parameters';

export interface ParameterSummary {
  readonly name: string;
  readonly shape: readonly number[];
  readonly count: number;
  readonly elementOffset: number;
  readonly min: number;
  readonly max: number;
  readonly mean: number;
  readonly std: number;
  readonly histogram: readonly number[];
  readonly parameter: Parameter;
  readonly traceStage: string;
}

export interface ParameterGroup {
  readonly name: string;
  readonly count: number;
  readonly entries: readonly ParameterSummary[];
}

export interface ParameterOverview {
  readonly groups: readonly ParameterGroup[];
  readonly entries: readonly ParameterSummary[];
  readonly tensorCount: number;
  readonly elementCount: number;
  readonly expectedCount: number;
}

export const HISTOGRAM_BINS = 20;
export const HEATMAP_TILE = 32;

export function traceStageForParameter(name: string): string {
  if (name === 'token_embedding.weight') return 'embedding.token';
  if (name === 'position_embedding.weight') return 'embedding.position';
  if (name === 'final_ln.scale' || name === 'final_ln.offset') return 'final_ln';
  if (name.startsWith('unembedding.')) return 'logits';
  const parts = name.split('.');
  if (parts[0] !== 'blocks' || parts.length < 4) throw new Error(`No trace stage for ${name}.`);
  const prefix = `${parts[0]}.${parts[1]}`;
  if (parts[2] === 'ln1' || parts[2] === 'ln2') return `${prefix}.${parts[2]}`;
  if (parts[2] === 'attn') {
    return parts[3] === 'out' ? `${prefix}.attn.projection` : `${prefix}.attn.${parts[3]}`;
  }
  if (parts[2] === 'mlp') {
    return parts[3] === 'in' ? `${prefix}.mlp.preactivation` : `${prefix}.mlp.output`;
  }
  throw new Error(`No trace stage for ${name}.`);
}

function groupName(name: string): string {
  const parts = name.split('.');
  return parts[0] === 'blocks' ? `${parts[0]}.${parts[1]}` : parts[0];
}

export function summarizeParameter(parameter: Parameter, elementOffset = 0): ParameterSummary {
  const values = parameter.value.data;
  if (values.length === 0) throw new Error(`Empty parameter ${parameter.name}.`);
  let min = Infinity;
  let max = -Infinity;
  let mean = 0;
  let m2 = 0;
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    if (!Number.isFinite(value)) throw new Error(`Non-finite parameter ${parameter.name}.`);
    min = Math.min(min, value);
    max = Math.max(max, value);
    const delta = value - mean;
    mean += delta / (index + 1);
    m2 += delta * (value - mean);
  }
  const histogram = Array<number>(HISTOGRAM_BINS).fill(0);
  for (const value of values) {
    const bin =
      min === max
        ? 0
        : Math.min(HISTOGRAM_BINS - 1, Math.floor(((value - min) / (max - min)) * HISTOGRAM_BINS));
    histogram[bin] += 1;
  }
  return {
    name: parameter.name,
    shape: parameter.shape,
    count: values.length,
    elementOffset,
    min,
    max,
    mean,
    std: Math.sqrt(m2 / values.length),
    histogram,
    parameter,
    traceStage: traceStageForParameter(parameter.name),
  };
}

export function buildParameterOverview(
  index: WeightIndexArtifact,
  registry: ParameterRegistry,
  config: ModelConfig,
): ParameterOverview {
  const expectedCount = parameterSpecs(config).reduce(
    (sum, spec) => sum + spec.shape.reduce((product, dimension) => product * dimension, 1),
    0,
  );
  const names = new Set<string>();
  const summaries = index.entries.map((entry) => {
    if (names.has(entry.name)) throw new Error(`Duplicate indexed parameter ${entry.name}.`);
    names.add(entry.name);
    const parameter = registry.get(entry.name);
    if (
      entry.elementCount !== parameter.value.data.length ||
      entry.shape.join(',') !== parameter.shape.join(',')
    ) {
      throw new Error(`Index and registry disagree for ${entry.name}.`);
    }
    return summarizeParameter(parameter, entry.elementOffset);
  });
  const elementCount = summaries.reduce((sum, item) => sum + item.count, 0);
  if (
    summaries.length !== registry.size ||
    elementCount !== registry.elementCount ||
    elementCount !== expectedCount
  ) {
    throw new Error('Parameter explorer does not cover the complete model registry.');
  }
  const groups = new Map<string, ParameterSummary[]>();
  for (const summary of summaries) {
    const name = groupName(summary.name);
    const entries = groups.get(name) ?? [];
    entries.push(summary);
    groups.set(name, entries);
  }
  return {
    groups: [...groups].map(([name, entries]) => ({
      name,
      count: entries.reduce((sum, item) => sum + item.count, 0),
      entries,
    })),
    entries: summaries,
    tensorCount: summaries.length,
    elementCount,
    expectedCount,
  };
}
