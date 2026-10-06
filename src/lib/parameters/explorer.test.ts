import { describe, expect, it } from 'vitest';
import { DEFAULT_MODEL_CONFIG } from '../model/config';
import { encodeCheckpointWeights } from '../model/checkpoint';
import { emptyParameters } from '../model/parameters';
import { forwardStages } from '../trace/forward-stages';
import { buildParameterOverview, HISTOGRAM_BINS } from './explorer';

describe('parameter explorer', () => {
  it('covers every indexed tensor exactly once and reconciles the config total', () => {
    const registry = emptyParameters(DEFAULT_MODEL_CONFIG);
    registry.setValues('final_ln.scale', Array(DEFAULT_MODEL_CONFIG.dModel).fill(2));
    const index = encodeCheckpointWeights(DEFAULT_MODEL_CONFIG, registry).index;
    const overview = buildParameterOverview(index, registry, DEFAULT_MODEL_CONFIG);
    expect(overview.tensorCount).toBe(registry.size);
    expect(overview.elementCount).toBe(169_728);
    expect(overview.elementCount).toBe(overview.expectedCount);
    expect(overview.groups.reduce((sum, group) => sum + group.count, 0)).toBe(169_728);
    expect(new Set(overview.entries.map((entry) => entry.name)).size).toBe(registry.size);
    expect(overview.entries.map((entry) => entry.elementOffset)).toEqual(
      index.entries.map((entry) => entry.elementOffset),
    );
    const stages = new Set(forwardStages(DEFAULT_MODEL_CONFIG.nLayers).map((stage) => stage.name));
    expect(overview.entries.every((entry) => stages.has(entry.traceStage))).toBe(true);
  });

  it('computes exact population statistics and a mass-conserving histogram', () => {
    const registry = emptyParameters(DEFAULT_MODEL_CONFIG);
    const values = Array(DEFAULT_MODEL_CONFIG.dModel).fill(0);
    values.splice(0, 4, -2, 0, 2, 4);
    registry.setValues('final_ln.offset', values);
    const overview = buildParameterOverview(
      encodeCheckpointWeights(DEFAULT_MODEL_CONFIG, registry).index,
      registry,
      DEFAULT_MODEL_CONFIG,
    );
    const summary = overview.entries.find((entry) => entry.name === 'final_ln.offset')!;
    expect(summary.count).toBe(64);
    expect(summary.min).toBe(-2);
    expect(summary.max).toBe(4);
    expect(summary.mean).toBeCloseTo(4 / 64, 12);
    expect(summary.std).toBeCloseTo(
      Math.sqrt(values.reduce((sum, value) => sum + (value - 4 / 64) ** 2, 0) / 64),
      12,
    );
    expect(summary.histogram).toHaveLength(HISTOGRAM_BINS);
    expect(summary.histogram.reduce((sum, count) => sum + count, 0)).toBe(64);
    const constant = overview.entries.find((entry) => entry.name === 'final_ln.scale')!;
    expect(constant.histogram[0]).toBe(constant.count);
  });

  it('rejects an index that omits a registered tensor', () => {
    const registry = emptyParameters(DEFAULT_MODEL_CONFIG);
    const index = encodeCheckpointWeights(DEFAULT_MODEL_CONFIG, registry).index;
    expect(() =>
      buildParameterOverview(
        { ...index, entries: index.entries.slice(1) },
        registry,
        DEFAULT_MODEL_CONFIG,
      ),
    ).toThrow('complete model registry');
  });
});
