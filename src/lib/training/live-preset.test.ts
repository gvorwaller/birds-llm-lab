import { describe, expect, it } from 'vitest';
import { DEFAULT_MODEL_CONFIG } from '../model/config';
import { decodeTokenIds } from '../tokenizer/codec';
import { SMALL_BIRD_NAME_PRESET_SUMMARY, smallBirdNamePreset } from './live-preset';

describe('small bird-name teaching preset', () => {
  it('keeps the bundled train and validation names separate with a smaller model', () => {
    const preset = smallBirdNamePreset();
    expect(preset.model.dModel).toBeLessThan(DEFAULT_MODEL_CONFIG.dModel);
    expect(preset.model.nLayers).toBeLessThan(DEFAULT_MODEL_CONFIG.nLayers);
    expect(preset.sequences).toHaveLength(12);
    expect(preset.validationSequences).toHaveLength(4);
    const train = preset.sequences.map((ids) => decodeTokenIds(ids));
    const validation = (preset.validationSequences ?? []).map((ids) => decodeTokenIds(ids));
    expect(new Set([...train, ...validation]).size).toBe(16);
    expect(train[0]).toBe('Bird name: robin.\n');
    expect(validation[0]).toBe('Bird name: heron.\n');
    expect(SMALL_BIRD_NAME_PRESET_SUMMARY.trainingNames).not.toContain('heron');
    expect(decodeTokenIds(preset.samplePromptIds ?? [])).toBe('Bird name: ');
  });
});
