import { describe, expect, it } from 'vitest';
import { routes } from '../navigation';
import { glossaryById, glossaryTerms } from './terms';

const required = [
  'token',
  'vocabulary',
  'embedding',
  'parameter-weight',
  'bias',
  'matrix-multiply',
  'dot-product',
  'attention',
  'head',
  'residual-stream',
  'layernorm',
  'mlp',
  'gelu',
  'logit',
  'softmax',
  'temperature',
  'cross-entropy',
  'loss',
  'gradient',
  'backpropagation',
  'learning-rate',
  'adam',
  'overfitting',
  'validation-set',
  'context-window',
  'hallucination',
];

describe('glossary', () => {
  it('covers every original-plan term once with working local cross-links', () => {
    expect(glossaryTerms.map((term) => term.id)).toEqual(required);
    expect(glossaryById.size).toBe(required.length);
    for (const term of glossaryTerms) {
      expect(term.plain.length).toBeGreaterThan(35);
      expect(term.detail.length).toBeGreaterThan(35);
      for (const related of term.related) expect(glossaryById.has(related)).toBe(true);
      expect(routes.some((route) => route.path === term.workbench.href.split('?')[0])).toBe(true);
    }
  });

  it('gives literal mechanisms for metaphor terms and formulas for mathematical terms', () => {
    expect(glossaryById.get('attention')?.detail).toContain('“Looks at” is a metaphor. Literally');
    expect(glossaryById.get('hallucination')?.detail).toContain(
      '“Hallucination” is a metaphor. Literally',
    );
    for (const id of [
      'matrix-multiply',
      'attention',
      'layernorm',
      'gelu',
      'softmax',
      'cross-entropy',
      'gradient',
      'adam',
    ]) {
      expect(glossaryById.get(id)?.formula).toBeTruthy();
    }
  });
});
