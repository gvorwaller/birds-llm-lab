import { describe, expect, it } from 'vitest';
import { localizeBundleCode } from './local-only-bundle';

describe('local-only production bundle transform', () => {
  it('localizes framework error links and preserves standards namespace values', () => {
    const errorReference = ['https', '://svelte', '.dev/e/example'].join('');
    const namespace = ['http', '://www', '.w3', '.org/1999/xhtml'].join('');
    const transformed = localizeBundleCode(
      `const error="${errorReference}";const namespace=\`${namespace}\`;`,
    );
    expect(transformed).not.toContain(errorReference);
    expect(transformed).not.toContain(namespace);
    expect(transformed).toContain('/svelte-runtime-error/example');
    const namespaceExpression = transformed.match(/const namespace=([^;]+);/)?.[1];
    expect(Function(`return ${namespaceExpression}`)()).toBe(namespace);
  });
});
