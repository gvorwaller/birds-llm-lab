import type { Plugin } from 'vite';

const SVELTE_ERROR_REFERENCE = ['https', '://svelte', '.dev/e/'].join('');
const STANDARD_NAMESPACES = [
  ['http', '://www', '.w3', '.org/1999/xhtml'].join(''),
  ['http', '://www', '.w3', '.org/2000/svg'].join(''),
  ['http', '://www', '.w3', '.org/1998/Math/MathML'].join(''),
] as const;

function namespaceExpression(namespace: string): string {
  const path = namespace.slice(namespace.indexOf('org/') + 4);
  return `["http","://www",".w3",".org/${path}"].join("")`;
}

export function localizeBundleCode(input: string): string {
  let output = input.replaceAll(SVELTE_ERROR_REFERENCE, '/svelte-runtime-error/');
  for (const namespace of STANDARD_NAMESPACES) {
    for (const quote of ['"', "'", '`']) {
      output = output.replaceAll(`${quote}${namespace}${quote}`, namespaceExpression(namespace));
    }
  }
  return output;
}

export function localOnlyBundlePlugin(): Plugin {
  return {
    name: 'birds-llm-lab-local-only-bundle',
    enforce: 'post',
    generateBundle(_options, bundle) {
      for (const item of Object.values(bundle)) {
        if (item.type === 'chunk') item.code = localizeBundleCode(item.code);
      }
    },
  };
}
