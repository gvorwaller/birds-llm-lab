import { svelte } from '@sveltejs/vite-plugin-svelte';
import { defineConfig } from 'vitest/config';
import { localOnlyBundlePlugin } from './scripts/local-only-bundle.ts';

export default defineConfig({
  plugins: [svelte(), localOnlyBundlePlugin()],
  server: {
    host: '127.0.0.1',
    port: 5301,
    strictPort: true,
  },
  preview: {
    host: '127.0.0.1',
    port: 5301,
    strictPort: true,
  },
  test: {
    include: [
      'src/**/*.test.ts',
      'server/**/*.test.ts',
      'scripts/**/*.test.ts',
      'tests/**/*.test.ts',
    ],
  },
});
