import { readFile, readdir } from 'node:fs/promises';
import { extname, join, relative } from 'node:path';
import { describe, expect, it } from 'vitest';

const repositoryRoot = process.cwd();
const roots = ['src', 'server', 'scripts'];
const checkedExtensions = new Set(['.ts', '.svelte', '.css', '.html', '.json']);

async function filesUnder(path: string): Promise<string[]> {
  const entries = await readdir(path, { withFileTypes: true });
  const files: string[] = [];
  for (const entry of entries) {
    const child = join(path, entry.name);
    if (entry.isDirectory()) files.push(...(await filesUnder(child)));
    else if (checkedExtensions.has(extname(entry.name))) files.push(child);
  }
  return files;
}

describe('executable source safety scan', () => {
  it('contains no production endpoint or embedded Postgres connection URI', async () => {
    const files = (
      await Promise.all(roots.map((root) => filesUnder(join(repositoryRoot, root))))
    ).flat();
    files.push(
      join(repositoryRoot, 'package.json'),
      join(repositoryRoot, 'vite.config.ts'),
      join(repositoryRoot, 'index.html'),
    );
    const forbiddenDomain = ['birds', 'gaylon', 'photos'].join('.');
    const connectionScheme = ['post', 'gres', 'ql://'].join('');
    const failures: string[] = [];
    for (const path of files) {
      const contents = await readFile(path, 'utf8');
      if (contents.includes(forbiddenDomain) || contents.includes(connectionScheme)) {
        failures.push(relative(repositoryRoot, path));
      }
    }
    expect(failures).toEqual([]);
  });
});
