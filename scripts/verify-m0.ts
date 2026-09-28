import { createHash } from 'node:crypto';
import { readFile, readdir } from 'node:fs/promises';
import { homedir } from 'node:os';
import { extname, join, relative } from 'node:path';
import { parseNamedEnvironment, renderTrainingDocument } from '../server/export/corpus-export.js';
import { parseCorpusManifest, parseCorpusRow, type CorpusSplit } from '../src/lib/data/schemas.js';

const root = process.cwd();
const scannedExtensions = new Set(['.ts', '.svelte', '.css', '.html', '.json', '.js']);

async function filesUnder(path: string): Promise<string[]> {
  const entries = await readdir(path, { withFileTypes: true });
  const files: string[] = [];
  for (const entry of entries) {
    const child = join(path, entry.name);
    if (entry.isDirectory()) files.push(...(await filesUnder(child)));
    else if (scannedExtensions.has(extname(entry.name))) files.push(child);
  }
  return files;
}

async function inspectableFiles(): Promise<string[]> {
  const files = (
    await Promise.all(
      ['src', 'server', 'scripts', 'dist'].map((directory) => filesUnder(join(root, directory))),
    )
  ).flat();
  files.push(
    join(root, 'index.html'),
    join(root, 'vite.config.ts'),
    join(root, 'package.json'),
    join(root, 'data', 'manifest.json'),
    join(root, 'data', 'job-state.json'),
  );
  return files;
}

function assertNoUnexpectedUrl(path: string, contents: string): void {
  for (const match of contents.matchAll(/https?:\/\/([^/"'`\s]+)/g)) {
    const host = match[1];
    if (host.startsWith('${')) continue;
    const loopbackPort = '(?::(?:\\d+|\\$\\{[^}]+\\}))?';
    if (
      !new RegExp(`^localhost${loopbackPort}$`).test(host) &&
      !new RegExp(`^127\\.0\\.0\\.1${loopbackPort}$`).test(host)
    ) {
      throw new Error(`${relative(root, path)} contains a non-loopback URL.`);
    }
  }
}

async function verifySafety(): Promise<void> {
  const environment = parseNamedEnvironment(
    await readFile(join(homedir(), 'birds', '.env.test'), 'utf8'),
  );
  const files = await inspectableFiles();
  for (const path of files) {
    const contents = await readFile(path, 'utf8');
    assertNoUnexpectedUrl(path, contents);
    if (contents.includes(environment.PGPASSWORD)) {
      throw new Error(`${relative(root, path)} contains the database password value.`);
    }
    if (
      (path.endsWith('manifest.json') || path.endsWith('job-state.json')) &&
      contents.includes(environment.PGUSER)
    ) {
      throw new Error(`${relative(root, path)} contains the database username value.`);
    }
  }
}

async function verifyCorpus(): Promise<void> {
  const corpus = await readFile(join(root, 'data', 'corpus.jsonl'));
  const manifest = parseCorpusManifest(
    JSON.parse(await readFile(join(root, 'data', 'manifest.json'), 'utf8')) as unknown,
  );
  const hash = createHash('sha256').update(corpus).digest('hex');
  if (hash !== manifest.corpusSha256)
    throw new Error('Corpus SHA-256 does not match the manifest.');
  if (corpus.byteLength !== manifest.corpusBytes)
    throw new Error('Corpus bytes do not match the manifest.');
  const lines = corpus.length === 0 ? [] : corpus.toString('utf8').trimEnd().split('\n');
  if (lines.length !== manifest.emittedRows)
    throw new Error('Corpus rows do not match the manifest.');

  const splits: Record<CorpusSplit, { documents: number; bytes: number }> = {
    train: { documents: 0, bytes: 0 },
    validation: { documents: 0, bytes: 0 },
    test: { documents: 0, bytes: 0 },
  };
  for (const line of lines) {
    const row = parseCorpusRow(JSON.parse(line) as unknown);
    splits[row.split].documents += 1;
    splits[row.split].bytes += Buffer.byteLength(renderTrainingDocument(row), 'utf8');
  }
  if (JSON.stringify(splits) !== JSON.stringify(manifest.splits)) {
    throw new Error('Corpus split counts or document bytes do not match the manifest.');
  }
  if (manifest.sourceRows !== manifest.emittedRows + manifest.skippedRows) {
    throw new Error('Source, emitted, and skipped row counts do not reconcile.');
  }
  console.log(
    `M0 artifact verification passed: ${manifest.emittedRows} rows, ${manifest.corpusBytes} bytes, SHA-256 ${manifest.corpusSha256}.`,
  );
}

await verifySafety();
await verifyCorpus();
