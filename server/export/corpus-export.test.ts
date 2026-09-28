import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import {
  CORPUS_QUERY,
  assignCorpusSplit,
  exportCorpus,
  parseNamedEnvironment,
  type DatabaseClient,
} from './corpus-export';

const temporaryDirectories: string[] = [];
const environmentContents = `
PGHOST=127.0.0.1
PGPORT=15436
PGDATABASE=birds_test
PGUSER="local-user"
PGPASSWORD='local-password'
UNRELATED_KEY=ignored
`;

afterEach(async () => {
  await Promise.all(
    temporaryDirectories.splice(0).map((path) => rm(path, { recursive: true, force: true })),
  );
});

class FakeClient implements DatabaseClient {
  readonly statements: string[] = [];
  connected = false;
  ended = false;

  constructor(
    private readonly sourceRows: Record<string, unknown>[],
    private readonly failureStatement?: string,
  ) {}

  async connect(): Promise<void> {
    this.connected = true;
  }

  async query<Row = Record<string, unknown>>(text: string): Promise<{ rows: Row[] }> {
    this.statements.push(text);
    if (text === this.failureStatement) throw new Error('planned query failure');
    if (text === 'SHOW transaction_read_only') {
      return { rows: [{ transaction_read_only: 'on' }] as Row[] };
    }
    if (text === CORPUS_QUERY) return { rows: this.sourceRows as Row[] };
    return { rows: [] };
  }

  async end(): Promise<void> {
    this.ended = true;
  }
}

async function setup(): Promise<{ directory: string; environmentPath: string }> {
  const directory = await mkdtemp(join(tmpdir(), 'birds-llm-export-test-'));
  temporaryDirectories.push(directory);
  const environmentPath = join(directory, '.env.test');
  await writeFile(environmentPath, environmentContents);
  return { directory, environmentPath };
}

function sourceRow(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    code: 'amekes',
    name: 'American Kestrel',
    sci: 'Falco sparverius',
    order: 'Falconiformes',
    family: 'Falconidae',
    extract: 'Line one.\r\nLine two.',
    sections: [
      { title: 'Habitat', text: 'Open country.' },
      { title: 'Broken', text: 4 },
    ],
    field_craft: 'AI-authored field note.',
    tags: ['habitat:open', 'habitat:open', ' diet:insects '],
    ...overrides,
  };
}

describe('local environment parsing', () => {
  it('reads only named values and accepts optional quotes', () => {
    expect(parseNamedEnvironment(environmentContents)).toEqual({
      PGHOST: '127.0.0.1',
      PGPORT: '15436',
      PGDATABASE: 'birds_test',
      PGUSER: 'local-user',
      PGPASSWORD: 'local-password',
    });
  });

  it('fails closed for any non-fixed endpoint', () => {
    expect(() =>
      parseNamedEnvironment(environmentContents.replace('127.0.0.1', 'localhost')),
    ).toThrow('fixed local birds_test endpoint');
  });
});

describe('corpus export', () => {
  it('uses a verified read-only transaction and writes reconciled artifacts', async () => {
    const { directory, environmentPath } = await setup();
    const client = new FakeClient([sourceRow()]);
    const result = await exportCorpus({
      outputDirectory: join(directory, 'data'),
      environmentPath,
      repositoryDirectory: directory,
      createClient: () => client,
      resolveGitRevision: async () => '1'.repeat(40),
      now: () => new Date('2026-09-28T12:00:00.000Z'),
    });

    expect(client.statements).toEqual([
      'BEGIN TRANSACTION READ ONLY',
      'SHOW transaction_read_only',
      CORPUS_QUERY,
      'COMMIT',
    ]);
    expect(client.ended).toBe(true);
    expect(CORPUS_QUERY).toContain("taxonomy.category = 'species'");
    expect(CORPUS_QUERY).toContain('enrichment.wikipedia_extract IS NOT NULL');
    expect(result.manifest).toMatchObject({
      sourceRows: 1,
      emittedRows: 1,
      skippedRows: 0,
      malformedSectionsSkipped: 1,
      splitOverrides: ['amekes'],
      sources: { fieldCraftExported: true, fieldCraftIncludedInTraining: false },
    });
    expect(result.manifest.splits.test.documents).toBe(1);
    const corpus = await readFile(result.corpusPath, 'utf8');
    const row = JSON.parse(corpus.trim()) as Record<string, unknown>;
    expect(row).toMatchObject({
      code: 'amekes',
      extract: 'Line one.\nLine two.',
      tags: ['diet:insects', 'habitat:open'],
      split: 'test',
    });
    expect(row.sections).toEqual([{ title: 'Habitat', text: 'Open country.' }]);
    const manifest = JSON.parse(await readFile(result.manifestPath, 'utf8')) as Record<
      string,
      unknown
    >;
    expect(manifest).not.toHaveProperty('username');
    expect(manifest).not.toHaveProperty('password');
  });

  it('rolls back and leaves no artifact when the source query fails', async () => {
    const { directory, environmentPath } = await setup();
    const client = new FakeClient([], CORPUS_QUERY);
    await expect(
      exportCorpus({
        outputDirectory: join(directory, 'data'),
        environmentPath,
        createClient: () => client,
      }),
    ).rejects.toThrow('planned query failure');
    expect(client.statements.at(-1)).toBe('ROLLBACK');
    expect(client.ended).toBe(true);
    await expect(readFile(join(directory, 'data', 'corpus.jsonl'))).rejects.toMatchObject({
      code: 'ENOENT',
    });
  });

  it('rolls back on cancellation before artifacts become visible', async () => {
    const { directory, environmentPath } = await setup();
    const client = new FakeClient([sourceRow()]);
    const controller = new AbortController();
    controller.abort();
    await expect(
      exportCorpus({
        outputDirectory: join(directory, 'data'),
        environmentPath,
        signal: controller.signal,
        createClient: () => client,
      }),
    ).rejects.toThrow('cancelled');
    expect(client.connected).toBe(false);
  });
});

describe('stable split assignment', () => {
  it('holds named demo species in the test split', () => {
    expect(assignCorpusSplit('amekes', 'American Kestrel')).toEqual({
      split: 'test',
      override: true,
    });
  });

  it('is deterministic for ordinary species', () => {
    expect(assignCorpusSplit('amerob', 'American Robin')).toEqual(
      assignCorpusSplit('amerob', 'American Robin'),
    );
  });
});
