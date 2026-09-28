import { createHash, randomBytes } from 'node:crypto';
import { execFile } from 'node:child_process';
import { mkdir, readFile, rename, rm, stat, writeFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { basename, dirname, join, resolve } from 'node:path';
import { promisify } from 'node:util';
import { Client } from 'pg';
import {
  ARTIFACT_FORMAT_VERSION,
  parseCorpusManifest,
  parseCorpusRow,
  serializeCorpusRow,
  serializeStableJson,
  type CorpusManifest,
  type CorpusRow,
  type CorpusSection,
  type CorpusSplit,
} from '../../src/lib/data/schemas.js';

const execFileAsync = promisify(execFile);

const EXPECTED_DATABASE = {
  host: '127.0.0.1',
  port: 15436,
  database: 'birds_test',
} as const;

const ENVIRONMENT_KEYS = ['PGHOST', 'PGPORT', 'PGDATABASE', 'PGUSER', 'PGPASSWORD'] as const;
type EnvironmentKey = (typeof ENVIRONMENT_KEYS)[number];
type NamedEnvironment = Record<EnvironmentKey, string>;

export const DEMO_SPECIES_TEST_OVERRIDES = [
  'American Kestrel',
  'Northern Cardinal',
  'Osprey',
  'Wandering Albatross',
] as const;

export const CORPUS_QUERY = `
SELECT
  enrichment.species_code AS code,
  taxonomy.com_name AS name,
  taxonomy.sci_name AS sci,
  taxonomy.order_name AS "order",
  taxonomy.family_sci_name AS family,
  enrichment.wikipedia_extract AS extract,
  enrichment.wikipedia_sections AS sections,
  enrichment.field_craft,
  enrichment.tags
FROM species_enrichment AS enrichment
INNER JOIN taxonomy_cache AS taxonomy
  ON taxonomy.species_code = enrichment.species_code
WHERE taxonomy.category = 'species'
  AND enrichment.wikipedia_extract IS NOT NULL
ORDER BY enrichment.species_code ASC
`.trim();

interface QueryResult<Row = Record<string, unknown>> {
  rows: Row[];
}

export interface DatabaseClient {
  connect(): Promise<void>;
  query<Row = Record<string, unknown>>(text: string): Promise<QueryResult<Row>>;
  end(): Promise<void>;
}

export type ExportPhase =
  'connecting' | 'reading' | 'normalizing' | 'writing' | 'committing' | 'complete';

export interface ExportProgress {
  phase: ExportPhase;
  processedRows: number;
  totalRows: number | null;
}

export interface ExportCorpusOptions {
  outputDirectory?: string;
  environmentPath?: string;
  repositoryDirectory?: string;
  signal?: AbortSignal;
  onProgress?: (progress: ExportProgress) => void;
  now?: () => Date;
  createClient?: (configuration: NamedEnvironment) => DatabaseClient;
  resolveGitRevision?: (repositoryDirectory: string) => Promise<string | null>;
}

export interface ExportCorpusResult {
  corpusPath: string;
  manifestPath: string;
  manifest: CorpusManifest;
}

interface SourceRow {
  code?: unknown;
  name?: unknown;
  sci?: unknown;
  order?: unknown;
  family?: unknown;
  extract?: unknown;
  sections?: unknown;
  field_craft?: unknown;
  tags?: unknown;
}

interface NormalizationResult {
  rows: CorpusRow[];
  skippedReasons: Record<string, number>;
  malformedSectionsSkipped: number;
  sourceBytes: CorpusManifest['sourceBytes'];
  splitOverrides: string[];
}

export class ExportCancelledError extends Error {
  constructor() {
    super('Corpus export was cancelled.');
    this.name = 'ExportCancelledError';
  }
}

function checkCancelled(signal: AbortSignal | undefined): void {
  if (signal?.aborted) throw new ExportCancelledError();
}

function stripOptionalQuotes(value: string): string {
  const trimmed = value.trim();
  if (trimmed.length >= 2) {
    const first = trimmed[0];
    const last = trimmed[trimmed.length - 1];
    if ((first === '"' && last === '"') || (first === "'" && last === "'")) {
      return trimmed.slice(1, -1);
    }
  }
  return trimmed;
}

export function parseNamedEnvironment(contents: string): NamedEnvironment {
  const parsed = new Map<string, string>();
  for (const rawLine of contents.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (line.length === 0 || line.startsWith('#')) continue;
    const equals = line.indexOf('=');
    if (equals < 1) continue;
    const key = line.slice(0, equals).trim();
    if (!ENVIRONMENT_KEYS.includes(key as EnvironmentKey)) continue;
    parsed.set(key, stripOptionalQuotes(line.slice(equals + 1)));
  }

  const values = Object.fromEntries(
    ENVIRONMENT_KEYS.map((key) => {
      const value = parsed.get(key);
      if (value === undefined || value.length === 0) {
        throw new Error(
          `The required ${key} setting is missing from the local test environment file.`,
        );
      }
      return [key, value];
    }),
  ) as NamedEnvironment;

  if (
    values.PGHOST !== EXPECTED_DATABASE.host ||
    values.PGPORT !== String(EXPECTED_DATABASE.port) ||
    values.PGDATABASE !== EXPECTED_DATABASE.database
  ) {
    throw new Error(
      'Refusing to export: the database settings do not match the fixed local birds_test endpoint.',
    );
  }

  return values;
}

function createPostgresClient(configuration: NamedEnvironment): DatabaseClient {
  const client = new Client({
    host: EXPECTED_DATABASE.host,
    port: EXPECTED_DATABASE.port,
    database: EXPECTED_DATABASE.database,
    user: configuration.PGUSER,
    password: configuration.PGPASSWORD,
    application_name: 'birds-llm-lab-read-only-export',
  });
  return {
    async connect() {
      await client.connect();
    },
    async query<Row = Record<string, unknown>>(text: string) {
      const result = await client.query(text);
      return { rows: result.rows as Row[] };
    },
    async end() {
      await client.end();
    },
  };
}

function normalizeText(value: unknown, allowEmpty = false): string | null {
  if (typeof value !== 'string') return null;
  const normalized = value.replace(/\r\n?/g, '\n').trim();
  return normalized.length > 0 || allowEmpty ? normalized : null;
}

function decodeJson(value: unknown): unknown {
  if (typeof value !== 'string') return value;
  const trimmed = value.trim();
  if (!trimmed.startsWith('[') && !trimmed.startsWith('{')) return value;
  try {
    return JSON.parse(trimmed) as unknown;
  } catch {
    return value;
  }
}

function normalizeSections(value: unknown): { sections: CorpusSection[]; skipped: number } {
  const decoded = decodeJson(value);
  if (decoded === null || decoded === undefined) return { sections: [], skipped: 0 };
  if (!Array.isArray(decoded)) return { sections: [], skipped: 1 };

  const sections: CorpusSection[] = [];
  let skipped = 0;
  for (const candidate of decoded) {
    if (typeof candidate !== 'object' || candidate === null || Array.isArray(candidate)) {
      skipped += 1;
      continue;
    }
    const section = candidate as Record<string, unknown>;
    const title = normalizeText(section.title);
    const text = normalizeText(section.text);
    if (title === null || text === null) {
      skipped += 1;
      continue;
    }
    sections.push({ title, text });
  }
  return { sections, skipped };
}

function normalizeTags(value: unknown): string[] {
  const decoded = decodeJson(value);
  if (!Array.isArray(decoded)) return [];
  return [
    ...new Set(
      decoded.map((tag) => normalizeText(tag)).filter((tag): tag is string => tag !== null),
    ),
  ].sort((left, right) => left.localeCompare(right));
}

export function assignCorpusSplit(
  code: string,
  name: string,
): { split: CorpusSplit; override: boolean } {
  if ((DEMO_SPECIES_TEST_OVERRIDES as readonly string[]).includes(name)) {
    return { split: 'test', override: true };
  }
  const digest = createHash('sha256').update(`split-v1:${code}`).digest();
  const bucket = digest.readUInt32BE(0) % 100;
  if (bucket < 90) return { split: 'train', override: false };
  if (bucket < 95) return { split: 'validation', override: false };
  return { split: 'test', override: false };
}

function byteLength(value: string | null): number {
  return value === null ? 0 : Buffer.byteLength(value, 'utf8');
}

function increment(record: Record<string, number>, key: string): void {
  record[key] = (record[key] ?? 0) + 1;
}

function normalizeSourceRows(
  sourceRows: SourceRow[],
  signal: AbortSignal | undefined,
  onProgress: ExportCorpusOptions['onProgress'],
): NormalizationResult {
  const rows: CorpusRow[] = [];
  const skippedReasons: Record<string, number> = {};
  const sourceBytes = { wikipediaExtracts: 0, wikipediaSections: 0, fieldCraft: 0 };
  const splitOverrides: string[] = [];
  let malformedSectionsSkipped = 0;

  for (const [index, source] of sourceRows.entries()) {
    checkCancelled(signal);
    const code = normalizeText(source.code);
    const name = normalizeText(source.name);
    const sci = normalizeText(source.sci);
    const extract = normalizeText(source.extract);
    if (code === null || name === null || sci === null || extract === null) {
      increment(skippedReasons, 'missing_required_text');
      continue;
    }

    const normalizedSections = normalizeSections(source.sections);
    malformedSectionsSkipped += normalizedSections.skipped;
    const fieldCraft = normalizeText(source.field_craft);
    const split = assignCorpusSplit(code, name);
    if (split.override) splitOverrides.push(code);

    const row = parseCorpusRow({
      code,
      name,
      sci,
      order: normalizeText(source.order, true),
      family: normalizeText(source.family, true),
      extract,
      sections: normalizedSections.sections,
      field_craft: fieldCraft,
      tags: normalizeTags(source.tags),
      split: split.split,
    });
    rows.push(row);
    sourceBytes.wikipediaExtracts += byteLength(row.extract);
    sourceBytes.wikipediaSections += row.sections.reduce(
      (total, section) => total + byteLength(section.title) + byteLength(section.text),
      0,
    );
    sourceBytes.fieldCraft += byteLength(row.field_craft);

    if (index % 100 === 0) {
      onProgress?.({
        phase: 'normalizing',
        processedRows: index + 1,
        totalRows: sourceRows.length,
      });
    }
  }

  rows.sort((left, right) => left.code.localeCompare(right.code));
  splitOverrides.sort((left, right) => left.localeCompare(right));
  return { rows, skippedReasons, malformedSectionsSkipped, sourceBytes, splitOverrides };
}

export function renderTrainingDocument(row: CorpusRow): string {
  const metadata = [
    `<|bos|>Common name: ${row.name}`,
    `Scientific name: ${row.sci}`,
    `Order: ${row.order ?? ''}`,
    `Family: ${row.family ?? ''}`,
    '',
    row.extract ?? '',
  ];
  for (const section of row.sections) {
    metadata.push('', section.title, section.text);
  }
  metadata.push('<|eos|>');
  return metadata.join('\n');
}

async function resolveRevision(repositoryDirectory: string): Promise<string | null> {
  try {
    const { stdout } = await execFileAsync('git', ['rev-parse', 'HEAD'], {
      cwd: repositoryDirectory,
      encoding: 'utf8',
    });
    const revision = stdout.trim();
    return /^[a-f0-9]{40}$/.test(revision) ? revision : null;
  } catch {
    return null;
  }
}

function temporaryPath(targetPath: string): string {
  return join(
    dirname(targetPath),
    `.${basename(targetPath)}.${process.pid}.${randomBytes(8).toString('hex')}.tmp`,
  );
}

async function validateCorpusFile(
  path: string,
  expectedRows: number,
  expectedHash: string,
): Promise<void> {
  const contents = await readFile(path);
  const actualHash = createHash('sha256').update(contents).digest('hex');
  if (actualHash !== expectedHash)
    throw new Error('Temporary corpus hash did not match the in-memory corpus hash.');
  const text = contents.toString('utf8');
  const lines = text.length === 0 ? [] : text.trimEnd().split('\n');
  if (lines.length !== expectedRows)
    throw new Error('Temporary corpus row count did not reconcile.');
  for (const line of lines) parseCorpusRow(JSON.parse(line) as unknown);
}

async function pathExists(path: string): Promise<boolean> {
  try {
    await stat(path);
    return true;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return false;
    throw error;
  }
}

async function replaceArtifacts(
  temporaryCorpusPath: string,
  corpusPath: string,
  temporaryManifestPath: string,
  manifestPath: string,
): Promise<void> {
  const backupSuffix = `.backup.${process.pid}.${randomBytes(8).toString('hex')}`;
  const corpusBackup = `${corpusPath}${backupSuffix}`;
  const manifestBackup = `${manifestPath}${backupSuffix}`;
  const hadCorpus = await pathExists(corpusPath);
  const hadManifest = await pathExists(manifestPath);
  try {
    if (hadCorpus) await rename(corpusPath, corpusBackup);
    if (hadManifest) await rename(manifestPath, manifestBackup);
    await rename(temporaryCorpusPath, corpusPath);
    await rename(temporaryManifestPath, manifestPath);
    await rm(corpusBackup, { force: true });
    await rm(manifestBackup, { force: true });
  } catch (error) {
    await rm(corpusPath, { force: true });
    await rm(manifestPath, { force: true });
    if (hadCorpus && (await pathExists(corpusBackup))) await rename(corpusBackup, corpusPath);
    if (hadManifest && (await pathExists(manifestBackup)))
      await rename(manifestBackup, manifestPath);
    throw error;
  }
}

export async function exportCorpus(options: ExportCorpusOptions = {}): Promise<ExportCorpusResult> {
  const outputDirectory = resolve(options.outputDirectory ?? join(process.cwd(), 'data'));
  const environmentPath = resolve(options.environmentPath ?? join(homedir(), 'birds', '.env.test'));
  const repositoryDirectory = resolve(options.repositoryDirectory ?? process.cwd());
  const onProgress = options.onProgress;
  const now = options.now ?? (() => new Date());
  const configuration = parseNamedEnvironment(await readFile(environmentPath, 'utf8'));
  const client = (options.createClient ?? createPostgresClient)(configuration);
  const corpusPath = join(outputDirectory, 'corpus.jsonl');
  const manifestPath = join(outputDirectory, 'manifest.json');
  const temporaryCorpusPath = temporaryPath(corpusPath);
  const temporaryManifestPath = temporaryPath(manifestPath);
  let transactionOpen = false;

  await mkdir(outputDirectory, { recursive: true });
  checkCancelled(options.signal);
  onProgress?.({ phase: 'connecting', processedRows: 0, totalRows: null });

  try {
    await client.connect();
    await client.query('BEGIN TRANSACTION READ ONLY');
    transactionOpen = true;
    const readOnly = await client.query<{ transaction_read_only: string }>(
      'SHOW transaction_read_only',
    );
    if (readOnly.rows[0]?.transaction_read_only !== 'on') {
      throw new Error('The database did not confirm a read-only transaction.');
    }

    checkCancelled(options.signal);
    onProgress?.({ phase: 'reading', processedRows: 0, totalRows: null });
    const queryResult = await client.query<SourceRow>(CORPUS_QUERY);
    checkCancelled(options.signal);
    const normalized = normalizeSourceRows(queryResult.rows, options.signal, onProgress);
    const corpusContents =
      normalized.rows.length === 0 ? '' : `${normalized.rows.map(serializeCorpusRow).join('\n')}\n`;
    const corpusBuffer = Buffer.from(corpusContents, 'utf8');
    const corpusSha256 = createHash('sha256').update(corpusBuffer).digest('hex');
    const splits: CorpusManifest['splits'] = {
      train: { documents: 0, bytes: 0 },
      validation: { documents: 0, bytes: 0 },
      test: { documents: 0, bytes: 0 },
    };
    for (const row of normalized.rows) {
      splits[row.split].documents += 1;
      splits[row.split].bytes += Buffer.byteLength(renderTrainingDocument(row), 'utf8');
    }

    const gitRevision = await (options.resolveGitRevision ?? resolveRevision)(repositoryDirectory);
    const warnings: string[] = [];
    if (normalized.malformedSectionsSkipped > 0) {
      warnings.push(
        `Skipped ${normalized.malformedSectionsSkipped} malformed Wikipedia section entries.`,
      );
    }
    if (gitRevision === null) warnings.push('Exporter git revision was unavailable.');
    const skippedRows = Object.values(normalized.skippedReasons).reduce(
      (sum, count) => sum + count,
      0,
    );
    const manifest = parseCorpusManifest({
      formatVersion: ARTIFACT_FORMAT_VERSION,
      exportedAt: now().toISOString(),
      database: {
        host: EXPECTED_DATABASE.host,
        port: EXPECTED_DATABASE.port,
        name: EXPECTED_DATABASE.database,
      },
      sourceRows: queryResult.rows.length,
      emittedRows: normalized.rows.length,
      skippedRows,
      skippedReasons: normalized.skippedReasons,
      malformedSectionsSkipped: normalized.malformedSectionsSkipped,
      splits,
      sourceBytes: normalized.sourceBytes,
      corpusBytes: corpusBuffer.byteLength,
      corpusSha256,
      splitAlgorithm: 'sha256-bucket',
      splitVersion: 'split-v1',
      splitOverrides: normalized.splitOverrides,
      templateVersion: 'corpus-v1',
      sources: {
        wikipediaExtracts: true,
        wikipediaSections: true,
        fieldCraftExported: true,
        fieldCraftIncludedInTraining: false,
      },
      gitRevision,
      warnings,
    });

    checkCancelled(options.signal);
    onProgress?.({
      phase: 'writing',
      processedRows: normalized.rows.length,
      totalRows: queryResult.rows.length,
    });
    await writeFile(temporaryCorpusPath, corpusBuffer, { flag: 'wx', mode: 0o600 });
    await validateCorpusFile(temporaryCorpusPath, manifest.emittedRows, manifest.corpusSha256);
    await writeFile(temporaryManifestPath, serializeStableJson(manifest), {
      flag: 'wx',
      mode: 0o600,
    });
    parseCorpusManifest(JSON.parse(await readFile(temporaryManifestPath, 'utf8')) as unknown);

    checkCancelled(options.signal);
    onProgress?.({
      phase: 'committing',
      processedRows: normalized.rows.length,
      totalRows: queryResult.rows.length,
    });
    await client.query('COMMIT');
    transactionOpen = false;
    await replaceArtifacts(temporaryCorpusPath, corpusPath, temporaryManifestPath, manifestPath);
    onProgress?.({
      phase: 'complete',
      processedRows: normalized.rows.length,
      totalRows: queryResult.rows.length,
    });
    return { corpusPath, manifestPath, manifest };
  } catch (error) {
    if (transactionOpen) {
      try {
        await client.query('ROLLBACK');
      } catch {
        // Preserve the original failure while still making the rollback attempt observable in tests/logs.
      }
    }
    await rm(temporaryCorpusPath, { force: true });
    await rm(temporaryManifestPath, { force: true });
    throw error;
  } finally {
    await client.end();
  }
}
