import { createHash } from 'node:crypto';
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import {
  serializeCorpusRow,
  type CorpusManifest,
  type CorpusRow,
} from '../../src/lib/data/schemas';
import { renderTrainingDocument } from '../export/corpus-export';
import { CorpusEvidenceIndex, EvidenceQueryError, EvidenceUnavailableError } from './index';

const directories: string[] = [];

afterEach(async () => {
  await Promise.all(
    directories.splice(0).map((path) => rm(path, { recursive: true, force: true })),
  );
});

function row(code: string, name: string, split: CorpusRow['split'], extract: string): CorpusRow {
  return {
    code,
    name,
    sci: `Avis ${code}`,
    order: 'Passeriformes',
    family: 'Birdidae',
    extract,
    sections: [{ title: 'Habitat', text: 'Coastal waters near the river.' }],
    field_craft: 'Secret-only field craft advice.',
    tags: ['secret-only-tag'],
    split,
  };
}

async function fixture(rows: CorpusRow[]): Promise<{
  index: CorpusEvidenceIndex;
  write: (nextRows: CorpusRow[]) => Promise<void>;
  corpusPath: string;
  manifestPath: string;
}> {
  const directory = await mkdtemp(join(tmpdir(), 'birds-evidence-test-'));
  directories.push(directory);
  await mkdir(directory, { recursive: true });
  const corpusPath = join(directory, 'corpus.jsonl');
  const manifestPath = join(directory, 'manifest.json');
  const write = async (nextRows: CorpusRow[]) => {
    const corpus = Buffer.from(`${nextRows.map(serializeCorpusRow).join('\n')}\n`);
    const splitCounts = { train: 0, validation: 0, test: 0 };
    for (const item of nextRows) splitCounts[item.split] += 1;
    const manifest: CorpusManifest = {
      formatVersion: 1,
      exportedAt: '2026-10-06T12:00:00.000Z',
      database: { host: '127.0.0.1', port: 15436, name: 'birds_test' },
      sourceRows: nextRows.length,
      emittedRows: nextRows.length,
      skippedRows: 0,
      skippedReasons: {},
      malformedSectionsSkipped: 0,
      splits: {
        train: { documents: splitCounts.train, bytes: 0 },
        validation: { documents: splitCounts.validation, bytes: 0 },
        test: { documents: splitCounts.test, bytes: 0 },
      },
      sourceBytes: { wikipediaExtracts: 0, wikipediaSections: 0, fieldCraft: 0 },
      corpusBytes: corpus.length,
      corpusSha256: createHash('sha256').update(corpus).digest('hex'),
      splitAlgorithm: 'sha256-bucket',
      splitVersion: 'split-v1',
      splitOverrides: [],
      templateVersion: 'corpus-v1',
      sources: {
        wikipediaExtracts: true,
        wikipediaSections: true,
        fieldCraftExported: true,
        fieldCraftIncludedInTraining: false,
      },
      gitRevision: null,
      warnings: [],
    };
    for (const item of nextRows) {
      manifest.splits[item.split].bytes += Buffer.byteLength(renderTrainingDocument(item));
    }
    await writeFile(corpusPath, corpus);
    await writeFile(manifestPath, JSON.stringify(manifest));
  };
  await write(rows);
  return {
    index: new CorpusEvidenceIndex(corpusPath, manifestPath),
    write,
    corpusPath,
    manifestPath,
  };
}

describe('corpus evidence index', () => {
  it('separates phrase, normalized-field, and species-document matches with exact source provenance', async () => {
    const { index } = await fixture([
      row('osprey', 'Osprey', 'test', 'Hunts over open ocean by day.'),
      row('heron', 'Blue Heron', 'train', 'Open land is far from the ocean.'),
      row('finch', 'Café Finch', 'validation', 'A small bird near cafés.'),
    ]);
    const phrase = await index.search('open ocean', 0);
    expect(phrase.indexedDocuments).toBe(3);
    expect(phrase.exactPhrase.total).toBe(1);
    expect(phrase.exactPhrase.hits[0]).toMatchObject({
      code: 'osprey',
      split: 'test',
      field: 'Wikipedia extract',
    });
    expect(phrase.exactPhrase.hits[0].snippet).toContain('open ocean');
    expect(phrase.normalizedTerms.total).toBe(2);
    expect(phrase.speciesDocuments.total).toBe(0);

    const reverse = await index.search('ocean open', 0);
    expect(reverse.exactPhrase.total).toBe(0);
    expect(reverse.normalizedTerms.total).toBe(2);
    const species = await index.search('osprey', 0);
    expect(species.speciesDocuments.hits).toMatchObject([
      { code: 'osprey', field: 'Common name', split: 'test' },
    ]);
    const accent = await index.search('cafe', 0);
    expect(accent.exactPhrase.total).toBe(0);
    expect(accent.speciesDocuments.hits).toMatchObject([{ code: 'finch', split: 'validation' }]);
    const section = await index.search('coastal waters', 0);
    expect(section.exactPhrase.hits[0]).toMatchObject({ field: 'Wikipedia section: Habitat' });
    expect(section.exactPhrase.hits[0].snippet).toContain('Coastal waters');
    expect((await index.search('secret-only', 0)).normalizedTerms.total).toBe(0);
  });

  it('reports every hit count while paging and reloads when a new export changes the hash', async () => {
    const rows = Array.from({ length: 25 }, (_, number) =>
      row(`bird-${number}`, `Coastal Bird ${number}`, 'train', `Coastal bird ${number}.`),
    );
    const { index, write } = await fixture(rows);
    const first = await index.search('coastal', 0);
    expect(first.speciesDocuments.total).toBe(25);
    expect(first.speciesDocuments.hits).toHaveLength(20);
    const second = await index.search('coastal', 1);
    expect(second.speciesDocuments.total).toBe(25);
    expect(second.speciesDocuments.hits).toHaveLength(5);
    await write([row('new', 'New Bird', 'train', 'Forest only.')]);
    expect((await index.search('coastal', 0)).speciesDocuments.total).toBe(0);
    expect((await index.search('forest', 0)).exactPhrase.total).toBe(1);
  });

  it('reports the current export time when identical corpus bytes are exported again', async () => {
    const { index, manifestPath } = await fixture([row('one', 'One Bird', 'train', 'Ocean.')]);
    await index.search('ocean', 0);
    const manifest = JSON.parse(await readFile(manifestPath, 'utf8')) as CorpusManifest;
    manifest.exportedAt = '2026-10-07T12:00:00.000Z';
    await writeFile(manifestPath, JSON.stringify(manifest));
    expect((await index.search('ocean', 0)).exportedAt).toBe(manifest.exportedAt);
  });

  it('rejects malformed queries and a corpus that does not match its manifest', async () => {
    const { index, corpusPath } = await fixture([row('one', 'One Bird', 'train', 'Ocean.')]);
    await expect(index.search('', 0)).rejects.toBeInstanceOf(EvidenceQueryError);
    await expect(index.search('one', -1)).rejects.toBeInstanceOf(EvidenceQueryError);
    await writeFile(corpusPath, 'changed\n');
    await expect(index.search('one', 0)).rejects.toBeInstanceOf(EvidenceUnavailableError);
  });
});
