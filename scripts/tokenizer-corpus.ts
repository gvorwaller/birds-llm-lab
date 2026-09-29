import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import {
  parseCorpusManifest,
  parseCorpusRow,
  type CorpusManifest,
  type CorpusRow,
} from '../src/lib/data/schemas.js';
import { trainingDocumentText } from '../src/lib/tokenizer/corpus.js';

export interface LoadedCorpus {
  manifest: CorpusManifest;
  rows: CorpusRow[];
  trainingDocuments: string[];
}

export interface LoadCorpusOptions {
  corpusPath?: string;
  manifestPath?: string;
}

export async function loadCorpus(options: LoadCorpusOptions = {}): Promise<LoadedCorpus> {
  const corpusPath = resolve(options.corpusPath ?? 'data/corpus.jsonl');
  const manifestPath = resolve(options.manifestPath ?? 'data/manifest.json');
  const manifest = parseCorpusManifest(JSON.parse(await readFile(manifestPath, 'utf8')) as unknown);
  const bytes = await readFile(corpusPath);
  const actualHash = createHash('sha256').update(bytes).digest('hex');
  if (actualHash !== manifest.corpusSha256) {
    throw new Error(
      `Corpus/manifest hash mismatch: file ${actualHash}, manifest ${manifest.corpusSha256}.`,
    );
  }
  const text = bytes.toString('utf8');
  const lines = text.length === 0 ? [] : text.trimEnd().split('\n');
  const rows = lines.map((line, index) => {
    try {
      return parseCorpusRow(JSON.parse(line) as unknown);
    } catch (error) {
      throw new Error(`Invalid corpus row ${index + 1}: ${(error as Error).message}`, {
        cause: error,
      });
    }
  });
  if (rows.length !== manifest.emittedRows) {
    throw new Error(
      `Corpus row count ${rows.length} does not match manifest ${manifest.emittedRows}.`,
    );
  }
  const trainingRows = rows.filter((row) => row.split === 'train');
  if (trainingRows.length !== manifest.splits.train.documents) {
    throw new Error('Training split row count does not match the corpus manifest.');
  }
  return {
    manifest,
    rows,
    trainingDocuments: trainingRows.map(trainingDocumentText),
  };
}
