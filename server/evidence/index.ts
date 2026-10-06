import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import {
  parseCorpusManifest,
  parseCorpusRow,
  type CorpusManifest,
  type CorpusRow,
  type CorpusSplit,
} from '../../src/lib/data/schemas.js';
import type {
  EvidenceHit,
  EvidenceMatchPage,
  EvidenceSearchResponse,
  OrderCooccurrenceEvidence,
} from '../../src/lib/data/api-types.js';
import { KESTREL_EXAMPLE } from '../../src/lib/evidence/saved-example.js';

export const EVIDENCE_PAGE_SIZE = 20;
const MAX_QUERY_LENGTH = 160;
const MAX_QUERY_TERMS = 12;

type FieldKind =
  | 'common-name'
  | 'scientific-name'
  | 'order'
  | 'family'
  | 'extract'
  | 'section-title'
  | 'section-text';

interface IndexedDocument {
  readonly code: string;
  readonly name: string;
  readonly sci: string;
  readonly split: CorpusSplit;
  readonly order: string | null;
  readonly family: string | null;
}

interface IndexedField {
  readonly documentId: number;
  readonly kind: FieldKind;
  readonly label: string;
  readonly text: string;
}

interface LoadedIndex {
  readonly corpusSha256: string;
  readonly exportedAt: string;
  readonly documents: readonly IndexedDocument[];
  readonly fields: readonly IndexedField[];
  readonly postings: ReadonlyMap<string, readonly number[]>;
}

export class EvidenceUnavailableError extends Error {}
export class EvidenceQueryError extends Error {}

function normalizedTerms(text: string): string[] {
  return (
    text
      .normalize('NFKD')
      .replace(/\p{M}/gu, '')
      .toLocaleLowerCase('en-US')
      .match(/[\p{L}\p{N}]+/gu) ?? []
  );
}

function validateSearch(query: string, page: number): string[] {
  const trimmed = query.trim();
  if (trimmed.length === 0 || trimmed.length > MAX_QUERY_LENGTH) {
    throw new EvidenceQueryError(`Search text must be 1–${MAX_QUERY_LENGTH} characters.`);
  }
  if (!Number.isSafeInteger(page) || page < 0 || page > 10_000) {
    throw new EvidenceQueryError('Page must be a non-negative integer no greater than 10,000.');
  }
  const terms = [...new Set(normalizedTerms(trimmed))];
  if (terms.length === 0 || terms.length > MAX_QUERY_TERMS) {
    throw new EvidenceQueryError(`Search text needs 1–${MAX_QUERY_TERMS} terms.`);
  }
  return terms;
}

function fieldsFor(row: CorpusRow, documentId: number): IndexedField[] {
  const fields: IndexedField[] = [
    { documentId, kind: 'common-name', label: 'Common name', text: row.name },
    { documentId, kind: 'scientific-name', label: 'Scientific name', text: row.sci },
  ];
  if (row.order) fields.push({ documentId, kind: 'order', label: 'Order', text: row.order });
  if (row.family) fields.push({ documentId, kind: 'family', label: 'Family', text: row.family });
  if (row.extract)
    fields.push({ documentId, kind: 'extract', label: 'Wikipedia extract', text: row.extract });
  for (const section of row.sections) {
    fields.push({
      documentId,
      kind: 'section-title',
      label: 'Wikipedia section title',
      text: section.title,
    });
    fields.push({
      documentId,
      kind: 'section-text',
      label: `Wikipedia section: ${section.title}`,
      text: section.text,
    });
  }
  return fields;
}

function buildIndex(corpus: Buffer, manifest: CorpusManifest): LoadedIndex {
  const actualHash = createHash('sha256').update(corpus).digest('hex');
  if (actualHash !== manifest.corpusSha256) {
    throw new EvidenceUnavailableError(
      'The exported corpus does not match its manifest. Re-export before searching.',
    );
  }
  const documents: IndexedDocument[] = [];
  const fields: IndexedField[] = [];
  const postings = new Map<string, number[]>();
  const lines = corpus.toString('utf8').split('\n');
  for (const line of lines) {
    if (line.length === 0) continue;
    const row = parseCorpusRow(JSON.parse(line) as unknown);
    const documentId = documents.length;
    documents.push({
      code: row.code,
      name: row.name,
      sci: row.sci,
      split: row.split,
      order: row.order,
      family: row.family,
    });
    for (const field of fieldsFor(row, documentId)) {
      const fieldId = fields.length;
      fields.push(field);
      for (const term of new Set(normalizedTerms(field.text))) {
        const posting = postings.get(term);
        if (posting) posting.push(fieldId);
        else postings.set(term, [fieldId]);
      }
    }
  }
  if (documents.length !== manifest.emittedRows) {
    throw new EvidenceUnavailableError(
      'The exported corpus row count does not match its manifest.',
    );
  }
  return {
    corpusSha256: manifest.corpusSha256,
    exportedAt: manifest.exportedAt,
    documents,
    fields,
    postings,
  };
}

function candidateFields(index: LoadedIndex, terms: readonly string[]): number[] {
  const lists = terms.map((term) => index.postings.get(term) ?? []);
  if (lists.some((list) => list.length === 0)) return [];
  lists.sort((left, right) => left.length - right.length);
  const otherSets = lists.slice(1).map((list) => new Set(list));
  return lists[0].filter((fieldId) => otherSets.every((set) => set.has(fieldId)));
}

function matchPosition(
  field: IndexedField,
  query: string,
  firstTerm: string,
  exact: boolean,
): number {
  if (exact) return field.text.toLocaleLowerCase('en-US').indexOf(query.toLocaleLowerCase('en-US'));
  for (const match of field.text.matchAll(/[\p{L}\p{N}]+/gu)) {
    if (normalizedTerms(match[0]).includes(firstTerm)) return match.index;
  }
  return 0;
}

function snippet(text: string, position: number): string {
  const radius = 95;
  const start = Math.max(0, position - radius);
  const end = Math.min(text.length, position + radius);
  const excerpt = text.slice(start, end).replace(/\s+/g, ' ').trim();
  return `${start > 0 ? '…' : ''}${excerpt}${end < text.length ? '…' : ''}`;
}

function hit(index: LoadedIndex, field: IndexedField, position: number): EvidenceHit {
  const document = index.documents[field.documentId];
  return {
    code: document.code,
    name: document.name,
    sci: document.sci,
    split: document.split,
    field: field.label,
    snippet: snippet(field.text, position),
  };
}

function pageMatches(
  index: LoadedIndex,
  fieldIds: readonly number[],
  page: number,
  query: string,
  firstTerm: string,
  mode: 'exact' | 'terms' | 'species',
): EvidenceMatchPage {
  let total = 0;
  const hits: EvidenceHit[] = [];
  const seenSpecies = new Set<number>();
  const start = page * EVIDENCE_PAGE_SIZE;
  for (const fieldId of fieldIds) {
    const field = index.fields[fieldId];
    if (mode === 'species') {
      if (field.kind !== 'common-name' && field.kind !== 'scientific-name') continue;
      if (seenSpecies.has(field.documentId)) continue;
      seenSpecies.add(field.documentId);
    }
    const position = matchPosition(field, query, firstTerm, mode === 'exact');
    if (mode === 'exact' && position < 0) continue;
    if (total >= start && hits.length < EVIDENCE_PAGE_SIZE) hits.push(hit(index, field, position));
    total += 1;
  }
  return { total, hits };
}

export function searchIndex(
  index: LoadedIndex,
  query: string,
  page: number,
): EvidenceSearchResponse {
  const trimmed = query.trim();
  const terms = validateSearch(trimmed, page);
  const candidates = candidateFields(index, terms);
  return {
    query: trimmed,
    page,
    pageSize: EVIDENCE_PAGE_SIZE,
    corpusSha256: index.corpusSha256,
    exportedAt: index.exportedAt,
    indexedDocuments: index.documents.length,
    exactPhrase: pageMatches(index, candidates, page, trimmed, terms[0], 'exact'),
    normalizedTerms: pageMatches(index, candidates, page, trimmed, terms[0], 'terms'),
    speciesDocuments: pageMatches(index, candidates, page, trimmed, terms[0], 'species'),
  };
}

function orderCooccurrence(
  index: LoadedIndex,
  speciesCode: string,
  generatedOrder: string,
): OrderCooccurrenceEvidence {
  const target = index.documents.find((document) => document.code === speciesCode);
  if (!target?.order) {
    throw new EvidenceUnavailableError('The example species is absent from this export.');
  }
  const train = index.documents.filter((document) => document.split === 'train');
  // The fixed corpus-v1 template writes these adjacent lines for every document.
  const matching = train.filter((document) => document.order === generatedOrder);
  return {
    corpusSha256: index.corpusSha256,
    trainDocuments: train.length,
    exactTemplateSpan: `\nOrder: ${generatedOrder}\nFamily: `,
    trainSpanDocuments: matching.length,
    trainExamples: matching.slice(0, 3).map((document) => ({
      code: document.code,
      name: document.name,
      sci: document.sci,
      split: document.split,
      field: 'Order and family in training template',
      snippet: `Order: ${document.order}\nFamily: ${document.family ?? ''}`,
    })),
    target: {
      code: target.code,
      name: target.name,
      sci: target.sci,
      split: target.split,
      field: 'Order and family in exported document',
      snippet: `Order: ${target.order}\nFamily: ${target.family ?? ''}`,
      order: target.order,
      family: target.family,
    },
  };
}

export class CorpusEvidenceIndex {
  private cached: LoadedIndex | null = null;
  private loading: { hash: string; promise: Promise<LoadedIndex> } | null = null;

  constructor(
    private readonly corpusPath: string,
    private readonly manifestPath: string,
  ) {}

  async search(query: string, page: number): Promise<EvidenceSearchResponse> {
    validateSearch(query, page);
    return searchIndex(await this.load(), query, page);
  }

  async cooccurrence(
    speciesCode: string,
    generatedOrder: string,
  ): Promise<OrderCooccurrenceEvidence> {
    return orderCooccurrence(await this.load(), speciesCode, generatedOrder);
  }

  async verifiedExampleEvidence(): Promise<OrderCooccurrenceEvidence> {
    const index = await this.load();
    if (index.corpusSha256 !== KESTREL_EXAMPLE.corpusSha256) {
      throw new EvidenceUnavailableError(
        'The current corpus export does not match the verified ready-v1 example.',
      );
    }
    const evidence = orderCooccurrence(index, KESTREL_EXAMPLE.speciesCode, 'Passeriformes');
    if (evidence.target.order !== KESTREL_EXAMPLE.referenceOrder) {
      throw new EvidenceUnavailableError(
        'The example species order does not match the verified export.',
      );
    }
    return evidence;
  }

  private async load(): Promise<LoadedIndex> {
    let manifest: CorpusManifest;
    try {
      manifest = parseCorpusManifest(
        JSON.parse(await readFile(this.manifestPath, 'utf8')) as unknown,
      );
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
        throw new EvidenceUnavailableError('No exported corpus is available yet.');
      }
      throw error;
    }
    if (this.cached?.corpusSha256 !== manifest.corpusSha256) {
      if (this.loading?.hash !== manifest.corpusSha256) {
        const promise = readFile(this.corpusPath)
          .catch((error: unknown) => {
            if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
              throw new EvidenceUnavailableError(
                'The exported corpus file is missing. Re-export before searching.',
              );
            }
            throw error;
          })
          .then((corpus) => buildIndex(corpus, manifest));
        this.loading = { hash: manifest.corpusSha256, promise };
      }
      const loading = this.loading;
      try {
        this.cached = await loading.promise;
      } finally {
        if (this.loading === loading) this.loading = null;
      }
    }
    if (!this.cached) throw new EvidenceUnavailableError('No exported corpus is available yet.');
    if (this.cached.exportedAt !== manifest.exportedAt) {
      this.cached = { ...this.cached, exportedAt: manifest.exportedAt };
    }
    return this.cached;
  }
}
