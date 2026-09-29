export const ARTIFACT_FORMAT_VERSION = 1 as const;

export type CorpusSplit = 'train' | 'validation' | 'test';

export interface CorpusSection {
  title: string;
  text: string;
}

export interface CorpusRow {
  code: string;
  name: string;
  sci: string;
  order: string | null;
  family: string | null;
  extract: string | null;
  sections: CorpusSection[];
  field_craft: string | null;
  tags: string[];
  split: CorpusSplit;
}

export interface SplitSummary {
  documents: number;
  bytes: number;
}

export interface CorpusManifest {
  formatVersion: 1;
  exportedAt: string;
  database: {
    host: '127.0.0.1';
    port: 15436;
    name: 'birds_test';
  };
  sourceRows: number;
  emittedRows: number;
  skippedRows: number;
  skippedReasons: Record<string, number>;
  malformedSectionsSkipped: number;
  splits: Record<CorpusSplit, SplitSummary>;
  sourceBytes: {
    wikipediaExtracts: number;
    wikipediaSections: number;
    fieldCraft: number;
  };
  corpusBytes: number;
  corpusSha256: string;
  splitAlgorithm: 'sha256-bucket';
  splitVersion: 'split-v1';
  splitOverrides: string[];
  templateVersion: 'corpus-v1';
  sources: {
    wikipediaExtracts: true;
    wikipediaSections: true;
    fieldCraftExported: true;
    fieldCraftIncludedInTraining: false;
  };
  gitRevision: string | null;
  warnings: string[];
}

export interface TokenizerArtifact {
  formatVersion: 1;
  specialIds: {
    bos: 256;
    eos: 257;
    pad: 258;
  };
  targetSize: number;
  merges: Array<{
    left: number;
    right: number;
    id: number;
    trainingCount: number;
  }>;
  tokens: Array<{
    id: number;
    bytes: number[];
    display: string;
  }>;
  corpusSha256: string;
  trainer: {
    algorithmVersion: 'bpe-v1';
    split: 'train';
    templateVersion: 'corpus-v1';
    documentCount: number;
    byteCount: number;
  };
}

export interface ModelConfig {
  formatVersion: 1;
  vocabSize: number;
  contextLength: number;
  dModel: number;
  nLayers: number;
  nHeads: number;
  dHead: number;
  dMlp: number;
  tiedEmbeddings: boolean;
  useBias: boolean;
  layerNormEpsilon: number;
  gelu: 'tanh-approximation';
}

export interface OptimizerConfig {
  name: 'adamw';
  beta1: number;
  beta2: number;
  epsilon: number;
  weightDecay: number;
  gradientClipNorm: number;
  learningRate: number;
  warmupSteps: number;
  totalSteps: number;
}

export interface CheckpointConfig {
  formatVersion: 1;
  model: ModelConfig;
  optimizer: OptimizerConfig;
  seed: number;
  corpusSha256: string;
  tokenizerSha256: string;
  trainingStep: number;
  sourceGitRevision: string | null;
}

export interface WeightIndexEntry {
  name: string;
  shape: number[];
  elementOffset: number;
  elementCount: number;
  byteOffset: number;
  byteLength: number;
}

export interface WeightIndexArtifact {
  formatVersion: 1;
  byteLength: number;
  entries: WeightIndexEntry[];
}

export interface TrainingLogEntry {
  step: number;
  split: 'train' | 'validation';
  meanLoss: number;
  perplexity: number;
  learningRate: number;
  gradientNorm: number;
  elapsedMs: number;
  sample: string | null;
}

export interface TrainingLogArtifact {
  formatVersion: 1;
  entries: TrainingLogEntry[];
}

export class ArtifactValidationError extends Error {
  constructor(
    readonly path: string,
    message: string,
  ) {
    super(`${path}: ${message}`);
    this.name = 'ArtifactValidationError';
  }
}

type JsonObject = Record<string, unknown>;

function objectAt(value: unknown, path: string): JsonObject {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new ArtifactValidationError(path, 'expected an object');
  }
  return value as JsonObject;
}

function exactKeys(value: JsonObject, keys: readonly string[], path: string): void {
  const expected = new Set(keys);
  for (const key of Object.keys(value)) {
    if (!expected.has(key)) {
      throw new ArtifactValidationError(`${path}.${key}`, 'unexpected property');
    }
  }
  for (const key of keys) {
    if (!(key in value)) {
      throw new ArtifactValidationError(`${path}.${key}`, 'missing property');
    }
  }
}

function stringAt(value: unknown, path: string, allowEmpty = false): string {
  if (typeof value !== 'string' || (!allowEmpty && value.length === 0)) {
    throw new ArtifactValidationError(
      path,
      allowEmpty ? 'expected a string' : 'expected a non-empty string',
    );
  }
  return value;
}

function nullableStringAt(value: unknown, path: string): string | null {
  return value === null ? null : stringAt(value, path, true);
}

function booleanAt(value: unknown, path: string): boolean {
  if (typeof value !== 'boolean') {
    throw new ArtifactValidationError(path, 'expected a boolean');
  }
  return value;
}

function finiteNumberAt(value: unknown, path: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new ArtifactValidationError(path, 'expected a finite number');
  }
  return value;
}

function integerAt(value: unknown, path: string, minimum = 0): number {
  const number = finiteNumberAt(value, path);
  if (!Number.isInteger(number) || number < minimum) {
    throw new ArtifactValidationError(path, `expected an integer >= ${minimum}`);
  }
  return number;
}

function literalAt<T extends string | number | boolean>(
  value: unknown,
  literal: T,
  path: string,
): T {
  if (value !== literal) {
    throw new ArtifactValidationError(path, `expected ${JSON.stringify(literal)}`);
  }
  return literal;
}

function enumAt<const T extends readonly string[]>(
  value: unknown,
  values: T,
  path: string,
): T[number] {
  if (typeof value !== 'string' || !values.includes(value)) {
    throw new ArtifactValidationError(path, `expected one of ${values.join(', ')}`);
  }
  return value as T[number];
}

function arrayAt<T>(
  value: unknown,
  path: string,
  parseItem: (item: unknown, itemPath: string) => T,
): T[] {
  if (!Array.isArray(value)) {
    throw new ArtifactValidationError(path, 'expected an array');
  }
  return value.map((item, index) => parseItem(item, `${path}[${index}]`));
}

function sha256At(value: unknown, path: string): string {
  const hash = stringAt(value, path);
  if (!/^[a-f0-9]{64}$/.test(hash)) {
    throw new ArtifactValidationError(path, 'expected a lowercase SHA-256 hex digest');
  }
  return hash;
}

function parseCorpusSection(value: unknown, path: string): CorpusSection {
  const section = objectAt(value, path);
  exactKeys(section, ['title', 'text'], path);
  return {
    title: stringAt(section.title, `${path}.title`),
    text: stringAt(section.text, `${path}.text`),
  };
}

export function parseCorpusRow(value: unknown): CorpusRow {
  const row = objectAt(value, '$');
  exactKeys(
    row,
    [
      'code',
      'name',
      'sci',
      'order',
      'family',
      'extract',
      'sections',
      'field_craft',
      'tags',
      'split',
    ],
    '$',
  );
  return {
    code: stringAt(row.code, '$.code'),
    name: stringAt(row.name, '$.name'),
    sci: stringAt(row.sci, '$.sci'),
    order: nullableStringAt(row.order, '$.order'),
    family: nullableStringAt(row.family, '$.family'),
    extract: nullableStringAt(row.extract, '$.extract'),
    sections: arrayAt(row.sections, '$.sections', parseCorpusSection),
    field_craft: nullableStringAt(row.field_craft, '$.field_craft'),
    tags: arrayAt(row.tags, '$.tags', (tag, path) => stringAt(tag, path)),
    split: enumAt(row.split, ['train', 'validation', 'test'] as const, '$.split'),
  };
}

function parseSplitSummary(value: unknown, path: string): SplitSummary {
  const summary = objectAt(value, path);
  exactKeys(summary, ['documents', 'bytes'], path);
  return {
    documents: integerAt(summary.documents, `${path}.documents`),
    bytes: integerAt(summary.bytes, `${path}.bytes`),
  };
}

function parseCountRecord(value: unknown, path: string): Record<string, number> {
  const input = objectAt(value, path);
  const output: Record<string, number> = {};
  for (const [key, count] of Object.entries(input)) {
    output[key] = integerAt(count, `${path}.${key}`);
  }
  return output;
}

export function parseCorpusManifest(value: unknown): CorpusManifest {
  const manifest = objectAt(value, '$');
  exactKeys(
    manifest,
    [
      'formatVersion',
      'exportedAt',
      'database',
      'sourceRows',
      'emittedRows',
      'skippedRows',
      'skippedReasons',
      'malformedSectionsSkipped',
      'splits',
      'sourceBytes',
      'corpusBytes',
      'corpusSha256',
      'splitAlgorithm',
      'splitVersion',
      'splitOverrides',
      'templateVersion',
      'sources',
      'gitRevision',
      'warnings',
    ],
    '$',
  );

  const database = objectAt(manifest.database, '$.database');
  exactKeys(database, ['host', 'port', 'name'], '$.database');
  const splits = objectAt(manifest.splits, '$.splits');
  exactKeys(splits, ['train', 'validation', 'test'], '$.splits');
  const sourceBytes = objectAt(manifest.sourceBytes, '$.sourceBytes');
  exactKeys(sourceBytes, ['wikipediaExtracts', 'wikipediaSections', 'fieldCraft'], '$.sourceBytes');
  const sources = objectAt(manifest.sources, '$.sources');
  exactKeys(
    sources,
    [
      'wikipediaExtracts',
      'wikipediaSections',
      'fieldCraftExported',
      'fieldCraftIncludedInTraining',
    ],
    '$.sources',
  );
  const exportedAt = stringAt(manifest.exportedAt, '$.exportedAt');
  if (Number.isNaN(Date.parse(exportedAt))) {
    throw new ArtifactValidationError('$.exportedAt', 'expected an ISO timestamp');
  }

  return {
    formatVersion: literalAt(manifest.formatVersion, 1, '$.formatVersion'),
    exportedAt,
    database: {
      host: literalAt(database.host, '127.0.0.1', '$.database.host'),
      port: literalAt(database.port, 15436, '$.database.port'),
      name: literalAt(database.name, 'birds_test', '$.database.name'),
    },
    sourceRows: integerAt(manifest.sourceRows, '$.sourceRows'),
    emittedRows: integerAt(manifest.emittedRows, '$.emittedRows'),
    skippedRows: integerAt(manifest.skippedRows, '$.skippedRows'),
    skippedReasons: parseCountRecord(manifest.skippedReasons, '$.skippedReasons'),
    malformedSectionsSkipped: integerAt(
      manifest.malformedSectionsSkipped,
      '$.malformedSectionsSkipped',
    ),
    splits: {
      train: parseSplitSummary(splits.train, '$.splits.train'),
      validation: parseSplitSummary(splits.validation, '$.splits.validation'),
      test: parseSplitSummary(splits.test, '$.splits.test'),
    },
    sourceBytes: {
      wikipediaExtracts: integerAt(
        sourceBytes.wikipediaExtracts,
        '$.sourceBytes.wikipediaExtracts',
      ),
      wikipediaSections: integerAt(
        sourceBytes.wikipediaSections,
        '$.sourceBytes.wikipediaSections',
      ),
      fieldCraft: integerAt(sourceBytes.fieldCraft, '$.sourceBytes.fieldCraft'),
    },
    corpusBytes: integerAt(manifest.corpusBytes, '$.corpusBytes'),
    corpusSha256: sha256At(manifest.corpusSha256, '$.corpusSha256'),
    splitAlgorithm: literalAt(manifest.splitAlgorithm, 'sha256-bucket', '$.splitAlgorithm'),
    splitVersion: literalAt(manifest.splitVersion, 'split-v1', '$.splitVersion'),
    splitOverrides: arrayAt(manifest.splitOverrides, '$.splitOverrides', (item, path) =>
      stringAt(item, path),
    ),
    templateVersion: literalAt(manifest.templateVersion, 'corpus-v1', '$.templateVersion'),
    sources: {
      wikipediaExtracts: literalAt(sources.wikipediaExtracts, true, '$.sources.wikipediaExtracts'),
      wikipediaSections: literalAt(sources.wikipediaSections, true, '$.sources.wikipediaSections'),
      fieldCraftExported: literalAt(
        sources.fieldCraftExported,
        true,
        '$.sources.fieldCraftExported',
      ),
      fieldCraftIncludedInTraining: literalAt(
        sources.fieldCraftIncludedInTraining,
        false,
        '$.sources.fieldCraftIncludedInTraining',
      ),
    },
    gitRevision:
      manifest.gitRevision === null ? null : stringAt(manifest.gitRevision, '$.gitRevision'),
    warnings: arrayAt(manifest.warnings, '$.warnings', (item, path) => stringAt(item, path, true)),
  };
}

export function parseTokenizerArtifact(value: unknown): TokenizerArtifact {
  const artifact = objectAt(value, '$');
  exactKeys(
    artifact,
    ['formatVersion', 'specialIds', 'targetSize', 'merges', 'tokens', 'corpusSha256', 'trainer'],
    '$',
  );
  const specialIds = objectAt(artifact.specialIds, '$.specialIds');
  exactKeys(specialIds, ['bos', 'eos', 'pad'], '$.specialIds');
  const trainer = objectAt(artifact.trainer, '$.trainer');
  exactKeys(
    trainer,
    ['algorithmVersion', 'split', 'templateVersion', 'documentCount', 'byteCount'],
    '$.trainer',
  );

  const parsed: TokenizerArtifact = {
    formatVersion: literalAt(artifact.formatVersion, 1, '$.formatVersion'),
    specialIds: {
      bos: literalAt(specialIds.bos, 256, '$.specialIds.bos'),
      eos: literalAt(specialIds.eos, 257, '$.specialIds.eos'),
      pad: literalAt(specialIds.pad, 258, '$.specialIds.pad'),
    },
    targetSize: integerAt(artifact.targetSize, '$.targetSize', 259),
    merges: arrayAt(artifact.merges, '$.merges', (item, path) => {
      const merge = objectAt(item, path);
      exactKeys(merge, ['left', 'right', 'id', 'trainingCount'], path);
      return {
        left: integerAt(merge.left, `${path}.left`),
        right: integerAt(merge.right, `${path}.right`),
        id: integerAt(merge.id, `${path}.id`, 259),
        trainingCount: integerAt(merge.trainingCount, `${path}.trainingCount`, 1),
      };
    }),
    tokens: arrayAt(artifact.tokens, '$.tokens', (item, path) => {
      const token = objectAt(item, path);
      exactKeys(token, ['id', 'bytes', 'display'], path);
      return {
        id: integerAt(token.id, `${path}.id`),
        bytes: arrayAt(token.bytes, `${path}.bytes`, (byte, bytePath) => {
          const parsed = integerAt(byte, bytePath);
          if (parsed > 255)
            throw new ArtifactValidationError(bytePath, 'expected a byte from 0 to 255');
          return parsed;
        }),
        display: stringAt(token.display, `${path}.display`, true),
      };
    }),
    corpusSha256: sha256At(artifact.corpusSha256, '$.corpusSha256'),
    trainer: {
      algorithmVersion: literalAt(trainer.algorithmVersion, 'bpe-v1', '$.trainer.algorithmVersion'),
      split: literalAt(trainer.split, 'train', '$.trainer.split'),
      templateVersion: literalAt(trainer.templateVersion, 'corpus-v1', '$.trainer.templateVersion'),
      documentCount: integerAt(trainer.documentCount, '$.trainer.documentCount'),
      byteCount: integerAt(trainer.byteCount, '$.trainer.byteCount'),
    },
  };

  if (parsed.tokens.length !== parsed.targetSize) {
    throw new ArtifactValidationError('$.tokens', 'length must equal targetSize');
  }
  if (parsed.merges.length !== parsed.targetSize - 259) {
    throw new ArtifactValidationError(
      '$.merges',
      'length must equal targetSize minus 259 base and special tokens',
    );
  }
  for (const [id, token] of parsed.tokens.entries()) {
    if (token.id !== id) {
      throw new ArtifactValidationError(`$.tokens[${id}].id`, `expected sequential id ${id}`);
    }
    if (id < 256 && (token.bytes.length !== 1 || token.bytes[0] !== id)) {
      throw new ArtifactValidationError(
        `$.tokens[${id}].bytes`,
        'byte-token bytes must contain exactly its id',
      );
    }
    if (id >= 256 && id <= 258 && token.bytes.length !== 0) {
      throw new ArtifactValidationError(
        `$.tokens[${id}].bytes`,
        'special tokens must not contain ordinary bytes',
      );
    }
  }
  for (const [rank, merge] of parsed.merges.entries()) {
    const expectedId = 259 + rank;
    if (merge.id !== expectedId) {
      throw new ArtifactValidationError(`$.merges[${rank}].id`, `expected ${expectedId}`);
    }
    if (
      merge.left >= merge.id ||
      merge.right >= merge.id ||
      (merge.left >= 256 && merge.left <= 258) ||
      (merge.right >= 256 && merge.right <= 258)
    ) {
      throw new ArtifactValidationError(
        `$.merges[${rank}]`,
        'operands must be earlier ordinary or learned tokens',
      );
    }
    const expectedBytes = [...parsed.tokens[merge.left].bytes, ...parsed.tokens[merge.right].bytes];
    if (
      expectedBytes.length !== parsed.tokens[merge.id].bytes.length ||
      expectedBytes.some((byte, index) => byte !== parsed.tokens[merge.id].bytes[index])
    ) {
      throw new ArtifactValidationError(
        `$.tokens[${merge.id}].bytes`,
        'learned-token bytes must concatenate its merge operands',
      );
    }
  }
  return parsed;
}

export function parseModelConfig(value: unknown): ModelConfig {
  const config = objectAt(value, '$');
  const keys = [
    'formatVersion',
    'vocabSize',
    'contextLength',
    'dModel',
    'nLayers',
    'nHeads',
    'dHead',
    'dMlp',
    'tiedEmbeddings',
    'useBias',
    'layerNormEpsilon',
    'gelu',
  ] as const;
  exactKeys(config, keys, '$');
  const nHeads = integerAt(config.nHeads, '$.nHeads', 1);
  const dHead = integerAt(config.dHead, '$.dHead', 1);
  const dModel = integerAt(config.dModel, '$.dModel', 1);
  if (nHeads * dHead !== dModel) {
    throw new ArtifactValidationError('$', 'nHeads * dHead must equal dModel');
  }
  return {
    formatVersion: literalAt(config.formatVersion, 1, '$.formatVersion'),
    vocabSize: integerAt(config.vocabSize, '$.vocabSize', 259),
    contextLength: integerAt(config.contextLength, '$.contextLength', 1),
    dModel,
    nLayers: integerAt(config.nLayers, '$.nLayers', 1),
    nHeads,
    dHead,
    dMlp: integerAt(config.dMlp, '$.dMlp', 1),
    tiedEmbeddings: booleanAt(config.tiedEmbeddings, '$.tiedEmbeddings'),
    useBias: booleanAt(config.useBias, '$.useBias'),
    layerNormEpsilon: finiteNumberAt(config.layerNormEpsilon, '$.layerNormEpsilon'),
    gelu: literalAt(config.gelu, 'tanh-approximation', '$.gelu'),
  };
}

function probabilityAt(value: unknown, path: string): number {
  const number = finiteNumberAt(value, path);
  if (!(number > 0 && number < 1)) {
    throw new ArtifactValidationError(path, 'expected a number strictly between 0 and 1');
  }
  return number;
}

function positiveNumberAt(value: unknown, path: string): number {
  const number = finiteNumberAt(value, path);
  if (!(number > 0)) throw new ArtifactValidationError(path, 'expected a positive number');
  return number;
}

function nonNegativeNumberAt(value: unknown, path: string): number {
  const number = finiteNumberAt(value, path);
  if (number < 0) throw new ArtifactValidationError(path, 'expected a non-negative number');
  return number;
}

export function parseCheckpointConfig(value: unknown): CheckpointConfig {
  const checkpoint = objectAt(value, '$');
  exactKeys(
    checkpoint,
    [
      'formatVersion',
      'model',
      'optimizer',
      'seed',
      'corpusSha256',
      'tokenizerSha256',
      'trainingStep',
      'sourceGitRevision',
    ],
    '$',
  );
  const optimizer = objectAt(checkpoint.optimizer, '$.optimizer');
  exactKeys(
    optimizer,
    [
      'name',
      'beta1',
      'beta2',
      'epsilon',
      'weightDecay',
      'gradientClipNorm',
      'learningRate',
      'warmupSteps',
      'totalSteps',
    ],
    '$.optimizer',
  );
  const warmupSteps = integerAt(optimizer.warmupSteps, '$.optimizer.warmupSteps');
  const totalSteps = integerAt(optimizer.totalSteps, '$.optimizer.totalSteps', 1);
  if (warmupSteps >= totalSteps) {
    throw new ArtifactValidationError('$.optimizer', 'warmupSteps must be smaller than totalSteps');
  }
  const trainingStep = integerAt(checkpoint.trainingStep, '$.trainingStep');
  if (trainingStep > totalSteps) {
    throw new ArtifactValidationError('$.trainingStep', 'must not exceed optimizer totalSteps');
  }
  const seed = integerAt(checkpoint.seed, '$.seed');
  if (seed > 0xffff_ffff) {
    throw new ArtifactValidationError('$.seed', 'expected an unsigned 32-bit integer');
  }
  const sourceGitRevision = nullableStringAt(checkpoint.sourceGitRevision, '$.sourceGitRevision');
  if (sourceGitRevision !== null && !/^[a-f0-9]{40}$/.test(sourceGitRevision)) {
    throw new ArtifactValidationError(
      '$.sourceGitRevision',
      'expected a lowercase 40-character git revision or null',
    );
  }
  const weightDecay = nonNegativeNumberAt(optimizer.weightDecay, '$.optimizer.weightDecay');
  return {
    formatVersion: literalAt(checkpoint.formatVersion, 1, '$.formatVersion'),
    model: parseModelConfig(checkpoint.model),
    optimizer: {
      name: literalAt(optimizer.name, 'adamw', '$.optimizer.name'),
      beta1: probabilityAt(optimizer.beta1, '$.optimizer.beta1'),
      beta2: probabilityAt(optimizer.beta2, '$.optimizer.beta2'),
      epsilon: positiveNumberAt(optimizer.epsilon, '$.optimizer.epsilon'),
      weightDecay,
      gradientClipNorm: positiveNumberAt(
        optimizer.gradientClipNorm,
        '$.optimizer.gradientClipNorm',
      ),
      learningRate: positiveNumberAt(optimizer.learningRate, '$.optimizer.learningRate'),
      warmupSteps,
      totalSteps,
    },
    seed,
    corpusSha256: sha256At(checkpoint.corpusSha256, '$.corpusSha256'),
    tokenizerSha256: sha256At(checkpoint.tokenizerSha256, '$.tokenizerSha256'),
    trainingStep,
    sourceGitRevision,
  };
}

export function parseWeightIndex(value: unknown): WeightIndexArtifact {
  const index = objectAt(value, '$');
  exactKeys(index, ['formatVersion', 'byteLength', 'entries'], '$');
  return {
    formatVersion: literalAt(index.formatVersion, 1, '$.formatVersion'),
    byteLength: integerAt(index.byteLength, '$.byteLength'),
    entries: arrayAt(index.entries, '$.entries', (item, path) => {
      const entry = objectAt(item, path);
      exactKeys(
        entry,
        ['name', 'shape', 'elementOffset', 'elementCount', 'byteOffset', 'byteLength'],
        path,
      );
      return {
        name: stringAt(entry.name, `${path}.name`),
        shape: arrayAt(entry.shape, `${path}.shape`, (dimension, dimensionPath) =>
          integerAt(dimension, dimensionPath, 1),
        ),
        elementOffset: integerAt(entry.elementOffset, `${path}.elementOffset`),
        elementCount: integerAt(entry.elementCount, `${path}.elementCount`),
        byteOffset: integerAt(entry.byteOffset, `${path}.byteOffset`),
        byteLength: integerAt(entry.byteLength, `${path}.byteLength`),
      };
    }),
  };
}

export function parseTrainingLog(value: unknown): TrainingLogArtifact {
  const log = objectAt(value, '$');
  exactKeys(log, ['formatVersion', 'entries'], '$');
  return {
    formatVersion: literalAt(log.formatVersion, 1, '$.formatVersion'),
    entries: arrayAt(log.entries, '$.entries', (item, path) => {
      const entry = objectAt(item, path);
      exactKeys(
        entry,
        [
          'step',
          'split',
          'meanLoss',
          'perplexity',
          'learningRate',
          'gradientNorm',
          'elapsedMs',
          'sample',
        ],
        path,
      );
      return {
        step: integerAt(entry.step, `${path}.step`),
        split: enumAt(entry.split, ['train', 'validation'] as const, `${path}.split`),
        meanLoss: nonNegativeNumberAt(entry.meanLoss, `${path}.meanLoss`),
        perplexity: positiveNumberAt(entry.perplexity, `${path}.perplexity`),
        learningRate: nonNegativeNumberAt(entry.learningRate, `${path}.learningRate`),
        gradientNorm: nonNegativeNumberAt(entry.gradientNorm, `${path}.gradientNorm`),
        elapsedMs: integerAt(entry.elapsedMs, `${path}.elapsedMs`),
        sample: nullableStringAt(entry.sample, `${path}.sample`),
      };
    }),
  };
}

function sortedJsonValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(sortedJsonValue);
  if (typeof value !== 'object' || value === null) return value;
  return Object.fromEntries(
    Object.entries(value as JsonObject)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, item]) => [key, sortedJsonValue(item)]),
  );
}

export function serializeStableJson(value: unknown): string {
  return `${JSON.stringify(sortedJsonValue(value), null, 2)}\n`;
}

export function serializeCorpusRow(row: CorpusRow): string {
  const validated = parseCorpusRow(row);
  return JSON.stringify({
    code: validated.code,
    name: validated.name,
    sci: validated.sci,
    order: validated.order,
    family: validated.family,
    extract: validated.extract,
    sections: validated.sections.map(({ title, text }) => ({ title, text })),
    field_craft: validated.field_craft,
    tags: validated.tags,
    split: validated.split,
  });
}
