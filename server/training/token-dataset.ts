import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import {
  lstat,
  mkdir,
  mkdtemp,
  open,
  readFile,
  rename,
  rm,
  writeFile,
  type FileHandle,
} from 'node:fs/promises';
import { createInterface } from 'node:readline';
import { basename, dirname, join, resolve } from 'node:path';
import {
  parseCorpusManifest,
  parseCorpusRow,
  parseTokenizerArtifact,
  serializeStableJson,
  type CorpusManifest,
  type TokenizerArtifact,
} from '../../src/lib/data/schemas';
import { SeededRandom } from '../../src/lib/math/rng';
import { encodeText } from '../../src/lib/tokenizer/bpe';
import { PAD_ID } from '../../src/lib/tokenizer/constants';
import { trainingDocumentText } from '../../src/lib/tokenizer/corpus';
import type { LanguageModelBatch } from '../../src/lib/training/batches';
import type { DataOrderState } from '../trainer-state';

type TrainingSplit = 'train' | 'validation';

interface TokenDocument {
  readonly split: TrainingSplit;
  readonly tokenOffset: number;
  readonly tokenCount: number;
}

export interface TokenDatasetIndex {
  readonly formatVersion: 1;
  readonly corpusSha256: string;
  readonly tokenizerSha256: string;
  readonly tokenCount: number;
  readonly documents: TokenDocument[];
  readonly sourceDocumentCounts: Record<'train' | 'validation' | 'test', number>;
}

export interface TokenDatasetPaths {
  readonly corpusPath?: string;
  readonly manifestPath?: string;
  readonly tokenizerPath?: string;
  readonly cacheRoot?: string;
}

interface WindowDescriptor {
  readonly tokenOffset: number;
  readonly predictionCount: number;
}

async function exists(path: string): Promise<boolean> {
  try {
    await lstat(path);
    return true;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return false;
    throw error;
  }
}

async function sha256File(path: string): Promise<string> {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) hash.update(chunk as Buffer);
  return hash.digest('hex');
}

async function writeAll(file: FileHandle, bytes: Buffer): Promise<void> {
  let offset = 0;
  while (offset < bytes.length) {
    const result = await file.write(bytes, offset, bytes.length - offset);
    if (result.bytesWritten === 0) throw new Error('Token dataset write made no progress.');
    offset += result.bytesWritten;
  }
}

function parseIndex(value: unknown): TokenDatasetIndex {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new Error('Token dataset index must be an object.');
  }
  const index = value as Partial<TokenDatasetIndex>;
  if (
    index.formatVersion !== 1 ||
    typeof index.corpusSha256 !== 'string' ||
    typeof index.tokenizerSha256 !== 'string' ||
    !Number.isSafeInteger(index.tokenCount) ||
    (index.tokenCount ?? -1) < 0 ||
    !Array.isArray(index.documents) ||
    typeof index.sourceDocumentCounts !== 'object' ||
    index.sourceDocumentCounts === null
  ) {
    throw new Error('Token dataset index metadata is invalid.');
  }
  let coveredTokens = 0;
  for (const [position, document] of index.documents.entries()) {
    if (
      typeof document !== 'object' ||
      document === null ||
      (document.split !== 'train' && document.split !== 'validation') ||
      !Number.isSafeInteger(document.tokenOffset) ||
      !Number.isSafeInteger(document.tokenCount) ||
      document.tokenOffset !== coveredTokens ||
      document.tokenCount < 2
    ) {
      throw new Error(`Token dataset document ${position} is invalid or non-contiguous.`);
    }
    coveredTokens += document.tokenCount;
  }
  if (coveredTokens !== index.tokenCount) {
    throw new Error('Token dataset token count does not match its document index.');
  }
  for (const split of ['train', 'validation', 'test'] as const) {
    const count = index.sourceDocumentCounts[split];
    if (!Number.isSafeInteger(count) || count < 0) {
      throw new Error(`Token dataset ${split} source count is invalid.`);
    }
  }
  return index as TokenDatasetIndex;
}

async function loadInputs(paths: Required<TokenDatasetPaths>): Promise<{
  manifest: CorpusManifest;
  tokenizer: TokenizerArtifact;
  tokenizerSha256: string;
}> {
  const [manifestText, tokenizerBytes, corpusSha256] = await Promise.all([
    readFile(paths.manifestPath, 'utf8'),
    readFile(paths.tokenizerPath),
    sha256File(paths.corpusPath),
  ]);
  const manifest = parseCorpusManifest(JSON.parse(manifestText) as unknown);
  if (manifest.corpusSha256 !== corpusSha256) {
    throw new Error(
      `Corpus/manifest hash mismatch: file ${corpusSha256}, manifest ${manifest.corpusSha256}.`,
    );
  }
  const tokenizer = parseTokenizerArtifact(JSON.parse(tokenizerBytes.toString('utf8')) as unknown);
  if (tokenizer.corpusSha256 !== corpusSha256) {
    throw new Error('Tokenizer was not trained from the selected corpus.');
  }
  return {
    manifest,
    tokenizer,
    tokenizerSha256: createHash('sha256').update(tokenizerBytes).digest('hex'),
  };
}

function resolvedPaths(paths: TokenDatasetPaths): Required<TokenDatasetPaths> {
  return {
    corpusPath: resolve(paths.corpusPath ?? 'data/corpus.jsonl'),
    manifestPath: resolve(paths.manifestPath ?? 'data/manifest.json'),
    tokenizerPath: resolve(paths.tokenizerPath ?? 'src/assets/tokenizer-1024.json'),
    cacheRoot: resolve(paths.cacheRoot ?? '.cache/training-dataset-v1'),
  };
}

async function buildCache(
  directory: string,
  paths: Required<TokenDatasetPaths>,
  manifest: CorpusManifest,
  tokenizer: TokenizerArtifact,
  tokenizerSha256: string,
): Promise<void> {
  const parent = dirname(directory);
  await mkdir(parent, { recursive: true, mode: 0o700 });
  const temporary = await mkdtemp(join(parent, `.${basename(directory)}.tmp-`));
  const tokenFile = await open(join(temporary, 'tokens.bin'), 'wx', 0o600);
  const documents: TokenDocument[] = [];
  const counts = { train: 0, validation: 0, test: 0 };
  let tokenCount = 0;
  let rowNumber = 0;
  try {
    const lines = createInterface({
      input: createReadStream(paths.corpusPath, { encoding: 'utf8' }),
      crlfDelay: Infinity,
    });
    for await (const line of lines) {
      rowNumber += 1;
      let row;
      try {
        row = parseCorpusRow(JSON.parse(line) as unknown);
      } catch (error) {
        throw new Error(`Invalid corpus row ${rowNumber}: ${(error as Error).message}`, {
          cause: error,
        });
      }
      counts[row.split] += 1;
      if (row.split === 'test') continue;
      const tokens = encodeText(trainingDocumentText(row), tokenizer, { bos: true, eos: true });
      if (tokens.some((token) => token > 0xffff)) {
        throw new Error('Tokenizer ID exceeds the uint16 token-cache format.');
      }
      const bytes = Buffer.allocUnsafe(tokens.length * Uint16Array.BYTES_PER_ELEMENT);
      tokens.forEach((token, index) => bytes.writeUInt16LE(token, index * 2));
      await writeAll(tokenFile, bytes);
      documents.push({ split: row.split, tokenOffset: tokenCount, tokenCount: tokens.length });
      tokenCount += tokens.length;
    }
    await tokenFile.close();
    if (rowNumber !== manifest.emittedRows) {
      throw new Error(
        `Corpus contains ${rowNumber} rows; manifest declares ${manifest.emittedRows}.`,
      );
    }
    for (const split of ['train', 'validation', 'test'] as const) {
      if (counts[split] !== manifest.splits[split].documents) {
        throw new Error(`${split} row count does not match the corpus manifest.`);
      }
    }
    const index: TokenDatasetIndex = {
      formatVersion: 1,
      corpusSha256: manifest.corpusSha256,
      tokenizerSha256,
      tokenCount,
      documents,
      sourceDocumentCounts: counts,
    };
    await writeFile(join(temporary, 'index.json'), serializeStableJson(index), {
      flag: 'wx',
      mode: 0o600,
    });
    await rename(temporary, directory);
  } catch (error) {
    await tokenFile.close().catch(() => undefined);
    await rm(temporary, { recursive: true, force: true });
    throw error;
  }
}

export class StreamedTokenDataset {
  private constructor(
    readonly index: TokenDatasetIndex,
    private readonly tokenFile: FileHandle,
  ) {}

  static async open(pathsInput: TokenDatasetPaths = {}): Promise<StreamedTokenDataset> {
    const paths = resolvedPaths(pathsInput);
    const { manifest, tokenizer, tokenizerSha256 } = await loadInputs(paths);
    const directory = join(paths.cacheRoot, `${manifest.corpusSha256}-${tokenizerSha256}`);
    if (!(await exists(directory))) {
      await buildCache(directory, paths, manifest, tokenizer, tokenizerSha256);
    }
    const index = parseIndex(JSON.parse(await readFile(join(directory, 'index.json'), 'utf8')));
    if (index.corpusSha256 !== manifest.corpusSha256 || index.tokenizerSha256 !== tokenizerSha256) {
      throw new Error('Token dataset cache identity does not match its inputs.');
    }
    const tokenPath = join(directory, 'tokens.bin');
    const stats = await lstat(tokenPath);
    if (stats.size !== index.tokenCount * Uint16Array.BYTES_PER_ELEMENT) {
      throw new Error('Token dataset binary length does not match its index.');
    }
    return new StreamedTokenDataset(index, await open(tokenPath, 'r'));
  }

  windows(split: TrainingSplit, contextLength: number): WindowDescriptor[] {
    if (!Number.isInteger(contextLength) || contextLength <= 0) {
      throw new Error('Context length must be a positive integer.');
    }
    const windows: WindowDescriptor[] = [];
    for (const document of this.index.documents) {
      if (document.split !== split) continue;
      for (let start = 0; start < document.tokenCount - 1; start += contextLength) {
        windows.push({
          tokenOffset: document.tokenOffset + start,
          predictionCount: Math.min(contextLength, document.tokenCount - 1 - start),
        });
      }
    }
    return windows;
  }

  async readBatch(
    windows: readonly WindowDescriptor[],
    contextLength: number,
  ): Promise<LanguageModelBatch> {
    const inputIds = Array<number>(windows.length * contextLength).fill(PAD_ID);
    const targetIds = Array<number>(windows.length * contextLength).fill(PAD_ID);
    let predictionCount = 0;
    for (const [batchIndex, window] of windows.entries()) {
      const bytes = Buffer.allocUnsafe((window.predictionCount + 1) * 2);
      const result = await this.tokenFile.read(bytes, 0, bytes.length, window.tokenOffset * 2);
      if (result.bytesRead !== bytes.length) throw new Error('Token dataset binary is truncated.');
      for (let offset = 0; offset < window.predictionCount; offset += 1) {
        inputIds[batchIndex * contextLength + offset] = bytes.readUInt16LE(offset * 2);
        targetIds[batchIndex * contextLength + offset] = bytes.readUInt16LE((offset + 1) * 2);
      }
      predictionCount += window.predictionCount;
    }
    return {
      inputIds,
      targetIds,
      batchSize: windows.length,
      sequenceLength: contextLength,
      predictionCount,
    };
  }

  async close(): Promise<void> {
    await this.tokenFile.close();
  }
}

function shuffledIndices(length: number, seed: number, epoch: number): number[] {
  const random = new SeededRandom((seed ^ Math.imul(epoch + 1, 0x9e37_79b1)) >>> 0);
  const indices = Array.from({ length }, (_, index) => index);
  for (let index = indices.length - 1; index > 0; index -= 1) {
    const selected = Math.floor(random.uniform() * (index + 1));
    [indices[index], indices[selected]] = [indices[selected], indices[index]];
  }
  return indices;
}

export class DeterministicBatchStream {
  private stateValue: DataOrderState;

  constructor(
    private readonly dataset: StreamedTokenDataset,
    private readonly windowsValue: readonly WindowDescriptor[],
    private readonly batchSize: number,
    private readonly contextLength: number,
    private readonly seed: number,
    state: DataOrderState = { epoch: 0, cursor: 0 },
  ) {
    if (windowsValue.length === 0) throw new Error('Batch stream requires at least one window.');
    if (!Number.isInteger(batchSize) || batchSize <= 0) {
      throw new Error('Batch size must be a positive integer.');
    }
    if (
      !Number.isSafeInteger(state.epoch) ||
      state.epoch < 0 ||
      !Number.isSafeInteger(state.cursor) ||
      state.cursor < 0 ||
      state.cursor > windowsValue.length
    ) {
      throw new Error('Saved data-order state is outside the dataset.');
    }
    this.stateValue = { ...state };
  }

  snapshot(): DataOrderState {
    return { ...this.stateValue };
  }

  async next(): Promise<LanguageModelBatch> {
    if (this.stateValue.cursor === this.windowsValue.length) {
      this.stateValue = { epoch: this.stateValue.epoch + 1, cursor: 0 };
    }
    const order = shuffledIndices(this.windowsValue.length, this.seed, this.stateValue.epoch);
    const end = Math.min(this.stateValue.cursor + this.batchSize, order.length);
    const selected = order
      .slice(this.stateValue.cursor, end)
      .map((index) => this.windowsValue[index]);
    this.stateValue = { epoch: this.stateValue.epoch, cursor: end };
    return this.dataset.readBatch(selected, this.contextLength);
  }
}
