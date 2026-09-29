import {
  ARTIFACT_FORMAT_VERSION,
  type CorpusManifest,
  type TokenizerArtifact,
} from '../data/schemas';
import {
  BASE_VOCAB_SIZE,
  BOS_ID,
  EOS_ID,
  isSpecialTokenId,
  PAD_ID,
  SPECIAL_TOKEN_TEXT,
} from './constants';
import {
  decodeTokenIds,
  displayBytes,
  encodeByteTokens,
  encodeUtf8,
  type SpecialTokenOptions,
} from './codec';

const PAIR_STRIDE = 65_536;
const MAX_VOCAB_SIZE = 4_096;

interface Pair {
  left: number;
  right: number;
}

interface RankedPair extends Pair {
  key: number;
  count: number;
}

interface EncodingCandidate extends Pair {
  leftIndex: number;
  rightIndex: number;
  rank: number;
  mergedId: number;
}

export interface TokenizerTrainingProgress {
  mergeCount: number;
  targetMergeCount: number;
  selectedPair: readonly [number, number] | null;
  trainingCount: number | null;
}

export interface TokenizerTrainingOptions {
  targetSize: number;
  corpusSha256: string;
  onProgress?: (progress: TokenizerTrainingProgress) => void;
}

export interface MergeReplayStep {
  rank: number;
  mergeId: number;
  left: number;
  right: number;
  trainingCount: number;
  before: readonly number[];
  after: readonly number[];
  applications: number;
}

function pairKey(left: number, right: number): number {
  return left * PAIR_STRIDE + right;
}

function pairFromKey(key: number): Pair {
  return { left: Math.floor(key / PAIR_STRIDE), right: key % PAIR_STRIDE };
}

function isMergeablePair(left: number, right: number): boolean {
  return !isSpecialTokenId(left) && !isSpecialTokenId(right);
}

function validateTrainingOptions(options: TokenizerTrainingOptions): void {
  if (
    !Number.isInteger(options.targetSize) ||
    options.targetSize < BASE_VOCAB_SIZE ||
    options.targetSize > MAX_VOCAB_SIZE
  ) {
    throw new Error(
      `Tokenizer target size must be an integer from ${BASE_VOCAB_SIZE} to ${MAX_VOCAB_SIZE}.`,
    );
  }
  if (!/^[a-f0-9]{64}$/.test(options.corpusSha256)) {
    throw new Error('Tokenizer training requires a lowercase corpus SHA-256 digest.');
  }
}

function trainingMetadata(documents: readonly string[]): {
  documentCount: number;
  byteCount: number;
} {
  return {
    documentCount: documents.length,
    byteCount: documents.reduce((total, document) => total + encodeUtf8(document).length, 0),
  };
}

function createBaseTokenBytes(): number[][] {
  const tokenBytes = Array.from({ length: BASE_VOCAB_SIZE }, (_, id) => (id < 256 ? [id] : []));
  return tokenBytes;
}

function makeArtifact(
  documents: readonly string[],
  options: TokenizerTrainingOptions,
  merges: TokenizerArtifact['merges'],
  tokenBytes: readonly (readonly number[])[],
): TokenizerArtifact {
  const metadata = trainingMetadata(documents);
  const actualSize = BASE_VOCAB_SIZE + merges.length;
  return {
    formatVersion: ARTIFACT_FORMAT_VERSION,
    specialIds: { bos: BOS_ID, eos: EOS_ID, pad: PAD_ID },
    targetSize: actualSize,
    merges,
    tokens: tokenBytes.slice(0, actualSize).map((bytes, id) => ({
      id,
      bytes: [...bytes],
      display: isSpecialTokenId(id) ? SPECIAL_TOKEN_TEXT[id] : displayBytes(bytes),
    })),
    corpusSha256: options.corpusSha256,
    trainer: {
      algorithmVersion: 'bpe-v1',
      split: 'train',
      templateVersion: 'corpus-v1',
      documentCount: metadata.documentCount,
      byteCount: metadata.byteCount,
    },
  };
}

function countPairs(sequences: readonly (readonly number[])[]): Map<number, number> {
  const counts = new Map<number, number>();
  for (const sequence of sequences) {
    for (let index = 0; index + 1 < sequence.length; index += 1) {
      const left = sequence[index];
      const right = sequence[index + 1];
      if (!isMergeablePair(left, right)) continue;
      const key = pairKey(left, right);
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
  }
  return counts;
}

function selectPair(counts: ReadonlyMap<number, number>): RankedPair | null {
  let selected: RankedPair | null = null;
  for (const [key, count] of counts) {
    if (count <= 0) continue;
    const pair = pairFromKey(key);
    if (
      selected === null ||
      count > selected.count ||
      (count === selected.count &&
        (pair.left < selected.left || (pair.left === selected.left && pair.right < selected.right)))
    ) {
      selected = { key, count, ...pair };
    }
  }
  return selected;
}

function replacePair(
  sequence: readonly number[],
  left: number,
  right: number,
  mergedId: number,
): { sequence: number[]; applications: number } {
  const output: number[] = [];
  let applications = 0;
  for (let index = 0; index < sequence.length; index += 1) {
    if (sequence[index] === left && sequence[index + 1] === right) {
      output.push(mergedId);
      applications += 1;
      index += 1;
    } else {
      output.push(sequence[index]);
    }
  }
  return { sequence: output, applications };
}

export function trainBpeReference(
  documents: readonly string[],
  options: TokenizerTrainingOptions,
): TokenizerArtifact {
  validateTrainingOptions(options);
  let sequences = documents.map((document) => encodeByteTokens(document, { bos: true, eos: true }));
  const tokenBytes = createBaseTokenBytes();
  const merges: TokenizerArtifact['merges'] = [];

  while (BASE_VOCAB_SIZE + merges.length < options.targetSize) {
    const selected = selectPair(countPairs(sequences));
    if (!selected) break;
    const id = BASE_VOCAB_SIZE + merges.length;
    sequences = sequences.map(
      (sequence) => replacePair(sequence, selected.left, selected.right, id).sequence,
    );
    tokenBytes[id] = [...tokenBytes[selected.left], ...tokenBytes[selected.right]];
    merges.push({
      left: selected.left,
      right: selected.right,
      id,
      trainingCount: selected.count,
    });
    options.onProgress?.({
      mergeCount: merges.length,
      targetMergeCount: options.targetSize - BASE_VOCAB_SIZE,
      selectedPair: [selected.left, selected.right],
      trainingCount: selected.count,
    });
  }
  return makeArtifact(documents, options, merges, tokenBytes);
}

class PairMaxHeap {
  private readonly values: RankedPair[] = [];

  constructor(values: Iterable<RankedPair>) {
    for (const value of values) this.push(value);
  }

  private preferred(left: RankedPair, right: RankedPair): boolean {
    return (
      left.count > right.count ||
      (left.count === right.count &&
        (left.left < right.left || (left.left === right.left && left.right < right.right)))
    );
  }

  push(value: RankedPair): void {
    let index = this.values.push(value) - 1;
    while (index > 0) {
      const parent = Math.floor((index - 1) / 2);
      if (this.preferred(this.values[parent], value)) break;
      this.values[index] = this.values[parent];
      index = parent;
    }
    this.values[index] = value;
  }

  pop(): RankedPair | null {
    if (this.values.length === 0) return null;
    const root = this.values[0];
    const tail = this.values.pop();
    if (this.values.length > 0 && tail) {
      let index = 0;
      while (true) {
        const left = index * 2 + 1;
        if (left >= this.values.length) break;
        const right = left + 1;
        let child = left;
        if (right < this.values.length && this.preferred(this.values[right], this.values[left])) {
          child = right;
        }
        if (this.preferred(tail, this.values[child])) break;
        this.values[index] = this.values[child];
        index = child;
      }
      this.values[index] = tail;
    }
    return root;
  }
}

function heapEntry(key: number, count: number): RankedPair {
  return { key, count, ...pairFromKey(key) };
}

export function trainBpe(
  documents: readonly string[],
  options: TokenizerTrainingOptions,
): TokenizerArtifact {
  validateTrainingOptions(options);
  const sequences = documents.map((document) =>
    encodeByteTokens(document, { bos: true, eos: true }),
  );
  const nodeCount = sequences.reduce((total, sequence) => total + sequence.length, 0);
  const tokens = new Int32Array(nodeCount);
  const previous = new Int32Array(nodeCount);
  const next = new Int32Array(nodeCount);
  previous.fill(-1);
  next.fill(-1);

  const counts = new Map<number, number>();
  const occurrences = new Map<number, number[]>();
  let offset = 0;
  for (const sequence of sequences) {
    for (let local = 0; local < sequence.length; local += 1) {
      const index = offset + local;
      tokens[index] = sequence[local];
      previous[index] = local === 0 ? -1 : index - 1;
      next[index] = local + 1 === sequence.length ? -1 : index + 1;
      if (local + 1 < sequence.length) {
        const right = sequence[local + 1];
        if (isMergeablePair(sequence[local], right)) {
          const key = pairKey(sequence[local], right);
          counts.set(key, (counts.get(key) ?? 0) + 1);
          const positions = occurrences.get(key);
          if (positions) positions.push(index);
          else occurrences.set(key, [index]);
        }
      }
    }
    offset += sequence.length;
  }

  const heap = new PairMaxHeap(Array.from(counts, ([key, count]) => heapEntry(key, count)));
  const tokenBytes = createBaseTokenBytes();
  const merges: TokenizerArtifact['merges'] = [];

  const addDelta = (
    deltas: Map<number, number>,
    left: number,
    right: number,
    delta: number,
  ): void => {
    if (!isMergeablePair(left, right)) return;
    const key = pairKey(left, right);
    deltas.set(key, (deltas.get(key) ?? 0) + delta);
  };

  const addOccurrence = (leftIndex: number, rightIndex: number): void => {
    const left = tokens[leftIndex];
    const right = tokens[rightIndex];
    if (!isMergeablePair(left, right)) return;
    const key = pairKey(left, right);
    const positions = occurrences.get(key);
    if (positions) positions.push(leftIndex);
    else occurrences.set(key, [leftIndex]);
  };

  while (BASE_VOCAB_SIZE + merges.length < options.targetSize) {
    let selected: RankedPair | null = null;
    while ((selected = heap.pop()) !== null) {
      if (counts.get(selected.key) === selected.count && selected.count > 0) break;
    }
    if (!selected) break;

    const positions = occurrences.get(selected.key) ?? [];
    positions.sort((left, right) => left - right);
    const mergedId = BASE_VOCAB_SIZE + merges.length;
    const deltas = new Map<number, number>();
    let applications = 0;

    for (const leftIndex of positions) {
      const rightIndex = next[leftIndex];
      if (
        rightIndex < 0 ||
        tokens[leftIndex] !== selected.left ||
        tokens[rightIndex] !== selected.right
      ) {
        continue;
      }
      const previousIndex = previous[leftIndex];
      const followingIndex = next[rightIndex];
      if (previousIndex >= 0) addDelta(deltas, tokens[previousIndex], selected.left, -1);
      addDelta(deltas, selected.left, selected.right, -1);
      if (followingIndex >= 0) addDelta(deltas, selected.right, tokens[followingIndex], -1);

      tokens[leftIndex] = mergedId;
      next[leftIndex] = followingIndex;
      if (followingIndex >= 0) previous[followingIndex] = leftIndex;
      previous[rightIndex] = -2;
      next[rightIndex] = -2;

      if (previousIndex >= 0) {
        addDelta(deltas, tokens[previousIndex], mergedId, 1);
        addOccurrence(previousIndex, leftIndex);
      }
      if (followingIndex >= 0) {
        addDelta(deltas, mergedId, tokens[followingIndex], 1);
        addOccurrence(leftIndex, followingIndex);
      }
      applications += 1;
    }

    if (applications === 0) {
      counts.set(selected.key, 0);
      continue;
    }

    for (const [key, delta] of deltas) {
      const updated = (counts.get(key) ?? 0) + delta;
      if (updated < 0) throw new Error('Incremental BPE pair count became negative.');
      counts.set(key, updated);
      if (updated > 0) heap.push(heapEntry(key, updated));
    }
    tokenBytes[mergedId] = [...tokenBytes[selected.left], ...tokenBytes[selected.right]];
    merges.push({
      left: selected.left,
      right: selected.right,
      id: mergedId,
      trainingCount: selected.count,
    });
    options.onProgress?.({
      mergeCount: merges.length,
      targetMergeCount: options.targetSize - BASE_VOCAB_SIZE,
      selectedPair: [selected.left, selected.right],
      trainingCount: selected.count,
    });
  }

  return makeArtifact(documents, options, merges, tokenBytes);
}

function mergeRanks(artifact: TokenizerArtifact): Map<number, { rank: number; id: number }> {
  return new Map(
    artifact.merges.map((merge, rank) => [
      pairKey(merge.left, merge.right),
      { rank, id: merge.id },
    ]),
  );
}

class EncodingMinHeap {
  private readonly values: EncodingCandidate[] = [];

  private preferred(left: EncodingCandidate, right: EncodingCandidate): boolean {
    return left.rank < right.rank || (left.rank === right.rank && left.leftIndex < right.leftIndex);
  }

  push(value: EncodingCandidate): void {
    let index = this.values.push(value) - 1;
    while (index > 0) {
      const parent = Math.floor((index - 1) / 2);
      if (this.preferred(this.values[parent], value)) break;
      this.values[index] = this.values[parent];
      index = parent;
    }
    this.values[index] = value;
  }

  pop(): EncodingCandidate | null {
    if (this.values.length === 0) return null;
    const root = this.values[0];
    const tail = this.values.pop();
    if (this.values.length > 0 && tail) {
      let index = 0;
      while (true) {
        const left = index * 2 + 1;
        if (left >= this.values.length) break;
        const right = left + 1;
        let child = left;
        if (right < this.values.length && this.preferred(this.values[right], this.values[left])) {
          child = right;
        }
        if (this.preferred(tail, this.values[child])) break;
        this.values[index] = this.values[child];
        index = child;
      }
      this.values[index] = tail;
    }
    return root;
  }
}

export function encodeIdsWithArtifact(
  initialIds: readonly number[],
  artifact: TokenizerArtifact,
): number[] {
  if (artifact.merges.length === 0 || initialIds.length < 2) return [...initialIds];
  const ranks = mergeRanks(artifact);
  const tokens = Int32Array.from(initialIds);
  const previous = new Int32Array(tokens.length);
  const next = new Int32Array(tokens.length);
  for (let index = 0; index < tokens.length; index += 1) {
    previous[index] = index - 1;
    next[index] = index + 1 < tokens.length ? index + 1 : -1;
  }
  const heap = new EncodingMinHeap();

  const addCandidate = (leftIndex: number, rightIndex: number): void => {
    if (leftIndex < 0 || rightIndex < 0) return;
    const left = tokens[leftIndex];
    const right = tokens[rightIndex];
    const ranked = ranks.get(pairKey(left, right));
    if (!ranked) return;
    heap.push({ left, right, leftIndex, rightIndex, rank: ranked.rank, mergedId: ranked.id });
  };

  for (let index = 0; index + 1 < tokens.length; index += 1) addCandidate(index, index + 1);
  let candidate: EncodingCandidate | null;
  while ((candidate = heap.pop()) !== null) {
    if (
      next[candidate.leftIndex] !== candidate.rightIndex ||
      tokens[candidate.leftIndex] !== candidate.left ||
      tokens[candidate.rightIndex] !== candidate.right
    ) {
      continue;
    }
    const before = previous[candidate.leftIndex];
    const after = next[candidate.rightIndex];
    tokens[candidate.leftIndex] = candidate.mergedId;
    next[candidate.leftIndex] = after;
    if (after >= 0) previous[after] = candidate.leftIndex;
    previous[candidate.rightIndex] = -2;
    next[candidate.rightIndex] = -2;
    if (before >= 0) addCandidate(before, candidate.leftIndex);
    if (after >= 0) addCandidate(candidate.leftIndex, after);
  }

  const output: number[] = [];
  let index = tokens.length === 0 ? -1 : 0;
  while (index >= 0) {
    output.push(tokens[index]);
    index = next[index];
  }
  return output;
}

export function encodeText(
  text: string,
  artifact: TokenizerArtifact,
  options: SpecialTokenOptions = {},
): number[] {
  return encodeIdsWithArtifact(encodeByteTokens(text, options), artifact);
}

export function encodeTextReference(
  text: string,
  artifact: TokenizerArtifact,
  options: SpecialTokenOptions = {},
): number[] {
  let ids = encodeByteTokens(text, options);
  for (const merge of artifact.merges) {
    ids = replacePair(ids, merge.left, merge.right, merge.id).sequence;
  }
  return ids;
}

export function decodeText(ids: readonly number[], artifact: TokenizerArtifact): string {
  return decodeTokenIds(ids, artifact);
}

export function replayMerges(text: string, artifact: TokenizerArtifact): MergeReplayStep[] {
  let ids = encodeByteTokens(text);
  const steps: MergeReplayStep[] = [];
  for (const [rank, merge] of artifact.merges.entries()) {
    const before = ids;
    const result = replacePair(before, merge.left, merge.right, merge.id);
    if (result.applications === 0) continue;
    ids = result.sequence;
    steps.push({
      rank,
      mergeId: merge.id,
      left: merge.left,
      right: merge.right,
      trainingCount: merge.trainingCount,
      before,
      after: ids,
      applications: result.applications,
    });
  }
  return steps;
}

export function assertArtifactCorpusHash(
  artifact: TokenizerArtifact,
  manifest: CorpusManifest,
): void {
  if (artifact.corpusSha256 !== manifest.corpusSha256) {
    throw new Error(
      `Tokenizer/corpus hash mismatch: tokenizer ${artifact.corpusSha256}, corpus ${manifest.corpusSha256}.`,
    );
  }
}
