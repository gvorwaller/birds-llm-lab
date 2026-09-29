import type { TokenizerArtifact } from '../data/schemas';
import { BASE_VOCAB_SIZE, BOS_ID, EOS_ID, isSpecialTokenId, SPECIAL_TOKEN_TEXT } from './constants';

const encoder = new TextEncoder();
const decoder = new TextDecoder('utf-8', { fatal: false });

export interface SpecialTokenOptions {
  bos?: boolean;
  eos?: boolean;
}

export function encodeUtf8(text: string): number[] {
  return Array.from(encoder.encode(text));
}

export function decodeUtf8(bytes: readonly number[]): string {
  return decoder.decode(Uint8Array.from(bytes));
}

export function encodeByteTokens(text: string, options: SpecialTokenOptions = {}): number[] {
  const ids = encodeUtf8(text);
  if (options.bos) ids.unshift(BOS_ID);
  if (options.eos) ids.push(EOS_ID);
  return ids;
}

export function displayBytes(bytes: readonly number[]): string {
  if (bytes.length === 0) return '';
  const visibleWhitespace = decodeUtf8(bytes)
    .replaceAll(' ', '␠')
    .replaceAll('\n', '↵')
    .replaceAll('\r', '␍')
    .replaceAll('\t', '⇥');
  return Array.from(visibleWhitespace, (character) => {
    const code = character.codePointAt(0) ?? 0;
    return code < 32 || code === 127 ? `\\x${code.toString(16).padStart(2, '0')}` : character;
  }).join('');
}

export function tokenByteTable(artifact: TokenizerArtifact): readonly (readonly number[])[] {
  const table: number[][] = Array.from({ length: artifact.tokens.length }, () => []);
  for (const token of artifact.tokens) table[token.id] = token.bytes;
  return table;
}

export function decodeTokenIds(
  ids: readonly number[],
  artifact?: TokenizerArtifact,
  specialTokens: 'omit' | 'literal' | 'error' = 'omit',
): string {
  const table = artifact ? tokenByteTable(artifact) : undefined;
  const bytes: number[] = [];
  let literal = '';

  const flush = (): void => {
    if (bytes.length === 0) return;
    literal += decodeUtf8(bytes);
    bytes.length = 0;
  };

  for (const id of ids) {
    if (isSpecialTokenId(id)) {
      if (specialTokens === 'error') throw new Error(`Cannot decode special token id ${id}.`);
      if (specialTokens === 'literal') {
        flush();
        literal += SPECIAL_TOKEN_TEXT[id];
      }
      continue;
    }
    if (id < 0 || !Number.isInteger(id)) throw new Error(`Invalid token id ${id}.`);
    if (id < BYTE_TOKEN_COUNT) {
      bytes.push(id);
      continue;
    }
    if (!artifact || id < BASE_VOCAB_SIZE || !table?.[id]) {
      throw new Error(`Token id ${id} is not present in the tokenizer artifact.`);
    }
    bytes.push(...table[id]);
  }
  flush();
  return literal;
}

const BYTE_TOKEN_COUNT = 256;
