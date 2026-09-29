import { describe, expect, it } from 'vitest';
import { BOS_ID, EOS_ID } from './constants';
import { decodeTokenIds, decodeUtf8, encodeByteTokens, encodeUtf8 } from './codec';

describe('UTF-8 byte and special-token codec', () => {
  const examples = [
    '',
    'Warbler',
    'café',
    '🐦‍⬛ birds',
    '东方白鹳',
    'literal <|bos|> and <|eos|>',
    'line one\nline two\t✓',
  ];

  it.each(examples)('round-trips ordinary Unicode text: %j', (text) => {
    expect(decodeUtf8(encodeUtf8(text))).toBe(text);
    expect(decodeTokenIds(encodeByteTokens(text))).toBe(text);
  });

  it('inserts special tokens only when explicitly requested', () => {
    const literal = '<|bos|>';
    expect(encodeByteTokens(literal)).toEqual(encodeUtf8(literal));
    const framed = encodeByteTokens(literal, { bos: true, eos: true });
    expect(framed[0]).toBe(BOS_ID);
    expect(framed.at(-1)).toBe(EOS_ID);
    expect(decodeTokenIds(framed)).toBe(literal);
    expect(decodeTokenIds(framed, undefined, 'literal')).toBe(`<|bos|>${literal}<|eos|>`);
  });
});
