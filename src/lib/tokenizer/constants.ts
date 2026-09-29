export const BYTE_TOKEN_COUNT = 256;
export const BOS_ID = 256;
export const EOS_ID = 257;
export const PAD_ID = 258;
export const BASE_VOCAB_SIZE = 259;

export const SPECIAL_TOKEN_TEXT = {
  [BOS_ID]: '<|bos|>',
  [EOS_ID]: '<|eos|>',
  [PAD_ID]: '<|pad|>',
} as const;

export type SpecialTokenId = keyof typeof SPECIAL_TOKEN_TEXT;

export function isSpecialTokenId(id: number): id is SpecialTokenId {
  return id === BOS_ID || id === EOS_ID || id === PAD_ID;
}
