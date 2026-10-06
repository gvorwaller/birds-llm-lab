/** Verified locally against ready-v1 and tokenizer-1024 on 2026-10-06. */
export const KESTREL_EXAMPLE = {
  checkpointId: 'ready-v1',
  trainingStep: 2_000,
  corpusSha256: 'fc166042bfa433860c147102e405741df021725e58fc89332741f1c8b57d9232',
  tokenizerSha256: 'e8544be8159e4a6f1bdd59fd4f84e754406072927d58c5256c0c585701551e5c',
  prompt: 'Common name: American Kestrel\nScientific name:',
  seed: 1,
  controls: { temperature: 0.8, topK: 20, topP: 1 },
  count: 18,
  generatedIds: [
    318, 71, 268, 263, 97, 370, 879, 80, 101, 271, 111, 261, 276, 667, 67, 413, 386, 103,
  ],
  continuation: '\n\nGarinaus\nOrder: Passeriformes\nFamily: Pealothen\nScientific name: Capus g',
  focusStep: 6,
  focusTokenId: 879,
  focusTokenText: '\nOrder: Passeriformes\nFamily: ',
  focusProbability: 0.27882142788080144,
  speciesCode: 'amekes',
  referenceOrder: 'Falconiformes',
} as const;
