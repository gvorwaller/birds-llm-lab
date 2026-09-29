import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import {
  parseTokenizerArtifact,
  serializeStableJson,
  type TokenizerArtifact,
} from '../src/lib/data/schemas.js';
import {
  assertArtifactCorpusHash,
  decodeText,
  encodeText,
  encodeTextReference,
  trainBpe,
} from '../src/lib/tokenizer/bpe.js';
import { trainingDocumentText } from '../src/lib/tokenizer/corpus.js';
import { loadCorpus } from './tokenizer-corpus.js';

async function loadArtifact(path: string): Promise<TokenizerArtifact> {
  return parseTokenizerArtifact(JSON.parse(await readFile(resolve(path), 'utf8')) as unknown);
}

async function main(): Promise<void> {
  const loaded = await loadCorpus();
  const byteArtifact = await loadArtifact('src/assets/tokenizer-byte-only.json');
  const trainedArtifact = await loadArtifact('src/assets/tokenizer-1024.json');
  assertArtifactCorpusHash(byteArtifact, loaded.manifest);
  assertArtifactCorpusHash(trainedArtifact, loaded.manifest);
  if (byteArtifact.targetSize !== 259 || trainedArtifact.targetSize !== 1024) {
    throw new Error('Prebuilt tokenizer artifacts have unexpected vocabulary sizes.');
  }

  let textBytes = 0;
  let encodedTokens = 0;
  for (const [index, row] of loaded.rows.entries()) {
    const text = trainingDocumentText(row);
    const ids = encodeText(text, trainedArtifact, { bos: true, eos: true });
    if (decodeText(ids, trainedArtifact) !== text) {
      throw new Error(`Full-corpus tokenizer round-trip failed at row ${index + 1}.`);
    }
    if (index % 257 === 0) {
      const reference = encodeTextReference(text, trainedArtifact, { bos: true, eos: true });
      if (
        ids.length !== reference.length ||
        ids.some((id, tokenIndex) => id !== reference[tokenIndex])
      ) {
        throw new Error(`Fast/reference encoder mismatch at row ${index + 1}.`);
      }
    }
    textBytes += new TextEncoder().encode(text).length;
    encodedTokens += ids.length;
  }

  const repeated = trainBpe(loaded.trainingDocuments, {
    targetSize: 1024,
    corpusSha256: loaded.manifest.corpusSha256,
  });
  if (serializeStableJson(repeated) !== serializeStableJson(trainedArtifact)) {
    throw new Error(
      'Repeated corpus/settings did not produce a byte-identical tokenizer artifact.',
    );
  }
  console.log(
    `M1 verified: ${loaded.rows.length} corpus documents, ${textBytes} text bytes, ${encodedTokens} encoded tokens, byte-identical 1024-token artifact.`,
  );
}

main().catch((error: unknown) => {
  console.error(
    error instanceof Error ? error.message : 'Unknown Milestone 1 verification failure.',
  );
  process.exitCode = 1;
});
