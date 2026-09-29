import { serializeStableJson } from '../src/lib/data/schemas.js';
import { trainBpe, trainBpeReference } from '../src/lib/tokenizer/bpe.js';
import { BASE_VOCAB_SIZE } from '../src/lib/tokenizer/constants.js';
import { encodeUtf8 } from '../src/lib/tokenizer/codec.js';
import { loadCorpus } from './tokenizer-corpus.js';

async function main(): Promise<void> {
  const loaded = await loadCorpus();
  const subset: string[] = [];
  let bytes = 0;
  for (const document of loaded.trainingDocuments) {
    if (bytes >= 1_000_000) break;
    subset.push(document);
    bytes += encodeUtf8(document).length;
  }
  const options = {
    targetSize: BASE_VOCAB_SIZE + 32,
    corpusSha256: loaded.manifest.corpusSha256,
  };
  const referenceStarted = performance.now();
  const reference = trainBpeReference(subset, options);
  const referenceMs = performance.now() - referenceStarted;
  const optimizedStarted = performance.now();
  const optimized = trainBpe(subset, options);
  const optimizedMs = performance.now() - optimizedStarted;
  if (serializeStableJson(reference) !== serializeStableJson(optimized)) {
    throw new Error('Optimized trainer did not match the reference artifact.');
  }
  console.log(
    JSON.stringify(
      {
        subsetBytes: bytes,
        documents: subset.length,
        learnedMerges: reference.merges.length,
        referenceMs: Number(referenceMs.toFixed(2)),
        optimizedMs: Number(optimizedMs.toFixed(2)),
        artifactsMatch: true,
      },
      null,
      2,
    ),
  );
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : 'Unknown tokenizer benchmark failure.');
  process.exitCode = 1;
});
