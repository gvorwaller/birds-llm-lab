import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { loadCheckpoint } from '../server/checkpoint-store';
import { CorpusEvidenceIndex } from '../server/evidence/index';
import { parseTokenizerArtifact } from '../src/lib/data/schemas';
import { KESTREL_EXAMPLE } from '../src/lib/evidence/saved-example';
import { replayKestrelExample } from '../src/lib/evidence/verified-example';

const checkpointPath = resolve('checkpoints/ready-v1');
const tokenizerPath = resolve('src/assets/tokenizer-1024.json');
const corpusPath = resolve('data/corpus.jsonl');
const manifestPath = resolve('data/manifest.json');
const [checkpoint, tokenizerBytes, corpusBytes] = await Promise.all([
  loadCheckpoint(checkpointPath),
  readFile(tokenizerPath),
  readFile(corpusPath),
]);
const tokenizerSha256 = createHash('sha256').update(tokenizerBytes).digest('hex');
const corpusSha256 = createHash('sha256').update(corpusBytes).digest('hex');
if (
  tokenizerSha256 !== KESTREL_EXAMPLE.tokenizerSha256 ||
  corpusSha256 !== KESTREL_EXAMPLE.corpusSha256
) {
  throw new Error('Local tokenizer or corpus differs from the saved example.');
}
const replay = replayKestrelExample({
  checkpointId: KESTREL_EXAMPLE.checkpointId,
  config: checkpoint.config,
  tokenizer: parseTokenizerArtifact(JSON.parse(tokenizerBytes.toString('utf8')) as unknown),
  registry: checkpoint.parameters,
});
const evidence = await new CorpusEvidenceIndex(corpusPath, manifestPath).verifiedExampleEvidence();
if (
  evidence.trainDocuments !== 9795 ||
  evidence.trainSpanDocuments !== 5939 ||
  evidence.target.split !== 'test' ||
  evidence.target.order !== KESTREL_EXAMPLE.referenceOrder
) {
  throw new Error('Corpus co-occurrence or target reference differs from the saved example.');
}
console.log(
  JSON.stringify(
    {
      status: 'passed',
      checkpoint: KESTREL_EXAMPLE.checkpointId,
      tokenIds: replay.replay.generatedIds,
      focusProbability: replay.focusProbability,
      focusRank: replay.focusRank,
      trainDocuments: evidence.trainDocuments,
      trainSpanDocuments: evidence.trainSpanDocuments,
      targetOrder: evidence.target.order,
      targetSplit: evidence.target.split,
    },
    null,
    2,
  ),
);
