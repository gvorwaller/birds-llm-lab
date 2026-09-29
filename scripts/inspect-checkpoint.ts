import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { parseCorpusManifest, parseTokenizerArtifact } from '../src/lib/data/schemas';
import { SeededRandom } from '../src/lib/math/rng';
import { loadCheckpoint } from '../server/checkpoint-store';
import { generateTextSample } from '../server/training/trainer';

const checkpointPath = resolve(process.argv[2] ?? 'checkpoints/ready-v1');
const tokenizerPath = resolve('src/assets/tokenizer-1024.json');
const [loaded, tokenizerBytes, manifestText, corpusBytes] = await Promise.all([
  loadCheckpoint(checkpointPath),
  readFile(tokenizerPath),
  readFile(resolve('data/manifest.json'), 'utf8'),
  readFile(resolve('data/corpus.jsonl')),
]);
const tokenizerSha256 = createHash('sha256').update(tokenizerBytes).digest('hex');
const corpusSha256 = createHash('sha256').update(corpusBytes).digest('hex');
const tokenizer = parseTokenizerArtifact(JSON.parse(tokenizerBytes.toString('utf8')) as unknown);
const manifest = parseCorpusManifest(JSON.parse(manifestText) as unknown);
if (
  loaded.config.tokenizerSha256 !== tokenizerSha256 ||
  loaded.config.corpusSha256 !== corpusSha256 ||
  manifest.corpusSha256 !== corpusSha256 ||
  tokenizer.corpusSha256 !== corpusSha256
) {
  throw new Error('Checkpoint, tokenizer, corpus, and manifest identities do not agree.');
}
if (loaded.resumeState === null) throw new Error('Ready checkpoint is missing resumable state.');

function boundary(split: 'train' | 'validation') {
  const entries = loaded.trainingLog.entries.filter((entry) => entry.split === split);
  if (entries.length < 2) throw new Error(`Checkpoint has insufficient ${split} metrics.`);
  return { initial: entries[0], final: entries.at(-1)! };
}

const train = boundary('train');
const validation = boundary('validation');
const sample = generateTextSample(
  loaded.parameters,
  loaded.config.model,
  tokenizer,
  new SeededRandom(loaded.config.seed ^ 0x1a2b_3c4d),
);
const passed =
  loaded.config.trainingStep === loaded.config.optimizer.totalSteps &&
  train.final.meanLoss < train.initial.meanLoss &&
  validation.final.meanLoss < validation.initial.meanLoss;

console.log(
  JSON.stringify(
    {
      status: passed ? 'passed' : 'failed',
      checkpointPath,
      trainingStep: loaded.config.trainingStep,
      totalSteps: loaded.config.optimizer.totalSteps,
      corpusSha256,
      tokenizerSha256,
      sourceGitRevision: loaded.config.sourceGitRevision,
      train: {
        initialLoss: train.initial.meanLoss,
        finalLoss: train.final.meanLoss,
        initialPredictions: train.initial.predictionCount,
        finalPredictions: train.final.predictionCount,
      },
      validation: {
        initialLoss: validation.initial.meanLoss,
        finalLoss: validation.final.meanLoss,
        initialPredictions: validation.initial.predictionCount,
        finalPredictions: validation.final.predictionCount,
      },
      generatedSample: sample,
    },
    null,
    2,
  ),
);
if (!passed) process.exitCode = 2;
