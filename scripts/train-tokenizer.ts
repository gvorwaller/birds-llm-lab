import { randomBytes } from 'node:crypto';
import { mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises';
import { basename, dirname, join, resolve } from 'node:path';
import {
  parseTokenizerArtifact,
  serializeStableJson,
  type TokenizerArtifact,
} from '../src/lib/data/schemas.js';
import { trainBpe } from '../src/lib/tokenizer/bpe.js';
import { BASE_VOCAB_SIZE } from '../src/lib/tokenizer/constants.js';
import { loadCorpus } from './tokenizer-corpus.js';

interface Arguments {
  vocabSize: number;
  outputPath: string;
  corpusPath: string;
  manifestPath: string;
}

function usage(): string {
  return [
    'Usage: npm run train:tokenizer -- --vocab-size 1024 [options]',
    '',
    'Options:',
    '  --output <path>    Artifact destination (default: data/tokenizer.json)',
    '  --corpus <path>    Exported corpus JSONL (default: data/corpus.jsonl)',
    '  --manifest <path>  Corpus manifest (default: data/manifest.json)',
  ].join('\n');
}

function parseInteger(value: string | undefined, name: string): number {
  const parsed = Number(value);
  if (!Number.isInteger(parsed)) throw new Error(`${name} must be an integer.`);
  return parsed;
}

export function parseArguments(argv: readonly string[]): Arguments {
  const values = new Map<string, string>();
  for (let index = 0; index < argv.length; index += 1) {
    const flag = argv[index];
    if (!['--vocab-size', '--output', '--corpus', '--manifest'].includes(flag)) {
      throw new Error(`Unknown argument ${flag}.\n\n${usage()}`);
    }
    const value = argv[index + 1];
    if (!value || value.startsWith('--')) throw new Error(`${flag} requires a value.`);
    values.set(flag, value);
    index += 1;
  }
  const vocabSize = parseInteger(values.get('--vocab-size'), '--vocab-size');
  if (vocabSize < BASE_VOCAB_SIZE || vocabSize > 4_096) {
    throw new Error(`--vocab-size must be from ${BASE_VOCAB_SIZE} to 4096.`);
  }
  return {
    vocabSize,
    outputPath: resolve(values.get('--output') ?? 'data/tokenizer.json'),
    corpusPath: resolve(values.get('--corpus') ?? 'data/corpus.jsonl'),
    manifestPath: resolve(values.get('--manifest') ?? 'data/manifest.json'),
  };
}

async function writeArtifactAtomically(
  artifact: TokenizerArtifact,
  outputPath: string,
): Promise<void> {
  await mkdir(dirname(outputPath), { recursive: true });
  const temporaryPath = join(
    dirname(outputPath),
    `.${basename(outputPath)}.${process.pid}.${randomBytes(8).toString('hex')}.tmp`,
  );
  try {
    await writeFile(temporaryPath, serializeStableJson(artifact), { flag: 'wx', mode: 0o600 });
    const validated = parseTokenizerArtifact(
      JSON.parse(await readFile(temporaryPath, 'utf8')) as unknown,
    );
    if (validated.corpusSha256 !== artifact.corpusSha256) {
      throw new Error('Temporary tokenizer artifact failed corpus-hash validation.');
    }
    await rename(temporaryPath, outputPath);
  } catch (error) {
    await rm(temporaryPath, { force: true });
    throw error;
  }
}

async function main(): Promise<void> {
  const args = parseArguments(process.argv.slice(2));
  const loaded = await loadCorpus({
    corpusPath: args.corpusPath,
    manifestPath: args.manifestPath,
  });
  console.log(
    `Training deterministic BPE on ${loaded.trainingDocuments.length} train documents (${loaded.manifest.splits.train.bytes} manifest bytes).`,
  );
  const started = performance.now();
  const artifact = trainBpe(loaded.trainingDocuments, {
    targetSize: args.vocabSize,
    corpusSha256: loaded.manifest.corpusSha256,
    onProgress(progress) {
      if (
        progress.mergeCount === 1 ||
        progress.mergeCount === progress.targetMergeCount ||
        progress.mergeCount % 25 === 0
      ) {
        console.log(
          `merge ${progress.mergeCount}/${progress.targetMergeCount}: ${progress.selectedPair?.join(',')} (${progress.trainingCount} occurrences)`,
        );
      }
    },
  });
  if (artifact.targetSize !== args.vocabSize) {
    throw new Error(
      `Training stopped at ${artifact.targetSize} tokens before requested size ${args.vocabSize}.`,
    );
  }
  await writeArtifactAtomically(artifact, args.outputPath);
  console.log(
    `Wrote ${artifact.targetSize}-token artifact to ${args.outputPath} in ${((performance.now() - started) / 1000).toFixed(2)} s.`,
  );
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((error: unknown) => {
    console.error(error instanceof Error ? error.message : 'Unknown tokenizer training failure.');
    process.exitCode = 1;
  });
}
