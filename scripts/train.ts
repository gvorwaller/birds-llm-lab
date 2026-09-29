import { trainReadyCheckpoint } from '../server/training/trainer';

interface Arguments {
  readonly resumeFrom?: string;
  readonly outputRoot?: string;
  readonly totalSteps?: number;
  readonly stopAfterStep?: number;
  readonly checkpointEvery?: number;
}

function integer(value: string | undefined, flag: string): number {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed <= 0) throw new Error(`${flag} requires an integer.`);
  return parsed;
}

function text(value: string | undefined, flag: string): string {
  if (value === undefined || value.length === 0) throw new Error(`${flag} requires a value.`);
  return value;
}

function parseArguments(argv: readonly string[]): Arguments {
  const result: {
    resumeFrom?: string;
    outputRoot?: string;
    totalSteps?: number;
    stopAfterStep?: number;
    checkpointEvery?: number;
  } = {};
  for (let index = 0; index < argv.length; index += 1) {
    const flag = argv[index];
    const value = argv[index + 1];
    if (flag === '--resume') result.resumeFrom = text(value, flag);
    else if (flag === '--output-root') result.outputRoot = text(value, flag);
    else if (flag === '--total-steps') result.totalSteps = integer(value, flag);
    else if (flag === '--stop-after') result.stopAfterStep = integer(value, flag);
    else if (flag === '--checkpoint-every') result.checkpointEvery = integer(value, flag);
    else throw new Error(`Unknown training argument: ${flag}.`);
    index += 1;
  }
  return result;
}

const result = await trainReadyCheckpoint({
  ...parseArguments(process.argv.slice(2)),
  onProgress(entry) {
    console.log(
      `${entry.split.padEnd(10)} step=${entry.step.toString().padStart(4)} loss=${entry.meanLoss.toFixed(6)} perplexity=${entry.perplexity.toFixed(2)} predictions=${entry.predictionCount} lr=${entry.learningRate.toExponential(3)} elapsed=${(entry.elapsedMs / 1_000).toFixed(1)}s`,
    );
    if (entry.sample !== null) console.log(`sample: ${JSON.stringify(entry.sample)}`);
  },
});

console.log(
  JSON.stringify(
    {
      status: 'complete',
      checkpointPath: result.checkpointPath,
      trainingStep: result.config.trainingStep,
      corpusSha256: result.config.corpusSha256,
      tokenizerSha256: result.config.tokenizerSha256,
      sourceGitRevision: result.config.sourceGitRevision,
    },
    null,
    2,
  ),
);
