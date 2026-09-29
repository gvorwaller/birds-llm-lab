import { lstat, mkdir, mkdtemp, readFile, rename, rm, writeFile } from 'node:fs/promises';
import { basename, dirname, join, resolve } from 'node:path';
import {
  parseCheckpointConfig,
  parseTrainingLog,
  parseWeightIndex,
  serializeStableJson,
  type CheckpointConfig,
  type TrainingLogArtifact,
  type WeightIndexArtifact,
} from '../src/lib/data/schemas';
import {
  CheckpointValidationError,
  decodeCheckpointWeights,
  encodeCheckpointWeights,
} from '../src/lib/model/checkpoint';
import type { ParameterRegistry } from '../src/lib/model/parameters';
import { parseTrainerState, serializeTrainerState, type TrainerResumeState } from './trainer-state';

const CHECKPOINT_FILES = {
  config: 'config.json',
  weights: 'weights.bin',
  weightIndex: 'weights.index.json',
  trainingLog: 'training-log.json',
  trainerState: 'trainer-state.json',
} as const;

export interface LoadedCheckpoint {
  readonly config: CheckpointConfig;
  readonly parameters: ParameterRegistry;
  readonly weightIndex: WeightIndexArtifact;
  readonly trainingLog: TrainingLogArtifact;
  readonly resumeState: TrainerResumeState | null;
}

async function pathExists(path: string): Promise<boolean> {
  try {
    await lstat(path);
    return true;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return false;
    throw error;
  }
}

async function readJson(path: string): Promise<unknown> {
  try {
    return JSON.parse(await readFile(path, 'utf8')) as unknown;
  } catch (error) {
    if (error instanceof SyntaxError) {
      throw new CheckpointValidationError(`${basename(path)} is not valid JSON: ${error.message}`);
    }
    throw error;
  }
}

function validateLog(config: CheckpointConfig, log: TrainingLogArtifact): void {
  for (const [index, entry] of log.entries.entries()) {
    if (entry.step > config.trainingStep) {
      throw new CheckpointValidationError(
        `training-log.json entry ${index} is at step ${entry.step}, after checkpoint step ${config.trainingStep}.`,
      );
    }
  }
}

export async function loadCheckpoint(directoryInput: string): Promise<LoadedCheckpoint> {
  const directory = resolve(directoryInput);
  const trainerStatePath = join(directory, CHECKPOINT_FILES.trainerState);
  const [configValue, weightIndexValue, trainingLogValue, weights, trainerStateValue] =
    await Promise.all([
      readJson(join(directory, CHECKPOINT_FILES.config)),
      readJson(join(directory, CHECKPOINT_FILES.weightIndex)),
      readJson(join(directory, CHECKPOINT_FILES.trainingLog)),
      readFile(join(directory, CHECKPOINT_FILES.weights)),
      pathExists(trainerStatePath).then((exists) => (exists ? readJson(trainerStatePath) : null)),
    ]);
  const config = parseCheckpointConfig(configValue);
  const trainingLog = parseTrainingLog(trainingLogValue);
  validateLog(config, trainingLog);
  const weightIndex = parseWeightIndex(weightIndexValue);
  const parameters = decodeCheckpointWeights(config.model, weightIndex, weights);
  const resumeState =
    trainerStateValue === null ? null : parseTrainerState(trainerStateValue, parameters);
  if (resumeState !== null && resumeState.optimizer.step !== config.trainingStep) {
    throw new CheckpointValidationError(
      `trainer-state.json optimizer step ${resumeState.optimizer.step} does not match checkpoint step ${config.trainingStep}.`,
    );
  }
  return { config, parameters, weightIndex, trainingLog, resumeState };
}

export async function writeCheckpoint(
  destinationInput: string,
  configInput: CheckpointConfig,
  parameters: ParameterRegistry,
  trainingLogInput: TrainingLogArtifact,
  resumeState: TrainerResumeState | null = null,
): Promise<LoadedCheckpoint> {
  const destination = resolve(destinationInput);
  const config = parseCheckpointConfig(configInput);
  const trainingLog = parseTrainingLog(trainingLogInput);
  validateLog(config, trainingLog);
  const encoded = encodeCheckpointWeights(config.model, parameters);
  if (await pathExists(destination)) {
    throw new Error(`Refusing to replace existing checkpoint path: ${destination}.`);
  }

  const parent = dirname(destination);
  await mkdir(parent, { recursive: true, mode: 0o700 });
  const temporary = await mkdtemp(join(parent, `.${basename(destination)}.tmp-`));
  try {
    await Promise.all([
      writeFile(join(temporary, CHECKPOINT_FILES.config), serializeStableJson(config), {
        flag: 'wx',
        mode: 0o600,
      }),
      writeFile(join(temporary, CHECKPOINT_FILES.weightIndex), serializeStableJson(encoded.index), {
        flag: 'wx',
        mode: 0o600,
      }),
      writeFile(join(temporary, CHECKPOINT_FILES.trainingLog), serializeStableJson(trainingLog), {
        flag: 'wx',
        mode: 0o600,
      }),
      writeFile(join(temporary, CHECKPOINT_FILES.weights), encoded.bytes, {
        flag: 'wx',
        mode: 0o600,
      }),
      ...(resumeState === null
        ? []
        : [
            writeFile(
              join(temporary, CHECKPOINT_FILES.trainerState),
              serializeStableJson(serializeTrainerState(resumeState)),
              { flag: 'wx', mode: 0o600 },
            ),
          ]),
    ]);
    const validated = await loadCheckpoint(temporary);
    await rename(temporary, destination);
    return validated;
  } catch (error) {
    await rm(temporary, { recursive: true, force: true });
    throw error;
  }
}
