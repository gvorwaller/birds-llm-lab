import type { ModelConfig, WeightIndexArtifact } from '../data/schemas';
import type { AdamWHyperparameters, AdamWSnapshot } from '../model/optimizer';

export const LIVE_TRAINING_PROTOCOL_VERSION = 1 as const;

export interface LiveTrainingConfig {
  readonly model: ModelConfig;
  /** Tokenized training sequences. */
  readonly sequences: readonly (readonly number[])[];
  readonly validationSequences?: readonly (readonly number[])[];
  readonly samplePromptIds?: readonly number[];
  readonly sampleNewTokens?: number;
  readonly selectedWeightName?: string;
  readonly seed: number;
  readonly batchSize: number;
  readonly totalSteps: number;
  /** Scalar progress and explicit checkpoint transfers are limited to this step cadence. */
  readonly displayEvery: number;
  readonly baseLearningRate: number;
  readonly warmupSteps: number;
  readonly optimizer: AdamWHyperparameters;
}

export interface LiveTrainingCheckpoint {
  readonly formatVersion: 1;
  readonly trainingIdentity: string;
  readonly step: number;
  readonly epoch: number;
  readonly batchIndex: number;
  readonly weightIndex: WeightIndexArtifact;
  readonly weightBytes: Uint8Array;
  readonly optimizer: AdamWSnapshot;
}

interface CommandBase {
  readonly version: typeof LIVE_TRAINING_PROTOCOL_VERSION;
  readonly runId: string;
}

export type LiveTrainingCommand =
  | (CommandBase & {
      readonly type: 'start';
      readonly config: LiveTrainingConfig;
      readonly startPaused?: boolean;
      readonly resumeFrom?: LiveTrainingCheckpoint;
    })
  | (CommandBase & { readonly type: 'pause' | 'resume' | 'step' | 'cancel' | 'checkpoint' });

export type LiveTrainingState = 'running' | 'paused' | 'cancelled' | 'completed' | 'error';

export interface SelectedWeightSnapshot {
  readonly name: string;
  readonly shape: readonly [number, number];
  readonly values: readonly number[];
  readonly histogramMinimum: number;
  readonly histogramMaximum: number;
  readonly histogramCounts: readonly number[];
}

export type LiveTrainingReply =
  | (CommandBase & {
      readonly type: 'status';
      readonly state: LiveTrainingState;
      readonly step: number;
      readonly totalSteps: number;
    })
  | (CommandBase & {
      readonly type: 'progress';
      readonly step: number;
      readonly totalSteps: number;
      readonly trainLoss: number;
      readonly validationLoss: number | null;
      readonly predictionCount: number;
      readonly learningRate: number;
      readonly gradientNorm: number;
      readonly sampleTokenIds: readonly number[] | null;
      readonly selectedWeight: SelectedWeightSnapshot | null;
    })
  | (CommandBase & {
      readonly type: 'checkpoint';
      readonly checkpoint: LiveTrainingCheckpoint;
    })
  | (CommandBase & {
      readonly type: 'error';
      readonly code:
        | 'invalid-message'
        | 'unsupported-version'
        | 'invalid-state'
        | 'invalid-config'
        | 'checkpoint-cadence'
        | 'training-failed';
      readonly message: string;
      readonly recoverable: boolean;
    });
