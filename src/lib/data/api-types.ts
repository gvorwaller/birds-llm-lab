import type { CorpusManifest } from './schemas';

export type ExportJobStatus =
  'queued' | 'running' | 'cancelling' | 'cancelled' | 'succeeded' | 'failed' | 'interrupted';

export interface ExportJobProgress {
  phase:
    'queued' | 'connecting' | 'reading' | 'normalizing' | 'writing' | 'committing' | 'complete';
  processedRows: number;
  totalRows: number | null;
}

export interface ExportJobState {
  id: string;
  kind: 'corpus-export';
  status: ExportJobStatus;
  createdAt: string;
  updatedAt: string;
  progress: ExportJobProgress;
  logs: string[];
  manifest: CorpusManifest | null;
  error: string | null;
}

export interface ServiceStatus {
  status: 'ok';
  version: 1;
  endpoint: '127.0.0.1:5301';
  corpusAvailable: boolean;
  activeExportJobId: string | null;
  latestExportJobId: string | null;
  activeTrainingJobId: string | null;
  latestTrainingJobId: string | null;
  recoveryWarning: string | null;
}

export type TrainingPresetId = 'quick' | 'ready';

export type TrainingJobStatus =
  'queued' | 'running' | 'cancelling' | 'cancelled' | 'succeeded' | 'failed' | 'interrupted';

export interface TrainingJobProgress {
  step: number;
  totalSteps: number;
  trainLoss: number | null;
  validationLoss: number | null;
  learningRate: number;
  elapsedMs: number;
  estimatedRemainingMs: number | null;
  latestSample: string | null;
}

export interface TrainingJobState {
  id: string;
  kind: 'model-training';
  preset: TrainingPresetId;
  status: TrainingJobStatus;
  createdAt: string;
  updatedAt: string;
  progress: TrainingJobProgress;
  logs: string[];
  checkpointId: string | null;
  error: string | null;
}

export interface CheckpointSummary {
  id: string;
  trainingStep: number;
  totalSteps: number;
  corpusSha256: string;
  tokenizerSha256: string;
  sourceGitRevision: string | null;
  finalTrainLoss: number | null;
  finalValidationLoss: number | null;
  modifiedAt: string;
}

export interface CheckpointList {
  checkpoints: CheckpointSummary[];
  activeCheckpointId: string | null;
  recoveryWarning: string | null;
}
