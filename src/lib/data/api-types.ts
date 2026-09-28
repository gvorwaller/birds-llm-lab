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
  recoveryWarning: string | null;
}

export interface CheckpointList {
  checkpoints: [];
  activeCheckpointId: null;
  availableInMilestone: 2;
}
