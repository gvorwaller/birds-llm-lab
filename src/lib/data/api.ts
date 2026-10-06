import type {
  CheckpointList,
  CheckpointInspectionBundle,
  EvidenceSearchResponse,
  OrderCooccurrenceEvidence,
  ExportJobState,
  ServiceStatus,
  TrainingJobState,
  TrainingPresetId,
} from './api-types';
import { parseCorpusManifest, type CorpusManifest } from './schemas';

function csrfToken(): string {
  const value = document.querySelector<HTMLMetaElement>('meta[name="birds-llm-lab-csrf"]')?.content;
  if (!value || value.startsWith('__BIRDS_')) {
    throw new Error(
      'The local request token is unavailable. Reload this page from the installed service.',
    );
  }
  return value;
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  if (init?.method === 'POST') headers.set('X-Birds-LLM-Lab-CSRF', csrfToken());
  const response = await fetch(path, { ...init, headers, credentials: 'same-origin' });
  const payload = (await response.json()) as T | { error?: string };
  if (!response.ok) {
    const message =
      'error' in (payload as object) ? (payload as { error?: string }).error : undefined;
    throw new Error(message ?? `The local service returned HTTP ${response.status}.`);
  }
  return payload as T;
}

export function getServiceStatus(): Promise<ServiceStatus> {
  return requestJson('/api/status');
}

export function searchCorpusEvidence(query: string, page = 0): Promise<EvidenceSearchResponse> {
  const parameters = new URLSearchParams({ q: query, page: String(page) });
  return requestJson(`/api/evidence/search?${parameters}`);
}

export function getVerifiedExampleEvidence(): Promise<OrderCooccurrenceEvidence> {
  return requestJson('/api/evidence/verified-example');
}

export async function getCorpusManifest(): Promise<CorpusManifest | null> {
  const response = await fetch('/api/corpus/manifest', { credentials: 'same-origin' });
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(`The local service returned HTTP ${response.status}.`);
  return parseCorpusManifest((await response.json()) as unknown);
}

export function startCorpusExport(): Promise<ExportJobState> {
  return requestJson('/api/corpus/export', { method: 'POST' });
}

export function getExportJob(id: string): Promise<ExportJobState> {
  return requestJson(`/api/jobs/${encodeURIComponent(id)}`);
}

export function cancelExportJob(id: string): Promise<ExportJobState> {
  return requestJson(`/api/jobs/${encodeURIComponent(id)}/cancel`, { method: 'POST' });
}

export function getCheckpoints(): Promise<CheckpointList> {
  return requestJson('/api/checkpoints');
}

export function getActiveCheckpointForInspection(): Promise<CheckpointInspectionBundle> {
  return requestJson('/api/checkpoints/active/inspect');
}

export function selectCheckpoint(id: string): Promise<CheckpointList> {
  return requestJson(`/api/checkpoints/${encodeURIComponent(id)}/select`, { method: 'POST' });
}

export function startTraining(preset: TrainingPresetId): Promise<TrainingJobState> {
  return requestJson('/api/training/jobs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ preset }),
  });
}

export function getTrainingJob(id: string): Promise<TrainingJobState> {
  return requestJson(`/api/jobs/${encodeURIComponent(id)}`);
}

export function cancelTrainingJob(id: string): Promise<TrainingJobState> {
  return requestJson(`/api/jobs/${encodeURIComponent(id)}/cancel`, { method: 'POST' });
}
