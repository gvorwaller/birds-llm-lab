<script lang="ts">
  import { onMount } from 'svelte';
  import type { ExportJobState, ServiceStatus } from '../lib/data/api-types';
  import {
    cancelExportJob,
    getCorpusManifest,
    getExportJob,
    getServiceStatus,
    startCorpusExport,
  } from '../lib/data/api';
  import type { CorpusManifest } from '../lib/data/schemas';

  let service = $state<ServiceStatus | null>(null);
  let manifest = $state<CorpusManifest | null>(null);
  let job = $state<ExportJobState | null>(null);
  let busy = $state(true);
  let error = $state<string | null>(null);
  let pollTimer: ReturnType<typeof setTimeout> | undefined;

  const active = $derived(job !== null && ['queued', 'running', 'cancelling'].includes(job.status));
  const progressValue = $derived(
    job?.progress.totalRows && job.progress.totalRows > 0
      ? Math.min(100, (job.progress.processedRows / job.progress.totalRows) * 100)
      : 0,
  );

  function message(value: unknown): string {
    return value instanceof Error ? value.message : 'The local service returned an unknown error.';
  }

  function schedulePoll(): void {
    if (!active) return;
    clearTimeout(pollTimer);
    pollTimer = setTimeout(() => void refreshJob(), 350);
  }

  async function refreshJob(): Promise<void> {
    if (!job) return;
    try {
      job = await getExportJob(job.id);
      if (job.status === 'succeeded') {
        manifest = job.manifest ?? (await getCorpusManifest());
        service = await getServiceStatus();
      }
    } catch (cause) {
      error = message(cause);
    } finally {
      schedulePoll();
    }
  }

  async function load(): Promise<void> {
    busy = true;
    error = null;
    try {
      service = await getServiceStatus();
      manifest = await getCorpusManifest();
      if (service.latestExportJobId) {
        job = await getExportJob(service.latestExportJobId);
        schedulePoll();
      }
    } catch (cause) {
      error = message(cause);
    } finally {
      busy = false;
    }
  }

  async function startExport(): Promise<void> {
    error = null;
    try {
      job = await startCorpusExport();
      schedulePoll();
    } catch (cause) {
      error = message(cause);
      await load();
    }
  }

  async function cancelExport(): Promise<void> {
    if (!job) return;
    error = null;
    try {
      job = await cancelExportJob(job.id);
      schedulePoll();
    } catch (cause) {
      error = message(cause);
    }
  }

  function formatNumber(value: number): string {
    return new Intl.NumberFormat('en-US').format(value);
  }

  function formatBytes(value: number): string {
    return `${(value / 1_000_000).toFixed(2)} MB`;
  }

  onMount(() => {
    void load();
    return () => clearTimeout(pollTimer);
  });
</script>

<section class="workbench" aria-labelledby="corpus-heading">
  <div class="section-heading">
    <div>
      <p class="eyebrow">Local database → validated JSONL</p>
      <h2 id="corpus-heading">Corpus export</h2>
    </div>
    <span class:online={service?.status === 'ok'} class="service-state">
      {service?.status === 'ok' ? 'Service online' : 'Checking service'}
    </span>
  </div>

  <p class="explanation">
    This action reads the fixed local <code>birds_test</code> database inside a verified read-only transaction.
    It cannot target another host or database.
  </p>

  {#if service?.recoveryWarning}
    <p class="notice warning" role="status">{service.recoveryWarning}</p>
  {/if}
  {#if error}
    <p class="notice error" role="alert">{error}</p>
  {/if}

  <div class="action-row">
    <button class="primary" disabled={busy || active} onclick={() => void startExport()}>
      {active ? 'Export in progress…' : 'Export corpus from local birds DB'}
    </button>
    {#if active}
      <button
        class="secondary"
        disabled={job?.status === 'cancelling'}
        onclick={() => void cancelExport()}
      >
        {job?.status === 'cancelling' ? 'Cancelling…' : 'Cancel safely'}
      </button>
    {/if}
  </div>

  {#if job}
    <div class="job-panel" aria-live="polite">
      <div class="job-line">
        <strong>Latest export</strong>
        <span class:failed={job.status === 'failed'}>{job.status}</span>
      </div>
      <div
        class="progress-track"
        role="progressbar"
        aria-label="Corpus export progress"
        aria-valuemin="0"
        aria-valuemax="100"
        aria-valuenow={Math.round(progressValue)}
      >
        <span style:width={`${progressValue}%`}></span>
      </div>
      <p class="job-detail">
        {job.progress.phase} · {formatNumber(job.progress.processedRows)}{job.progress.totalRows
          ? ` / ${formatNumber(job.progress.totalRows)}`
          : ''}
      </p>
      {#if job.error}<p class="job-error">{job.error}</p>{/if}
    </div>
  {/if}
</section>

<section class="workbench" aria-labelledby="manifest-heading">
  <div class="section-heading">
    <div>
      <p class="eyebrow">Inspectable artifact</p>
      <h2 id="manifest-heading">Corpus manifest</h2>
    </div>
    <span class="service-state">{manifest ? 'Available' : 'Not exported'}</span>
  </div>

  {#if manifest}
    <dl class="metrics">
      <div>
        <dt>Source rows</dt>
        <dd>{formatNumber(manifest.sourceRows)}</dd>
      </div>
      <div>
        <dt>Emitted rows</dt>
        <dd>{formatNumber(manifest.emittedRows)}</dd>
      </div>
      <div>
        <dt>Corpus size</dt>
        <dd>{formatBytes(manifest.corpusBytes)}</dd>
      </div>
      <div>
        <dt>Skipped rows</dt>
        <dd>{formatNumber(manifest.skippedRows)}</dd>
      </div>
    </dl>
    <div class="split-grid">
      {#each ['train', 'validation', 'test'] as split}
        <div>
          <span>{split}</span>
          <strong
            >{formatNumber(
              manifest.splits[split as keyof typeof manifest.splits].documents,
            )}</strong
          >
          <small>{formatBytes(manifest.splits[split as keyof typeof manifest.splits].bytes)}</small>
        </div>
      {/each}
    </div>
    <dl class="manifest-details">
      <div>
        <dt>Exported</dt>
        <dd>{new Date(manifest.exportedAt).toLocaleString()}</dd>
      </div>
      <div>
        <dt>SHA-256</dt>
        <dd><code>{manifest.corpusSha256}</code></dd>
      </div>
      <div>
        <dt>Split contract</dt>
        <dd>{manifest.splitVersion} · {manifest.splitAlgorithm}</dd>
      </div>
      <div>
        <dt>Database</dt>
        <dd>{manifest.database.host}:{manifest.database.port}/{manifest.database.name}</dd>
      </div>
    </dl>
    <p class="source-note">
      Wikipedia extracts and sections are included in training documents. AI-generated field craft
      is exported and measured, but excluded from default training and demos.
    </p>
  {:else if busy}
    <p class="empty-state">Reading the local manifest…</p>
  {:else}
    <p class="empty-state">No corpus has been exported yet. Use the button above to create one.</p>
  {/if}
</section>

<section class="workbench future" aria-labelledby="training-heading">
  <div class="section-heading">
    <div>
      <p class="eyebrow">Milestone 2</p>
      <h2 id="training-heading">Training & checkpoints</h2>
    </div>
    <span class="service-state">Not available yet</span>
  </div>
  <p class="explanation">
    The controls are intentionally disabled until the from-scratch model, optimizer, and checkpoint
    validators pass their mathematical gates. This page will remain the owner-facing control
    surface.
  </p>
  <button class="secondary" disabled>Train new checkpoint</button>
</section>

<style>
  .workbench {
    max-width: 980px;
    margin-top: 3rem;
    padding: 2rem;
    border: 1px solid var(--line);
    background: var(--paper-raised);
  }

  .section-heading,
  .action-row,
  .job-line {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
  }

  .section-heading h2 {
    margin: 0;
    font-size: 2rem;
  }

  .section-heading .eyebrow {
    margin-bottom: 0.45rem;
  }

  .service-state {
    padding: 0.35rem 0.55rem;
    border: 1px solid var(--line-strong);
    color: var(--muted);
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
  }

  .service-state.online {
    border-color: var(--forest);
    color: var(--forest);
  }

  .explanation,
  .source-note,
  .empty-state {
    max-width: 760px;
    color: var(--muted);
    line-height: 1.65;
  }

  .action-row {
    justify-content: flex-start;
    margin-top: 1.5rem;
  }

  button {
    min-height: 44px;
    padding: 0.75rem 1rem;
    border: 1px solid var(--forest-deep);
    cursor: pointer;
  }

  button.primary {
    color: #fff;
    background: var(--forest-deep);
  }

  button.secondary {
    color: var(--ink);
    background: transparent;
  }

  button:disabled {
    cursor: not-allowed;
    opacity: 0.58;
  }

  .notice {
    padding: 0.8rem 1rem;
    border-left: 4px solid var(--signal);
    background: rgb(213 108 63 / 0.1);
  }

  .notice.error,
  .job-error,
  .failed {
    color: #a23e2d;
  }

  .job-panel {
    margin-top: 1.5rem;
    padding: 1rem;
    border: 1px solid var(--line);
  }

  .progress-track {
    height: 8px;
    margin-top: 0.9rem;
    overflow: hidden;
    background: var(--line);
  }

  .progress-track span {
    display: block;
    height: 100%;
    background: var(--forest);
  }

  .job-detail,
  .job-error {
    margin: 0.65rem 0 0;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.76rem;
  }

  .metrics,
  .split-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    margin: 1.5rem 0;
    border: 1px solid var(--line);
    background: var(--line);
  }

  .metrics div,
  .split-grid div {
    padding: 1rem;
    background: var(--paper);
  }

  dt,
  .split-grid span,
  .split-grid small {
    color: var(--muted);
    font-size: 0.72rem;
    text-transform: uppercase;
  }

  dd {
    margin: 0.35rem 0 0;
  }

  .metrics dd,
  .split-grid strong {
    display: block;
    font-family: Georgia, 'Times New Roman', serif;
    font-size: 1.45rem;
  }

  .split-grid {
    grid-template-columns: repeat(3, 1fr);
  }

  .split-grid small {
    display: block;
    margin-top: 0.3rem;
  }

  .manifest-details {
    display: grid;
    gap: 0.75rem;
  }

  .manifest-details div {
    display: grid;
    grid-template-columns: 130px minmax(0, 1fr);
    gap: 1rem;
  }

  code {
    overflow-wrap: anywhere;
    font-size: 0.78rem;
  }

  .source-note {
    margin-top: 1.5rem;
    padding-top: 1rem;
    border-top: 1px solid var(--line);
  }

  .future {
    margin-bottom: 4rem;
    border-style: dashed;
  }
</style>
