<script lang="ts">
  import { onMount } from 'svelte';
  import type {
    ExportJobState,
    ServiceStatus,
    TrainingJobState,
    TrainingPresetId,
  } from '../lib/data/api-types';
  import {
    cancelExportJob,
    cancelTrainingJob,
    getCheckpoints,
    getCorpusManifest,
    getExportJob,
    getServiceStatus,
    getTrainingJob,
    selectCheckpoint,
    startCorpusExport,
    startTraining,
  } from '../lib/data/api';
  import type { CorpusManifest } from '../lib/data/schemas';
  import { checkpointCatalog } from '../lib/stores/checkpoints';

  let service = $state<ServiceStatus | null>(null);
  let manifest = $state<CorpusManifest | null>(null);
  let job = $state<ExportJobState | null>(null);
  let trainingJob = $state<TrainingJobState | null>(null);
  let trainingPreset = $state<TrainingPresetId>('quick');
  let busy = $state(true);
  let error = $state<string | null>(null);
  let exportPollTimer: ReturnType<typeof setTimeout> | undefined;
  let trainingPollTimer: ReturnType<typeof setTimeout> | undefined;

  const active = $derived(job !== null && ['queued', 'running', 'cancelling'].includes(job.status));
  const trainingActive = $derived(
    trainingJob !== null && ['queued', 'running', 'cancelling'].includes(trainingJob.status),
  );
  const progressValue = $derived(
    job?.progress.totalRows && job.progress.totalRows > 0
      ? Math.min(100, (job.progress.processedRows / job.progress.totalRows) * 100)
      : 0,
  );
  const trainingProgress = $derived(
    trainingJob && trainingJob.progress.totalSteps > 0
      ? Math.min(100, (trainingJob.progress.step / trainingJob.progress.totalSteps) * 100)
      : 0,
  );

  function message(value: unknown): string {
    return value instanceof Error ? value.message : 'The local service returned an unknown error.';
  }

  function schedulePoll(): void {
    if (!active) return;
    clearTimeout(exportPollTimer);
    exportPollTimer = setTimeout(() => void refreshJob(), 350);
  }

  function scheduleTrainingPoll(): void {
    if (!trainingActive) return;
    clearTimeout(trainingPollTimer);
    trainingPollTimer = setTimeout(() => void refreshTrainingJob(), 500);
  }

  async function refreshTrainingJob(): Promise<void> {
    if (!trainingJob) return;
    try {
      trainingJob = await getTrainingJob(trainingJob.id);
      if (trainingJob.status === 'succeeded') checkpointCatalog.set(await getCheckpoints());
    } catch (cause) {
      error = message(cause);
    } finally {
      scheduleTrainingPoll();
    }
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
      checkpointCatalog.set(await getCheckpoints());
      if (service.latestExportJobId) {
        job = await getExportJob(service.latestExportJobId);
        schedulePoll();
      }
      if (service.latestTrainingJobId) {
        trainingJob = await getTrainingJob(service.latestTrainingJobId);
        scheduleTrainingPoll();
      }
    } catch (cause) {
      error = message(cause);
    } finally {
      busy = false;
    }
  }

  async function startTrainingJob(): Promise<void> {
    error = null;
    try {
      trainingJob = await startTraining(trainingPreset);
      scheduleTrainingPoll();
    } catch (cause) {
      error = message(cause);
      await load();
    }
  }

  async function cancelTraining(): Promise<void> {
    if (!trainingJob) return;
    error = null;
    try {
      trainingJob = await cancelTrainingJob(trainingJob.id);
      scheduleTrainingPoll();
    } catch (cause) {
      error = message(cause);
    }
  }

  async function activateCheckpoint(id: string): Promise<void> {
    error = null;
    try {
      checkpointCatalog.set(await selectCheckpoint(id));
    } catch (cause) {
      error = message(cause);
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

  function formatDuration(value: number | null): string {
    if (value === null) return 'Calculating…';
    const seconds = Math.max(0, Math.round(value / 1_000));
    const minutes = Math.floor(seconds / 60);
    return `${minutes}m ${seconds % 60}s`;
  }

  onMount(() => {
    void load();
    return () => {
      clearTimeout(exportPollTimer);
      clearTimeout(trainingPollTimer);
    };
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

<section class="workbench" aria-labelledby="training-heading">
  <div class="section-heading">
    <div>
      <p class="eyebrow">From-scratch transformer</p>
      <h2 id="training-heading">Training & checkpoints</h2>
    </div>
    <span class:online={!trainingActive} class="service-state">
      {trainingActive ? trainingJob?.status : 'Ready'}
    </span>
  </div>
  <p class="explanation">
    Train with one of two bounded presets. The quick run proves the complete workflow; the ready run
    uses the verified 2,000-step schedule. Only atomically completed, validated checkpoints appear
    below.
  </p>

  {#if $checkpointCatalog.recoveryWarning}
    <p class="notice warning" role="status">{$checkpointCatalog.recoveryWarning}</p>
  {/if}

  <div class="training-controls">
    <label>
      <span class="control-label">Training preset</span>
      <select bind:value={trainingPreset} disabled={trainingActive}>
        <option value="quick">Quick proof · 100 steps</option>
        <option value="ready">Ready model · 2,000 steps</option>
      </select>
    </label>
    <button
      class="primary"
      disabled={busy || trainingActive}
      onclick={() => void startTrainingJob()}
    >
      {trainingActive ? 'Training in progress…' : 'Train new checkpoint'}
    </button>
    {#if trainingActive}
      <button
        class="secondary"
        disabled={trainingJob?.status === 'cancelling'}
        onclick={() => void cancelTraining()}
      >
        {trainingJob?.status === 'cancelling' ? 'Cancelling safely…' : 'Cancel safely'}
      </button>
    {/if}
  </div>

  {#if trainingJob}
    <div class="job-panel training-job" aria-live="polite">
      <div class="job-line">
        <strong>Latest training · {trainingJob.preset}</strong>
        <span class:failed={['failed', 'interrupted'].includes(trainingJob.status)}
          >{trainingJob.status}</span
        >
      </div>
      <div
        class="progress-track"
        role="progressbar"
        aria-label="Model training progress"
        aria-valuemin="0"
        aria-valuemax="100"
        aria-valuenow={Math.round(trainingProgress)}
      >
        <span style:width={`${trainingProgress}%`}></span>
      </div>
      <dl class="training-metrics">
        <div>
          <dt>Step</dt>
          <dd>{trainingJob.progress.step} / {trainingJob.progress.totalSteps}</dd>
        </div>
        <div>
          <dt>Train loss</dt>
          <dd>{trainingJob.progress.trainLoss?.toFixed(4) ?? '—'}</dd>
        </div>
        <div>
          <dt>Validation loss</dt>
          <dd>{trainingJob.progress.validationLoss?.toFixed(4) ?? '—'}</dd>
        </div>
        <div>
          <dt>Learning rate</dt>
          <dd>{trainingJob.progress.learningRate.toExponential(2)}</dd>
        </div>
        <div>
          <dt>Elapsed</dt>
          <dd>{formatDuration(trainingJob.progress.elapsedMs)}</dd>
        </div>
        <div>
          <dt>Estimated left</dt>
          <dd>{formatDuration(trainingJob.progress.estimatedRemainingMs)}</dd>
        </div>
      </dl>
      <p class="job-detail">
        Cancellation: {trainingJob.status === 'cancelling'
          ? 'requested'
          : trainingActive
            ? 'available between optimizer steps'
            : 'not active'}
      </p>
      {#if trainingJob.progress.latestSample}
        <div class="sample">
          <span>Latest seeded sample</span>
          <pre>{trainingJob.progress.latestSample}</pre>
        </div>
      {/if}
      {#if trainingJob.error}<p class="job-error">{trainingJob.error}</p>{/if}
    </div>
  {/if}

  <div class="checkpoint-heading">
    <h3>Validated checkpoints</h3>
    <span>{$checkpointCatalog.checkpoints.length} available</span>
  </div>
  {#if $checkpointCatalog.checkpoints.length === 0}
    <p class="empty-state">No validated checkpoint is available yet.</p>
  {:else}
    <div class="checkpoint-list">
      {#each $checkpointCatalog.checkpoints as checkpoint}
        <article class:active-checkpoint={$checkpointCatalog.activeCheckpointId === checkpoint.id}>
          <div>
            <strong>{checkpoint.id}</strong>
            <small>step {checkpoint.trainingStep} / {checkpoint.totalSteps}</small>
          </div>
          <dl>
            <div>
              <dt>Train loss</dt>
              <dd>{checkpoint.finalTrainLoss?.toFixed(4) ?? '—'}</dd>
            </div>
            <div>
              <dt>Validation</dt>
              <dd>{checkpoint.finalValidationLoss?.toFixed(4) ?? '—'}</dd>
            </div>
          </dl>
          <button
            class="secondary"
            disabled={$checkpointCatalog.activeCheckpointId === checkpoint.id}
            onclick={() => void activateCheckpoint(checkpoint.id)}
          >
            {$checkpointCatalog.activeCheckpointId === checkpoint.id ? 'Active' : 'Use checkpoint'}
          </button>
        </article>
      {/each}
    </div>
  {/if}
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

  .training-controls {
    display: flex;
    flex-wrap: wrap;
    align-items: end;
    gap: 0.75rem;
    margin-top: 1.5rem;
  }

  .training-controls label {
    min-width: 240px;
  }

  .control-label {
    display: block;
    margin-bottom: 0.4rem;
    color: var(--muted);
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
  }

  select {
    width: 100%;
    min-height: 44px;
    padding: 0.6rem;
    border: 1px solid var(--line-strong);
    color: var(--ink);
    background: var(--paper);
  }

  .training-metrics {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 1px;
    margin: 1rem 0;
    background: var(--line);
  }

  .training-metrics div {
    min-width: 0;
    padding: 0.75rem;
    background: var(--paper);
  }

  .training-metrics dd {
    overflow-wrap: anywhere;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.85rem;
  }

  .sample {
    margin-top: 1rem;
    padding: 0.8rem;
    border-left: 3px solid var(--lichen);
    background: var(--paper);
  }

  .sample span {
    color: var(--muted);
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
  }

  .sample pre {
    margin: 0.6rem 0 0;
    overflow: auto;
    white-space: pre-wrap;
    font:
      0.78rem/1.5 ui-monospace,
      SFMono-Regular,
      Menlo,
      monospace;
  }

  .checkpoint-heading {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 1rem;
    margin-top: 2rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--line);
  }

  .checkpoint-heading h3 {
    margin: 0;
    font-family: Georgia, 'Times New Roman', serif;
    font-size: 1.35rem;
    font-weight: 500;
  }

  .checkpoint-heading span {
    color: var(--muted);
    font-size: 0.75rem;
  }

  .checkpoint-list {
    display: grid;
    gap: 0.65rem;
    margin-top: 1rem;
  }

  .checkpoint-list article {
    display: grid;
    grid-template-columns: minmax(180px, 1fr) minmax(220px, 1fr) auto;
    align-items: center;
    gap: 1rem;
    padding: 0.9rem;
    border: 1px solid var(--line);
    background: var(--paper);
  }

  .checkpoint-list article.active-checkpoint {
    border-color: var(--forest);
    box-shadow: inset 4px 0 var(--forest);
  }

  .checkpoint-list strong,
  .checkpoint-list small {
    display: block;
  }

  .checkpoint-list small {
    margin-top: 0.25rem;
    color: var(--muted);
  }

  .checkpoint-list dl {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.75rem;
    margin: 0;
  }

  .checkpoint-list dd {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.8rem;
  }

  @media (max-width: 1150px) {
    .checkpoint-list article {
      grid-template-columns: 1fr auto;
    }

    .checkpoint-list dl {
      grid-column: 1 / -1;
      grid-row: 2;
    }
  }
</style>
