<script lang="ts">
  import { onMount } from 'svelte';
  import {
    createLiveTrainingWorker,
    onLiveTrainingReply,
    sendLiveTrainingCommand,
  } from '../lib/training/worker-client';
  import {
    LIVE_TRAINING_PROTOCOL_VERSION as VERSION,
    type LiveTrainingCheckpoint,
    type LiveTrainingCommand,
    type LiveTrainingConfig,
    type LiveTrainingReply,
    type LiveTrainingState,
  } from '../lib/training/worker-protocol';

  const fixture: Omit<LiveTrainingConfig, 'displayEvery'> = {
    model: {
      formatVersion: 1,
      vocabSize: 259,
      contextLength: 4,
      dModel: 4,
      nLayers: 2,
      nHeads: 2,
      dHead: 2,
      dMlp: 6,
      tiedEmbeddings: true,
      useBias: true,
      layerNormEpsilon: 1e-5,
      gelu: 'tanh-approximation',
    },
    sequences: [
      [256, 97, 98, 99, 257],
      [256, 98, 99, 100, 257],
      [256, 99, 100, 101, 257],
    ],
    seed: 123,
    batchSize: 2,
    totalSteps: 1_000,
    baseLearningRate: 0.001,
    warmupSteps: 10,
    optimizer: {
      beta1: 0.9,
      beta2: 0.95,
      epsilon: 1e-8,
      weightDecay: 0.1,
      gradientClipNorm: 1,
    },
  };

  let worker: Worker | null = null;
  let unsubscribe: (() => void) | null = null;
  let runId = '';
  let displayEvery = $state(25);
  let workerState = $state<LiveTrainingState | 'idle'>('idle');
  let step = $state(0);
  let progress = $state<Extract<LiveTrainingReply, { type: 'progress' }> | null>(null);
  let checkpoint = $state.raw<LiveTrainingCheckpoint | null>(null);
  let lastError = $state('');
  let uiTicks = $state(0);

  function ensureWorker(): Worker {
    if (worker) return worker;
    worker = createLiveTrainingWorker();
    unsubscribe = onLiveTrainingReply(worker, (reply) => {
      if (reply.runId !== runId) return;
      if (reply.type === 'status') {
        workerState = reply.state;
        step = reply.step;
      } else if (reply.type === 'progress') {
        progress = reply;
        step = reply.step;
      } else if (reply.type === 'checkpoint') {
        checkpoint = reply.checkpoint;
      } else {
        lastError = `${reply.code}: ${reply.message}`;
      }
    });
    worker.addEventListener('error', () => {
      lastError = 'The training worker stopped unexpectedly.';
      workerState = 'error';
    });
    return worker;
  }

  function send(command: LiveTrainingCommand): void {
    lastError = '';
    sendLiveTrainingCommand(ensureWorker(), command);
  }

  function start(resumeFrom?: LiveTrainingCheckpoint): void {
    runId = crypto.randomUUID();
    checkpoint = null;
    progress = null;
    step = resumeFrom?.step ?? 0;
    workerState = 'idle';
    send({
      version: VERSION,
      runId,
      type: 'start',
      config: { ...fixture, displayEvery },
      ...(resumeFrom ? { resumeFrom, startPaused: true } : {}),
    });
  }

  function command(type: Exclude<LiveTrainingCommand['type'], 'start'>): void {
    send({ version: VERSION, runId, type });
  }

  onMount(() => {
    const timer = window.setInterval(() => (uiTicks += 1), 100);
    return () => {
      window.clearInterval(timer);
      unsubscribe?.();
      worker?.terminate();
    };
  });
</script>

<section class="protocol" aria-label="Live training worker protocol">
  <div class="lab-panel">
    <p class="eyebrow">Worker protocol · synthetic fixture</p>
    <h2>Train off the main thread</h2>
    <p>
      This small, synthetic token set exercises the worker controls. It does not use the bird corpus
      or the checkpoint selected in the sidebar. The live training charts and corpus preset come
      next.
    </p>
    <p class="fixture">
      Three five-token sequences · 259-token vocabulary · two layers, four channels · 1,000
      optimizer steps
    </p>
    <label for="display-cadence">Display cadence (steps)</label>
    <input
      id="display-cadence"
      type="number"
      min="1"
      max="1000"
      step="1"
      bind:value={displayEvery}
      disabled={workerState === 'running' || workerState === 'paused'}
    />
    <div class="actions">
      <button
        onclick={() => start()}
        disabled={workerState === 'running' || workerState === 'paused'}
        >Start fixture training</button
      >
      <button onclick={() => command('pause')} disabled={workerState !== 'running'}>Pause</button>
      <button onclick={() => command('resume')} disabled={workerState !== 'paused'}>Resume</button>
      <button onclick={() => command('step')} disabled={workerState !== 'paused'}>One batch</button>
      <button
        onclick={() => command('cancel')}
        disabled={workerState !== 'running' && workerState !== 'paused'}>Cancel</button
      >
      <button
        onclick={() => command('checkpoint')}
        disabled={workerState === 'idle' || workerState === 'error'}>Checkpoint</button
      >
      <button
        onclick={() => checkpoint && start(checkpoint)}
        disabled={!checkpoint || workerState === 'running'}>Restore saved checkpoint</button
      >
    </div>
  </div>

  <div class="lab-panel" aria-live="polite">
    <h2>Worker state</h2>
    <dl>
      <div>
        <dt>State</dt>
        <dd data-testid="worker-state">{workerState}</dd>
      </div>
      <div>
        <dt>Completed batches</dt>
        <dd data-testid="worker-step">{step} / {fixture.totalSteps}</dd>
      </div>
      <div>
        <dt>Main-thread heartbeat</dt>
        <dd data-testid="ui-heartbeat">{uiTicks}</dd>
      </div>
    </dl>
    {#if progress}
      <p>
        Last displayed batch {progress.step}: loss {progress.trainLoss.toFixed(6)}, learning rate {progress.learningRate.toExponential(
          3,
        )}, gradient norm {progress.gradientNorm.toFixed(6)}.
      </p>
    {/if}
    {#if checkpoint}
      <p data-testid="checkpoint-summary">
        Checkpoint at batch {checkpoint.step} · {checkpoint.weightBytes.byteLength} weight bytes · optimizer
        state saved.
      </p>
    {/if}
    {#if lastError}<p class="error" role="alert">{lastError}</p>{/if}
    <p class="note">
      Progress sends scalar values at the selected cadence. Full weights and optimizer moments
      transfer only when you request a checkpoint, with the same cadence limit.
    </p>
  </div>
</section>

<style>
  .protocol {
    display: grid;
    gap: 1.25rem;
    max-width: 1100px;
  }
  .protocol h2 {
    margin: 0 0 0.8rem;
  }
  .protocol p {
    line-height: 1.55;
  }
  .fixture,
  .note {
    color: var(--muted);
    font-size: 0.83rem;
  }
  label {
    display: block;
    margin-top: 1.5rem;
    font-weight: 700;
  }
  input {
    width: 9rem;
    margin-top: 0.4rem;
    padding: 0.5rem;
    border: 1px solid var(--line-strong);
    color: var(--ink);
    background: var(--paper);
    font: inherit;
  }
  .actions {
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem;
    margin-top: 1.2rem;
  }
  button {
    min-height: 2.4rem;
    padding: 0.5rem 0.7rem;
    border: 1px solid var(--forest);
    color: var(--ink);
    background: var(--paper);
    cursor: pointer;
  }
  button:disabled {
    cursor: default;
    opacity: 0.45;
  }
  dl {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
  }
  dl div {
    min-width: 150px;
    padding: 0.7rem;
    border: 1px solid var(--line);
  }
  dt {
    color: var(--muted);
    font-size: 0.75rem;
  }
  dd {
    margin: 0.2rem 0 0;
    font:
      0.9rem ui-monospace,
      SFMono-Regular,
      Menlo,
      monospace;
  }
  .error {
    color: #af421c;
  }
</style>
