<script lang="ts">
  import { onMount } from 'svelte';
  import MetricPlot from './MetricPlot.svelte';
  import type { AdamWScalarTrace } from '../lib/model/optimizer';
  import { decodeTokenIds } from '../lib/tokenizer/codec';
  import { smallBirdNamePreset, SMALL_BIRD_NAME_PRESET_SUMMARY } from '../lib/training/live-preset';
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

  type Progress = Extract<LiveTrainingReply, { type: 'progress' }>;
  type PresetId = 'small' | 'protocol';
  const protocolFixture: Omit<LiveTrainingConfig, 'displayEvery'> = {
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
  let renderTimer: number | null = null;
  let queuedProgress: Progress[] = [];
  let runId = '';
  let presetId = $state<PresetId>('small');
  let displayEvery = $state(10);
  let workerState = $state<LiveTrainingState | 'idle'>('idle');
  let step = $state(0);
  let runConfig = $state.raw<LiveTrainingConfig | null>(null);
  let renderedProgress = $state.raw<Progress[]>([]);
  let latest = $state.raw<Progress | null>(null);
  let adamTrace = $state.raw<AdamWScalarTrace | null>(null);
  let checkpoint = $state.raw<LiveTrainingCheckpoint | null>(null);
  let lastError = $state('');
  let uiTicks = $state(0);

  const maximumStep = $derived(runConfig?.totalSteps ?? (presetId === 'small' ? 300 : 1_000));
  const lossSeries = $derived([
    {
      label: 'Train batch',
      color: '#356c4b',
      points: renderedProgress.map((item) => ({ step: item.step, value: item.trainLoss })),
    },
    {
      label: 'Validation split',
      color: '#c16b3d',
      points: renderedProgress
        .filter((item) => item.validationLoss !== null)
        .map((item) => ({ step: item.step, value: item.validationLoss as number })),
    },
  ]);
  const learningRateSeries = $derived([
    {
      label: 'Learning rate',
      color: '#356c4b',
      points: renderedProgress.map((item) => ({ step: item.step, value: item.learningRate })),
    },
  ]);
  const gradientSeries = $derived([
    {
      label: 'Gradient norm before clipping',
      color: '#965d95',
      points: renderedProgress.map((item) => ({ step: item.step, value: item.gradientNorm })),
    },
  ]);
  const weight = $derived(latest?.selectedWeight ?? null);
  const histogramPeak = $derived(weight ? Math.max(...weight.histogramCounts, 1) : 1);
  const matrixMagnitude = $derived(weight ? Math.max(...weight.values.map(Math.abs), 1e-9) : 1);
  const samples = $derived(
    renderedProgress
      .filter((item) => item.sampleTokenIds !== null)
      .slice(-5)
      .reverse(),
  );

  function matrixColor(value: number): string {
    const strength = Math.min(0.8, (Math.abs(value) / matrixMagnitude) * 0.8);
    return value >= 0 ? `rgba(42, 112, 77, ${strength})` : `rgba(197, 96, 58, ${strength})`;
  }

  function scalar(value: number): string {
    return value.toPrecision(8);
  }

  function flushProgress(): void {
    renderTimer = null;
    if (queuedProgress.length === 0) return;
    renderedProgress = [...renderedProgress, ...queuedProgress];
    latest = queuedProgress.at(-1) ?? latest;
    step = Math.max(step, latest?.step ?? step);
    queuedProgress = [];
  }

  function queueProgress(reply: Progress): void {
    queuedProgress.push(reply);
    if (renderTimer === null) renderTimer = window.setTimeout(flushProgress, 100);
  }

  function ensureWorker(): Worker {
    if (worker) return worker;
    worker = createLiveTrainingWorker();
    unsubscribe = onLiveTrainingReply(worker, (reply) => {
      if (reply.runId !== runId) return;
      if (reply.type === 'status') {
        workerState = reply.state;
        step = reply.step;
        if (reply.state === 'completed') flushProgress();
      } else if (reply.type === 'progress') {
        queueProgress(reply);
        if (reply.adamStep) adamTrace = reply.adamStep;
      } else if (reply.type === 'adam-step') {
        adamTrace = reply.trace;
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

  function selectPreset(): void {
    displayEvery = presetId === 'small' ? 10 : 25;
    checkpoint = null;
    renderedProgress = [];
    latest = null;
    adamTrace = null;
    runConfig = null;
    step = 0;
  }

  function start(resumeFrom?: LiveTrainingCheckpoint): void {
    runId = crypto.randomUUID();
    checkpoint = null;
    renderedProgress = [];
    latest = null;
    adamTrace = null;
    queuedProgress = [];
    if (renderTimer !== null) window.clearTimeout(renderTimer);
    renderTimer = null;
    step = resumeFrom?.step ?? 0;
    workerState = 'idle';
    runConfig =
      presetId === 'small'
        ? smallBirdNamePreset(displayEvery)
        : { ...protocolFixture, displayEvery };
    send({
      version: VERSION,
      runId,
      type: 'start',
      config: runConfig,
      ...(resumeFrom ? { resumeFrom, startPaused: true } : {}),
    });
  }

  function command(type: Exclude<LiveTrainingCommand['type'], 'start'>): void {
    send({ version: VERSION, runId, type });
  }

  onMount(() => {
    const heartbeat = window.setInterval(() => (uiTicks += 1), 100);
    return () => {
      window.clearInterval(heartbeat);
      if (renderTimer !== null) window.clearTimeout(renderTimer);
      unsubscribe?.();
      worker?.terminate();
    };
  });
</script>

<section class="training" aria-label="Live training demonstration">
  <div class="lab-panel">
    <p class="eyebrow">Small live-training preset</p>
    <h2>Train off the main thread</h2>
    <p>
      The default preset learns from 12 bundled bird-name teaching lines and evaluates on four
      different names. These short illustrative lines are separate from the exported bird corpus and
      the checkpoint selected in the sidebar.
    </p>
    <p class="glossary-ref">
      Terms: <a href="/glossary#loss">loss</a> · <a href="/glossary#gradient">gradient</a> ·
      <a href="/glossary#adam">AdamW</a>
      · <a href="/glossary#validation-set">validation set</a>
    </p>
    <div class="setup">
      <label for="training-preset">Preset</label>
      <select
        id="training-preset"
        bind:value={presetId}
        onchange={selectPreset}
        disabled={workerState === 'running' || workerState === 'paused'}
      >
        <option value="small">Small bird-name teaching corpus</option>
        <option value="protocol">Synthetic protocol fixture</option>
      </select>
      <label for="display-cadence">Display cadence (steps)</label>
      <input
        id="display-cadence"
        type="number"
        min="1"
        max={maximumStep}
        step="1"
        bind:value={displayEvery}
        disabled={workerState === 'running' || workerState === 'paused'}
      />
    </div>
    {#if presetId === 'small'}
      <p class="fixture">
        12 train lines · 4 validation lines · 259 byte/special tokens · one layer, 12 channels · 300
        optimizer steps
      </p>
      <details>
        <summary>See the teaching corpus</summary>
        <p>Template: <code>{SMALL_BIRD_NAME_PRESET_SUMMARY.template}</code></p>
        <p>Train: {SMALL_BIRD_NAME_PRESET_SUMMARY.trainingNames.join(', ')}.</p>
        <p>Validation: {SMALL_BIRD_NAME_PRESET_SUMMARY.validationNames.join(', ')}.</p>
      </details>
    {:else}
      <p class="fixture">
        Three five-token synthetic sequences · 259-token vocabulary · two layers, four channels ·
        1,000 optimizer steps
      </p>
    {/if}
    <div class="actions">
      <button
        onclick={() => start()}
        disabled={workerState === 'running' || workerState === 'paused'}
        >{presetId === 'small' ? 'Start small preset' : 'Start fixture training'}</button
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
    <h2>Training state</h2>
    <dl class="status-grid">
      <div>
        <dt>State</dt>
        <dd data-testid="worker-state">{workerState}</dd>
      </div>
      <div>
        <dt>Completed batches</dt>
        <dd data-testid="worker-step">{step} / {maximumStep}</dd>
      </div>
      <div>
        <dt>Main-thread heartbeat</dt>
        <dd data-testid="ui-heartbeat">{uiTicks}</dd>
      </div>
      {#if latest}
        <div>
          <dt>Train batch loss</dt>
          <dd data-testid="train-loss">{latest.trainLoss.toFixed(4)}</dd>
        </div>
        <div>
          <dt>Validation loss</dt>
          <dd data-testid="validation-loss">
            {latest.validationLoss?.toFixed(4) ?? 'Not configured'}
          </dd>
        </div>
        <div>
          <dt>Learning rate</dt>
          <dd>{latest.learningRate.toExponential(3)}</dd>
        </div>
        <div>
          <dt>Gradient norm</dt>
          <dd>{latest.gradientNorm.toFixed(4)}</dd>
        </div>
      {/if}
    </dl>
    {#if checkpoint}<p data-testid="checkpoint-summary">
        Checkpoint at batch {checkpoint.step} · {checkpoint.weightBytes.byteLength} weight bytes · optimizer
        state saved.
      </p>{/if}
    {#if lastError}<p class="error" role="alert">{lastError}</p>{/if}
    <p class="note">
      The Worker trains every batch. It computes and sends display values only at the selected step
      cadence, except for a scalar trace requested by “One batch.” This screen draws charts at most
      once per 100 ms; skipped frames do not skip training steps or logged points.
    </p>
  </div>

  {#if runConfig?.validationSequences}
    <div class="charts" aria-label="Live training plots">
      <MetricPlot title="Train and validation loss" maxStep={maximumStep} series={lossSeries} />
      <MetricPlot title="Learning rate" maxStep={maximumStep} series={learningRateSeries} />
      <MetricPlot
        title="Gradient norm before clipping"
        maxStep={maximumStep}
        series={gradientSeries}
      />
    </div>
    <p class="chart-note">
      Train loss is from the batch before its update; validation loss, sample, and weight snapshot
      use the updated model at that step.
    </p>
    <div class="lab-panel" aria-label="Slowed Adam step">
      <h2>Slowed AdamW step</h2>
      <p class="note">
        Pause and choose “One batch” to inspect one update. During a run, this display refreshes at
        the selected cadence. It follows a single named weight, not the whole matrix.
      </p>
      {#if adamTrace}
        <p class="fixture" data-testid="adam-scalar-name">
          {adamTrace.name}[{adamTrace.index}] · update {adamTrace.step}
        </p>
        <ol class="adam-stages" data-testid="adam-stages">
          <li>
            <strong>Before update</strong><span data-testid="adam-before"
              >{scalar(adamTrace.valueBefore)}</span
            >
          </li>
          <li><strong>Raw gradient</strong><span>{scalar(adamTrace.rawGradient)}</span></li>
          <li>
            <strong>Clip global gradient</strong><span
              >{scalar(adamTrace.rawGradient)} × {scalar(adamTrace.clipScale)} = {scalar(
                adamTrace.clippedGradient,
              )}</span
            >
          </li>
          <li>
            <strong>First moment</strong><span
              >m: {scalar(adamTrace.firstMomentBefore)} → {scalar(adamTrace.firstMoment)}</span
            >
          </li>
          <li>
            <strong>Second moment</strong><span
              >v: {scalar(adamTrace.secondMomentBefore)} → {scalar(adamTrace.secondMoment)}</span
            >
          </li>
          <li>
            <strong>Bias correction</strong><span
              >1 − β₁<sup>t</sup> = {scalar(adamTrace.firstBiasCorrection)}; 1 − β₂<sup>t</sup> = {scalar(
                adamTrace.secondBiasCorrection,
              )}</span
            >
          </li>
          <li>
            <strong>Corrected moments</strong><span
              >m̂ = {scalar(adamTrace.firstEstimate)}; v̂ = {scalar(adamTrace.secondEstimate)}</span
            >
          </li>
          <li>
            <strong>Adam + weight decay</strong><span
              >{scalar(adamTrace.adaptiveTerm)} + {scalar(adamTrace.decayTerm)}</span
            >
          </li>
          <li>
            <strong>Updated value</strong><span data-testid="adam-after"
              >{scalar(adamTrace.valueBefore)} − {scalar(adamTrace.learningRate)} × ({scalar(
                adamTrace.adaptiveTerm,
              )} + {scalar(adamTrace.decayTerm)}) = {scalar(adamTrace.valueAfter)}</span
            >
          </li>
        </ol>
        <p class="note">
          Moments and the stored weight use Float32 rounding. The final value above is read back
          from the updated parameter.
        </p>
      {:else}
        <p>No Adam update to inspect yet.</p>
      {/if}
    </div>
    <div class="lab-panel">
      <h2>Seeded sample generations</h2>
      <p class="note">
        Each sample starts with “Bird name: ” and uses a step-specific seed. These are model
        outputs, not claims verified against the corpus.
      </p>
      {#if samples.length === 0}<p>No displayed sample yet.</p>{:else}
        <ol class="samples">
          {#each samples as item}<li>
              <span>Step {item.step}</span>
              <pre>{decodeTokenIds(item.sampleTokenIds ?? [])}</pre>
            </li>{/each}
        </ol>
      {/if}
    </div>
    <div class="lab-panel">
      <h2>Selected weight: blocks.0.attn.q.weight</h2>
      {#if weight}
        <p class="note">
          Stored Float32 values at displayed step {latest?.step}. Histogram counts all {weight
            .values.length} cells; the matrix shows each cell's signed value.
        </p>
        <h3>Weight histogram</h3>
        <div
          class="histogram"
          role="img"
          aria-label={`Histogram of ${weight.name}, ${weight.histogramCounts.length} bins from ${weight.histogramMinimum.toFixed(4)} to ${weight.histogramMaximum.toFixed(4)}`}
        >
          {#each weight.histogramCounts as count, index}<div
              title={`Bin ${index + 1}: ${count} weights`}
            >
              <span style:height={`${(count / histogramPeak) * 100}%`}></span><small>{count}</small>
            </div>{/each}
        </div>
        <p class="range">
          {weight.histogramMinimum.toFixed(4)} to {weight.histogramMaximum.toFixed(4)}
        </p>
        <h3>Matrix snapshot · {weight.shape[0]} × {weight.shape[1]}</h3>
        <div class="matrix-scroll">
          <div
            class="matrix"
            style:grid-template-columns={`repeat(${weight.shape[1]}, minmax(2.6rem, 1fr))`}
            role="grid"
            aria-label={`${weight.name} values`}
          >
            {#each weight.values as value, index}<span
                role="gridcell"
                aria-label={`Row ${Math.floor(index / weight.shape[1])}, column ${index % weight.shape[1]}, value ${value}`}
                style:background={matrixColor(value)}>{value.toFixed(2)}</span
              >{/each}
          </div>
        </div>
      {:else}<p>No displayed weight snapshot yet.</p>{/if}
    </div>
  {/if}
</section>

<style>
  .training {
    display: grid;
    gap: 1.25rem;
    max-width: 1100px;
    min-width: 0;
  }
  .training h2 {
    margin: 0 0 0.8rem;
  }
  .training h3 {
    margin: 1.2rem 0 0.6rem;
    font-family: Georgia, 'Times New Roman', serif;
    font-weight: 500;
  }
  .training p {
    line-height: 1.55;
  }
  .fixture,
  .note,
  .chart-note,
  .range {
    color: var(--muted);
    font-size: 0.83rem;
  }
  .setup {
    display: flex;
    flex-wrap: wrap;
    align-items: end;
    gap: 1rem;
  }
  label {
    display: block;
    margin-top: 1rem;
    font-weight: 700;
  }
  select,
  input {
    display: block;
    margin-top: 0.4rem;
    padding: 0.5rem;
    border: 1px solid var(--line-strong);
    color: var(--ink);
    background: var(--paper);
    font: inherit;
  }
  input {
    width: 9rem;
  }
  select {
    max-width: 100%;
  }
  details {
    margin-top: 0.8rem;
  }
  summary {
    cursor: pointer;
    font-weight: 700;
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
  .status-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
  }
  .status-grid div {
    min-width: 135px;
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
  .charts {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1rem;
    min-width: 0;
  }
  .samples {
    display: grid;
    gap: 0.6rem;
    padding-left: 1.4rem;
  }
  .adam-stages {
    display: grid;
    gap: 0;
    padding-left: 1.6rem;
  }
  .adam-stages li {
    padding: 0.6rem;
    border-bottom: 1px solid var(--line);
  }
  .adam-stages li::marker {
    color: var(--muted);
  }
  .adam-stages strong {
    display: block;
    margin-bottom: 0.2rem;
  }
  .adam-stages span {
    font:
      0.8rem ui-monospace,
      SFMono-Regular,
      Menlo,
      monospace;
    overflow-wrap: anywhere;
  }
  .samples li span {
    color: var(--muted);
    font-size: 0.75rem;
  }
  .samples pre {
    margin: 0.2rem 0;
    padding: 0.6rem;
    overflow: auto;
    border: 1px solid var(--line);
    background: var(--paper);
    white-space: pre-wrap;
    overflow-wrap: anywhere;
  }
  .histogram {
    display: flex;
    align-items: end;
    height: 120px;
    gap: 0.25rem;
    border-bottom: 1px solid var(--line-strong);
  }
  .histogram div {
    display: flex;
    flex: 1;
    height: 100%;
    flex-direction: column;
    justify-content: end;
    align-items: center;
    min-width: 0;
  }
  .histogram span {
    display: block;
    width: 100%;
    background: var(--forest-deep);
  }
  .histogram small {
    font-size: 0.65rem;
  }
  .matrix-scroll {
    overflow: auto;
  }
  .matrix {
    display: grid;
    gap: 2px;
    min-width: max-content;
  }
  .matrix span {
    display: grid;
    min-height: 2.4rem;
    place-items: center;
    border: 1px solid var(--line);
    font:
      0.66rem ui-monospace,
      SFMono-Regular,
      Menlo,
      monospace;
  }
</style>
