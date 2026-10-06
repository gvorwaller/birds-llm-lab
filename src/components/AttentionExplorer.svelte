<script lang="ts">
  import { onMount } from 'svelte';
  import { checkpointCatalog } from '../lib/stores/checkpoints';
  import { loadInspectionModel, type InspectionModel } from '../lib/data/inspection';
  import { encodeText } from '../lib/tokenizer/bpe';
  import { languageModelForward } from '../lib/model/model';
  import type { ForwardTrace } from '../lib/trace/forward-trace';
  import { averageAttentionEntry } from '../lib/trace/attention-view';
  import TraceGrid from './TraceGrid.svelte';

  const examples = [
    'The osprey dives into the water to catch',
    'The northern cardinal sings from a branch',
    'A seabird flies across the open ocean',
  ];
  let prompt = $state(examples[0]);
  let inspection = $state.raw<InspectionModel | null>(null);
  let trace = $state.raw<ForwardTrace | null>(null);
  let ids = $state.raw<number[]>([]);
  let error = $state<string | null>(null);
  let loading = $state(false);
  let layer = $state(0);
  let head = $state<number | 'average'>(0);
  let row = $state(0);
  let column = $state(0);
  let requestNumber = 0;

  const entry = $derived(
    trace && inspection
      ? head === 'average'
        ? averageAttentionEntry(trace, layer, inspection.config.model.nHeads)
        : trace.get(`blocks.${layer}.attn.probabilities`)
      : null,
  );
  const plane = $derived(head === 'average' ? 0 : head);
  const selected = $derived([plane, row, column]);
  const explanation = $derived(entry?.explainCell(selected) ?? null);
  const strongest = $derived.by(() => {
    if (!entry || ids.length === 0) return null;
    const length = ids.length;
    let best = 0;
    let value = -Infinity;
    for (let key = 0; key < length; key += 1) {
      const probability = entry.output.data[(plane * length + row) * length + key];
      if (probability > value) {
        best = key;
        value = probability;
      }
    }
    return { position: best, value };
  });

  function compute(model: InspectionModel): void {
    const nextIds = encodeText(prompt, model.tokenizer, { bos: true });
    if (nextIds.length === 0 || nextIds.length > model.config.model.contextLength) {
      throw new Error(
        `Prompt must fit within ${model.config.model.contextLength} tokens including BOS.`,
      );
    }
    const result = languageModelForward(
      nextIds,
      [1, nextIds.length],
      model.registry,
      model.config.model,
      undefined,
      { trace: true },
    );
    ids = nextIds;
    trace = result.trace ?? null;
    row = 0;
    column = 0;
  }

  async function load(id: string): Promise<void> {
    const request = ++requestNumber;
    loading = true;
    error = null;
    try {
      const next = await loadInspectionModel();
      if (request !== requestNumber || next.checkpointId !== id) return;
      inspection = next;
      layer = 0;
      head = 0;
      compute(next);
    } catch (cause) {
      if (request !== requestNumber) return;
      error = cause instanceof Error ? cause.message : 'Attention trace unavailable.';
      trace = null;
    } finally {
      if (request === requestNumber) loading = false;
    }
  }

  function inspectPrompt(): void {
    if (!inspection) return;
    try {
      error = null;
      compute(inspection);
    } catch (cause) {
      error = cause instanceof Error ? cause.message : 'Prompt could not be inspected.';
    }
  }

  onMount(() => {
    let lastId: string | null = null;
    const unsubscribe = checkpointCatalog.subscribe((value) => {
      const id = value.activeCheckpointId;
      if (id === lastId) return;
      lastId = id;
      if (id) void load(id);
      else {
        requestNumber += 1;
        inspection = null;
        trace = null;
        loading = false;
      }
    });
    return () => {
      requestNumber += 1;
      unsubscribe();
    };
  });
</script>

<section class="attention-explorer" aria-label="Attention explorer">
  <div class="panel">
    <p class="eyebrow">Observed probabilities · active checkpoint</p>
    <h2>Which earlier positions contribute?</h2>
    <p>
      Each row is a query position. Each column is a key position. A value is the fraction of that
      head’s weighted mix assigned to that key.
    </p>
    <label for="preset-prompt">Example prompt</label>
    <select
      id="preset-prompt"
      onchange={(event) => {
        prompt = event.currentTarget.value;
        inspectPrompt();
      }}
    >
      {#each examples as example}<option value={example}>{example}</option>{/each}
    </select>
    <label for="attention-prompt">Prompt to inspect</label>
    <textarea id="attention-prompt" bind:value={prompt} rows="2" spellcheck="false"></textarea>
    <button onclick={inspectPrompt} disabled={!inspection}>Inspect prompt</button>
    {#if inspection}<small
        >Checkpoint {inspection.checkpointId} · step {inspection.config.trainingStep}</small
      >{/if}
  </div>

  {#if loading}<p role="status">Loading the selected checkpoint…</p>{/if}
  {#if error}<p role="alert" class="error">{error}</p>{/if}
  {#if !loading && !inspection && !error}<p>Select a validated checkpoint in the sidebar.</p>{/if}

  {#if entry && trace && inspection}
    <div class="tokens" aria-label="Prompt positions">
      {#each ids as id, position}<span
          ><strong>{position}</strong> {inspection.tokenizer.tokens[id]?.display || `#${id}`}</span
        >{/each}
    </div>
    <div class="controls">
      <label for="attention-layer">Layer</label>
      <select
        id="attention-layer"
        value={layer}
        onchange={(event) => {
          layer = Number(event.currentTarget.value);
          row = 0;
          column = 0;
        }}
      >
        {#each Array(inspection.config.model.nLayers) as _, index}<option value={index}
            >Layer {index}</option
          >{/each}
      </select>
      <label for="attention-head">View</label>
      <select
        id="attention-head"
        value={head}
        onchange={(event) => {
          head =
            event.currentTarget.value === 'average' ? 'average' : Number(event.currentTarget.value);
          row = 0;
          column = 0;
        }}
      >
        {#each Array(inspection.config.model.nHeads) as _, index}<option value={index}
            >Head {index}</option
          >{/each}
        <option value="average">Average heads</option>
      </select>
    </div>
    <div class="panel">
      <h2>
        {head === 'average' ? 'Average of head probabilities' : `Layer ${layer}, head ${head}`}
      </h2>
      <p>
        The average is computed after each head’s softmax. These prompts are examples; the map shows
        observed values and does not claim a head learned a specific language rule.
      </p>
      <TraceGrid
        {entry}
        {plane}
        {selected}
        onSelect={(indices) => {
          row = indices[1];
          column = indices[2];
        }}
      />
      {#if strongest}<p>
          For query position {row}, the largest displayed probability is at key position {strongest.position}:
          {strongest.value}.
        </p>{/if}
    </div>
    {#if explanation}
      <div class="panel">
        <h3>Query {row} → key {column}</h3>
        <p>Full Float32 value: <code>{explanation.output}</code></p>
        <details open>
          <summary>What's happening</summary>
          <p>
            {head === 'average'
              ? 'The cell averages the probabilities from all heads at these positions.'
              : 'The cell is the probability from this individual attention head.'}
          </p>
        </details>
        <details open>
          <summary>The actual math</summary>
          <p><code>{explanation.formula}</code></p>
          <div class="terms">
            <table>
              <thead><tr><th>Term</th><th>Left</th><th>Right</th><th>Product</th></tr></thead>
              <tbody
                >{#each explanation.terms as term}<tr
                    ><th>{term.label}</th><td>{term.left}</td><td>{term.right}</td><td
                      >{term.product}</td
                    ></tr
                  >{/each}</tbody
              >
            </table>
          </div>
        </details>
      </div>
    {/if}
  {/if}
</section>

<style>
  .attention-explorer {
    display: grid;
    gap: 1.2rem;
    min-width: 0;
  }
  .panel {
    padding: 1.4rem;
    background: #1c251e;
    color: #eef5e8;
    border: 1px solid #53614d;
    min-width: 0;
  }
  .panel .eyebrow {
    color: #b9ce90;
  }
  .panel h2 {
    margin: 0.2rem 0 0.8rem;
  }
  label {
    display: block;
    margin: 0.7rem 0 0.25rem;
    font-weight: 700;
  }
  select,
  textarea {
    max-width: 100%;
    width: min(100%, 34rem);
    min-height: 2.4rem;
    padding: 0.4rem;
    color: #eef5e8;
    background: #253328;
    border: 1px solid #60785b;
  }
  textarea {
    display: block;
    width: 100%;
  }
  button {
    margin-top: 0.7rem;
    min-height: 2.4rem;
    padding: 0.4rem 0.8rem;
    color: #eef5e8;
    background: #283b2d;
    border: 1px solid #829f75;
    cursor: pointer;
  }
  button:focus-visible,
  select:focus-visible,
  textarea:focus-visible {
    outline: 2px solid #f0c576;
  }
  small {
    display: block;
    margin-top: 0.7rem;
    color: #b9c8b2;
  }
  .tokens {
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
  }
  .tokens span {
    padding: 0.4rem;
    background: #202b21;
    color: #eef5e8;
    border: 1px solid #53614d;
  }
  .controls {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.6rem;
  }
  .controls label {
    margin: 0;
  }
  .controls select {
    width: auto;
  }
  .error {
    color: #a43922;
  }
  .terms {
    max-height: 14rem;
    overflow: auto;
  }
  table {
    border-collapse: collapse;
    font:
      0.72rem ui-monospace,
      SFMono-Regular,
      Menlo,
      monospace;
  }
  th,
  td {
    padding: 0.3rem 0.6rem;
    border-bottom: 1px solid #415043;
    text-align: right;
  }
</style>
