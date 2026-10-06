<script lang="ts">
  import { onMount } from 'svelte';
  import { checkpointCatalog } from '../lib/stores/checkpoints';
  import { loadInspectionModel, type InspectionModel } from '../lib/data/inspection';
  import { encodeText, decodeText } from '../lib/tokenizer/bpe';
  import { generateReplay, type GenerationReplay } from '../lib/generation/replay';

  let inspection = $state.raw<InspectionModel | null>(null);
  let replay = $state.raw<GenerationReplay | null>(null);
  let prompt = $state('The osprey dives into the water to catch');
  let temperature = $state(1);
  let topK = $state(0);
  let topP = $state(1);
  let seed = $state(42);
  let count = $state(8);
  let stepIndex = $state(0);
  let query = $state('');
  let page = $state(0);
  let loading = $state(false);
  let error = $state<string | null>(null);
  let requestNumber = 0;
  const pageSize = 25;

  const currentStep = $derived(replay?.steps[stepIndex] ?? null);
  const distribution = $derived(currentStep?.distribution ?? null);
  const filteredIds = $derived.by(() => {
    if (!distribution || !inspection) return [];
    const needle = query.trim().toLowerCase();
    const tokens = inspection.tokenizer.tokens;
    return distribution.rankOrder.filter(
      (id) =>
        !needle ||
        String(id).includes(needle) ||
        tokens[id]?.display.toLowerCase().includes(needle),
    );
  });
  const pageCount = $derived(Math.max(1, Math.ceil(filteredIds.length / pageSize)));
  const visibleIds = $derived(filteredIds.slice(page * pageSize, (page + 1) * pageSize));
  const remainderBefore = $derived(
    distribution
      ? Math.max(0, 1 - visibleIds.reduce((sum, id) => sum + distribution.beforeFilters[id], 0))
      : 0,
  );
  const remainderAfter = $derived(
    distribution
      ? Math.max(0, 1 - visibleIds.reduce((sum, id) => sum + distribution.probabilities[id], 0))
      : 0,
  );
  const topIds = $derived(distribution?.rankOrder.slice(0, 10) ?? []);
  const selectedId = $derived(currentStep?.draw.tokenId ?? 0);
  const outputIds = $derived(replay?.generatedIds.slice(0, stepIndex + 1) ?? []);

  function tokenLabel(id: number): string {
    return inspection?.tokenizer.tokens[id]?.display || `#${id}`;
  }

  function number(value: number): string {
    return value.toPrecision(8);
  }

  function intervalStart(id: number): number {
    return id === 0 ? 0 : (distribution?.cumulative[id - 1] ?? 0);
  }

  function invalidate(): void {
    replay = null;
    error = null;
    stepIndex = 0;
    page = 0;
  }

  async function load(id: string): Promise<void> {
    const request = ++requestNumber;
    loading = true;
    error = null;
    replay = null;
    try {
      const next = await loadInspectionModel();
      if (request !== requestNumber || next.checkpointId !== id) return;
      inspection = next;
      run(1);
    } catch (cause) {
      if (request !== requestNumber) return;
      inspection = null;
      error = cause instanceof Error ? cause.message : 'The checkpoint could not be inspected.';
    } finally {
      if (request === requestNumber) loading = false;
    }
  }

  function run(steps: number): void {
    if (!inspection) return;
    try {
      const promptIds = encodeText(prompt, inspection.tokenizer, { bos: true });
      replay = generateReplay(
        promptIds,
        seed,
        { temperature, topK, topP },
        steps,
        inspection.registry,
        inspection.config.model,
      );
      stepIndex = 0;
      page = 0;
      error = null;
    } catch (cause) {
      replay = null;
      error = cause instanceof Error ? cause.message : 'The generation could not be computed.';
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
        replay = null;
        loading = false;
      }
    });
    return () => {
      requestNumber += 1;
      unsubscribe();
    };
  });
</script>

<section class="generation" aria-label="Generation inspection">
  <div class="panel intro">
    <p class="eyebrow">Active checkpoint · real model forward pass</p>
    <h2>Inspect a next-token draw</h2>
    <p>
      Each decision recomputes the model on the current context. The controls and seed are saved
      with the replay.
    </p>
    <p class="glossary-ref">
      Terms: <a href="/glossary#logit">logit</a> · <a href="/glossary#temperature">temperature</a> ·
      <a href="/glossary#context-window">context window</a>
    </p>
    {#if inspection}<small
        >Checkpoint {inspection.checkpointId} · training step {inspection.config
          .trainingStep}</small
      >{/if}
  </div>

  {#if loading}<p role="status">Loading the selected checkpoint…</p>{/if}
  {#if error}<p class="error" role="alert">{error}</p>{/if}
  {#if !loading && !inspection && !error}<p>
      Select a validated checkpoint in the sidebar to inspect generation.
    </p>{/if}

  {#if inspection}
    <div class="panel controls">
      <label for="generation-prompt">Prompt</label>
      <textarea id="generation-prompt" rows="3" bind:value={prompt} oninput={invalidate}></textarea>
      <div class="control-grid">
        <label
          >Temperature <input
            aria-label="Temperature"
            type="number"
            min="0.001"
            step="0.1"
            bind:value={temperature}
            oninput={invalidate}
          /></label
        >
        <label
          >Top-k <input
            aria-label="Top-k"
            type="number"
            min="0"
            max={inspection.config.model.vocabSize}
            step="1"
            bind:value={topK}
            oninput={invalidate}
          /></label
        >
        <label
          >Top-p <input
            aria-label="Top-p"
            type="number"
            min="0.001"
            max="1"
            step="0.05"
            bind:value={topP}
            oninput={invalidate}
          /></label
        >
        <label
          >Seed <input
            aria-label="Seed"
            type="number"
            step="1"
            bind:value={seed}
            oninput={invalidate}
          /></label
        >
        <label
          >Tokens to generate <input
            aria-label="Tokens to generate"
            type="number"
            min="1"
            max="32"
            step="1"
            bind:value={count}
            oninput={invalidate}
          /></label
        >
      </div>
      <div class="actions">
        <button onclick={() => run(1)}>Inspect next token</button>
        <button onclick={() => run(count)}>Generate and save replay</button>
      </div>
      <p class="hint">
        Order: temperature → softmax → top-k → top-p → renormalize → seeded draw. Top-k 0 and top-p
        1 disable those filters.
      </p>
    </div>
  {/if}

  {#if replay && currentStep && distribution && inspection}
    <div class="panel replay">
      <p class="eyebrow">Saved replay · seed {replay.seed}</p>
      <h2>Decision {stepIndex + 1} of {replay.steps.length}</h2>
      <div class="actions">
        <button onclick={() => (stepIndex = Math.max(0, stepIndex - 1))} disabled={stepIndex === 0}
          >← Previous decision</button
        >
        <button
          onclick={() => (stepIndex = Math.min((replay?.steps.length ?? 1) - 1, stepIndex + 1))}
          disabled={stepIndex === replay.steps.length - 1}>Next decision →</button
        >
      </div>
      <p>
        Generated IDs: <code data-testid="generated-ids">{replay.generatedIds.join(', ')}</code>
      </p>
      <p>Visible continuation: <code>{decodeText(outputIds, inspection.tokenizer)}</code></p>
      <p class="hint">Previous/next reads saved decisions; it does not draw a new random number.</p>
    </div>

    <div class="panel context">
      <h2>Context for this model call</h2>
      <p>
        {currentStep.context.length} of {inspection.config.model.contextLength} positions used. Next token
        has absolute position {currentStep.nextAbsolutePosition}.
      </p>
      <div class="token-strip" aria-label="Model context tokens">
        {#each currentStep.context as token}
          <span class="token-chip" title={`Token #${token.id}`}
            ><strong>{tokenLabel(token.id)}</strong><small
              >absolute {token.absolutePosition} · model {token.modelPosition}</small
            ></span
          >
        {/each}
      </div>
      <details>
        <summary>Dropped leading tokens ({currentStep.dropped.length})</summary>
        {#if currentStep.dropped.length === 0}<p>None. The context window has not slid.</p>{:else}
          <div class="token-strip" aria-label="Dropped leading tokens">
            {#each currentStep.dropped as token}
              <span class="token-chip"
                ><strong>{tokenLabel(token.id)}</strong><small
                  >absolute {token.absolutePosition} · #{token.id}</small
                ></span
              >
            {/each}
          </div>
        {/if}
      </details>
      <p class="hint">
        Model positions restart at zero after the window slides; absolute positions keep counting.
      </p>
    </div>

    <div class="panel decision">
      <h2>Selected token: {tokenLabel(selectedId)} <small>#{selectedId}</small></h2>
      <dl>
        <div>
          <dt>Retained mass</dt>
          <dd data-testid="retained-mass">{number(distribution.retainedMass)}</dd>
        </div>
        <div>
          <dt>Removed mass</dt>
          <dd data-testid="removed-mass">{number(distribution.removedMass)}</dd>
        </div>
        <div>
          <dt>Random draw</dt>
          <dd data-testid="random-draw">{number(currentStep.draw.randomNumber)}</dd>
        </div>
        <div>
          <dt>Cumulative interval</dt>
          <dd data-testid="selected-interval">
            [{number(currentStep.draw.intervalStart)}, {number(currentStep.draw.intervalEnd)})
          </dd>
        </div>
        <div>
          <dt>Selected probability</dt>
          <dd>{number(distribution.probabilities[selectedId])}</dd>
        </div>
      </dl>
      <p class="hint">
        Cumulative intervals are laid out in token-ID order. The draw lands inside the selected
        token's interval.
      </p>
    </div>

    <div class="panel vocabulary">
      <h2>Top logits and probabilities</h2>
      <div class="table-wrap">
        <table aria-label="Top ten next-token logits">
          <thead
            ><tr
              ><th>ID</th><th>Token</th><th>Logit</th><th>Before filters</th><th>After filters</th
              ></tr
            ></thead
          ><tbody>
            {#each topIds as id}
              <tr class:selected={id === selectedId}
                ><td>#{id}</td><td>{tokenLabel(id)}</td><td>{number(distribution.logits[id])}</td
                ><td>{number(distribution.beforeFilters[id])}</td><td
                  >{number(distribution.probabilities[id])}</td
                ></tr
              >
            {/each}
          </tbody>
        </table>
      </div>
    </div>

    <div class="panel vocabulary">
      <h2>Full-vocabulary probability mass</h2>
      <p>
        Search by token ID or displayed bytes. Rows are sorted by pre-filter probability; intervals
        remain in token-ID order.
      </p>
      <label for="probability-search">Search vocabulary</label>
      <input id="probability-search" type="search" bind:value={query} oninput={() => (page = 0)} />
      <p>{filteredIds.length} matching tokens · page {page + 1} of {pageCount}</p>
      <div class="actions">
        <button onclick={() => (page = Math.max(0, page - 1))} disabled={page === 0}
          >← Previous page</button
        >
        <button
          onclick={() => (page = Math.min(pageCount - 1, page + 1))}
          disabled={page === pageCount - 1}>Next page →</button
        >
      </div>
      <div class="table-wrap">
        <table aria-label="Vocabulary probabilities">
          <thead
            ><tr
              ><th>ID</th><th>Token</th><th>Logit</th><th>Before</th><th>After</th><th
                >Cumulative interval</th
              ></tr
            ></thead
          ><tbody>
            {#each visibleIds as id}
              <tr
                class:selected={id === selectedId}
                data-token-id={id}
                data-before={distribution.beforeFilters[id]}
                data-after={distribution.probabilities[id]}
                data-interval-start={intervalStart(id)}
                data-interval-end={distribution.cumulative[id]}
                ><td>#{id}</td><td>{tokenLabel(id)}</td><td>{number(distribution.logits[id])}</td
                ><td>{number(distribution.beforeFilters[id])}</td><td
                  >{number(distribution.probabilities[id])}</td
                ><td>[{number(intervalStart(id))}, {number(distribution.cumulative[id])})</td></tr
              >
            {/each}
            <tr class="remainder"
              ><th colspan="3">Other {distribution.logits.length - visibleIds.length} tokens</th><td
                data-testid="remainder-before">{number(remainderBefore)}</td
              ><td data-testid="remainder-after">{number(remainderAfter)}</td><td>—</td></tr
            >
          </tbody>
        </table>
      </div>
      <p class="hint">
        The remainder aggregates every token outside this page, including search exclusions. Page
        rows plus remainder sum to the full mass.
      </p>
    </div>
  {/if}
</section>

<style>
  .generation {
    display: grid;
    gap: 1.25rem;
    min-width: 0;
  }
  .panel {
    min-width: 0;
    padding: 1.4rem;
    border: 1px solid var(--line);
    background: var(--paper-raised);
  }
  .panel h2 {
    margin: 0.25rem 0 0.7rem;
    font-size: 1.75rem;
  }
  .panel p {
    line-height: 1.5;
  }
  .panel small,
  .hint {
    color: var(--muted);
  }
  .error {
    color: #af421c;
  }
  .controls label,
  .vocabulary label {
    display: grid;
    gap: 0.35rem;
    font-weight: 700;
  }
  .controls textarea,
  .controls input,
  .vocabulary input {
    width: 100%;
    min-width: 0;
    padding: 0.55rem;
    border: 1px solid var(--line-strong);
    color: var(--ink);
    background: var(--paper);
    font: inherit;
  }
  .control-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(145px, 1fr));
    gap: 0.8rem;
    margin-top: 1rem;
  }
  .actions {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    margin-top: 1rem;
  }
  button {
    min-height: 2.4rem;
    padding: 0.45rem 0.8rem;
    border: 1px solid var(--forest);
    color: var(--ink);
    background: var(--paper);
    cursor: pointer;
  }
  button:disabled {
    cursor: default;
    opacity: 0.45;
  }
  .replay code {
    overflow-wrap: anywhere;
  }
  .context .token-strip {
    max-height: 14rem;
    overflow: auto;
  }
  .context details {
    margin-top: 1rem;
  }
  .context summary {
    cursor: pointer;
    font-weight: 700;
  }
  .decision dl {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 0.6rem;
  }
  .decision dl div {
    padding: 0.7rem;
    border: 1px solid var(--line);
  }
  .decision dt {
    color: var(--muted);
    font-size: 0.75rem;
  }
  .decision dd {
    margin: 0.3rem 0 0;
    font:
      0.85rem ui-monospace,
      SFMono-Regular,
      Menlo,
      monospace;
    overflow-wrap: anywhere;
  }
  .table-wrap {
    max-height: 30rem;
    overflow: auto;
    margin-top: 0.8rem;
    border: 1px solid var(--line);
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font:
      0.78rem ui-monospace,
      SFMono-Regular,
      Menlo,
      monospace;
  }
  th,
  td {
    padding: 0.55rem;
    border-bottom: 1px solid var(--line);
    text-align: left;
    overflow-wrap: anywhere;
  }
  thead th {
    position: sticky;
    top: 0;
    color: var(--sidebar-ink);
    background: var(--sidebar);
  }
  tr.selected {
    background: color-mix(in srgb, var(--paper) 65%, var(--lichen));
  }
  tr.remainder {
    font-weight: 700;
  }
  .vocabulary input {
    max-width: 28rem;
  }
</style>
