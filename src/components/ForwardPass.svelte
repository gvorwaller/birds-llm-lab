<script lang="ts">
  import { onMount } from 'svelte';
  import { checkpointCatalog } from '../lib/stores/checkpoints';
  import { loadInspectionModel, type InspectionModel } from '../lib/data/inspection';
  import { encodeText } from '../lib/tokenizer/bpe';
  import { languageModelForward } from '../lib/model/model';
  import type { ForwardTrace } from '../lib/trace/forward-trace';
  import { formatTraceValue } from '../lib/trace/display';
  import { forwardStages } from '../lib/trace/forward-stages';
  import TraceGrid from './TraceGrid.svelte';

  const prompt = 'The osprey dives into the water to catch';
  let inspection = $state.raw<InspectionModel | null>(null);
  let trace = $state.raw<ForwardTrace | null>(null);
  let tokenIds = $state.raw<number[]>([]);
  let error = $state<string | null>(null);
  let loading = $state(false);
  let step = $state(0);
  let selectedByStage = $state<Record<string, number[]>>({});
  let requestNumber = 0;

  const stages = $derived(forwardStages(inspection?.config.model.nLayers ?? 1));
  const stage = $derived(stages[step] ?? stages[0]);
  const entry = $derived(trace?.get(stage.name) ?? null);
  const topFive = $derived.by(() => {
    if (!trace || tokenIds.length === 0) return [];
    const logits = trace.get('logits').output;
    const width = logits.shape[2];
    const offset = (tokenIds.length - 1) * width;
    return Array.from({ length: width }, (_, id) => ({ id, value: logits.data[offset + id] }))
      .sort((a, b) => b.value - a.value || a.id - b.id)
      .slice(0, 5);
  });
  const selected = $derived(
    selectedByStage[stage.name] ??
      (stage.name === 'logits' ? [0, tokenIds.length - 1, topFive[0]?.id ?? 0] : [0, 0, 0]),
  );
  const explanation = $derived(entry && trace ? entry.explainCell(selected) : null);

  async function load(id: string): Promise<void> {
    const request = ++requestNumber;
    loading = true;
    error = null;
    try {
      const next = await loadInspectionModel();
      if (request !== requestNumber || next.checkpointId !== id) return;
      const ids = encodeText(prompt, next.tokenizer, { bos: true });
      if (ids.length > next.config.model.contextLength) {
        throw new Error('The example prompt exceeds this checkpoint’s context length.');
      }
      const result = languageModelForward(
        ids,
        [1, ids.length],
        next.registry,
        next.config.model,
        undefined,
        { trace: true },
      );
      if (request !== requestNumber) return;
      inspection = next;
      tokenIds = ids;
      trace = result.trace ?? null;
      selectedByStage = {};
      const requestedStage = new URLSearchParams(window.location.search).get('stage');
      step = Math.max(
        0,
        stages.findIndex((candidate) => candidate.name === requestedStage),
      );
    } catch (cause) {
      if (request !== requestNumber) return;
      error = cause instanceof Error ? cause.message : 'The checkpoint could not be inspected.';
      inspection = null;
      trace = null;
    } finally {
      if (request === requestNumber) loading = false;
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

  function select(indices: number[]): void {
    selectedByStage = { ...selectedByStage, [stage.name]: indices };
  }

  function copyValue(): void {
    if (explanation) void navigator.clipboard.writeText(String(explanation.output));
  }
</script>

<section class="inspection" aria-label="Forward-pass inspection">
  <div class="intro">
    <p class="eyebrow">Fixed example · active checkpoint</p>
    <h2>Follow one real calculation</h2>
    <p>“{prompt}”</p>
    {#if inspection}<small
        >Checkpoint {inspection.checkpointId} · training step {inspection.config
          .trainingStep}</small
      >{/if}
  </div>

  {#if loading}<p role="status">Loading the selected checkpoint and computing its trace…</p>{/if}
  {#if error}<p class="error" role="alert">{error}</p>{/if}
  {#if !loading && !inspection && !error}<p>
      Select a validated checkpoint in the sidebar to inspect this prompt.
    </p>{/if}

  {#if trace && inspection && entry}
    <div class="token-list" aria-label="Prompt tokens">
      {#each tokenIds as id, position}
        <span title={`Token id ${id}`}
          ><strong>{inspection.tokenizer.tokens[id]?.display || `#${id}`}</strong><small
            >{position}: #{id}</small
          ></span
        >
      {/each}
    </div>

    <div class="step-controls">
      <button onclick={() => (step = Math.max(0, step - 1))} disabled={step === 0}
        >← Previous</button
      >
      <span aria-live="polite">Step {step + 1} of {stages.length}</span>
      <button
        onclick={() => (step = Math.min(stages.length - 1, step + 1))}
        disabled={step === stages.length - 1}>Next →</button
      >
    </div>
    <label class="jump-label" for="forward-stage">Jump to operation</label>
    <select
      id="forward-stage"
      value={step}
      onchange={(event) => (step = Number(event.currentTarget.value))}
    >
      {#each stages as candidate, index}
        <option value={index}>{index + 1}. {candidate.title}</option>
      {/each}
    </select>

    <section class="stage" aria-labelledby="stage-title">
      <p class="eyebrow">{entry.name}</p>
      <h2 id="stage-title">{stage.title}</h2>
      <p>{stage.description}</p>
      {#if stage.name === 'logits'}
        <div class="top-logits" role="group" aria-label="Top five logits">
          {#each topFive as candidate}
            <button
              class:selected={selected[2] === candidate.id}
              aria-pressed={selected[2] === candidate.id}
              onclick={() => select([0, tokenIds.length - 1, candidate.id])}
            >
              <strong>#{candidate.id} {inspection.tokenizer.tokens[candidate.id]?.display}</strong>
              <span>{formatTraceValue(candidate.value)}</span>
            </button>
          {/each}
        </div>
      {:else}
        <TraceGrid {entry} plane={0} {selected} onSelect={select} />
      {/if}
    </section>

    {#if explanation}
      <aside class="explanation" aria-label="Selected cell explanation">
        <h3>Selected cell {selected.join(', ')}</h3>
        <p>
          <strong>Full Float32 value:</strong> <code>{explanation.output}</code>
          <button onclick={copyValue}>Copy</button>
        </p>
        <details open>
          <summary>What's happening</summary>
          <p>{stage.description}</p>
        </details>
        <details open>
          <summary>The actual math</summary>
          <p><code>{explanation.formula}</code></p>
          {#if Object.keys(explanation.scalars).length > 0}
            <dl>
              {#each Object.entries(explanation.scalars) as [label, value]}<div>
                  <dt>{label}</dt>
                  <dd>{value}</dd>
                </div>{/each}
            </dl>
          {/if}
          {#if explanation.terms.length > 0}
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
          {/if}
          {#if explanation.note}<p class="rounding">{explanation.note}</p>{/if}
        </details>
      </aside>
    {/if}
  {/if}
</section>

<style>
  .inspection {
    display: grid;
    gap: 1.4rem;
    min-width: 0;
  }
  .intro,
  .stage,
  .explanation {
    padding: 1.4rem;
    border: 1px solid #53614d;
    background: #1c251e;
    min-width: 0;
    color: #eef5e8;
  }
  .intro .eyebrow,
  .stage .eyebrow {
    color: #b9ce90;
  }
  .intro h2,
  .stage h2 {
    margin: 0.3rem 0 0.6rem;
  }
  .intro small {
    color: #b9c8b2;
  }
  .error {
    color: #ffb591;
  }
  .token-list {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
  }
  .token-list span {
    display: grid;
    gap: 0.2rem;
    padding: 0.45rem;
    border: 1px solid #53614d;
    background: #202b21;
    color: #eef5e8;
  }
  .token-list small {
    color: #b9c8b2;
  }
  .step-controls {
    display: flex;
    align-items: center;
    gap: 1rem;
  }
  .jump-label {
    display: block;
    font-weight: 700;
  }
  #forward-stage {
    width: min(100%, 34rem);
    min-height: 2.3rem;
    padding: 0.3rem;
    color: #eef5e8;
    background: #253328;
    border: 1px solid #60785b;
  }
  .step-controls button,
  .explanation button {
    min-height: 2.3rem;
    padding: 0.35rem 0.7rem;
    color: #e9f2df;
    background: #283b2d;
    border: 1px solid #829f75;
    cursor: pointer;
  }
  button:disabled {
    opacity: 0.5;
    cursor: default;
  }
  button:focus-visible,
  summary:focus-visible {
    outline: 2px solid #f0c576;
  }
  .top-logits {
    display: grid;
    gap: 0.4rem;
  }
  .top-logits button {
    display: flex;
    justify-content: space-between;
    padding: 0.75rem;
    color: #eef5e8;
    background: #253328;
    border: 1px solid #60785b;
    cursor: pointer;
    text-align: left;
  }
  .top-logits button.selected {
    outline: 2px solid #f0c576;
  }
  .explanation h3 {
    margin-top: 0;
  }
  .explanation details {
    margin-top: 0.8rem;
  }
  .explanation summary {
    cursor: pointer;
    font-weight: 700;
  }
  .explanation dl {
    display: grid;
    gap: 0.3rem;
  }
  .explanation dl div {
    display: flex;
    gap: 1rem;
  }
  .explanation dt {
    min-width: 7rem;
    color: #b9c8b2;
  }
  .explanation dd {
    margin: 0;
    overflow-wrap: anywhere;
  }
  .terms {
    max-height: 18rem;
    overflow: auto;
  }
  .terms table {
    border-collapse: collapse;
    font:
      0.72rem ui-monospace,
      SFMono-Regular,
      Menlo,
      monospace;
  }
  .terms th,
  .terms td {
    padding: 0.3rem 0.6rem;
    border-bottom: 1px solid #415043;
    text-align: right;
  }
  .rounding {
    color: #e5bc83;
  }
</style>
