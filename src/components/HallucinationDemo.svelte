<script lang="ts">
  import { onMount } from 'svelte';
  import { getVerifiedExampleEvidence } from '../lib/data/api';
  import type { OrderCooccurrenceEvidence } from '../lib/data/api-types';
  import { loadInspectionModel } from '../lib/data/inspection';
  import {
    replayKestrelExample,
    type VerifiedExampleReplay,
  } from '../lib/evidence/verified-example';
  import { KESTREL_EXAMPLE } from '../lib/evidence/saved-example';
  import { checkpointCatalog } from '../lib/stores/checkpoints';

  let activeCheckpointId = $state<string | null>(null);
  let verified = $state.raw<VerifiedExampleReplay | null>(null);
  let evidence = $state.raw<OrderCooccurrenceEvidence | null>(null);
  let loading = $state(false);
  let error = $state('');
  let requestNumber = 0;

  const focus = $derived(verified?.replay.steps[KESTREL_EXAMPLE.focusStep] ?? null);
  const before = KESTREL_EXAMPLE.continuation.split(KESTREL_EXAMPLE.focusTokenText)[0];
  const after = KESTREL_EXAMPLE.continuation.split(KESTREL_EXAMPLE.focusTokenText)[1];
  const topTokens = $derived(
    focus
      ? focus.distribution.rankOrder.slice(0, 5).map((id, index) => ({
          id,
          rank: index + 1,
          text: id === KESTREL_EXAMPLE.focusTokenId ? KESTREL_EXAMPLE.focusTokenText : null,
          probability: focus.distribution.probabilities[id],
        }))
      : [],
  );

  function percent(value: number): string {
    return `${(value * 100).toFixed(2)}%`;
  }

  async function replay(): Promise<void> {
    const request = ++requestNumber;
    loading = true;
    error = '';
    verified = null;
    evidence = null;
    try {
      const [inspection, corpusEvidence] = await Promise.all([
        loadInspectionModel(),
        getVerifiedExampleEvidence(),
      ]);
      if (request !== requestNumber) return;
      const result = replayKestrelExample(inspection);
      if (corpusEvidence.corpusSha256 !== KESTREL_EXAMPLE.corpusSha256) {
        throw new Error('The corpus evidence does not match this checkpoint.');
      }
      verified = result;
      evidence = corpusEvidence;
    } catch (cause) {
      if (request === requestNumber) {
        error = cause instanceof Error ? cause.message : 'The saved example could not be replayed.';
      }
    } finally {
      if (request === requestNumber) loading = false;
    }
  }

  onMount(() => {
    let lastId: string | null = null;
    const unsubscribe = checkpointCatalog.subscribe((catalog) => {
      const id = catalog.activeCheckpointId;
      if (id === lastId) return;
      lastId = id;
      activeCheckpointId = id;
      requestNumber += 1;
      verified = null;
      evidence = null;
      loading = false;
      error = '';
      if (id === KESTREL_EXAMPLE.checkpointId) void replay();
    });
    return () => {
      requestNumber += 1;
      unsubscribe();
    };
  });
</script>

<section class="lab-panel demo" aria-labelledby="hallucination-heading">
  <p class="eyebrow">M6.2 · Verified generation example</p>
  <h2 id="hallucination-heading">A likely token can attach the wrong order</h2>
  <p>
    This saved example uses checkpoint <code>ready-v1</code>, seed {KESTREL_EXAMPLE.seed},
    temperature {KESTREL_EXAMPLE.controls.temperature}, top-k {KESTREL_EXAMPLE.controls.topK}, and {KESTREL_EXAMPLE.count}
    generated tokens. The screen replays it with the selected model and checks every token before reporting
    a probability.
  </p>
  <div class="pair">
    <div>
      <h3>Prompt</h3>
      <pre>{KESTREL_EXAMPLE.prompt}</pre>
    </div>
    <div>
      <h3>Saved continuation</h3>
      <pre>{before}<mark>{KESTREL_EXAMPLE.focusTokenText}</mark>{after}</pre>
    </div>
  </div>
  {#if activeCheckpointId !== KESTREL_EXAMPLE.checkpointId}
    <p class="note">
      Select <code>ready-v1</code> in the sidebar to replay this example. The current checkpoint cannot
      stand in for it.
    </p>
  {:else}
    <button onclick={() => void replay()} disabled={loading}>Replay and check the example</button>
  {/if}
  {#if loading}<p role="status">Replaying the checkpoint and reading corpus evidence…</p>{/if}
  {#if error}<p class="error" role="alert">{error}</p>{/if}

  {#if verified && focus && evidence}
    <div class="findings" aria-live="polite">
      <div class="finding">
        <h3>What the model actually chose</h3>
        <p>
          At generated token {KESTREL_EXAMPLE.focusStep + 1}, token
          <code>#{KESTREL_EXAMPLE.focusTokenId}</code> contains the whole highlighted
          <code>Order: Passeriformes</code> span and the following family label. It ranked {verified.focusRank}
          of {focus.distribution.probabilities.length} tokens.
        </p>
        <dl class="numbers">
          <div>
            <dt>Probability after sampling controls</dt>
            <dd data-testid="example-probability">{percent(verified.focusProbability)}</dd>
          </div>
          <div>
            <dt>Seeded random draw</dt>
            <dd>{focus.draw.randomNumber.toPrecision(8)}</dd>
          </div>
          <div>
            <dt>Token interval</dt>
            <dd>
              [{focus.draw.intervalStart.toPrecision(8)}, {focus.draw.intervalEnd.toPrecision(8)})
            </dd>
          </div>
        </dl>
        <p class="note">
          Temperature changes the softmax, then top-k retains 20 tokens. Top-p is 1, so it removes
          none of those 20. The seeded draw landed in this token's interval.
        </p>
        <details>
          <summary>Top five tokens at this decision</summary>
          <ol class="top-tokens">
            {#each topTokens as token}
              <li>
                <span>#{token.rank} · token {token.id}{token.text ? ' · selected' : ''}</span
                ><strong>{percent(token.probability)}</strong>
              </li>
            {/each}
          </ol>
        </details>
      </div>
      <div class="finding">
        <h3>What the exported corpus shows</h3>
        <p>
          The exact <code>Order: Passeriformes</code> / <code>Family:</code> template span occurs in
          <strong data-testid="example-cooccurrence"
            >{evidence.trainSpanDocuments.toLocaleString()}</strong
          >
          of {evidence.trainDocuments.toLocaleString()} train documents for other species. These are source
          co-occurrences, not a probability assigned by the model.
        </p>
        <ul class="source-list">
          {#each evidence.trainExamples as item}
            <li>
              <strong>{item.name}</strong><span>{item.split} · {item.field}</span>
              <pre>{item.snippet}</pre>
            </li>
          {/each}
        </ul>
        <p>
          The exported <strong>{evidence.target.name}</strong> document is in the
          <strong>{evidence.target.split}</strong> split. Its recorded order is
          <strong data-testid="example-reference-order">{evidence.target.order}</strong>.
        </p>
        <pre class="target-snippet">{evidence.target.snippet}</pre>
      </div>
    </div>
    <div class="interpretation">
      <h3>Interpretation</h3>
      <p>
        The model selected a common training-template token after inventing a scientific name. It
        did not look up the Kestrel's order before continuing. The corpus count shows how often that
        span appeared in training; it does not prove which internal weights caused this draw. The
        Kestrel's held-out document gives a different order.
      </p>
      <p>
        The Birds tag lesson is similar: a fluent continuation is not a source-backed tag decision.
        Tags need classification and evidence checks rather than being assigned from generated
        prose.
      </p>
    </div>
  {/if}
</section>

<style>
  .demo {
    min-width: 0;
  }
  h2 {
    margin: 0 0 0.7rem;
  }
  h3 {
    margin: 0 0 0.5rem;
    font-family: Georgia, 'Times New Roman', serif;
    font-weight: 500;
  }
  p {
    line-height: 1.55;
  }
  .pair,
  .findings {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(min(100%, 24rem), 1fr));
    gap: 1rem;
  }
  .pair > div,
  .finding,
  .interpretation {
    min-width: 0;
    padding: 1rem;
    border: 1px solid var(--line);
    background: var(--paper-raised);
  }
  pre {
    margin: 0;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    font:
      0.84rem/1.5 ui-monospace,
      SFMono-Regular,
      Menlo,
      monospace;
  }
  .pair pre,
  .target-snippet {
    padding: 0.8rem;
    background: var(--paper);
    border: 1px solid var(--line);
  }
  mark {
    background: #f4ce84;
    color: #30230b;
  }
  button {
    margin: 1rem 0;
    padding: 0.65rem 0.9rem;
    border: 1px solid var(--forest);
    background: var(--paper);
    color: var(--ink);
    cursor: pointer;
  }
  button:disabled {
    opacity: 0.5;
    cursor: default;
  }
  .note {
    color: var(--muted);
    font-size: 0.84rem;
  }
  .error {
    color: #af421c;
  }
  .findings {
    margin-top: 1rem;
  }
  .numbers {
    display: grid;
    gap: 0.4rem;
  }
  .numbers div {
    display: flex;
    gap: 0.6rem;
    justify-content: space-between;
    flex-wrap: wrap;
    border-bottom: 1px solid var(--line);
    padding: 0.4rem 0;
  }
  dt {
    color: var(--muted);
  }
  dd {
    margin: 0;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  summary {
    cursor: pointer;
  }
  .top-tokens {
    padding-left: 1.4rem;
  }
  .top-tokens li {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    padding: 0.3rem 0;
  }
  .source-list {
    display: grid;
    gap: 0.6rem;
    padding-left: 1.2rem;
  }
  .source-list li span {
    display: block;
    color: var(--muted);
    font-size: 0.78rem;
  }
  .source-list pre {
    margin-top: 0.25rem;
  }
  .interpretation {
    margin-top: 1rem;
    border-left: 3px solid var(--forest);
  }
</style>
