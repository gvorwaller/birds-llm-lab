<script lang="ts">
  import { onMount } from 'svelte';
  import { analyzeGpt2, getGpt2Status, type Gpt2Status } from '../lib/data/api';
  import type { Gpt2Analysis } from '../../server/gpt2/companion';
  import { loadInspectionModel, type InspectionModel } from '../lib/data/inspection';
  import { encodeText, decodeText } from '../lib/tokenizer/bpe';
  import { generateReplay, type GenerationReplay } from '../lib/generation/replay';
  import { checkpointCatalog } from '../lib/stores/checkpoints';

  let status = $state<Gpt2Status | null>(null);
  let inspection = $state.raw<InspectionModel | null>(null);
  let prompt = $state('The osprey dives into the water to catch');
  let result = $state.raw<Gpt2Analysis | null>(null);
  let tinyReplay = $state.raw<GenerationReplay | null>(null);
  let tinyGreedyReplay = $state.raw<GenerationReplay | null>(null);
  let tinyTokens = $state<number[]>([]);
  let tinyParameterCount = $state(0);
  let loading = $state(false);
  let error = $state<string | null>(null);
  let checkpointError = $state<string | null>(null);
  let checkpointChecked = $state(false);
  let checkpointLoad = 0;
  const tinyTop = $derived(tinyReplay?.steps[0].distribution.rankOrder.slice(0, 10) ?? []);

  function percent(probability: number): string {
    return `${(probability * 100).toFixed(probability < 0.001 ? 3 : 2)}%`;
  }

  function invalidate(): void {
    result = null;
    tinyReplay = null;
    tinyGreedyReplay = null;
    tinyTokens = [];
    error = null;
  }

  async function analyze(): Promise<void> {
    if (!status?.installed || loading) return;
    loading = true;
    invalidate();
    try {
      const next = await analyzeGpt2(prompt);
      result = next;
      if (inspection) {
        tinyTokens = encodeText(prompt, inspection.tokenizer, { bos: true });
        tinyReplay = generateReplay(
          tinyTokens,
          42,
          { temperature: 1, topK: 0, topP: 1 },
          1,
          inspection.registry,
          inspection.config.model,
        );
        tinyGreedyReplay = generateReplay(
          tinyTokens,
          42,
          { temperature: 1, topK: 1, topP: 1 },
          20,
          inspection.registry,
          inspection.config.model,
        );
      }
    } catch (cause) {
      error = cause instanceof Error ? cause.message : 'The comparison could not be run.';
    } finally {
      loading = false;
    }
  }

  async function loadTiny(expectedId: string | null = null): Promise<void> {
    const request = ++checkpointLoad;
    checkpointChecked = false;
    checkpointError = null;
    inspection = null;
    invalidate();
    try {
      const value = await loadInspectionModel();
      if (request !== checkpointLoad) return;
      if (expectedId !== null && value.checkpointId !== expectedId) {
        throw new Error('The selected checkpoint changed. Reload this page.');
      }
      inspection = value;
      tinyParameterCount = [...value.registry].reduce(
        (total, parameter) => total + parameter.value.data.length,
        0,
      );
    } catch (cause) {
      if (request === checkpointLoad) {
        checkpointError =
          cause instanceof Error ? cause.message : 'The tiny checkpoint is unavailable.';
      }
    } finally {
      if (request === checkpointLoad) checkpointChecked = true;
    }
  }

  onMount(() => {
    void getGpt2Status()
      .then((value) => (status = value))
      .catch((cause) => (error = cause instanceof Error ? cause.message : 'Service unavailable.'));
    void loadTiny();
    let observedId: string | null = null;
    const unsubscribe = checkpointCatalog.subscribe((value) => {
      const id = value.activeCheckpointId;
      if (id === null || id === observedId) return;
      observedId = id;
      if (inspection?.checkpointId !== id) void loadTiny(id);
    });
    return () => {
      checkpointLoad += 1;
      unsubscribe();
    };
  });
</script>

<section class="comparison" aria-label="GPT-2 comparison">
  <div class="panel">
    <p class="eyebrow">Optional local model · no remote inference</p>
    <h2>One prompt, two learned models</h2>
    <p>
      GPT-2 small is a pretrained general-text model with about 124 million parameters. The tiny
      model was trained here on bird text. Their vocabularies and training data differ, so compare
      the mechanics and scale rather than treating their percentages as the same experiment.
    </p>
    {#if status?.installed}
      <p class="success" role="status">
        GPT-2 is installed locally ({(status.bytes / 1024 / 1024).toFixed(0)} MiB cached). Analysis uses
        that cache without network access.
      </p>
    {:else if status}
      <p role="status">GPT-2 is optional and has not been installed in this checkout.</p>
    {:else}
      <p role="status">Checking the local GPT-2 cache…</p>
    {/if}
    {#if inspection}
      <p>
        Selected tiny checkpoint: {inspection.checkpointId} · {tinyParameterCount.toLocaleString()} parameters.
      </p>
    {:else if checkpointError}
      <p>The tiny checkpoint is unavailable: {checkpointError}</p>
    {:else}
      <p role="status">Checking the selected tiny checkpoint…</p>
    {/if}
  </div>

  <form
    class="panel"
    onsubmit={(event) => {
      event.preventDefault();
      void analyze();
    }}
  >
    <label for="comparison-prompt">Prompt</label>
    <textarea
      id="comparison-prompt"
      rows="3"
      maxlength="300"
      bind:value={prompt}
      oninput={invalidate}></textarea>
    <button
      class="primary-action"
      type="submit"
      disabled={!status?.installed || !checkpointChecked || loading}
    >
      {loading ? 'Running both models…' : 'Compare this prompt'}
    </button>
    <p class="muted">
      Both models use unfiltered next-token probabilities. The continuations use greedy decoding for
      20 tokens.
    </p>
  </form>

  {#if error}<p class="error" role="alert">{error}</p>{/if}

  {#if result}
    <div class="comparison-grid">
      {#if inspection && tinyReplay && tinyGreedyReplay}
        <article class="panel">
          <p class="eyebrow">Birds LLM Lab · {inspection.checkpointId}</p>
          <h2>Tiny model</h2>
          <p>{tinyTokens.length} prompt tokens, including the start token.</p>
          <div class="token-list" aria-label="Tiny model prompt tokens">
            {#each tinyTokens as id}
              <span title={`Token ${id}`}
                >{inspection.tokenizer.tokens[id]?.display ?? `#${id}`}</span
              >
            {/each}
          </div>
          <h3>Next token</h3>
          <ol class="prediction-list">
            {#each tinyTop as id}
              <li>
                <code>{inspection.tokenizer.tokens[id]?.display ?? `#${id}`}</code><strong
                  >{percent(tinyReplay.steps[0].distribution.probabilities[id])}</strong
                >
              </li>
            {/each}
          </ol>
          <h3>Greedy continuation</h3>
          <p class="continuation">
            {prompt}<mark>{decodeText(tinyGreedyReplay.generatedIds, inspection.tokenizer)}</mark>
          </p>
          <p><a href="/attention">Inspect this model's attention heads →</a></p>
        </article>
      {/if}

      <article class="panel">
        <p class="eyebrow">Pretrained general-text model · 124M parameters</p>
        <h2>GPT-2 small</h2>
        <p>{result.tokens.length} prompt tokens.</p>
        <div class="token-list" aria-label="GPT-2 prompt tokens">
          {#each result.tokens as token}
            <span title={`Token ${token.id}`}>{JSON.stringify(token.text)}</span>
          {/each}
        </div>
        <h3>Next token</h3>
        <ol class="prediction-list">
          {#each result.predictions as prediction}
            <li>
              <code>{JSON.stringify(prediction.text)}</code><strong
                >{percent(prediction.probability)}</strong
              >
            </li>
          {/each}
        </ol>
        <h3>Greedy continuation</h3>
        <p class="continuation">{prompt}<mark>{result.continuation}</mark></p>
        <p class="muted">
          This cached ONNX model exposes logits and generated text, but does not expose attention
          matrices.
        </p>
      </article>
    </div>
  {/if}
</section>

<style>
  .comparison {
    display: grid;
    gap: 1.2rem;
  }
  .comparison-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1.2rem;
    align-items: start;
  }
  .comparison .panel {
    min-width: 0;
  }
  .comparison textarea {
    display: block;
    width: 100%;
    margin: 0.5rem 0 1rem;
    padding: 0.8rem;
    font: inherit;
  }
  .comparison button {
    cursor: pointer;
    border: 0;
  }
  .comparison button:disabled {
    cursor: wait;
    opacity: 0.5;
  }
  .token-list {
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
    max-height: 11rem;
    overflow-y: auto;
  }
  .token-list span {
    padding: 0.2rem 0.4rem;
    border: 1px solid var(--line);
    border-radius: 0.2rem;
    background: var(--paper-raised);
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.75rem;
    overflow-wrap: anywhere;
  }
  .prediction-list {
    padding-left: 1.5rem;
  }
  .prediction-list li {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    padding: 0.25rem 0;
    border-bottom: 1px solid var(--line);
  }
  .prediction-list code {
    overflow-wrap: anywhere;
  }
  .continuation {
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    line-height: 1.7;
  }
  .continuation mark {
    padding: 0.15rem;
    background: var(--lichen);
    color: var(--ink);
  }
  .muted {
    color: var(--muted);
    font-size: 0.9rem;
  }
  .success {
    color: var(--forest-deep);
    font-weight: 600;
  }
  @media (max-width: 1180px) {
    .comparison-grid {
      grid-template-columns: 1fr;
    }
  }
</style>
