<script lang="ts">
  import { onMount } from 'svelte';
  import { checkpointCatalog } from '../lib/stores/checkpoints';
  import { loadInspectionModel, type InspectionModel } from '../lib/data/inspection';
  import { initializeParameters } from '../lib/model/parameters';
  import type { Tensor } from '../lib/math/tensor';
  import {
    fitEmbeddingPca,
    nearestCosine,
    projectEmbeddings,
    type PcaBasis,
  } from '../lib/trace/embeddings';
  import { formatTraceValue, heatAlpha } from '../lib/trace/display';
  import PcaMap from './PcaMap.svelte';

  interface EmbeddingView {
    readonly model: InspectionModel;
    readonly initial: Tensor;
    readonly trained: Tensor;
    readonly basis: PcaBasis;
    readonly initialPoints: Float32Array;
    readonly trainedPoints: Float32Array;
    readonly bound: number;
  }

  let view = $state.raw<EmbeddingView | null>(null);
  let loading = $state(false);
  let error = $state<string | null>(null);
  let query = $state('');
  let selectedId = $state(97);
  let selectedChannel = $state(0);
  let requestNumber = 0;

  const matches = $derived.by(() => {
    if (!view) return [];
    const normalized = query.trim().toLocaleLowerCase();
    if (!normalized) return view.model.tokenizer.tokens;
    return view.model.tokenizer.tokens.filter(
      (token) =>
        String(token.id).includes(normalized) ||
        token.display.toLocaleLowerCase().includes(normalized),
    );
  });
  const initialNeighbours = $derived(view ? nearestCosine(view.initial, selectedId, 8) : []);
  const trainedNeighbours = $derived(view ? nearestCosine(view.trained, selectedId, 8) : []);
  const width = $derived(view?.trained.shape[1] ?? 0);
  const initialValue = $derived(view?.initial.data[selectedId * width + selectedChannel] ?? 0);
  const trainedValue = $derived(view?.trained.data[selectedId * width + selectedChannel] ?? 0);

  async function load(id: string): Promise<void> {
    const request = ++requestNumber;
    loading = true;
    error = null;
    try {
      const model = await loadInspectionModel();
      if (request !== requestNumber || model.checkpointId !== id) return;
      const initial = initializeParameters(model.config.model, model.config.seed).get(
        'token_embedding.weight',
      ).value;
      const trained = model.registry.get('token_embedding.weight').value;
      const basis = fitEmbeddingPca(trained);
      const initialPoints = projectEmbeddings(initial, basis);
      const trainedPoints = projectEmbeddings(trained, basis);
      let bound = 1e-12;
      for (const point of initialPoints) bound = Math.max(bound, Math.abs(point));
      for (const point of trainedPoints) bound = Math.max(bound, Math.abs(point));
      view = { model, initial, trained, basis, initialPoints, trainedPoints, bound };
      selectedId = Math.min(97, model.config.model.vocabSize - 1);
      selectedChannel = 0;
    } catch (cause) {
      if (request !== requestNumber) return;
      error = cause instanceof Error ? cause.message : 'Embeddings unavailable.';
      view = null;
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
        view = null;
        loading = false;
      }
    });
    return () => {
      requestNumber += 1;
      unsubscribe();
    };
  });
</script>

<section class="embeddings" aria-label="Embedding laboratory">
  <div class="panel">
    <p class="eyebrow">One row per token id</p>
    <h2>Choose a token</h2>
    <p>
      An embedding is a learned row of {width || 'model-width'} Float32 numbers. Search by token id or
      visible text; the selected row and neighbours below use the active checkpoint.
    </p>
    <label for="embedding-search">Search tokens</label>
    <input id="embedding-search" bind:value={query} placeholder="Token text or id" />
    <p>{matches.length} matching tokens</p>
    <label for="embedding-token">Token list</label>
    <select
      id="embedding-token"
      size="6"
      value={selectedId}
      onchange={(event) => (selectedId = Number(event.currentTarget.value))}
    >
      {#each matches as token}<option value={token.id}
          >#{token.id} · {token.display || 'empty bytes'}</option
        >{/each}
    </select>
    {#if view}<small
        >Checkpoint {view.model.checkpointId} · step {view.model.config.trainingStep}</small
      >{/if}
  </div>

  {#if loading}<p role="status">Loading embeddings and fitting PCA…</p>{/if}
  {#if error}<p role="alert" class="error">{error}</p>{/if}
  {#if !loading && !view && !error}<p>Select a validated checkpoint in the sidebar.</p>{/if}

  {#if view}
    <div class="panel">
      <h2>Selected token #{selectedId} · {view.model.tokenizer.tokens[selectedId]?.display}</h2>
      <p>
        Click or focus a channel to inspect its stored trained value and reconstructed initial
        value. The initial row is regenerated from the checkpoint seed and model configuration; it
        was not stored in the checkpoint.
      </p>
      <div class="strip" role="group" aria-label="Trained embedding channels">
        {#each Array(width) as _, channel}
          {@const value = view.trained.data[selectedId * width + channel]}
          <button
            class:selected={selectedChannel === channel}
            aria-pressed={selectedChannel === channel}
            aria-label={`Channel ${channel}, trained value ${value}`}
            title={`Full Float32 value: ${value}`}
            style={`background: ${value < 0 ? `rgba(213, 108, 63, ${heatAlpha(value)})` : `rgba(185, 206, 144, ${heatAlpha(value)})`}`}
            onclick={() => (selectedChannel = channel)}
          >
            <small>{channel}</small>{formatTraceValue(value, 3)}
          </button>
        {/each}
      </div>
      <details open>
        <summary>What's happening</summary>
        <p>
          Training changes the token's weight row. These numbers become the token contribution at
          every input position where this id appears.
        </p>
      </details>
      <details open>
        <summary>The actual math · channel {selectedChannel}</summary>
        <dl>
          <div>
            <dt>Initial, reconstructed</dt>
            <dd>{initialValue}</dd>
          </div>
          <div>
            <dt>Trained, stored</dt>
            <dd>{trainedValue}</dd>
          </div>
          <div>
            <dt>Change</dt>
            <dd>{trainedValue - initialValue}</dd>
          </div>
        </dl>
        <p>
          <code>change = trained − initial</code>. Full Float32 values are shown; a rounded strip
          may not make the subtraction appear exact.
        </p>
      </details>
    </div>

    <div class="panel">
      <h2>Two-dimensional map</h2>
      <p>
        The PCA basis is fitted to the trained embedding rows. The initial reconstructed rows and
        trained stored rows are projected onto that same basis with the same axis scale, so their
        coordinates can be compared. PCA keeps only two directions; nearby dots here do not prove
        the tokens mean the same thing.
      </p>
      <div class="maps">
        <PcaMap
          title="Initial · reconstructed"
          points={view.initialPoints}
          bound={view.bound}
          {selectedId}
          tokenizer={view.model.tokenizer}
        />
        <PcaMap
          title="Trained · stored"
          points={view.trainedPoints}
          bound={view.bound}
          {selectedId}
          tokenizer={view.model.tokenizer}
        />
      </div>
      <details>
        <summary>The actual math · PCA method</summary>
        <p>
          Subtract each trained channel mean. Compute sample covariance by dividing each centred
          cross-product sum by token count minus one. Symmetric Jacobi rotations diagonalize that
          covariance. Sort eigenvalues descending; orient each axis so its largest-magnitude loading
          is positive. Both maps use the trained mean and axes.
        </p>
        <dl>
          <div>
            <dt>PC1 eigenvalue</dt>
            <dd>{view.basis.eigenvalues[0]}</dd>
          </div>
          <div>
            <dt>PC2 eigenvalue</dt>
            <dd>{view.basis.eigenvalues[1]}</dd>
          </div>
          <div>
            <dt>Jacobi sweeps</dt>
            <dd>{view.basis.sweeps}</dd>
          </div>
          <div>
            <dt>Relative tolerance</dt>
            <dd>{view.basis.tolerance}</dd>
          </div>
          <div>
            <dt>Selected initial point</dt>
            <dd>
              ({view.initialPoints[selectedId * 2]}, {view.initialPoints[selectedId * 2 + 1]})
            </dd>
          </div>
          <div>
            <dt>Selected trained point</dt>
            <dd>
              ({view.trainedPoints[selectedId * 2]}, {view.trainedPoints[selectedId * 2 + 1]})
            </dd>
          </div>
        </dl>
      </details>
    </div>

    <div class="panel">
      <h2>Nearest rows by cosine similarity</h2>
      <p>
        Cosine compares vector direction, not length. These are nearest rows in this tiny
        checkpoint; a high score alone does not establish a semantic relationship.
      </p>
      <div class="neighbours">
        <div>
          <h3>Initial · reconstructed</h3>
          <ol>
            {#each initialNeighbours as item}<li>
                <span>#{item.id} {view.model.tokenizer.tokens[item.id]?.display}</span><code
                  >{item.similarity}</code
                >
              </li>{/each}
          </ol>
        </div>
        <div>
          <h3>Trained · stored</h3>
          <ol>
            {#each trainedNeighbours as item}<li>
                <span>#{item.id} {view.model.tokenizer.tokens[item.id]?.display}</span><code
                  >{item.similarity}</code
                >
              </li>{/each}
          </ol>
        </div>
      </div>
      <details>
        <summary>The actual math · cosine</summary>
        <p>
          <code>cosine(a, b) = sum(aᵢ bᵢ) / sqrt(sum(aᵢ²) sum(bᵢ²))</code>. Zero-length rows have no
          cosine neighbour result. Ties sort by token id.
        </p>
      </details>
    </div>
  {/if}
</section>

<style>
  .embeddings {
    display: grid;
    gap: 1.2rem;
    min-width: 0;
  }
  .panel {
    min-width: 0;
    padding: 1.4rem;
    color: #eef5e8;
    background: #1c251e;
    border: 1px solid #53614d;
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
  input,
  select {
    max-width: 100%;
    width: min(100%, 32rem);
    padding: 0.5rem;
    color: #eef5e8;
    background: #253328;
    border: 1px solid #60785b;
  }
  select {
    display: block;
  }
  input:focus-visible,
  select:focus-visible,
  button:focus-visible,
  summary:focus-visible {
    outline: 2px solid #f0c576;
  }
  small {
    color: #b9c8b2;
  }
  .panel > small {
    display: block;
    margin-top: 0.7rem;
  }
  .strip {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(5.2rem, 1fr));
    gap: 0.25rem;
  }
  .strip button {
    display: grid;
    min-height: 2.8rem;
    border: 1px solid #60785b;
    color: #eef5e8;
    font:
      0.7rem ui-monospace,
      SFMono-Regular,
      Menlo,
      monospace;
    cursor: pointer;
  }
  .strip button.selected {
    outline: 2px solid #f0c576;
  }
  .strip small {
    font-size: 0.6rem;
  }
  details {
    margin-top: 1rem;
  }
  summary {
    cursor: pointer;
    font-weight: 700;
  }
  dl {
    display: grid;
    gap: 0.35rem;
  }
  dl div {
    display: flex;
    flex-wrap: wrap;
    gap: 0.7rem;
  }
  dt {
    min-width: 10rem;
    color: #b9c8b2;
  }
  dd {
    margin: 0;
    overflow-wrap: anywhere;
  }
  .maps,
  .neighbours {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1rem;
  }
  .neighbours ol {
    padding-left: 1.7rem;
  }
  .neighbours li {
    padding: 0.3rem 0;
    overflow-wrap: anywhere;
  }
  .neighbours li span,
  .neighbours li code {
    display: block;
  }
  .error {
    color: #a43922;
  }
  @media (max-width: 1150px) {
    .maps,
    .neighbours {
      grid-template-columns: 1fr;
    }
  }
</style>
