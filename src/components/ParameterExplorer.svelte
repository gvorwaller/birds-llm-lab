<script lang="ts">
  import { onMount } from 'svelte';
  import { checkpointCatalog } from '../lib/stores/checkpoints';
  import { loadInspectionModel } from '../lib/data/inspection';
  import { formatTraceValue } from '../lib/trace/display';
  import {
    buildParameterOverview,
    HEATMAP_TILE,
    type ParameterOverview,
  } from '../lib/parameters/explorer';

  let overview = $state.raw<ParameterOverview | null>(null);
  let checkpointId = $state('');
  let selectedName = $state('');
  let rowStart = $state(0);
  let columnStart = $state(0);
  let selectedRow = $state(0);
  let selectedColumn = $state(0);
  let loading = $state(false);
  let error = $state('');
  let canvas = $state<HTMLCanvasElement | null>(null);
  let requestNumber = 0;

  const selected = $derived(overview?.entries.find((entry) => entry.name === selectedName) ?? null);
  const rows = $derived(selected ? (selected.shape.length === 1 ? 1 : selected.shape[0]) : 0);
  const columns = $derived(
    selected ? (selected.shape.length === 1 ? selected.shape[0] : selected.shape[1]) : 0,
  );
  const tileRows = $derived(Math.min(HEATMAP_TILE, rows - rowStart));
  const tileColumns = $derived(Math.min(HEATMAP_TILE, columns - columnStart));
  const selectedValue = $derived(
    selected?.parameter.value.data[selectedRow * columns + selectedColumn] ?? 0,
  );
  const peakBin = $derived(selected ? Math.max(...selected.histogram) : 1);

  function selectParameter(name: string): void {
    selectedName = name;
    rowStart = 0;
    columnStart = 0;
    selectedRow = 0;
    selectedColumn = 0;
  }

  async function load(id: string): Promise<void> {
    const request = ++requestNumber;
    loading = true;
    error = '';
    overview = null;
    try {
      const model = await loadInspectionModel();
      if (request !== requestNumber || model.checkpointId !== id) return;
      const next = buildParameterOverview(model.weightIndex, model.registry, model.config.model);
      overview = next;
      checkpointId = id;
      selectParameter(next.entries[0]?.name ?? '');
    } catch (cause) {
      if (request === requestNumber)
        error = cause instanceof Error ? cause.message : 'The checkpoint could not be inspected.';
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
      if (id) void load(id);
      else {
        requestNumber += 1;
        overview = null;
        loading = false;
      }
    });
    return () => {
      requestNumber += 1;
      unsubscribe();
    };
  });

  $effect(() => {
    if (!canvas || !selected || tileRows <= 0 || tileColumns <= 0) return;
    const context = canvas.getContext('2d');
    if (!context) return;
    const cell = 12;
    canvas.width = tileColumns * cell;
    canvas.height = tileRows * cell;
    const bound = Math.max(Math.abs(selected.min), Math.abs(selected.max), 1e-12);
    for (let row = 0; row < tileRows; row += 1) {
      for (let column = 0; column < tileColumns; column += 1) {
        const value =
          selected.parameter.value.data[(rowStart + row) * columns + columnStart + column];
        const intensity = Math.round(55 + 190 * Math.min(1, Math.abs(value) / bound));
        context.fillStyle = value < 0 ? `rgb(${intensity},85,61)` : `rgb(62,${intensity},119)`;
        context.fillRect(column * cell, row * cell, cell, cell);
      }
    }
    if (
      selectedRow >= rowStart &&
      selectedRow < rowStart + tileRows &&
      selectedColumn >= columnStart &&
      selectedColumn < columnStart + tileColumns
    ) {
      context.strokeStyle = '#fff';
      context.lineWidth = 2;
      context.strokeRect(
        (selectedColumn - columnStart) * cell + 1,
        (selectedRow - rowStart) * cell + 1,
        cell - 2,
        cell - 2,
      );
    }
  });

  function chooseCell(event: MouseEvent): void {
    if (!canvas) return;
    const bounds = canvas.getBoundingClientRect();
    const column = Math.floor(((event.clientX - bounds.left) / bounds.width) * tileColumns);
    const row = Math.floor(((event.clientY - bounds.top) / bounds.height) * tileRows);
    selectedRow = rowStart + Math.max(0, Math.min(tileRows - 1, row));
    selectedColumn = columnStart + Math.max(0, Math.min(tileColumns - 1, column));
  }

  function moveCell(event: KeyboardEvent): void {
    if (event.key === 'ArrowUp') setRow(selectedRow - 1);
    else if (event.key === 'ArrowDown') setRow(selectedRow + 1);
    else if (event.key === 'ArrowLeft') setColumn(selectedColumn - 1);
    else if (event.key === 'ArrowRight') setColumn(selectedColumn + 1);
    else return;
    event.preventDefault();
  }

  function setRow(value: number): void {
    selectedRow = Math.max(0, Math.min(rows - 1, Math.trunc(value) || 0));
    rowStart = Math.floor(selectedRow / HEATMAP_TILE) * HEATMAP_TILE;
  }

  function setColumn(value: number): void {
    selectedColumn = Math.max(0, Math.min(columns - 1, Math.trunc(value) || 0));
    columnStart = Math.floor(selectedColumn / HEATMAP_TILE) * HEATMAP_TILE;
  }

  function shiftRows(direction: -1 | 1): void {
    setRow(Math.max(0, Math.min(rows - 1, rowStart + direction * HEATMAP_TILE)));
  }

  function shiftColumns(direction: -1 | 1): void {
    setColumn(Math.max(0, Math.min(columns - 1, columnStart + direction * HEATMAP_TILE)));
  }
</script>

<section class="parameter-explorer" aria-label="Parameter explorer">
  <div class="lab-panel introduction">
    <p class="eyebrow">M6.3 · Stored checkpoint weights</p>
    <h2>Every named parameter, one place</h2>
    <p>
      Choose a tensor to inspect a summary of its stored Float32 values and a 32 × 32 window. The
      tree comes from the checkpoint weight index. Colors show sign and relative magnitude; the
      selected cell shows the stored number.
    </p>
    {#if loading}<p role="status">Loading and summarizing checkpoint weights…</p>{/if}
    {#if error}<p class="error" role="alert">{error}</p>{/if}
    {#if !loading && !overview && !error}<p>Select a validated checkpoint in the sidebar.</p>{/if}
    {#if overview}
      <p data-testid="parameter-reconciliation">
        <strong>{overview.tensorCount}</strong> named tensors ·
        <strong>{overview.elementCount.toLocaleString()}</strong>
        values · expected from model config
        <strong>{overview.expectedCount.toLocaleString()}</strong>
        · checkpoint <code>{checkpointId}</code>
      </p>
    {/if}
  </div>

  {#if overview && selected}
    <div class="explorer-layout">
      <nav class="lab-panel tree" aria-label="Checkpoint parameters">
        <h2>Parameter tree</h2>
        {#each overview.groups as group}
          <details open>
            <summary>{group.name} <small>{group.count.toLocaleString()} values</small></summary>
            <ul>
              {#each group.entries as entry}
                <li>
                  <button
                    class:chosen={entry.name === selectedName}
                    aria-current={entry.name === selectedName ? 'true' : undefined}
                    onclick={() => selectParameter(entry.name)}
                    ><span>{entry.name}</span><small
                      >[{entry.shape.join(' × ')}] · {entry.count.toLocaleString()}</small
                    ></button
                  >
                </li>
              {/each}
            </ul>
          </details>
        {/each}
      </nav>
      <section class="lab-panel detail" aria-label="Selected parameter">
        <p class="eyebrow">Stored tensor</p>
        <h2>{selected.name}</h2>
        <p>
          Shape [{selected.shape.join(' × ')}] · {selected.count.toLocaleString()} Float32 values · weight-index
          offset {selected.elementOffset.toLocaleString()}
        </p>
        <dl class="stats">
          <div>
            <dt>Count</dt>
            <dd>{selected.count.toLocaleString()}</dd>
          </div>
          <div>
            <dt>Minimum</dt>
            <dd>{formatTraceValue(selected.min, 6)}</dd>
          </div>
          <div>
            <dt>Maximum</dt>
            <dd>{formatTraceValue(selected.max, 6)}</dd>
          </div>
          <div>
            <dt>Mean</dt>
            <dd>{formatTraceValue(selected.mean, 6)}</dd>
          </div>
          <div>
            <dt>Population std</dt>
            <dd>{formatTraceValue(selected.std, 6)}</dd>
          </div>
        </dl>
        <h3>Value histogram</h3>
        <p class="note">
          20 equal-width bins from minimum to maximum; bar height counts stored values. A constant
          tensor occupies the first bin.
        </p>
        <div
          class="histogram"
          role="img"
          aria-label={`20-bin histogram for ${selected.name}, ${selected.count} values`}
        >
          {#each selected.histogram as count, bin}
            <div
              title={`Bin ${bin + 1}: ${count} values`}
              style={`height: ${Math.max(2, (count / peakBin) * 100)}%`}
            ></div>
          {/each}
        </div>
        <p class="histogram-axis">
          <span>{formatTraceValue(selected.min, 4)}</span><span
            >{formatTraceValue(selected.max, 4)}</span
          >
        </p>
        <h3>Weight heatmap</h3>
        <p class="note">
          Positive values are green, negative values orange. Brightness uses the largest absolute
          value in this tensor. One window has at most {HEATMAP_TILE * HEATMAP_TILE} cells.
        </p>
        <div class="tile-controls">
          <button onclick={() => shiftRows(-1)} disabled={rowStart === 0}>Previous rows</button>
          <button onclick={() => shiftRows(1)} disabled={rowStart + HEATMAP_TILE >= rows}
            >Next rows</button
          >
          <button onclick={() => shiftColumns(-1)} disabled={columnStart === 0}
            >Previous columns</button
          >
          <button onclick={() => shiftColumns(1)} disabled={columnStart + HEATMAP_TILE >= columns}
            >Next columns</button
          >
        </div>
        <p class="note">
          Rows {rowStart}–{rowStart + tileRows - 1}; columns {columnStart}–{columnStart +
            tileColumns -
            1}.
        </p>
        <div class="heatmap-scroll">
          <canvas
            bind:this={canvas}
            onclick={chooseCell}
            role="button"
            tabindex="0"
            onkeydown={moveCell}
            aria-label={`Heatmap window for ${selected.name}; arrow keys select a cell`}
          ></canvas>
        </div>
        <div class="cell-controls">
          <label
            >Row <input
              type="number"
              min="0"
              max={rows - 1}
              value={selectedRow}
              onchange={(event) => setRow(Number(event.currentTarget.value))}
            /></label
          >
          <label
            >Column <input
              type="number"
              min="0"
              max={columns - 1}
              value={selectedColumn}
              onchange={(event) => setColumn(Number(event.currentTarget.value))}
            /></label
          >
        </div>
        <p class="cell-value" data-testid="parameter-cell-value">
          [{selectedRow}, {selectedColumn}] = <code>{selectedValue}</code>
        </p>
        <a href={`/forward-pass?stage=${encodeURIComponent(selected.traceStage)}`}
          >See this parameter in the {selected.traceStage} forward trace →</a
        >
      </section>
    </div>
  {/if}
</section>

<style>
  .parameter-explorer {
    display: grid;
    gap: 1rem;
    min-width: 0;
  }
  .introduction h2,
  .tree h2,
  .detail h2 {
    margin-top: 0;
  }
  .explorer-layout {
    display: grid;
    grid-template-columns: minmax(16rem, 0.8fr) minmax(0, 1.5fr);
    gap: 1rem;
    min-width: 0;
  }
  .tree,
  .detail {
    min-width: 0;
  }
  .tree {
    max-height: 70vh;
    overflow: auto;
  }
  .tree details {
    border-top: 1px solid var(--line);
    padding: 0.5rem 0;
  }
  .tree summary {
    cursor: pointer;
    font-weight: 600;
  }
  .tree summary small {
    color: var(--muted);
    font-weight: 400;
    margin-left: 0.4rem;
  }
  .tree ul {
    list-style: none;
    margin: 0.4rem 0 0;
    padding: 0;
  }
  .tree li button {
    display: grid;
    text-align: left;
    width: 100%;
    padding: 0.45rem;
    color: var(--ink);
    background: transparent;
    border: 0;
    cursor: pointer;
    overflow-wrap: anywhere;
  }
  .tree li button:hover,
  .tree li button:focus-visible,
  .tree li button.chosen {
    background: var(--paper-raised);
    outline: 1px solid var(--forest);
  }
  .tree li small {
    color: var(--muted);
  }
  .stats {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(9rem, 1fr));
    gap: 0.5rem;
  }
  .stats div {
    padding: 0.5rem;
    border: 1px solid var(--line);
  }
  .stats dt {
    color: var(--muted);
    font-size: 0.8rem;
  }
  .stats dd {
    margin: 0.2rem 0 0;
    font:
      0.85rem ui-monospace,
      SFMono-Regular,
      Menlo,
      monospace;
  }
  .note {
    color: var(--muted);
    font-size: 0.84rem;
  }
  .histogram {
    height: 7rem;
    display: flex;
    align-items: end;
    gap: 2px;
    border-bottom: 1px solid var(--ink);
  }
  .histogram div {
    flex: 1;
    background: var(--forest);
    min-width: 0;
  }
  .histogram-axis {
    display: flex;
    justify-content: space-between;
    font:
      0.75rem ui-monospace,
      SFMono-Regular,
      Menlo,
      monospace;
  }
  .tile-controls,
  .cell-controls {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin: 0.7rem 0;
  }
  .tile-controls button {
    background: var(--paper-raised);
    border: 1px solid var(--line);
    padding: 0.45rem;
    cursor: pointer;
  }
  .tile-controls button:disabled {
    opacity: 0.5;
    cursor: default;
  }
  .heatmap-scroll {
    max-width: 100%;
    overflow: auto;
    border: 1px solid var(--line);
    background: #17231c;
  }
  canvas {
    display: block;
    image-rendering: pixelated;
    cursor: crosshair;
  }
  .cell-controls label {
    display: grid;
    gap: 0.2rem;
  }
  .cell-controls input {
    width: 6rem;
    padding: 0.35rem;
  }
  .cell-value {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  .error {
    color: #af421c;
  }
  @media (max-width: 800px) {
    .explorer-layout {
      grid-template-columns: 1fr;
    }
    .tree {
      max-height: 22rem;
    }
  }
</style>
