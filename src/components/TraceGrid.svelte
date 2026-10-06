<script lang="ts">
  import type { TraceEntry } from '../lib/trace/forward-trace';
  import { formatTraceValue, heatAlpha } from '../lib/trace/display';

  interface Props {
    entry: TraceEntry;
    plane: number;
    selected: readonly number[];
    onSelect: (indices: number[]) => void;
  }

  let { entry, plane, selected, onSelect }: Props = $props();
  const rows = $derived(entry.output.shape[1]);
  const columns = $derived(entry.output.shape[2]);

  function value(row: number, column: number): number {
    return entry.output.data[(plane * rows + row) * columns + column];
  }
</script>

<div class="grid-scroll" role="region" aria-label={`${entry.name} values`}>
  <table>
    <thead>
      <tr
        ><th scope="col">Position</th>{#each Array(columns) as _, column}<th scope="col"
            >{column}</th
          >{/each}</tr
      >
    </thead>
    <tbody>
      {#each Array(rows) as _, row}
        <tr>
          <th scope="row">{row}</th>
          {#each Array(columns) as _, column}
            {@const current = value(row, column)}
            <td>
              <button
                class:selected={selected[0] === plane &&
                  selected[1] === row &&
                  selected[2] === column}
                style={`background: ${current < 0 ? `rgba(213, 108, 63, ${heatAlpha(current)})` : `rgba(185, 206, 144, ${heatAlpha(current)})`}`}
                title={`Full Float32 value: ${current}`}
                aria-label={`Position ${row}, column ${column}, value ${current}`}
                aria-pressed={selected[0] === plane &&
                  selected[1] === row &&
                  selected[2] === column}
                onclick={() => onSelect([plane, row, column])}
                >{formatTraceValue(current, 3)}</button
              >
            </td>
          {/each}
        </tr>
      {/each}
    </tbody>
  </table>
</div>

<style>
  .grid-scroll {
    max-width: 100%;
    overflow: auto;
    border: 1px solid #53614d;
  }
  table {
    border-collapse: collapse;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.65rem;
  }
  th {
    position: sticky;
    top: 0;
    background: #202820;
    color: #dce6d5;
    padding: 0.3rem;
  }
  tbody th {
    left: 0;
    z-index: 1;
  }
  td {
    padding: 1px;
  }
  button {
    min-width: 4.7rem;
    min-height: 2rem;
    border: 1px solid #40503b;
    color: #f5f8ee;
    font: inherit;
    cursor: pointer;
  }
  button:hover,
  button:focus-visible,
  button.selected {
    outline: 2px solid #f0c576;
    outline-offset: -2px;
  }
</style>
