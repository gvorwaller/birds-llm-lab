<script lang="ts">
  interface Point {
    readonly step: number;
    readonly value: number;
  }
  interface Series {
    readonly label: string;
    readonly color: string;
    readonly points: readonly Point[];
  }
  interface Props {
    title: string;
    maxStep: number;
    series: readonly Series[];
  }
  let { title, maxStep, series }: Props = $props();
  const width = 600;
  const height = 170;
  const left = 50;
  const right = 12;
  const top = 14;
  const bottom = 28;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;
  const values = $derived(series.flatMap((item) => item.points.map((point) => point.value)));
  const low = $derived(values.length ? Math.min(...values) : 0);
  const high = $derived(values.length ? Math.max(...values) : 1);
  const range = $derived(high - low || Math.max(Math.abs(high) * 0.1, 1e-6));
  const lower = $derived(low - range * 0.08);
  const upper = $derived(high + range * 0.08);

  function path(points: readonly Point[]): string {
    return points
      .map((point, index) => {
        const x = left + (Math.max(0, Math.min(maxStep, point.step)) / maxStep) * plotWidth;
        const y = top + ((upper - point.value) / (upper - lower)) * plotHeight;
        return `${index === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`;
      })
      .join(' ');
  }
</script>

<div class="plot">
  <h3>{title}</h3>
  <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${title} by training step`}>
    <title>{title} by training step</title>
    <line x1={left} y1={top} x2={left} y2={height - bottom} class="axis" />
    <line x1={left} y1={height - bottom} x2={width - right} y2={height - bottom} class="axis" />
    <text x="4" y={top + 5}>{upper.toPrecision(3)}</text>
    <text x="4" y={height - bottom}>{lower.toPrecision(3)}</text>
    <text x={left} y={height - 6}>0</text>
    <text x={width - right} y={height - 6} text-anchor="end">{maxStep} steps</text>
    {#each series as item}
      {#if item.points.length > 0}
        <path
          d={path(item.points)}
          fill="none"
          stroke={item.color}
          stroke-width="2.5"
          stroke-linecap="round"
          stroke-linejoin="round"
        />
        {#if item.points.length === 1}
          <circle
            cx={left + (item.points[0].step / maxStep) * plotWidth}
            cy={top + ((upper - item.points[0].value) / (upper - lower)) * plotHeight}
            r="4"
            fill={item.color}
          />
        {/if}
      {/if}
    {/each}
  </svg>
  <div class="legend">
    {#each series as item}<span><i style:background={item.color}></i>{item.label}</span>{/each}
  </div>
  {#if values.length === 0}<p>No displayed steps yet.</p>{/if}
</div>

<style>
  .plot {
    min-width: 0;
    padding: 1rem;
    border: 1px solid var(--line);
    background: var(--paper-raised);
  }
  h3 {
    margin: 0 0 0.7rem;
    font-family: Georgia, 'Times New Roman', serif;
    font-weight: 500;
  }
  svg {
    display: block;
    width: 100%;
    height: auto;
  }
  .axis {
    stroke: var(--line-strong);
    stroke-width: 1;
  }
  text {
    fill: var(--muted);
    font:
      11px ui-monospace,
      SFMono-Regular,
      Menlo,
      monospace;
  }
  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: 0.8rem;
    font-size: 0.78rem;
  }
  .legend span {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
  }
  .legend i {
    display: inline-block;
    width: 1rem;
    height: 0.2rem;
  }
  p {
    color: var(--muted);
    font-size: 0.8rem;
  }
</style>
