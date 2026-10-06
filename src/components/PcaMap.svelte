<script lang="ts">
  import type { TokenizerArtifact } from '../lib/data/schemas';
  interface Props {
    title: string;
    points: Float32Array;
    bound: number;
    selectedId: number;
    tokenizer: TokenizerArtifact;
  }
  let { title, points, bound, selectedId, tokenizer }: Props = $props();
  const count = $derived(points.length / 2);
  const x = (id: number) => 150 + (points[id * 2] / bound) * 138;
  const y = (id: number) => 150 - (points[id * 2 + 1] / bound) * 138;
</script>

<figure>
  <figcaption>{title}</figcaption>
  <svg viewBox="0 0 300 300" role="img" aria-label={`${title} PCA map of ${count} tokens`}>
    <rect x="0" y="0" width="300" height="300" fill="#172019" />
    <line x1="150" y1="0" x2="150" y2="300" stroke="#64725f" />
    <line x1="0" y1="150" x2="300" y2="150" stroke="#64725f" />
    {#each Array(count) as _, id}
      <circle
        cx={x(id)}
        cy={y(id)}
        r={id === selectedId ? 5 : 1.5}
        fill={id === selectedId ? '#f0c576' : '#a9c893'}
        opacity={id === selectedId ? 1 : 0.55}
      >
        <title
          >#{id}
          {tokenizer.tokens[id]?.display || 'empty'} · ({points[id * 2]}, {points[
            id * 2 + 1
          ]})</title
        >
      </circle>
    {/each}
  </svg>
  <small>PC1 → · PC2 ↑ · highlighted token #{selectedId}</small>
</figure>

<style>
  figure {
    margin: 0;
    min-width: 0;
  }
  figcaption {
    margin-bottom: 0.5rem;
    font-weight: 700;
  }
  svg {
    display: block;
    width: 100%;
    max-width: 360px;
    border: 1px solid #53614d;
  }
  small {
    display: block;
    margin-top: 0.35rem;
    color: #b9c8b2;
  }
</style>
