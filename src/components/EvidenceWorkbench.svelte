<script lang="ts">
  import { onMount } from 'svelte';
  import HallucinationDemo from './HallucinationDemo.svelte';
  import { getCorpusManifest, searchCorpusEvidence } from '../lib/data/api';
  import type { EvidenceMatchPage, EvidenceSearchResponse } from '../lib/data/api-types';
  import type { CorpusManifest } from '../lib/data/schemas';

  let manifest = $state<CorpusManifest | null>(null);
  let loadingManifest = $state(true);
  let query = $state('');
  let result = $state<EvidenceSearchResponse | null>(null);
  let searching = $state(false);
  let error = $state('');
  let requestNumber = 0;

  const categories = $derived(
    result
      ? [
          {
            title: 'Exact phrase',
            explanation:
              'The entered words appear together in one exported field, with the same punctuation and accents. Letter case is ignored.',
            page: result.exactPhrase,
            testId: 'exact-phrase-results',
          },
          {
            title: 'Normalized terms',
            explanation:
              'All entered terms appear in one field. Case, accents, and word order are ignored; this does not prove the phrase occurs there.',
            page: result.normalizedTerms,
            testId: 'normalized-term-results',
          },
          {
            title: 'Species documents',
            explanation:
              'The terms match the common or scientific name of an exported species document.',
            page: result.speciesDocuments,
            testId: 'species-document-results',
          },
        ]
      : [],
  );
  const hasNext = $derived.by(() => {
    const current = result;
    return (
      current !== null &&
      categories.some((category) => (current.page + 1) * current.pageSize < category.page.total)
    );
  });
  const noHits = $derived(
    result !== null && categories.every((category) => category.page.total === 0),
  );

  async function runSearch(page = 0, searchText = query): Promise<void> {
    const submitted = searchText.trim();
    if (!submitted) {
      error = 'Enter words or a phrase to search.';
      return;
    }
    const currentRequest = ++requestNumber;
    searching = true;
    error = '';
    try {
      const found = await searchCorpusEvidence(submitted, page);
      if (currentRequest === requestNumber) result = found;
    } catch (cause) {
      if (currentRequest === requestNumber) {
        result = null;
        error = cause instanceof Error ? cause.message : 'Evidence search failed.';
      }
    } finally {
      if (currentRequest === requestNumber) searching = false;
    }
  }

  function countLabel(page: EvidenceMatchPage): string {
    if (!result || page.total === 0) return '0 matches';
    const noun = page.total === 1 ? 'match' : 'matches';
    if (page.hits.length === 0) return `${page.total.toLocaleString()} ${noun} · none on this page`;
    const first = result.page * result.pageSize + 1;
    const last = Math.min((result.page + 1) * result.pageSize, page.total);
    return `${page.total.toLocaleString()} ${noun} · showing ${first.toLocaleString()}–${last.toLocaleString()}`;
  }

  onMount(() => {
    void (async () => {
      try {
        manifest = await getCorpusManifest();
      } catch (cause) {
        error = cause instanceof Error ? cause.message : 'Could not read the corpus manifest.';
      } finally {
        loadingManifest = false;
      }
    })();
  });
</script>

<section class="evidence-bench" aria-labelledby="evidence-heading">
  <div class="lab-panel">
    <p class="eyebrow">M6.1 · Corpus evidence search</p>
    <h2 id="evidence-heading">Search the exported documents</h2>
    <p>
      This index searches only fields in the training-document template: species names and taxonomy,
      Wikipedia extracts, and section titles and text. Exported field craft and tags are excluded.
      Results can come from train, validation, or test; only the train split updated the model.
    </p>
    <p class="glossary-ref">
      Terms: <a href="/glossary#hallucination">hallucination</a> ·
      <a href="/glossary#validation-set">validation set</a>
    </p>
    {#if loadingManifest}
      <p role="status">Checking for an exported corpus…</p>
    {:else if manifest}
      <p class="provenance">
        {manifest.emittedRows.toLocaleString()} exported documents ·
        {new Date(manifest.exportedAt).toLocaleString()} · corpus
        <code>{manifest.corpusSha256.slice(0, 12)}</code>
      </p>
      <form
        class="search-form"
        onsubmit={(event) => {
          event.preventDefault();
          void runSearch();
        }}
      >
        <label for="evidence-query">Words or exact phrase</label>
        <div>
          <input
            id="evidence-query"
            type="search"
            bind:value={query}
            maxlength="160"
            placeholder="e.g. Wandering Albatross or open ocean"
          />
          <button type="submit" disabled={searching}>Search evidence</button>
        </div>
      </form>
    {:else}
      <p>
        No exported corpus is available yet. <a href="/data">Open the Data workbench</a> to create one.
      </p>
    {/if}
    {#if error}<p class="error" role="alert">{error}</p>{/if}
  </div>

  <HallucinationDemo />

  {#if searching}<p role="status">Searching the local corpus…</p>{/if}
  {#if result}
    <div class="lab-panel summary" aria-live="polite">
      <h2>Search results for “{result.query}”</h2>
      <p>
        Indexed {result.indexedDocuments.toLocaleString()} exported documents. Results below are source
        text matches, not a verdict about whether a claim is true. Exact and normalized counts are matching
        fields, so one species may appear more than once.
      </p>
      <p class="provenance">
        Searched export {new Date(result.exportedAt).toLocaleString()} · corpus
        <code>{result.corpusSha256.slice(0, 12)}</code>
      </p>
      {#if noHits}<p class="no-hits" data-testid="evidence-no-hits">
          No matching evidence found with these searches. A missing hit does not establish that the
          claim is false or absent from other sources.
        </p>{/if}
      <div class="paging">
        <button
          onclick={() => void runSearch((result?.page ?? 1) - 1, result?.query ?? query)}
          disabled={searching || result.page === 0}>Previous 20</button
        >
        <span>Page {result.page + 1}</span>
        <button
          onclick={() => void runSearch((result?.page ?? 0) + 1, result?.query ?? query)}
          disabled={searching || !hasNext}>Next 20</button
        >
      </div>
    </div>
    {#each categories as category}
      <section
        class="lab-panel result-group"
        aria-label={category.title}
        data-testid={category.testId}
      >
        <div class="group-heading">
          <h2>{category.title}</h2>
          <strong>{countLabel(category.page)}</strong>
        </div>
        <p class="note">{category.explanation}</p>
        {#if category.page.hits.length === 0}
          <p>No matches on this page.</p>
        {:else}
          <ol>
            {#each category.page.hits as item}
              <li>
                <div class="hit-heading">
                  <strong>{item.name}</strong>
                  <em>{item.sci}</em>
                  <code>{item.code}</code>
                </div>
                <p class="source"><span class="split">{item.split}</span> · {item.field}</p>
                <blockquote>{item.snippet}</blockquote>
              </li>
            {/each}
          </ol>
        {/if}
      </section>
    {/each}
  {/if}
</section>

<style>
  .evidence-bench {
    display: grid;
    gap: 1.25rem;
    max-width: 1100px;
    min-width: 0;
  }
  h2 {
    margin: 0 0 0.7rem;
  }
  p {
    line-height: 1.55;
  }
  .provenance,
  .note,
  .source {
    color: var(--muted);
    font-size: 0.84rem;
  }
  .provenance code {
    overflow-wrap: anywhere;
  }
  .search-form {
    margin-top: 1.4rem;
  }
  .search-form label {
    display: block;
    margin-bottom: 0.45rem;
    font-weight: 700;
  }
  .search-form > div {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
  }
  input {
    flex: 1 1 20rem;
    min-width: 0;
    padding: 0.7rem;
    border: 1px solid var(--line-strong);
    background: var(--paper);
    color: var(--ink);
    font: inherit;
  }
  button {
    padding: 0.65rem 0.9rem;
    border: 1px solid var(--forest);
    background: var(--paper);
    color: var(--ink);
    cursor: pointer;
  }
  button:disabled {
    opacity: 0.45;
    cursor: default;
  }
  .error {
    color: #af421c;
  }
  .no-hits {
    padding: 0.9rem;
    border-left: 3px solid var(--forest);
    background: var(--paper);
  }
  .paging {
    display: flex;
    gap: 0.8rem;
    align-items: center;
    flex-wrap: wrap;
    margin-top: 1rem;
  }
  .result-group {
    min-width: 0;
  }
  .group-heading {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    align-items: baseline;
    flex-wrap: wrap;
  }
  .group-heading strong {
    color: var(--forest-deep);
    font-size: 0.85rem;
  }
  ol {
    display: grid;
    gap: 0;
    padding-left: 1.3rem;
  }
  li {
    padding: 0.8rem 0.3rem;
    border-bottom: 1px solid var(--line);
  }
  li::marker {
    color: var(--muted);
  }
  .hit-heading {
    display: flex;
    gap: 0.55rem;
    flex-wrap: wrap;
    align-items: baseline;
  }
  .hit-heading em,
  .hit-heading code {
    color: var(--muted);
    font-size: 0.82rem;
  }
  .source {
    margin: 0.3rem 0;
  }
  .split {
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  blockquote {
    margin: 0;
    padding: 0.6rem 0.8rem;
    background: var(--paper);
    border-left: 2px solid var(--line-strong);
    overflow-wrap: anywhere;
    line-height: 1.5;
  }
</style>
