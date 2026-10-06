<script lang="ts">
  import { onMount } from 'svelte';
  import { glossaryById, glossaryTerms } from '../lib/glossary/terms';

  function focusHash(): void {
    const id = decodeURIComponent(window.location.hash.slice(1));
    if (!glossaryById.has(id)) return;
    requestAnimationFrame(() => {
      const target = document.getElementById(id);
      target?.scrollIntoView({ block: 'start' });
      target?.focus({ preventScroll: true });
    });
  }

  onMount(() => {
    focusHash();
    window.addEventListener('hashchange', focusHash);
    return () => window.removeEventListener('hashchange', focusHash);
  });
</script>

<section class="glossary" aria-label="Model glossary">
  <div class="lab-panel introduction">
    <p class="eyebrow">M6.4 · Plain language and actual math</p>
    <h2>Find a term, then inspect it in the lab</h2>
    <p>
      Each definition describes this app’s small transformer. Formulas show the operation where one
      applies; the workbench links show real checkpoint or training values. Metaphors are marked and
      followed by the literal calculation.
    </p>
    <nav class="term-index" aria-label="Glossary term index">
      {#each glossaryTerms as term}
        <a href={`#${term.id}`}>{term.label}</a>
      {/each}
    </nav>
  </div>

  <div class="terms">
    {#each glossaryTerms as term}
      <article
        class="lab-panel term"
        id={term.id}
        tabindex="-1"
        aria-labelledby={`${term.id}-heading`}
      >
        <p class="eyebrow">Model term</p>
        <h2 id={`${term.id}-heading`}>{term.label}</h2>
        <p class="plain">{term.plain}</p>
        {#if term.formula}
          <div class="formula">
            <h3>Operation</h3>
            <code>{term.formula}</code>
          </div>
        {/if}
        <p>{term.detail}</p>
        <div class="links">
          <a href={term.workbench.href}>{term.workbench.label} →</a>
          {#if term.related.length > 0}
            <span>Related:</span>
            {#each term.related as id}
              {@const related = glossaryById.get(id)}
              {#if related}<a href={`#${id}`}>{related.label}</a>{/if}
            {/each}
          {/if}
        </div>
      </article>
    {/each}
  </div>
</section>

<style>
  .glossary {
    display: grid;
    gap: 1.2rem;
    min-width: 0;
  }
  .introduction h2,
  .term h2 {
    margin: 0 0 0.7rem;
    font-family: Georgia, 'Times New Roman', serif;
  }
  .introduction p,
  .term p {
    line-height: 1.6;
  }
  .term-index {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 1.2rem;
  }
  .term-index a {
    padding: 0.4rem 0.55rem;
    border: 1px solid var(--line-strong);
    color: var(--forest);
    text-decoration: none;
    font-size: 0.82rem;
  }
  .term-index a:hover,
  .term-index a:focus-visible {
    border-color: var(--forest);
    background: var(--paper);
  }
  .terms {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(min(100%, 21rem), 1fr));
    align-items: start;
    gap: 1rem;
  }
  .term {
    scroll-margin-top: 3.5rem;
  }
  .term:target,
  .term:focus {
    outline: 3px solid var(--focus);
    outline-offset: 2px;
  }
  .term .eyebrow {
    margin-bottom: 0.5rem;
  }
  .term .plain {
    font-weight: 600;
  }
  .formula {
    padding: 0.8rem;
    border-left: 3px solid var(--forest);
    background: var(--paper);
  }
  .formula h3 {
    margin: 0 0 0.35rem;
    font:
      700 0.72rem ui-monospace,
      SFMono-Regular,
      Menlo,
      monospace;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .formula code {
    white-space: normal;
    overflow-wrap: anywhere;
    font-size: 0.82rem;
  }
  .links {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.3rem 0.7rem;
    padding-top: 0.7rem;
    border-top: 1px solid var(--line);
    font-size: 0.82rem;
  }
  .links span {
    color: var(--muted);
  }
  .links a {
    color: var(--forest);
    text-underline-offset: 0.2rem;
  }
</style>
