<script lang="ts">
  import { onMount } from 'svelte';
  import DataWorkbench from './components/DataWorkbench.svelte';
  import TokenizerLab from './components/TokenizerLab.svelte';
  import { routeForPath, routes } from './lib/navigation';

  let pathname = $state('/');

  const currentRoute = $derived(routeForPath(pathname));

  function navigate(event: MouseEvent, path: string): void {
    if (
      event.defaultPrevented ||
      event.button !== 0 ||
      event.metaKey ||
      event.ctrlKey ||
      event.shiftKey ||
      event.altKey
    ) {
      return;
    }

    event.preventDefault();
    history.pushState({}, '', path);
    pathname = path;
    window.scrollTo({ top: 0, behavior: 'instant' });
  }

  onMount(() => {
    pathname = window.location.pathname;
    const updatePath = () => (pathname = window.location.pathname);
    window.addEventListener('popstate', updatePath);
    return () => window.removeEventListener('popstate', updatePath);
  });
</script>

<svelte:head>
  <title>{currentRoute.label} · Birds LLM Lab</title>
  <meta name="description" content={currentRoute.summary} />
</svelte:head>

<div class="safety-banner" role="status">
  <span aria-hidden="true">●</span>
  Local teaching app — no production data
</div>

<div class="app-shell">
  <aside class="sidebar">
    <a class="brand" href="/" onclick={(event) => navigate(event, '/')}>
      <span class="brand-mark" aria-hidden="true">B/LLM</span>
      <span>
        <strong>Birds LLM Lab</strong>
        <small>Inspectable by design</small>
      </span>
    </a>

    <nav aria-label="Lab screens">
      <p class="nav-label">Workbenches</p>
      {#each routes as route}
        <a
          href={route.path}
          class:active={currentRoute.path === route.path}
          aria-current={currentRoute.path === route.path ? 'page' : undefined}
          onclick={(event) => navigate(event, route.path)}
        >
          <span>{route.label}</span>
          <small>{route.milestone}</small>
        </a>
      {/each}
    </nav>

    <div class="sidebar-note">
      <span class="note-index">01</span>
      <p>Every value shown in later milestones will trace back to stored data and readable code.</p>
    </div>
  </aside>

  <main id="main-content">
    <div class="notebook-rule" aria-hidden="true"></div>
    <section class="hero" aria-labelledby="page-title">
      <p class="eyebrow">{currentRoute.eyebrow} · {currentRoute.milestone}</p>
      <h1 id="page-title">{currentRoute.title}</h1>
      <p class="lede">{currentRoute.summary}</p>
    </section>

    {#if currentRoute.path === '/'}
      <section class="card-grid" aria-label="Core principles">
        <article>
          <span class="card-number">01</span>
          <h2>Real machinery</h2>
          <p>
            The small model uses the same token, attention, and gradient mechanisms as larger
            models.
          </p>
        </article>
        <article>
          <span class="card-number">02</span>
          <h2>Actual numbers</h2>
          <p>
            Visual explanations will expose operands and formulas instead of replacing them with
            metaphor.
          </p>
        </article>
        <article>
          <span class="card-number">03</span>
          <h2>Local boundaries</h2>
          <p>
            The app stays on this Mac and reads only an explicitly exported test-database corpus.
          </p>
        </article>
      </section>
      <a class="primary-action" href="/data" onclick={(event) => navigate(event, '/data')}>
        Open the Data workbench
        <span aria-hidden="true">→</span>
      </a>
    {:else if currentRoute.path === '/data'}
      <DataWorkbench />
    {:else if currentRoute.path === '/tokenizer'}
      <TokenizerLab />
    {:else}
      <section class="placeholder" aria-labelledby="placeholder-title">
        <div>
          <p class="eyebrow">Bench status</p>
          <h2 id="placeholder-title">Scheduled for {currentRoute.milestone}</h2>
        </div>
        <p>
          This route is in place so the complete laboratory stays legible while each verified
          milestone is added. It does not simulate results that have not been computed yet.
        </p>
      </section>
    {/if}
  </main>
</div>
