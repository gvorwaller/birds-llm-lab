<script lang="ts">
  import { onMount } from 'svelte';
  import { getCheckpoints, selectCheckpoint } from '../lib/data/api';
  import { checkpointCatalog } from '../lib/stores/checkpoints';

  let busy = $state(false);
  let error = $state<string | null>(null);

  async function load(): Promise<void> {
    try {
      checkpointCatalog.set(await getCheckpoints());
    } catch (cause) {
      error = cause instanceof Error ? cause.message : 'Checkpoint list unavailable.';
    }
  }

  async function select(event: Event): Promise<void> {
    const id = (event.currentTarget as HTMLSelectElement).value;
    if (!id) return;
    busy = true;
    error = null;
    try {
      checkpointCatalog.set(await selectCheckpoint(id));
    } catch (cause) {
      error = cause instanceof Error ? cause.message : 'Checkpoint selection failed.';
      await load();
    } finally {
      busy = false;
    }
  }

  onMount(() => void load());
</script>

<div class="checkpoint-picker">
  <label for="global-checkpoint">Active checkpoint</label>
  <select
    id="global-checkpoint"
    value={$checkpointCatalog.activeCheckpointId ?? ''}
    disabled={busy || $checkpointCatalog.checkpoints.length === 0}
    onchange={(event) => void select(event)}
  >
    {#if $checkpointCatalog.checkpoints.length === 0}
      <option value="">No validated checkpoint</option>
    {:else}
      <option value="" disabled>Select checkpoint</option>
      {#each $checkpointCatalog.checkpoints as checkpoint}
        <option value={checkpoint.id}>{checkpoint.id} · step {checkpoint.trainingStep}</option>
      {/each}
    {/if}
  </select>
  {#if error}<small role="alert">{error}</small>{/if}
</div>

<style>
  .checkpoint-picker {
    margin-top: 1.5rem;
    padding: 0.8rem;
    border: 1px solid #364037;
  }

  label,
  select,
  small {
    display: block;
    width: 100%;
  }

  label {
    margin-bottom: 0.45rem;
    color: #879487;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }

  select {
    min-height: 38px;
    padding: 0.4rem;
    border: 1px solid #667363;
    color: #eff3e9;
    background: #171d18;
    font-size: 0.72rem;
  }

  small {
    margin-top: 0.45rem;
    color: #f1a17e;
    line-height: 1.35;
  }
</style>
