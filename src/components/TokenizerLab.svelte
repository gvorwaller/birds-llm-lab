<script lang="ts">
  import byteArtifactJson from '../assets/tokenizer-byte-only.json';
  import trainedArtifactJson from '../assets/tokenizer-1024.json';
  import { parseTokenizerArtifact, type TokenizerArtifact } from '../lib/data/schemas';
  import { decodeText, encodeText, replayMerges } from '../lib/tokenizer/bpe';
  import { BASE_VOCAB_SIZE, isSpecialTokenId } from '../lib/tokenizer/constants';
  import { encodeUtf8 } from '../lib/tokenizer/codec';

  const byteArtifact = parseTokenizerArtifact(byteArtifactJson);
  const trainedArtifact = parseTokenizerArtifact(trainedArtifactJson);

  let input = $state('The osprey dives into the water.');
  let selectedArtifact = $state<'trained' | 'byte'>('trained');
  let replayText = $state('Warbler');
  let replayIndex = $state(0);
  let vocabularyQuery = $state('');

  const artifact = $derived(selectedArtifact === 'trained' ? trainedArtifact : byteArtifact);
  const inputBytes = $derived(encodeUtf8(input));
  const tokenIds = $derived(encodeText(input, artifact));
  const decoded = $derived(decodeText(tokenIds, artifact));
  const replaySteps = $derived(replayMerges(replayText, trainedArtifact));
  const boundedReplayIndex = $derived(Math.min(replayIndex, replaySteps.length));
  const replayIds = $derived(
    boundedReplayIndex === 0
      ? encodeUtf8(replayText)
      : [...replaySteps[boundedReplayIndex - 1].after],
  );
  const activeMerge = $derived(
    boundedReplayIndex === 0 ? null : replaySteps[boundedReplayIndex - 1],
  );
  const vocabulary = $derived.by(() => {
    const query = vocabularyQuery.trim().toLocaleLowerCase();
    if (!query) return artifact.tokens;
    return artifact.tokens.filter(
      (token) =>
        String(token.id).includes(query) ||
        token.display.toLocaleLowerCase().includes(query) ||
        token.bytes.join(' ').includes(query),
    );
  });

  function token(artifactToUse: TokenizerArtifact, id: number) {
    const found = artifactToUse.tokens[id];
    if (!found) throw new Error(`Tokenizer artifact is missing token ${id}.`);
    return found;
  }

  function label(artifactToUse: TokenizerArtifact, id: number): string {
    const display = token(artifactToUse, id).display;
    if (display.length > 0) return display;
    return 'empty';
  }

  function bytes(artifactToUse: TokenizerArtifact, id: number): string {
    const values = token(artifactToUse, id).bytes;
    return values.length === 0
      ? '—'
      : values.map((value) => `0x${value.toString(16).padStart(2, '0')}`).join(' ');
  }

  function selectArtifact(value: 'trained' | 'byte'): void {
    selectedArtifact = value;
    vocabularyQuery = '';
  }

  function setReplayText(value: string): void {
    replayText = value;
    replayIndex = 0;
  }
</script>

<section class="tokenizer-layout" aria-label="Tokenizer laboratory">
  <section class="lab-panel token-input-panel" aria-labelledby="token-input-heading">
    <div class="panel-heading">
      <div>
        <p class="eyebrow">Text → token ids</p>
        <h2 id="token-input-heading">Tokenize any text</h2>
      </div>
      <fieldset class="artifact-switcher">
        <legend>Vocabulary</legend>
        <button
          class:active={selectedArtifact === 'byte'}
          aria-pressed={selectedArtifact === 'byte'}
          onclick={() => selectArtifact('byte')}>256 byte*</button
        >
        <button
          class:active={selectedArtifact === 'trained'}
          aria-pressed={selectedArtifact === 'trained'}
          onclick={() => selectArtifact('trained')}>1,024 BPE</button
        >
      </fieldset>
    </div>

    <label class="field-label" for="tokenizer-input">Input text</label>
    <textarea id="tokenizer-input" bind:value={input} rows="4" spellcheck="false"></textarea>

    <div class="token-stats" aria-live="polite">
      <span><strong>{inputBytes.length}</strong> UTF-8 bytes</span>
      <span><strong>{tokenIds.length}</strong> tokens</span>
      <span><strong>{artifact.targetSize.toLocaleString()}</strong> vocabulary entries</span>
      <span class:verified={decoded === input}>
        {decoded === input ? 'Exact round-trip' : 'Round-trip failed'}
      </span>
    </div>

    <div class="token-strip" aria-label="Encoded tokens">
      {#each tokenIds as id, index (`${index}-${id}`)}
        <span
          class:learned={id >= BASE_VOCAB_SIZE}
          class:special={isSpecialTokenId(id)}
          class="token-chip"
          title={`Token ${id}; bytes ${bytes(artifact, id)}`}
        >
          <strong>{label(artifact, id)}</strong>
          <small>#{id}</small>
        </span>
      {/each}
      {#if tokenIds.length === 0}<span class="empty-state">No bytes yet.</span>{/if}
    </div>

    <p class="footnote">
      * “256 byte” is the familiar shorthand. The artifact has 259 ids: 256 bytes plus explicit BOS,
      EOS, and padding tokens. Literal text such as &lt;|bos|&gt; remains ordinary bytes. A
      replacement glyph on one chip marks a partial UTF-8 preview; decoding the complete token
      stream remains exact.
    </p>
  </section>

  <section class="lab-panel" aria-labelledby="replay-heading">
    <div class="panel-heading">
      <div>
        <p class="eyebrow">Pure replay · no retraining</p>
        <h2 id="replay-heading">Watch the merges</h2>
      </div>
      <span class="step-counter">{boundedReplayIndex} / {replaySteps.length}</span>
    </div>

    <label class="field-label" for="replay-input">Word or phrase</label>
    <input
      id="replay-input"
      value={replayText}
      oninput={(event) => setReplayText(event.currentTarget.value)}
      spellcheck="false"
    />

    <div
      class="token-strip replay-strip"
      aria-live="polite"
      aria-label="Current merge replay state"
    >
      {#each replayIds as id, index (`replay-${boundedReplayIndex}-${index}-${id}`)}
        <span class:learned={id >= BASE_VOCAB_SIZE} class="token-chip">
          <strong>{label(trainedArtifact, id)}</strong>
          <small>#{id}</small>
        </span>
      {/each}
    </div>

    <div class="replay-controls">
      <button
        class="secondary"
        disabled={boundedReplayIndex === 0}
        onclick={() => (replayIndex = 0)}
      >
        Reset
      </button>
      <button
        class="secondary"
        disabled={boundedReplayIndex === 0}
        onclick={() => (replayIndex = Math.max(0, boundedReplayIndex - 1))}
      >
        ← Previous
      </button>
      <button
        class="primary"
        disabled={boundedReplayIndex >= replaySteps.length}
        onclick={() => (replayIndex = Math.min(replaySteps.length, boundedReplayIndex + 1))}
      >
        Next merge →
      </button>
    </div>

    {#if activeMerge}
      <div class="merge-explanation">
        <span class="merge-rank">Learned rank {activeMerge.rank + 1}</span>
        <p>
          <code>#{activeMerge.left}</code> + <code>#{activeMerge.right}</code> became
          <code>#{activeMerge.mergeId}</code>. This adjacent pair occurred
          <strong>{activeMerge.trainingCount.toLocaleString()}</strong> times when it was learned;
          it was applied {activeMerge.applications} time{activeMerge.applications === 1 ? '' : 's'}
          here.
        </p>
      </div>
    {:else}
      <div class="merge-explanation">
        <span class="merge-rank">Starting state</span>
        <p>Each chip is one UTF-8 byte. Step forward to apply only the ranked merges that match.</p>
      </div>
    {/if}
  </section>

  <section class="lab-panel vocabulary-panel" aria-labelledby="vocabulary-heading">
    <div class="panel-heading">
      <div>
        <p class="eyebrow">Stored artifact</p>
        <h2 id="vocabulary-heading">Vocabulary</h2>
      </div>
      <span class="step-counter">{vocabulary.length} shown</span>
    </div>

    <label class="field-label" for="vocabulary-search">Filter by id, text, or decimal byte</label>
    <input
      id="vocabulary-search"
      type="search"
      bind:value={vocabularyQuery}
      placeholder="e.g. osprey, 259, or 32"
    />

    <div class="table-scroll" role="region" aria-label="Scrollable tokenizer vocabulary">
      <table>
        <thead>
          <tr><th>Id</th><th>Display</th><th>Bytes</th><th>Learned count</th></tr>
        </thead>
        <tbody>
          {#each vocabulary as entry (entry.id)}
            <tr>
              <td><code>#{entry.id}</code></td>
              <td>{entry.display || 'empty'}</td>
              <td><code>{bytes(artifact, entry.id)}</code></td>
              <td>
                {entry.id >= BASE_VOCAB_SIZE
                  ? artifact.merges[entry.id - BASE_VOCAB_SIZE]?.trainingCount.toLocaleString()
                  : '—'}
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  </section>

  <section class="lab-panel explanation-panel" aria-labelledby="explanation-heading">
    <p class="eyebrow">Mechanism, not metaphor</p>
    <h2 id="explanation-heading">Why names split differently</h2>
    <details open>
      <summary>What’s happening</summary>
      <p>
        Every string first becomes its exact UTF-8 bytes. The trained tokenizer then replaces
        frequent adjacent pairs in a fixed learned order. Common text usually becomes fewer, longer
        tokens; rare names and number patterns often remain in smaller pieces.
      </p>
    </details>
    <details>
      <summary>The actual algorithm</summary>
      <p>
        During training, the pair with maximum count is selected. A tie chooses the smaller integer
        pair <code>(left id, right id)</code> lexicographically. Encoding replays those merges by rank
        until no learned pair remains. Pairs never cross documents or special tokens.
      </p>
      <p class="artifact-proof">
        Corpus <code>{artifact.corpusSha256.slice(0, 12)}…</code> ·
        {artifact.trainer.documentCount.toLocaleString()} training documents ·
        {artifact.merges.length.toLocaleString()} learned merges
      </p>
    </details>
  </section>
</section>
