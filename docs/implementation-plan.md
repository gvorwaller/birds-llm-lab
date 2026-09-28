# birds-llm-lab implementation plan

**Status:** implementation handoff v1, 2026-09-28

**Source:** [Claude Code's product plan](plan.md)

**Audience:** an implementation agent that needs explicit contracts, small tasks,
and objective stop conditions.

This document turns the product plan into an executable sequence. The product
plan remains authoritative for purpose and teaching content. When the two
documents differ, follow the decisions and scope controls in this implementation
plan, then record the difference in the devlog.

## 1. Read this before changing code

1. Read `AGENTS.md`, `docs/plan.md`, and this document completely.
2. Run `td usage --new-session`, then inspect `td next` and `td status`.
3. Do not write to, reset, seed, or directly migrate the Birds database. If the
   local test database is unavailable, the agent may run `npm run test:db:up` in
   `~/birds`; this is the only permitted command outside this repository. The
   Milestone 0 export must use a read-only transaction.
4. Do not modify `~/birds`. It is a separate repository.
5. Do not add an ML framework. Core model math must be readable TypeScript over
   `Float32Array`.
6. Do not add outbound or non-loopback network calls. Localhost serves the app
   and its control API. The optional external comparison milestone is blocked
   until the owner explicitly approves it.
7. Work on one numbered task below at a time. Add or update tests in the same
   task. Do not start the next milestone until the current exit gate passes.
8. Add evidence to `docs/devlog/YYYY-MM-DD.md`, then make the milestone commit.
9. If a stated contract is impossible or materially too slow, stop, record the
   measurement, and ask the owner. Do not silently replace it.

## 2. Binding implementation decisions from plan review

The following decisions are part of the implementation plan, not optional
suggestions. Section 7 records the owner's approved product choices.

1. **Separate the numerical engine from visualization.** The model returns a
   normal output by default and an opt-in `ForwardTrace` for one inspected
   example. Never retain every intermediate for a whole training batch.
2. **Add a performance proof before the full model.** A tiny overfit test and a
   default-config benchmark must pass before a long corpus training run. A pure
   scalar implementation may be correct but miss the 15-minute target.
3. **Build one vertical inspection slice before all screens.** After forward and
   backward math are correct, display one prompt through embedding, one
   attention head, and top logits. This validates the trace/UI boundary early.
4. **Treat live browser training as a demonstration, not checkpoint production.**
   Use a smaller preset in the Worker. Produce the ready-made checkpoint with
   the Node trainer. Both must call the same math and optimizer modules.
5. **Make the hallucination result honest.** Corpus substring/search evidence can
   show that wording or terms were not found; it cannot prove that a semantic
   claim is absent. Label results `matching evidence found` or
   `no matching evidence found`, never `claim proved false`.
6. **Defer the 256/1,024/4,096 tokenizer comparison.** Ship one correct tokenizer
   first. Train comparison tokenizers offline and load their artifacts in the
   UI; do not retrain a 4,096-token vocabulary during interaction.
7. **Keep optional M7 out of the dependency graph.** Core milestones must build,
   test, and run without comparison-model packages or downloaded artifacts.

## 3. Decisions and exact contracts

These contracts remove choices the implementing agent should not invent.

### 3.1 Supported runtime

- Node: `>=22 <23`; npm is the package manager.
- Browser: current desktop Chrome and Safari at 1280 px and 1024 px widths.
- Phone layout, remote deployment, authentication, and multi-user storage are
  out of scope. A loopback-only Node control service is required so the owner can
  export data, train, and select checkpoints without using the command line.
- Vite + Svelte 5 runes + TypeScript with strict type checking.
- Vitest is the unit/integration test runner. Add Playwright only when the first
  interactive screen exists.

### 3.2 Repository layout

Create only folders needed by the current task.

```text
src/
  lib/
    math/            # tensor primitives; no Svelte imports
    model/           # config, parameters, forward, backward, optimizer
    tokenizer/       # BPE train/encode/decode and artifact types
    trace/           # stable UI-facing trace schema and inspectors
    data/            # corpus/checkpoint loaders and validation
    generation/      # softmax filtering and seeded sampling
    stores/          # app state only
    components/      # reusable visual components
  routes/            # one route per lab screen plus glossary
  workers/           # live-training Worker entry
server/
  index.ts            # loopback HTTP server and static app host
  jobs/               # bounded export/training job coordinator
  routes/             # fixed local API; no arbitrary command execution
scripts/
  export-corpus.ts
  train-tokenizer.ts
  train.ts
  inspect-checkpoint.ts
tests/
  fixtures/          # tiny committed deterministic fixtures only
  reference/         # hand-computed cases; no production snapshots
data/                # generated locally; large artifacts ignored
checkpoints/         # generated locally; ignored by default
docs/devlog/
```

No module under `src/lib/math`, `model`, `tokenizer`, `trace`, or `generation`
may import Svelte, browser globals, Node-only modules, or database code.

### 3.3 Local control service and owner workflow

- The built app and JSON API are served by one Node process bound only to
  `127.0.0.1:5301`. Do not bind to `0.0.0.0`, `::`, a LAN address, or a public
  interface.
- The service serves built static assets and these fixed operations:
  `GET /api/status`, `GET /api/corpus/manifest`,
  `POST /api/corpus/export`, `GET /api/jobs/:id`,
  `POST /api/jobs/:id/cancel`, `GET /api/checkpoints`,
  `POST /api/training/jobs`, and `POST /api/checkpoints/:id/select`.
- API handlers call typed project modules directly. Never accept a shell command,
  script path, database string, output path, or arbitrary trainer arguments from
  the browser.
- Permit only same-origin requests. Reject unexpected `Host` and `Origin`
  headers, send no CORS opt-in headers, use a per-process CSRF token embedded in
  the served app, and set a restrictive Content Security Policy. These controls
  protect against a malicious website driving localhost endpoints.
- Allow only one export and one training job at a time. Jobs have typed states,
  progress, bounded retained logs, cancellation, and clear restart recovery.
- The Data page is the owner's control surface: database/corpus status, export
  button, manifest, training controls/progress, checkpoint list, active
  checkpoint, errors, and recovery instructions. Owner-facing instructions must
  never require Terminal.
- Scripts remain available for agent automation and tests, but every owner task
  must have an equivalent Data-page action.
- A versioned `scripts/install-launch-agent.ts` writes
  `~/Library/LaunchAgents/com.gaylon.birds-llm-lab.plist` using resolved absolute
  paths, runs the built service at login, and writes stdout/stderr to
  `~/Library/Logs/birds-llm-lab.log`. It validates the plist before loading it.
- Add uninstall/status/restart scripts. Installation and removal must target only
  this exact label and plist. Never unload or edit unrelated LaunchAgents.
- After each milestone, the implementing agent builds, restarts this service,
  verifies `/api/status`, and smoke-tests the changed owner workflow at
  `http://localhost:5301`.

### 3.4 Tensor representation

- A tensor is `{ data: Float32Array, shape: readonly number[] }`.
- Data is row-major. The last axis is contiguous.
- Validate dimensions at public function boundaries in development/test code.
- Use `Float64Array` only inside test reference calculations and corpus summary
  calculations. Stored weights and runtime activations are Float32.
- Every operation that participates in training has an explicit forward cache
  type and an explicit backward function. Do not build a general autograd engine.
- Reuse caller-provided output/scratch buffers on hot paths after correctness is
  established. The first implementation may allocate for clarity.

### 3.5 Model convention

Use this exact default unless the owner approves a change:

```ts
{
  vocabSize: 1024,
  contextLength: 64,
  dModel: 64,
  nLayers: 2,
  nHeads: 4,
  dHead: 16,
  dMlp: 256,
  tiedEmbeddings: true,
  useBias: true,
  layerNormEpsilon: 1e-5,
  gelu: "tanh-approximation"
}
```

- Pre-LayerNorm block: `x = x + attention(ln1(x))`, then
  `x = x + mlp(ln2(x))`; apply final LayerNorm before logits.
- Q, K, V and attention output are separate `[dModel, dModel]` weight
  matrices with `[dModel]` biases. Do not fuse them in the stored checkpoint.
- MLP weights are `[dModel, dMlp]` and `[dMlp, dModel]` with biases.
- Learned token embeddings are `[vocabSize, dModel]`; learned positions are
  `[contextLength, dModel]`.
- Tied unembedding uses the transpose of token embeddings and has no separate
  output bias. Untied mode is tested but is not required in the first checkpoint.
- Default parameter count with the conventions above is **169,728**. Add a unit
  test for this number so shape or bias drift is intentional.
- Causal attention masks future positions with negative infinity before stable
  softmax. Padding positions must also be masked if batching uses padding.

### 3.6 Initialization, loss, and optimizer

- Implement one documented seeded PRNG usable in Node, browser, and Worker.
  Never use `Math.random()` in model initialization, split assignment, shuffling,
  or sampling.
- Initialize matrix weights from a zero-mean normal distribution with standard
  deviation `0.02`; initialize linear biases to zero; initialize LayerNorm scale
  to one and offset to zero.
- Loss is mean next-token cross-entropy over non-pad targets. State precisely
  which tokens contribute to each batch in the training log.
- AdamW defaults: beta1 `0.9`, beta2 `0.95`, epsilon `1e-8`, weight decay `0.1`.
  Do not decay biases or LayerNorm parameters.
- Global gradient norm clipping default is `1.0`.
- Learning-rate defaults are decided only after the tiny overfit test. Record
  the selected value and evidence in the devlog and checkpoint config.
- Cosine decay is by optimizer step after linear warmup. Define step zero and the
  final step in a schedule unit test; avoid an off-by-one hidden in the trainer.

### 3.7 BPE contract

- Raw text is encoded to UTF-8 bytes. Token ids `0..255` are byte tokens.
- Reserve ids 256, 257, and 258 for `<|bos|>`, `<|eos|>`, and `<|pad|>`.
- Learned merge ids start at 259 and are assigned in merge order.
- Training counts adjacent token pairs independently within each training
  document. A pair never crosses a document boundary or a special token.
- On equal counts, choose the pair with the lexicographically smaller pair of
  integer ids. This makes training deterministic.
- Encoding applies learned merges by merge rank until no ranked pair remains.
  Use a correct simple version first; optimize only with equivalence tests.
- Decoding concatenates token bytes and uses UTF-8 replacement decoding only for
  an explicitly partial token preview. Full ordinary-text round trips must be
  byte-exact and string-exact.
- Literal text equal to a special-token spelling is ordinary bytes unless the
  caller explicitly passes a special token id.
- `tokenizer.json` includes `formatVersion`, special ids, target size, ordered
  merge pairs, token byte arrays, display strings, training counts, corpus hash,
  and trainer settings.
- Target vocabulary means total vocabulary including bytes and special tokens.
  Therefore the minimum accepted configured size is 259, not 256. The UI label
  `256` is shorthand for byte-only and actually contains 259 ids. Explain this.

### 3.8 Corpus contract

The exporter reads connection values from `~/birds/.env.test`; it must not log
them or copy them into generated files.

For each eligible species, normalize database values as follows:

- preserve Unicode; normalize line endings to `\n`;
- trim leading/trailing whitespace from individual text fields;
- represent missing extract/field craft as `null`, missing sections/tags as `[]`;
- validate section JSON and skip malformed individual sections with a counted
  warning; do not silently discard an entire row;
- sort output rows by species code and object keys in the documented order;
- sort and deduplicate tags; preserve section order from the source value.

Use a stable SHA-256 assignment for splitting. Hash `split-v1:<species-code>`,
interpret the first 32 bits unsigned, then use buckets 0-89 train, 90-94
validation, and 95-99 test. Famous demo species are explicit test overrides in
one named constant. The manifest lists overrides separately so ratios remain
auditable.

Training documents use a versioned, deterministic template:

```text
<|bos|>Common name: {name}
Scientific name: {sci}
Order: {order}
Family: {family}

{extract}

{section title}
{section text}
...
<|eos|>
```

Special markers above mean ids inserted by the document builder, not literal
text fed to the BPE encoder. Include Wikipedia sections by default, because the
stated corpus-size measurements and learning target assume them. Exclude field
craft by default. Record selected sources and template version in the manifest.

`manifest.json` has a versioned schema and includes export timestamp, source row
counts, emitted/skipped counts with reasons, per-split document/byte counts,
source-field byte counts, corpus SHA-256, split algorithm/version, template
version, exporter git revision (or `null` with a warning), and database identity
limited to host/port/database name. It must never include username or password.

### 3.9 Checkpoint contract

- Every artifact has `formatVersion: 1`.
- `config.json` stores the complete model, optimizer, seed, corpus hash,
  tokenizer hash, training step, and source git revision.
- `weights.bin` is little-endian Float32 data.
- `weights.index.json` is the sole ordering authority. Each entry stores name,
  shape, element offset, element count, byte offset, and byte length.
- Loader validation rejects duplicate names, shape/count mismatches, overlapping
  ranges, truncated/extra bytes, unsupported versions, and non-finite weights.
- Write checkpoints to a temporary sibling path, validate them by loading, then
  rename into place. Never leave a valid-looking partial checkpoint.
- `training-log.json` records step, split, mean loss, perplexity, learning rate,
  gradient norm before clipping, elapsed time, and seeded sample generations at
  configured intervals.
- Publish the verified ready checkpoint as a GitHub Release asset with SHA-256
  hashes and format/config metadata. Do not use Git LFS initially. The running
  app uses an already-installed local copy and never downloads one implicitly.

### 3.10 Trace and display-fidelity contract

`forward(input, options)` returns logits and optionally a `ForwardTrace`. Trace
entries use stable semantic names such as `blocks.0.attn.scores`; UI code never
reaches into private model caches.

For an inspected prompt, retain:

- token/position embedding contributions and residual streams;
- LayerNorm input, mean, variance, normalized values, scale, and offset;
- Q/K/V inputs, weights, biases, and results;
- per-head raw attention scores, masks, probabilities, value-weighted results;
- attention output projection and residual addition;
- MLP preactivation, GELU result, output, and residual addition;
- final normalization, logits, stable softmax probabilities, and sampling trace.

Each visualized operation exposes an `explainCell(indices)` result containing the
exact operand values, multiplication terms, reduction value, formula label, and
output value. To control memory, compute term lists on demand from retained
inputs and weights. Tests compare `explainCell` with the normal kernel.

Display decimal values with a stated precision, but provide the underlying
Float32 value through hover/click/copy. Text must say when rounding makes shown
terms appear not to sum exactly to the shown result.

## 4. Execution plan

Each task below should fit in one focused agent turn. A task is complete only
when its named checks pass and the devlog contains the command and result.

### Milestone 0 — safe scaffold and corpus artifact

#### M0.1 Scaffold only

- Create the Vite/Svelte/TypeScript project without overwriting `docs/` or
  `AGENTS.md`.
- Add scripts: `check`, `test`, `test:run`, `build`, `export`,
  `train:tokenizer`, `train`, `inspect:checkpoint`, `serve`, `service:install`,
  `service:status`, `service:restart`, and `service:uninstall`.
- Add design-token CSS, an app shell, placeholder routes, and a no-network
  Content Security Policy suitable for development and production builds.
- Add a visible `Local teaching app — no production data` banner.
- Tests/checks: clean install, `npm run check`, `npm test -- --run`, build.

#### M0.2 Define schemas before database access

- Implement and test JSON types/validators for corpus row, manifest, tokenizer,
  model config, weight index, and training log.
- Add tiny committed fixture rows containing Unicode, empty optionals, malformed
  section input, and a literal special-token spelling.
- Tests/checks: validator accept/reject cases and stable JSON serialization.

#### M0.3 Implement read-only export

- Parse only the named keys from `~/birds/.env.test` without mutating process
  configuration globally.
- Connect to `127.0.0.1:15436/birds_test` only; fail closed if resolved settings
  differ. If configurable local values are truly needed, ask first.
- Execute `BEGIN TRANSACTION READ ONLY`, verify transaction read-only state, run
  the documented query, write artifacts, then `COMMIT`; rollback on failure.
- Write to temporary files and atomically rename after hashing/validation.
- Never print the connection string or password.
- Tests/checks: query shape test with a fake client; failure-path rollback;
  source scan for prohibited hosts/secrets. If the local test database is not
  running, the agent may start it only with `npm run test:db:up` in `~/birds`.
  Never reset or seed it.

#### M0.4 Add the local service, Data page, and LaunchAgent

- Implement the exact loopback service and security rules in section 3.3.
- Wire the Data-page export button to the same tested export module used by the
  CLI; show progress, completion, errors, and the resulting manifest.
- Add typed job state and cancellation. A browser refresh reconnects to the
  current job rather than starting a duplicate.
- Implement idempotent LaunchAgent install/status/restart/uninstall scripts.
  Installation refuses to overwrite a differently owned or differently labeled
  plist and prints no secrets.
- Tests/checks: API contract tests, foreign Host/Origin/CSRF rejection, duplicate
  job rejection, cancellation, static fallback, plist snapshot/validation, and
  a real localhost smoke test from export click to displayed manifest.

#### M0 exit gate and commit

- Export completes without changing the source DB.
- Manifest counts reconcile exactly with emitted JSONL.
- No credential value is present in tracked or generated inspectable metadata.
- The owner can export and inspect the manifest at `http://localhost:5301`
  without opening Terminal; the service survives a browser refresh and restarts
  through its LaunchAgent.
- Commit: `feat: scaffold app and add read-only corpus export`.

### Milestone 1 — tokenizer engine, artifact, then UI

#### M1.1 Byte and special-token codec

- Implement UTF-8 byte encoding, explicit special-token insertion, and decoding.
- Property-test ordinary Unicode strings and every corpus document if exported.

#### M1.2 Deterministic BPE trainer

- Implement the simple reference trainer exactly as section 3.7.
- Test pair boundaries, tie-breaking, merge ordering, target sizes, repeated
  pairs, Unicode, empty documents, and deterministic artifact bytes.
- Add a benchmark log for a 1 MB subset before optimizing.

#### M1.3 Fast encoder and artifact CLI

- Implement rank-based encoding and equivalence tests against a deliberately
  slow reference encoder.
- `npm run train:tokenizer -- --vocab-size 1024` consumes train documents only,
  writes atomically, and records corpus hash/settings.
- Refuse tokenizer/corpus hash mismatches in loaders.

#### M1.4 Tokenizer Lab

- Build input token chips, ids, bytes, display strings, merge replay, vocabulary
  table, and explanatory/math panels.
- Keep the merge replay as pure derived state so step forward/back is exact.
- Add prebuilt byte-only and 1,024-token artifacts first. Add 4,096 only after
  its offline training time and artifact size are acceptable.
- Tests/checks: component behavior, keyboard operation, 1024/1280 screenshots,
  no horizontal-page overflow, round-trip of user input.

#### M1 exit gate and commit

- Full exported corpus round-trip passes.
- Same corpus/settings produce byte-identical merge data.
- UI can explain each merge using artifact counts/ranks.
- Commit: `feat: add deterministic BPE tokenizer lab`.

### Milestone 2 — numerical engine and trainable checkpoint

#### M2.1 Math kernels

- Implement seeded RNG, tensor validation, matmul/batched matmul, add, reductions,
  stable softmax, LayerNorm, GELU, embedding lookup/scatter-add, and
  cross-entropy.
- Implement backward functions beside forward functions.
- Test against hand-computed cases, invariants, and Float64 test references.

#### M2.2 Parameter registry and checkpoint I/O

- Build named parameters in one registry used by initialization, optimizer,
  serialization, parameter count, and explorer.
- Implement versioned checkpoint writer/loader and corruption tests.
- Assert default parameter names, shapes, and total `169_728`.

#### M2.3 One transformer block forward/backward

- Implement single-head attention first, then multi-head reshape/transpose, then
  one full pre-LN block. Add causal and padding masks.
- Add finite-difference checks per parameter group on a tiny config. Use central
  differences in Float64 reference calculations and document chosen epsilon.
- Relative error rule: `abs(a-n) / max(1e-6, abs(a)+abs(n)) < 1e-3`; also apply
  an absolute-error bound near zero.

#### M2.4 Full model and optimizer

- Stack blocks, final LayerNorm, tied unembedding, cross-entropy, AdamW, clipping,
  warmup/cosine schedule, deterministic batches, and seeded sampling.
- Test optimizer state and one complete update against a small hand/reference
  case. Test tied gradients accumulate into the shared embedding exactly once.

#### M2.5 Learning and speed gates

- Overfit one tiny batch: loss must fall by at least 80% and generated next-token
  choices must match the tiny sequence.
- Benchmark default-config forward+backward for a documented batch/context on
  the target M4. Estimate full-run time from measured optimizer steps.
- If estimated ready-checkpoint training exceeds 15 minutes, profile. Optimize
  buffer reuse and loop ordering without changing results. Ask before adding
  native/WASM/GPU dependencies or reducing the promised default.

#### M2.6 Node trainer and ready checkpoint

- Stream/shuffle deterministic training examples; do not materialize redundant
  token copies of the full corpus.
- Save resumable optimizer/RNG/data-order state as well as inference weights.
- Report train and validation metrics separately; test split is not used for
  tuning.
- Inspect generated samples during training but do not weaken correctness gates
  to force subjectively good prose.

#### M2.7 Add owner-facing training and checkpoint controls

- Connect the Data-page Train button to the same trainer module as the CLI via a
  bounded background job. Use a safe preset form, not arbitrary CLI arguments.
- Show current step/total, loss, validation loss, learning rate, elapsed/estimated
  time, latest sample, pause/cancel state, and terminal errors.
- List only validated checkpoints under the configured checkpoint directory.
  Selection persists locally and immediately updates the global picker.
- A service or browser restart recovers completed artifacts and clearly marks an
  interrupted job; it never presents a partial checkpoint as usable.
- Tests/checks: job lifecycle, invalid settings, duplicate start, cancellation,
  interrupted recovery, checkpoint validation/selection, and localhost smoke.

#### M2 exit gate and commit

- All math and gradient checks pass.
- Two fixed-seed 100-step runs have identical logged numbers on the same runtime.
- Train and validation loss fall from initialization.
- Ready checkpoint loads, generates, and records exact config/corpus hashes.
- The owner can start training, watch progress, cancel safely, and select a valid
  checkpoint without opening Terminal.
- Commit: `feat: add from-scratch transformer training and checkpoints`.

### Milestone 3 — trace boundary and first inspection slice

#### M3.1 ForwardTrace and cell explanations

- Add opt-in trace capture without changing non-trace logits.
- Add `explainCell` for embedding addition, LayerNorm, matmul, attention score,
  softmax row, weighted value, projection, GELU, residual, and logit.
- Golden test one fixed tiny model/prompt; store small expected values only.

#### M3.2 Vertical slice

- One route walks a fixed prompt through token/position embeddings, layer 0 head
  0 attention, and top-five logits.
- Implement shared grid/heatmap selection, value formatting, formula, and actual
  operand panels. Step forward/back must be pure and reversible.
- Verify every visible selected value against the golden trace test.

#### M3.3 Complete forward-pass and attention screens

- Extend the vertical slice across all blocks and operations.
- Add per-layer/head and average-head attention views. Average probabilities,
  not pre-softmax scores.
- Preset prompts are examples, never tests that a particular head learned a
  desired linguistic behavior. Describe observed patterns cautiously.

#### M3.4 Embeddings screen

- Implement cosine neighbours and PCA from first principles. Define centering,
  covariance, eigensolver, convergence/tolerance, and sign stabilization so the
  plotted orientation is deterministic.
- Compare initial embeddings reconstructed from checkpoint seed/config with the
  trained weights. Label what is reconstructed versus stored.

#### M3 exit gate and commit

- Trace on/off logits are bit-identical.
- Selected cells on each operation match engine values.
- Screens remain usable at 1024 and 1280 widths and by keyboard.
- Commit: `feat: add inspectable forward pass attention and embeddings`.

### Milestone 4 — sampling and generation

#### M4.1 Distribution transforms

- Implement temperature, top-k, top-p, renormalization, and seeded categorical
  sampling as pure functions.
- Define order: temperature -> softmax -> top-k -> top-p -> renormalize -> draw.
- Define edge cases: temperature must be positive; `k=0` means disabled;
  `p=1` means disabled; top-p retains at least one token; stable ties use id.
- Tests cover mass sums, ties, extremes, invalid inputs, and fixed draws.

#### M4.2 Next-token screen

- Show top logits, probabilities before/after filters, retained/removed mass,
  random draw, cumulative interval, and selected token.
- “Full-vocabulary probability mass” means a searchable/virtualized table plus
  an aggregate remainder, not thousands of always-rendered DOM nodes.

#### M4.3 Generation loop

- Re-run the same model for each appended token. Initially recompute the full
  context; KV caching is explicitly out of scope unless performance requires it.
- Show absolute token positions and exactly which leading tokens were dropped
  when the context window slides.
- Replay stores seed plus generated ids; stepping backward never resamples.

#### M4 exit gate and commit

- Fixed config/prompt/seed gives identical generated ids in Node and browser.
- UI probabilities and cumulative intervals equal pure-function outputs.
- Commit: `feat: add inspectable sampling and generation`.

### Milestone 5 — live training demonstration

#### M5.1 Worker protocol

- Define versioned start/pause/resume/step/cancel/checkpoint messages and typed
  progress/error replies.
- Transfer snapshots no more often than the chosen display cadence. Never send
  all activations/weights every optimizer step.
- Cancellation must stop within one batch and leave the UI responsive.

#### M5.2 Small live-training preset

- Start with a deliberately smaller model/corpus preset and label it clearly.
- Plot train/validation loss, LR, gradient norm, samples, selected-weight
  histogram, and selected matrix snapshot.
- Throttle rendering separately from training; dropped visual frames must not
  alter training results.

#### M5.3 Slowed Adam step

- Snapshot one named scalar parameter before update, raw gradient, clipped
  gradient, first/second moments, bias corrections, decay term, and new value.
- Recompute the displayed result in an independent unit test.

#### M5 exit gate and commit

- Training does not block main-thread interaction.
- Pause/resume is deterministic; cancel is prompt; displayed scalar update
  equals optimizer state.
- Commit: `feat: add worker-based live training demonstration`.

### Milestone 6 — evidence, parameter explorer, and teaching copy

#### M6.1 Corpus evidence search

- Build a local index over exported training documents only.
- Show exact-phrase, normalized-term, and species-document matches separately.
- Always display source split, field, and snippets. Say “no matching evidence
  found with these searches,” not “the corpus never said this.”

#### M6.2 Hallucination demonstration

- Save seeded examples only after verifying the output is reproducible with the
  shipped checkpoint.
- Explain likely-token mechanics with actual probability and corpus co-occurrence
  evidence. Label interpretation as interpretation.
- Mention the Birds tag lesson without importing or modifying Birds code/data.

#### M6.3 Parameter explorer

- Derive the tree entirely from the parameter registry/weight index.
- Compute count/min/max/mean/std, histogram, heatmap, and links into trace views.
- Virtualize or tile large matrices; do not create one DOM node per weight.

#### M6.4 Glossary and copy audit

- Add all original-plan terms, cross-links, formulas, and plain-language text.
- Audit metaphors so each is labeled and followed by the literal mechanism.
- Audit accessibility: focus, labels, reduced motion, non-color encoding, contrast.

#### M6 exit gate and commit

- Evidence screen distinguishes absence of search hits from factual falsity.
- Every registered parameter appears exactly once in the explorer and counts sum
  to the config total.
- Commit: `feat: add corpus evidence parameter explorer and glossary`.

### Milestone 7 — optional external comparison

Do not create tickets, dependencies, download scripts, caches, or UI placeholders
for this milestone until the owner explicitly says to proceed.

If approved, first write a separate mini-plan covering package choice, exact
download size/license/source, cache location, offline behavior, deletion, CSP,
checkpoint integrity, and how attention/token probabilities are extracted. The
one-time download must require deliberate confirmation in the app and the core
app must continue working with an empty cache.

## 5. Required verification matrix

Run the smallest relevant subset during a task and the full applicable set at a
milestone gate.

| Area | Required evidence |
|---|---|
| Formatting/types | formatter check, TypeScript strict build, `svelte-check` |
| Unit math | hand cases, invariants, Float64 reference comparisons |
| Gradients | central finite differences for every parameter group |
| Learning | tiny-batch overfit and falling train/validation corpus losses |
| Determinism | fixed artifact bytes/log prefix/generated token ids |
| Serialization | round trip plus corrupt/truncated/version mismatch failures |
| UI fidelity | selected-cell engine/trace/UI equality tests |
| Worker | responsiveness, pause/resume, cancellation, error propagation |
| Safety | read-only transaction test, secret scan, prohibited-host scan |
| Local API | loopback binding, Host/Origin/CSRF rejection, bounded jobs |
| Owner workflow | Data-page export/train/select without Terminal |
| Layout/a11y | keyboard pass, reduced motion, 1024/1280 visual snapshots |
| Release | clean clone/install/check/test/build; LaunchAgent restart and smoke |

The prohibited-host scan applies to executable source, configuration, generated
bundles, and package scripts. Documentation may name forbidden production hosts
to state safety rules; allowlisting documentation must not allow them in code.

## 6. Agent handoff template

At the end of every task, leave this in the devlog and `td handoff` if work is
not ready for review:

```text
Task:
Changed files:
Contract implemented:
Commands run and results:
Manual checks:
Measurements:
Known limitations:
Next exact task:
Questions requiring owner decision:
```

Never report a check as passing if it was not executed. Distinguish test evidence
from visual/manual observation.

## 7. Owner decisions — approved 2026-09-28

These choices are final implementation requirements unless the owner explicitly
changes one later.

| ID | Approved decision |
|---|---|
| Q1 | License the public repository under the MIT License. |
| Q2 | Train on Wikipedia extracts and sections by default. |
| Q3 | Export and label AI-generated field craft, but exclude it from default training and demos. |
| Q4 | Support current Chrome and Safari at widths of 1024 px and above; phone support is out of scope. |
| Q5 | Distribute the ready checkpoint as a GitHub Release asset with hashes; do not use Git LFS. |
| Q6 | Defer the GPT-2 comparison until core Milestones 0–6 are complete. |
| Q7 | Use a distinct laboratory/notebook visual identity with strong accessibility. |

## 8. Definition of project completion

The core project is complete when Milestones 0–6 pass their gates from a clean
clone, the LaunchAgent keeps the loopback app available, the owner can export
the corpus and train/select checkpoints entirely through the Data page, a
verified ready checkpoint can be installed without credentials, every displayed
numerical claim is traceable to engine data, and no core runtime path requires
outbound or non-loopback network access. Milestone 7 is optional and does not
block completion.
