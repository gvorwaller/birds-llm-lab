# birds-llm-lab — an animated, inspectable tiny LLM (plan)

**Status:** PLAN v1 (2026-09-28). Written by Claude Code (the birds CC session) for the owner, Gaylon Vorwaller, to hand to a separate implementing agent: another Claude session, ChatGPT/Codex, or AGY. **Nothing is built yet.** Read this whole document before writing any code.

## 1. Purpose

A **local-only teaching app** that gives the owner a solid conceptual understanding of how a large language model works. It does that by **training a real, tiny, decoder-only transformer on bird text from the birds app's local test database, and animating every step**:
- how text becomes tokens;
- how tokens become vectors;
- how weights (parameters) transform those vectors, and how attention decides what to look at;
- how the model produces a probability for every possible next token;
- how one token is picked, and how it all repeats;
- how training adjusts the weights;
- and why the process can confidently produce false statements ("hallucination").

**Who it's for:** the owner, a programmer for 42 years who is expert in rule-based decision systems and new to modern ML. **Explain concepts plainly, and never hide the math:** every number shown on screen must be traceable to code and stored values. Prefer "show the actual numbers" over metaphors. Where a metaphor helps, label it as one.

**Why a tiny model:** production LLMs have billions of weights and can't be shown. A model with roughly 100k–300k weights uses **exactly the same machinery** (tokenizer, embeddings, attention, MLP, residual stream, softmax, cross-entropy, gradient descent), and every weight can be displayed.

## 2. Hard rules for the implementing agent

1. **Local only.** Never deploy. Never contact the production droplet, `birds.gaylon.photos`, or any production database. **No Anthropic, OpenAI or other LLM API calls** from the app.
2. **Data comes only from the local birds test database,** read-only:
   - Postgres 17 at `127.0.0.1:15436`, database `birds_test`;
   - connection settings are read at export time from `~/birds/.env.test` (`PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`, `PGPASSWORD`).
   - **Never copy credentials into this repo.** The export runs inside `BEGIN READ ONLY`.
   - If the database isn't running, tell the owner to run `npm run test:db:up` in `~/birds`. Never start, reset or modify it yourself.
3. **Never modify `~/birds`.** This is a separate repo at `~/birds-llm-lab` with its own `git init`. Don't copy code or secrets from birds, beyond reading the DB as above.
4. **No network at runtime,** except the optional real-tokenizer and GPT-2 comparison (§6.9). Its model download must be opt-in, owner-approved, and happen once into a local cache.
5. **The core model is written from scratch in TypeScript** using `Float32Array` math. No TensorFlow.js, ONNX, PyTorch or ML library for the tiny model. The point is that every multiply is visible, readable code. (Test-only numeric helpers are fine.)
6. **Correctness before animation.** Every mathematical component has unit tests, including a finite-difference gradient check (§8), before any UI is built on it.
7. **Commit per milestone,** with a short summary. Ask the owner when a requirement is unclear; don't guess.
8. **Keep a `docs/devlog/YYYY-MM-DD.md`** with what was built, what was verified, and open questions.

## 3. Stack

- **Vite + Svelte 5 (runes) + TypeScript**, the same as the owner's birds app. Plain CSS with design tokens, light and dark themes.
- **Node ≥ 22** for scripts. `vitest` for tests.
- **Training** runs in a **Web Worker** in the browser for the live training view, *and* as a Node script (`npm run train`) that produces a ready-made checkpoint. The app then works immediately without training first.
- **Storage:** plain JSON and binary files in `data/` and `checkpoints/`, gitignored if large (>5 MB). IndexedDB only for user-saved checkpoints in the browser.
- **Charts and heatmaps:** SVG or Canvas drawn directly. An optional small charting library is fine for the loss curve, but heatmaps should be drawn directly so cell-to-value mapping is exact.

## 4. Data export (Milestone 0)

`npm run export` runs `scripts/export-corpus.ts`, which connects read-only using `~/birds/.env.test` and writes `data/corpus.jsonl` (gitignored). Each line:

```json
{ "code": "amekes", "name": "American Kestrel", "sci": "Falco sparverius",
  "order": "Falconiformes", "family": "Falconidae",
  "extract": "…", "sections": [{"title": "Habitat", "text": "…"}],
  "field_craft": "…", "tags": ["habitat:grassland", "…"] }
```

- **Source query (read-only):** `species_enrichment` (`wikipedia_extract`, `wikipedia_sections`, `field_craft`, `tags`), joined to `taxonomy_cache` (`com_name`, `sci_name`, `order_name`, `family_sci_name`) where `category='species'` and `wikipedia_extract IS NOT NULL`.
- **Measured 2026-09-28:** 10,886 articles; extracts ≈4.6 MB total (≈430 characters each); section JSON ≈33 MB; field craft ≈4.9 MB.
- **Corpus toggles:**
  - **Wikipedia text** (default for training): human-written.
  - **Field craft:** *AI-generated* by the birds app, so label it that way in the UI wherever it's used. It's excluded from default training, which keeps the "the model only learns from the text we give it" story clean.
- **Split** by species into train, validation and test (90/5/5), seeded and **stable across runs**. Hold out a few famous species by name for the demos, such as American Kestrel, Osprey, Wandering Albatross and Northern Cardinal.
- `data/manifest.json` records row counts, byte counts, the export time, the git commit and a hash of the corpus file.

## 5. Model specification (Milestone 1–2)

**Tokenizer: byte-pair encoding (BPE), trained from scratch on the train split.**
- Start from bytes (256 base tokens) plus special tokens `<|bos|>`, `<|eos|>` and `<|pad|>`.
- Learn merges up to a target vocabulary. Default **1,024**, configurable 256–4,096.
- Store `tokenizer.json`: the ordered merge list, the vocabulary (id → byte sequence → display string) and the frequency of each merge at the time it was learned.
- Round-trip guarantee: `decode(encode(s)) === s` for any string.

**Transformer: decoder-only (GPT-style), pre-LayerNorm.**
- **Defaults:** `vocab 1024`, `d_model 64`, `layers 2`, `heads 4` (head size 16), `context 64` tokens, `mlp 256` (4×), GELU, learned positional embeddings, and tied input/output embeddings (optional toggle).
- **Parameters:** the app must compute and display the exact count per matrix and in total. It's about 100k–200k at the defaults.
- **Per block:** `x + Attn(LN(x))`, then `x + MLP(LN(x))`, with a causal mask. The final LN goes to the unembedding (logits).
- **Deterministic:** a seeded RNG for initialization and sampling.
- **Training:** cross-entropy on next-token prediction, AdamW, cosine learning-rate schedule with warmup, gradient clipping, batch size and steps configurable. It reports train and validation loss, and perplexity.
- **Target:** the ready-made checkpoint trains on the M4 in ≤15 minutes via `npm run train`, and produces recognizable bird-like sentences.

**Checkpoints** (`checkpoints/<name>/`):
- `config.json`, `tokenizer.json`, `weights.bin` (Float32, in a documented order), `weights.index.json` (name, shape, offset for each matrix);
- `training-log.json` (loss per step, with sample generations every N steps).

**Backpropagation** is implemented by hand, one function per layer, with its own backward pass. This is the educational core; don't hide it.

## 6. Screens (Milestones 3–6)

Every screen has three things:
- (a) the live visualization;
- (b) a collapsible **"What's happening"** panel in plain English;
- (c) a collapsible **"The actual math"** panel showing the formula *and the real numbers for one highlighted cell*, e.g. how one attention score was computed from the actual q and k values.

Global controls: model and checkpoint picker, a prompt box, and step controls (play/pause/step-forward/step-back, with a speed slider). Everything works at 1280 px and at laptop widths.

1. **Tokenizer lab.**
   - Type any text: see the token boundaries (colored chips), token ids and byte sequences.
   - **Merge replay:** animate BPE learning from bytes. Pick a word ("Warbler") and watch its pieces merge step by step.
   - Show the vocabulary sorted by merge order, with counts.
   - Show "why numbers and rare names split into many tokens".
   - Compare the token count for the same sentence under vocabularies of 256, 1,024 and 4,096 (retraining the tokenizer quickly on a subset is fine).
2. **Embeddings.**
   - Each token id is a row of `d_model` numbers; show the actual row as a strip of colored cells with values on hover.
   - A 2-D map (PCA, computed in code, no library magic) of token embeddings, before training (random) vs after training (structure).
   - Search a token and show its nearest neighbours by cosine similarity, with the similarity numbers.
3. **Forward pass, animated.** Using a prompt (default: "The osprey dives into the water to catch"), step through:
   - tokens → embedding rows + position rows → the residual stream (one vector per position, shown as a grid);
   - each block's LayerNorm → Q/K/V projections (show the weight matrices as heatmaps and the resulting vectors);
   - attention scores (the QKᵀ/√d heatmap with the causal mask visible) → softmax per row → the weighted sum of V;
   - output projection → added to the residual stream;
   - MLP (expand, GELU, contract) → added to the residual.

   Animate the data flow, and let the owner click any cell to see its exact computation.
4. **Attention explorer.** For each layer and head, a heatmap of which earlier tokens each token attends to, with values. Toggle between heads, and include an "average heads" view. Include a few prompts where a head visibly attends to the bird name or the verb.
5. **Next-token prediction.**
   - Final vector → unembedding → one **logit** per vocabulary entry.
   - Show the top 20 logits and their softmax probabilities as bars, and the full-vocabulary probability mass.
   - Sliders for **temperature**, **top-k** and **top-p** that redraw the distribution live.
   - "Pick": show the random number drawn and where it lands in the cumulative distribution.
6. **Generation loop.** Repeat 5 token by token, appending as it goes. Show that it's the same computation run again, not "thinking ahead". Show the context window sliding when the prompt exceeds 64 tokens.
7. **Training view** (the Web Worker trains live from a fresh random init):
   - loss curves for train and validation;
   - learning rate;
   - a histogram of each weight matrix over time;
   - a chosen weight matrix heatmap updating;
   - sample generations every N steps, going from gibberish to bird-like sentences;
   - gradient-norm plot;
   - "one gradient step, slowed down": pick one parameter and show its value, its gradient, the Adam moment estimates and its new value.
8. **Hallucination demo.**
   - Prompts where the model confidently continues with something the training text never said about that bird, e.g. "The American Kestrel feeds far out at" → "sea".
   - For each continuation, **search the training corpus** for the generated claim about that species and report whether it appears. If it doesn't, explain *mechanically* why it was produced: the continuation that other species' sentences made likely.
   - Link the idea to the birds app's tag problem: plausible patterns, no source of truth.
9. **Scale comparison (optional; network opt-in).**
   - Tokenize the same sentences with a real production-grade tokenizer (e.g. GPT-2's BPE, or a modern one via a local package) and compare the token boundaries.
   - Optionally load **GPT-2 small (124M parameters)** locally via `transformers.js` into a local cache, **only after the owner approves the one-time download**. Show its attention maps and next-token probabilities for the same prompt next to the tiny model's.
   - A static page puts the sizes in perspective: this model's parameter count vs GPT-2 small vs published large-model sizes, labelled "public estimates" where not officially disclosed.
10. **Parameter explorer.** A tree of every weight matrix (name, shape, count, min/max/mean/std) with a heatmap. Clicking a matrix shows where it's used in the forward-pass screen.

**Glossary** (always reachable): token, vocabulary, embedding, parameter/weight, bias, matrix multiply, dot product, attention, head, residual stream, LayerNorm, MLP, GELU, logit, softmax, temperature, cross-entropy, loss, gradient, backpropagation, learning rate, Adam, overfitting, validation set, context window, hallucination.

## 7. Milestones

| M | Deliverable | Done when |
|---|---|---|
| 0 | Repo scaffold, `export-corpus.ts`, data manifest | corpus exported read-only; counts match §4; no secrets in repo |
| 1 | BPE tokenizer + Tokenizer lab (screen 1) | round-trip tests pass; merge replay animates; vocabulary sizes compare |
| 2 | Transformer forward/backward + `npm run train` + checkpoint | gradient checks pass; train/val loss falls; checkpoint generates bird-like text; param count displayed |
| 3 | Forward-pass animation + attention explorer (screens 3–4) + embeddings (screen 2) | every displayed number equals the value recomputed in tests for a fixed seed/prompt |
| 4 | Next-token + generation (screens 5–6) | sampling reproducible with a seed; sliders change distribution correctly |
| 5 | Live training view (screen 7) | trains in a Worker without freezing UI; single-parameter Adam step matches the math |
| 6 | Hallucination demo + parameter explorer + glossary (8, 10) | corpus search shows claim absent; explorer covers all matrices |
| 7 | Optional scale comparison (9) | only after owner approves the download |

After each milestone: commit, write the devlog entry, and give the owner a 5-line summary with how to run it.

## 8. Verification

- **Tokenizer:** round-trip on the entire corpus; merges deterministic for a seed; special tokens are never split.
- **Math:** unit tests for matmul, softmax (numerically stable), LayerNorm, GELU and cross-entropy against hand-computed small cases.
- **Gradient check:** finite differences vs analytic gradients for every parameter group on a tiny config. Relative error < 1e-3.
- **Determinism:** a fixed seed produces the same init, the same training losses for the first 100 steps, and the same generation.
- **Display fidelity:** for a fixed checkpoint and prompt, a test computes the attention matrix, logits and top-5 probabilities, and asserts the UI data layer returns identical numbers.
- **Safety:**
  - a test asserts the export script opens a `READ ONLY` transaction;
  - a repo scan asserts no credentials, and no hostnames other than `127.0.0.1` or `localhost`.
- `npm run check` (svelte-check), `npm test` and `npm run build` pass.

## 9. Owner decisions already made

- The location is `~/birds-llm-lab`, a separate repo, local only.
- Data comes from the birds **test** database, never production.
- The purpose is conceptual understanding, so correctness and traceability of every number beat polish.

## 10. Open questions for the owner (ask before building the relevant milestone)

1. Is the optional GPT-2 comparison (M7, a one-time download of ~500 MB) wanted at all?
2. Should field craft ever be used as training text? It's AI-generated, so the default is no.
3. Is a desktop browser enough? (That's the assumption; phone isn't required.)
4. Is this plan reviewed by CODEX1 before handing off? (Recommended.)

## 11. Relation to the birds app

This lab is separate from the birds app's tag-accuracy work (td-894144, `~/birds/docs/2026-09-26-ai-tag-accuracy-plan-CC.md`). That plan's Phase 2 uses *classifiers* (never generators) for tags. This lab demonstrates *generation*, including why it hallucinates, which is exactly why the birds app refuses to let generation assign tags.
