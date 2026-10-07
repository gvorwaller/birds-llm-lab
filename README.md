# birds-llm-lab

A local-only teaching app for inspecting how a small decoder-only transformer
tokenizes bird text, trains, attends, predicts, generates, and hallucinates.

Milestones 0–6 are implemented: the local app
shell, read-only corpus workflow, secured loopback service, deterministic byte-pair
tokenizer, trainable from-scratch transformer, inspectable forward pass and
embeddings, seeded generation, and worker-based live training.

The owner-facing app is available at [http://localhost:5301](http://localhost:5301).
The Data workbench exports the corpus, trains, and selects checkpoints without
requiring Terminal. The Tokenizer workbench exposes bytes, ids, and merge replay.
The Forward pass, Attention, and Embeddings workbenches use the selected validated
checkpoint and expose real model values. The Generation and Training workbenches
show actual sampling and optimizer updates. The Evidence workbench searches
exported corpus fields with source split and field labels. It also replays a
verified checkpoint example and compares its selected token with the exported
corpus and the held-out species record.
The Parameters workbench covers every named tensor in the selected checkpoint,
shows distribution statistics and a bounded heatmap, and links to its forward trace.
The always-reachable Glossary defines every term from the original plan with
plain language, applicable formulas, and links into the workbenches.
The optional GPT-2 comparison at `/comparison` uses a separately cached,
pretrained GPT-2 small model. It compares token boundaries, unfiltered next-token
probabilities, and 20-token greedy continuations with the selected tiny checkpoint.
The ONNX model used here does not expose attention matrices; inspect the tiny
model's heads in the Attention workbench.

## Optional local GPT-2 setup

GPT-2 is not downloaded by `npm install` or by opening the app. On a Mac with
Node 22 available, run `npm run gpt2:install` once from this checkout. That
downloads about 480 MiB into the ignored `.cache/gpt2` directory and verifies a
local-only inference call. Restart the installed service with
`npm run service:restart`, then open [the comparison workbench](http://localhost:5301/comparison).
Later analyses use the cache with remote model loading disabled. The original
tiny-model workbenches work without GPT-2 installed. To remove GPT-2, delete
`.cache/gpt2` and restart the service.

If this Mac has a global `libvips` installation and `npm install` tries to build
`sharp` from source, run the install with `SHARP_IGNORE_GLOBAL_LIBVIPS=1`.

- [Original product plan](docs/plan.md)
- [Implementation plan](docs/implementation-plan.md)

The implementation must not connect to production services or include database
credentials. Its only application data source will be an explicitly exported,
read-only snapshot from the local `birds_test` database.

## Agent commands

Use the project-local Node 22 runtime installed through npm scripts:

- `npm run check` — Svelte and strict TypeScript checks.
- `npm test -- --run` — unit and integration tests.
- `npm run build` — production app build.
- `npm run export` — the same fixed-endpoint read-only export used by the Data page.
- `npm run train:tokenizer -- --vocab-size 1024` — train an atomic tokenizer artifact
  from the exported train split.
- `npm run verify:m0` — artifact reconciliation and local-only/credential scan.
- `npm run verify:m1` — full-corpus tokenizer round-trip and determinism gate.
- `npm run verify:m2-determinism` — fixed-seed training/resume determinism gate.
- `npm run verify:m6-example` — replay the saved example against the local
  `ready-v1` checkpoint and verify its corpus evidence.
- `npm run verify:m6-parameters` — reconcile every `ready-v1` tensor and its
  histogram with the model config and forward trace.
- `npm run test:browser` — keyboard/layout smoke at 1024 px and 1280 px widths.
- `npm run service:status` — report the LaunchAgent state.
- `npm run gpt2:install` — optional one-time GPT-2 download and offline check.

Licensed under the [MIT License](LICENSE).
