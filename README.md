# birds-llm-lab

A local-only teaching app for inspecting how a small decoder-only transformer
tokenizes bird text, trains, attends, predicts, generates, and hallucinates.

Milestones 0–3 are implemented: the local app shell, read-only corpus workflow,
secured loopback service, deterministic byte-pair tokenizer, trainable from-scratch
transformer, and inspectable forward-pass, attention, and embeddings screens.

The owner-facing app is available at [http://localhost:5301](http://localhost:5301).
The Data workbench exports the corpus, trains, and selects checkpoints without
requiring Terminal. The Tokenizer workbench exposes bytes, ids, and merge replay.
The Forward pass, Attention, and Embeddings workbenches use the selected validated
checkpoint and expose real model values. Sampling/generation, live browser training,
and corpus-evidence screens are planned for Milestones 4–6.

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
- `npm run test:browser` — keyboard/layout smoke at 1024 px and 1280 px widths.
- `npm run service:status` — report the LaunchAgent state.

Licensed under the [MIT License](LICENSE).
