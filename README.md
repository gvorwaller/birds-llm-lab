# birds-llm-lab

A local-only teaching app for inspecting how a small decoder-only transformer
tokenizes bird text, trains, attends, predicts, generates, and hallucinates.

Milestones 0 and 1 are implemented: the local app shell, read-only corpus workflow,
secured loopback service, deterministic byte-pair tokenizer, and interactive Tokenizer Lab.

The owner-facing app is available at [http://localhost:5301](http://localhost:5301).
The Data workbench exports and inspects the corpus without requiring Terminal. The
Tokenizer workbench tokenizes arbitrary text, replays learned merges, and exposes the
stored 259-entry byte vocabulary and 1,024-entry trained vocabulary.

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
- `npm run test:browser` — keyboard/layout smoke at 1024 px and 1280 px widths.
- `npm run service:status` — report the LaunchAgent state.

Licensed under the [MIT License](LICENSE).
