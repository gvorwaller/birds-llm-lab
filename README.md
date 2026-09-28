# birds-llm-lab

A local-only teaching app for inspecting how a small decoder-only transformer
tokenizes bird text, trains, attends, predicts, generates, and hallucinates.

Milestone 0 is implemented: the local app shell, validated corpus schemas,
read-only export workflow, secured loopback service, and macOS LaunchAgent.

The owner-facing app is available at [http://localhost:5301](http://localhost:5301).
The Data workbench exports and inspects the corpus without requiring Terminal.

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
- `npm run verify:m0` — artifact reconciliation and local-only/credential scan.
- `npm run service:status` — report the LaunchAgent state.

Licensed under the [MIT License](LICENSE).
