# birds-llm-lab

A local-only teaching app for inspecting how a small decoder-only transformer
tokenizes bird text, trains, attends, predicts, generates, and hallucinates.

This repository is currently in the planning stage. Nothing has been built yet.

- [Original product plan](docs/plan.md)
- [Implementation plan](docs/implementation-plan.md)

The implementation must not connect to production services or include database
credentials. Its only application data source will be an explicitly exported,
read-only snapshot from the local `birds_test` database.

Licensed under the [MIT License](LICENSE).
