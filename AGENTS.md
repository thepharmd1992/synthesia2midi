# AGENTS.md

Canonical agent operating model for this repo. Keep this file short: it is the one agent-facing contract, not a command runbook.

## Repository Orientation

- Synthesia2MIDI is a PySide6 desktop app that analyzes Synthesia piano videos and exports MIDI.
- Code should keep one-way dependencies: `GUI → workflows → detection → core`. See [`ARCHITECTURE.MD`](ARCHITECTURE.MD) for the full map.
- `main.py` is a root-window composition facade. Do not add workflow bodies or signal-compatibility wrappers there; wire Qt signals to focused controllers/workflows where possible.
- Preserve existing app behavior and per-video config/calibration compatibility unless the task explicitly includes a migration.

## Read Before Editing

- Project status, rationale, blockers, and parking lot: [`PROJECT_LOG.md`](PROJECT_LOG.md)
- Architecture and subsystem boundaries: [`ARCHITECTURE.MD`](ARCHITECTURE.MD)
- Task ownership, scoping, and handoff format: [`docs/task-boundaries.md`](docs/task-boundaries.md)
- Verification and testing commands: [`docs/testing.md`](docs/testing.md)
- Main-window refactor status, when touching that area: [`docs/main-py-refactor-checklist.md`](docs/main-py-refactor-checklist.md)

## Operating Rules

- Start every task with `git status --short --branch` and protect unrelated user/agent changes.
- Keep changes bounded to the assigned task; do not opportunistically refactor neighboring systems.
- Add or update tests before behavior changes and before risky refactors.
- Do not commit generated media, logs, extracted frames, MIDI files, `.venv`, or Rust `target/` output.
- For multi-step refactors, setup changes, and UI/layout work, make frequent small checkpoint commits after each coherent slice passes relevant verification.

## Command and Runbook Ownership

Do not duplicate setup, launch, test, or smoke-command blocks in this file. Canonical command locations are:

- User setup and launch: [`README.md`](README.md)
- Local verification and test strategy: [`docs/testing.md`](docs/testing.md)
- Architecture-specific smoke context: [`ARCHITECTURE.MD`](ARCHITECTURE.MD)

When a command changes, update the owning document first, then update links or short references elsewhere.

## Documentation Updates

- Keep docs focused on stable contracts, commands, decisions, and boundaries.
- Record durable project decisions, scope changes, blocker/parking-lot calls, and next actions in [`PROJECT_LOG.md`](PROJECT_LOG.md); keep transient task chatter in Kanban/comments.
- Update [`ARCHITECTURE.MD`](ARCHITECTURE.MD) for architecture changes, [`docs/task-boundaries.md`](docs/task-boundaries.md) for ownership/scoping changes, and [`docs/testing.md`](docs/testing.md) for verification changes.
- Do not create a second canonical agent operating model. If this file is ever retired, replace it with one linked successor instead of keeping parallel guidance.
