# AGENTS.md

Canonical agent operating model for this repo. Keep this file short: it is the one agent-facing contract, not a command runbook.

## Repository Orientation

- Synthesia2MIDI is a PySide6 desktop app that analyzes Synthesia piano videos and exports MIDI.
- Code should keep one-way dependencies: `GUI → workflows → detection → core`. See [`ARCHITECTURE.MD`](ARCHITECTURE.MD) for the full map.
- `main.py` is a root-window composition facade. Do not add workflow bodies or signal-compatibility wrappers there; wire Qt signals to focused controllers/workflows where possible.
- Preserve existing app behavior and per-video config/calibration compatibility unless the task explicitly includes a migration.

## Read Before Editing

- Project status, rationale, blockers, and parking lot: [`PROJECT_LOG.md`](PROJECT_LOG.md)
- Current and historical work: [`backlog/tasks/`](backlog/tasks/)
- Durable workflow/product docs and decisions: [`backlog/docs/`](backlog/docs/) and [`backlog/decisions/`](backlog/decisions/)
- Current Spec Kit feature pointer: [`.specify/feature.json`](.specify/feature.json), then the referenced feature under [`specs/`](specs/) if active.
- Architecture and subsystem boundaries: [`ARCHITECTURE.MD`](ARCHITECTURE.MD)
- Task ownership, scoping, and handoff format: [`docs/task-boundaries.md`](docs/task-boundaries.md)
- Verification and testing commands: [`docs/testing.md`](docs/testing.md)
- Main-window refactor status, when touching that area: [`docs/main-py-refactor-checklist.md`](docs/main-py-refactor-checklist.md)

## Operating Rules

- Start every task with `git status --short --branch` and protect unrelated user/agent changes.
- Do not create a worktree unless Jeff explicitly authorizes it.
- Before starting a new feature branch or task, audit existing branches and worktrees. Any completed feature must be verified and merged into `main` first; if merge approval is required, stop and ask rather than stacking new work on an unmerged branch.
- After a feature is merged, remove its worktree and delete its merged local branch unless Jeff explicitly asks to retain it. Treat remote branch deletion as a push operation and wait for authorization.
- Keep changes bounded to the assigned task; do not opportunistically refactor neighboring systems.
- Add or update tests before behavior changes and before risky refactors.
- Do not commit generated media, logs, extracted frames, MIDI files, `.venv`, or Rust `target/` output.
- For multi-step refactors, setup changes, and UI/layout work, make frequent small checkpoint commits after each coherent slice passes relevant verification.
- Use one Backlog task with acceptance criteria for non-trivial work. Backlog owns status.
- Use Spec Kit for non-trivial feature planning. Spec Kit owns feature detail, not project status.

## Command and Runbook Ownership

Do not duplicate setup, launch, test, or smoke-command blocks in this file. Canonical command locations are:

- User setup and launch: [`README.md`](README.md)
- Local verification, test strategy, GUI smoke, setup/launcher checks, and Rust editor checks: [`docs/testing.md`](docs/testing.md)
- Architecture context and subsystem boundaries: [`ARCHITECTURE.MD`](ARCHITECTURE.MD)

When a command changes, update the owning document first, then update links or short references elsewhere.

## Documentation Updates

- Keep docs focused on stable contracts, commands, decisions, and boundaries.
- Record durable project decisions in [`backlog/decisions/`](backlog/decisions/) and durable workflow/product planning in [`backlog/docs/`](backlog/docs/).
- Keep [`PROJECT_LOG.md`](PROJECT_LOG.md) as a concise state handoff until specific content is migrated to Backlog.
- Update [`ARCHITECTURE.MD`](ARCHITECTURE.MD) for architecture changes, [`docs/task-boundaries.md`](docs/task-boundaries.md) for ownership/scoping changes, and [`docs/testing.md`](docs/testing.md) for verification changes.
- Do not create a second canonical agent operating model. If this file is ever retired, replace it with one linked successor instead of keeping parallel guidance.

<!-- SPECKIT START -->
Current Spec Kit feature: `specs/001-release-recovery/`. Read `.specify/feature.json`, then `spec.md`, `plan.md`, and `tasks.md` in that directory.
<!-- SPECKIT END -->
