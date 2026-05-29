# Agent Task Boundaries

Use this file to route coding-agent work, prevent overlapping edits, and make handoffs reviewable. It is not a command runbook.

Canonical docs:

- Agent operating model: [`../AGENTS.md`](../AGENTS.md)
- Project state and durable decisions: [`../PROJECT_LOG.md`](../PROJECT_LOG.md)
- Architecture and subsystem boundaries: [`../ARCHITECTURE.MD`](../ARCHITECTURE.MD)
- Verification runbook: [`testing.md`](testing.md)

`AGENTS.md` is the canonical agent contract. This file only defines ownership and handoff expectations.

## Universal Preflight

Every implementation task starts with the repo preflight required by `AGENTS.md`: inspect git status, identify unrelated local changes, and record the intended files before editing.

Do not overwrite another user or agent's work. If the intended files are already modified, either narrow the task, coordinate in Kanban, or hand off/block with a clear owner and dependency.

## Ownership and Routing

| Route | Involve when | Primary paths / areas | Notes |
|---|---|---|---|
| `default` / general | Agent docs, audits, decomposition, project-log upkeep, small glue edits, or unclear ownership triage. | `AGENTS.md`, `PROJECT_LOG.md`, `docs/*.md`, Kanban task setup. | Keep source-of-truth docs concise and non-duplicative. Route implementation to a specialist once the affected subsystem is clear. |
| `backend` | Behavior, data flow, state, detection, MIDI conversion, workflow orchestration, compatibility facades. | `synthesia2midi/synthesia2midi/core/`, `detection/`, `workflows/`, `midi_generator.py`, `midi_reader.py`, backend-facing parts of `main.py`. | Preserve existing behavior and per-video config/calibration compatibility unless the task explicitly includes a migration. Require tests or characterization evidence for risky behavior changes. |
| `frontend` | Qt UI, window/widget behavior, layout, signal wiring, user-facing interactions. | `synthesia2midi/synthesia2midi/gui/`, GUI-facing parts of `main.py`, controller wiring that directly affects UI behavior. | Avoid algorithm or persistence changes unless paired with a backend owner. Use offscreen smoke verification for GUI wiring/refactor work. |
| `ops` | Setup, launch, CI, dependencies, platform assumptions, FFmpeg/YouTube/network tooling, release packaging. | `.github/workflows/`, `setup_env.py`, `run.py`, dependency files, `utils/ffmpeg_helper.py`, platform-sensitive loader/downloader paths. | Keep setup boring and cross-platform. Do not duplicate setup/run instructions across docs; link the canonical runbook instead. |
| `reviewer` | Any code change before it is treated as done; docs/config changes when they affect routing, setup, behavior, or long-term contracts. | All changed files. | Review scope control, behavior drift, generated files, test evidence, and whether the change matches the owning profile's boundary. |

If a task crosses multiple rows, name the primary owner and the secondary owner in the Kanban comment or handoff. Do not silently broaden a card into a cross-subsystem refactor.

## File Ownership Expectations

- One card should own one coherent slice. Split work when a change needs independent backend, frontend, and ops decisions.
- A worker may inspect outside its owned paths, but should only edit outside them when the task explicitly requires it and the handoff names why.
- Existing uncommitted changes belong to whoever made them until proven otherwise. Avoid those files or coordinate before editing.
- Keep compatibility facades stable while extracting internals. For this repo, `main.py` and `MonolithicPianoDetector` are especially sensitive boundary surfaces.
- Generated media, logs, extracted frames, MIDI outputs, virtualenvs, and Rust `target/` output are never owned deliverables unless a task explicitly says otherwise.

## Handoff Rules

Use Kanban for task state and `PROJECT_LOG.md` for durable project state.

Hand off to:

- `backend` when the next step changes app behavior, detection, MIDI conversion, state/config, compatibility facades, or workflow execution.
- `frontend` when the next step changes Qt widgets, layout, visual feedback, signal wiring, or user interaction.
- `ops` when the next step changes CI, setup, dependencies, launch behavior, platform checks, or packaging/release mechanics.
- `reviewer` after implementation work that changes code, behavior, setup, routing contracts, or other long-lived project rules.
- `default` / general when the next step is triage, documentation consolidation, planning, audit, or cross-agent coordination rather than subsystem implementation.

A handoff must say:

```text
Owner / next profile:
Files changed:
Files intentionally avoided:
Behavior changes: none / describe
Dependencies or blockers:
Verification run:
Known risks:
Project log updated: yes/no + section
Next recommended task:
```

## Dependencies, Blockers, and Ownership Changes

Record transient execution details in Kanban comments. Record durable project state in `PROJECT_LOG.md`.

Update `PROJECT_LOG.md` when any of these will matter to the next worker:

- A blocker changes the active path or definition of done.
- A dependency must finish before another task can proceed.
- Ownership changes because the affected subsystem is different from the original card.
- A decision changes the agent operating model, architecture boundary, setup contract, or verification expectations.
- A non-blocking discovery should be parked instead of fixed immediately.

Use the existing project-log sections instead of inventing new ones:

- `Active Task`: current owner and bounded objective, if a long-running slice is active.
- `Blockers`: blocker, owner needed, affected paths, and Kanban task ID when available.
- `Decisions / Rationale`: durable ownership, architecture, setup, or verification decisions.
- `Parking Lot`: real but non-blocking follow-up work.
- `Next Action`: the next owner/profile and the smallest useful next step.

Keep project-log entries short. Do not paste command output, full diffs, or temporary debugging notes there; put those in Kanban comments or the task handoff.

## Scope Control

- Extract one responsibility at a time.
- Avoid mixing extraction with behavior changes.
- Add or update characterization tests before changing fragile orchestration.
- Stop and split the work if a controller, workflow, or agent plan starts becoming a new god object.
- When a discovered issue is real but not blocking the current card, park it in `PROJECT_LOG.md` or create a follow-up task instead of expanding the current task.
