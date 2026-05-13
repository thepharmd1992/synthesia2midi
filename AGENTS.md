# AGENTS.md

Coding-agent instructions for this repo.

## Read First

- Architecture and subsystem boundaries: [`ARCHITECTURE.MD`](ARCHITECTURE.MD)
- Main-window refactor status: [`docs/main-py-refactor-checklist.md`](docs/main-py-refactor-checklist.md)
- Task routing conventions: [`docs/task-boundaries.md`](docs/task-boundaries.md)

## Non-Negotiables

- Preserve existing app behavior unless the task explicitly says otherwise.
- Preserve per-video config/calibration compatibility unless the task explicitly includes a migration.
- Do not commit generated media, logs, extracted frames, MIDI files, `.venv`, or Rust `target/` output.
- Start every task with `git status --short --branch` and protect unrelated user/agent changes.
- For multi-step refactors or setup changes, make frequent small commits after passing relevant verification.
- Keep changes bounded. Do not opportunistically refactor neighboring systems.
- Add or update tests before behavior changes and before risky refactors.
- Keep docs focused on stable contracts, commands, decisions, and boundaries.

## Verification Defaults

Run from repo root:

```bash
git diff --check
.venv/bin/python -m compileall -q synthesia2midi
PYTHONPATH=synthesia2midi QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q
```

For GUI wiring/refactor work, also run the offscreen smoke in `ARCHITECTURE.MD`.

For setup/launcher work:

```bash
PYTHONPATH=synthesia2midi QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q tests/test_setup_and_launch.py
.venv/bin/python setup_env.py --check
```

For Rust touch-up editor work:

```bash
cargo check --manifest-path tools/midi_touchup_editor_rust/Cargo.toml
```

## Setup / Launcher Contract

Supported user commands are:

```bash
python3 setup_env.py
python3 run.py
```

Windows equivalents:

```powershell
py setup_env.py
py run.py
```

Do not reintroduce Textual/TUI installers or OS-specific setup/run wrapper scripts unless explicitly requested.

## Current Architecture Rule

`main.py` is a root-window composition facade. Do not add workflow bodies or signal-compatibility wrappers there. Wire Qt signals directly to focused controllers/workflows where possible. See `ARCHITECTURE.MD` for the controller map and dependency direction.

## Kanban

Tenant/board usage may vary by task. If using Hermes Kanban, keep cards scoped to bounded work and update them when checkpoint commits land.
