# Project Log

## Original Goal

Make Synthesia2MIDI an agent-first codebase so future development can be done safely and efficiently by coding agents.

## Definition of Done

- `main.py` no longer owns major unrelated responsibilities as a god object.
- `detection/monolithic_detector.py` no longer contains the full manual auto-detection implementation as a god object; it remains an API-compatible facade.
- Agent onboarding, task boundaries, and verification commands are documented in tracked repo files.
- A pytest scaffold and CI gates exist before broad refactor work proceeds.
- Refactor behavior is protected by compile/import/unit/offscreen smoke checks.
- Confirmed utility bugs discovered during audit are fixed with regression tests.

## Smallest Useful Version

A refactor wave that keeps behavior stable while extracting low-risk `main.py` responsibilities into controllers/workflows, splitting the manual ROI detector into focused stages, and establishing tests/docs/CI for future agents.

## Current Focus

Backlog and Spec Kit workflow adoption: root `AGENTS.md` remains the canonical agent operating contract, Backlog now owns planning/status memory, Spec Kit owns non-trivial feature planning, and existing root docs remain canonical until deliberately migrated.

## Completed

- 2026-06-08 Backlog/Spec Kit workflow foundation: installed Backlog under `backlog/`, Spec Kit under `.specify/` with Codex skills, created an inactive Spec Kit feature pointer, recorded the planning decision, and added starter process/product/parking-lot Backlog docs.
- 2026-05-29 M0 docs consistency pass: added a docs index, confirmed root `AGENTS.md` as the single canonical agent operating model, and moved duplicated active command/runbook instructions back to `README.md` and `docs/testing.md` links.
- 2026-05-29 M0 operating-model update: confirmed the existing root `PROJECT_LOG.md` is the project state log; root `AGENTS.md` remains tracked and canonical for agent instructions; follow-on doc-link work should point README/docs index readers here without duplicating command runbooks.
- Fixed blocking f-string syntax error in `synthesia2midi/synthesia2midi/main.py`.
- Created Kanban roadmap under tenant `synthesia2midi`; root epic `t_77771f49`.
- Added implementation plan at `docs/plans/2026-05-11-agent-first-refactor.md`.
- Added tracked `AGENTS.md` as the repo's public agent contract.
- Added pytest scaffold, baseline tests, and CI gates for Python/Rust.
- Fixed confirmed ROI overlay-crop and relative-path MIDI-save bugs with regression tests.
- Extracted `main.py` responsibilities into video-to-frames, MIDI touch-up, video session, calibration interaction, calibration wizard, spark, shadow, and overlay interaction controllers.
- Added synthetic manual auto-detector characterization tests covering black/white key detection and type-aware note assignment.
- Split `detection/monolithic_detector.py` into an API-compatible facade plus focused black-key, white-key solver, note-assignment, defaults, geometry, and visualization modules.
- Replaced the Textual/TUI installer and OS-specific setup/run wrappers with one cross-platform `setup_env.py` and venv-aware `run.py`; FFmpeg is a hard requirement.
- Ran independent code review and addressed the two findings: Windows CI shell compatibility and a non-widget `QMessageBox` parent.

## Active Task

TASK-26 is in progress on `001-release-recovery`: repair packaged helper resolution and validation, publish verified v0.2.2, fix Guide alignment review and Unicode MIDI metadata, then reconcile TASK-9. Issue #6 is out of scope because its requested range/trim behavior already exists.

## Blockers

None known.

## Parking Lot

- Larger redesign of detection algorithms beyond behavior-preserving extraction.
- Full packaging/release overhaul.
- UI redesign or feature additions.
- Large real-video fixture corpus.

## Decisions / Rationale

- Use Backlog as the planning memory and status surface. Use Spec Kit for non-trivial feature execution planning. Keep existing root docs canonical until specific content is deliberately migrated.
- Track `AGENTS.md` in this repo because future development is intentionally agent-driven. The previous `.gitignore` rule for `AGENTS.md` was local-agent oriented and conflicted with the new operating model.
- Use characterization-first extraction instead of rewriting `main.py` from scratch. The app reportedly works, so risk is behavior drift, not missing architecture.
- Keep thin compatibility wrappers in `main.py` during extraction so existing Qt signal wiring can remain stable.
- Keep `MonolithicPianoDetector` as the stable detector API while splitting implementation by stage; auto-detect adapter and per-video tuning should not need migration.
- Keep setup boring: one cross-platform setup script and one venv-aware launcher. README/docs/testing own the exact user and verification commands; this log should only record the decision.
- Keep root `AGENTS.md` as the canonical agent operating model. Do not create `docs/agent-operating-model.md` unless `AGENTS.md` becomes too large; link the canonical file instead.

## Next Action

Create or update a Backlog task before the next non-trivial product/refactor slice, then decide whether that slice needs Spec Kit planning.
