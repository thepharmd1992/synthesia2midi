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

Agent-first refactor wave complete; setup/launcher surface simplified to one setup script plus one venv-aware launcher.

## Completed

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

None.

## Blockers

None known.

## Parking Lot

- Larger redesign of detection algorithms beyond behavior-preserving extraction.
- Full packaging/release overhaul.
- UI redesign or feature additions.
- Large real-video fixture corpus.

## Decisions / Rationale

- Track `AGENTS.md` in this repo because future development is intentionally agent-driven. The previous `.gitignore` rule for `AGENTS.md` was local-agent oriented and conflicted with the new operating model.
- Use characterization-first extraction instead of rewriting `main.py` from scratch. The app reportedly works, so risk is behavior drift, not missing architecture.
- Keep thin compatibility wrappers in `main.py` during extraction so existing Qt signal wiring can remain stable.
- Keep `MonolithicPianoDetector` as the stable detector API while splitting implementation by stage; auto-detect adapter and per-video tuning should not need migration.
- Keep setup boring: `python3 setup_env.py`/`python3 run.py` on macOS/Linux and `py setup_env.py`/`py run.py` on Windows. Setup creates `.venv`, installs deps, and requires FFmpeg; launch re-execs into `.venv`. Do not reintroduce a TUI or OS-specific wrapper scripts without an explicit product need.

## Next Action

No immediate refactor action remains for the current wave. Recommended next task: expand detector coverage with real-world fixture cases before making algorithmic improvements, or extract remaining low-value menu/layout glue from `main.py` if GUI work resumes.
