---
id: TASK-24
title: Make source selector the only startup window
status: Done
assignee: []
created_date: '2026-07-12 00:00'
updated_date: '2026-07-12 00:00'
labels:
  - ux
  - startup
  - qt
dependencies:
  - TASK-23
priority: high
ordinal: 24000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the main workspace hidden during startup, return secondary-dialog cancellations to Select Video Source, and exit when the source selector itself is cancelled.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Select Video Source is the only visible Synthesia2MIDI window at initial startup.
- [x] #2 The main workspace appears only after a local, recent, or downloaded video loads successfully.
- [x] #3 Cancelling the local file picker or YouTube dialog leaves Select Video Source open and the main workspace hidden.
- [x] #4 Cancelling or closing Select Video Source exits Synthesia2MIDI completely.
- [x] #5 Existing in-session File menu and empty-state video actions remain available and behaviorally compatible.
- [x] #6 Source, launcher, UI-matrix, and packaged-startup verification pass before local integration.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Approved design: `docs/superpowers/specs/2026-07-12-startup-source-flow-design.md`.

Do not show the main window temporarily and then hide it. The launch paths must leave it hidden until the startup coordinator confirms a loaded video session.

Local verification completed on 2026-07-12:

- The focused startup regression gate passed all `32` tests. The full Python suite passed all `558` tests with the existing Qt deprecation warnings only; compileall, Ruff's syntax/name gate, and both diff checks passed.
- Rust formatting and `cargo check` passed, and all `21` Rust tests passed with the existing unused `AudioTelemetry::meter` warning only.
- The `qps` UI matrix rendered `25` nonblank surfaces with no detected clipping under `logs/ux-audit/startup-source-flow/`.
- The Desktop developer launcher reached the source selector without exposing the main workspace. The packaged macOS app was then exercised with Computer Use: cancelling the native file picker returned to the same selector, and cancelling the selector terminated the process.
- `packaging/build_release.py --version v0.2.1-dev` passed its startup smoke and produced `Synthesia2MIDI-macos-arm64-v0.2.1-dev.zip` (`497655689` bytes). The archive passed integrity checks and contains arm64 app and Rust editor executables, FFmpeg/FFprobe, Deno, the app icon, and all production translation catalogs.
- No translation source or compiled catalog changed. The UI-string manifest was regenerated only to update source line locations after the startup refactor.
- Independent review found no Critical, Important, or Minor findings. Remaining remote Windows/macOS package and cross-platform CI checks are intentionally deferred until Jeff authorizes the combined push.
<!-- SECTION:NOTES:END -->
