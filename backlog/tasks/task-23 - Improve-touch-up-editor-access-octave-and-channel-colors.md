---
id: TASK-23
title: Improve touch-up editor access octave and channel colors
status: In Progress
assignee: []
created_date: '2026-07-12 00:00'
updated_date: '2026-07-12 00:00'
labels:
  - midi
  - touch-up
  - ux
  - rust
dependencies:
  - TASK-22
priority: high
ordinal: 23000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the MIDI touch-up editor easier to open, add safe whole-file octave adjustment, and preserve Synthesia2MIDI color-family identity in falling bars and active piano keys through portable MIDI metadata.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The File menu and Settings footer can open any MIDI in the touch-up editor without a loaded video, while the Welcome dialog remains unchanged.
- [x] #2 The MIDI picker starts in the Synthesia2MIDI Desktop export folder when available.
- [x] #3 Octave down/up shifts the entire document by 12 semitones as one Undo/Redo command and displays the cumulative offset.
- [x] #4 An octave shift is rejected atomically if any note would leave the visible A0-C8 piano range.
- [x] #5 Generated MIDI files carry validated version-one channel metadata for calibrated Natural and Sharp / Flat colors.
- [x] #6 The Rust editor preserves valid color metadata and safely falls back for ordinary or malformed MIDI files.
- [x] #7 Falling bars and active keys use readable exemplar-derived colors or distinct fallback colors for channels 0 through 3.
- [x] #8 Simultaneous same-pitch notes on different channels remain separately visible in bars and key highlights.
- [ ] #9 Qt translations, Python/Rust tests, UI matrices, and Windows/macOS packaged smokes pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Approved design: `docs/superpowers/specs/2026-07-12-touch-up-editor-access-octave-colors-design.md`.

Use a versioned standard MIDI text meta event as the portable color contract. Do not introduce a sidecar file or a color-bearing process argument.

Local verification on 2026-07-12:

- `548` Python tests passed with the existing Qt deprecation warnings only; compileall, Ruff's syntax/name gate, and `git diff --check` passed.
- Rust formatting passed, all `21` Rust tests passed, and `cargo check` passed with the existing unused `AudioTelemetry::meter` warning only.
- The expanded `qps` UI matrix rendered `25` nonblank surfaces with no detected clipping under `logs/ux-audit/touchup-access-review-fixes/`.
- `packaging/build_release.py --version v0.2.1-dev` smoke-launched and produced `Synthesia2MIDI-macos-arm64-v0.2.1-dev.zip` (`497654173` bytes). The archive contains arm64 app and touch-up editor executables, FFmpeg/FFprobe, `TouchUpPiano.sf2`, the app icon, and all six production translation catalogs.
- Independent review found no critical defects. Follow-up fixes made whole-document octave changes linear in document size, kept black/white/near-white derived morphology colors distinct and readable, and aligned Python's empty/invalid metadata rejection with the Rust parser.

Remaining remote gate: the branch is intentionally unpushed. Criterion #9 stays open until Jeff authorizes a push, the GitHub Python/Rust Windows and macOS matrices pass, and the tag-triggered Windows x64 plus Apple Silicon package smokes pass. TASK-23 remains `In Progress` until that evidence exists.
<!-- SECTION:NOTES:END -->
