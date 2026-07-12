---
id: TASK-23
title: Improve touch-up editor access octave and channel colors
status: To Do
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
- [ ] #1 The File menu and Settings footer can open any MIDI in the touch-up editor without a loaded video, while the Welcome dialog remains unchanged.
- [ ] #2 The MIDI picker starts in the Synthesia2MIDI Desktop export folder when available.
- [ ] #3 Octave down/up shifts the entire document by 12 semitones as one Undo/Redo command and displays the cumulative offset.
- [ ] #4 An octave shift is rejected atomically if any note would leave the visible A0-C8 piano range.
- [ ] #5 Generated MIDI files carry validated version-one channel metadata for calibrated Natural and Sharp / Flat colors.
- [ ] #6 The Rust editor preserves valid color metadata and safely falls back for ordinary or malformed MIDI files.
- [ ] #7 Falling bars and active keys use readable exemplar-derived colors or distinct fallback colors for channels 0 through 3.
- [ ] #8 Simultaneous same-pitch notes on different channels remain separately visible in bars and key highlights.
- [ ] #9 Qt translations, Python/Rust tests, UI matrices, and Windows/macOS packaged smokes pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Approved design: `docs/superpowers/specs/2026-07-12-touch-up-editor-access-octave-colors-design.md`.

Use a versioned standard MIDI text meta event as the portable color contract. Do not introduce a sidecar file or a color-bearing process argument.
<!-- SECTION:NOTES:END -->
