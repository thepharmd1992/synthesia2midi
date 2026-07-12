---
id: TASK-22
title: Support four color families and eight lit exemplars
status: To Do
assignee: []
created_date: '2026-07-12 00:00'
updated_date: '2026-07-12 00:00'
labels:
  - calibration
  - detection
  - midi
  - ux
dependencies:
  - TASK-20
priority: high
ordinal: 22000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Support Synthesia videos containing up to four stable note-color families, with separate Natural and Sharp / Flat lit exemplars for each family. Discover additional families automatically, retain a bounded manual fallback, and export each active family to its own MIDI channel without breaking existing two-family calibration files.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The calibration model supports one through four color families with Natural and Sharp / Flat exemplars.
- [ ] #2 Existing LW/LB/RW/RB configurations load unchanged and retain their family/channel identity.
- [ ] #3 The assisted scanner performs lightweight 10-frame discovery through the full video unless four complete families are found, while detailed refinement runs only around promising events.
- [ ] #4 Repeated evidence and temporal stability prevent transient flashes or intro animations from creating families.
- [ ] #5 The Calibration and assisted-review interfaces use compact dynamic family grids with Add/Remove controls capped at four families.
- [ ] #6 Missing exemplar rows can be manually set or marked not present, and incomplete present rows block conversion.
- [ ] #7 Colors 1 through 4 export to separate MIDI channels and remain stable across rescans.
- [ ] #8 Dynamic exemplar state and diagnostics round-trip through saved configuration formats.
- [ ] #9 All shipped translations, UI matrices, focused calibration tests, full tests, and Windows/macOS release smokes pass before v0.2.0.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Approved design: `docs/superpowers/specs/2026-07-12-four-color-families-design.md`.

Reference acceptance video: `https://www.youtube.com/watch?v=7i9ZcXGk4ZI`. Use it only as a local acceptance input; do not commit the video or extracted frames.
<!-- SECTION:NOTES:END -->
