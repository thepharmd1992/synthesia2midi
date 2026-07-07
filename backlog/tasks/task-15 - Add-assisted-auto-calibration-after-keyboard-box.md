---
id: TASK-15
title: Add assisted auto-calibration after keyboard box selection
status: To Do
assignee: []
created_date: '2026-07-07 00:00'
labels:
  - ui
  - calibration
  - detection
dependencies:
  - TASK-14
modified_files: []
priority: high
ordinal: 15000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a V1 assisted calibration flow that starts after the user draws a keyboard bounding box on a good unlit frame. The app should generate overlays, warn if the baseline frame appears to contain lit keys, capture unlit references, scan the video for lit exemplar colors, assign legacy exemplar slots by color family, and ask for confirmation before saving.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The user still draws the keyboard bounding box manually before assisted calibration begins.
- [ ] #2 Successful auto-detection can immediately capture unlit reference colors and histograms from the selected baseline frame.
- [ ] #3 A reusable soft warning detects likely lit overlays during unlit calibration and names likely lit notes when confidence is high.
- [ ] #4 The existing manual "Calibrate Unlit All Keys" path uses the same soft warning before overwriting unlit data.
- [ ] #5 The assisted scan searches overlay ROIs across video frames for lit exemplar candidates without relying on physical left/right keyboard position.
- [ ] #6 Candidate lit colors are clustered into color families and mapped into legacy `LW`, `LB`, `RW`, and `RB` slots by family and key color.
- [ ] #7 One-color or partial-color videos can leave absent exemplar slots disabled or unchanged only after user confirmation.
- [ ] #8 The user sees a progress/cancel path while scanning and a confirmation summary before exemplar changes are saved.
- [ ] #9 Tests cover the unlit-frame guard, exemplar candidate detection, color-family assignment, partial results, cancellation, and proposal application.
- [ ] #10 Local exploratory validation compares the Game of Thrones video proposal against the saved target INI and overlays, excluding octave transpose.
<!-- AC:END -->

## Verification

<!-- SECTION:NOTES:BEGIN -->
- Design spec: `docs/superpowers/specs/2026-07-07-assisted-auto-calibration-design.md`
- Expected gates will include `git diff --check`, compileall, focused calibration tests, and full pytest after implementation.
<!-- SECTION:NOTES:END -->
