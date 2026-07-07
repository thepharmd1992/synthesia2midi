---
id: TASK-14
title: Warn when lit exemplar matches unlit calibration
status: Done
assignee: []
created_date: '2026-07-06 00:00'
labels:
  - ui
  - calibration
  - detection
dependencies: []
modified_files:
  - synthesia2midi/synthesia2midi/gui/calibration_interaction_controller.py
  - tests/test_bugfix_regressions.py
priority: high
ordinal: 14000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent users from accidentally capturing lit exemplar colors from an unlit frame by warning when the sampled lit exemplar is effectively the same as the selected overlay's unlit calibration.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Lit exemplar capture compares the sampled overlay color against the overlay's unlit reference color when available.
- [x] #2 Near-identical lit/unlit samples are rejected before exemplar color, histogram, hue calibration, or autosave state changes are written.
- [x] #3 The user sees a warning telling them the sample does not look lit enough and to move to a frame where the key is lit.
- [x] #4 Valid lit samples still calibrate through the existing success path.
- [x] #5 Regression tests cover rejected near-unlit samples and valid distinct samples.
<!-- AC:END -->

## Verification

<!-- SECTION:NOTES:BEGIN -->
- `.venv/bin/python -m pytest tests/test_bugfix_regressions.py tests/test_localization.py tests/test_ui_string_audit.py`
<!-- SECTION:NOTES:END -->
