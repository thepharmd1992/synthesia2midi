---
id: TASK-4
title: Lock Manual Fit overlays to keyboard box
status: Done
assignee: []
created_date: '2026-06-09 00:00'
labels:
  - ui
  - calibration
  - manual-fit
dependencies: []
modified_files:
  - synthesia2midi/synthesia2midi/core/app_state.py
  - synthesia2midi/synthesia2midi/config_manager.py
  - synthesia2midi/synthesia2midi/workflows/manual_keyboard_fit.py
  - synthesia2midi/synthesia2midi/gui/manual_keyboard_fit_dialog.py
  - synthesia2midi/synthesia2midi/gui/manual_keyboard_fit_controller.py
  - tests
priority: high
ordinal: 4000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Persist the manual keyboard box, keep Manual Fit overlay geometry inside that box, let users redraw the box, and warn when the lower box edge looks like non-key background.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Manual keyboard box persists with calibration config.
- [x] #2 Generated manual overlays store the setup keyboard box.
- [x] #3 Group, width, local, and single-overlay transforms keep final overlay geometry inside the keyboard box.
- [x] #4 Rotated overlay corners are constrained inside the keyboard box.
- [x] #5 Manual Fit exposes an Edit Keyboard Box action that redraws the boundary and returns to fit mode.
- [x] #6 Apply warns when lower keyboard-box end bands look like background.
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
- Manual Fit stores the keyboard box in calibration config and overlay metadata.
- Final overlay geometry is constrained against the keyboard box, including rotated corners.
- Users can redraw the keyboard box from Manual Fit.
- Apply warns when lower edge bands look like background instead of white keys.
- Verification: `git diff --check`, `.venv/bin/python -m compileall -q synthesia2midi`, `.venv/bin/python -m pytest`.
<!-- SECTION:NOTES:END -->
