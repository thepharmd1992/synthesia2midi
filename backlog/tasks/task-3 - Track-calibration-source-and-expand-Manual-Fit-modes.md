---
id: TASK-3
title: Track calibration source and expand Manual Fit modes
status: Done
assignee: []
created_date: '2026-06-09 00:28'
labels:
  - ui
  - calibration
  - manual-fit
dependencies: []
modified_files:
  - synthesia2midi/synthesia2midi/core/app_state.py
  - synthesia2midi/synthesia2midi/config_manager.py
  - synthesia2midi/synthesia2midi/gui/calibration_wizard_controller.py
  - synthesia2midi/synthesia2midi/gui/manual_keyboard_fit_dialog.py
  - synthesia2midi/synthesia2midi/gui/manual_keyboard_fit_controller.py
  - tests
priority: high
ordinal: 3000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Persist whether current overlays came from auto or manual calibration, route Edit Current Calibration accordingly, stabilize Manual Fit mode layout, and add all-white/all-black fit modes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Auto-generated overlays persist an auto calibration source and Edit Current Calibration routes to auto tuning
- [x] #2 Manual-generated overlays persist a manual calibration source and Edit Current Calibration routes to Manual Fit
- [x] #3 Manual Fit mode selector stays in a stable position when switching to Single Overlay
- [x] #4 Manual Fit includes All Whites and All Blacks modes
- [x] #5 All Overlays exposes both white and black width controls
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
- Calibration source is persisted in config and overlay metadata.
- Edit Current Calibration routes manual overlays to Manual Fit and auto overlays to auto tuning.
- Manual Fit now supports All Overlays, All Whites, All Blacks, Select Overlays, and Single Overlay.
- Verification: `git diff --check`, `.venv/bin/python -m compileall -q synthesia2midi`, `.venv/bin/python -m pytest`.
<!-- SECTION:NOTES:END -->
