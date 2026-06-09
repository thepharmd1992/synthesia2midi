---
id: TASK-6
title: Clarify keyboard box edge edit confirmation
status: Done
assignee: []
created_date: '2026-06-09 00:00'
labels:
  - ui
  - calibration
  - manual-fit
dependencies: []
modified_files:
  - synthesia2midi/synthesia2midi/gui/manual_keyboard_fit_dialog.py
  - synthesia2midi/synthesia2midi/gui/manual_keyboard_fit_controller.py
  - tests
priority: high
ordinal: 6000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clarify Edit Keyboard Box instructions so users understand green boundary bars can be adjusted, and expose an OK action after boundary-bar movement.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Edit Keyboard Box instructions mention adjusting green boundary bars.
- [x] #2 Edge-edit mode does not imply redraw is the only path forward.
- [x] #3 Moving a boundary bar shows an OK action.
- [x] #4 OK returns the user to Manual Fit fine-tune mode.
- [x] #5 Full redraw remains available as a fallback.
<!-- AC:END -->

## Verification

<!-- SECTION:NOTES:BEGIN -->
- `git diff --check`
- `.venv/bin/python -m compileall -q synthesia2midi`
- `.venv/bin/python -m pytest`
<!-- SECTION:NOTES:END -->
