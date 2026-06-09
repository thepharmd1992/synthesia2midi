---
id: TASK-5
title: Add Manual Fit keyboard box edge handles
status: Done
assignee: []
created_date: '2026-06-09 00:00'
labels:
  - ui
  - calibration
  - manual-fit
dependencies: []
modified_files:
  - synthesia2midi/synthesia2midi/workflows/manual_keyboard_fit.py
  - synthesia2midi/synthesia2midi/gui/manual_keyboard_fit_controller.py
  - synthesia2midi/synthesia2midi/gui/canvas/interaction.py
  - synthesia2midi/synthesia2midi/gui/keyboard_canvas.py
  - tests
priority: high
ordinal: 5000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make Manual Fit keyboard-box editing clearer by adding visible side-edge handles and allowing users to drag one side boundary at a time while preserving whole-box redraw as a fallback.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Edit Keyboard Box enters side-edge edit mode.
- [x] #2 Dragging a side handle changes only that side of the keyboard box.
- [x] #3 The opposite side and vertical bounds are preserved during side-edge edits.
- [x] #4 Dragging away from side handles still supports whole-box redraw.
- [x] #5 Keyboard-box side handles draw as thick bright green vertical lines that protrude above the box.
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
- Edit Keyboard Box now enters a side-edge mode with thick bright green vertical handles.
- Dragging a side handle changes only that side; dragging elsewhere can still redraw the whole box.
- Verification: `git diff --check`, `.venv/bin/python -m compileall -q synthesia2midi`, `.venv/bin/python -m pytest`.
<!-- SECTION:NOTES:END -->
