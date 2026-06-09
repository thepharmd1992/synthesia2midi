---
id: TASK-7
title: Add keyboard box edge grab arms
status: Done
assignee: []
created_date: '2026-06-09 00:00'
labels:
  - ui
  - calibration
  - manual-fit
dependencies: []
modified_files:
  - synthesia2midi/synthesia2midi/gui/canvas/interaction.py
  - synthesia2midi/synthesia2midi/gui/keyboard_canvas.py
  - tests
priority: high
ordinal: 7000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add inward horizontal grab arms to Manual Fit keyboard-box side handles so users can recover and drag a side boundary that has moved partly off screen.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Keyboard-box side handles draw inward horizontal arms from the top of each green boundary bar.
- [x] #2 Each arm has a visible end cap that reads as a grab target.
- [x] #3 Clicking and dragging an inward arm moves the same keyboard-box side boundary.
- [x] #4 Dragging an arm preserves the grab offset instead of snapping the boundary to the cursor.
- [x] #5 Existing vertical side-handle dragging still works.
<!-- AC:END -->

## Verification

<!-- SECTION:NOTES:BEGIN -->
- `git diff --check`
- `.venv/bin/python -m compileall -q synthesia2midi`
- `.venv/bin/python -m pytest`
<!-- SECTION:NOTES:END -->
