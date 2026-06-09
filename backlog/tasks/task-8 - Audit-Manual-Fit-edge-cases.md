---
id: TASK-8
title: Audit Manual Fit edge cases
status: Done
assignee: []
created_date: '2026-06-09 00:00'
labels:
  - review
  - ui
  - calibration
  - manual-fit
dependencies: []
modified_files:
  - synthesia2midi/synthesia2midi/workflows/manual_keyboard_fit.py
  - synthesia2midi/synthesia2midi/gui/manual_keyboard_fit_controller.py
  - tests
priority: high
ordinal: 8000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Audit the current Manual Fit work-session commits and fix confirmed edge cases around user backtracking, reset, cancel, and keyboard-box edit instructions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A gitignored edge-case report is written.
- [x] #2 Edit Keyboard Box canvas instructions mention green boundary bars.
- [x] #3 Cancel restores the previous manual keyboard box.
- [x] #4 Reset All restores the current session-baseline manual keyboard box.
- [x] #5 Full verification gate passes.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Wrote ignored report at `logs/manual-fit-edge-case-analysis-2026-06-08.md`.
- Fixed stale canvas instruction for Edit Keyboard Box.
- Fixed Manual Fit keyboard-box restoration for Cancel and Reset All.
- Verification: `git diff --check`, `.venv/bin/python -m compileall -q synthesia2midi`, `.venv/bin/python -m pytest`.
<!-- SECTION:NOTES:END -->
