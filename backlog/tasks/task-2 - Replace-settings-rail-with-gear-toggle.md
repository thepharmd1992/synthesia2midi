---
id: TASK-2
title: Replace settings rail with gear toggle
status: Done
assignee: []
created_date: '2026-06-09 00:06'
labels:
  - ui
  - settings
dependencies: []
modified_files:
  - synthesia2midi/synthesia2midi/main.py
  - tests/test_main_window_layout.py
priority: medium
ordinal: 2000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove the persistent settings rail from the video workspace and use a compact gear toggle for the floating settings window.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Main window uses a small gear settings toggle instead of a wide vertical settings rail
- [x] #2 Gear tooltip/state changes between show and hide settings
- [x] #3 Settings tool window still opens upper-right and preserves user-positioned geometry
- [x] #4 Focus-video action hides/restores the settings toggle and settings window consistently
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed the persistent main-window settings rail and replaced it with a compact gear toggle above the video area. The toggle now opens, hides, and tracks the floating settings window while preserving user-positioned settings geometry.
<!-- SECTION:FINAL_SUMMARY:END -->
