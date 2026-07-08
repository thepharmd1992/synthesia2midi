---
id: TASK-17
title: Phase 1 UX hardening
status: To Do
assignee: []
created_date: '2026-07-08 00:00'
labels:
  - ui
  - ux
  - calibration
dependencies:
  - TASK-15
  - TASK-16
documentation:
  - docs/superpowers/specs/2026-07-08-phase-1-ux-hardening-design.md
modified_files: []
priority: high
ordinal: 17000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the first UX hardening slice from the internal settings/calibration audit. The goal is to make the normal video-to-MIDI path clearer without changing detector behavior, saved calibration/config formats, or the overall settings architecture.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Convert area explains the first missing prerequisite instead of showing ready text while disabled.
- [ ] #2 Calibration shows visible instructions for keyboard-box selection, no-key frame capture, and pressed-key examples without requiring Help expansion.
- [ ] #3 Left/Right pressed-key terminology remains visible and is clarified as Synthesia note color/family language rather than physical keyboard position.
- [ ] #4 Overlay quick adjustments show current values and reset controls, including Left Slant and Right Slant.
- [ ] #5 Calibration Wizard and Auto-Detect Tuning use plain-language instructions while preserving existing detection/tuning behavior.
- [ ] #6 Detection, Spark, MIDI range, Trim, Optional, and YouTube fallback settings are reframed with clearer user-facing copy.
- [ ] #7 Destructive Trim is clearly separated from non-destructive MIDI processing range.
- [ ] #8 Existing saved configs, overlay sidecars, detection parameters, and conversion behavior remain compatible.
- [ ] #9 Tests cover the changed visible UI behavior and localization/audit gates are updated for changed strings.
<!-- AC:END -->

## Verification

<!-- SECTION:NOTES:BEGIN -->
- Design spec: `docs/superpowers/specs/2026-07-08-phase-1-ux-hardening-design.md`
- Expected focused gates include:
  - `git diff --check`
  - `.venv/bin/python -m compileall -q synthesia2midi`
  - `.venv/bin/python -m pytest tests/test_controls_qt.py tests/test_startup_dialog.py tests/test_youtube_download_dialog.py tests/test_auto_detect_tuning_dialog.py tests/test_ui_string_audit.py tests/test_localization.py`
  - `.venv/bin/python -m pytest`
- Implementation plan will be created after spec review.
<!-- SECTION:NOTES:END -->
