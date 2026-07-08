---
id: TASK-17
title: Phase 1 UX hardening
status: Done
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
  - docs/superpowers/plans/2026-07-08-phase-1-ux-hardening.md
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
- [x] #1 The Convert area explains the first missing prerequisite instead of showing ready text while disabled.
- [x] #2 Calibration shows visible instructions for keyboard-box selection, no-key frame capture, and pressed-key examples without requiring Help expansion.
- [x] #3 Left/Right pressed-key terminology remains visible and is clarified as Synthesia note color/family language rather than physical keyboard position.
- [x] #4 Overlay quick adjustments show current values and reset controls, including Left Slant and Right Slant.
- [x] #5 Calibration Wizard and Auto-Detect Tuning use plain-language instructions while preserving existing detection/tuning behavior.
- [x] #6 Detection, Spark, MIDI range, Trim, Optional, and YouTube fallback settings are reframed with clearer user-facing copy.
- [x] #7 Destructive Trim is clearly separated from non-destructive MIDI processing range.
- [x] #8 Existing saved configs, overlay sidecars, detection parameters, and conversion behavior remain compatible.
- [x] #9 Tests cover the changed visible UI behavior and localization/audit gates are updated for changed strings.
<!-- AC:END -->

## Verification

<!-- SECTION:NOTES:BEGIN -->
- Design spec: `docs/superpowers/specs/2026-07-08-phase-1-ux-hardening-design.md`
- Implementation plan: `docs/superpowers/plans/2026-07-08-phase-1-ux-hardening.md`
- Expected focused gates include:
  - `git diff --check`
  - `.venv/bin/python -m compileall -q synthesia2midi`
  - `.venv/bin/python -m pytest tests/test_controls_qt.py tests/test_startup_dialog.py tests/test_youtube_download_dialog.py tests/test_auto_detect_tuning_dialog.py tests/test_ui_string_audit.py tests/test_localization.py`
  - `.venv/bin/python -m pytest`
- Implementation plan will be created after spec review.
- Task 7 actual verification on `2026-07-08`:
  - `.venv/bin/python -m synthesia2midi.tools.audit_ui_strings --output docs/localization/ui-string-manifest.json`
    - Passed (`Wrote 569 UI string candidates`)
  - `.venv/bin/pyside6-lupdate -extensions py synthesia2midi/synthesia2midi -ts synthesia2midi/synthesia2midi/translations/synthesia2midi_es.ts synthesia2midi/synthesia2midi/translations/synthesia2midi_ja.ts synthesia2midi/synthesia2midi/translations/synthesia2midi_ru.ts synthesia2midi/synthesia2midi/translations/synthesia2midi_zh_CN.ts synthesia2midi/synthesia2midi/translations/synthesia2midi_ko.ts synthesia2midi/synthesia2midi/translations/synthesia2midi_pt_BR.ts`
    - Passed (`0 new`, `548 already existing` for each production catalog)
  - `.venv/bin/python -m synthesia2midi.tools.export_translation_packet --source-ts synthesia2midi/synthesia2midi/translations/synthesia2midi_es.ts --output docs/localization/translation-agent-packet.json`
    - Passed (`Wrote 548 translation entries`)
  - `for ts_file in synthesia2midi/synthesia2midi/translations/synthesia2midi_*.ts; do locale_name=$(basename "$ts_file" .ts | sed 's/^synthesia2midi_//'); .venv/bin/pyside6-lrelease "$ts_file" -qm "synthesia2midi/synthesia2midi/translations/synthesia2midi_${locale_name}.qm"; done`
    - Passed (`548 finished`, `0 unfinished` for `es`, `ja`, `ko`, `pt_BR`, `ru`, `zh_CN`)
  - `.venv/bin/python -m pytest tests/test_localization.py tests/test_ui_string_audit.py -v`
    - Passed (`24 passed`)
  - `.venv/bin/python -m pytest tests/test_controls_qt.py tests/test_main_window_layout.py tests/test_calibration_wizard_copy.py tests/test_assisted_calibration_copy.py tests/test_auto_detect_tuning_dialog.py tests/test_youtube_download_dialog.py tests/test_startup_dialog.py -v`
    - Passed (`53 passed`)
  - `git diff --check`
    - Passed
  - `.venv/bin/python -m compileall -q synthesia2midi`
    - Passed
  - `.venv/bin/python -m pytest`
    - Passed (`321 passed`; warnings only)
<!-- SECTION:NOTES:END -->
