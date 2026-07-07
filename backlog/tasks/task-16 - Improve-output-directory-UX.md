---
id: TASK-16
title: Improve output directory UX
status: Done
assignee: []
created_date: '2026-07-07 00:00'
labels:
  - ui
  - files
  - workflow
dependencies: []
documentation:
  - docs/superpowers/plans/2026-07-07-output-directory-ux.md
modified_files:
  - docs/localization
  - docs/superpowers/plans
  - synthesia2midi/synthesia2midi
  - tests
priority: high
ordinal: 16000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Move user-facing output and technical working files into clearer default locations. Final MIDI files should be easy to find, downloaded source videos should use Downloads, and frame/config/project data should live under app-managed storage while preserving compatibility with existing sidecar files.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Final MIDI files default to `Desktop/Synthesia2MIDI MIDI Files/`.
- [x] #2 YouTube downloads default to `Downloads/Synthesia2MIDI/(per-video slug)/`.
- [x] #3 Frame series, config, overlays, and conversion settings are written under app-managed project data.
- [x] #4 Old sidecar config, overlays, and frame folders still load.
- [x] #5 The conversion-complete dialog offers `Open Touch-Up Editor` and `Show MIDI in Folder`.
- [x] #6 Tests cover the new path helpers, legacy fallbacks, and dialog behavior.
- [x] #7 Localization/audit artifacts are updated for new visible strings.
<!-- AC:END -->

## Verification

<!-- SECTION:NOTES:BEGIN -->
- Implementation plan: `docs/superpowers/plans/2026-07-07-output-directory-ux.md`
- Expected gates include `git diff --check`, `.venv/bin/python -m compileall -q synthesia2midi`, focused path/controller tests, localization/audit tests, and full pytest.
- Verification completed on 2026-07-07:
  - `git diff --check`
  - `.venv/bin/python -m compileall -q synthesia2midi`
  - `.venv/bin/python -m pytest` (`300 passed`)
  - `.venv/bin/python -m pytest tests/test_runtime_paths.py tests/test_config_manager.py tests/test_midi_conversion_controller.py tests/test_midi_touchup_controller.py tests/test_packaged_entrypoint.py` (`29 passed`)
<!-- SECTION:NOTES:END -->
