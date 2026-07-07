---
id: TASK-16
title: Improve output directory UX
status: To Do
assignee: []
created_date: '2026-07-07 00:00'
labels:
  - ui
  - files
  - workflow
dependencies: []
documentation:
  - docs/superpowers/plans/2026-07-07-output-directory-ux.md
modified_files: []
priority: high
ordinal: 16000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Move user-facing output and technical working files into clearer default locations. Final MIDI files should be easy to find, downloaded source videos should use Downloads, and frame/config/project data should live under app-managed storage while preserving compatibility with existing sidecar files.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Final MIDI files default to `Desktop/Synthesia2MIDI MIDI Files/`.
- [ ] #2 YouTube downloads default to `Downloads/Synthesia2MIDI/(per-video slug)/`.
- [ ] #3 Frame series, config, overlays, and conversion settings are written under app-managed project data.
- [ ] #4 Old sidecar config, overlays, and frame folders still load.
- [ ] #5 The conversion-complete dialog offers `Open Touch-Up Editor` and `Show MIDI in Folder`.
- [ ] #6 Tests cover the new path helpers, legacy fallbacks, and dialog behavior.
- [ ] #7 Localization/audit artifacts are updated for new visible strings.
<!-- AC:END -->

## Verification

<!-- SECTION:NOTES:BEGIN -->
- Implementation plan: `docs/superpowers/plans/2026-07-07-output-directory-ux.md`
- Expected gates include `git diff --check`, `.venv/bin/python -m compileall -q synthesia2midi`, focused path/controller tests, localization/audit tests, and full pytest.
<!-- SECTION:NOTES:END -->
