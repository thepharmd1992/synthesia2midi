---
id: TASK-18
title: Complete UX audit phases 2-4
status: Done
assignee: []
created_date: '2026-07-11 00:00'
labels:
  - ui
  - ux
  - accessibility
  - localization
dependencies:
  - TASK-17
documentation:
  - docs/superpowers/specs/2026-07-11-ux-phases-2-4-design.md
priority: high
ordinal: 18000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Complete the guided calibration, advanced-settings reorganization, and accessibility/localization phases from the internal UX audit while preserving detector behavior and saved project compatibility.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 A first-position Guide shows video, overlays, no-key, pressed-color, and conversion steps with status and next actions.
- [x] #2 The no-video main window and startup dialog clearly explain how to begin with a suitable Synthesia-style video.
- [x] #3 Assisted calibration uses a swatch-based dialog; accepting applies the proposal while retry, cancel, and keep-current paths preserve prior calibration.
- [x] #4 Manual Fit shows a visible explanation for every mode, and conversion completion defaults to showing the MIDI file in its folder.
- [x] #5 Detection specialist modes, repeated-note controls, and permanent Trim are collapsed under a symptom-led Advanced page by default.
- [x] #6 Auto-Detect advanced controls are explicitly expert-only and collapsed by default, and a lightweight user glossary explains necessary terms.
- [x] #7 Existing saved configs, detector values, signals, overlay sidecars, and conversion behavior remain compatible after controls move.
- [x] #8 Small interactive controls meet the target-size requirements, essential instructions are not tooltip-only, status colors are readable, and native warning styling is used.
- [x] #9 Startup, Guide/settings, Calibration Wizard, and YouTube Download have tested logical keyboard focus order and useful accessible names.
- [x] #10 Settings rail labels fit every shipped locale and qps at default, 125%, and 150% fonts.
- [x] #11 A deterministic qps 150% offscreen screenshot/clipping matrix covers the required core UI surfaces.
- [x] #12 All changed UI strings are audited, translated in every production locale, compiled to qm assets, and pass localization integrity tests.
- [x] #13 Focused tests, the complete pytest suite, compileall, and git diff checks pass.
<!-- AC:END -->

## Notes

<!-- SECTION:NOTES:BEGIN -->
- Internal audit: `logs/ux-audit/2026-07-07-settings-ux-audit/ux-audit-report.md` (intentionally ignored).
- Design: `docs/superpowers/specs/2026-07-11-ux-phases-2-4-design.md`.
- No worktree and no push.
- Keep Left/Right as user-facing Synthesia color-family terminology.
- Final verification: 373 tests passed; qps 150% matrix covered 21 opening states with zero clipping, overlap, or blank findings; all six production catalogs compiled 611 finished translations with zero unfinished.
<!-- SECTION:NOTES:END -->
