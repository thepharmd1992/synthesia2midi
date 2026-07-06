---
id: TASK-10
title: Add comprehensive GUI localization audit
status: Done
assignee: []
created_date: '2026-07-06 00:00'
labels:
  - localization
  - gui
  - packaging
dependencies: []
documentation: []
modified_files:
  - backlog/tasks
  - docs/localization
  - packaging/Synthesia2MIDI.spec
  - synthesia2midi/synthesia2midi
  - tests
priority: high
ordinal: 1010
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add deterministic localization infrastructure and audit coverage for app-visible Qt UI strings and packaged-app dialogs. Scope excludes backend internals, logs, setup scripts, release scripts, config keys, paths, URLs, and developer console output.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Qt translation loading exists with English default and pseudo-locale test support
- [x] #2 Static and runtime audit tooling produce a stable reviewable manifest of app-visible string candidates
- [x] #3 GUI and packaged-app-visible strings are converted to Qt translation calls where classified as translatable
- [x] #4 Translation assets are included in packaged builds
- [x] #5 Tests cover audit extraction, runtime widget crawling, pseudo-locale behavior, and lupdate extraction
<!-- AC:END -->
