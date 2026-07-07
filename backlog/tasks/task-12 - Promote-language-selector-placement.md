---
id: TASK-12
title: Promote language selector placement
status: Done
assignee: []
created_date: '2026-07-06 00:00'
labels:
  - localization
  - gui
dependencies:
  - TASK-11
documentation: []
modified_files:
  - backlog/tasks
  - docs/localization
  - synthesia2midi/synthesia2midi
  - tests
priority: high
ordinal: 1012
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Move language selection out of Optional settings and make it prominent in the startup welcome dialog and as its own top-level settings section.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Startup welcome dialog shows the language selector directly under the welcome title
- [x] #2 Settings pane has a first-class Language section in the section rail
- [x] #3 Optional settings no longer owns the language selector
- [x] #4 Spanish translation assets and UI string manifest are refreshed for the new placement
- [x] #5 Tests cover startup placement, settings section placement, and saved preference behavior
<!-- AC:END -->
