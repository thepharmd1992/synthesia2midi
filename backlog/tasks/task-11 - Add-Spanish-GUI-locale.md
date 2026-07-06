---
id: TASK-11
title: Add Spanish GUI locale
status: Done
assignee: []
created_date: '2026-07-06 00:00'
labels:
  - localization
  - gui
  - packaging
dependencies:
  - TASK-10
documentation: []
modified_files:
  - backlog/tasks
  - docs/localization
  - synthesia2midi/synthesia2midi
  - tests
priority: high
ordinal: 1011
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add Spanish as the first real GUI locale using the existing Qt localization audit infrastructure. Keep English as the source/default language, use neutral Latin American Spanish, and make language changes apply after restart.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Spanish `.ts` and compiled `.qm` translation assets exist and are loaded by the app
- [x] #2 All current extracted GUI source strings have finished Spanish translations with placeholders preserved
- [x] #3 App startup resolves locale from environment override first, then saved user preference, then English fallback
- [x] #4 Optional settings expose an English/Español language selector that hides the pseudo-locale
- [x] #5 Tests cover Spanish loading, translation asset integrity, preference persistence, and selector behavior
<!-- AC:END -->
