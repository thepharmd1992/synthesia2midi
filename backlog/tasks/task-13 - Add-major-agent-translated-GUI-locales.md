---
id: TASK-13
title: Add major agent-translated GUI locales
status: In Progress
assignee: []
created_date: '2026-07-07 00:00'
labels:
  - localization
  - gui
  - packaging
dependencies:
  - TASK-11
  - TASK-12
documentation:
  - docs/localization/translation-agent-instructions.md
modified_files:
  - backlog/tasks
  - docs/localization
  - docs/testing.md
  - synthesia2midi/synthesia2midi
  - tests
priority: high
ordinal: 1013
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add first-pass agent-translated GUI locales for Japanese, Russian, Simplified Chinese, Korean, and Brazilian Portuguese using the existing Qt localization pipeline.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A reusable translation-agent packet and instructions exist for serial locale delegation
- [ ] #2 Japanese, Russian, Simplified Chinese, Korean, and Brazilian Portuguese `.ts` and `.qm` assets are tracked
- [ ] #3 The language selector exposes all production locales and still hides the pseudo-locale
- [ ] #4 Locale validation tests cover every tracked production locale for loadability, completeness, and placeholder preservation
- [ ] #5 Localization docs describe packet export and compiling every production `.ts` file
<!-- AC:END -->
