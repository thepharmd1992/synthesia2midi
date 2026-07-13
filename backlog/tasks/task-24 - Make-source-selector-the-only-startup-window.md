---
id: TASK-24
title: Make source selector the only startup window
status: In Progress
assignee: []
created_date: '2026-07-12 00:00'
updated_date: '2026-07-12 00:00'
labels:
  - ux
  - startup
  - qt
dependencies:
  - TASK-23
priority: high
ordinal: 24000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the main workspace hidden during startup, return secondary-dialog cancellations to Select Video Source, and exit when the source selector itself is cancelled.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Select Video Source is the only visible Synthesia2MIDI window at initial startup.
- [ ] #2 The main workspace appears only after a local, recent, or downloaded video loads successfully.
- [ ] #3 Cancelling the local file picker or YouTube dialog leaves Select Video Source open and the main workspace hidden.
- [ ] #4 Cancelling or closing Select Video Source exits Synthesia2MIDI completely.
- [ ] #5 Existing in-session File menu and empty-state video actions remain available and behaviorally compatible.
- [ ] #6 Source, launcher, UI-matrix, and packaged-startup verification pass before local integration.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Approved design: `docs/superpowers/specs/2026-07-12-startup-source-flow-design.md`.

Do not show the main window temporarily and then hide it. The launch paths must leave it hidden until the startup coordinator confirms a loaded video session.
<!-- SECTION:NOTES:END -->
