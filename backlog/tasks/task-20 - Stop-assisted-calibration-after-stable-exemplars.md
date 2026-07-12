---
id: TASK-20
title: Stop assisted calibration after stable complete exemplars
status: Done
assignee: []
created_date: '2026-07-11 00:00'
updated_date: '2026-07-11 00:00'
labels:
  - calibration
  - detection
  - performance
dependencies:
  - TASK-15
priority: high
ordinal: 20000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
End an assisted lit-exemplar scan early once two distinct Synthesia color families each have confident white-key and black-key evidence. Keep scanning one-family and incomplete-family videos through the configured end frame, and require repeated temporally separated evidence plus a confirmation tail so a transient animation cannot trigger early completion.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A scan stops before the end frame after two distinct color families each have confirmed white-key and black-key exemplars.
- [x] #2 Early completion requires repeated, temporally separated evidence and a short confirmation tail.
- [x] #3 A transient animation that produces simultaneous color candidates does not trigger early completion.
- [x] #4 One-family and incomplete-family videos retain full-range scanning behavior.
- [x] #5 Cancellation, candidate assignment, and assisted-calibration proposal behavior remain compatible.
- [x] #6 Focused assisted-calibration tests and the full Python verification gate pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implement this as a conservative stopping condition only. Sequential video-read optimization and probabilistic timeline sampling remain separate follow-up work.

The completed stop rule requires two confident, temporally separated completed bursts for every white/black slot in both hue-clustered families. A six-checkpoint confirmation period must then receive a fresh completed burst for every slot. Family evidence is matched by circular hue distance so rolling recent history cannot swap family identities. Final exemplar ranking and bounded recent stop evidence are maintained separately.

The Game of Thrones reference probe (`baseline=430`, `end=2500`, `stride=10`) stopped after 80 coarse checkpoints with the last requested refinement frame at 1226. It found both complete families with `LW=(134,168,205)`, `LB=(74,114,174)`, `RW=(253,176,71)`, and `RB=(255,132,54)`.

Final verification: Ruff passed; `git diff --check` passed; compileall passed; 409 Python tests passed with 29 pre-existing Qt deprecation warnings. Independent read-only review reproduced and verified fixes for transient pulses, saturated output buckets, rolling family order, and repeated unchanged-evidence clustering; no concrete findings remained.
<!-- SECTION:NOTES:END -->
