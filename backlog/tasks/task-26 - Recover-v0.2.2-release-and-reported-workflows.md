---
id: TASK-26
title: Recover v0.2.2 release and reported workflows
status: In Progress
assignee: []
created_date: '2026-08-31 00:00'
updated_date: '2026-08-31 00:00'
labels:
  - release
  - packaging
  - guide
  - midi
dependencies:
  - TASK-25
documentation:
  - specs/001-release-recovery/spec.md
priority: high
ordinal: 26000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Repair the packaged Windows helper failures, publish a verified v0.2.2 containing the existing YouTube 403 fallback, make accepted alignment review advance the Guide, make Unicode filenames safe in MIDI metadata, and reconcile TASK-9 from release evidence. GitHub issue #6 is explicitly excluded because its range and destructive-trim capabilities already exist.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Frozen Windows and macOS apps resolve every bundled helper and SoundFont through package-owned paths.
- [ ] #2 The package builder rejects missing, system-only, shimmed, or non-runnable helpers before creating an archive.
- [ ] #3 Reviewed PyInstaller inputs and the Windows FFmpeg package version are pinned.
- [ ] #4 Accepting manual or automatic alignment review advances the Guide; canceling does not; assisted-scan restoration preserves acceptance.
- [ ] #5 ASCII, accented, fullwidth-punctuation, CJK, and emoji video names produce safe MIDI track metadata without conversion failure.
- [ ] #6 Focused tests, the full Python gate, Rust gates, and both remote package preflight jobs pass.
- [ ] #7 v0.2.2 is publicly available with verified Windows x64 and macOS arm64 assets before GitHub issue #9 is closed.
- [ ] #8 TASK-9 is reconciled against v0.2.2 evidence and issue #6 remains unchanged.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Active Spec Kit feature: `specs/001-release-recovery/`.

Preserve the unrelated untracked `uv.lock`. Release evidence and final verification counts will be recorded here after publication.
<!-- SECTION:NOTES:END -->
