---
id: TASK-26
title: Recover v0.2.2 release and reported workflows
status: In Progress
assignee: []
created_date: '2026-08-31 00:00'
updated_date: '2026-08-31 02:20'
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
- [x] #2 The package builder rejects missing, system-only, shimmed, or non-runnable helpers before creating an archive.
- [x] #3 Reviewed PyInstaller inputs and the Windows FFmpeg package version are pinned.
- [x] #4 Accepting manual or automatic alignment review advances the Guide; canceling does not; assisted-scan restoration preserves acceptance.
- [x] #5 ASCII, accented, fullwidth-punctuation, CJK, and emoji video names produce safe MIDI track metadata without conversion failure.
- [ ] #6 Focused tests, the full Python gate, Rust gates, and both remote package preflight jobs pass.
- [ ] #7 v0.2.2 is publicly available with verified Windows x64 and macOS arm64 assets before GitHub issue #9 is closed.
- [ ] #8 TASK-9 is reconciled against v0.2.2 evidence and issue #6 remains unchanged.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Active Spec Kit feature: `specs/001-release-recovery/`.

Preserve the unrelated untracked `uv.lock`.

Local candidate evidence (2026-08-31):

- Full Python gate: 602 passed with 29 existing Qt deprecation warnings; compileall and `git diff --check` passed.
- Rust editor: format check passed, 21 tests passed, and `cargo check` passed with the existing unused `AudioTelemetry::meter` warning.
- Local macOS v0.2.2-dev package self-check passed all six package-owned helper/asset checks. FFmpeg, ffprobe, Deno, and the Rust editor were native arm64 Mach-O executables; the GUI smoke and zip integrity check passed.
- Local archive: 559 MB, SHA-256 `97531cb0b176c06f083d560b61a77322bfa3c151edb2bf8169a679537fa5311d`.
- The first local package inspection exposed a 353-byte Python FFmpeg launcher tied to the build machine. The builder and packaged self-check now reject non-native launchers; rebuilding with the underlying 49 MB native FFmpeg/ffprobe executables passed.
- Exact issue #9 URL `https://www.youtube.com/watch?v=B33CSGTwwmQ`: initial `HTTP Error 403: Forbidden`, emitted the alternate-client retry status, then downloaded a 19,377,627-byte 480p MP4 successfully.

Remote preflight, public release, and TASK-9 reconciliation remain pending.
<!-- SECTION:NOTES:END -->
