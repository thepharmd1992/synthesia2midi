---
id: TASK-9
title: Package user release and harden YouTube download
status: To Do
assignee: []
created_date: '2026-06-08 21:20'
labels:
  - release
  - packaging
  - youtube
dependencies: []
documentation:
  - docs/superpowers/specs/2026-06-08-packaged-release-youtube-hardening-design.md
modified_files:
  - .github/workflows
  - backlog/tasks
  - docs
  - run.py
  - setup_env.py
  - synthesia2midi/run.py
  - synthesia2midi/synthesia2midi
  - tests
  - tools/midi_touchup_editor_rust
priority: high
ordinal: 1009
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Package Synthesia2MIDI as a portable end-user app for Windows x64 and macOS Apple Silicon, bundle required helper binaries, and harden the YouTube downloader with JS runtime support, browser-cookie retry, and packaged-runtime path resolution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Portable packaged builds exist for Windows x64 and macOS Apple Silicon from version tags
- [ ] #2 Packaged runtime resolves bundled FFmpeg, ffprobe, Deno, Rust touch-up editor, and SoundFont without repo-root assumptions
- [ ] #3 YouTube download supports bundled JS runtime, remembered preferred browser cookies, and auto-retry for known auth/challenge failures
- [ ] #4 Manual local video loading remains the stable fallback and packaged failure messaging is explicit
- [ ] #5 CI builds and publishes zipped GitHub Release assets from tags
- [ ] #6 Tests cover runtime path resolution and YouTube retry policy
<!-- AC:END -->
