---
id: TASK-9
title: Package user release and harden YouTube download
status: Done
assignee: []
created_date: '2026-06-08 21:20'
updated_date: '2026-08-31 02:45'
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
- [x] #1 Portable packaged builds exist for Windows x64 and macOS Apple Silicon from version tags
- [x] #2 Packaged runtime resolves bundled FFmpeg, ffprobe, Deno, Rust touch-up editor, and SoundFont without repo-root assumptions
- [x] #3 YouTube download supports bundled JS runtime, remembered preferred browser cookies, and auto-retry for known auth/challenge failures
- [x] #4 Manual local video loading remains the stable fallback and packaged failure messaging is explicit
- [x] #5 CI builds and publishes zipped GitHub Release assets from tags
- [x] #6 Tests cover runtime path resolution and YouTube retry policy
<!-- AC:END -->

## Completion Evidence

- The public [v0.2.2 release](https://github.com/thepharmd1992/synthesia2midi/releases/tag/v0.2.2) contains versioned and stable `latest` zip assets for Windows x64 and macOS arm64.
- Both tag-built packaged apps passed the required package-owned helper/asset self-check before publication. A downloaded public macOS asset repeated all six checks successfully on a separate machine.
- The public Windows archive contains real 102 MB FFmpeg/ffprobe PE executables under `_internal/bin`, not Chocolatey shims; the runtime resolver uses PyInstaller's bundle root rather than repo paths.
- The exact reported YouTube URL reproduced an initial 403, invoked the alternate-client retry, and completed a 19,377,627-byte 480p download. Existing JS runtime, remembered browser-cookie retry, explicit error messaging, and local-file loading paths remain covered by the test suite.
- The final local gate passed 602 Python tests and 21 Rust tests. The cross-platform preflight passed Python and Rust jobs on Windows, macOS, and Ubuntu plus both packaged-release jobs.
