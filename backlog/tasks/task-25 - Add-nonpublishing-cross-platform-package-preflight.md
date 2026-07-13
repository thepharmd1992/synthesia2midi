---
id: TASK-25
title: Add nonpublishing cross-platform package preflight
status: Done
assignee: []
created_date: '2026-07-12 00:00'
updated_date: '2026-07-12 00:00'
labels:
  - ci
  - packaging
  - release
dependencies:
  - TASK-24
priority: high
ordinal: 25000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Allow Windows x64 and Apple Silicon packages to be built and inspected before updating remote `main` or creating a version tag, without publishing a GitHub release.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Pushing a `codex/*-preflight` branch runs the existing Windows and Apple Silicon package recipe.
- [x] #2 Manual package preflight is available after the workflow reaches the default branch.
- [x] #3 Non-tag runs upload short-lived workflow artifacts without creating or modifying a GitHub release.
- [x] #4 Version tags retain the existing release creation, stable archive alias, and release upload behavior.
- [x] #5 Python and Rust CI pass on Windows, macOS, and Linux, and both package preflight jobs pass before local integration.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Use the existing `Release` workflow as the single package-build recipe. Preflight branches and manual dispatches must never execute tag-only publishing steps.

Verification completed on 2026-07-12:

- Local `git diff --check`, workflow YAML parsing, compileall, Ruff's syntax/name gate, all `561` Python tests, Rust formatting, all `21` Rust tests, and `cargo check` passed.
- The `qps` 150% UI matrix rendered all `25` surfaces without clipping, including the wider Windows-font stress test.
- GitHub CI run `29219523378` passed Python and Rust jobs on Windows, macOS, and Linux. The preflight exposed a Windows-only Settings-width assumption before integration; the final fix allows the floating Settings window to widen for translated labels without increasing footer height.
- GitHub Release run `29219523383` passed Windows x64 and Apple Silicon package builds and uploaded both archives as seven-day workflow artifacts.
- `create-release`, stable `latest` aliases, and GitHub release uploads were skipped. The public release list remained unchanged with `v0.2.0` as latest.
<!-- SECTION:NOTES:END -->
