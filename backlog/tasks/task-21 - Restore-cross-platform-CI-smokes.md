---
id: TASK-21
title: Restore cross-platform CI smokes
status: In Progress
assignee: []
created_date: '2026-07-11 00:00'
updated_date: '2026-07-11 00:00'
labels:
  - ci
  - ux
  - cross-platform
dependencies: []
priority: high
ordinal: 21000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the full Python and Rust smoke matrices pass on GitHub's Windows, macOS, and Linux runners before merging the completed UX and assisted-calibration work to main.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Python package and developer-tool imports resolve consistently in CI.
- [x] #2 The UI string manifest is byte-stable across Windows, macOS, and Linux path conventions.
- [x] #3 Settings footer and Manual Fit numeric controls accommodate shipped locales and font scales on every runner.
- [ ] #4 The pseudo-locale UI matrix reports no unexplained clipping on any runner.
- [ ] #5 Python and Rust jobs pass on Windows, macOS, and Linux for the pull request.
- [ ] #6 The verified pull request is merged to main and the resulting main-branch CI run passes.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The first branch run reproduced the old package-shadowing failure and exposed latent platform differences. The import bootstrap now prioritizes the real source package, the assisted-calibration probe has a repo-root command wrapper, manifest source paths use POSIX separators, Manual Fit spinboxes size from their current font and value range, and footer content uses one column so translated controls do not create competing minimum widths.

Local CI-equivalent verification passes under `PYTHONPATH=synthesia2midi` and `QT_QPA_PLATFORM=offscreen`. The remaining acceptance criteria require fresh GitHub runner evidence.
<!-- SECTION:NOTES:END -->
