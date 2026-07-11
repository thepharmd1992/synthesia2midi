---
id: TASK-19
title: Implement structural UX audit phases A-D
status: In Progress
assignee: []
created_date: '2026-07-11 22:01'
labels:
  - ui
  - ux
  - accessibility
dependencies:
  - TASK-18
references:
  - logs/ux-audit/2026-07-11-structural-ui-audit/structural-ui-ux-audit.md
priority: high
ordinal: 19000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement every required Phase A-D change from the internal 2026-07-11 structural UI/UX audit. Preserve saved project compatibility and detector behavior while correcting interaction safety, window/scroll architecture, responsive workflow structure, and deterministic regression coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Auto-Detect tuning is transactional: Save commits, while Cancel, Escape, and window-close restore parameters, overlays, cached context, and prior dirty state without saving.
- [x] #2 Sliders, spinboxes, and closed combo boxes never change from wheel/trackpad scrolling; wheel input continues to the owning scroll viewport.
- [x] #3 Conversion shortcuts obey the same readiness state as Convert, normal canvas mode cannot accidentally move overlays, selection modes expose persistent instructions/cancel/Escape/retry feedback, trim extrema fit, and Auto-Detect Return activates Save.
- [x] #4 Settings uses fixed navigation and fixed global actions with one page-owned scroll viewport; short pages do not scroll and no active nested scroll areas remain.
- [x] #5 Repeated Notes has a dedicated tool, Auto-Detect Expert shows one category at a time, and floating windows use the parent window screen.
- [x] #6 Guide completion/current/future states compress appropriately, Manual Fit reflows and contracts by mode, YouTube fallback is progressive, file/menu hierarchy is simplified, and the Rust editor responsive/status issues are addressed.
- [ ] #7 Real-window, scroll-ownership, wheel-propagation, numeric-extrema, transactional-dialog, keyboard/default, populated-metadata, and large-text window-bound regression gates pass.
- [ ] #8 All changed GUI strings are audited and translated in every production locale; compileall, complete pytest, localization, visual matrix, git diff, and relevant Rust/package gates pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Phase A: interaction correctness. Phase B: scrolling/window architecture. Phase C: workflow compression/responsiveness. Phase D: deterministic regression gates. Use test-first slices and checkpoint commits; no worktrees and no push.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Authoritative internal audit: logs/ux-audit/2026-07-11-structural-ui-audit/structural-ui-ux-audit.md (ignored by design).

Phase A completed with focused regression coverage for transactional tuning, wheel routing, keyboard defaults, canvas selection/edit boundaries, cancellation lifecycle, and numeric-extrema sizing.

Phase B completed with a fixed Settings rail/footer, page-owned scrolling, a dedicated Repeated Notes tool, master-detail Auto-Detect Expert controls, and parent-screen placement.

Phase C completed with a compact state-aware Guide, responsive Manual Fit modes, progressive YouTube recovery controls, separate file/folder pickers, nested diagnostics, bounded touch-up failures, recent-name elision, and Rust editor input/layout/status hardening.
<!-- SECTION:NOTES:END -->
