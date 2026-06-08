# Synthesia2MIDI Constitution

## Core Principles

### I. Preserve The Working App

Changes must preserve existing video loading, calibration, overlay, detection, per-video config, and MIDI export behavior unless the active task explicitly includes a migration.

### II. Keep Boundaries One-Way

Code must keep the repo dependency direction: `GUI -> workflows -> detection -> core`. `main.py` stays a composition facade, not a workflow body.

### III. Make UI Modes Explicit

Calibration, overlay fitting, detection tuning, and MIDI actions must expose clear user-facing modes. Do not rely on ambiguous gestures when a visible mode or state is safer.

### IV. Detection Must Match Geometry

Overlay geometry used for visual feedback must stay aligned with detection geometry. Visual-only changes that misrepresent sampled regions are not acceptable.

### V. Verify Before Completion

Behavior changes require tests or a stated reason tests are not practical. Verification commands live in `docs/testing.md`.

## Development Workflow

Backlog owns project status. Spec Kit owns non-trivial feature planning. `AGENTS.md` is the top-level agent contract.

Use Backlog tasks for acceptance criteria and final summaries. Use Spec Kit specs, plans, and tasks only when the implementation needs that detail.

## Governance

This constitution applies to Spec Kit feature work. Amendments require updating this file and recording the durable reason in `backlog/decisions/`.

**Version**: 1.0.0 | **Ratified**: 2026-06-08 | **Last Amended**: 2026-06-08
