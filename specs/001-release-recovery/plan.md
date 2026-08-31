# Implementation Plan: v0.2.2 Release Recovery

**Branch**: `001-release-recovery` | **Date**: 2026-08-31 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/001-release-recovery/spec.md`

## Summary

Ship `v0.2.2` as a verified replacement for `v0.2.1`: resolve PyInstaller 6
one-folder resources through `sys._MEIPASS`, stage real FFmpeg executables rather
than Chocolatey shims, and make the packaged application execute a fail-closed
self-check before the archive is produced. In the same bounded release, add
non-persisted alignment-review evidence for the Guide and normalize arbitrary
Unicode source names into MIDIUtil's supported text encoding. Tests precede each
behavior change, and release publication follows a successful cross-platform
preflight.

## Technical Context

**Language/Version**: Python 3.12 in release CI; Rust stable for the touch-up editor

**Primary Dependencies**: PySide6, PyInstaller 6.22.2,
pyinstaller-hooks-contrib 2026.7, MIDIUtil 1.2.1, yt-dlp, FFmpeg/ffprobe, Deno,
and the existing Rust editor

**Storage**: Existing per-video INI/JSON project files; one temporary JSON package
self-check report during release builds

**Testing**: pytest, compileall, git diff checks, Cargo fmt/test/check, packaged
self-check, GUI smoke, GitHub Windows x64/macOS arm64 preflight

**Target Platform**: Windows x64 and macOS Apple Silicon packaged apps; source
compatibility on supported desktop platforms

**Project Type**: PySide6 desktop application with a Rust companion executable

**Performance Goals**: Package self-check completes within 60 seconds; no added
work on normal interactive startup; Guide transitions immediately after an
accepted editor

**Constraints**: Portable archives cannot depend on build-agent paths or system
helpers; helper probes must be non-interactive and bounded; existing saved
config formats and macOS bundle resolution must remain compatible

**Scale/Scope**: Four reported release/workflow defects, one patch release, and
focused changes across packaging, runtime paths, Guide state, and MIDI metadata

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **I. Preserve The Working App**: PASS. No config migration; source fallbacks,
  existing downstream-calibration Guide evidence, and normal ASCII track names
  remain supported.
- **II. Keep Boundaries One-Way**: PASS. Runtime/package checks stay in focused
  modules; Guide derivation stays in GUI over core state; MIDI normalization
  stays in the MIDI writer. `main.py` is untouched.
- **III. Make UI Modes Explicit**: PASS. Alignment completion becomes explicit
  current-session state set only by accepted editor outcomes.
- **IV. Detection Must Match Geometry**: PASS. No detection or overlay geometry
  algorithm changes are planned.
- **V. Verify Before Completion**: PASS. Regression tests are written first,
  followed by local gates, packaged self-checks, and both remote package jobs.

Post-design re-check: PASS. The data model and contracts below preserve the
same boundaries and add no constitution exceptions.

## Project Structure

### Documentation (this feature)

```text
specs/001-release-recovery/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── alignment-review.md
│   └── package-self-check.md
└── tasks.md
```

### Source Code (repository root)

```text
.github/workflows/release.yml
packaging/
├── Synthesia2MIDI.spec
├── build_release.py
└── requirements-build.txt
synthesia2midi/
├── run.py
└── synthesia2midi/
    ├── core/app_state.py
    ├── gui/calibration_guide.py
    ├── gui/calibration_wizard_controller.py
    ├── gui/manual_keyboard_fit_controller.py
    ├── midi_generator.py
    ├── package_self_check.py
    ├── runtime_paths.py
    ├── version.py
    └── workflows/video_loading.py
tests/
├── test_build_release.py
├── test_calibration_guide.py
├── test_manual_keyboard_fit_controller.py
├── test_midi_generator.py
├── test_package_self_check.py
├── test_release_workflow.py
├── test_runtime_paths.py
└── test_version.py
```

**Structure Decision**: Extend the existing single desktop application. Package
self-check behavior gets one focused module; all other changes stay with their
current owner. No new service layer, config format, or `main.py` workflow body is
introduced.

## Complexity Tracking

No constitution violations require justification.
