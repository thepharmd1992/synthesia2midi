# Tasks: v0.2.2 Release Recovery

**Input**: Design documents from `/specs/001-release-recovery/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`,
`contracts/`, and `quickstart.md`

**Tests**: Regression tests are mandatory and are written before each behavior
change.

## Phase 1: Setup and Scope Control

**Purpose**: Establish one status owner and preserve unrelated work.

- [x] T001 Create TASK-26, activate `specs/001-release-recovery/`, record issue #6 as out of scope, and preserve untracked `uv.lock` in `backlog/tasks/task-26 - Recover-v0.2.2-release-and-reported-workflows.md`, `.specify/feature.json`, `AGENTS.md`, and `PROJECT_LOG.md`
- [x] T002 Commit the completed specification, research, contracts, plan, task list, and TASK-26 planning state without staging `uv.lock`

---

## Phase 2: User Story 1 - Self-Contained Working Release (Priority: P1) MVP

**Goal**: Produce a portable app that resolves and executes its own helpers, uses
reviewed packaging inputs, and rejects broken candidates before an archive exists.

**Independent Test**: In a frozen-layout fixture with an executable root and a
separate `_internal` root, the app resolves only package-owned helpers and writes
a passing self-check report; missing/system-only/dead helpers produce a failing
report and prevent archive creation.

### Tests for User Story 1

- [x] T003 [US1] Add failing `_MEIPASS` detection, frozen bundle precedence, and macOS compatibility regressions in `tests/test_runtime_paths.py`
- [x] T004 [US1] Add failing package-owned path, helper probe, timeout, missing asset, report schema, and launcher-mode tests in `tests/test_package_self_check.py` and `tests/test_packaged_entrypoint.py`
- [x] T005 [US1] Add failing pinned-packager, Chocolatey real-target, dead-binary, self-check-before-archive, and archive rejection tests in `tests/test_build_release.py`
- [x] T006 [US1] Add failing workflow assertions for the pinned Windows FFmpeg package and next preflight version in `tests/test_release_workflow.py`

### Implementation for User Story 1

- [x] T007 [US1] Add final-field `bundle_root` detection and frozen bundle-root precedence while preserving source and macOS fallbacks in `synthesia2midi/synthesia2midi/runtime_paths.py`
- [x] T008 [US1] Implement the versioned, bounded, package-owned helper and asset report in `synthesia2midi/synthesia2midi/package_self_check.py`
- [x] T009 [US1] Route `--package-self-check <report>` before Qt startup in `synthesia2midi/run.py`
- [x] T010 [US1] Pin PyInstaller 6.22.2 and pyinstaller-hooks-contrib 2026.7 in `packaging/requirements-build.txt` and install that file from `packaging/build_release.py`
- [x] T011 [US1] Resolve Chocolatey shims to unambiguous real FFmpeg/ffprobe package binaries and probe all staged executable inputs in `packaging/build_release.py`
- [x] T012 [US1] Run and parse the packaged self-check before GUI smoke and move zip creation after all validation in `packaging/build_release.py`
- [x] T013 [US1] Pin Chocolatey FFmpeg 9.0.1 and set the v0.2.2 development preflight label in `.github/workflows/release.yml`
- [x] T014 [US1] Run the focused packaging tests and create a packaging checkpoint commit after they pass

**Checkpoint**: A candidate package cannot be archived unless the frozen app
itself finds and executes all bundled helpers.

---

## Phase 3: User Story 2 - Complete Alignment Review (Priority: P2)

**Goal**: Advance the Guide after an accepted alignment review without confusing
editor opening, cancellation, or assisted-scan results with acceptance.

**Independent Test**: Confirm unchanged manual and automatic alignment and see
`Capture No-Key Frame`; reject either editor and remain on `Review Alignment`;
cancel a queued assisted scan after acceptance and remain advanced.

### Tests for User Story 2

- [ ] T015 [US2] Add failing Guide derivation tests for explicit review evidence, legacy downstream evidence, and initial false state in `tests/test_calibration_guide.py` and `tests/test_app_state.py`
- [ ] T016 [US2] Add failing Manual Fit accept/reject tests and immediate Guide refresh assertions in `tests/test_manual_keyboard_fit_controller.py`
- [ ] T017 [US2] Add failing auto-tuning accept/reject, assisted restoration, new overlay, and new video invalidation tests in `tests/test_calibration_wizard_controller.py`, `tests/test_bugfix_regressions.py`, and `tests/test_video_loading_paths.py`

### Implementation for User Story 2

- [ ] T018 [US2] Add non-persisted `CalibrationConfig.alignment_reviewed` and use it as independent Guide evidence in `synthesia2midi/synthesia2midi/core/app_state.py` and `synthesia2midi/synthesia2midi/gui/calibration_guide.py`
- [ ] T019 [US2] Set review evidence only on accepted Manual Fit and refresh the control panel in `synthesia2midi/synthesia2midi/gui/manual_keyboard_fit_controller.py`
- [ ] T020 [US2] Set review evidence before accepted auto-tuning launches assisted calibration, preserve it across assisted restoration, and clear it for replacement overlays in `synthesia2midi/synthesia2midi/gui/calibration_wizard_controller.py`
- [ ] T021 [US2] Clear current-session review on new/closed video and calibration reset in `synthesia2midi/synthesia2midi/workflows/video_loading.py` and `synthesia2midi/synthesia2midi/workflows/calibration.py`
- [ ] T022 [US2] Run focused Guide/controller tests and create an alignment checkpoint commit after they pass

**Checkpoint**: Accepted review advances exactly once; cancellation advances zero
times; later assisted-scan outcomes cannot reverse accepted review.

---

## Phase 4: User Story 3 - International Filename MIDI Metadata (Priority: P2)

**Goal**: Prevent arbitrary source filenames from crashing MIDI track-name creation.

**Independent Test**: Create track metadata for ASCII, accented Latin, fullwidth
colon, CJK, and emoji names and save each MIDI file without an encoding exception.

### Tests for User Story 3

- [ ] T023 [US3] Add failing parameterized normalization, MIDIUtil event, and save regressions for representative names in `tests/test_midi_generator.py`

### Implementation for User Story 3

- [ ] T024 [US3] Normalize track names with NFKC and deterministic ISO-8859-1 replacement before passing them to MIDIUtil in `synthesia2midi/synthesia2midi/midi_generator.py`
- [ ] T025 [US3] Update the development version to `0.2.2-dev` and its regression in `synthesia2midi/synthesia2midi/version.py` and `tests/test_version.py`
- [ ] T026 [US3] Run focused MIDI/version tests and create a Unicode metadata checkpoint commit after they pass

**Checkpoint**: International names no longer abort conversion and ASCII metadata
is unchanged.

---

## Phase 5: Cross-Cutting Verification and Release Preparation

**Purpose**: Validate the combined blast radius before any publication.

- [ ] T027 Update the canonical packaged release gate and self-check expectations in `docs/testing.md`
- [ ] T028 Run `git diff --check`, compileall, the full pytest suite, and all Rust fmt/test/check gates from `specs/001-release-recovery/quickstart.md`
- [ ] T029 Build the local macOS v0.2.2-dev package and verify its package-self-check report, GUI smoke, archive integrity, helpers, SoundFont, architecture, and version
- [ ] T030 Re-run the exact GitHub issue #9 YouTube URL through the existing source downloader and record the initial 403/alternate-client/success evidence without changing its fallback policy
- [ ] T031 Record local evidence in TASK-26 and create the release-preparation checkpoint commit without staging generated build outputs or `uv.lock`

---

## Phase 6: Remote Preflight, Publication, and Reconciliation

**Purpose**: Publish only the exact candidate proven by local and remote gates,
then make project status match reality.

- [ ] T032 Push the verified candidate to `codex/v0.2.2-preflight` and wait for the Release workflow plus Windows/macOS/Linux CI to finish
- [ ] T033 Download both preflight artifacts, verify zip integrity/layout/version, and confirm the Windows log shows real FFmpeg/ffprobe plus a passing packaged self-check
- [ ] T034 Fast-forward the verified feature commit into local and remote `main`, create annotated tag `v0.2.2`, push it, and wait for the tag Release workflow
- [ ] T035 Verify the public v0.2.2 Windows x64/macOS arm64 versioned and latest assets, their digests/layouts, and release visibility
- [ ] T036 Close GitHub issue #9 with exact-video and v0.2.2 release evidence while leaving issue #6 unchanged
- [ ] T037 Reconcile every TASK-9 acceptance criterion from current evidence, mark TASK-26 complete, update `PROJECT_LOG.md`, deactivate `.specify/feature.json`, and restore the AGENTS.md Spec Kit pointer to none
- [ ] T038 Commit and push the final status reconciliation on `main`, delete the merged local feature branch, retain the remote preflight branch because remote deletion was not authorized, and confirm `uv.lock` remains untouched

---

## Dependencies and Execution Order

- Phase 1 completes before behavior changes.
- User Story 1 is the release-blocking MVP and completes before other stories.
- User Stories 2 and 3 are independent after Phase 1, but are executed
  sequentially to keep checkpoint commits auditable.
- Each story's tests must fail before its implementation tasks begin and pass
  before its checkpoint.
- Full local verification precedes every remote push.
- The preflight commit must be the same commit merged and tagged for `v0.2.2`.
- TASK-9 and GitHub issue #9 cannot be completed before public release evidence.

## Implementation Strategy

1. Lock packaging correctness first because it invalidates the current public
   portable build across several workflows.
2. Land the Guide and MIDI fixes as isolated, test-backed checkpoints.
3. Exercise the combined build locally, then use the existing nonpublishing
   cross-platform preflight.
4. Publish the proven commit only once, verify public artifacts, and reconcile
   status after the evidence exists.
