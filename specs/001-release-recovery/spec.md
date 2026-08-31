# Feature Specification: v0.2.2 Release Recovery

**Feature Branch**: `001-release-recovery`

**Created**: 2026-08-31

**Status**: Ready for planning

**Input**: User description: "Repair the Windows packaged-helper failures, validate and publish a replacement release for the existing YouTube fix, make Guide alignment review advance reliably, make Unicode video names safe in MIDI conversion, and reconcile TASK-9. Issue #6 is out of scope because trim/range support already exists."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Download a self-contained working release (Priority: P1)

As a Windows user, I can download the portable release and use video loading, YouTube support, and the MIDI touch-up editor without installing the helper programs separately.

**Why this priority**: The current Windows release ships the required files but cannot locate them, and two shipped video helpers are not independently runnable. This invalidates the portable-release promise and blocks multiple major workflows.

**Independent Test**: Download the candidate Windows archive onto a clean Windows environment, run its package verification, then open the app and exercise helper discovery without adding helper programs to the system path.

**Acceptance Scenarios**:

1. **Given** a clean Windows environment with no system FFmpeg, Deno, or touch-up editor, **When** the packaged verification runs, **Then** every bundled helper and required soundfont is found and every executable helper completes a non-interactive version or help check.
2. **Given** the exact YouTube video from issue #9, **When** its initial media request is rejected with HTTP 403, **Then** the packaged app retries with the supported alternate client and completes the download.
3. **Given** a missing, misplaced, or non-runnable packaged helper, **When** release verification runs, **Then** the candidate release is rejected before publication with the failing helper identified.
4. **Given** the existing macOS release flow, **When** the same verification runs, **Then** macOS packaged helper and asset discovery continues to work.

---

### User Story 2 - Complete alignment review without a Guide loop (Priority: P2)

As a user following the Guide, I can review and confirm correctly aligned keyboard overlays and then continue to the no-key frame step without performing an unrelated workaround.

**Why this priority**: The current Guide can remain on `Review Alignment` after a successful save because it has no direct evidence that review was confirmed.

**Independent Test**: Start with a video and uncalibrated overlays, confirm alignment in the appropriate editor without changing geometry, and verify the Guide advances to `Capture No-Key Frame`. Repeat with a canceled review and verify it remains on alignment review.

**Acceptance Scenarios**:

1. **Given** auto-detected overlays with no downstream calibration, **When** the user confirms the alignment editor, **Then** the Guide marks alignment review complete and exposes the no-key step.
2. **Given** manual overlays with no downstream calibration, **When** the user confirms the manual alignment editor, **Then** the Guide marks alignment review complete and exposes the no-key step.
3. **Given** an alignment editor that is canceled, **When** the Guide refreshes, **Then** alignment remains marked for review.
4. **Given** a confirmed alignment followed by an automatic pressed-key scan that is canceled or finds nothing, **When** prior calibration samples are restored, **Then** the confirmed alignment state is not lost.

---

### User Story 3 - Convert videos with international filenames (Priority: P2)

As a user, I can create MIDI from a locally named video even when its filename contains fullwidth punctuation, non-Latin text, or emoji.

**Why this priority**: Conversion currently fails before processing can complete because the source filename is copied into a metadata field with a restricted character encoding.

**Independent Test**: Generate MIDI for representative filenames containing a fullwidth colon, CJK text, accented Latin text, and emoji; every file saves successfully and contains a readable, safe track name.

**Acceptance Scenarios**:

1. **Given** a filename containing `：`, **When** MIDI metadata is created, **Then** conversion completes without an encoding exception.
2. **Given** a filename containing characters that cannot be represented by the MIDI writer, **When** the track name is prepared, **Then** unsupported characters are converted safely while supported text remains recognizable.
3. **Given** an ordinary ASCII filename, **When** MIDI is created, **Then** its existing track-name behavior remains unchanged.

### Edge Cases

- A valid system helper must not conceal a missing packaged helper during release verification.
- A broken helper located beside the application must not cause a candidate portable release to pass.
- Packaged paths may differ between Windows one-folder layout and macOS application bundles.
- Package verification must terminate helpers non-interactively and must not open the GUI editor.
- Alignment confirmation must be reset when a new overlay set or video session replaces the reviewed overlays.
- A failed or canceled assisted calibration may restore samples, but it must not undo a separately confirmed alignment review.
- An international output filename must remain usable even if its MIDI metadata track name needs a restricted representation.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The portable release MUST locate its packaged video helpers, JavaScript runtime, touch-up editor, and soundfont without depending on a developer checkout or system installation.
- **FR-002**: Release verification MUST execute each packaged helper through a bounded, non-interactive probe and reject missing or non-runnable helpers.
- **FR-003**: Release verification MUST prove that the application itself resolves the packaged locations used at runtime, rather than validating only archive contents.
- **FR-004**: The release build MUST use reviewed, repeatable packager inputs so a dependency update cannot silently change the bundle layout.
- **FR-005**: A replacement patch release MUST include the existing media-403 retry behavior and MUST be verified before issue #9 is closed.
- **FR-006**: Existing macOS packaged resolution and source-checkout helper fallbacks MUST remain compatible.
- **FR-007**: Confirming alignment review MUST create explicit current-session evidence that the Guide can use independently of no-key or exemplar calibration.
- **FR-008**: Canceling alignment review MUST NOT mark the review complete.
- **FR-009**: Replacing the video or overlay set MUST invalidate prior current-session alignment confirmation.
- **FR-010**: Restoring calibration samples after an assisted-scan failure or cancellation MUST NOT erase a confirmed alignment review.
- **FR-011**: MIDI track-name creation MUST accept arbitrary Unicode source filenames without raising an encoding exception.
- **FR-012**: Safe metadata conversion MUST preserve already-supported characters and provide a deterministic fallback for unsupported characters.
- **FR-013**: Automated regression coverage MUST exercise packaged runtime detection, dead-helper rejection, Guide confirmation/cancellation, and representative international track names.
- **FR-014**: Existing project range/trim behavior and GitHub issue #6 MUST remain out of scope.
- **FR-015**: Backlog status for the original packaging task MUST be reconciled against verified release evidence rather than duplicated or assumed complete.

### Key Entities

- **Release Candidate**: A versioned portable archive plus its target platform, packaged helpers, required assets, verification report, and publication status.
- **Alignment Review State**: Current-session evidence tied to the active video and overlay set indicating whether the user confirmed alignment.
- **MIDI Track Name**: The user-derived metadata label written into the generated MIDI file after conversion to the writer's supported character set.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of required packaged helpers and assets are both resolved by the packaged application and validated before a release archive can be published.
- **SC-002**: The exact issue #9 video completes from the replacement packaged release after the expected initial 403 retry path.
- **SC-003**: A confirmed alignment review advances the Guide in one attempt, while a canceled review advances it zero times.
- **SC-004**: All representative ASCII, accented, fullwidth-punctuation, CJK, and emoji filenames complete MIDI track-name creation without an encoding failure.
- **SC-005**: The focused regression tests, full Python test suite, Rust editor gate, and both platform package preflight jobs pass before publication.
- **SC-006**: GitHub issue #9 is closed only after the replacement release asset is publicly available and verified.

## Assumptions

- The replacement release version is `v0.2.2`, the next patch after `v0.2.1`.
- The existing media-403 fallback on `main` is retained and re-verified rather than redesigned.
- Alignment review evidence is session state; existing downstream calibration remains sufficient proof for older saved projects.
- The portable package continues to bundle its helpers instead of requiring end users to install them.
- Existing source-mode workflows, per-video configuration compatibility, and macOS bundle conventions remain unchanged.
- The user authorized branch publication, preflight execution, version tagging, GitHub Release publication, and closure of issue #9 after verification.
