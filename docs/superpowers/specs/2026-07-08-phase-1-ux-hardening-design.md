# Phase 1 UX Hardening Design

## Status

Approved for planning on branch `codex/ux-guided-calibration`.

## Goal

Make the existing Synthesia2MIDI workflow easier to complete successfully without changing detector behavior, saved calibration formats, or the overall settings architecture.

## Scope

This is the first implementation slice from the UX audit. It focuses on the normal path from loaded video to usable MIDI:

1. Understand what is missing before conversion.
2. Calibrate overlays, no-key frame, and pressed-key examples with visible instructions.
3. Use existing auto-detect tuning and assisted calibration with clearer guidance.
4. Avoid confusing optional and destructive settings.
5. Preserve current backend behavior and config compatibility.

Out of scope:

- A full guided calibration wizard/stepper replacement.
- Live language switching.
- Detection algorithm changes.
- New real-video fixture corpus.
- Rust touch-up editor UI changes.
- Removing advanced settings.

## User Model

The target user wants a MIDI file and may not understand calibration terminology. The UI should tell them:

- what to do next;
- what frame state is required;
- whether a step is complete;
- what is missing before conversion;
- which settings are advanced recovery tools.

## Design Principles

- Prefer visible one-line guidance over popups and tooltips.
- Keep warnings as backup guardrails, not as the primary instruction path.
- Keep internal technical names where needed for compatibility, but use plain labels around them.
- Keep existing settings sections for this phase.
- Make risky/destructive controls visually and textually distinct.
- Keep Left/Right terminology for Synthesia note color families, but clarify that it does not mean physical keyboard position.

## Architecture

Most work belongs in Qt UI classes and existing controllers:

- `ControlPanelQt` owns settings sections, conversion status, visible calibration copy, overlay adjustment controls, Detection/Spark/MIDI/Trim/Optional wording, and settings-section tests.
- `CalibrationWizard` owns the initial keyboard-box dialog copy.
- `AutoDetectTuningDialog` owns auto-detect tuning guidance and reset labels.
- `CalibrationWizardController` owns the assisted calibration proposal text or custom proposal dialog if the implementation plan chooses that route.
- `YouTubeDownloadDialog` owns fallback framing and quality wording.
- `MidiTouchupController` may change default conversion-complete button focus if included in the final plan.
- `runtime_paths.py`, detection modules, conversion modules, config persistence, and video loading should not change except for tests or harmless imports.

No new backend dependency should point from detection/core into GUI.

## Components

### Conversion Readiness

Add a small UI-facing readiness model in or near `ControlPanelQt`.

It should answer:

- Is conversion allowed?
- If not, what is the first user-actionable missing prerequisite?
- What label should the bottom rail show?

Candidate messages:

- `Load a video to convert.`
- `Create key overlays first.`
- `Capture a no-key frame.`
- `Capture at least one pressed-key example.`
- `Ready to create MIDI.`

`_can_convert()` may remain as the boolean compatibility method, but the UI should no longer show `Ready to convert` when the button is disabled.

### Calibration Section

Keep the existing Calibration settings section, but make the normal path visible without expanding Help.

Visible rows should communicate:

- `Find the keyboard`
  - `Pause on a clear frame where the full keyboard is visible.`
  - Button: `Draw Keyboard Box and Find Keys` or equivalent.
- `Capture no-key frame`
  - `Pause where no keys are glowing.`
  - Button: `Capture No-Key Frame` or equivalent.
- `Capture pressed-key examples`
  - `Pause where a key is glowing, then click that key.`
  - Keep buttons for `Set Left White`, `Set Left Black`, `Set Right White`, `Set Right Black`.
  - Add clarification: `Left/Right refer to Synthesia note colors, not the physical side of the keyboard.`

The existing collapsed Help section can remain, but critical instructions must not depend on it.

### Overlay Quick Adjustments

The current plus/minus controls should expose current adjustment values and reset affordances.

Minimum acceptable phase-1 behavior:

- Each quick adjustment row shows a visible current value.
- Each quick adjustment row has a reset-to-zero control.
- Left/Right Slant values start at `0` and update when the user clicks plus/minus.

Preferred behavior:

- Follow the Manual Fit pattern with a numeric spinbox and reset button where practical.

This phase should not change Manual Fit behavior or overlay persistence semantics unless the implementation plan identifies an existing state object that already owns those values safely.

### Calibration Wizard Copy

Rename implementation-heavy text:

- `Select Keyboard Region With Autodetector` becomes a plain action such as `Draw Keyboard Box and Find Keys`.
- Add visible instruction before the button: `Pause on a clear frame where the full keyboard is visible.`
- `Edit Current Calibration` should show an inline disabled reason when unavailable, not only a tooltip.

### Auto-Detect Tuning

Add top guidance:

`Check the overlays on the video. If they line up with the keys, click Save. If the edges are off, adjust the edge controls.`

Rename:

- `Reset All to Active Defaults` to `Reset to Recommended Settings`.
- Advanced tab to `Advanced Detector Settings` or another clear expert label.

Keep current preview behavior and tuning parameters unchanged.

### Assisted Calibration Confirmation

Keep the current assisted calibration algorithm unchanged.

Improve the user-facing confirmation:

- Avoid raw RGB tuple emphasis in the main message.
- Explain whether one or two Synthesia note color families were found.
- Use Left/Right wording only with the color-family clarification.
- Keep technical details in logs or secondary details if a custom dialog is added.

If a custom swatch dialog is too large for Phase 1, the implementation plan may use a simpler `QMessageBox` copy improvement, as long as raw tuples are not the primary message.

### Detection Section

Rename and frame current controls around symptoms:

- `Detection Threshold` becomes `Detection Sensitivity`.
- Visible helper: `Missing notes? Lower it. Extra notes? Raise it.`
- Histogram Detection copy explains gradients/uneven pressed colors.
- Delta Detection copy explains pressed colors fading in/out.
- Black Key Filter copy explains false black-key notes.

Existing controls and state wiring remain.

### Spark Section

Keep the section available in Phase 1, but frame it as a niche repeated-note recovery feature.

Add visible guidance:

`Use this only if repeated notes merge into one long note.`

Rename user-facing `ROI` wording to `area` or `region`, for example `Select Spark Area Above Keys`.

Hide or de-emphasize disabled calibration controls if feasible without layout churn. If not feasible in Phase 1, the visible guidance is required.

### MIDI and Trim Sections

Separate non-destructive conversion range from destructive project trim.

MIDI section:

- Rename `Custom MIDI Processing Range` to `Convert Only Part of the Video`.
- Clarify that it affects MIDI creation only.

Trim section:

- Rename to `Permanently Trim Project`.
- Add visible warning before the Trim button:
  - `Most users should use MIDI range instead. Trim changes the working video session, not the original video file.`
- Keep cancellation as the default in the confirmation dialog.

### Optional Section

Rename Hand Assignment to plain language:

`Put each hand/color on a separate MIDI channel`

Add visible helper:

`Use this only if the video uses different colors for left and right hand notes.`

### YouTube Dialog

Keep downloader behavior and preferences unchanged.

Improve framing:

- Collapse or visually frame browser-cookie fallback under `If YouTube blocks the download`.
- Explain that browser cookies are used only as a fallback.
- Quality labels should favor MIDI accuracy language:
  - `1080p - recommended for best MIDI detection`
  - `720p - faster, may be less accurate`
  - `480p - fastest, highest risk of bad calibration`

## Data Flow

No persistent data format changes are planned.

Conversion readiness reads existing `AppState` fields:

- loaded video path;
- overlays;
- unlit reference colors/histograms;
- enabled exemplar slots;
- effective exemplar colors;
- detection threshold;
- MIDI tempo.

Overlay quick-adjust display may track transient session deltas in the UI if no persisted source exists. Any transient display state must reset when overlays are regenerated or the video/session changes.

## Error Handling

Existing warning paths remain:

- unlit frame may contain lit keys;
- lit exemplar sample looks unchanged;
- disabled exemplar type;
- invalid trim range;
- YouTube fetch/download errors.

Phase 1 should improve warning copy only when it directly supports the UX goals above. It should not weaken soft-warning bypasses where users can intentionally continue.

## Localization

All new user-visible strings must use existing Qt translation patterns:

- `QCoreApplication.translate(...)`
- `translate("ControlPanelQt", "...")`
- `self.tr(...)` only where already locally appropriate

After UI copy changes, update localization extraction/audit artifacts and production `.ts` files according to the repository's localization gate. Human translation quality is not part of this phase, but assets must remain structurally valid.

## Testing

Focused tests should cover:

- Conversion readiness status for missing video, missing overlays, missing no-key frame, missing pressed-key examples, and ready state.
- Calibration section shows visible instructions without expanding Help.
- Left/Right clarification appears near pressed-key examples.
- Overlay quick adjustment values and reset controls work.
- Calibration Wizard uses plain keyboard-box wording and visible frame instruction.
- Auto-Detect Tuning shows top guidance and recommended reset wording.
- Detection/Spark/MIDI/Trim/Optional labels use the new plain-language framing.
- YouTube fallback and quality labels remain functional and preferences still persist.
- Existing localization integrity tests pass after changed strings.

Default verification after implementation:

```bash
git diff --check
.venv/bin/python -m compileall -q synthesia2midi
.venv/bin/python -m pytest tests/test_controls_qt.py tests/test_startup_dialog.py tests/test_youtube_download_dialog.py tests/test_auto_detect_tuning_dialog.py tests/test_ui_string_audit.py tests/test_localization.py
.venv/bin/python -m pytest
```

## Acceptance Criteria

- The Convert area never claims readiness while disabled; it states the first missing prerequisite.
- Calibration instructions for keyboard box, no-key frame, and pressed-key examples are visible by default.
- Left/Right terminology is preserved and clarified as Synthesia note color/family language.
- Overlay quick adjustments show current values and provide reset controls.
- Calibration Wizard, Auto-Detect Tuning, Detection, Spark, MIDI, Trim, Optional, and YouTube copy are clearer without backend behavior changes.
- Destructive Trim is visually and textually separated from non-destructive MIDI range.
- Existing per-video configs, overlay sidecars, detection parameters, and conversion behavior remain compatible.
- Tests and localization gates are updated for all changed visible strings.

## Risks

- `ControlPanelQt` is already large; implementation should add small helpers rather than broad refactors.
- Overlay quick adjustment values may not have a persisted state source; transient UI state must not misrepresent saved config.
- Localization churn is likely because many user-visible strings will change.
- Changing labels may break tests that assert exact text.

## Open Decisions Resolved

- Phase 1 is the current branch scope.
- The full guided calibration redesign is deferred.
- Left/Right terminology stays in the UI, with clarification.
- No new worktree will be created unless Jeff asks for one later.
