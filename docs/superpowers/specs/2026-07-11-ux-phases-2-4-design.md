# UX Phases 2-4 Design

## Status

Approved by the instruction to complete Phases 2, 3, and 4 from the internal UX audit. This design consolidates the already-reviewed audit into an implementation contract; it does not reopen Phase 1 decisions.

## Goal

Make the normal video-to-MIDI path self-explanatory, move specialist recovery controls out of the beginner path, and make the shipped Qt interface robust across keyboard navigation, larger fonts, and every supported locale.

## Scope

This work completes the remaining implementation phases in `logs/ux-audit/2026-07-07-settings-ux-audit/ux-audit-report.md`:

1. Phase 2: guided calibration workflow.
2. Phase 3: advanced settings reorganization.
3. Phase 4: accessibility and localization hardening.

It also closes the audit findings that directly support those phases and were not included in Phase 1:

- a usable empty state when startup is dismissed;
- a short startup cue describing suitable input;
- visible Manual Fit mode explanations;
- `Show MIDI in Folder` as the conversion-complete default;
- consistent user-facing terminology;
- native, accessible trim confirmation styling.

Out of scope:

- detector algorithm changes;
- automatic keyboard-box discovery;
- live language switching;
- new saved calibration or config fields;
- changes to MIDI generation;
- changes to the Rust touch-up editor;
- human-certified translations.

## Binding Decisions

- Keep `Left` and `Right` in user-facing calibration language. Explain that they mean Synthesia note color families, not physical keyboard position.
- English remains the source/default language. Language changes still apply after restart.
- Keep all existing config keys, detector parameter names, `.ini` fields, and overlay sidecar formats unchanged.
- Do not create a linked worktree and do not push.
- The normal path must not require reading a tooltip or opening Help.
- Advanced controls remain available, but are collapsed and symptom-led by default.

## Phase 2: Guided Workflow

### Guide Page

Add a first-position `Guide` page to the settings rail. Keep the existing `Calibration` page as the detailed/manual surface.

The Guide shows these steps:

1. Open or download a video.
2. Find and check the keyboard overlays.
3. Capture a no-key frame.
4. Find pressed-key colors.
5. Create MIDI.

Each step shows:

- a stable title;
- `Done`, `Next`, `Needs review`, or `Not ready` status;
- one short frame/action instruction;
- one primary action;
- a recovery action only when it is useful.

The step model is derived from existing `AppState` data:

- video: `video.filepath` exists;
- overlays: at least one overlay exists;
- alignment review: overlays with no downstream calibration show `Needs review`; any existing no-key or exemplar calibration proves the user has progressed past review;
- no-key frame: every active overlay has an unlit reference, plus histogram data when histogram matching is enabled;
- pressed colors: every required enabled exemplar has an effective color;
- conversion: existing `ConversionReadiness` is true.

No new progress value is persisted. A loaded older project is interpreted from its existing calibration data.

Guide actions reuse existing signals/controllers:

- Open Video and Download from YouTube route to `VideoSessionUiController` through `ControlSignalManager`.
- Find Keyboard routes to the calibration wizard.
- Review Alignment opens the correct existing editor: Auto-Detect Tuning for reusable auto-detect context, Manual Fit for manual calibration, or the calibration wizard as fallback.
- Capture No-Key Frame reuses the existing manual calibration request.
- Find Pressed-Key Colors invokes the existing assisted scan from the current no-key frame.
- Create MIDI reuses the existing conversion request.

### Visual Examples

The Guide includes compact programmatic illustrations for:

- a keyboard box that contains the full keyboard;
- a no-key frame with no glowing overlays;
- a pressed-key frame with one clearly glowing key.

The illustrations use Qt painting rather than bundled sample-video frames. This keeps them deterministic, small, license-free, and neutral across locales. Captions remain translated text outside the drawing.

### Main Empty State

When no video is loaded, the canvas area shows:

- `Open a Synthesia-style video to begin`;
- the same one-line suitable-input cue used at startup;
- `Open Video` and `Download from YouTube` buttons;
- a visible Settings action.

The empty state disappears after a video session loads. It does not change video loading order or workflow ownership.

### Assisted Calibration Review

Replace the generic proposal question with `AssistedCalibrationDialog`.

The dialog shows:

- the number of candidate samples and color families found;
- the Left/Right color-family clarification;
- one row each for Left White, Left Black, Right White, and Right Black;
- a real color swatch and `Found`, `Not found`, or `Not used` state;
- no raw RGB tuple in primary UI.

Actions:

- `Use These Examples`: apply the proposal and save through the existing workflow.
- `Try Another Frame`: restore the pre-scan no-key references and all prior exemplar values, then leave visible Guide instructions for moving to another no-key frame and scanning again.
- `Keep Current Examples`: restore the pre-scan no-key references and preserve all prior exemplar values.
- Closing the window behaves like `Keep Current Examples`.

The controller reports transient assisted-calibration state to the Guide (`scanning`, `applied`, `none_found`, `retry`, or `kept`). None of that state is persisted.

### Supporting Audit Items

- Startup adds one concise suitable-input line and marks missing recent files instead of allowing a later failure. Duplicate filenames show parent-folder context.
- Manual Fit displays a translated explanation for the selected mode. `Select Overlays` explicitly tells the user to draw around problem keys.
- Conversion Complete keeps both actions but defaults keyboard activation to `Show MIDI in Folder`.

## Phase 3: Advanced Settings

### Rail Structure

Use this settings order:

1. Guide
2. Calibration
3. Overlays
4. Detection
5. MIDI
6. Advanced
7. Optional
8. Language

`Language` remains the bottom item directly under `Optional`, as previously requested.

### Detection

The default Detection page contains only Detection Sensitivity and its missing-notes/extra-notes guidance.

Move the existing mode controls into collapsed symptom sections on the Advanced page:

- `Gradient or uneven pressed colors` owns histogram matching.
- `Pressed colors fade in or out` owns frame-delta matching.
- `False black-key notes` owns the black-key filter.

Existing widget attributes, signals, state updates, and persisted values remain intact. Reorganization changes only widget ownership/layout and visible labels.

### Repeated Notes

Move the current Spark page into a collapsed `Repeated notes merge together` section on Advanced.

Primary labels use `Repeated Notes Fix`, `flash area`, and `repeated-note setup`. The internal subsystem, signals, and config fields remain named Spark. Detailed controls remain hidden until the feature checkbox is enabled.

### Trim

Move `Permanently Trim Project` into a collapsed Advanced section. Keep the non-destructive MIDI range on the MIDI page. The confirmation uses a native `QMessageBox` warning with Cancel as default and no custom high-contrast-breaking stylesheet.

### Auto-Detect Expert Controls

Rename the advanced tab to `Advanced (Expert)`, add a visible note that it is only for cases where basic edge alignment cannot find the keys, and keep every advanced category collapsed on first open. Basic edge controls remain the default tab.

### Glossary

Add a small collapsed glossary on Advanced for terms that remain necessary:

- keyboard box;
- overlay;
- Left/Right color family;
- detection sensitivity;
- repeated-notes flashes.

Internal names such as ROI, histogram, and delta may remain in code, logs, or secondary technical tooltips, but not as unexplained primary labels.

## Phase 4: Accessibility And Localization

### Interactive Targets

- Raise 30-32 px icon/adjustment targets to at least 36 px.
- Use 40 px for the main Settings button.
- Give icon-only and symbol-only controls accessible names and descriptions.
- Keep controls responsive; do not introduce fixed widths that clip translated text.

### Visible Instructions And Contrast

- Essential workflow instructions remain visible in Guide or action rows.
- Replace low-contrast `#888` status text with at least `#595959` on light backgrounds.
- Replace light green status text with a darker accessible green such as `#2e7d32`.
- Keep platform-native disabled and warning styling where possible.

### Dynamic Settings Rail

Calculate rail width from the actual translated item text and current font metrics. Use 98 px as a minimum, not a fixed width. Resize the lower action rail to the same width.

Tests must prove that each rail label fits for:

- English;
- Spanish;
- Japanese;
- Russian;
- Simplified Chinese;
- Korean;
- Brazilian Portuguese;
- pseudo-locale `qps`.

The fit check runs at default, 125%, and 150% font sizes.

### Keyboard Navigation

Define and test explicit, logical focus order for:

- StartupDialog;
- the Guide/settings rail and global actions;
- CalibrationWizard;
- YouTubeDownloadDialog.

Default actions must match the beginner path: Open Video at startup, Use These Examples in assisted calibration, Save in tuning, and Show MIDI in Folder after conversion.

### Deterministic Visual Matrix

Add a read-only/offscreen UI renderer that creates screenshots and a machine-readable clipping report without videos or network access.

The pseudo-locale 150% pass covers:

- startup;
- every settings page;
- calibration wizard;
- assisted calibration proposal;
- Auto-Detect Tuning basic and expert tabs;
- Manual Fit;
- YouTube download.

The renderer is a verification tool, not runtime app behavior. Output goes to ignored `logs/ux-audit/` paths unless a temporary directory is supplied.

## Architecture

New focused GUI modules:

- `gui/calibration_guide.py`: pure guide snapshot derivation, step widgets, and illustrations.
- `gui/assisted_calibration_dialog.py`: proposal presentation and user decision only.
- `gui/video_empty_state.py`: empty canvas actions and copy.
- `gui/ui_glossary.py`: translated user-facing glossary entries.
- `tools/render_ui_matrix.py`: deterministic offscreen screenshot/clipping verifier.

Existing owners remain responsible for behavior:

- `ControlPanelQt` composes settings pages, exposes existing signals, and updates guide/advanced widgets from `AppState`.
- `CalibrationWizardController` runs assisted scanning and applies/restores proposals.
- `ControlSignalManager` connects new UI requests directly to existing controllers.
- `VideoSessionCoordinator` preserves load ordering and only toggles the empty-state presentation after a successful load.
- `main.py` remains composition and public UI adaptation; no workflow body is added.

Dependency direction remains `GUI -> workflows -> detection -> core`.

## Compatibility

- No config schema or migration.
- No detector parameter rename.
- No overlay or MIDI behavior change.
- Existing saved configs load into the reorganized controls and retain every value.
- Guide status is derived from existing data and can be recomputed at any time.
- All existing ControlPanel widget attributes used by controllers/tests remain available.

## Localization

Every new visible string uses Qt translation calls. After extraction:

- update all six production `.ts` catalogs;
- compile all `.qm` files;
- preserve placeholders and markup;
- use the existing serial per-language agent workflow for new translations;
- keep `qps` hidden from user selectors but available to verification.

## Testing

Phase 2:

- pure guide-state tests for fresh, video-only, overlays, no-key, exemplars, and conversion-ready states;
- action-routing tests;
- assisted dialog swatch/state/default-action tests;
- apply, retry, cancel, and no-result preservation tests;
- empty-state and startup recent-file tests;
- Manual Fit explanation and conversion-default tests.

Phase 3:

- rail structure and default visibility tests;
- advanced symptom sections own the existing controls;
- Spark/Trim are absent as top-level pages;
- existing saved values still populate every moved control;
- Auto-Detect expert note/categories default collapsed;
- glossary copy is visible on expansion.

Phase 4:

- minimum target and accessible-name tests;
- contrast-token tests for custom status styles;
- all-locale rail-fit matrix at three font scales;
- keyboard focus-order tests for four required surfaces;
- pseudo-locale screenshot renderer smoke and clipping report;
- localization manifest/catalog integrity tests.

Default final gate:

```bash
git diff --check
.venv/bin/python -m compileall -q synthesia2midi
.venv/bin/python -m pytest
```

Additional final gates:

```bash
.venv/bin/python -m synthesia2midi.tools.audit_ui_strings --output docs/localization/ui-string-manifest.json
.venv/bin/python -m synthesia2midi.tools.render_ui_matrix --locale qps --font-scale 1.5 --output logs/ux-audit/phase-4-qps
.venv/bin/pyside6-lupdate -extensions py synthesia2midi/synthesia2midi -ts /tmp/synthesia2midi_lupdate_probe.ts
for ts_file in synthesia2midi/synthesia2midi/translations/synthesia2midi_*.ts; do
  locale_name=$(basename "$ts_file" .ts | sed 's/^synthesia2midi_//')
  .venv/bin/pyside6-lrelease "$ts_file" -qm "/tmp/synthesia2midi_${locale_name}_probe.qm"
done
```

## Acceptance Criteria

- A new user can see the next calibration action and current completion state without opening Help.
- The Guide covers video, overlays, no-key capture, pressed colors, and conversion.
- Assisted proposals use swatches and preserve prior state unless explicitly accepted.
- Empty/startup states explain how to begin without long instructional popups.
- Detection specialist modes, repeated-note setup, and Trim are collapsed under Advanced by default.
- Auto-Detect expert controls are clearly marked and collapsed.
- Existing settings and saved configs retain their values and behavior.
- Required actions do not depend on tooltips.
- Core controls meet the target-size and keyboard-order requirements.
- No settings rail label clips in any shipped locale or `qps` at 150% font size.
- The pseudo-locale visual matrix completes without reported clipping or empty screenshots.
- All production translation catalogs are complete and compile.
- The full automated suite passes.

## Risks

- `ControlPanelQt` is large. New reusable presentation belongs in focused modules, while compatibility attributes stay on the panel.
- Qt focus chains can include internal child widgets. Tests must verify user-reachable order rather than assuming creation order.
- Font and style metrics vary by platform. Deterministic tests should measure text against allocated geometry and retain a small safety margin.
- Moving controls can accidentally break signal wiring. Tests must assert both state restoration and signal emission for moved controls.
- The assisted scan currently captures no-key references before proposing examples. Retry/cancel must restore that snapshot exactly.
