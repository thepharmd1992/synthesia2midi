# Touch-Up Editor Access, Octave, and Channel Colors Design

## Goal

Make the MIDI touch-up editor easier to open, add a safe whole-file octave adjustment, and preserve the visual identity of Synthesia2MIDI color families when a generated MIDI is reopened in the editor.

## Scope

This feature adds:

- An always-available `Open MIDI in Touch-Up Editor...` action in the main File menu.
- A clearly named `Open Touch-Up Editor` action in the Settings footer.
- Whole-file octave down/up controls in the Rust editor.
- Optional Synthesia2MIDI channel-color metadata in generated MIDI files.
- Exemplar-derived falling-bar and active-key colors in the Rust editor.
- Distinct fallback colors for MIDI files without Synthesia2MIDI metadata.
- Side-by-side visualization when multiple channels play the same pitch at the same time.

The Welcome dialog remains unchanged.

## Non-Goals

- Per-channel or selected-note octave commands.
- Manual channel-color editing in the Rust editor.
- A sidecar color file or a live dependency on the source video configuration.
- Changing the audible instrument by channel.
- Localizing the currently English-only Rust editor as part of this feature.
- Overwriting the source MIDI file.

## Editor Entry Points

The File menu and Settings footer must call the same `MidiTouchupController.open_from_picker()` path. There must not be separate editor-launch implementations.

The File menu adds `Open MIDI in Touch-Up Editor...`. The Settings footer renames the existing `Edit MIDI` button to `Open Touch-Up Editor`. Both remain available before or after a video is loaded.

The file picker starts in `RuntimePaths.midi_exports_dir()` when that directory exists. It falls back to the user's home directory. Existing binary discovery, process ownership, setup guidance, result parsing, and shutdown behavior remain owned by `MidiTouchupController`.

## Whole-File Octave Adjustment

The Rust toolbar adds a compact control:

```text
Octave  [-]  0  [+]
```

Each click proposes a 12-semitone shift for every note in the open document. The operation is atomic: it applies only if every resulting pitch remains within the editor's visible 88-key piano range, MIDI pitches 21 through 108 (`A0-C8`). If any note would leave that range, no note changes and the editor shows a short warning identifying whether the lowest or highest note blocked the shift.

One successful click is one undoable command. Undo and Redo restore every affected pitch together and update the displayed cumulative octave offset. The offset starts at zero for each newly opened MIDI and reflects whole-file octave commands relative to that loaded file. Individual note edits do not change the offset.

The editor pauses playback before applying a successful octave command, preserves the current selection where possible, refreshes synthesized song data, and redraws immediately. Saving writes the transposed pitches to the normal separate touch-up output path; the source MIDI remains unchanged.

## Portable Channel-Color Metadata

### MIDI Record

Synthesia2MIDI-generated MIDI files add one standard MIDI text meta event at tick zero on the primary track. Its payload uses this versioned prefix and compact JSON body:

```text
Synthesia2MIDI:color-map:v1:{"channels":{"0":{"natural":[90,168,255],"sharp_flat":[65,121,184]}}}
```

Rules:

- Channel keys are zero-based MIDI channels `0` through `15`.
- `natural` and `sharp_flat` values are RGB integer triples in `0` through `255`.
- Only active color families are written.
- A family may contain one or both morphology colors.
- The complete text payload is capped at 4 KiB.
- Normal MIDI software may ignore the namespaced text event without affecting playback.

The Python MIDI writer owns serialization of this record. The conversion workflow supplies the effective calibrated Natural and Sharp / Flat colors keyed by each family's stable MIDI channel. The metadata does not change note or channel generation.

### Rust Parsing

The Rust loader scans preserved text meta events for the exact versioned prefix. It validates the payload size, JSON shape, channel range, component count, and RGB values. The last valid version-one record at tick zero wins. Unknown versions, invalid values, and malformed JSON are ignored without preventing the MIDI from opening.

The existing preserved-event path keeps the text event intact when the editor saves a touch-up MIDI. If another program removes it, the editor uses its fallback palette the next time the file is opened.

## Rendering Colors

For a MIDI containing valid Synthesia2MIDI metadata:

- Natural-key notes use the channel's `natural` color.
- Sharp / Flat notes use the channel's `sharp_flat` color.
- If the requested morphology is missing, the editor derives a readable variant from the other morphology before falling back to the built-in channel palette.
- Colors that already read clearly are left unchanged.
- Colors that are too dark or too pale are adjusted only enough to remain visible on the dark falling-bar surface and the piano keys. The family hue remains recognizable.
- Existing outlines remain so pale neighboring bars do not merge visually.

For all other MIDI files, the editor uses a deterministic built-in palette with distinct colors for at least channels 0 through 3. Natural and Sharp / Flat notes remain visibly related, with the accidental variant darker when no explicit pair exists.

Both falling bars and active piano-key highlights resolve color through the same channel-and-morphology function.

## Same-Pitch Multi-Channel Notes

Different MIDI channels can play the same pitch during overlapping time ranges. A full-width draw order would hide one channel.

The falling-bar renderer groups notes by pitch and overlapping time span. Notes from distinct channels in the same overlap group receive stable left-to-right sublanes ordered by channel. This subdivision remains stable for the full displayed note so colors do not jump while the playhead moves.

At the current playhead, a piano key gathers all distinct active channels for that pitch. One channel fills the key normally; multiple channels split the key into equal side-by-side color bands ordered by channel.

Duplicate overlapping notes on the same channel do not create duplicate color bands.

## Failure Behavior

- A missing editor binary keeps the existing setup/re-download guidance.
- A missing MIDI path keeps the existing warning.
- Invalid color metadata never blocks opening or saving.
- A blocked octave shift changes nothing and does not create an Undo entry.
- Metadata serialization failure must not silently produce a partially written MIDI. Conversion reports the save failure through the existing error path.

## Architecture Boundaries

- `main.py` only creates the File-menu action and wires it to the existing controller.
- `controls_qt.py` owns the Settings-footer label and signal.
- `midi_touchup_controller.py` remains the sole Qt-to-Rust process boundary.
- `midi_generator.py` owns MIDI color-metadata serialization.
- `workflows/conversion.py` translates stable app color-family state into channel metadata.
- `tools/midi_touchup_editor_rust/` owns metadata parsing, octave editing, overlap layout, and rendering.

The Python-to-Rust launch arguments do not carry colors. The MIDI file is the portable contract.

## Localization

All new or renamed Qt strings are added to every tracked translation catalog and compiled `.qm` file. The pseudo-locale and UI-string manifest are regenerated. The Rust editor continues to use its existing English-only UI in this feature.

## Verification

Python coverage must verify:

- The File-menu action and Settings-footer button both reach `open_from_picker()`.
- No touch-up action is added to the Welcome dialog.
- The picker prefers `midi_exports_dir()` and falls back safely.
- Version-one metadata is deterministic, bounded, and contains stable channel mappings.
- Missing morphology colors serialize predictably.
- Existing MIDI channel assignment and legacy calibration behavior remain unchanged.
- Translation catalogs and the UI-string manifest remain complete.

Rust coverage must verify:

- Valid metadata parsing and malformed/unknown-version fallback.
- Metadata preservation through load and save.
- Distinct fallback colors for channels 0 through 3.
- Natural and Sharp / Flat color selection and readability adjustment.
- Stable same-pitch overlap sublanes and split active-key bands.
- Whole-file octave up/down, cumulative offset, Undo/Redo, and audio-data refresh.
- Atomic rejection at pitches 21 and 108 with no dirty or Undo state change.

Final verification includes the full Python suite, Rust tests and `cargo check`, the pseudo-locale UI matrix, and Windows x64 plus Apple Silicon packaged release smokes.
