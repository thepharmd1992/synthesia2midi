# Touch-Up Editor Access, Octave, and Channel Colors Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the touch-up editor easy to open, add atomic whole-file octave adjustment, and carry readable Synthesia2MIDI color-family identity through portable MIDI metadata.

**Architecture:** The Qt app keeps one `MidiTouchupController` launch path and adds two discoverable entry points. Python writes a versioned text meta event containing stable channel colors; the Rust editor parses that optional record, uses pure color/layout helpers for rendering, and retains the event through its existing preserved-event save path. Whole-file octave changes use the existing Rust command history so one click remains one Undo/Redo operation.

**Tech Stack:** Python 3.12, PySide6/Qt, midiutil, pytest, Rust 2021, eframe/egui, midly, serde/serde_json, Cargo, PyInstaller, GitHub Actions.

## Global Constraints

- Work on `codex/touchup-editor-access-octave-colors` without creating a worktree.
- Do not push this branch until Jeff explicitly requests it.
- Preserve existing per-video configuration and legacy two-family MIDI channel behavior.
- Keep the Welcome dialog unchanged; editor access belongs only in the File menu and Settings footer.
- The source MIDI is never overwritten; the Rust editor continues to create a separate touch-up file.
- Octave shifts are whole-document operations of exactly 12 semitones and must stay within MIDI pitches 21 through 108 (`A0-C8`).
- Color metadata uses a standard MIDI text event with prefix `Synthesia2MIDI:color-map:v1:` and a maximum complete payload size of 4096 bytes.
- Colors never travel in process arguments or a sidecar file.
- Invalid or unknown color metadata never prevents a MIDI from opening.
- All new Qt text must be extracted, translated in every production locale, pseudo-localized, compiled, and included in the UI-string manifest.
- The Rust editor remains English-only in this feature.
- Keep `main.py` as composition/wiring; workflow and process bodies remain in focused modules.
- Do not stage or modify the unrelated untracked `uv.lock`.

---

## File Map

**Qt access and launch path**

- Modify `synthesia2midi/synthesia2midi/main.py`: add the File-menu action and direct controller wiring.
- Modify `synthesia2midi/synthesia2midi/gui/controls_qt.py`: rename the footer action.
- Modify `synthesia2midi/synthesia2midi/gui/midi_touchup_controller.py`: prefer the MIDI export directory in the picker.
- Test `tests/test_main_window_layout.py`, `tests/test_controls_qt.py`, `tests/test_midi_touchup_controller.py`, and `tests/test_startup_dialog.py`.

**Portable Python metadata**

- Modify `synthesia2midi/synthesia2midi/midi_generator.py`: validate, serialize, and add the color-map text event.
- Modify `synthesia2midi/synthesia2midi/workflows/conversion.py`: map enabled family exemplars to stable channels.
- Test `tests/test_midi_generator.py` and `tests/test_color_family_channels.py`.

**Rust parsing and rendering**

- Create `tools/midi_touchup_editor_rust/src/color_map.rs`: parse metadata, derive missing morphology colors, adapt readability, and provide deterministic fallbacks.
- Create `tools/midi_touchup_editor_rust/src/channel_layout.rs`: compute stable same-pitch overlap lanes and active channel sets.
- Modify `tools/midi_touchup_editor_rust/src/main.rs`: retain parsed colors in `MidiDocument`, render with helpers, cache lanes, and add octave commands/UI.

**Localization, gates, and docs**

- Modify `synthesia2midi/synthesia2midi/translations/synthesia2midi_{es,ja,ru,zh_CN,ko,pt_BR}.ts` and matching `.qm` files.
- Modify `docs/localization/ui-string-manifest.json` and `docs/localization/translation-agent-packet.json`.
- Modify `tests/test_localization.py` and `tests/test_ui_string_audit.py` only when a new assertion is needed; do not weaken existing gates.
- Modify `.github/workflows/ci.yml` and `docs/testing.md` so Rust unit tests run in addition to `cargo check`.
- Update `ARCHITECTURE.MD` for the MIDI metadata contract.
- Update `backlog/tasks/task-23 - Improve-touch-up-editor-access-octave-and-channel-colors.md` as checkpoints complete.

---

### Task 1: Add Discoverable Editor Entry Points

**Files:**
- Modify: `synthesia2midi/synthesia2midi/main.py:143-168`
- Modify: `synthesia2midi/synthesia2midi/gui/controls_qt.py:313-356`
- Modify: `synthesia2midi/synthesia2midi/gui/midi_touchup_controller.py:84-103`
- Test: `tests/test_main_window_layout.py:150-220`
- Test: `tests/test_controls_qt.py:9-17`
- Test: `tests/test_midi_touchup_controller.py`
- Test: `tests/test_startup_dialog.py`

**Interfaces:**
- Consumes: `MidiTouchupController.open_from_picker() -> None`, `RuntimePaths.midi_exports_dir() -> Path`.
- Produces: `Video2MidiApp.open_midi_touchup_action: QAction`; clearer `ControlPanelQt.midi_touchup_button`; one unchanged controller picker path.

- [ ] **Step 1: Mark TASK-23 In Progress**

Change only the Backlog front matter:

```yaml
status: In Progress
```

- [ ] **Step 2: Write failing entry-point and picker tests**

Add to `tests/test_midi_touchup_controller.py`:

```python
def test_open_from_picker_prefers_midi_exports_directory(monkeypatch, tmp_path):
    home = tmp_path / "home"
    exports = home / "Desktop" / "Synthesia2MIDI MIDI Files"
    exports.mkdir(parents=True)
    runtime_paths = RuntimePaths(
        frozen=False,
        app_root=tmp_path,
        repo_root=tmp_path,
        home_dir=home,
        platform_name="darwin",
    )
    calls = []
    monkeypatch.setattr(
        "synthesia2midi.gui.midi_touchup_controller.detect_runtime_paths",
        lambda: runtime_paths,
    )
    monkeypatch.setattr(
        "synthesia2midi.gui.midi_touchup_controller.QFileDialog.getOpenFileName",
        lambda parent, title, start_dir, filters: calls.append(
            (title, start_dir, filters)
        ) or ("", ""),
    )

    MidiTouchupController(_fake_app()).open_from_picker()

    assert calls[0][1] == str(exports)
```

Add a companion test where the export folder does not exist and assert the start directory is `str(home)`.

Extend `test_main_menu_separates_primary_files_from_advanced_diagnostics` in `tests/test_main_window_layout.py`:

```python
assert "Open MIDI in Touch-Up Editor..." in file_labels
assert app.open_midi_touchup_action in app.file_menu.actions()
```

Add a trigger test that monkeypatches `QFileDialog.getOpenFileName`, triggers `app.open_midi_touchup_action`, and asserts the picker was called once. Extend `tests/test_controls_qt.py`:

```python
assert panel.midi_touchup_button.text() == "Open Touch-Up Editor"
```

Extend `tests/test_startup_dialog.py`:

```python
assert not hasattr(dialog, "midi_touchup_button")
assert [dialog.local_file_btn.text(), dialog.youtube_btn.text()] == [
    "Open Video File",
    "Download from YouTube",
]
```

- [ ] **Step 3: Run the focused tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_midi_touchup_controller.py tests/test_controls_qt.py tests/test_main_window_layout.py tests/test_startup_dialog.py -q
```

Expected: failures report the old `Edit MIDI` label, missing File-menu action, and Movies/Documents picker start path.

- [ ] **Step 4: Implement the two access points and export-folder picker**

In `main.py`, after the YouTube action:

```python
self.open_midi_touchup_action = QAction(
    QCoreApplication.translate(
        "Video2MidiApp", "Open MIDI in Touch-Up Editor..."
    ),
    self,
)
self.open_midi_touchup_action.triggered.connect(
    self.midi_touchup_controller.open_from_picker
)
filemenu.addAction(self.open_midi_touchup_action)
```

In `controls_qt.py`, retain the existing signal and change only the source label:

```python
self.midi_touchup_button = QPushButton(
    QCoreApplication.translate("ControlPanelQt", "Open Touch-Up Editor")
)
```

In `MidiTouchupController.open_from_picker()`:

```python
runtime_paths = detect_runtime_paths()
exports_dir = runtime_paths.midi_exports_dir()
start_dir = str(exports_dir if exports_dir.exists() else runtime_paths.home_dir)
```

Do not add a StartupDialog signal or button.

- [ ] **Step 5: Run the focused tests and verify GREEN**

Run the Step 3 command.

Expected: all selected tests pass. The localization manifest may remain stale until Task 6; do not run or weaken that gate here.

- [ ] **Step 6: Commit the access checkpoint**

```bash
git add \
  'backlog/tasks/task-23 - Improve-touch-up-editor-access-octave-and-channel-colors.md' \
  synthesia2midi/synthesia2midi/main.py \
  synthesia2midi/synthesia2midi/gui/controls_qt.py \
  synthesia2midi/synthesia2midi/gui/midi_touchup_controller.py \
  tests/test_main_window_layout.py \
  tests/test_controls_qt.py \
  tests/test_midi_touchup_controller.py \
  tests/test_startup_dialog.py
git commit -m "feat: expose the MIDI touch-up editor"
```

---

### Task 2: Write Portable Channel-Color Metadata

**Files:**
- Modify: `synthesia2midi/synthesia2midi/midi_generator.py`
- Modify: `synthesia2midi/synthesia2midi/workflows/conversion.py:20-49,286-299`
- Test: `tests/test_midi_generator.py`
- Test: `tests/test_color_family_channels.py`

**Interfaces:**
- Consumes: `DetectionState.get_effective_exemplar_lit_colors()`, `COLOR_FAMILIES`, and each family's stable `midi_channel`.
- Produces: `COLOR_MAP_META_PREFIX`, `serialize_channel_color_map(channel_colors) -> str`, `MidiWriter.add_channel_color_map(...) -> None`, and `_midi_channel_color_map(app_state) -> dict[int, dict[str, tuple[int, int, int]]]`.

- [ ] **Step 1: Write failing serializer and conversion-map tests**

Add to `tests/test_midi_generator.py`:

```python
import pytest

from synthesia2midi.midi_generator import (
    COLOR_MAP_META_PREFIX,
    serialize_channel_color_map,
)


def test_channel_color_metadata_is_deterministic_and_compact():
    payload = serialize_channel_color_map(
        {
            1: {"sharp_flat": (12, 34, 56), "natural": (90, 120, 150)},
            0: {"natural": (1, 2, 3)},
        }
    )

    assert payload == (
        COLOR_MAP_META_PREFIX
        + '{"channels":{"0":{"natural":[1,2,3]},'
        '"1":{"natural":[90,120,150],"sharp_flat":[12,34,56]}}}'
    )
    assert len(payload.encode("utf-8")) <= 4096


@pytest.mark.parametrize(
    "channel_colors",
    [
        {-1: {"natural": (1, 2, 3)}},
        {16: {"natural": (1, 2, 3)}},
        {0: {"natural": (1, 2)}},
        {0: {"natural": (1, 2, 999)}},
        {0: {"unknown": (1, 2, 3)}},
    ],
)
def test_channel_color_metadata_rejects_invalid_values(channel_colors):
    with pytest.raises(ValueError):
        serialize_channel_color_map(channel_colors)
```

Add a test that calls `MidiWriter.add_channel_color_map(...)` and asserts one midiutil `Text` event begins with `COLOR_MAP_META_PREFIX`.

Add to `tests/test_color_family_channels.py`:

```python
def test_conversion_builds_metadata_from_enabled_family_slots():
    state = AppState()
    state.detection.exemplar_lit_colors.update(
        {
            "LW": (10, 20, 30),
            "LB": (5, 10, 15),
            "COLOR_3_W": (100, 110, 120),
        }
    )
    state.detection.exemplar_key_type_enabled.update(
        {"LW": True, "LB": True, "RW": False, "RB": False,
         "COLOR_3_W": True, "COLOR_3_B": False}
    )

    assert conversion._midi_channel_color_map(state) == {
        0: {"natural": (10, 20, 30), "sharp_flat": (5, 10, 15)},
        2: {"natural": (100, 110, 120)},
    }
```

- [ ] **Step 2: Run tests and verify RED**

```bash
.venv/bin/python -m pytest tests/test_midi_generator.py tests/test_color_family_channels.py -q
```

Expected: import failures for the new metadata interfaces.

- [ ] **Step 3: Implement validation and serialization in `midi_generator.py`**

Add imports and constants:

```python
import json
from collections.abc import Mapping, Sequence

COLOR_MAP_META_PREFIX = "Synthesia2MIDI:color-map:v1:"
MAX_COLOR_MAP_META_BYTES = 4096
COLOR_MORPHOLOGIES = ("natural", "sharp_flat")
```

Add a deterministic serializer. Normalize channel keys in numeric order, include only known morphology names, require exactly three integer components in `0..255`, reject channels outside `0..15`, and check the byte limit after prefixing:

```python
def serialize_channel_color_map(channel_colors: Mapping[int, Mapping[str, Sequence[int]]]) -> str:
    channels: dict[str, dict[str, list[int]]] = {}
    for channel in sorted(channel_colors):
        if not isinstance(channel, int) or not 0 <= channel <= 15:
            raise ValueError(f"Invalid MIDI channel: {channel}")
        source = channel_colors[channel]
        unknown = set(source) - set(COLOR_MORPHOLOGIES)
        if unknown:
            raise ValueError(f"Unknown color morphology: {sorted(unknown)}")
        encoded: dict[str, list[int]] = {}
        for morphology in COLOR_MORPHOLOGIES:
            if morphology not in source:
                continue
            components = list(source[morphology])
            if (
                len(components) != 3
                or any(type(component) is not int for component in components)
                or any(not 0 <= component <= 255 for component in components)
            ):
                raise ValueError(f"Invalid RGB color for channel {channel} {morphology}")
            encoded[morphology] = components
        if encoded:
            channels[str(channel)] = encoded

    body = json.dumps(
        {"channels": channels},
        ensure_ascii=True,
        separators=(",", ":"),
    )
    payload = COLOR_MAP_META_PREFIX + body
    if len(payload.encode("utf-8")) > MAX_COLOR_MAP_META_BYTES:
        raise ValueError("Channel color metadata exceeds 4096 bytes")
    return payload
```

Add to `MidiWriter`:

```python
def add_channel_color_map(self, channel_colors, track: int = 0, time: float = 0.0) -> None:
    self.mf.addText(track, time, serialize_channel_color_map(channel_colors))
```

- [ ] **Step 4: Map effective exemplars in `conversion.py`**

Import `COLOR_FAMILIES` and add:

```python
def _midi_channel_color_map(app_state: AppState) -> dict[int, dict[str, tuple[int, int, int]]]:
    effective = app_state.detection.get_effective_exemplar_lit_colors()
    result: dict[int, dict[str, tuple[int, int, int]]] = {}
    for family in COLOR_FAMILIES:
        colors: dict[str, tuple[int, int, int]] = {}
        natural = effective.get(family.natural_slot)
        sharp_flat = effective.get(family.accidental_slot)
        if natural is not None:
            colors["natural"] = natural
        if sharp_flat is not None:
            colors["sharp_flat"] = sharp_flat
        if colors:
            result[family.midi_channel] = colors
    return result
```

In `_setup_midi_writer()`, add metadata after track name and tempo setup only when at least one effective color exists:

```python
channel_colors = _midi_channel_color_map(self.app_state)
if channel_colors:
    midi_writer.add_channel_color_map(channel_colors)
```

Let validation errors reach the existing conversion failure path; do not silently omit malformed app state.

- [ ] **Step 5: Run focused and compatibility tests**

```bash
.venv/bin/python -m pytest \
  tests/test_midi_generator.py \
  tests/test_color_family_channels.py \
  tests/test_standard_detection.py \
  tests/test_conversion_workflow_seams.py -q
```

Expected: all selected tests pass and existing channel assignment remains unchanged.

- [ ] **Step 6: Commit the metadata-writer checkpoint**

```bash
git add \
  synthesia2midi/synthesia2midi/midi_generator.py \
  synthesia2midi/synthesia2midi/workflows/conversion.py \
  tests/test_midi_generator.py \
  tests/test_color_family_channels.py
git commit -m "feat: embed MIDI channel color metadata"
```

---

### Task 3: Parse Metadata and Resolve Readable Rust Colors

**Files:**
- Create: `tools/midi_touchup_editor_rust/src/color_map.rs`
- Modify: `tools/midi_touchup_editor_rust/src/main.rs:1-35,314-344,2314-2340,2440-2620`
- Test: inline unit tests in `tools/midi_touchup_editor_rust/src/color_map.rs`

**Interfaces:**
- Consumes: exact text prefix and JSON schema from Task 2.
- Produces: `ChannelColorMap`, `parse_color_map_text(&[u8]) -> Option<ChannelColorMap>`, and `note_color(&ChannelColorMap, channel: u8, pitch: u8) -> Color32`.

- [ ] **Step 1: Add failing parser and color-policy tests**

Create `color_map.rs` with tests first:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_valid_version_one_color_map() {
        let parsed = parse_color_map_text(
            br#"Synthesia2MIDI:color-map:v1:{"channels":{"0":{"natural":[10,20,30],"sharp_flat":[4,8,12]},"3":{"natural":[200,100,40]}}}"#,
        )
        .expect("valid color map");

        assert_eq!(parsed.get(&0).unwrap().natural, Some([10, 20, 30]));
        assert_eq!(parsed.get(&0).unwrap().sharp_flat, Some([4, 8, 12]));
        assert_eq!(parsed.get(&3).unwrap().natural, Some([200, 100, 40]));
    }

    #[test]
    fn ignores_unknown_malformed_and_oversized_metadata() {
        assert!(parse_color_map_text(b"Synthesia2MIDI:color-map:v2:{}").is_none());
        assert!(parse_color_map_text(b"Synthesia2MIDI:color-map:v1:{bad").is_none());
        assert!(parse_color_map_text(&vec![b'x'; MAX_COLOR_MAP_META_BYTES + 1]).is_none());
    }

    #[test]
    fn fallback_palette_distinguishes_first_four_channels() {
        let colors: Vec<Color32> = (0..4)
            .map(|channel| note_color(&ChannelColorMap::new(), channel, 60))
            .collect();
        let unique: std::collections::HashSet<_> = colors.iter().copied().collect();
        assert_eq!(unique.len(), 4);
    }

    #[test]
    fn explicit_morphologies_and_derived_pair_are_related_but_distinct() {
        let mut map = ChannelColorMap::new();
        map.insert(
            0,
            ChannelColors {
                natural: Some([90, 180, 220]),
                sharp_flat: Some([40, 90, 120]),
            },
        );
        assert_ne!(note_color(&map, 0, 60), note_color(&map, 0, 61));

        map.get_mut(&0).unwrap().sharp_flat = None;
        assert_ne!(note_color(&map, 0, 60), note_color(&map, 0, 61));
    }
}
```

Declare `mod color_map;` in `main.rs` so Cargo compiles the module.

- [ ] **Step 2: Run Rust tests and verify RED**

```bash
cargo test --manifest-path tools/midi_touchup_editor_rust/Cargo.toml color_map
```

Expected: compile failures for the undefined module interfaces.

- [ ] **Step 3: Implement strict parsing and color resolution**

In `color_map.rs`, define:

```rust
use std::collections::BTreeMap;

use eframe::egui::Color32;
use serde::Deserialize;

pub(crate) const COLOR_MAP_META_PREFIX: &[u8] = b"Synthesia2MIDI:color-map:v1:";
pub(crate) const MAX_COLOR_MAP_META_BYTES: usize = 4096;

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct ChannelColors {
    pub(crate) natural: Option<[u8; 3]>,
    pub(crate) sharp_flat: Option<[u8; 3]>,
}

pub(crate) type ChannelColorMap = BTreeMap<u8, ChannelColors>;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ColorMapPayload {
    channels: BTreeMap<String, ChannelColors>,
}
```

`parse_color_map_text` must reject text over 4096 bytes, require the exact prefix, deserialize the remaining UTF-8 JSON, reject channel keys outside `0..15`, and return only non-empty channel entries. JSON arrays into `[u8; 3]` provide component-count and range validation.

Implement a 16-entry deterministic fallback palette whose first four colors are visibly distinct. Implement `note_color` with this order:

1. Explicit morphology color for the channel.
2. Missing Sharp / Flat: darken the Natural color.
3. Missing Natural: lighten the Sharp / Flat color.
4. Built-in channel color, darkened for Sharp / Flat pitches.

Only adapt colors whose relative luminance is below `0.18` or above `0.88`; blend minimally toward white or black and retain the original alpha. Keep `is_black_key(pitch)` in one place in this module.

- [ ] **Step 4: Store the parsed map on `MidiDocument`**

Import the module interfaces in `main.rs`:

```rust
mod color_map;

use color_map::{note_color, parse_color_map_text, ChannelColorMap};
```

Add:

```rust
channel_colors: ChannelColorMap,
```

to `MidiDocument`. In `load_midi_document`, initialize an empty map and, while iterating track events, accept only text events at absolute tick zero:

```rust
if absolute_tick == 0 {
    if let TrackEventKind::Meta(MetaMessage::Text(bytes)) = event.kind {
        if let Some(parsed) = parse_color_map_text(bytes) {
            channel_colors = parsed;
        }
    }
}
```

Assign the final valid map to `MidiDocument`. Do not remove the text event from `preserved_tracks`; the existing raw meta-event encoding must continue to save it.

Replace the old three-color `channel_base_color`/`note_color` implementation in `main.rs` with the module function and pass `&self.document.channel_colors` at both falling-bar and active-key call sites.

- [ ] **Step 5: Add a load/save preservation test**

In the existing `ui_policy_tests` module, construct a minimal `midly::Smf` with a tick-zero `MetaMessage::Text` event and `MetaMessage::EndOfTrack`. Use this test-path helper so no dependency is added:

```rust
fn unique_test_midi(name: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "synthesia2midi-{name}-{}-{nanos}.mid",
        std::process::id()
    ))
}
```

Build the fixture with `Header::new(Format::SingleTrack, Timing::Metrical(u15::new(480)))`, one `TrackEvent` containing the color payload, and a final EndOfTrack event. Save it with `smf.save(&source_path)`, call `load_midi_document`, save through `save_midi_document`, reload the returned path, and assert channel zero retains both morphology colors. Remove both source and output paths at test end.

- [ ] **Step 6: Run and format Rust tests**

```bash
cargo fmt --manifest-path tools/midi_touchup_editor_rust/Cargo.toml --check
cargo test --manifest-path tools/midi_touchup_editor_rust/Cargo.toml
cargo check --manifest-path tools/midi_touchup_editor_rust/Cargo.toml
```

Expected: parser, preservation, existing UI policy tests, and compilation all pass.

- [ ] **Step 7: Commit the Rust metadata checkpoint**

```bash
git add \
  tools/midi_touchup_editor_rust/src/color_map.rs \
  tools/midi_touchup_editor_rust/src/main.rs
git commit -m "feat: render MIDI channel color metadata"
```

---

### Task 4: Keep Same-Pitch Channels Separately Visible

**Files:**
- Create: `tools/midi_touchup_editor_rust/src/channel_layout.rs`
- Modify: `tools/midi_touchup_editor_rust/src/main.rs:540-620,870-925,1086-1365`
- Test: inline unit tests in `tools/midi_touchup_editor_rust/src/channel_layout.rs`

**Interfaces:**
- Consumes: note IDs, pitches, channels, and tick intervals from `EditableNote`.
- Produces: `compute_lane_assignments(&[NoteSpan]) -> HashMap<u64, LaneAssignment>` and `active_channels_at_tick(&[NoteSpan], tick: u64) -> BTreeMap<u8, Vec<u8>>`.

- [ ] **Step 1: Write failing interval-layout tests**

Create `channel_layout.rs` with public crate-local data types and tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn span(id: u64, pitch: u8, channel: u8, start: u64, end: u64) -> NoteSpan {
        NoteSpan { note_id: id, pitch, channel, start_tick: start, end_tick: end }
    }

    #[test]
    fn overlapping_channels_receive_stable_channel_ordered_lanes() {
        let notes = vec![
            span(1, 60, 3, 0, 100),
            span(2, 60, 0, 20, 80),
            span(3, 60, 2, 40, 120),
        ];
        let lanes = compute_lane_assignments(&notes);

        assert_eq!(lanes[&2], LaneAssignment { index: 0, count: 3 });
        assert_eq!(lanes[&3], LaneAssignment { index: 1, count: 3 });
        assert_eq!(lanes[&1], LaneAssignment { index: 2, count: 3 });
    }

    #[test]
    fn same_channel_duplicates_share_one_visual_lane() {
        let notes = vec![span(1, 60, 1, 0, 100), span(2, 60, 1, 20, 80)];
        let lanes = compute_lane_assignments(&notes);
        assert_eq!(lanes[&1], LaneAssignment { index: 0, count: 1 });
        assert_eq!(lanes[&2], LaneAssignment { index: 0, count: 1 });
    }

    #[test]
    fn active_channels_are_sorted_unique_per_pitch() {
        let notes = vec![
            span(1, 60, 3, 0, 100),
            span(2, 60, 0, 0, 100),
            span(3, 60, 3, 10, 90),
        ];
        assert_eq!(active_channels_at_tick(&notes, 50)[&60], vec![0, 3]);
    }
}
```

- [ ] **Step 2: Run the module tests and verify RED**

```bash
cargo test --manifest-path tools/midi_touchup_editor_rust/Cargo.toml channel_layout
```

Expected: compile failure until the module is declared and implemented.

- [ ] **Step 3: Implement overlap components and active-channel sets**

Define:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct NoteSpan {
    pub(crate) note_id: u64,
    pub(crate) pitch: u8,
    pub(crate) channel: u8,
    pub(crate) start_tick: u64,
    pub(crate) end_tick: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LaneAssignment {
    pub(crate) index: usize,
    pub(crate) count: usize,
}
```

For `compute_lane_assignments`:

1. Group spans by pitch in a `BTreeMap`.
2. Sort each pitch by `(start_tick, end_tick, channel, note_id)`.
3. Form connected overlap components while the next `start_tick` is strictly less than the component's maximum `end_tick`.
4. Sort the component's unique channels.
5. Assign every note the index of its channel and the unique-channel count.

For `active_channels_at_tick`, filter `start_tick <= tick && tick < end_tick`, gather channels in a `BTreeSet` per pitch, and return sorted vectors.

- [ ] **Step 4: Cache lanes and apply them to falling bars**

Add `mod channel_layout;` and an app field:

```rust
note_lanes: HashMap<u64, LaneAssignment>,
```

Add a helper that maps every `EditableNote` to `NoteSpan` and recomputes the cache. Call it after opening a MIDI and after `push_command`, Undo, and Redo. Do not recompute inside each paint loop.

In `note_rect`, after obtaining the piano-key `x` and width, apply the cached lane:

```rust
let lane = self.note_lanes.get(&note.note_id).copied().unwrap_or(
    LaneAssignment { index: 0, count: 1 },
);
let lane_width = w / lane.count.max(1) as f32;
let lane_x = x + lane.index as f32 * lane_width;
```

Use `lane_x` and `lane_width` for the rectangle. Preserve the existing outline and selected-note treatment.

- [ ] **Step 5: Split active piano keys by distinct channel**

Replace `active_note_channels_at_playhead() -> HashMap<u8, u8>` with a sorted multi-channel map built through `active_channels_at_tick`.

For each key rectangle, one channel fills the entire key. For multiple channels, divide the rectangle width by channel count and paint equal vertical bands in sorted channel order. Draw the key's existing outer border after all bands so subdivisions do not erase the piano outline.

- [ ] **Step 6: Run Rust tests and checks**

```bash
cargo fmt --manifest-path tools/midi_touchup_editor_rust/Cargo.toml --check
cargo test --manifest-path tools/midi_touchup_editor_rust/Cargo.toml
cargo check --manifest-path tools/midi_touchup_editor_rust/Cargo.toml
```

Expected: all tests pass, including color-map and channel-layout modules.

- [ ] **Step 7: Commit the overlap-rendering checkpoint**

```bash
git add \
  tools/midi_touchup_editor_rust/src/channel_layout.rs \
  tools/midi_touchup_editor_rust/src/main.rs
git commit -m "feat: show overlapping MIDI channel colors"
```

---

### Task 5: Add Atomic Whole-File Octave Editing

**Files:**
- Modify: `tools/midi_touchup_editor_rust/src/main.rs:170-181,355-363,540-609,870-925,1498-1660`
- Test: `tools/midi_touchup_editor_rust/src/main.rs` `ui_policy_tests`

**Interfaces:**
- Consumes: `EditableNote`, existing `EditCommand`, Undo/Redo stack, audio refresh, `MIN_PITCH=21`, and `MAX_PITCH=108`.
- Produces: `plan_octave_shift(...)`, `MidiTouchupApp.apply_octave_shift(delta_octaves: i8) -> Result<(), OctaveShiftBlock>`, and toolbar state `octave_offset: i8`.

- [ ] **Step 1: Write failing octave-policy tests**

Add pure tests for planning:

```rust
#[test]
fn octave_plan_moves_every_note_by_twelve() {
    let notes = vec![(1_u64, 48_u8), (2, 72)];
    assert_eq!(
        plan_octave_shift(notes.into_iter(), 1).unwrap(),
        vec![PitchChange { note_id: 1, before: 48, after: 60 },
             PitchChange { note_id: 2, before: 72, after: 84 }]
    );
}

#[test]
fn octave_plan_rejects_entire_shift_at_piano_bounds() {
    assert_eq!(
        plan_octave_shift([(1_u64, 21_u8)].into_iter(), -1),
        Err(OctaveShiftBlock::BelowPiano { pitch: 21 })
    );
    assert_eq!(
        plan_octave_shift([(1_u64, 108_u8)].into_iter(), 1),
        Err(OctaveShiftBlock::AbovePiano { pitch: 108 })
    );
}
```

Refactor app construction into `from_document(...)` so tests can create an app with `audio_engine=None`. Add an app-level test:

```rust
#[test]
fn octave_shift_is_one_undoable_redoable_command() {
    let mut app = test_app_with_pitches(&[48, 72]);

    app.apply_octave_shift(1).unwrap();
    assert_eq!(app.document.notes.iter().map(|n| n.pitch).collect::<Vec<_>>(), vec![60, 84]);
    assert_eq!(app.octave_offset, 1);
    assert_eq!(app.undo_stack.len(), 1);

    app.undo();
    assert_eq!(app.document.notes.iter().map(|n| n.pitch).collect::<Vec<_>>(), vec![48, 72]);
    assert_eq!(app.octave_offset, 0);

    app.redo();
    assert_eq!(app.document.notes.iter().map(|n| n.pitch).collect::<Vec<_>>(), vec![60, 84]);
    assert_eq!(app.octave_offset, 1);
}
```

Add a blocked app-level test asserting pitches, dirty state, Undo length, and offset remain unchanged.

- [ ] **Step 2: Run the Rust tests and verify RED**

```bash
cargo test --manifest-path tools/midi_touchup_editor_rust/Cargo.toml octave
```

Expected: compile failures for missing pitch-change and app methods.

- [ ] **Step 3: Add the bulk command model**

Define:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PitchChange {
    note_id: u64,
    before: u8,
    after: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OctaveShiftBlock {
    BelowPiano { pitch: u8 },
    AbovePiano { pitch: u8 },
}
```

Add to `EditCommand`:

```rust
Transpose {
    changes: Vec<PitchChange>,
    delta_octaves: i8,
},
```

`plan_octave_shift` must accept only `-1` or `1`, calculate every result before returning, reject the complete plan if one result is below 21 or above 108, and return changes in note iteration order.

Extend `apply_command` so forward application uses `after` and increments `octave_offset`; reverse uses `before` and decrements it. Update all notes by ID before the existing dirty/recompute/audio-refresh path runs.

- [ ] **Step 4: Add the app operation and bounded warning**

Add `octave_offset: i8`, reset it to zero in `load_new_midi`, and implement:

```rust
fn apply_octave_shift(&mut self, delta_octaves: i8) -> Result<(), OctaveShiftBlock> {
    let changes = plan_octave_shift(
        self.document.notes.iter().map(|note| (note.note_id, note.pitch)),
        delta_octaves,
    )?;
    if changes.is_empty() {
        return Ok(());
    }
    self.set_playing(false);
    self.push_command(EditCommand::Transpose { changes, delta_octaves });
    self.status_line = format!("Octave adjustment: {:+}", self.octave_offset);
    Ok(())
}
```

Add `midi_pitch_label(pitch) -> String` using pitch classes and `(pitch / 12) - 1`. The toolbar wrapper catches `OctaveShiftBlock` and shows one short `MessageDialog` explaining that all notes must remain between A0 and C8 and naming the blocking note. A rejection must not call `push_command`.

- [ ] **Step 5: Add the compact toolbar control**

Place the control in the existing `horizontal_wrapped` toolbar before Undo/Redo:

```rust
ui.label(RichText::new("Octave").size(control_font_size - 2.0));
if ui.small_button("-").on_hover_text("Shift every note down one octave").clicked() {
    self.request_octave_shift(-1);
}
ui.label(RichText::new(format!("{:+}", self.octave_offset)).size(control_font_size));
if ui.small_button("+").on_hover_text("Shift every note up one octave").clicked() {
    self.request_octave_shift(1);
}
```

Display `0` without a plus sign at the baseline; display `+1`, `+2`, etc. above it. Keep fixed button dimensions so the toolbar does not shift when the value changes.

- [ ] **Step 6: Run all Rust verification**

```bash
cargo fmt --manifest-path tools/midi_touchup_editor_rust/Cargo.toml
cargo fmt --manifest-path tools/midi_touchup_editor_rust/Cargo.toml --check
cargo test --manifest-path tools/midi_touchup_editor_rust/Cargo.toml
cargo check --manifest-path tools/midi_touchup_editor_rust/Cargo.toml
```

Expected: all parser, overlap, octave, and existing UI-policy tests pass.

- [ ] **Step 7: Commit the octave checkpoint**

```bash
git add tools/midi_touchup_editor_rust/src/main.rs
git commit -m "feat: add whole-file octave touch-up"
```

---

### Task 6: Refresh Localization and Run Rust Tests in CI

**Files:**
- Modify: `synthesia2midi/synthesia2midi/translations/synthesia2midi_{es,ja,ru,zh_CN,ko,pt_BR}.ts`
- Modify: matching `.qm` files
- Modify: `docs/localization/ui-string-manifest.json`
- Modify: `docs/localization/translation-agent-packet.json`
- Modify: `tests/test_localization.py`
- Modify: `.github/workflows/ci.yml:62-75`
- Modify: `docs/testing.md:136-140`
- Modify: `ARCHITECTURE.MD:143-159`

**Interfaces:**
- Consumes: the two new/renamed Qt source strings and all Rust unit tests from Tasks 3-5.
- Produces: complete Qt catalogs, deterministic audit artifacts, and a CI Rust gate that executes tests before checking compilation.

- [ ] **Step 1: Add localization expectations before regenerating assets**

Add these source strings to the production-locale expectation list in `tests/test_localization.py`:

```python
TOUCHUP_ACCESS_PRODUCTION_LOCALE_STRINGS = [
    "Open MIDI in Touch-Up Editor...",
    "Open Touch-Up Editor",
]
```

Include this list in the existing per-locale source/translation completeness assertion.

- [ ] **Step 2: Run localization tests and verify RED**

```bash
.venv/bin/python -m pytest tests/test_localization.py tests/test_ui_string_audit.py -q
```

Expected: stale catalogs/manifest and missing production-locale source strings.

- [ ] **Step 3: Regenerate TS catalogs and supply translations**

Run:

```bash
.venv/bin/pyside6-lupdate -extensions py \
  synthesia2midi/synthesia2midi \
  -ts \
  synthesia2midi/synthesia2midi/translations/synthesia2midi_es.ts \
  synthesia2midi/synthesia2midi/translations/synthesia2midi_ja.ts \
  synthesia2midi/synthesia2midi/translations/synthesia2midi_ru.ts \
  synthesia2midi/synthesia2midi/translations/synthesia2midi_zh_CN.ts \
  synthesia2midi/synthesia2midi/translations/synthesia2midi_ko.ts \
  synthesia2midi/synthesia2midi/translations/synthesia2midi_pt_BR.ts
```

Then populate the two strings with these reviewed first-pass translations:

| Locale | Open MIDI in Touch-Up Editor... | Open Touch-Up Editor |
|---|---|---|
| `es` | `Abrir MIDI en el Editor de retoque...` | `Abrir Editor de retoque` |
| `ja` | `MIDIをタッチアップエディターで開く...` | `Touch-Up Editorを開く` |
| `ru` | `Открыть MIDI в редакторе правки...` | `Открыть редактор правки` |
| `zh_CN` | `在微调编辑器中打开 MIDI...` | `打开微调编辑器` |
| `ko` | `수정 편집기에서 MIDI 열기...` | `수정 편집기 열기` |
| `pt_BR` | `Abrir MIDI no Editor de Retoques...` | `Abrir Editor de Retoques` |

Reuse the existing `Open Touch-Up Editor` translation from the `MidiTouchupController` context where available. Remove only obsolete `Edit MIDI` entries produced by lupdate; do not manually reorder unrelated catalog content.

- [ ] **Step 4: Compile every QM and regenerate deterministic audit outputs**

```bash
for ts_file in synthesia2midi/synthesia2midi/translations/synthesia2midi_*.ts; do
  locale_name=$(basename "$ts_file" .ts | sed 's/^synthesia2midi_//')
  .venv/bin/pyside6-lrelease "$ts_file" \
    -qm "synthesia2midi/synthesia2midi/translations/synthesia2midi_${locale_name}.qm"
done
.venv/bin/python -m synthesia2midi.tools.audit_ui_strings \
  --output docs/localization/ui-string-manifest.json
.venv/bin/python -m synthesia2midi.tools.export_translation_packet \
  --source-ts synthesia2midi/synthesia2midi/translations/synthesia2midi_es.ts \
  --output docs/localization/translation-agent-packet.json
```

- [ ] **Step 5: Upgrade the Rust CI and documented gate**

In `.github/workflows/ci.yml`, change the Rust step to:

```yaml
- name: Test MIDI touch-up editor
  run: cargo test --manifest-path tools/midi_touchup_editor_rust/Cargo.toml

- name: Check MIDI touch-up editor
  run: cargo check --manifest-path tools/midi_touchup_editor_rust/Cargo.toml
```

In `docs/testing.md`, make the canonical Rust gate:

```bash
cargo fmt --manifest-path tools/midi_touchup_editor_rust/Cargo.toml --check
cargo test --manifest-path tools/midi_touchup_editor_rust/Cargo.toml
cargo check --manifest-path tools/midi_touchup_editor_rust/Cargo.toml
```

Update `ARCHITECTURE.MD` to state that generated MIDI may contain a namespaced versioned channel-color text event, the Rust loader treats it as optional, and the event remains the portable Python-to-Rust color contract.

- [ ] **Step 6: Run localization and Rust gates**

```bash
.venv/bin/python -m pytest tests/test_localization.py tests/test_ui_string_audit.py -q
cargo fmt --manifest-path tools/midi_touchup_editor_rust/Cargo.toml --check
cargo test --manifest-path tools/midi_touchup_editor_rust/Cargo.toml
cargo check --manifest-path tools/midi_touchup_editor_rust/Cargo.toml
```

Expected: all commands pass with no unfinished/empty translations or stale manifest entries.

- [ ] **Step 7: Commit localization and CI hardening**

```bash
git add \
  .github/workflows/ci.yml \
  ARCHITECTURE.MD \
  docs/testing.md \
  docs/localization/ui-string-manifest.json \
  docs/localization/translation-agent-packet.json \
  synthesia2midi/synthesia2midi/translations \
  tests/test_localization.py
git commit -m "test: gate touch-up colors and octave editing"
```

---

### Task 7: Integrated Verification, Review, and Task Closure

**Files:**
- Modify: `backlog/tasks/task-23 - Improve-touch-up-editor-access-octave-and-channel-colors.md`
- Modify only if verification finds defects: files owned by Tasks 1-6.

**Interfaces:**
- Consumes: all completed checkpoints and the approved design.
- Produces: review evidence, a locally smoke-tested Apple Silicon package, a clean branch ready for Jeff-authorized push, and eventual Windows/macOS remote evidence before TASK-23 is marked Done.

- [ ] **Step 1: Run focused Python integration tests**

```bash
.venv/bin/python -m pytest \
  tests/test_midi_touchup_controller.py \
  tests/test_midi_generator.py \
  tests/test_color_family_channels.py \
  tests/test_controls_qt.py \
  tests/test_main_window_layout.py \
  tests/test_startup_dialog.py \
  tests/test_localization.py \
  tests/test_ui_string_audit.py -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run the complete local source gate**

```bash
git diff --check
.venv/bin/python -m compileall -q synthesia2midi
.venv/bin/python -m ruff check synthesia2midi tests --select=E9,F63,F7,F82
.venv/bin/python -m pytest
cargo fmt --manifest-path tools/midi_touchup_editor_rust/Cargo.toml --check
cargo test --manifest-path tools/midi_touchup_editor_rust/Cargo.toml
cargo check --manifest-path tools/midi_touchup_editor_rust/Cargo.toml
```

Expected: zero failures. Existing Qt/Rust warnings must be reported but are not failures unless newly introduced by this branch.

- [ ] **Step 3: Run deterministic UI and metadata probes**

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m synthesia2midi.tools.render_ui_matrix \
  --locale qps \
  --font-scale 1.5 \
  --output logs/ux-audit/touchup-access
```

Create a four-channel synthetic MIDI through `MidiWriter`, inspect its text event, load it with a Rust unit/integration fixture, and verify all four channel colors remain distinct. Keep probe output under ignored `logs/`; do not commit generated MIDI or screenshots.

- [ ] **Step 4: Build and smoke-launch the local Apple Silicon package**

```bash
.venv/bin/python packaging/build_release.py --version v0.2.1-dev
```

Expected: `dist/release/Synthesia2MIDI-macos-arm64-v0.2.1-dev.zip` is created and the packaged startup smoke exits successfully. Inspect the archive for the Rust editor, SoundFont, translations, FFmpeg, app icon, and build version.

- [ ] **Step 5: Request two-stage code review and fix concrete findings**

Use `superpowers:requesting-code-review` for:

1. Spec compliance against `docs/superpowers/specs/2026-07-12-touch-up-editor-access-octave-colors-design.md`.
2. Code quality, metadata compatibility, rendering performance, Undo/Redo integrity, and missing tests.

Apply only verified findings. Rerun the focused gate for every fix and the complete gate after the final fix.

- [ ] **Step 6: Commit final verified repairs and record local evidence**

Update TASK-23 implementation notes with exact test counts, Rust results, package archive name, and any residual limitations. Leave status `In Progress` until remote Windows/macOS jobs pass.

```bash
git add \
  .github/workflows/ci.yml \
  ARCHITECTURE.MD \
  docs/testing.md \
  docs/localization/ui-string-manifest.json \
  docs/localization/translation-agent-packet.json \
  'backlog/tasks/task-23 - Improve-touch-up-editor-access-octave-and-channel-colors.md' \
  synthesia2midi/synthesia2midi/main.py \
  synthesia2midi/synthesia2midi/midi_generator.py \
  synthesia2midi/synthesia2midi/workflows/conversion.py \
  synthesia2midi/synthesia2midi/gui/controls_qt.py \
  synthesia2midi/synthesia2midi/gui/midi_touchup_controller.py \
  synthesia2midi/synthesia2midi/translations \
  tests/test_color_family_channels.py \
  tests/test_controls_qt.py \
  tests/test_localization.py \
  tests/test_main_window_layout.py \
  tests/test_midi_generator.py \
  tests/test_midi_touchup_controller.py \
  tests/test_startup_dialog.py \
  tools/midi_touchup_editor_rust/src/channel_layout.rs \
  tools/midi_touchup_editor_rust/src/color_map.rs \
  tools/midi_touchup_editor_rust/src/main.rs
git commit -m "fix: close touch-up editor review findings"
```

If review produces no code changes, commit only the Backlog evidence as `docs: record touch-up editor verification`.

- [ ] **Step 7: Stop at the push gate**

Report the branch, commits, local Python/Rust results, and local package smoke to Jeff. Do not push until he explicitly says to push.

After Jeff authorizes a push:

```bash
git push -u origin codex/touchup-editor-access-octave-colors
```

Wait for the GitHub Actions Python and Rust matrices on Windows and macOS. If either platform fails, use `superpowers:systematic-debugging`, fix on the same branch, rerun local gates, commit, and push only under the existing authorization.

- [ ] **Step 8: Complete the package/release gate when Jeff authorizes a version**

Merge only after the branch and resulting `main` CI matrices pass. When Jeff authorizes the next version tag, use the existing tag-driven Release workflow and confirm both `build-and-upload (windows-latest)` and `build-and-upload (macos-14)` succeed with versioned and `latest` archives.

Only then check all TASK-23 acceptance criteria, set `status: Done`, record the GitHub run IDs and release asset names, and commit the final Backlog closure.
