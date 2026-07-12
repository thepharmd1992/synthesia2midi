# Four Color Families Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Support one through four detected note-color families, with Natural and Sharp / Flat exemplars for each family and deterministic MIDI channel 1-4 export.

**Architecture:** Introduce one canonical family registry used by state, UI, scanning, detection, persistence, and conversion. Keep assisted calibration as a two-stage pipeline: cheap discovery every 10 frames across the video, followed by targeted histogram refinement only for stable family/morphology events. Standard detection records the winning exemplar slot for every pressed key, Spark delegates that identity, and conversion maps the winning slot directly to the family's MIDI channel.

**Tech Stack:** Python 3, PySide6/Qt, OpenCV, NumPy, pytest, Qt Linguist (`pyside6-lupdate` and `pyside6-lrelease`), existing PyInstaller packaging and GitHub Actions smoke workflows.

## Global Constraints

- Support at most four color families and eight lit exemplars.
- Preserve `LW`, `LB`, `RW`, and `RB` as internal compatibility keys; do not rename or reinterpret old INI/JSON data.
- Use `COLOR_3_W`, `COLOR_3_B`, `COLOR_4_W`, and `COLOR_4_B` for the new internal slots.
- User-visible terminology is `Color 1` through `Color 4`, `Natural`, and `Sharp / Flat`; do not show Left/Right or legacy slot identifiers in this feature's new UI.
- Color 1 is always available. Show other families only when enabled, detected, or already saved.
- Preserve the existing 10-frame coarse scan interval.
- Lightweight discovery continues after two complete families; detailed refinement becomes quiescent until new useful evidence appears.
- Stop all scanning once four stable families each have confirmed Natural and Sharp / Flat evidence.
- Colors 1-4 map to MIDI channels 1-4; internal channel values remain 0-3.
- Cancellation and failed rescans must restore enabled flags, colors, histograms, and channel-assignment behavior exactly.
- Do not commit the reference YouTube video, extracted frames, MIDI output, logs, `.venv`, Rust `target/`, or `uv.lock`.
- Update all production locales (`es`, `ja`, `ru`, `zh_CN`, `ko`, `pt_BR`) and pseudo-locale coverage after source copy stabilizes.
- Release targets remain Windows x64 and Apple Silicon macOS. Do not tag `v0.2.0` in this plan.
- Work on `codex/four-color-families` without creating a worktree and do not push unless Jeff explicitly requests it.

---

### Task 1: Canonical Color-Family Registry and State

**Files:**
- Create: `synthesia2midi/synthesia2midi/core/color_families.py`
- Modify: `synthesia2midi/synthesia2midi/core/app_state.py`
- Test: `tests/test_color_families.py`
- Create: `tests/test_app_state.py`

**Interfaces:**
- Produces: `ColorFamilyDefinition`, `COLOR_FAMILIES`, `SUPPORTED_EXEMPLAR_SLOTS`, `family_for_slot()`, `slots_for_family()`, `morphology_for_slot()`, `exemplar_display_parts()`, and `active_family_numbers()`.
- Produces: `DetectionConfig.get_required_exemplar_types() -> list[str]` while retaining `get_required_base_exemplar_types()` as a compatibility alias.
- Consumed by every later task; no later file may define a second slot-to-family mapping.

- [ ] **Step 1: Write failing registry and state tests**

```python
from synthesia2midi.core.app_state import DetectionConfig
from synthesia2midi.core.color_families import (
    SUPPORTED_EXEMPLAR_SLOTS,
    active_family_numbers,
    exemplar_display_parts,
    family_for_slot,
    morphology_for_slot,
    slots_for_family,
)


def test_color_family_registry_preserves_legacy_slots_and_channels():
    assert SUPPORTED_EXEMPLAR_SLOTS == (
        "LW", "LB", "RW", "RB",
        "COLOR_3_W", "COLOR_3_B", "COLOR_4_W", "COLOR_4_B",
    )
    assert slots_for_family(1) == ("LW", "LB")
    assert slots_for_family(4) == ("COLOR_4_W", "COLOR_4_B")
    assert family_for_slot("COLOR_3_B").midi_channel == 2
    assert morphology_for_slot("RW") == "natural"
    assert morphology_for_slot("RB") == "accidental"
    assert exemplar_display_parts("COLOR_4_B") == (4, "Sharp / Flat")


def test_detection_config_defaults_new_families_off_and_masks_them():
    config = DetectionConfig()
    assert config.exemplar_key_type_enabled["LW"] is True
    assert config.exemplar_key_type_enabled["COLOR_3_W"] is False
    config.exemplar_lit_colors["COLOR_3_W"] = (12, 34, 56)
    assert config.get_effective_exemplar_lit_colors()["COLOR_3_W"] is None
    config.exemplar_key_type_enabled["COLOR_3_W"] = True
    assert config.get_required_exemplar_types() == ["LW", "LB", "RW", "RB", "COLOR_3_W"]


def test_active_families_include_enabled_or_saved_slots_but_always_color_one():
    enabled = {slot: False for slot in SUPPORTED_EXEMPLAR_SLOTS}
    colors = {slot: None for slot in SUPPORTED_EXEMPLAR_SLOTS}
    colors["COLOR_4_W"] = (1, 2, 3)
    assert active_family_numbers(enabled, colors) == (1, 4)
```

- [ ] **Step 2: Run the tests and verify they fail for missing registry/new slots**

Run: `.venv/bin/python -m pytest tests/test_color_families.py tests/test_app_state.py -q`

Expected: FAIL because `synthesia2midi.core.color_families` and the eight-slot state do not exist.

- [ ] **Step 3: Add the canonical registry**

```python
from dataclasses import dataclass
from typing import Literal, Mapping

Morphology = Literal["natural", "accidental"]


@dataclass(frozen=True)
class ColorFamilyDefinition:
    number: int
    natural_slot: str
    accidental_slot: str
    midi_channel: int


COLOR_FAMILIES = (
    ColorFamilyDefinition(1, "LW", "LB", 0),
    ColorFamilyDefinition(2, "RW", "RB", 1),
    ColorFamilyDefinition(3, "COLOR_3_W", "COLOR_3_B", 2),
    ColorFamilyDefinition(4, "COLOR_4_W", "COLOR_4_B", 3),
)
SUPPORTED_EXEMPLAR_SLOTS = tuple(
    slot
    for family in COLOR_FAMILIES
    for slot in (family.natural_slot, family.accidental_slot)
)


def family_for_slot(slot: str) -> ColorFamilyDefinition | None:
    return next(
        (family for family in COLOR_FAMILIES if slot in (family.natural_slot, family.accidental_slot)),
        None,
    )


def slots_for_family(number: int) -> tuple[str, str]:
    family = next(family for family in COLOR_FAMILIES if family.number == number)
    return family.natural_slot, family.accidental_slot


def morphology_for_slot(slot: str) -> Morphology | None:
    family = family_for_slot(slot)
    if family is None:
        return None
    return "natural" if slot == family.natural_slot else "accidental"


def exemplar_display_parts(slot: str) -> tuple[int, str]:
    family = family_for_slot(slot)
    if family is None:
        raise ValueError(f"Unsupported exemplar slot: {slot}")
    label = "Natural" if slot == family.natural_slot else "Sharp / Flat"
    return family.number, label


def active_family_numbers(
    enabled: Mapping[str, bool], colors: Mapping[str, object]
) -> tuple[int, ...]:
    active = {1}
    for family in COLOR_FAMILIES:
        slots = (family.natural_slot, family.accidental_slot)
        if any(enabled.get(slot, False) or colors.get(slot) is not None for slot in slots):
            active.add(family.number)
    return tuple(sorted(active))
```

- [ ] **Step 4: Expand `DetectionConfig` to initialize and mask all eight slots**

Use `SUPPORTED_EXEMPLAR_SLOTS` for color, histogram, and enabled dictionaries. Enable the four legacy slots by default and disable Color 3/4. Implement:

```python
def get_required_exemplar_types(self) -> list[str]:
    return [
        slot
        for slot in SUPPORTED_EXEMPLAR_SLOTS
        if self.exemplar_key_type_enabled.get(slot, False)
    ]

def get_required_base_exemplar_types(self) -> list[str]:
    return self.get_required_exemplar_types()
```

Update effective color/histogram getters to iterate `SUPPORTED_EXEMPLAR_SLOTS`, preserving unknown dynamic entries only if existing callers rely on them.

- [ ] **Step 5: Run focused tests**

Run: `.venv/bin/python -m pytest tests/test_color_families.py tests/test_app_state.py tests/test_bugfix_regressions.py -q`

Expected: PASS.

- [ ] **Step 6: Commit the registry and state slice**

```bash
git add synthesia2midi/synthesia2midi/core/color_families.py synthesia2midi/synthesia2midi/core/app_state.py tests/test_color_families.py tests/test_app_state.py
git commit -m "feat: add canonical color family model"
```

### Task 2: Dynamic Persistence With Old-File Compatibility

**Files:**
- Modify: `synthesia2midi/synthesia2midi/config_manager.py`
- Test: `tests/test_config_manager.py`

**Interfaces:**
- Consumes: `SUPPORTED_EXEMPLAR_SLOTS` from Task 1.
- Produces: an `[ExemplarEnabled]` INI section containing all eight supported flags while continuing to read and write the four legacy `exemplar_enabled_*` settings.

- [ ] **Step 1: Write failing old-file and round-trip tests**

Add tests using the existing ConfigManager fixtures that assert: an old INI without `[ExemplarEnabled]` preserves legacy LW/LB/RW/RB flags and defaults Color 3/4 off; a new save/reload round-trips `COLOR_3_W` and `COLOR_4_B` enabled flags, RGB tuples, and NumPy histograms exactly; and an invalid Color 3 sample is ignored while its valid sibling survives and remains enabled for normal readiness guidance.

- [ ] **Step 2: Run the new tests and verify the dynamic round trip fails**

Run: `.venv/bin/python -m pytest tests/test_config_manager.py -k "old_ini_without_dynamic or four_family_enabled" -q`

Expected: the old-file test passes or exposes a compatibility regression; the four-family test FAILS because enabled flags are hard-coded to four slots.

- [ ] **Step 3: Add dynamic enabled-flag loading and saving**

Load legacy flags first. Then, if `[ExemplarEnabled]` exists, apply only recognized keys:

```python
for slot in SUPPORTED_EXEMPLAR_SLOTS:
    if parser.has_option("ExemplarEnabled", slot):
        detection.exemplar_key_type_enabled[slot] = parser.getboolean(
            "ExemplarEnabled", slot
        )
```

On save, retain the existing four legacy settings and add:

```python
parser["ExemplarEnabled"] = {
    slot: str(bool(detection.exemplar_key_type_enabled.get(slot, False))).lower()
    for slot in SUPPORTED_EXEMPLAR_SLOTS
}
```

Continue using the existing dynamic color/histogram serialization; do not rewrite unrelated sections.

- [ ] **Step 4: Run persistence tests**

Run: `.venv/bin/python -m pytest tests/test_config_manager.py -q`

Expected: PASS.

- [ ] **Step 5: Commit persistence compatibility**

```bash
git add synthesia2midi/synthesia2midi/config_manager.py tests/test_config_manager.py
git commit -m "feat: persist four color family calibration"
```

### Task 3: Stable Four-Family Assignment Engine

**Files:**
- Create: `synthesia2midi/detection/color_family_assignment.py`
- Modify: `synthesia2midi/detection/assisted_calibration.py`
- Create: `tests/test_color_family_assignment.py`
- Modify: `tests/test_assisted_calibration.py`

**Interfaces:**
- Consumes: family registry from Task 1.
- Produces: `FamilyEvidence`, `FamilyAssignment`, `cluster_family_evidence()`, and `assign_family_slots()`.
- Preserves: existing `ExemplarAssignmentResult` public result shape.

- [ ] **Step 1: Write failing synthetic assignment tests**

Create deterministic candidate fixtures for one, two, three, four, and five families. Assert all eight slots for four stable families; saved RGB anchors retain Color 1/2 identity when candidate order reverses; one isolated flash does not create a family; nearby hues merge; and five stable families retain only the four strongest plus warning `More than four stable color families were found.`

- [ ] **Step 2: Run the new assignment tests and verify import failure**

Run: `.venv/bin/python -m pytest tests/test_color_family_assignment.py -q`

Expected: FAIL because `color_family_assignment.py` does not exist.

- [ ] **Step 3: Implement focused evidence and assignment types**

```python
@dataclass(frozen=True)
class FamilyEvidence:
    frame_index: int
    key_id: int
    morphology: Literal["natural", "accidental"]
    rgb: tuple[int, int, int]
    score: float


@dataclass
class FamilyAssignment:
    family_number: int
    natural: FamilyEvidence | None = None
    accidental: FamilyEvidence | None = None
    confidence: float = 0.0

    @property
    def complete(self) -> bool:
        return self.natural is not None and self.accidental is not None
```

Implement circular HSV hue distance, saturation/value guards, temporal separation, distinct-key preference, and capped agglomerative clustering. A family becomes stable only with at least two temporally separated events; a second key increases confidence but is not mandatory because sparse videos may repeat only one key.

- [ ] **Step 4: Implement anchored deterministic slot assignment**

Match each stable cluster to the nearest morphology-compatible saved family anchor under the existing family-distance threshold. Assign unmatched clusters by deterministic first-evidence frame, then hue, then RGB tuple into the first unused family number. Return a warning when more than four stable clusters exist or evidence conflicts with two anchored identities.

- [ ] **Step 5: Replace the two-family cap in assisted calibration**

Update `assign_exemplar_slots()` to call `assign_family_slots()` and map each family through `slots_for_family()`. Remove the current two-family length cap and hard-coded `(LW, LB)/(RW, RB)` pairing. Preserve legacy result attributes/call signatures used by existing tests.

- [ ] **Step 6: Run assignment and existing assisted-calibration tests**

Run: `.venv/bin/python -m pytest tests/test_color_family_assignment.py tests/test_assisted_calibration.py -q`

Expected: PASS.

- [ ] **Step 7: Commit the assignment engine**

```bash
git add synthesia2midi/detection/color_family_assignment.py synthesia2midi/detection/assisted_calibration.py tests/test_color_family_assignment.py tests/test_assisted_calibration.py
git commit -m "feat: assign up to four stable color families"
```

### Task 4: Cheap Discovery and Targeted Refinement Scanner

**Files:**
- Modify: `synthesia2midi/detection/assisted_calibration.py`
- Modify: `tests/test_assisted_calibration.py`

**Interfaces:**
- Consumes: stable assignment engine from Task 3.
- Produces: optional `ExemplarScanDiagnostics` counters without changing the existing three-value return from `scan_lit_exemplar_candidates()`.

- [ ] **Step 1: Write failing scanner behavior and operation-count tests**

Add synthetic-provider tests proving: discovery continues past two complete families and finds a third at frame 900; a long two-family video checks every coarse checkpoint while `refined_frames < discovery_frames` and `refined_events <= 4`; a late third family raises refined events to six; and four complete families stop before a requested frame 5000 end.

- [ ] **Step 2: Run the four tests and verify diagnostics/early-stop failures**

Run: `.venv/bin/python -m pytest tests/test_assisted_calibration.py -k "continues_after_two or quiesces or reactivates or stop_before" -q`

Expected: FAIL because refinement currently runs around every coarse candidate and completion assumes two families.

- [ ] **Step 3: Add scan diagnostics without breaking callers**

```python
@dataclass
class ExemplarScanDiagnostics:
    discovery_frames: int = 0
    refined_frames: int = 0
    refined_events: int = 0


def scan_lit_exemplar_candidates(
    frame_provider,
    overlays,
    settings,
    *,
    progress_callback=None,
    cancel_requested=None,
    diagnostics: ExemplarScanDiagnostics | None = None,
) -> tuple[list[ExemplarCandidate], int, bool]:
    diagnostics = diagnostics or ExemplarScanDiagnostics()
    return _scan_candidates_with_diagnostics(
        frame_provider=frame_provider,
        overlays=overlays,
        settings=settings,
        progress_callback=progress_callback,
        cancel_requested=cancel_requested,
        diagnostics=diagnostics,
    )
```

Move the current scan loop into `_scan_candidates_with_diagnostics()` with the same inputs plus the required diagnostics object; Step 4 replaces that loop's refinement policy.

- [ ] **Step 4: Split lightweight discovery from detailed refinement**

At each existing 10-frame checkpoint, sample average RGB/HSV only and collapse sustained evidence from the same key/family into one event. Do not compute histograms in this pass. When repeated, temporally separated evidence stabilizes a missing family/morphology, refine only the best event within `frame_index ± settings.refine_radius`, incrementing `refined_frames` for each neighbor decoded and `refined_events` once per family/morphology.

Use this control rule:

```python
if len(stable_families) == 4 and all(family.complete for family in stable_families):
    break
if evidence_improves_missing_slot(stable_families, event):
    refined = refine_promising_event(event)
    keep_best_refined_candidate(refined)
```

Do not stop at two families. Once two are complete, avoid replacing their refined exemplars unless a new event materially improves score or corrects an anchored identity.

- [ ] **Step 5: Add animation, sustained-note, nearby-hue, and 10-frame regressions**

Use synthetic frames/providers, not the downloaded video. Assert one-frame intro flashes are rejected, sustained notes produce one evidence event, nearby hues merge, and an event landing on a coarse checkpoint remains discoverable.

- [ ] **Step 6: Run the scanner suite**

Run: `.venv/bin/python -m pytest tests/test_assisted_calibration.py tests/test_color_family_assignment.py -q`

Expected: PASS, including operation-count assertions.

- [ ] **Step 7: Commit scanner optimization**

```bash
git add synthesia2midi/detection/assisted_calibration.py tests/test_assisted_calibration.py
git commit -m "perf: target four-family exemplar refinement"
```

### Task 5: Reusable Compact Family Grid

**Files:**
- Create: `synthesia2midi/synthesia2midi/gui/color_family_grid.py`
- Create: `tests/test_color_family_grid.py`

**Interfaces:**
- Consumes: family registry from Task 1.
- Produces: `ExemplarRowWidgets` and `ColorFamilyGrid` for both settings and assisted review.

- [ ] **Step 1: Write failing offscreen widget tests**

Test that families `(1, 3)` create rows `LW`, `LB`, `COLOR_3_W`, `COLOR_3_B`; labels are Natural and Sharp / Flat; the Color 3 heading is correct; clicking Set emits the slot; and review mode replaces editable controls with Found/Missing status.

- [ ] **Step 2: Run tests and verify the component is missing**

Run: `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/test_color_family_grid.py -q`

Expected: FAIL on import.

- [ ] **Step 3: Implement the compact component**

```python
@dataclass
class ExemplarRowWidgets:
    label: QLabel
    swatch: QLabel
    set_button: QPushButton | None = None
    present: QCheckBox | None = None
    status: QLabel | None = None


class ColorFamilyGrid(QWidget):
    exemplar_requested = Signal(str)
    exemplar_enabled_changed = Signal(str, bool)
    family_add_requested = Signal()
    family_remove_requested = Signal(int)

    def __init__(self, *, mode: Literal["calibration", "review"], parent=None):
        super().__init__(parent)
        self.mode = mode
        self.rows: dict[str, ExemplarRowWidgets] = {}
        self._family_headings: dict[int, QLabel] = {}
        self._layout = QGridLayout(self)

    def set_families(
        self,
        family_numbers: Sequence[int],
        *,
        colors: Mapping[str, tuple[int, int, int] | None],
        enabled: Mapping[str, bool],
        assignments: Mapping[str, object] | None = None,
    ) -> None:
        self._rebuild_rows(
            tuple(family_numbers), colors, enabled, assignments or {}
        )
```

Implement `_rebuild_rows()` as the single owner of widget teardown/creation and signal binding, and `family_heading(number: int) -> QLabel` as the tested heading accessor.

Use a `QGridLayout` with one family heading and two fixed rows per family. Put label, swatch, Set button, and Present checkbox on the same row. Add a compact icon remove button only for families 2-4 and an `Add Color Family` button while fewer than four families are visible. Set sensible minimum widths so 150% font scaling wraps headings without horizontal scrolling.

- [ ] **Step 4: Add a 150% font-scale geometry test**

Set the application font to 1.5 times its default point size, show the grid offscreen, process events, and assert `grid.minimumSizeHint().width() <= 760` and every child geometry is contained within `grid.rect()` after resize to its size hint.

- [ ] **Step 5: Run component tests**

Run: `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/test_color_family_grid.py -q`

Expected: PASS.

- [ ] **Step 6: Commit the reusable grid**

```bash
git add synthesia2midi/synthesia2midi/gui/color_family_grid.py tests/test_color_family_grid.py
git commit -m "feat: add compact color family grid"
```

### Task 6: Replace Calibration and Review UIs

**Files:**
- Modify: `synthesia2midi/synthesia2midi/gui/controls_qt.py`
- Modify: `synthesia2midi/synthesia2midi/gui/assisted_calibration_dialog.py`
- Modify: `synthesia2midi/synthesia2midi/gui/main_action_controller.py`
- Modify: `synthesia2midi/synthesia2midi/gui/signal_manager.py`
- Test: `tests/test_controls_qt.py`
- Test: `tests/test_assisted_calibration_dialog.py`
- Test: `tests/test_main_action_controller.py`

**Interfaces:**
- Consumes: `ColorFamilyGrid` and `active_family_numbers()`.
- Produces: add/remove family actions capped at four, destructive confirmation, and a dynamic assisted-review presentation.

- [ ] **Step 1: Write failing calibration UI tests**

Assert the panel exposes `color_family_grid`; no `Set Left White` button remains; each row uses a short Set button; Add emits the existing add-family signal and selects the first unused family; removing calibrated Color 3 with a mocked No response retains all data; Color 1 has no remove action; Color 4 hides/disables Add; and a one-family saved configuration hides Color 2 without deleting its compatibility slots.

- [ ] **Step 2: Write failing review-dialog tests**

Build a proposal with complete Color 1 and only `COLOR_3_W`. Assert review shows Color 3 Natural Found and Sharp / Flat Missing, plus any scanner warning banner.

- [ ] **Step 3: Run the new UI tests and verify hard-coded layout failures**

Run: `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/test_controls_qt.py tests/test_assisted_calibration_dialog.py tests/test_main_action_controller.py -q`

Expected: FAIL because the current UI hard-codes four large Left/Right buttons.

- [ ] **Step 4: Embed `ColorFamilyGrid` in the calibration section**

Replace the four large buttons and Left/Right explanatory copy. Keep the existing dictionaries as aliases to the component's row widgets where needed to minimize controller churn. Refresh visible families from `active_family_numbers(enabled, colors)` whenever state changes.

- [ ] **Step 5: Wire Add/Remove through the existing signal manager**

Connect `add_additional_color_requested` and `remove_additional_color_requested` in `signal_manager.py`. Add controller handlers that enable the first unused family and clear an approved removed family's two colors, histograms, and flags. Color 1 cannot be removed; cap at Color 4. Ask for confirmation only when either slot has saved color/histogram data.

- [ ] **Step 6: Reuse the component in assisted-review mode**

Replace the review dialog's hard-coded `LW/LB/RW/RB` rows. Show all proposal families, including partial ones. Display a concise warning banner when assignment reports more than four stable families or anchored-family conflict.

- [ ] **Step 7: Run UI tests**

Run: `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/test_color_family_grid.py tests/test_controls_qt.py tests/test_assisted_calibration_dialog.py tests/test_main_action_controller.py -q`

Expected: PASS.

- [ ] **Step 8: Commit calibration UI changes**

```bash
git add synthesia2midi/synthesia2midi/gui/controls_qt.py synthesia2midi/synthesia2midi/gui/assisted_calibration_dialog.py synthesia2midi/synthesia2midi/gui/main_action_controller.py synthesia2midi/synthesia2midi/gui/signal_manager.py tests/test_controls_qt.py tests/test_assisted_calibration_dialog.py tests/test_main_action_controller.py
git commit -m "feat: show dynamic color family calibration rows"
```

### Task 7: Atomic Proposal Apply, Cancel, and Readiness

**Files:**
- Modify: `synthesia2midi/synthesia2midi/gui/calibration_wizard_controller.py`
- Modify: `synthesia2midi/synthesia2midi/gui/controls_qt.py`
- Modify: `synthesia2midi/synthesia2midi/workflows/conversion.py`
- Create: `tests/test_calibration_wizard_controller.py`
- Test: `tests/test_controls_qt.py`
- Test: `tests/test_conversion_workflow_seams.py`

**Interfaces:**
- Consumes: dynamic proposal assignments and family display helpers.
- Produces: full state rollback and dynamic readiness checks for every enabled slot.

- [ ] **Step 1: Write failing atomicity and readiness tests**

Assert cancel, review-window close, and a raised scan error each restore all eight color/histogram/enabled values and `hand_assignment_enabled`; accepted Color 3 enables separate assignment; and enabled `COLOR_4_B` with no color blocks conversion with `Color 4 Sharp / Flat` in the message.

- [ ] **Step 2: Run focused tests and verify dynamic-state failures**

Run: `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/test_calibration_wizard_controller.py tests/test_controls_qt.py tests/test_conversion_workflow_seams.py -q`

Expected: FAIL because summaries/readiness are four-slot hard-coded and assignment flag is not part of rollback.

- [ ] **Step 3: Expand proposal snapshot and restore**

Snapshot deep copies of all supported colors, histograms, enabled flags, and `hand_assignment_enabled` before review. Apply assignments only after Yes. Restore the snapshot on No, window close, cancellation, or scan exception. When a proposal contains family 3 or 4, set `hand_assignment_enabled = True` only as part of accepted apply.

- [ ] **Step 4: Generalize proposal summary and readiness**

Build labels through `exemplar_display_parts()` and Qt translation calls. Use `get_required_exemplar_types()` in both `ControlPanelQt._conversion_readiness()` and conversion preflight. A checked Present slot with no valid color blocks conversion; an unchecked slot does not.

- [ ] **Step 5: Run controller/readiness tests**

Run: `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/test_calibration_wizard_controller.py tests/test_controls_qt.py tests/test_conversion_workflow_seams.py -q`

Expected: PASS.

- [ ] **Step 6: Commit atomic proposal and readiness behavior**

```bash
git add synthesia2midi/synthesia2midi/gui/calibration_wizard_controller.py synthesia2midi/synthesia2midi/gui/controls_qt.py synthesia2midi/synthesia2midi/workflows/conversion.py tests/test_calibration_wizard_controller.py tests/test_controls_qt.py tests/test_conversion_workflow_seams.py
git commit -m "feat: validate dynamic color family calibration"
```

### Task 8: Winning-Family Detection Contract

**Files:**
- Modify: `synthesia2midi/synthesia2midi/detection/base.py`
- Modify: `synthesia2midi/synthesia2midi/detection/standard.py`
- Modify: `synthesia2midi/synthesia2midi/detection/spark_integrated.py`
- Create: `tests/test_standard_detection.py`
- Test: `tests/test_bugfix_regressions.py`

**Interfaces:**
- Produces: `DetectionMethod.get_last_exemplar_match(key_id: int) -> str | None`.
- Standard detection records only matches for keys surviving all filters/delta logic.
- Spark delegates winning family identity to its `standard_detector` and returns `None` for a key suppressed by Spark splitting.

- [ ] **Step 1: Write failing strongest-family tests**

Create Natural and Sharp / Flat overlays with four-family colors. Assert a purple Natural press returns `COLOR_4_W`, an accidental overlay never returns a Natural slot, filtered/nonpressed keys have no match, and Spark exposes the Standard winner only for keys it returns.

- [ ] **Step 2: Run detection tests and verify missing API**

Run: `.venv/bin/python -m pytest tests/test_standard_detection.py tests/test_bugfix_regressions.py -k "strongest_natural or never_matches or exposes_standard" -q`

Expected: FAIL because detectors return only pressed key IDs.

- [ ] **Step 3: Add the base match API and Standard state**

Add a default method to `DetectionMethod` returning `None`. In `StandardDetection`, reset `self.last_exemplar_matches` at each frame. Record the winning slot alongside progression/histogram data, then filter the map down to final `pressed_key_ids` before return:

```python
self.last_exemplar_matches = {
    key_id: overlay_detection_data[key_id]["winning_exemplar_slot"]
    for key_id in pressed_key_ids
    if overlay_detection_data[key_id]["winning_exemplar_slot"] is not None
}
```

Modify color progression selection to expose the slot with maximum valid progression. Modify histogram comparison to return `(passed: bool, winning_slot: str | None)` using the highest valid histogram ratio. Prefer the color winner when color passes; otherwise use the histogram winner. Preserve existing thresholds and final lit/off behavior.

- [ ] **Step 4: Delegate identity through Spark**

```python
def get_last_exemplar_match(self, key_id: int) -> str | None:
    if key_id not in self.previous_detected_keys:
        return None
    return self.standard_detector.get_last_exemplar_match(key_id)
```

Ensure `reset_state()` clears the Standard detector's match state.

- [ ] **Step 5: Run full detector tests**

Run: `.venv/bin/python -m pytest tests/test_standard_detection.py tests/test_bugfix_regressions.py -q`

Expected: PASS.

- [ ] **Step 6: Commit winning-family detection**

```bash
git add synthesia2midi/synthesia2midi/detection/base.py synthesia2midi/synthesia2midi/detection/standard.py synthesia2midi/synthesia2midi/detection/spark_integrated.py tests/test_standard_detection.py tests/test_bugfix_regressions.py
git commit -m "feat: retain winning color family during detection"
```

### Task 9: Route Detected Families to Four MIDI Channels

**Files:**
- Modify: `synthesia2midi/synthesia2midi/workflows/conversion.py`
- Create: `tests/test_color_family_channels.py`
- Modify: `tests/test_conversion_workflow_seams.py`

**Interfaces:**
- Consumes: `detector.get_last_exemplar_match()` and `family_for_slot()`.
- Produces: `_midi_channel_for_exemplar(slot: str | None) -> int` and passes frame-local winning slots into MIDI event creation.

- [ ] **Step 1: Write failing channel mapping tests**

Parametrize all eight slots and assert channels `0,0,1,1,2,2,3,3`. Process four simultaneous synthetic notes with winning slots from all four families and assert the MIDI writer starts notes on channels `[0, 1, 2, 3]`.

- [ ] **Step 2: Run channel tests and verify failures**

Run: `.venv/bin/python -m pytest tests/test_color_family_channels.py -q`

Expected: FAIL because conversion recomputes Left/Right color and cannot map Color 3/4.

- [ ] **Step 3: Pass frame-local exemplar matches into MIDI processing**

Immediately after `detect_frame()`, build:

```python
exemplar_matches = {
    key_id: detector.get_last_exemplar_match(key_id)
    for key_id in pressed_key_ids
}
```

Extend `_process_midi_events(pressed_key_ids, frame_idx, active_notes, midi_writer, frame_bgr, overlays, exemplar_matches: Mapping[int, str | None] | None = None)`. For every note-on, map a known slot directly through `family_for_slot(slot).midi_channel`. Retain the existing nearest-color routine only as a compatibility fallback when a detector implementation returns no identity.

- [ ] **Step 4: Make diagnostics dynamic**

Replace hard-coded four-slot settings-log dictionaries with comprehensions over `SUPPORTED_EXEMPLAR_SLOTS`, serializing only enabled or calibrated entries. Keep existing legacy field names where consumers expect them.

- [ ] **Step 5: Run conversion tests**

Run: `.venv/bin/python -m pytest tests/test_color_family_channels.py tests/test_conversion_workflow_seams.py tests/test_midi_conversion_controller.py -q`

Expected: PASS.

- [ ] **Step 6: Commit MIDI channel routing**

```bash
git add synthesia2midi/synthesia2midi/workflows/conversion.py tests/test_color_family_channels.py tests/test_conversion_workflow_seams.py
git commit -m "feat: export color families on separate MIDI channels"
```

### Task 10: Localization and Deterministic UI Audit

**Files:**
- Modify: six production `.ts` and `.qm` pairs under `synthesia2midi/synthesia2midi/translations/`
- Modify: `docs/localization/ui-string-manifest.json`
- Modify: `tests/test_localization.py`
- Modify: `tests/test_ui_string_audit.py`

**Interfaces:**
- Consumes: final user-visible strings from Tasks 5-7.
- Produces: complete production translations and pseudo-locale/audit coverage.

- [ ] **Step 1: Regenerate all `.ts` catalogs after copy is stable**

Run `pyside6-lupdate` over `synthesia2midi/synthesia2midi` with all six production `.ts` output paths in one invocation. Expected: catalogs update and new source strings appear as unfinished.

- [ ] **Step 2: Translate every new source string in all six catalogs**

Translate `Color {number}`, `Natural`, `Sharp / Flat`, `Set`, `Present`, `Found`, `Missing`, `Add Color Family`, remove/confirmation copy, readiness copy, and scanner warning copy. Preserve named placeholders and punctuation exactly. Remove obsolete Left/Right entries only when `lupdate` marks them vanished; do not hand-delete still-referenced messages.

- [ ] **Step 3: Compile every production catalog**

```bash
for locale in es ja ru zh_CN ko pt_BR; do
  .venv/bin/pyside6-lrelease \
    "synthesia2midi/synthesia2midi/translations/synthesia2midi_${locale}.ts" \
    -qm "synthesia2midi/synthesia2midi/translations/synthesia2midi_${locale}.qm" || exit 1
done
```

Expected: six successful release summaries with zero unfinished active messages.

- [ ] **Step 4: Regenerate the deterministic manifest**

Run: `.venv/bin/python -m synthesia2midi.tools.audit_ui_strings --output docs/localization/ui-string-manifest.json`

Expected: manifest updates only for intentional new/removed UI copy.

- [ ] **Step 5: Add locale matrix assertions**

Extend tests to instantiate `ColorFamilyGrid` for every production locale and `qps`, verify headings/actions are non-empty, and verify pseudo-locale visibly transforms new strings. Parse every `.ts` and assert no active translation is empty/unfinished and placeholders match source.

Add the family grid to the existing offscreen screenshot matrix at default and 150% font scaling. Assert one-, two-, three-, and four-family variants render without clipped text, overlapping controls, or horizontal scrollbars.

- [ ] **Step 6: Run localization and UI audit gates**

Run: `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/test_localization.py tests/test_ui_string_audit.py tests/test_color_family_grid.py -q`

Expected: PASS.

- [ ] **Step 7: Commit localization assets**

```bash
git add synthesia2midi/synthesia2midi/translations docs/localization/ui-string-manifest.json tests/test_localization.py tests/test_ui_string_audit.py tests/test_color_family_grid.py
git commit -m "feat: localize four color family controls"
```

### Task 11: Reference-Video Acceptance Harness and Documentation

**Files:**
- Create: `synthesia2midi/tools/inspect_color_family_scan.py`
- Create: `tests/test_inspect_color_family_scan.py`
- Modify: `backlog/tasks/task-22 - Support-four-color-families-and-eight-lit-exemplars.md`
- Modify: `PROJECT_LOG.md`

**Interfaces:**
- Consumes: scanner and assignment pipeline.
- Produces: `build_scan_report(result, diagnostics)`, `serialize_family_assignments(result)`, `build_argument_parser()`, `scan_configured_video(args)`, and `write_report(report, output_path)` for a read-only diagnostic CLI. It never writes video frames by default.

- [ ] **Step 1: Write failing CLI serialization test**

Assert a synthetic four-family result serializes `family_count == 4`, Color 4 Natural slot `COLOR_4_W`, refinement counters, stable ordering, and creates no media files.

- [ ] **Step 2: Run the test and verify the harness is missing**

Run: `.venv/bin/python -m pytest tests/test_inspect_color_family_scan.py -q`

Expected: FAIL on import.

- [ ] **Step 3: Implement the read-only diagnostic module**

```python
def build_scan_report(
    result: ExemplarAssignmentResult,
    diagnostics: ExemplarScanDiagnostics,
) -> dict[str, object]:
    return {
        "family_count": len(result.families),
        "families": serialize_family_assignments(result),
        "warnings": list(result.warnings),
        "diagnostics": asdict(diagnostics),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result, diagnostics = scan_configured_video(args)
    report = build_scan_report(result, diagnostics)
    write_report(report, output_path=args.output_json)
    return 0
```

CLI arguments: `video`, `--config`, `--start-frame`, `--end-frame`, and optional `--output-json`. Require an existing calibration/config that supplies overlays and unlit references. Print JSON to stdout by default. Do not download URLs or write extracted frames.

- [ ] **Step 4: Run the local reference acceptance input**

Run against `/tmp/synthesia2midi-dual-piano.mp4` and the locally saved calibration INI for that video. Expected: `family_count` is 4; orange, blue, yellow, and purple each have stable identities; all observed morphologies have slots; refined work is limited to promising events; no repository files are generated. Locate the actual INI with `find ~/Desktop ~/Downloads ~/Library/Application\ Support -name '*.ini' -mtime -30` and confirm its video path/title before use. If no matching INI exists, exercise the same flow interactively in the app and record the missing local fixture as an acceptance limitation rather than committing media.

- [ ] **Step 5: Export a local MIDI and inspect channels**

Use the app's normal conversion flow with the accepted proposal. Inspect the generated MIDI with the existing MIDI test/parser utility and confirm note-on events occur on internal channels `{0, 1, 2, 3}`. Keep the MIDI under `/tmp` or the user's output directory; do not add it to Git.

- [ ] **Step 6: Update Backlog and project handoff**

Record deterministic test completion and reference-video observations in Task 22. Keep the task In Progress until full gates and package smokes pass. Add a concise `PROJECT_LOG.md` handoff stating feature branch, latest commit, completed gates, acceptance result, and remaining release boundary.

- [ ] **Step 7: Run harness tests and commit**

Run: `.venv/bin/python -m pytest tests/test_inspect_color_family_scan.py -q`

Expected: PASS.

```bash
git add synthesia2midi/tools/inspect_color_family_scan.py tests/test_inspect_color_family_scan.py "backlog/tasks/task-22 - Support-four-color-families-and-eight-lit-exemplars.md" PROJECT_LOG.md
git commit -m "test: add four color family acceptance harness"
```

### Task 12: Full Verification, Review, and Package Gates

**Files:**
- Modify only if failures reveal feature defects: files already listed above.
- Modify after all gates pass: `backlog/tasks/task-22 - Support-four-color-families-and-eight-lit-exemplars.md`
- Modify after all gates pass: `PROJECT_LOG.md`

**Interfaces:**
- Produces: reviewed, merge-ready branch. It does not merge, push, tag, or publish.

- [ ] **Step 1: Run static repository checks**

```bash
git diff --check
.venv/bin/python -m compileall -q synthesia2midi
.venv/bin/pyside6-lupdate -extensions py synthesia2midi/synthesia2midi -ts /tmp/synthesia2midi_lupdate_probe.ts
```

Expected: no whitespace errors, compileall exits 0, and lupdate extracts nonzero source texts without parser errors.

- [ ] **Step 2: Run focused feature suites**

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest \
  tests/test_color_families.py \
  tests/test_color_family_assignment.py \
  tests/test_assisted_calibration.py \
  tests/test_color_family_grid.py \
  tests/test_assisted_calibration_dialog.py \
  tests/test_color_family_channels.py \
  tests/test_config_manager.py \
  tests/test_localization.py \
  tests/test_ui_string_audit.py -q
```

Expected: PASS.

- [ ] **Step 3: Run the full Python and Rust gates from `docs/testing.md`**

Run the exact current canonical commands documented in `docs/testing.md`, including `.venv/bin/python -m pytest` and the Rust touch-up editor checks. Do not duplicate or alter canonical commands in this plan if that document has changed. Expected: every gate exits 0.

- [ ] **Step 4: Run an independent code review**

Use `superpowers:requesting-code-review`. Review specifically for old-INI compatibility, accidental slot/morphology mismatch, scanner overfitting, cancellation rollback, channel identity drift, and UI overflow at 150% scaling. Fix every confirmed P0/P1 finding with a failing regression test first; rerun focused and full suites after fixes.

- [ ] **Step 5: Verify packaged translation/data collection locally**

Run the current packaged-entrypoint/spec tests from `docs/testing.md`. Inspect the PyInstaller analysis output and assert all six production `.qm` files are collected. Expected: PASS without a release tag.

- [ ] **Step 6: Push only after explicit approval, then run remote package smokes**

After Jeff explicitly requests a push, push `codex/four-color-families` and run the GitHub Actions smoke matrix for Windows x64 and Apple Silicon macOS. Do not require Intel macOS or Linux. Expected: both requested jobs green.

- [ ] **Step 7: Close implementation records after all required gates pass**

Mark Task 22 Done with acceptance evidence and update `PROJECT_LOG.md`. Commit only those status changes:

```bash
git add "backlog/tasks/task-22 - Support-four-color-families-and-eight-lit-exemplars.md" PROJECT_LOG.md
git commit -m "docs: close four color family implementation"
```

- [ ] **Step 8: Stop at the release boundary**

Report branch commits, local test totals, reference-video result, review findings, and Windows/macOS smoke URLs. Do not merge to `main`, tag `v0.2.0`, or publish packages until Jeff explicitly approves those actions.
