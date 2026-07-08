# Phase 1 UX Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing Synthesia2MIDI video-to-MIDI workflow clearer for nontechnical users by improving visible guidance, readiness feedback, calibration wording, advanced-setting framing, and YouTube fallback copy without changing detection behavior or persisted project formats.

**Architecture:** Keep this as a frontend/Qt slice. `ControlPanelQt` owns bottom-rail readiness, settings-section copy, calibration guidance, and quick-adjust UI state; `CalibrationWizard`, `AutoDetectTuningDialog`, `CalibrationWizardController`, and `YouTubeDownloadDialog` own their existing dialog copy. No detector, conversion, config, runtime path, or `main.py` workflow behavior changes are part of this plan.

**Tech Stack:** Python 3, PySide6, pytest, Qt `.ts`/`.qm` translation assets, existing Synthesia2MIDI GUI/controllers.

## Global Constraints

- Do not push.
- Do not create a worktree.
- Start implementation with `git status --short --branch` and preserve unrelated changes.
- Keep branch `codex/ux-guided-calibration` unless Jeff explicitly redirects.
- Keep Left/Right terminology visible in calibration copy; clarify that Left/Right means Synthesia note color/family, not physical keyboard position.
- Do not change detector algorithms, auto-detect parameters, MIDI conversion behavior, saved `.ini` formats, or overlay sidecar formats.
- Do not add workflow bodies or signal-compatibility wrappers to `synthesia2midi/synthesia2midi/main.py`.
- All new user-visible strings must use the existing Qt translation patterns.
- Update production `.ts` and `.qm` translation assets after copy changes.
- Run focused tests after each coherent slice and commit each slice that passes.

---

## File Structure

```text
backlog/tasks/task-17 - Phase-1-UX-hardening.md
  Track plan link and final acceptance status.

docs/superpowers/plans/2026-07-08-phase-1-ux-hardening.md
  This implementation plan.

synthesia2midi/synthesia2midi/gui/controls_qt.py
  Conversion readiness model, bottom rail status, Calibration/Overlays/Detection/Spark/MIDI/Trim/Optional copy, quick-adjust value/reset controls.

synthesia2midi/synthesia2midi/gui/wizard.py
  Calibration Wizard keyboard-box wording and inline disabled reason.

synthesia2midi/synthesia2midi/gui/calibration_wizard_controller.py
  Assisted calibration confirmation copy, keeping the algorithm unchanged.

synthesia2midi/synthesia2midi/gui/auto_detect_tuning_dialog.py
  Auto-detect tuning guidance, reset label, advanced tab label.

synthesia2midi/synthesia2midi/gui/youtube_download_dialog.py
  YouTube fallback framing and quality labels.

tests/test_controls_qt.py
  Direct ControlPanelQt behavior tests for readiness, calibration copy, quick adjustments, and section wording.

tests/test_main_window_layout.py
  Existing layout regression tests updated only where labels changed.

tests/test_auto_detect_tuning_dialog.py
  Auto-detect tuning copy/layout tests.

tests/test_youtube_download_dialog.py
  YouTube fallback and quality wording tests.

tests/test_calibration_wizard_copy.py
  New focused tests for CalibrationWizard visible wording.

tests/test_assisted_calibration_copy.py
  New focused tests for proposal summary wording without raw RGB-first messages.

docs/localization/ui-string-manifest.json
docs/localization/translation-agent-packet.json
synthesia2midi/synthesia2midi/translations/synthesia2midi_*.ts
synthesia2midi/synthesia2midi/translations/synthesia2midi_*.qm
  Regenerated localization audit/catalog assets for changed user-visible strings.
```

---

## Task 1: Conversion Readiness Status

**Files:**
- Modify: `synthesia2midi/synthesia2midi/gui/controls_qt.py`
- Test: `tests/test_controls_qt.py`

**Interfaces:**
- Produces: `ConversionReadiness(can_convert: bool, status_text: str)` in `controls_qt.py`
- Produces: `ControlPanelQt._conversion_readiness(self) -> ConversionReadiness`
- Keeps: `ControlPanelQt._can_convert(self) -> bool`
- Consumes: existing `AppState.video.filepath`, `AppState.overlays`, `DetectionConfig.get_required_base_exemplar_types()`, `DetectionConfig.get_effective_exemplar_lit_colors()`, and `MIDIConfig.tempo`

- [ ] **Step 1: Write failing readiness tests**

Append these tests to `tests/test_controls_qt.py`:

```python
from synthesia2midi.app_config import OverlayConfig
from synthesia2midi.core.app_state import AppState


def _panel_with_state(app_state: AppState) -> ControlPanelQt:
    QApplication.instance() or QApplication([])
    return ControlPanelQt(app_state=app_state)


def _basic_overlay(*, unlit=True, unlit_hist=None) -> OverlayConfig:
    return OverlayConfig(
        key_id=1,
        note_octave=4,
        note_name_in_octave="C",
        x=0,
        y=0,
        width=10,
        height=40,
        key_type="white",
        unlit_reference_color=(12, 12, 12) if unlit else None,
        unlit_hist=unlit_hist,
    )


def _calibrate_all_exemplars(app_state: AppState) -> None:
    app_state.detection.exemplar_lit_colors = {
        "LW": (255, 0, 0),
        "LB": (160, 0, 0),
        "RW": (0, 120, 255),
        "RB": (0, 70, 180),
    }


def test_conversion_readiness_explains_first_missing_prerequisite():
    state = AppState()
    panel = _panel_with_state(state)
    try:
        assert not panel.convert_button.isEnabled()
        assert panel.conversion_status.text() == "Load a video to convert."

        state.video.filepath = "/tmp/source.mp4"
        panel.update_controls_from_state()
        assert not panel.convert_button.isEnabled()
        assert panel.conversion_status.text() == "Create key overlays first."

        state.overlays = [_basic_overlay(unlit=False)]
        panel.update_controls_from_state()
        assert not panel.convert_button.isEnabled()
        assert panel.conversion_status.text() == "Capture a no-key frame."

        state.overlays = [_basic_overlay(unlit=True)]
        panel.update_controls_from_state()
        assert not panel.convert_button.isEnabled()
        assert panel.conversion_status.text() == "Capture at least one pressed-key example."

        _calibrate_all_exemplars(state)
        panel.update_controls_from_state()
        assert panel.convert_button.isEnabled()
        assert panel.conversion_status.text() == "Ready to create MIDI."
    finally:
        panel.close()
        panel.deleteLater()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest tests/test_controls_qt.py::test_conversion_readiness_explains_first_missing_prerequisite -v
```

Expected: FAIL because `conversion_status` currently starts as `Ready to convert` and `_conversion_readiness` does not exist.

- [ ] **Step 3: Implement readiness model**

In `synthesia2midi/synthesia2midi/gui/controls_qt.py`, change imports:

```python
import logging
from dataclasses import dataclass
from typing import Optional
```

Add this above `class CollapsibleSection(QWidget):`

```python
@dataclass(frozen=True)
class ConversionReadiness:
    can_convert: bool
    status_text: str
```

Change `_create_global_action_widgets()` status initialization to:

```python
self.conversion_status = QLabel(
    QCoreApplication.translate("ControlPanelQt", "Load a video to convert.")
)
self.conversion_status.setWordWrap(True)
```

Replace `_can_convert()` with this compatibility wrapper and add the new helpers directly above it:

```python
def _conversion_readiness(self) -> ConversionReadiness:
    """Return conversion availability plus the first user-actionable missing step."""
    if not self.app_state or not hasattr(self.app_state, "video"):
        return ConversionReadiness(
            False,
            translate("ControlPanelQt", "Load a video to convert."),
        )

    if not getattr(self.app_state.video, "filepath", None):
        return ConversionReadiness(
            False,
            translate("ControlPanelQt", "Load a video to convert."),
        )

    overlays = getattr(self.app_state, "overlays", None) or []
    if not overlays:
        return ConversionReadiness(
            False,
            translate("ControlPanelQt", "Create key overlays first."),
        )

    missing_unlit = [
        overlay.key_id
        for overlay in overlays
        if getattr(overlay, "unlit_reference_color", None) is None
    ]
    if missing_unlit:
        return ConversionReadiness(
            False,
            translate("ControlPanelQt", "Capture a no-key frame."),
        )

    if getattr(self.app_state.detection, "use_histogram_detection", False):
        missing_hist = [
            overlay.key_id
            for overlay in overlays
            if getattr(overlay, "unlit_hist", None) is None
        ]
        if missing_hist:
            return ConversionReadiness(
                False,
                translate("ControlPanelQt", "Capture a no-key frame."),
            )

    required_exemplars = self.app_state.detection.get_required_base_exemplar_types()
    if not required_exemplars:
        return ConversionReadiness(
            False,
            translate("ControlPanelQt", "Capture at least one pressed-key example."),
        )

    exemplar_colors = self.app_state.detection.get_effective_exemplar_lit_colors()
    for exemplar in required_exemplars:
        if exemplar_colors.get(exemplar) is None:
            return ConversionReadiness(
                False,
                translate("ControlPanelQt", "Capture at least one pressed-key example."),
            )

    detection_threshold = getattr(self.app_state.detection, "detection_threshold", 0.0)
    if not 0.1 <= detection_threshold <= 0.99:
        return ConversionReadiness(
            False,
            translate("ControlPanelQt", "Check detection sensitivity."),
        )

    if getattr(self.app_state.midi, "tempo", 0) <= 0:
        return ConversionReadiness(
            False,
            translate("ControlPanelQt", "Check MIDI tempo."),
        )

    return ConversionReadiness(
        True,
        translate("ControlPanelQt", "Ready to create MIDI."),
    )


def _update_conversion_readiness_display(self) -> None:
    readiness = self._conversion_readiness()
    self.convert_button.setEnabled(readiness.can_convert)
    self.conversion_status.setText(readiness.status_text)


def _can_convert(self) -> bool:
    """Return True if MIDI conversion prerequisites are satisfied."""
    return self._conversion_readiness().can_convert
```

In `update_controls_from_state()`, replace the current convert-button update:

```python
if hasattr(self, "convert_button"):
    self.convert_button.setEnabled(self._can_convert())
```

with:

```python
if hasattr(self, "convert_button"):
    self._update_conversion_readiness_display()
```

In `set_conversion_result()`, replace:

```python
self.convert_button.setEnabled(True)
```

with:

```python
self.convert_button.setEnabled(self._can_convert())
```

- [ ] **Step 4: Run focused readiness test**

Run:

```bash
.venv/bin/python -m pytest tests/test_controls_qt.py::test_conversion_readiness_explains_first_missing_prerequisite -v
```

Expected: PASS.

- [ ] **Step 5: Run nearby control-panel tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_controls_qt.py tests/test_main_window_layout.py::test_settings_lower_rail_holds_global_actions_and_status -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add synthesia2midi/synthesia2midi/gui/controls_qt.py tests/test_controls_qt.py
git commit -m "Improve conversion readiness feedback"
```

---

## Task 2: Calibration Section And Wizard Copy

**Files:**
- Modify: `synthesia2midi/synthesia2midi/gui/controls_qt.py`
- Modify: `synthesia2midi/synthesia2midi/gui/wizard.py`
- Test: `tests/test_controls_qt.py`
- Create: `tests/test_calibration_wizard_copy.py`

**Interfaces:**
- Produces: `ControlPanelQt.calibration_instruction_labels: dict[str, QLabel]`
- Produces: `CalibrationWizard.edit_current_reason_label: QLabel`
- Keeps: existing calibration signals and button attribute names

- [ ] **Step 1: Write failing ControlPanel calibration copy test**

Append to `tests/test_controls_qt.py`:

```python
def test_calibration_section_shows_visible_step_instructions():
    QApplication.instance() or QApplication([])
    panel = ControlPanelQt()
    try:
        assert panel.calibration_wizard_button.text() == "Draw Keyboard Box and Find Keys"
        assert panel.calibrate_unlit_button.text() == "Capture No-Key Frame"
        assert panel.calibration_instruction_labels["keyboard"].text() == (
            "Pause on a clear frame where the full keyboard is visible."
        )
        assert panel.calibration_instruction_labels["unlit"].text() == "Pause where no keys are glowing."
        assert panel.calibration_instruction_labels["pressed"].text() == (
            "Pause where a key is glowing, then click that key."
        )
        assert panel.left_right_color_family_note.text() == (
            "Left/Right refer to Synthesia note colors, not the physical side of the keyboard."
        )
    finally:
        panel.close()
        panel.deleteLater()
```

- [ ] **Step 2: Write failing CalibrationWizard copy test**

Create `tests/test_calibration_wizard_copy.py`:

```python
from PySide6.QtWidgets import QApplication, QLabel

from synthesia2midi.core.app_state import AppState
from synthesia2midi.gui.wizard import CalibrationWizard


def test_calibration_wizard_uses_plain_keyboard_box_language():
    QApplication.instance() or QApplication([])
    dialog = CalibrationWizard(None, AppState())
    try:
        dialog.show()
        QApplication.processEvents()
        button_texts = [button.text() for button in dialog.findChildren(type(dialog.edit_current_calibration_button))]
        label_texts = [label.text() for label in dialog.findChildren(QLabel)]

        assert "Draw Keyboard Box and Find Keys" in button_texts
        assert "Select Keyboard Region With Autodetector" not in button_texts
        assert "Pause on a clear frame where the full keyboard is visible." in label_texts
        assert dialog.edit_current_reason_label.text() == (
            "Edit becomes available after you create key overlays."
        )
        assert dialog.edit_current_reason_label.isVisible()

        dialog.set_edit_current_calibration_enabled(True)

        assert dialog.edit_current_calibration_button.isEnabled()
        assert not dialog.edit_current_reason_label.isVisible()
    finally:
        dialog.close()
        dialog.deleteLater()
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_controls_qt.py::test_calibration_section_shows_visible_step_instructions tests/test_calibration_wizard_copy.py -v
```

Expected: FAIL because the current UI still uses `Calibrate`, old wizard wording, and no inline disabled reason label.

- [ ] **Step 4: Implement visible Calibration section copy**

In `_create_mandatory_calibration_tab()` in `controls_qt.py`, replace the compact `calibration_grid` row labels for overlays/unlit with a vertical normal-path layout. Keep `self.calibration_wizard_button`, `self.calibrate_unlit_button`, `self.unlit_status_label`, `self.exemplar_buttons`, `self.exemplar_swatches`, and `self.exemplar_presence_checkboxes`.

Use this local helper inside `_create_mandatory_calibration_tab()` before the three rows:

```python
self.calibration_instruction_labels = {}

def add_instruction_row(row_key: str, title: str, instruction: str, action_widget: QWidget) -> None:
    row_widget = QWidget()
    row = QVBoxLayout(row_widget)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(4)

    title_label = QLabel(title)
    title_label.setStyleSheet("font-weight: bold; font-size: 11pt;")
    row.addWidget(title_label)

    instruction_label = QLabel(instruction)
    instruction_label.setWordWrap(True)
    instruction_label.setStyleSheet("color: #555;")
    self.calibration_instruction_labels[row_key] = instruction_label
    row.addWidget(instruction_label)
    row.addWidget(action_widget)

    layout.addWidget(row_widget)
```

Build the keyboard row:

```python
self.calibration_wizard_button = QPushButton(
    translate("ControlPanelQt", "Draw Keyboard Box and Find Keys")
)
self.calibration_wizard_button.setMinimumWidth(180)
self.calibration_wizard_button.clicked.connect(self.calibration_wizard_requested.emit)
self.calibration_wizard_button.setToolTip(
    translate("ControlPanelQt", "Creates overlays for the keyboard in your video. Re-run if overlays don't line up.")
)
add_instruction_row(
    "keyboard",
    translate("ControlPanelQt", "Find the keyboard"),
    translate("ControlPanelQt", "Pause on a clear frame where the full keyboard is visible."),
    self.calibration_wizard_button,
)
```

Keep the Octave spinbox row as a small `QGridLayout` immediately after the keyboard row:

```python
octave_grid = QGridLayout()
octave_grid.setHorizontalSpacing(8)
octave_label = QLabel(translate("ControlPanelQt", "Octave"))
octave_label.setStyleSheet("font-weight: bold; font-size: 11pt;")
octave_grid.addWidget(octave_label, 0, 0)
self.octave_transpose_spin = QSpinBox()
self.octave_transpose_spin.setRange(-5, 5)
self.octave_transpose_spin.setValue(0)
self.octave_transpose_spin.setFixedWidth(64)
self.octave_transpose_spin.valueChanged.connect(self.octave_transpose_changed.emit)
self.octave_transpose_spin.setToolTip(translate("ControlPanelQt", "Shifts the MIDI output up/down by octaves."))
octave_grid.addWidget(self.octave_transpose_spin, 0, 1)
octave_grid.setColumnStretch(2, 1)
layout.addLayout(octave_grid)
```

Build the no-key row:

```python
self.calibrate_unlit_button = QPushButton(translate("ControlPanelQt", "Capture No-Key Frame"))
self.calibrate_unlit_button.setMinimumWidth(180)
self.calibrate_unlit_button.clicked.connect(self.calibrate_unlit_requested.emit)
self.calibrate_unlit_button.setToolTip(
    translate(
        "ControlPanelQt",
        "Captures what unpressed overlays look like from the current frame. Pause on a frame with no highlighted notes first.",
    )
)

unlit_widget = QWidget()
unlit_value_layout = QVBoxLayout(unlit_widget)
unlit_value_layout.setContentsMargins(0, 0, 0, 0)
unlit_value_layout.setSpacing(3)
unlit_value_layout.addWidget(self.calibrate_unlit_button)
self.unlit_status_label = QLabel(translate("ControlPanelQt", "Not Set"))
self.unlit_status_label.setStyleSheet("font-style: italic; color: #888;")
unlit_value_layout.addWidget(self.unlit_status_label)

add_instruction_row(
    "unlit",
    translate("ControlPanelQt", "Capture no-key frame"),
    translate("ControlPanelQt", "Pause where no keys are glowing."),
    unlit_widget,
)
```

Before the exemplar buttons, add:

```python
pressed_title = QLabel(translate("ControlPanelQt", "Capture pressed-key examples"))
pressed_title.setStyleSheet("font-weight: bold; font-size: 11pt;")
layout.addWidget(pressed_title)

pressed_instruction = QLabel(
    translate("ControlPanelQt", "Pause where a key is glowing, then click that key.")
)
pressed_instruction.setWordWrap(True)
pressed_instruction.setStyleSheet("color: #555;")
self.calibration_instruction_labels["pressed"] = pressed_instruction
layout.addWidget(pressed_instruction)

self.left_right_color_family_note = QLabel(
    translate(
        "ControlPanelQt",
        "Left/Right refer to Synthesia note colors, not the physical side of the keyboard.",
    )
)
self.left_right_color_family_note.setWordWrap(True)
self.left_right_color_family_note.setStyleSheet("color: #555; font-style: italic;")
layout.addWidget(self.left_right_color_family_note)
```

Keep buttons `Set Left White`, `Set Left Black`, `Set Right White`, and `Set Right Black`.

- [ ] **Step 5: Implement CalibrationWizard copy**

In `wizard.py`, replace the autodetect button text:

```python
auto_selection_button = QPushButton(
    QCoreApplication.translate("CalibrationWizard", "Draw Keyboard Box and Find Keys")
)
```

Add this visible label immediately above the autodetect button:

```python
auto_instruction_label = QLabel(
    QCoreApplication.translate(
        "CalibrationWizard",
        "Pause on a clear frame where the full keyboard is visible.",
    )
)
auto_instruction_label.setWordWrap(True)
layout.addWidget(auto_instruction_label, 0, 0, 1, 3)
```

Move the autodetect button to row `1`, the edit-current button to row `2`, and increment the existing manual rows by `+1`.

After `self.edit_current_calibration_button`, add:

```python
self.edit_current_reason_label = QLabel(
    QCoreApplication.translate(
        "CalibrationWizard",
        "Edit becomes available after you create key overlays.",
    )
)
self.edit_current_reason_label.setWordWrap(True)
self.edit_current_reason_label.setStyleSheet("color: #666; font-style: italic;")
layout.addWidget(self.edit_current_reason_label, 3, 0, 1, 3)
```

Then move the manual label and following rows down by one more row. Update `set_edit_current_calibration_enabled()`:

```python
def set_edit_current_calibration_enabled(self, enabled: bool, tooltip: Optional[str] = None) -> None:
    self.edit_current_calibration_button.setEnabled(enabled)
    self.edit_current_reason_label.setVisible(not enabled)
    if tooltip:
        self.edit_current_calibration_button.setToolTip(tooltip)
        if not enabled:
            self.edit_current_reason_label.setText(tooltip)
```

- [ ] **Step 6: Run focused calibration copy tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_controls_qt.py::test_calibration_section_shows_visible_step_instructions tests/test_calibration_wizard_copy.py -v
```

Expected: PASS.

- [ ] **Step 7: Run layout tests for settings rail and Calibration tab**

Run:

```bash
.venv/bin/python -m pytest tests/test_main_window_layout.py tests/test_controls_qt.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

Run:

```bash
git add synthesia2midi/synthesia2midi/gui/controls_qt.py synthesia2midi/synthesia2midi/gui/wizard.py tests/test_controls_qt.py tests/test_calibration_wizard_copy.py
git commit -m "Clarify calibration setup copy"
```

---

## Task 3: Overlay Quick-Adjustment Values And Reset

**Files:**
- Modify: `synthesia2midi/synthesia2midi/gui/controls_qt.py`
- Modify: `tests/test_main_window_layout.py`

**Interfaces:**
- Produces: `ControlPanelQt._apply_overlay_adjustment(self, key_color: str, dimension: str, delta: int) -> None`
- Produces: `ControlPanelQt._reset_overlay_adjustment(self, key_color: str, dimension: str) -> None`
- Produces visible value labels and reset buttons, including `left_slant_value_label`, `left_slant_reset_button`, `right_slant_value_label`, `right_slant_reset_button`
- Keeps: `overlay_size_adjustment_requested(str, str, int)` semantics

- [ ] **Step 1: Write failing quick-adjust test**

Replace `test_overlays_tab_exposes_left_and_right_slant_controls` in `tests/test_main_window_layout.py` with:

```python
def test_overlays_tab_exposes_left_and_right_slant_controls(monkeypatch):
    app = _make_app(monkeypatch)
    try:
        emitted = []
        try:
            app.control_panel.overlay_size_adjustment_requested.disconnect()
        except (TypeError, RuntimeError):
            pass
        app.control_panel.overlay_size_adjustment_requested.connect(
            lambda key_color, dimension, delta: emitted.append((key_color, dimension, delta))
        )

        assert app.control_panel.left_slant_label.text() == "Left Slant"
        assert app.control_panel.right_slant_label.text() == "Right Slant"
        assert app.control_panel.left_slant_value_label.text() == "0"
        assert app.control_panel.right_slant_value_label.text() == "0"
        assert app.control_panel.left_slant_reset_button.text() == "Reset"
        assert app.control_panel.right_slant_reset_button.text() == "Reset"

        app.control_panel.left_slant_inc_button.click()
        app.control_panel.right_slant_dec_button.click()

        assert app.control_panel.left_slant_value_label.text() == "1"
        assert app.control_panel.right_slant_value_label.text() == "-1"

        app.control_panel.left_slant_reset_button.click()

        assert app.control_panel.left_slant_value_label.text() == "0"
        assert emitted == [
            ("all", "left_slant", 1),
            ("all", "right_slant", -1),
            ("all", "left_slant", -1),
        ]
    finally:
        app.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest tests/test_main_window_layout.py::test_overlays_tab_exposes_left_and_right_slant_controls -v
```

Expected: FAIL because value labels and reset buttons do not exist.

- [ ] **Step 3: Add transient adjustment state**

In `ControlPanelQt.__init__`, before `_setup_ui()`:

```python
self._overlay_adjustment_values: dict[tuple[str, str], int] = {}
self._overlay_adjustment_value_labels: dict[tuple[str, str], QLabel] = {}
```

Add methods near `_update_color_square()`:

```python
def _overlay_adjustment_key(self, key_color: str, dimension: str) -> tuple[str, str]:
    return key_color, dimension


def _set_overlay_adjustment_value(self, key_color: str, dimension: str, value: int) -> None:
    key = self._overlay_adjustment_key(key_color, dimension)
    self._overlay_adjustment_values[key] = value
    label = self._overlay_adjustment_value_labels.get(key)
    if label is not None:
        label.setText(str(value))


def _apply_overlay_adjustment(self, key_color: str, dimension: str, delta: int) -> None:
    key = self._overlay_adjustment_key(key_color, dimension)
    current_value = self._overlay_adjustment_values.get(key, 0)
    self._set_overlay_adjustment_value(key_color, dimension, current_value + delta)
    self.overlay_size_adjustment_requested.emit(key_color, dimension, delta)


def _reset_overlay_adjustment(self, key_color: str, dimension: str) -> None:
    key = self._overlay_adjustment_key(key_color, dimension)
    current_value = self._overlay_adjustment_values.get(key, 0)
    if current_value == 0:
        self._set_overlay_adjustment_value(key_color, dimension, 0)
        return
    self._set_overlay_adjustment_value(key_color, dimension, 0)
    self.overlay_size_adjustment_requested.emit(key_color, dimension, -current_value)
```

- [ ] **Step 4: Replace size-control helper**

Inside `_create_overlay_settings_tab()`, replace `add_size_control(row, column, label, dec_button, inc_button)` with:

```python
def add_size_control(row, column, label, dec_button, inc_button, key_color, dimension):
    key = self._overlay_adjustment_key(key_color, dimension)
    cell = QVBoxLayout()
    cell.setContentsMargins(0, 0, 0, 0)
    cell.setSpacing(4)
    cell.addWidget(label)

    value_row = QHBoxLayout()
    value_row.setContentsMargins(0, 0, 0, 0)
    value_row.setSpacing(6)
    value_caption = QLabel(translate("ControlPanelQt", "Current:"))
    value_label = QLabel("0")
    value_label.setMinimumWidth(24)
    reset_button = QPushButton(translate("ControlPanelQt", "Reset"))
    reset_button.setMaximumWidth(72)
    reset_button.clicked.connect(lambda checked=False, kc=key_color, dim=dimension: self._reset_overlay_adjustment(kc, dim))
    value_row.addWidget(value_caption)
    value_row.addWidget(value_label)
    value_row.addWidget(reset_button)
    value_row.addStretch()
    cell.addLayout(value_row)

    button_row = QHBoxLayout()
    button_row.setContentsMargins(0, 0, 0, 0)
    button_row.setSpacing(4)
    button_row.addWidget(dec_button)
    button_row.addWidget(inc_button)
    button_row.addStretch()
    cell.addLayout(button_row)

    self._overlay_adjustment_value_labels[key] = value_label
    self._set_overlay_adjustment_value(key_color, dimension, 0)
    size_grid.addLayout(cell, row, column)
    return value_label, reset_button
```

For each decrement/increment button, replace direct `.emit(...)` lambdas with `_apply_overlay_adjustment(...)`. For Left Slant:

```python
self.left_slant_dec_button.clicked.connect(lambda: self._apply_overlay_adjustment("all", "left_slant", -1))
self.left_slant_inc_button.clicked.connect(lambda: self._apply_overlay_adjustment("all", "left_slant", 1))
self.left_slant_value_label, self.left_slant_reset_button = add_size_control(
    2,
    0,
    self.left_slant_label,
    self.left_slant_dec_button,
    self.left_slant_inc_button,
    "all",
    "left_slant",
)
```

Do the same for `right_slant`, `white height`, `white width`, `black height`, and `black width`, using their existing key color, dimension, and delta values.

- [ ] **Step 5: Run quick-adjust and overlay layout tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_main_window_layout.py::test_overlays_tab_exposes_left_and_right_slant_controls tests/test_main_window_layout.py::test_overlay_size_controls_stack_for_narrow_settings_window -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add synthesia2midi/synthesia2midi/gui/controls_qt.py tests/test_main_window_layout.py
git commit -m "Show overlay adjustment values and reset controls"
```

---

## Task 4: Auto-Detect Tuning And Assisted Calibration Copy

**Files:**
- Modify: `synthesia2midi/synthesia2midi/gui/auto_detect_tuning_dialog.py`
- Modify: `synthesia2midi/synthesia2midi/gui/calibration_wizard_controller.py`
- Modify: `tests/test_auto_detect_tuning_dialog.py`
- Create: `tests/test_assisted_calibration_copy.py`

**Interfaces:**
- Produces: `AutoDetectTuningDialog.guidance_label: QLabel`
- Produces: `AutoDetectTuningDialog.reset_all_button: QPushButton`
- Produces: `AutoDetectTuningDialog.tabs: QTabWidget`
- Keeps: `_reset_all_to_defaults`, preview status, `accept()`, and assisted calibration proposal application unchanged

- [ ] **Step 1: Write failing Auto-Detect Tuning copy test**

Append to `tests/test_auto_detect_tuning_dialog.py`:

```python
from PySide6.QtWidgets import QPushButton, QTabWidget


def test_auto_detect_tuning_dialog_uses_user_guidance_copy():
    QApplication.instance() or QApplication([])
    dialog = AutoDetectTuningDialog(
        None,
        AppState(),
        np.zeros((8, 8, 3), dtype=np.uint8),
        (0, 0, 8, 8),
        initial_detection_results={"total_keys": 88},
        fallback_used=False,
        apply_detection_callback=lambda _results: True,
    )

    try:
        label_texts = [label.text() for label in dialog.findChildren(QLabel)]
        button_texts = [button.text() for button in dialog.findChildren(QPushButton)]
        tabs = dialog.findChild(QTabWidget)

        assert (
            "Check the overlays on the video. If they line up with the keys, click Save. "
            "If the edges are off, adjust the edge controls."
        ) in label_texts
        assert "Reset to Recommended Settings" in button_texts
        assert tabs.tabText(1) == "Advanced Detector Settings"
    finally:
        dialog.close()
```

- [ ] **Step 2: Write failing assisted calibration summary test**

Create `tests/test_assisted_calibration_copy.py`:

```python
from types import SimpleNamespace

from PySide6.QtWidgets import QApplication

from synthesia2midi.core.app_state import AppState
from synthesia2midi.gui.calibration_wizard_controller import CalibrationWizardController


def test_assisted_calibration_summary_explains_color_families_without_rgb_first():
    QApplication.instance() or QApplication([])
    controller = CalibrationWizardController.__new__(CalibrationWizardController)
    proposal = SimpleNamespace(
        candidate_count=12,
        assignment_result=SimpleNamespace(
            family_count=2,
            assignments={
                "LW": SimpleNamespace(enabled=True, rgb=(255, 0, 0)),
                "LB": SimpleNamespace(enabled=True, rgb=(160, 0, 0)),
                "RW": SimpleNamespace(enabled=True, rgb=(0, 120, 255)),
                "RB": SimpleNamespace(enabled=False, rgb=None),
            },
        ),
    )

    text = controller._proposal_summary_text(proposal)

    assert "Assisted calibration found 12 possible pressed-key samples." in text
    assert "Found 2 Synthesia note color families." in text
    assert "Left/Right refer to Synthesia note colors, not the physical side of the keyboard." in text
    assert "Left White: found" in text
    assert "Right Black: not present in this video" in text
    assert "(255, 0, 0)" not in text
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_auto_detect_tuning_dialog.py::test_auto_detect_tuning_dialog_uses_user_guidance_copy tests/test_assisted_calibration_copy.py -v
```

Expected: FAIL because the old tuning labels and raw RGB proposal summary are still present.

- [ ] **Step 4: Implement Auto-Detect Tuning copy**

In `_setup_ui()` in `auto_detect_tuning_dialog.py`, after the fallback banner block and before `controls_row`, add:

```python
self.guidance_label = QLabel(
    QCoreApplication.translate(
        "AutoDetectTuningDialog",
        "Check the overlays on the video. If they line up with the keys, click Save. If the edges are off, adjust the edge controls.",
    )
)
self.guidance_label.setWordWrap(True)
self.guidance_label.setStyleSheet("color: #444;")
layout.addWidget(self.guidance_label)
```

Change the reset button construction:

```python
self.reset_all_button = QPushButton(
    QCoreApplication.translate("AutoDetectTuningDialog", "Reset to Recommended Settings")
)
self.reset_all_button.clicked.connect(self._reset_all_to_defaults)
controls_row.addWidget(self.reset_all_button)
```

Replace the local `tabs` variable with `self.tabs`:

```python
self.tabs = QTabWidget()
self.tabs.tabBar().setExpanding(False)
self.tabs.setStyleSheet("QTabWidget::tab-bar { alignment: right; }")
self.tabs.addTab(
    self._build_param_tab(get_basic_auto_detect_param_keys()),
    QCoreApplication.translate("AutoDetectTuningDialog", "Basic"),
)
self.tabs.addTab(
    self._build_param_tab(get_advanced_auto_detect_param_keys()),
    QCoreApplication.translate("AutoDetectTuningDialog", "Advanced Detector Settings"),
)
layout.addWidget(self.tabs, 1)
```

- [ ] **Step 5: Implement assisted calibration summary copy**

Replace `_proposal_summary_text()` in `calibration_wizard_controller.py` with:

```python
def _proposal_summary_text(self, proposal) -> str:
    slot_labels = {
        "LW": translate("CalibrationWizardController", "Left White"),
        "LB": translate("CalibrationWizardController", "Left Black"),
        "RW": translate("CalibrationWizardController", "Right White"),
        "RB": translate("CalibrationWizardController", "Right Black"),
    }
    lines = [
        translate(
            "CalibrationWizardController",
            "Assisted calibration found {count} possible pressed-key samples.",
        ).format(count=proposal.candidate_count),
        translate(
            "CalibrationWizardController",
            "Found {count} Synthesia note color families.",
        ).format(count=proposal.assignment_result.family_count),
        translate(
            "CalibrationWizardController",
            "Left/Right refer to Synthesia note colors, not the physical side of the keyboard.",
        ),
    ]
    for slot in ("LW", "LB", "RW", "RB"):
        assignment = proposal.assignment_result.assignments.get(slot)
        if assignment is None:
            continue
        label = slot_labels[slot]
        if not assignment.enabled:
            lines.append(
                translate(
                    "CalibrationWizardController",
                    "{label}: not present in this video",
                ).format(label=label)
            )
        elif assignment.rgb is not None:
            lines.append(
                translate(
                    "CalibrationWizardController",
                    "{label}: found",
                ).format(label=label)
            )
        else:
            lines.append(
                translate(
                    "CalibrationWizardController",
                    "{label}: not found",
                ).format(label=label)
            )
    return "\n".join(lines)
```

- [ ] **Step 6: Run focused dialog tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_auto_detect_tuning_dialog.py tests/test_assisted_calibration_copy.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
git add synthesia2midi/synthesia2midi/gui/auto_detect_tuning_dialog.py synthesia2midi/synthesia2midi/gui/calibration_wizard_controller.py tests/test_auto_detect_tuning_dialog.py tests/test_assisted_calibration_copy.py
git commit -m "Clarify auto-detect calibration guidance"
```

---

## Task 5: Settings Section Copy For Detection, Spark, MIDI, Trim, Optional

**Files:**
- Modify: `synthesia2midi/synthesia2midi/gui/controls_qt.py`
- Modify: `tests/test_controls_qt.py`
- Modify: `tests/test_main_window_layout.py`

**Interfaces:**
- Produces: visible labels for symptom-based settings explanations
- Keeps: existing sliders, checkboxes, ranges, signals, and app-state updates

- [ ] **Step 1: Write failing settings-copy tests**

Append to `tests/test_controls_qt.py`:

```python
def _all_label_texts(widget):
    return [label.text() for label in widget.findChildren(QLabel)]


def _all_group_titles(widget):
    return [group.title() for group in widget.findChildren(QGroupBox)]


def test_detection_section_uses_sensitivity_and_symptom_copy():
    QApplication.instance() or QApplication([])
    panel = ControlPanelQt()
    try:
        texts = _all_label_texts(panel)
        titles = _all_group_titles(panel)

        assert "Detection Sensitivity" in titles
        assert "Detection Threshold" not in titles
        assert "Detection Sensitivity:" in texts
        assert "Missing notes? Lower it. Extra notes? Raise it." in texts
    finally:
        panel.close()
        panel.deleteLater()


def test_spark_midi_trim_optional_sections_use_plain_recovery_copy():
    QApplication.instance() or QApplication([])
    panel = ControlPanelQt()
    try:
        texts = _all_label_texts(panel)
        titles = _all_group_titles(panel)

        assert "Use this only if repeated notes merge into one long note." in texts
        assert panel.spark_roi_select_button.text() == "Select Spark Area Above Keys"
        assert "Convert Only Part of the Video" in titles
        assert "This affects MIDI creation only. It does not trim or change the video session." in texts
        assert "Permanently Trim Project" in titles
        assert (
            "Most users should use MIDI range instead. Trim changes the working video session, not the original video file."
        ) in texts
        assert panel.trim_video_button.text() == "Permanently Trim Project"
        assert panel.hand_assignment_cb.text() == "Put each hand/color on a separate MIDI channel"
        assert "Use this only if the video uses different colors for left and right hand notes." in texts
    finally:
        panel.close()
        panel.deleteLater()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_controls_qt.py::test_detection_section_uses_sensitivity_and_symptom_copy tests/test_controls_qt.py::test_spark_midi_trim_optional_sections_use_plain_recovery_copy -v
```

Expected: FAIL because old labels are still present.

- [ ] **Step 3: Update Detection copy**

In `_create_basic_detection_tab()`:

Replace help lines with:

```python
help_lines = [
    translate("ControlPanelQt", "Before tuning detection: capture a no-key frame and at least one pressed-key example."),
    translate("ControlPanelQt", "Detection Sensitivity: main setting for pressed vs unpressed keys."),
    translate("ControlPanelQt", "Missing notes? Lower it. Extra notes? Raise it."),
    translate(
        "ControlPanelQt",
        "Histogram Detection helps when pressed colors have gradients or uneven lighting.",
    ),
    translate(
        "ControlPanelQt",
        "Delta Detection helps when pressed colors fade in or out instead of switching cleanly.",
    ),
    translate("ControlPanelQt", "Black Key Filter reduces false black-key notes caused by nearby overlays."),
]
```

Change the threshold group title and label:

```python
threshold_group = QGroupBox(translate("ControlPanelQt", "Detection Sensitivity"))
...
threshold_layout.addWidget(QLabel(translate("ControlPanelQt", "Detection Sensitivity:")))
threshold_layout.addWidget(QLabel(translate("ControlPanelQt", "Missing notes? Lower it. Extra notes? Raise it.")))
```

Keep the existing slider range, default, value label, and signal.

- [ ] **Step 4: Update Spark copy**

In `_create_spark_detection_tab()`, add this visible label at the top of `main_layout` before the checkbox:

```python
spark_guidance_label = QLabel(
    translate("ControlPanelQt", "Use this only if repeated notes merge into one long note.")
)
spark_guidance_label.setWordWrap(True)
spark_guidance_label.setStyleSheet("color: #555;")
main_layout.addWidget(spark_guidance_label)
```

Change spark ROI button text and tooltip:

```python
self.spark_roi_select_button = QPushButton(translate("ControlPanelQt", "Select Spark Area Above Keys"))
...
self.spark_roi_select_button.setToolTip(
    translate("ControlPanelQt", "Select the area above the keys where spark bars and flashes appear.")
)
```

Change the toggle tooltip:

```python
self.spark_roi_toggle_button.setToolTip(
    translate("ControlPanelQt", "Show or hide the spark area overlay on the video.")
)
```

Do not rename internal signal names or variables in this phase.

- [ ] **Step 5: Update MIDI range copy**

In `_create_midi_settings_tab()`, change the processing range group:

```python
processing_range_group = QGroupBox(translate("ControlPanelQt", "Convert Only Part of the Video"))
processing_range_layout = QVBoxLayout(processing_range_group)
processing_hint = QLabel(
    translate(
        "ControlPanelQt",
        "This affects MIDI creation only. It does not trim or change the video session.",
    )
)
processing_hint.setWordWrap(True)
processing_hint.setStyleSheet("color: #555;")
processing_range_layout.addWidget(processing_hint)
```

Keep the frame controls below this hint.

- [ ] **Step 6: Update Trim copy and confirmation**

In `_create_video_trim_tab()`, change:

```python
trim_group = QGroupBox(translate("ControlPanelQt", "Permanently Trim Project"))
...
trim_warning = QLabel(
    translate(
        "ControlPanelQt",
        "Most users should use MIDI range instead. Trim changes the working video session, not the original video file.",
    )
)
trim_warning.setWordWrap(True)
trim_warning.setStyleSheet("color: #8a4b00; font-weight: 600;")
trim_layout.addWidget(trim_warning)
```

Change trim button:

```python
self.trim_video_button = QPushButton(translate("ControlPanelQt", "Permanently Trim Project"))
```

In `_handle_trim_video_request()`, remove emoji from the window title and Yes button:

```python
msg_box.setWindowTitle(translate("ControlPanelQt", "Permanently Trim Project"))
...
msg_box.setText(
    translate(
        "ControlPanelQt",
        "<b>This will permanently trim the working video session.</b><br><br>"
        "Frames outside {start_frame} to {end_text} will be unavailable in this project session.<br><br>"
        "Most users should cancel and use the MIDI range controls instead.",
    ).format(start_frame=start_frame, end_text=end_text)
)
...
yes_button.setText(translate("ControlPanelQt", "Trim Project"))
```

Keep `QMessageBox.Cancel` as the default.

- [ ] **Step 7: Update Optional copy**

In `_create_optional_settings_tab()`:

```python
self.hand_assignment_cb = QCheckBox(
    translate("ControlPanelQt", "Put each hand/color on a separate MIDI channel")
)
self.hand_assignment_cb.toggled.connect(self.hand_assignment_toggled.emit)
optional_layout.addWidget(self.hand_assignment_cb)

hand_assignment_hint = QLabel(
    translate(
        "ControlPanelQt",
        "Use this only if the video uses different colors for left and right hand notes.",
    )
)
hand_assignment_hint.setWordWrap(True)
hand_assignment_hint.setStyleSheet("color: #555;")
optional_layout.addWidget(hand_assignment_hint)
```

- [ ] **Step 8: Run focused settings-copy tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_controls_qt.py::test_detection_section_uses_sensitivity_and_symptom_copy tests/test_controls_qt.py::test_spark_midi_trim_optional_sections_use_plain_recovery_copy tests/test_main_window_layout.py::test_spark_roi_controls_stack_and_stay_inside_panel -v
```

Expected: PASS.

- [ ] **Step 9: Commit**

Run:

```bash
git add synthesia2midi/synthesia2midi/gui/controls_qt.py tests/test_controls_qt.py tests/test_main_window_layout.py
git commit -m "Clarify advanced settings copy"
```

---

## Task 6: YouTube Dialog Fallback And Quality Wording

**Files:**
- Modify: `synthesia2midi/synthesia2midi/gui/youtube_download_dialog.py`
- Modify: `tests/test_youtube_download_dialog.py`

**Interfaces:**
- Keeps: `YOUTUBE_PREFERRED_BROWSER_KEY`, `YOUTUBE_AUTO_COOKIE_RETRY_KEY`, default browser loading/saving, auto retry behavior, and quality `itemData` values
- Changes: user-visible fallback group title, fallback hint, and quality labels

- [ ] **Step 1: Update failing YouTube wording tests**

Replace assertions in `test_dialog_uses_refresh_info_label_and_default_1080p_quality`:

```python
assert dialog.quality_combo.itemText(0) == "1080p - recommended for best MIDI detection"
assert dialog.quality_combo.itemText(1) == "720p - faster, may be less accurate"
assert dialog.quality_combo.itemText(2) == "480p - fastest, highest risk of bad calibration"
assert dialog.fallback_group.title() == "If YouTube blocks the download"
assert dialog.fallback_hint_label.text() == (
    "Synthesia2MIDI can retry using saved browser cookies only if YouTube blocks the normal download."
)
```

In `test_video_info_success_uses_real_available_quality_options`, update the note fixtures:

```python
"1080p": {"available": True, "actual_height": 720, "note": "recommended for best MIDI detection"},
"720p": {"available": True, "actual_height": 720, "note": "faster, may be less accurate"},
"480p": {"available": True, "actual_height": 360, "note": "fastest, highest risk of bad calibration"},
```

Keep the existing `itemData` assertions unchanged.

- [ ] **Step 2: Run YouTube test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest tests/test_youtube_download_dialog.py::test_dialog_uses_refresh_info_label_and_default_1080p_quality -v
```

Expected: FAIL because old quality and fallback labels are present.

- [ ] **Step 3: Implement fallback framing**

In `setup_ui()` in `youtube_download_dialog.py`, replace:

```python
fallback_group = QGroupBox(QCoreApplication.translate("YouTubeDownloadDialog", "YouTube Access Fallback"))
fallback_layout = QVBoxLayout()
```

with:

```python
self.fallback_group = QGroupBox(
    QCoreApplication.translate("YouTubeDownloadDialog", "If YouTube blocks the download")
)
fallback_layout = QVBoxLayout()
self.fallback_hint_label = QLabel(
    QCoreApplication.translate(
        "YouTubeDownloadDialog",
        "Synthesia2MIDI can retry using saved browser cookies only if YouTube blocks the normal download.",
    )
)
self.fallback_hint_label.setWordWrap(True)
fallback_layout.addWidget(self.fallback_hint_label)
```

Then replace later `fallback_group` references:

```python
self.fallback_group.setLayout(fallback_layout)
layout.addWidget(self.fallback_group)
```

- [ ] **Step 4: Implement quality labels**

In `_reset_quality_options()`, replace the three default labels:

```python
self.quality_combo.addItem(
    QCoreApplication.translate(
        "YouTubeDownloadDialog", "1080p - recommended for best MIDI detection"
    ),
    "1080p",
)
self.quality_combo.addItem(
    QCoreApplication.translate(
        "YouTubeDownloadDialog", "720p - faster, may be less accurate"
    ),
    "720p",
)
self.quality_combo.addItem(
    QCoreApplication.translate(
        "YouTubeDownloadDialog", "480p - fastest, highest risk of bad calibration"
    ),
    "480p",
)
```

In `_quality_option_label()`, keep the existing `Up to ...` behavior but allow the clearer note text to flow through:

```python
if note:
    return QCoreApplication.translate(
        "YouTubeDownloadDialog",
        "Up to {preset} ({actual_height}p source) - {note}",
    ).format(preset=preset, actual_height=actual_height, note=note)
```

Do not change quality `itemData` values.

- [ ] **Step 5: Run YouTube dialog tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_youtube_download_dialog.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add synthesia2midi/synthesia2midi/gui/youtube_download_dialog.py tests/test_youtube_download_dialog.py
git commit -m "Clarify YouTube download choices"
```

---

## Task 7: Localization Assets, Backlog Status, And Full Verification

**Files:**
- Modify: `backlog/tasks/task-17 - Phase-1-UX-hardening.md`
- Modify: `docs/localization/ui-string-manifest.json`
- Modify: `docs/localization/translation-agent-packet.json`
- Modify: `synthesia2midi/synthesia2midi/translations/synthesia2midi_*.ts`
- Modify: `synthesia2midi/synthesia2midi/translations/synthesia2midi_*.qm`
- Test: `tests/test_localization.py`
- Test: `tests/test_ui_string_audit.py`

**Interfaces:**
- Keeps: existing locale registry and language selector behavior
- Produces: updated Qt catalogs with no unfinished production messages and matching Python source strings

- [ ] **Step 1: Run extraction and audit commands**

Run:

```bash
.venv/bin/python -m synthesia2midi.tools.audit_ui_strings --output docs/localization/ui-string-manifest.json
.venv/bin/pyside6-lupdate -extensions py synthesia2midi/synthesia2midi \
  -ts \
  synthesia2midi/synthesia2midi/translations/synthesia2midi_es.ts \
  synthesia2midi/synthesia2midi/translations/synthesia2midi_ja.ts \
  synthesia2midi/synthesia2midi/translations/synthesia2midi_ru.ts \
  synthesia2midi/synthesia2midi/translations/synthesia2midi_zh_CN.ts \
  synthesia2midi/synthesia2midi/translations/synthesia2midi_ko.ts \
  synthesia2midi/synthesia2midi/translations/synthesia2midi_pt_BR.ts
.venv/bin/python -m synthesia2midi.tools.export_translation_packet \
  --source-ts synthesia2midi/synthesia2midi/translations/synthesia2midi_es.ts \
  --output docs/localization/translation-agent-packet.json
```

Expected: Commands exit 0. The `.ts` files may contain unfinished messages for the new Phase 1 strings at this point.

- [ ] **Step 2: Complete new production catalog translations**

For every production `.ts` file, search for unfinished messages:

```bash
rg -n "type=\"unfinished\"" synthesia2midi/synthesia2midi/translations
```

For each new Phase 1 source string, write a concise first-pass production translation in:

```text
synthesia2midi/synthesia2midi/translations/synthesia2midi_es.ts
synthesia2midi/synthesia2midi/translations/synthesia2midi_ja.ts
synthesia2midi/synthesia2midi/translations/synthesia2midi_ru.ts
synthesia2midi/synthesia2midi/translations/synthesia2midi_zh_CN.ts
synthesia2midi/synthesia2midi/translations/synthesia2midi_ko.ts
synthesia2midi/synthesia2midi/translations/synthesia2midi_pt_BR.ts
```

Translation rules:

```text
Preserve Synthesia2MIDI, Synthesia, MIDI, YouTube, FFmpeg, Rust, browser names, file extensions, URLs, keyboard note names, and Python format fields such as {count}, {label}, {start_frame}, and {end_text}.
Keep UI labels short.
Do not translate config keys, path-like values, or quality itemData values such as 1080p, 720p, and 480p.
```

Use `apply_patch` for manual `.ts` edits. Do not leave any production message empty or unfinished.

- [ ] **Step 3: Compile every production translation**

Run:

```bash
for ts_file in synthesia2midi/synthesia2midi/translations/synthesia2midi_*.ts; do
  locale_name=$(basename "$ts_file" .ts | sed 's/^synthesia2midi_//')
  .venv/bin/pyside6-lrelease "$ts_file" -qm "synthesia2midi/synthesia2midi/translations/synthesia2midi_${locale_name}.qm"
done
```

Expected: each locale reports generated translations and writes its `.qm`.

- [ ] **Step 4: Run localization tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_localization.py tests/test_ui_string_audit.py -v
```

Expected: PASS. If `test_translation_agent_packet_matches_source_catalog` fails, rerun `export_translation_packet` after final `.ts` edits.

- [ ] **Step 5: Run focused GUI tests**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_controls_qt.py \
  tests/test_main_window_layout.py \
  tests/test_calibration_wizard_copy.py \
  tests/test_assisted_calibration_copy.py \
  tests/test_auto_detect_tuning_dialog.py \
  tests/test_youtube_download_dialog.py \
  tests/test_startup_dialog.py \
  -v
```

Expected: PASS.

- [ ] **Step 6: Run full verification gate**

Run:

```bash
git diff --check
.venv/bin/python -m compileall -q synthesia2midi
.venv/bin/python -m pytest
```

Expected: all commands exit 0.

- [ ] **Step 7: Update Backlog acceptance**

In `backlog/tasks/task-17 - Phase-1-UX-hardening.md`, change:

```yaml
status: To Do
```

to:

```yaml
status: Done
```

Check every acceptance item that is satisfied:

```markdown
- [x] #1 The Convert area explains the first missing prerequisite instead of showing ready text while disabled.
- [x] #2 Calibration shows visible instructions for keyboard-box selection, no-key frame capture, and pressed-key examples without requiring Help expansion.
- [x] #3 Left/Right pressed-key terminology remains visible and is clarified as Synthesia note color/family language rather than physical keyboard position.
- [x] #4 Overlay quick adjustments show current values and reset controls, including Left Slant and Right Slant.
- [x] #5 Calibration Wizard and Auto-Detect Tuning use plain-language instructions while preserving existing detection/tuning behavior.
- [x] #6 Detection, Spark, MIDI range, Trim, Optional, and YouTube fallback settings are reframed with clearer user-facing copy.
- [x] #7 Destructive Trim is clearly separated from non-destructive MIDI processing range.
- [x] #8 Existing saved configs, overlay sidecars, detection parameters, and conversion behavior remain compatible.
- [x] #9 Tests cover the changed visible UI behavior and localization/audit gates are updated for changed strings.
```

Append the verification commands actually run under `## Verification`.

- [ ] **Step 8: Commit**

Run:

```bash
git add \
  backlog/tasks/task-17\ -\ Phase-1-UX-hardening.md \
  docs/localization/ui-string-manifest.json \
  docs/localization/translation-agent-packet.json \
  synthesia2midi/synthesia2midi/translations \
  tests/test_localization.py \
  tests/test_ui_string_audit.py
git commit -m "Update localization for UX hardening copy"
```

If `tests/test_localization.py` and `tests/test_ui_string_audit.py` did not need source edits, omit them from `git add`.

---

## Self-Review Checklist

- [ ] Spec coverage: Task 1 covers conversion readiness; Task 2 covers Calibration and Wizard copy; Task 3 covers quick-adjust values/reset; Task 4 covers Auto-Detect and assisted confirmation copy; Task 5 covers Detection/Spark/MIDI/Trim/Optional; Task 6 covers YouTube; Task 7 covers localization, backlog, and final verification.
- [ ] Scope check: no planned detector, conversion, config persistence, path layout, `main.py` workflow, or Rust touch-up editor changes.
- [ ] Left/Right terminology: preserved in buttons and labels, clarified as Synthesia note colors/families.
- [ ] User-visible strings: every new string is shown inside `translate(...)` or `QCoreApplication.translate(...)`.
- [ ] Type consistency: `ConversionReadiness`, `_conversion_readiness`, `_update_conversion_readiness_display`, `_apply_overlay_adjustment`, and `_reset_overlay_adjustment` names match across tests and implementation steps.
- [ ] Test-first flow: every behavior-changing task starts with a failing test command, then implementation, then passing focused tests, then commit.
- [ ] Verification: final gate includes `git diff --check`, compileall, full pytest, localization audit, lupdate, lrelease, and focused GUI tests.
