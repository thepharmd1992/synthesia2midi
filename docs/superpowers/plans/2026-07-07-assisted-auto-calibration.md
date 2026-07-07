# Assisted Auto-Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build V1 assisted calibration after manual keyboard-box selection: auto-capture unlit references, warn on likely-lit baseline frames, scan for lit exemplars, assign color-family slots, and confirm before saving.

**Architecture:** Keep GUI orchestration in `CalibrationWizardController`/`CalibrationWorkflow`, and put frame/overlay analysis in pure detection helpers. The analyzer returns an explicit proposal; UI code applies that proposal only after user confirmation.

**Tech Stack:** Python 3.10+, PySide6, OpenCV, NumPy, pytest, existing `OverlayConfig`, existing `ConfigManager`, existing Qt translation flow.

## Global Constraints

- V1 does not implement fully automatic keyboard ROI discovery.
- The user still draws the keyboard bounding box manually on a good unlit frame.
- The app must not assign `LW/LB/RW/RB` by physical keyboard position.
- Treat `LW/LB/RW/RB` as legacy color-family slots: family A white/black and family B white/black.
- The unlit-frame warning is soft and bypassable.
- The scan must use overlay ROIs, not full-frame object detection or a new ML dependency.
- Default tests must not require real videos, network access, or visible GUI windows.
- GUI code must keep one-way dependencies: `GUI -> workflows -> detection -> core`.
- New user-visible strings must use `QCoreApplication.translate` and the localization audit/translation files must be updated.
- No git worktrees and no pushes unless the user explicitly requests them.

---

## File Structure

- Create `synthesia2midi/synthesia2midi/detection/assisted_calibration.py`
  - Pure dataclasses and deterministic helper functions for overlay sampling, unlit assessment, candidate scanning, family assignment, and proposal application.
- Modify `synthesia2midi/synthesia2midi/workflows/calibration.py`
  - Reuse the unlit-frame guard in the existing "Calibrate Unlit All Keys" path.
  - Add a helper that captures unlit references from an explicit RGB frame and overlay list.
- Modify `synthesia2midi/synthesia2midi/gui/calibration_wizard_controller.py`
  - Run the assisted flow after successful keyboard-region auto-detection.
  - Show unlit soft warning, progress, and confirmation dialogs.
- Modify `synthesia2midi/synthesia2midi/gui/wizard.py`
  - Preserve the baseline frame and selected ROI context already captured during auto-detect.
  - Expose enough context for the controller to run assisted calibration without re-reading the current canvas frame.
- Add `synthesia2midi/synthesia2midi/tools/probe_assisted_calibration.py`
  - Optional local probe for real-video validation against a saved target INI/overlays; excluded from default tests.
- Add `tests/test_assisted_calibration.py`
  - Unit tests for pure analysis helpers.
- Modify `tests/test_bugfix_regressions.py`
  - Controller/workflow regression tests for warning, cancellation, and proposal application.
- Modify localization assets:
  - `docs/localization/ui-string-manifest.json`
  - `docs/localization/translation-agent-packet.json`
  - `synthesia2midi/synthesia2midi/translations/synthesia2midi_*.ts`
  - `synthesia2midi/synthesia2midi/translations/synthesia2midi_*.qm`
- Modify `backlog/tasks/task-15 - Add-assisted-auto-calibration-after-keyboard-box.md`
  - Check off acceptance criteria only as they are implemented and verified.

---

### Task 1: Pure Data Models And Overlay Sampling

**Files:**
- Create: `synthesia2midi/synthesia2midi/detection/assisted_calibration.py`
- Test: `tests/test_assisted_calibration.py`

**Interfaces:**
- Consumes: `OverlayConfig`, RGB `np.ndarray` frames.
- Produces:
  - `overlay_note_label(overlay: OverlayConfig) -> str`
  - `overlay_key_color(overlay: OverlayConfig) -> str`
  - `sample_overlay_rgb(frame_rgb: np.ndarray, overlay: OverlayConfig) -> tuple[int, int, int] | None`
  - `sample_overlay_bgr(frame_rgb: np.ndarray, overlay: OverlayConfig) -> np.ndarray | None`
  - dataclasses used by later tasks.

- [ ] **Step 1: Write failing sampling tests**

Add `tests/test_assisted_calibration.py`:

```python
import numpy as np

from synthesia2midi.app_config import OverlayConfig
from synthesia2midi.detection.assisted_calibration import (
    overlay_key_color,
    overlay_note_label,
    sample_overlay_bgr,
    sample_overlay_rgb,
)


def _overlay(key_id=1, note="C", octave=4, x=1, y=1, width=3, height=2, key_type="LW"):
    return OverlayConfig(
        key_id=key_id,
        note_octave=octave,
        note_name_in_octave=note,
        x=x,
        y=y,
        width=width,
        height=height,
        key_type=key_type,
    )


def test_overlay_sampling_uses_clipped_integer_roi():
    frame = np.zeros((5, 6, 3), dtype=np.uint8)
    frame[1:3, 1:4] = (10, 20, 30)

    assert sample_overlay_rgb(frame, _overlay()) == (10, 20, 30)
    assert sample_overlay_bgr(frame, _overlay()).mean(axis=(0, 1)).astype(int).tolist() == [30, 20, 10]


def test_overlay_sampling_returns_none_for_empty_roi():
    frame = np.zeros((5, 6, 3), dtype=np.uint8)

    assert sample_overlay_rgb(frame, _overlay(x=99, y=99)) is None
    assert sample_overlay_bgr(frame, _overlay(x=99, y=99)) is None


def test_overlay_note_label_and_key_color_use_existing_overlay_data():
    assert overlay_note_label(_overlay(note="E", octave=4)) == "E4"
    assert overlay_key_color(_overlay(key_type="LB")) == "B"
    assert overlay_key_color(_overlay(key_type="RW")) == "W"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_assisted_calibration.py -q
```

Expected: fail with `ModuleNotFoundError` for `synthesia2midi.detection.assisted_calibration`.

- [ ] **Step 3: Implement dataclasses and sampling helpers**

Create `synthesia2midi/synthesia2midi/detection/assisted_calibration.py`:

```python
"""Assisted calibration analysis helpers.

This module is intentionally GUI-free. It works from explicit RGB frames,
overlay rectangles, and callbacks supplied by workflow/UI code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Literal, Optional, Sequence, Tuple

import cv2
import numpy as np

from synthesia2midi.app_config import OverlayConfig


KeyColor = Literal["W", "B"]
AssessmentStatus = Literal["clean", "warning", "unknown"]
FrameProvider = Callable[[int], Optional[np.ndarray]]
ProgressCallback = Callable[[int, int], bool]


@dataclass(frozen=True)
class LikelyLitOverlay:
    key_id: int
    note_label: str
    key_color: KeyColor
    rgb: Tuple[int, int, int]
    delta: float
    saturation: float
    confidence: float


@dataclass(frozen=True)
class UnlitFrameAssessment:
    status: AssessmentStatus
    likely_lit: Tuple[LikelyLitOverlay, ...] = ()
    reason: str = ""

    @property
    def should_warn(self) -> bool:
        return self.status == "warning" and bool(self.likely_lit)


@dataclass(frozen=True)
class ExemplarCandidate:
    slot_color: KeyColor
    key_id: int
    note_label: str
    frame_index: int
    rgb: Tuple[int, int, int]
    hsv: Tuple[float, float, float]
    delta_from_unlit: float
    confidence: float
    hist: Optional[np.ndarray] = field(default=None, compare=False)


@dataclass(frozen=True)
class AssignedExemplar:
    slot: str
    rgb: Optional[Tuple[int, int, int]]
    hist: Optional[np.ndarray]
    source: Optional[ExemplarCandidate]
    enabled: bool


@dataclass(frozen=True)
class ExemplarAssignmentResult:
    assignments: Dict[str, AssignedExemplar]
    missing_slots: Tuple[str, ...]
    disabled_slots: Tuple[str, ...]
    family_count: int
    confidence: float


@dataclass(frozen=True)
class AssistedCalibrationProposal:
    baseline_frame_index: int
    unlit_assessment: UnlitFrameAssessment
    assignment_result: ExemplarAssignmentResult
    scanned_frame_count: int
    candidate_count: int
    canceled: bool = False


@dataclass(frozen=True)
class ExemplarScanSettings:
    coarse_stride: int = 10
    refine_radius: int = 5
    min_rgb_delta: float = 35.0
    min_saturation: float = 35.0
    max_candidates_per_key: int = 6


def overlay_note_label(overlay: OverlayConfig) -> str:
    return overlay.get_full_note_name()


def overlay_key_color(overlay: OverlayConfig) -> KeyColor:
    suffix = (overlay.key_type or "")[-1:]
    return "B" if suffix == "B" else "W"


def _overlay_bounds(frame_rgb: np.ndarray, overlay: OverlayConfig) -> Optional[Tuple[int, int, int, int]]:
    height, width = frame_rgb.shape[:2]
    x1 = max(0, int(round(overlay.x)))
    y1 = max(0, int(round(overlay.y)))
    x2 = min(width, int(round(overlay.x + overlay.width)))
    y2 = min(height, int(round(overlay.y + overlay.height)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def sample_overlay_rgb(frame_rgb: np.ndarray, overlay: OverlayConfig) -> Optional[Tuple[int, int, int]]:
    bounds = _overlay_bounds(frame_rgb, overlay)
    if bounds is None:
        return None
    x1, y1, x2, y2 = bounds
    roi = frame_rgb[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    rgb = roi.mean(axis=(0, 1)).round().astype(int)
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


def sample_overlay_bgr(frame_rgb: np.ndarray, overlay: OverlayConfig) -> Optional[np.ndarray]:
    bounds = _overlay_bounds(frame_rgb, overlay)
    if bounds is None:
        return None
    x1, y1, x2, y2 = bounds
    roi_rgb = frame_rgb[y1:y2, x1:x2]
    if roi_rgb.size == 0:
        return None
    return cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
.venv/bin/python -m pytest tests/test_assisted_calibration.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add synthesia2midi/synthesia2midi/detection/assisted_calibration.py tests/test_assisted_calibration.py
git commit -m "feat: add assisted calibration sampling helpers"
```

---

### Task 2: Reusable Unlit-Frame Guard

**Files:**
- Modify: `synthesia2midi/synthesia2midi/detection/assisted_calibration.py`
- Test: `tests/test_assisted_calibration.py`

**Interfaces:**
- Consumes: `sample_overlay_rgb`, overlays with optional `unlit_reference_color`.
- Produces: `assess_unlit_frame(frame_rgb: np.ndarray, overlays: Sequence[OverlayConfig]) -> UnlitFrameAssessment`.

- [ ] **Step 1: Write failing tests for clean and warning assessments**

Append to `tests/test_assisted_calibration.py`:

```python
from synthesia2midi.detection.assisted_calibration import assess_unlit_frame


def test_unlit_frame_guard_returns_clean_for_uniform_keyboard_groups():
    frame = np.zeros((20, 80, 3), dtype=np.uint8)
    overlays = []
    for i in range(4):
        overlays.append(_overlay(key_id=i, note="C", octave=4, x=i * 10, y=0, width=8, height=8, key_type="LW"))
        frame[0:8, i * 10:i * 10 + 8] = (245, 245, 235)
    for i in range(4):
        overlays.append(_overlay(key_id=10 + i, note="C♯", octave=4, x=i * 10, y=10, width=8, height=8, key_type="LB"))
        frame[10:18, i * 10:i * 10 + 8] = (25, 25, 25)

    assessment = assess_unlit_frame(frame, overlays)

    assert assessment.status == "clean"
    assert assessment.likely_lit == ()


def test_unlit_frame_guard_warns_with_likely_lit_note_name():
    frame = np.zeros((20, 80, 3), dtype=np.uint8)
    overlays = []
    for i in range(6):
        overlays.append(_overlay(key_id=i, note="E", octave=4, x=i * 10, y=0, width=8, height=8, key_type="LW"))
        frame[0:8, i * 10:i * 10 + 8] = (245, 245, 235)
    overlays[2].note_name_in_octave = "G"
    frame[0:8, 20:28] = (235, 150, 40)

    assessment = assess_unlit_frame(frame, overlays)

    assert assessment.status == "warning"
    assert [item.note_label for item in assessment.likely_lit] == ["G4"]
    assert assessment.likely_lit[0].confidence > 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_assisted_calibration.py::test_unlit_frame_guard_returns_clean_for_uniform_keyboard_groups tests/test_assisted_calibration.py::test_unlit_frame_guard_warns_with_likely_lit_note_name -q
```

Expected: fail with `ImportError` for `assess_unlit_frame`.

- [ ] **Step 3: Implement `assess_unlit_frame`**

Append to `assisted_calibration.py`:

```python
def _rgb_distance(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
    return float(np.linalg.norm(np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)))


def _rgb_to_hsv_tuple(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    pixel = np.array([[rgb]], dtype=np.uint8)
    hsv = cv2.cvtColor(pixel, cv2.COLOR_RGB2HSV)[0, 0]
    return float(hsv[0]), float(hsv[1]), float(hsv[2])


def assess_unlit_frame(
    frame_rgb: np.ndarray,
    overlays: Sequence[OverlayConfig],
    *,
    min_group_delta: float = 45.0,
    min_reference_delta: float = 35.0,
    min_saturation_delta: float = 25.0,
    max_reported: int = 6,
) -> UnlitFrameAssessment:
    samples: list[tuple[OverlayConfig, KeyColor, Tuple[int, int, int], Tuple[float, float, float]]] = []
    for overlay in overlays:
        rgb = sample_overlay_rgb(frame_rgb, overlay)
        if rgb is None:
            continue
        samples.append((overlay, overlay_key_color(overlay), rgb, _rgb_to_hsv_tuple(rgb)))

    if len(samples) < 4:
        return UnlitFrameAssessment(status="unknown", reason="not enough overlay samples")

    likely: list[LikelyLitOverlay] = []
    for key_color in ("W", "B"):
        group = [sample for sample in samples if sample[1] == key_color]
        if len(group) < 3:
            continue

        group_rgbs = np.array([sample[2] for sample in group], dtype=np.float32)
        group_sats = np.array([sample[3][1] for sample in group], dtype=np.float32)
        median_rgb = tuple(np.median(group_rgbs, axis=0).round().astype(int).tolist())
        median_sat = float(np.median(group_sats))

        for overlay, _, rgb, hsv in group:
            group_delta = _rgb_distance(rgb, median_rgb)
            reference_delta = 0.0
            if overlay.unlit_reference_color is not None:
                reference_delta = _rgb_distance(rgb, overlay.unlit_reference_color)

            saturation_delta = hsv[1] - median_sat
            strong_group_outlier = group_delta >= min_group_delta and saturation_delta >= min_saturation_delta
            strong_reference_outlier = reference_delta >= min_reference_delta and hsv[1] >= 35.0
            if not strong_group_outlier and not strong_reference_outlier:
                continue

            confidence = min(1.0, max(group_delta / 120.0, reference_delta / 120.0))
            likely.append(
                LikelyLitOverlay(
                    key_id=overlay.key_id,
                    note_label=overlay_note_label(overlay),
                    key_color=overlay_key_color(overlay),
                    rgb=rgb,
                    delta=max(group_delta, reference_delta),
                    saturation=hsv[1],
                    confidence=confidence,
                )
            )

    if not likely:
        return UnlitFrameAssessment(status="clean")

    likely.sort(key=lambda item: (-item.confidence, item.note_label, item.key_id))
    return UnlitFrameAssessment(
        status="warning",
        likely_lit=tuple(likely[:max_reported]),
        reason="one or more overlays are color outliers for the unlit frame",
    )
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_assisted_calibration.py -q
```

Expected: all tests in `tests/test_assisted_calibration.py` pass.

- [ ] **Step 5: Commit**

```bash
git add synthesia2midi/synthesia2midi/detection/assisted_calibration.py tests/test_assisted_calibration.py
git commit -m "feat: detect likely lit keys in unlit frames"
```

---

### Task 3: Capture Unlit References From The Baseline Frame

**Files:**
- Modify: `synthesia2midi/synthesia2midi/detection/assisted_calibration.py`
- Modify: `synthesia2midi/synthesia2midi/workflows/calibration.py`
- Test: `tests/test_assisted_calibration.py`
- Test: `tests/test_bugfix_regressions.py`

**Interfaces:**
- Produces: `capture_unlit_references_from_frame(frame_rgb: np.ndarray, overlays: Sequence[OverlayConfig]) -> int`.
- `CalibrationWorkflow.handle_calibrate_unlit_all_keys()` uses `assess_unlit_frame` before overwriting existing unlit data.

- [ ] **Step 1: Write failing pure capture test**

Append to `tests/test_assisted_calibration.py`:

```python
from synthesia2midi.detection.assisted_calibration import capture_unlit_references_from_frame


def test_capture_unlit_references_sets_rgb_and_histogram():
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    frame[1:5, 1:5] = (100, 120, 140)
    overlay = _overlay(x=1, y=1, width=4, height=4)

    count = capture_unlit_references_from_frame(frame, [overlay])

    assert count == 1
    assert overlay.unlit_reference_color == (100, 120, 140)
    assert overlay.unlit_hist is not None
```

- [ ] **Step 2: Write failing workflow warning test**

Append to `tests/test_bugfix_regressions.py`:

```python
def test_unlit_calibration_warns_when_frame_has_likely_lit_key(monkeypatch):
    warnings = []
    infos = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: warnings.append(args) or QMessageBox.StandardButton.Cancel)
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: infos.append(args))

    app_state = AppState()
    app_state.overlays = [
        OverlayConfig(key_id=1, note_octave=4, note_name_in_octave="C", x=0, y=0, width=4, height=4, key_type="LW"),
        OverlayConfig(key_id=2, note_octave=4, note_name_in_octave="D", x=5, y=0, width=4, height=4, key_type="LW"),
        OverlayConfig(key_id=3, note_octave=4, note_name_in_octave="E", x=10, y=0, width=4, height=4, key_type="LW"),
        OverlayConfig(key_id=4, note_octave=4, note_name_in_octave="F", x=15, y=0, width=4, height=4, key_type="LW"),
    ]
    frame_rgb = np.full((8, 24, 3), (245, 245, 235), dtype=np.uint8)
    frame_rgb[0:4, 10:14] = (235, 150, 40)
    frame_bgr = frame_rgb[:, :, ::-1]
    canvas = SimpleNamespace(
        current_frame_rgb=frame_rgb,
        get_roi_bgr=lambda overlay: frame_bgr[int(overlay.y):int(overlay.y + overlay.height), int(overlay.x):int(overlay.x + overlay.width)],
    )
    parent = SimpleNamespace(keyboard_canvas=canvas, control_panel=SimpleNamespace(update_controls_from_state=lambda: None))
    workflow = CalibrationWorkflow(app_state, SimpleNamespace(), parent)

    workflow.handle_calibrate_unlit_all_keys()

    assert warnings
    assert "E4" in warnings[0][2]
    assert infos == []
    assert all(overlay.unlit_reference_color is None for overlay in app_state.overlays)
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_assisted_calibration.py::test_capture_unlit_references_sets_rgb_and_histogram tests/test_bugfix_regressions.py::test_unlit_calibration_warns_when_frame_has_likely_lit_key -q
```

Expected: pure test fails with missing import; workflow test fails because no warning is shown.

- [ ] **Step 4: Implement capture helper**

Append to `assisted_calibration.py`:

```python
def capture_unlit_references_from_frame(
    frame_rgb: np.ndarray,
    overlays: Sequence[OverlayConfig],
) -> int:
    from synthesia2midi.detection.roi_utils import get_hist_feature

    calibrated = 0
    for overlay in overlays:
        rgb = sample_overlay_rgb(frame_rgb, overlay)
        bgr = sample_overlay_bgr(frame_rgb, overlay)
        if rgb is None or bgr is None:
            continue
        overlay.unlit_reference_color = rgb
        overlay.unlit_hist = get_hist_feature(bgr)
        calibrated += 1
    return calibrated
```

- [ ] **Step 5: Wire soft warning into existing unlit calibration**

In `calibration.py`, update the QtCore import and add a module-level translate alias:

```python
from PySide6.QtCore import Qt, QCoreApplication

translate = QCoreApplication.translate
```

In `CalibrationWorkflow.handle_calibrate_unlit_all_keys`, before the loop that writes `overlay.unlit_reference_color`, add:

```python
        frame_rgb = getattr(keyboard_canvas, "current_frame_rgb", None)
        if frame_rgb is not None:
            from PySide6.QtWidgets import QMessageBox
            from synthesia2midi.detection.assisted_calibration import assess_unlit_frame

            assessment = assess_unlit_frame(frame_rgb, self.app_state.overlays)
            if assessment.should_warn:
                note_list = ", ".join(item.note_label for item in assessment.likely_lit)
                response = QMessageBox.warning(
                    self.parent_widget,
                    translate("CalibrationWorkflow", "Unlit Frame May Contain Lit Keys"),
                    translate(
                        "CalibrationWorkflow",
                        "It looks like these keys may be lit: {notes}.\n\nMove to a frame where no keys are lit, or continue if this is expected.",
                    ).format(notes=note_list),
                    QMessageBox.StandardButton.Ignore | QMessageBox.StandardButton.Cancel,
                    QMessageBox.StandardButton.Cancel,
                )
                if response == QMessageBox.StandardButton.Cancel:
                    return
```

Then keep the existing unlit write loop unchanged for this task.

- [ ] **Step 6: Run focused tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_assisted_calibration.py tests/test_bugfix_regressions.py::test_unlit_calibration_warns_when_frame_has_likely_lit_key -q
```

Expected: tests pass.

- [ ] **Step 7: Commit**

```bash
git add synthesia2midi/synthesia2midi/detection/assisted_calibration.py synthesia2midi/synthesia2midi/workflows/calibration.py tests/test_assisted_calibration.py tests/test_bugfix_regressions.py
git commit -m "feat: warn before unlit calibration on lit frames"
```

---

### Task 4: Lit Exemplar Scanner

**Files:**
- Modify: `synthesia2midi/synthesia2midi/detection/assisted_calibration.py`
- Test: `tests/test_assisted_calibration.py`

**Interfaces:**
- Produces: `scan_lit_exemplar_candidates(frame_provider, overlays, start_frame, end_frame, settings=ExemplarScanSettings(), progress_callback=None) -> tuple[list[ExemplarCandidate], int, bool]`.
- `frame_provider(frame_index)` returns RGB frames or `None`.
- `progress_callback(current_frame, end_frame)` returns `False` to cancel.

- [ ] **Step 1: Write failing scanner tests**

Append to `tests/test_assisted_calibration.py`:

```python
from synthesia2midi.detection.assisted_calibration import ExemplarScanSettings, scan_lit_exemplar_candidates


def test_scanner_finds_lit_candidates_from_overlay_deltas():
    overlays = [
        _overlay(key_id=1, note="C", octave=4, x=0, y=0, width=4, height=4, key_type="LW"),
        _overlay(key_id=2, note="C♯", octave=4, x=5, y=0, width=4, height=4, key_type="LB"),
    ]
    overlays[0].unlit_reference_color = (245, 245, 235)
    overlays[1].unlit_reference_color = (25, 25, 25)

    frames = {}
    for index in range(0, 31):
        frame = np.zeros((8, 16, 3), dtype=np.uint8)
        frame[:, :] = (10, 10, 10)
        frame[0:4, 0:4] = (245, 245, 235)
        frame[0:4, 5:9] = (25, 25, 25)
        frames[index] = frame
    frames[20][0:4, 0:4] = (130, 165, 205)
    frames[21][0:4, 5:9] = (70, 110, 170)

    candidates, scanned, canceled = scan_lit_exemplar_candidates(
        lambda index: frames.get(index),
        overlays,
        0,
        30,
        settings=ExemplarScanSettings(coarse_stride=10, refine_radius=2, min_rgb_delta=30.0),
    )

    assert canceled is False
    assert scanned > 0
    assert {candidate.note_label for candidate in candidates} >= {"C4", "C♯4"}
    assert any(candidate.slot_color == "W" and candidate.rgb == (130, 165, 205) for candidate in candidates)
    assert any(candidate.slot_color == "B" and candidate.rgb == (70, 110, 170) for candidate in candidates)


def test_scanner_honors_cancel_callback():
    overlay = _overlay()
    overlay.unlit_reference_color = (245, 245, 235)
    frame = np.full((8, 8, 3), (245, 245, 235), dtype=np.uint8)

    candidates, scanned, canceled = scan_lit_exemplar_candidates(
        lambda _index: frame,
        [overlay],
        0,
        100,
        progress_callback=lambda _current, _end: False,
    )

    assert candidates == []
    assert scanned == 0
    assert canceled is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_assisted_calibration.py::test_scanner_finds_lit_candidates_from_overlay_deltas tests/test_assisted_calibration.py::test_scanner_honors_cancel_callback -q
```

Expected: fail with missing `scan_lit_exemplar_candidates`.

- [ ] **Step 3: Implement scanner**

Append to `assisted_calibration.py`:

```python
def _candidate_from_sample(
    frame_index: int,
    overlay: OverlayConfig,
    rgb: Tuple[int, int, int],
    hist: Optional[np.ndarray],
) -> Optional[ExemplarCandidate]:
    if overlay.unlit_reference_color is None:
        return None
    delta = _rgb_distance(rgb, overlay.unlit_reference_color)
    hsv = _rgb_to_hsv_tuple(rgb)
    confidence = min(1.0, delta / 180.0)
    return ExemplarCandidate(
        slot_color=overlay_key_color(overlay),
        key_id=overlay.key_id,
        note_label=overlay_note_label(overlay),
        frame_index=frame_index,
        rgb=rgb,
        hsv=hsv,
        delta_from_unlit=delta,
        confidence=confidence,
        hist=hist,
    )


def _frame_candidate_for_overlay(
    frame_rgb: np.ndarray,
    frame_index: int,
    overlay: OverlayConfig,
    settings: ExemplarScanSettings,
) -> Optional[ExemplarCandidate]:
    rgb = sample_overlay_rgb(frame_rgb, overlay)
    bgr = sample_overlay_bgr(frame_rgb, overlay)
    if rgb is None or bgr is None or overlay.unlit_reference_color is None:
        return None
    delta = _rgb_distance(rgb, overlay.unlit_reference_color)
    hsv = _rgb_to_hsv_tuple(rgb)
    if delta < settings.min_rgb_delta or hsv[1] < settings.min_saturation:
        return None
    from synthesia2midi.detection.roi_utils import get_hist_feature

    return _candidate_from_sample(frame_index, overlay, rgb, get_hist_feature(bgr))


def scan_lit_exemplar_candidates(
    frame_provider: FrameProvider,
    overlays: Sequence[OverlayConfig],
    start_frame: int,
    end_frame: int,
    *,
    settings: ExemplarScanSettings = ExemplarScanSettings(),
    progress_callback: Optional[ProgressCallback] = None,
) -> Tuple[list[ExemplarCandidate], int, bool]:
    candidates_by_key: dict[int, list[ExemplarCandidate]] = {}
    scanned = 0
    end_frame = max(start_frame, end_frame)

    for frame_index in range(start_frame, end_frame + 1, max(1, settings.coarse_stride)):
        if progress_callback is not None and not progress_callback(frame_index, end_frame):
            return [], scanned, True

        frame = frame_provider(frame_index)
        if frame is None:
            continue
        scanned += 1

        for overlay in overlays:
            coarse_candidate = _frame_candidate_for_overlay(frame, frame_index, overlay, settings)
            if coarse_candidate is None:
                continue

            best = coarse_candidate
            refine_start = max(start_frame, frame_index - settings.refine_radius)
            refine_end = min(end_frame, frame_index + settings.refine_radius)
            for refined_index in range(refine_start, refine_end + 1):
                refined_frame = frame_provider(refined_index)
                if refined_frame is None:
                    continue
                refined_candidate = _frame_candidate_for_overlay(refined_frame, refined_index, overlay, settings)
                if refined_candidate is not None and refined_candidate.confidence > best.confidence:
                    best = refined_candidate

            bucket = candidates_by_key.setdefault(overlay.key_id, [])
            bucket.append(best)
            bucket.sort(key=lambda item: (-item.confidence, item.frame_index))
            del bucket[settings.max_candidates_per_key:]

    flattened = [candidate for bucket in candidates_by_key.values() for candidate in bucket]
    flattened.sort(key=lambda item: (-item.confidence, item.frame_index, item.key_id))
    return flattened, scanned, False
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_assisted_calibration.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add synthesia2midi/synthesia2midi/detection/assisted_calibration.py tests/test_assisted_calibration.py
git commit -m "feat: scan lit exemplar candidates"
```

---

### Task 5: Color-Family Assignment And Proposal Application

**Files:**
- Modify: `synthesia2midi/synthesia2midi/detection/assisted_calibration.py`
- Test: `tests/test_assisted_calibration.py`

**Interfaces:**
- Produces:
  - `assign_exemplar_slots(candidates: Sequence[ExemplarCandidate]) -> ExemplarAssignmentResult`
  - `apply_assisted_calibration_proposal(app_state: AppState, proposal: AssistedCalibrationProposal) -> None`

- [ ] **Step 1: Write failing assignment tests**

Append to `tests/test_assisted_calibration.py`:

```python
import cv2

from synthesia2midi.detection.assisted_calibration import (
    AssignedExemplar,
    AssistedCalibrationProposal,
    ExemplarCandidate,
    UnlitFrameAssessment,
    apply_assisted_calibration_proposal,
    assign_exemplar_slots,
)
from synthesia2midi.core.app_state import AppState


def _candidate(slot_color, rgb, frame_index=10, note="C4", key_id=1, confidence=0.9):
    hsv = cv2.cvtColor(np.array([[rgb]], dtype=np.uint8), cv2.COLOR_RGB2HSV)[0, 0]
    return ExemplarCandidate(
        slot_color=slot_color,
        key_id=key_id,
        note_label=note,
        frame_index=frame_index,
        rgb=rgb,
        hsv=(float(hsv[0]), float(hsv[1]), float(hsv[2])),
        delta_from_unlit=100.0,
        confidence=confidence,
        hist=np.array([1.0], dtype=np.float32),
    )


def test_assign_exemplar_slots_maps_two_color_families_by_hue_not_position():
    candidates = [
        _candidate("W", (130, 165, 205), key_id=50, note="D5"),
        _candidate("B", (70, 110, 170), key_id=30, note="C♯4"),
        _candidate("W", (243, 176, 68), key_id=10, note="A2"),
        _candidate("B", (243, 131, 46), key_id=20, note="A♯2"),
    ]

    result = assign_exemplar_slots(candidates)

    assert result.family_count == 2
    assert result.assignments["LW"].rgb == (130, 165, 205)
    assert result.assignments["LB"].rgb == (70, 110, 170)
    assert result.assignments["RW"].rgb == (243, 176, 68)
    assert result.assignments["RB"].rgb == (243, 131, 46)
    assert result.disabled_slots == ()


def test_assign_exemplar_slots_disables_absent_second_family():
    result = assign_exemplar_slots([
        _candidate("W", (130, 165, 205), key_id=1),
        _candidate("B", (70, 110, 170), key_id=2),
    ])

    assert result.family_count == 1
    assert result.assignments["LW"].enabled is True
    assert result.assignments["LB"].enabled is True
    assert result.assignments["RW"].enabled is False
    assert result.assignments["RB"].enabled is False
    assert result.disabled_slots == ("RW", "RB")


def test_apply_assisted_calibration_proposal_updates_colors_histograms_and_enabled_slots():
    app_state = AppState()
    assignment = assign_exemplar_slots([
        _candidate("W", (130, 165, 205), key_id=1),
        _candidate("B", (70, 110, 170), key_id=2),
    ])
    proposal = AssistedCalibrationProposal(
        baseline_frame_index=12,
        unlit_assessment=UnlitFrameAssessment(status="clean"),
        assignment_result=assignment,
        scanned_frame_count=3,
        candidate_count=2,
    )

    apply_assisted_calibration_proposal(app_state, proposal)

    assert app_state.detection.exemplar_lit_colors["LW"] == (130, 165, 205)
    assert app_state.detection.exemplar_lit_colors["LB"] == (70, 110, 170)
    assert app_state.detection.exemplar_lit_colors["RW"] is None
    assert app_state.detection.exemplar_lit_colors["RB"] is None
    assert app_state.detection.exemplar_key_type_enabled["RW"] is False
    assert app_state.detection.exemplar_key_type_enabled["RB"] is False
    assert app_state.unsaved_changes is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_assisted_calibration.py::test_assign_exemplar_slots_maps_two_color_families_by_hue_not_position tests/test_assisted_calibration.py::test_assign_exemplar_slots_disables_absent_second_family tests/test_assisted_calibration.py::test_apply_assisted_calibration_proposal_updates_colors_histograms_and_enabled_slots -q
```

Expected: fail with missing functions.

- [ ] **Step 3: Implement family assignment and proposal application**

Append to `assisted_calibration.py`:

```python
def _circular_hue_distance(a: float, b: float) -> float:
    delta = abs(a - b)
    return min(delta, 180.0 - delta)


def _family_hue(candidate: ExemplarCandidate) -> float:
    return candidate.hsv[0] if candidate.hsv[1] > 0 else _rgb_to_hsv_tuple(candidate.rgb)[0]


def _family_sort_key(bucket: list[ExemplarCandidate]) -> tuple[int, int, float]:
    mean_hue = float(np.mean([_family_hue(item) for item in bucket]))
    cool_family = 60.0 <= mean_hue <= 140.0
    first_frame = min(item.frame_index for item in bucket)
    return (0 if cool_family else 1, first_frame, mean_hue)


def _best_candidate(candidates: Iterable[ExemplarCandidate]) -> Optional[ExemplarCandidate]:
    ordered = sorted(candidates, key=lambda item: (-item.confidence, item.frame_index, item.key_id))
    return ordered[0] if ordered else None


def assign_exemplar_slots(
    candidates: Sequence[ExemplarCandidate],
    *,
    family_hue_threshold: float = 22.0,
) -> ExemplarAssignmentResult:
    family_buckets: list[list[ExemplarCandidate]] = []
    for candidate in sorted(candidates, key=lambda item: (-item.confidence, item.frame_index, item.key_id)):
        hue = _family_hue(candidate)
        target_bucket: Optional[list[ExemplarCandidate]] = None
        for bucket in family_buckets:
            bucket_hue = float(np.mean([_family_hue(item) for item in bucket]))
            if _circular_hue_distance(hue, bucket_hue) <= family_hue_threshold:
                target_bucket = bucket
                break
        if target_bucket is None:
            if len(family_buckets) >= 2:
                continue
            target_bucket = []
            family_buckets.append(target_bucket)
        target_bucket.append(candidate)

    family_buckets.sort(key=_family_sort_key)
    slot_pairs = [("LW", "LB"), ("RW", "RB")]
    assignments: Dict[str, AssignedExemplar] = {}
    missing: list[str] = []
    disabled: list[str] = []
    confidences: list[float] = []

    for family_index, slots in enumerate(slot_pairs):
        bucket = family_buckets[family_index] if family_index < len(family_buckets) else []
        for slot, key_color in zip(slots, ("W", "B")):
            source = _best_candidate(item for item in bucket if item.slot_color == key_color)
            if source is None:
                enabled = family_index < len(family_buckets)
                assignments[slot] = AssignedExemplar(slot=slot, rgb=None, hist=None, source=None, enabled=enabled)
                if enabled:
                    missing.append(slot)
                else:
                    disabled.append(slot)
                continue
            assignments[slot] = AssignedExemplar(
                slot=slot,
                rgb=source.rgb,
                hist=source.hist,
                source=source,
                enabled=True,
            )
            confidences.append(source.confidence)

    confidence = float(np.mean(confidences)) if confidences else 0.0
    return ExemplarAssignmentResult(
        assignments=assignments,
        missing_slots=tuple(missing),
        disabled_slots=tuple(disabled),
        family_count=len(family_buckets),
        confidence=confidence,
    )


def apply_assisted_calibration_proposal(app_state, proposal: AssistedCalibrationProposal) -> None:
    for slot, assignment in proposal.assignment_result.assignments.items():
        app_state.detection.exemplar_key_type_enabled[slot] = assignment.enabled
        app_state.detection.exemplar_lit_colors[slot] = assignment.rgb if assignment.enabled else None
        app_state.detection.exemplar_lit_histograms[slot] = assignment.hist if assignment.enabled else None
    app_state.unsaved_changes = True
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_assisted_calibration.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add synthesia2midi/synthesia2midi/detection/assisted_calibration.py tests/test_assisted_calibration.py
git commit -m "feat: assign assisted calibration color families"
```

---

### Task 6: Analyzer Facade And Local Probe Command

**Files:**
- Modify: `synthesia2midi/synthesia2midi/detection/assisted_calibration.py`
- Create: `synthesia2midi/synthesia2midi/tools/probe_assisted_calibration.py`
- Test: `tests/test_assisted_calibration.py`

**Interfaces:**
- Produces: `build_assisted_calibration_proposal(frame_provider, overlays, baseline_frame_index, end_frame, settings=ExemplarScanSettings(), progress_callback=None) -> AssistedCalibrationProposal`.
- Produces CLI module: `.venv/bin/python -m synthesia2midi.tools.probe_assisted_calibration --video <path> --overlays <path> --ini <path> --baseline-frame <n>`.

- [ ] **Step 1: Write failing analyzer test**

Append to `tests/test_assisted_calibration.py`:

```python
from synthesia2midi.detection.assisted_calibration import build_assisted_calibration_proposal


def test_build_assisted_calibration_proposal_combines_guard_scan_and_assignment():
    overlay = _overlay(key_id=1, x=0, y=0, width=4, height=4)
    baseline = np.full((8, 8, 3), (245, 245, 235), dtype=np.uint8)
    lit = baseline.copy()
    lit[0:4, 0:4] = (130, 165, 205)
    frames = {0: baseline, 10: lit}

    capture_unlit_references_from_frame(baseline, [overlay])
    proposal = build_assisted_calibration_proposal(
        lambda index: frames.get(index, baseline),
        [overlay],
        baseline_frame_index=0,
        end_frame=10,
        settings=ExemplarScanSettings(coarse_stride=10, refine_radius=0, min_rgb_delta=30.0),
    )

    assert proposal.baseline_frame_index == 0
    assert proposal.unlit_assessment.status == "clean"
    assert proposal.candidate_count == 1
    assert proposal.assignment_result.assignments["LW"].rgb == (130, 165, 205)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_assisted_calibration.py::test_build_assisted_calibration_proposal_combines_guard_scan_and_assignment -q
```

Expected: fail with missing function.

- [ ] **Step 3: Implement analyzer facade**

Append to `assisted_calibration.py`:

```python
def build_assisted_calibration_proposal(
    frame_provider: FrameProvider,
    overlays: Sequence[OverlayConfig],
    *,
    baseline_frame_index: int,
    end_frame: int,
    settings: ExemplarScanSettings = ExemplarScanSettings(),
    progress_callback: Optional[ProgressCallback] = None,
) -> AssistedCalibrationProposal:
    baseline_frame = frame_provider(baseline_frame_index)
    assessment = (
        assess_unlit_frame(baseline_frame, overlays)
        if baseline_frame is not None
        else UnlitFrameAssessment(status="unknown", reason="baseline frame unavailable")
    )
    candidates, scanned, canceled = scan_lit_exemplar_candidates(
        frame_provider,
        overlays,
        baseline_frame_index + 1,
        end_frame,
        settings=settings,
        progress_callback=progress_callback,
    )
    assignment = assign_exemplar_slots(candidates)
    return AssistedCalibrationProposal(
        baseline_frame_index=baseline_frame_index,
        unlit_assessment=assessment,
        assignment_result=assignment,
        scanned_frame_count=scanned,
        candidate_count=len(candidates),
        canceled=canceled,
    )
```

- [ ] **Step 4: Add local probe command**

Create `synthesia2midi/synthesia2midi/tools/probe_assisted_calibration.py`:

```python
"""Local assisted-calibration probe for real videos.

This is a developer tool. It is not part of the default pytest gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from synthesia2midi.app_config import OverlayConfig
from synthesia2midi.detection.assisted_calibration import (
    ExemplarScanSettings,
    build_assisted_calibration_proposal,
    capture_unlit_references_from_frame,
)
from synthesia2midi.video_loader import create_video_session


def _load_overlays(path: Path) -> list[OverlayConfig]:
    data = json.loads(path.read_text())
    raw_overlays = data if isinstance(data, list) else data.get("overlays", [])
    return [
        OverlayConfig(
            key_id=int(item["key_id"]),
            note_octave=int(item["note_octave"]),
            note_name_in_octave=str(item["note_name_in_octave"]),
            x=float(item["x"]),
            y=float(item["y"]),
            width=float(item["width"]),
            height=float(item["height"]),
            key_type=item.get("key_type"),
            unlit_reference_color=tuple(item["unlit_reference_color"]) if item.get("unlit_reference_color") else None,
        )
        for item in raw_overlays
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--overlays", required=True)
    parser.add_argument("--baseline-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--stride", type=int, default=10)
    args = parser.parse_args()

    session = create_video_session(args.video)
    overlays = _load_overlays(Path(args.overlays))
    end_frame = args.end_frame if args.end_frame is not None else session.total_frames - 1

    def frame_provider(index: int):
        success, frame_bgr = session.get_frame(index)
        if not success or frame_bgr is None:
            return None
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    baseline = frame_provider(args.baseline_frame)
    if baseline is None:
        raise SystemExit(f"Could not load baseline frame {args.baseline_frame}")
    capture_unlit_references_from_frame(baseline, overlays)
    proposal = build_assisted_calibration_proposal(
        frame_provider,
        overlays,
        baseline_frame_index=args.baseline_frame,
        end_frame=end_frame,
        settings=ExemplarScanSettings(coarse_stride=args.stride),
    )

    print(f"baseline_frame={proposal.baseline_frame_index}")
    print(f"unlit_status={proposal.unlit_assessment.status}")
    print(f"candidates={proposal.candidate_count}")
    print(f"families={proposal.assignment_result.family_count}")
    for slot, assignment in proposal.assignment_result.assignments.items():
        print(f"{slot}: enabled={assignment.enabled} rgb={assignment.rgb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run focused tests and import smoke**

Run:

```bash
.venv/bin/python -m pytest tests/test_assisted_calibration.py -q
PYTHONPATH=synthesia2midi .venv/bin/python -m synthesia2midi.tools.probe_assisted_calibration --help
```

Expected: tests pass; probe command prints argparse help.

- [ ] **Step 6: Commit**

```bash
git add synthesia2midi/synthesia2midi/detection/assisted_calibration.py synthesia2midi/synthesia2midi/tools/probe_assisted_calibration.py tests/test_assisted_calibration.py
git commit -m "feat: build assisted calibration proposals"
```

---

### Task 7: Assisted Wizard Flow

**Files:**
- Modify: `synthesia2midi/synthesia2midi/gui/calibration_wizard_controller.py`
- Modify: `synthesia2midi/synthesia2midi/gui/wizard.py`
- Modify: `synthesia2midi/synthesia2midi/workflows/calibration.py`
- Test: `tests/test_bugfix_regressions.py`

**Interfaces:**
- Consumes: `capture_unlit_references_from_frame`, `build_assisted_calibration_proposal`, `apply_assisted_calibration_proposal`.
- Adds controller method: `_run_assisted_auto_calibration(self, baseline_frame_rgb: np.ndarray, baseline_frame_index: int) -> bool`.

- [ ] **Step 1: Write failing controller test for accepted proposal**

Append to `tests/test_bugfix_regressions.py`:

```python
def test_keyboard_region_selection_runs_assisted_calibration_and_saves(monkeypatch):
    applied = []
    saved = []

    class _Wizard:
        auto_detect_source_frame_rgb = np.full((8, 8, 3), (245, 245, 235), dtype=np.uint8)

        def handle_keyboard_region_selected(self, *_args):
            app.app_state.overlays = [
                OverlayConfig(key_id=1, note_octave=4, note_name_in_octave="C", x=0, y=0, width=4, height=4, key_type="LW")
            ]
            return True

        def get_auto_detect_tuning_context(self):
            return {"frame_rgb": self.auto_detect_source_frame_rgb, "keyboard_roi": (1, 2, 3, 4)}

    app = SimpleNamespace(
        app_state=AppState(),
        calibration_workflow=SimpleNamespace(apply_template_styles_to_overlays=lambda: None),
        control_panel=SimpleNamespace(
            convert_button=SimpleNamespace(setEnabled=lambda _enabled: None),
            _can_convert=lambda: True,
            update_controls_from_state=lambda: None,
            update_trim_controls_from_state=lambda: None,
            update_selected_overlay_display=lambda: None,
        ),
        keyboard_canvas=SimpleNamespace(setCursor=lambda _cursor: None, display_frame=lambda _frame_idx: None, update=lambda: None),
        show_overlays_action=DummyShowOverlaysAction(),
        video_loading_workflow=SimpleNamespace(save_current_config=lambda: saved.append("save") or True),
        video_session=SimpleNamespace(total_frames=12, get_frame=lambda _index: (True, np.full((8, 8, 3), (235, 245, 255), dtype=np.uint8))),
    )
    app.app_state.video.current_frame_index = 3
    controller = CalibrationWizardController(app, DummyAutoDetectTuningControllerForRestore())
    controller.calibration_wizard = _Wizard()

    monkeypatch.setattr(
        "synthesia2midi.gui.calibration_wizard_controller.apply_assisted_calibration_proposal",
        lambda app_state, proposal: applied.append(proposal),
    )
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Yes)

    controller._handle_keyboard_region_selected(1, 2, 3, 4)

    assert applied
    assert saved == ["save"]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest tests/test_bugfix_regressions.py::test_keyboard_region_selection_runs_assisted_calibration_and_saves -q
```

Expected: fail because the controller has no assisted calibration flow.

- [ ] **Step 3: Add imports and helper methods to controller**

In `calibration_wizard_controller.py`, add imports:

```python
import cv2
import numpy as np
from PySide6.QtWidgets import QMessageBox, QProgressDialog

from synthesia2midi.detection.assisted_calibration import (
    ExemplarScanSettings,
    apply_assisted_calibration_proposal,
    build_assisted_calibration_proposal,
    capture_unlit_references_from_frame,
)
```

Add methods to `CalibrationWizardController`:

```python
    def _frame_provider_rgb(self, frame_index: int):
        if not self.video_session:
            return None
        success, frame_bgr = self.video_session.get_frame(frame_index)
        if not success or frame_bgr is None:
            return None
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def _proposal_summary_text(self, proposal) -> str:
        lines = [
            translate("CalibrationWizardController", "Assisted calibration found {count} candidate samples.").format(
                count=proposal.candidate_count
            ),
            translate("CalibrationWizardController", "Color families found: {count}").format(
                count=proposal.assignment_result.family_count
            ),
        ]
        for slot in ("LW", "LB", "RW", "RB"):
            assignment = proposal.assignment_result.assignments.get(slot)
            if assignment is None:
                continue
            if not assignment.enabled:
                lines.append(f"{slot}: " + translate("CalibrationWizardController", "not present in this video"))
            elif assignment.rgb is not None:
                lines.append(f"{slot}: {assignment.rgb}")
            else:
                lines.append(f"{slot}: " + translate("CalibrationWizardController", "not found"))
        return "\n".join(lines)

    def _run_assisted_auto_calibration(self, baseline_frame_rgb, baseline_frame_index: int) -> bool:
        if baseline_frame_rgb is None or not self.app_state.overlays:
            return False

        capture_unlit_references_from_frame(baseline_frame_rgb, self.app_state.overlays)

        end_frame = max(baseline_frame_index, getattr(self.video_session, "total_frames", baseline_frame_index + 1) - 1)
        progress = QProgressDialog(
            translate("CalibrationWizardController", "Scanning for lit key examples..."),
            translate("CalibrationWizardController", "Cancel"),
            baseline_frame_index,
            end_frame,
            self.app,
        )
        progress.setWindowTitle(translate("CalibrationWizardController", "Assisted Calibration"))
        progress.setMinimumDuration(0)

        def progress_callback(current_frame: int, final_frame: int) -> bool:
            progress.setMaximum(final_frame)
            progress.setValue(current_frame)
            return not progress.wasCanceled()

        proposal = build_assisted_calibration_proposal(
            self._frame_provider_rgb,
            self.app_state.overlays,
            baseline_frame_index=baseline_frame_index,
            end_frame=end_frame,
            settings=ExemplarScanSettings(),
            progress_callback=progress_callback,
        )
        progress.close()
        if proposal.canceled:
            return False

        response = QMessageBox.question(
            self.app,
            translate("CalibrationWizardController", "Assisted Calibration"),
            self._proposal_summary_text(proposal)
            + "\n\n"
            + translate("CalibrationWizardController", "Apply these calibration updates?"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if response != QMessageBox.StandardButton.Yes:
            return False

        apply_assisted_calibration_proposal(self.app_state, proposal)
        if self.video_loading_workflow:
            self.video_loading_workflow.save_current_config()
        return True
```

- [ ] **Step 4: Invoke helper after successful auto-detect**

In `_handle_keyboard_region_selected`, after the display/update block and before `_open_auto_detect_tuning_dialog()`, add:

```python
                baseline_frame_rgb = wizard_context.get("frame_rgb") if wizard_context else None
                baseline_frame_index = self.app_state.video.current_frame_index or 0
                self._run_assisted_auto_calibration(baseline_frame_rgb, baseline_frame_index)
                self.control_panel.update_controls_from_state()
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_bugfix_regressions.py::test_keyboard_region_selection_runs_assisted_calibration_and_saves tests/test_bugfix_regressions.py::test_auto_detect_keyboard_region_marks_overlay_generation_source_auto -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add synthesia2midi/synthesia2midi/gui/calibration_wizard_controller.py tests/test_bugfix_regressions.py
git commit -m "feat: run assisted calibration after keyboard box"
```

---

### Task 8: Unlit Warning In Assisted Flow

**Files:**
- Modify: `synthesia2midi/synthesia2midi/gui/calibration_wizard_controller.py`
- Test: `tests/test_bugfix_regressions.py`

**Interfaces:**
- Consumes: `assess_unlit_frame`.
- The assisted flow returns without writing when the user cancels the soft warning.

- [ ] **Step 1: Write failing cancellation test**

Append to `tests/test_bugfix_regressions.py`:

```python
def test_assisted_calibration_unlit_warning_cancel_skips_apply(monkeypatch):
    applied = []
    app = SimpleNamespace(
        app_state=AppState(),
        video_session=SimpleNamespace(total_frames=4, get_frame=lambda _index: (True, np.zeros((8, 8, 3), dtype=np.uint8))),
        video_loading_workflow=SimpleNamespace(save_current_config=lambda: True),
        control_panel=SimpleNamespace(update_controls_from_state=lambda: None),
    )
    app.app_state.overlays = [
        OverlayConfig(key_id=i, note_octave=4, note_name_in_octave="E", x=i * 2, y=0, width=2, height=2, key_type="LW")
        for i in range(4)
    ]
    baseline = np.full((8, 12, 3), (245, 245, 235), dtype=np.uint8)
    baseline[0:2, 4:6] = (240, 140, 40)
    controller = CalibrationWizardController(app, DummyAutoDetectTuningControllerForRestore())

    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: QMessageBox.StandardButton.Cancel)
    monkeypatch.setattr(
        "synthesia2midi.gui.calibration_wizard_controller.apply_assisted_calibration_proposal",
        lambda app_state, proposal: applied.append(proposal),
    )

    assert controller._run_assisted_auto_calibration(baseline, 0) is False
    assert applied == []
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest tests/test_bugfix_regressions.py::test_assisted_calibration_unlit_warning_cancel_skips_apply -q
```

Expected: fail because warning cancellation is not checked.

- [ ] **Step 3: Add warning before unlit capture**

In `_run_assisted_auto_calibration`, before `capture_unlit_references_from_frame`, add:

```python
        from synthesia2midi.detection.assisted_calibration import assess_unlit_frame

        assessment = assess_unlit_frame(baseline_frame_rgb, self.app_state.overlays)
        if assessment.should_warn:
            note_list = ", ".join(item.note_label for item in assessment.likely_lit)
            response = QMessageBox.warning(
                self.app,
                translate("CalibrationWizardController", "Unlit Frame May Contain Lit Keys"),
                translate(
                    "CalibrationWizardController",
                    "It looks like these keys may be lit: {notes}.\n\nMove to a frame where no keys are lit, or continue if this is expected.",
                ).format(notes=note_list),
                QMessageBox.StandardButton.Ignore | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if response == QMessageBox.StandardButton.Cancel:
                return False
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_bugfix_regressions.py::test_assisted_calibration_unlit_warning_cancel_skips_apply tests/test_bugfix_regressions.py::test_keyboard_region_selection_runs_assisted_calibration_and_saves -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add synthesia2midi/synthesia2midi/gui/calibration_wizard_controller.py tests/test_bugfix_regressions.py
git commit -m "feat: guard assisted calibration baseline frames"
```

---

### Task 9: Localization Updates

**Files:**
- Modify: `docs/localization/ui-string-manifest.json`
- Modify: `docs/localization/translation-agent-packet.json`
- Modify: `synthesia2midi/synthesia2midi/translations/synthesia2midi_*.ts`
- Modify: `synthesia2midi/synthesia2midi/translations/synthesia2midi_*.qm`
- Test: `tests/test_localization.py`
- Test: `tests/test_ui_string_audit.py`

**Interfaces:**
- Consumes: new `QCoreApplication.translate` strings from Tasks 3, 7, and 8.
- Produces: complete translated `.ts` and compiled `.qm` files with no unfinished messages.

- [ ] **Step 1: Run extraction and observe expected failures**

Run:

```bash
.venv/bin/python -m synthesia2midi.tools.audit_ui_strings --output docs/localization/ui-string-manifest.json
.venv/bin/pyside6-lupdate -extensions py synthesia2midi/synthesia2midi -ts /tmp/synthesia2midi_lupdate_probe.ts
.venv/bin/python -m pytest tests/test_localization.py tests/test_ui_string_audit.py -q
```

Expected: audit or localization tests fail until `.ts` files include translated new strings.

- [ ] **Step 2: Update production `.ts` files**

Run:

```bash
for ts_file in synthesia2midi/synthesia2midi/translations/synthesia2midi_*.ts; do
  .venv/bin/pyside6-lupdate -extensions py synthesia2midi/synthesia2midi -ts "$ts_file"
done
```

Edit new messages only. Use these translations:

```text
English source: Unlit Frame May Contain Lit Keys
es: El fotograma sin iluminar puede contener teclas iluminadas
ja: 未点灯フレームに点灯したキーが含まれている可能性があります
ru: В кадре без подсветки могут быть подсвеченные клавиши
zh_CN: 未点亮帧可能包含已点亮的琴键
ko: 꺼진 상태 프레임에 켜진 건반이 포함되어 있을 수 있습니다
pt_BR: O quadro sem iluminação pode conter teclas iluminadas

English source: It looks like these keys may be lit: {notes}.\n\nMove to a frame where no keys are lit, or continue if this is expected.
es: Parece que estas teclas pueden estar iluminadas: {notes}.\n\nVe a un fotograma donde no haya teclas iluminadas, o continúa si esto es esperado.
ja: これらのキーが点灯している可能性があります: {notes}。\n\n点灯しているキーがないフレームに移動するか、これが想定どおりなら続行してください。
ru: Похоже, эти клавиши могут быть подсвечены: {notes}.\n\nПерейдите к кадру без подсвеченных клавиш или продолжите, если это ожидаемо.
zh_CN: 以下琴键可能已点亮：{notes}。\n\n请移到没有琴键点亮的帧，或者如果这是预期情况则继续。
ko: 다음 건반이 켜져 있을 수 있습니다: {notes}.\n\n켜진 건반이 없는 프레임으로 이동하거나, 예상된 상태라면 계속하세요.
pt_BR: Parece que estas teclas podem estar iluminadas: {notes}.\n\nVá para um quadro em que nenhuma tecla esteja iluminada ou continue se isso for esperado.

English source: Scanning for lit key examples...
es: Buscando ejemplos de teclas iluminadas...
ja: 点灯したキーの例を検索しています...
ru: Поиск примеров подсвеченных клавиш...
zh_CN: 正在扫描已点亮琴键示例...
ko: 켜진 건반 예시를 검색하는 중...
pt_BR: Procurando exemplos de teclas iluminadas...

English source: Assisted Calibration
es: Calibración asistida
ja: アシスト付きキャリブレーション
ru: Помощник калибровки
zh_CN: 辅助校准
ko: 보조 보정
pt_BR: Calibração assistida

English source: Assisted calibration found {count} candidate samples.
es: La calibración asistida encontró {count} muestras candidatas.
ja: アシスト付きキャリブレーションで {count} 件の候補サンプルが見つかりました。
ru: Помощник калибровки нашел кандидатов: {count}.
zh_CN: 辅助校准找到了 {count} 个候选样本。
ko: 보조 보정에서 후보 샘플 {count}개를 찾았습니다.
pt_BR: A calibração assistida encontrou {count} amostras candidatas.

English source: Color families found: {count}
es: Familias de color encontradas: {count}
ja: 見つかった色ファミリー: {count}
ru: Найдено цветовых групп: {count}
zh_CN: 找到的颜色组：{count}
ko: 찾은 색상 계열: {count}
pt_BR: Famílias de cores encontradas: {count}

English source: not present in this video
es: no presente en este video
ja: この動画にはありません
ru: отсутствует в этом видео
zh_CN: 此视频中不存在
ko: 이 비디오에 없음
pt_BR: não presente neste vídeo

English source: not found
es: no encontrado
ja: 見つかりません
ru: не найдено
zh_CN: 未找到
ko: 찾을 수 없음
pt_BR: não encontrado

English source: Apply these calibration updates?
es: ¿Aplicar estas actualizaciones de calibración?
ja: これらのキャリブレーション更新を適用しますか？
ru: Применить эти обновления калибровки?
zh_CN: 应用这些校准更新吗？
ko: 이 보정 업데이트를 적용할까요?
pt_BR: Aplicar estas atualizações de calibração?
```

- [ ] **Step 3: Compile `.qm` files**

Run:

```bash
for ts_file in synthesia2midi/synthesia2midi/translations/synthesia2midi_*.ts; do
  qm_file="${ts_file%.ts}.qm"
  .venv/bin/pyside6-lrelease "$ts_file" -qm "$qm_file"
done
```

Expected: each locale compiles successfully.

- [ ] **Step 4: Update translation packet**

Run:

```bash
.venv/bin/python -m synthesia2midi.tools.export_translation_packet --source-ts synthesia2midi/synthesia2midi/translations/synthesia2midi_es.ts --output docs/localization/translation-agent-packet.json
```

- [ ] **Step 5: Run localization tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_localization.py tests/test_ui_string_audit.py -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add docs/localization/ui-string-manifest.json docs/localization/translation-agent-packet.json synthesia2midi/synthesia2midi/translations
git commit -m "chore: localize assisted calibration UI"
```

---

### Task 10: Local Video Probe And Threshold Tuning

**Files:**
- Modify: `synthesia2midi/synthesia2midi/detection/assisted_calibration.py`
- Modify: `docs/superpowers/specs/2026-07-07-assisted-auto-calibration-design.md` only if findings change the accepted design.
- Modify: `backlog/tasks/task-15 - Add-assisted-auto-calibration-after-keyboard-box.md`

**Interfaces:**
- Consumes: local Game of Thrones video and saved overlay/INI files.
- Produces: threshold adjustments backed by probe output; no checked-in media.

- [ ] **Step 1: Run local probe against target video**

Run:

```bash
PYTHONPATH=synthesia2midi .venv/bin/python -m synthesia2midi.tools.probe_assisted_calibration \
  --video /Users/jeff/Movies/game_of_thrones_main_theme_synthesia_piano_tutorial/game_of_thrones_main_theme_synthesia_piano_tutorial_1080p.mp4 \
  --overlays /Users/jeff/Movies/game_of_thrones_main_theme_synthesia_piano_tutorial/game_of_thrones_main_theme_synthesia_piano_tutorial_1080p_overlays.json \
  --baseline-frame 430 \
  --end-frame 2500 \
  --stride 10
```

Expected: output includes RGB values close to the target INI:

```text
LW: enabled=True rgb=(133, 166, 203) approximately
LB: enabled=True rgb=(72, 110, 170) approximately
RW: enabled=True rgb=(243, 176, 68) approximately
RB: enabled=True rgb=(243, 131, 46) approximately
```

- [ ] **Step 2: Tune only general thresholds if needed**

If probe output misses the target classes, adjust only generic defaults in `ExemplarScanSettings` or `assign_exemplar_slots`:

```python
@dataclass(frozen=True)
class ExemplarScanSettings:
    coarse_stride: int = 10
    refine_radius: int = 5
    min_rgb_delta: float = 30.0
    min_saturation: float = 30.0
    max_candidates_per_key: int = 6
```

Do not hard-code video path, frame numbers, note names, or target colors.

- [ ] **Step 3: Add regression coverage for any tuned behavior**

If thresholds changed, add or update a synthetic test in `tests/test_assisted_calibration.py` with the failing color pattern. Example:

```python
def test_assign_exemplar_slots_pairs_dark_black_key_with_nearest_white_family():
    result = assign_exemplar_slots([
        _candidate("W", (133, 166, 203), key_id=1),
        _candidate("B", (72, 110, 170), key_id=2),
        _candidate("W", (243, 176, 68), key_id=3),
        _candidate("B", (243, 131, 46), key_id=4),
    ])

    assert result.assignments["LB"].rgb == (72, 110, 170)
    assert result.assignments["RB"].rgb == (243, 131, 46)
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_assisted_calibration.py -q
```

Expected: pass.

- [ ] **Step 5: Update Backlog notes**

In `backlog/tasks/task-15 - Add-assisted-auto-calibration-after-keyboard-box.md`, add a verification note with the probe command and observed approximate RGB output. Do not check acceptance criteria unless the implementation is complete.

- [ ] **Step 6: Commit**

```bash
git add synthesia2midi/synthesia2midi/detection/assisted_calibration.py tests/test_assisted_calibration.py 'backlog/tasks/task-15 - Add-assisted-auto-calibration-after-keyboard-box.md'
git commit -m "test: tune assisted calibration on reference video"
```

---

### Task 11: Full Verification And Backlog Closeout

**Files:**
- Modify: `backlog/tasks/task-15 - Add-assisted-auto-calibration-after-keyboard-box.md`
- Modify: docs only if verification changes a documented command.

**Interfaces:**
- Produces: completed task record and verified branch.

- [ ] **Step 1: Run focused calibration tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_assisted_calibration.py tests/test_bugfix_regressions.py tests/test_manual_overlay_generation.py -q
```

Expected: pass.

- [ ] **Step 2: Run localization gate**

Run:

```bash
.venv/bin/python -m synthesia2midi.tools.audit_ui_strings --output docs/localization/ui-string-manifest.json
.venv/bin/python -m pytest tests/test_localization.py tests/test_ui_string_audit.py -q
```

Expected: pass with no manifest diff after the command.

- [ ] **Step 3: Run default gate**

Run:

```bash
git diff --check
.venv/bin/python -m compileall -q synthesia2midi
.venv/bin/python -m pytest
```

Expected: all pass.

- [ ] **Step 4: Update Backlog acceptance criteria**

Check completed items in `backlog/tasks/task-15 - Add-assisted-auto-calibration-after-keyboard-box.md`. The final criteria should be checked only after the default gate passes:

```markdown
- [x] #1 The user still draws the keyboard bounding box manually before assisted calibration begins.
- [x] #2 Successful auto-detection can immediately capture unlit reference colors and histograms from the selected baseline frame.
- [x] #3 A reusable soft warning detects likely lit overlays during unlit calibration and names likely lit notes when confidence is high.
- [x] #4 The existing manual "Calibrate Unlit All Keys" path uses the same soft warning before overwriting unlit data.
- [x] #5 The assisted scan searches overlay ROIs across video frames for lit exemplar candidates without relying on physical left/right keyboard position.
- [x] #6 Candidate lit colors are clustered into color families and mapped into legacy `LW`, `LB`, `RW`, and `RB` slots by family and key color.
- [x] #7 One-color or partial-color videos can leave absent exemplar slots disabled or unchanged only after user confirmation.
- [x] #8 The user sees a progress/cancel path while scanning and a confirmation summary before exemplar changes are saved.
- [x] #9 Tests cover the unlit-frame guard, exemplar candidate detection, color-family assignment, partial results, cancellation, and proposal application.
- [x] #10 Local exploratory validation compares the Game of Thrones video proposal against the saved target INI and overlays, excluding octave transpose.
```

Set frontmatter `status: Done` only if the implementation and verification are complete.

- [ ] **Step 5: Commit closeout**

```bash
git add 'backlog/tasks/task-15 - Add-assisted-auto-calibration-after-keyboard-box.md'
git commit -m "docs: close assisted calibration task"
```

- [ ] **Step 6: Final branch status**

Run:

```bash
git status --short --branch
git log --oneline -8
```

Expected: clean working tree on `codex/auto-calibration-exploration`; no pushes performed.
