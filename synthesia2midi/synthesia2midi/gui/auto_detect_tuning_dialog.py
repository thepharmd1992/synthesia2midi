"""
Modal tuning dialog for monolithic auto-detect parameters.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from synthesia2midi.core.app_state import AppState
from synthesia2midi.detection.auto_detect_adapter import AutoDetectAdapter
from synthesia2midi.detection.auto_detect_param_specs import (
    AUTO_DETECT_PARAM_CATEGORIES,
    AUTO_DETECT_PARAM_SPECS,
    coerce_auto_detect_param_value,
    coerce_auto_detect_params,
    get_active_auto_detect_defaults,
    get_category_param_keys,
    humanize_auto_detect_param_name,
)
from synthesia2midi.gui.controls_qt import CollapsibleSection


class AutoDetectTuningDialog(QDialog):
    """Modal dialog for live auto-detect parameter tuning."""

    def __init__(
        self,
        parent,
        app_state: AppState,
        source_frame_rgb: np.ndarray,
        keyboard_roi: Tuple[int, int, int, int],
        *,
        initial_detection_results: Optional[Dict[str, Any]],
        fallback_used: bool,
        apply_detection_callback: Callable[[Dict[str, Any]], bool],
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Auto-Detect Tuning")
        self.setModal(True)
        self.resize(980, 760)

        self.logger = logging.getLogger(f"{__name__}.AutoDetectTuningDialog")
        self.app_state = app_state
        self._source_frame_rgb = np.copy(source_frame_rgb)
        self._keyboard_roi = keyboard_roi
        self._fallback_used = bool(fallback_used)
        self._apply_detection_callback = apply_detection_callback
        self._initial_detection_results = initial_detection_results or {}
        self._defaults = get_active_auto_detect_defaults()
        self._current_params = coerce_auto_detect_params(self.app_state.calibration.auto_detect_params)
        self._control_widgets: Dict[str, Dict[str, Any]] = {}
        self._suppress_events = False

        self._adapter = AutoDetectAdapter()
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(120)
        self._debounce_timer.timeout.connect(self._run_preview_detection)

        self._status_labels: Dict[str, QLabel] = {}
        self._warning_label: Optional[QLabel] = None
        self._setup_ui()
        self._update_status_panel(self._initial_detection_results)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        banner = QLabel()
        banner.setWordWrap(True)
        if self._fallback_used:
            banner.setText(
                "Saved auto-detect params failed on initial run. Built-in fallback profile was used. "
                "Tune and save to update this video's parameters."
            )
            banner.setStyleSheet("color: #8a6d00; background: #fff8db; border: 1px solid #e6d390; padding: 6px;")
        else:
            banner.setText("Initial auto-detect used saved parameters.")
            banner.setStyleSheet("color: #2f5d2f; background: #eaf7ea; border: 1px solid #bcdcbc; padding: 6px;")
        layout.addWidget(banner)

        controls_row = QHBoxLayout()
        reset_all_btn = QPushButton("Reset All to Active Defaults")
        reset_all_btn.clicked.connect(self._reset_all_to_defaults)
        controls_row.addWidget(reset_all_btn)
        controls_row.addStretch()
        layout.addLayout(controls_row)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sections_container = QWidget()
        sections_layout = QVBoxLayout(sections_container)
        sections_layout.setContentsMargins(0, 0, 0, 0)
        sections_layout.setSpacing(8)

        for idx, category in enumerate(AUTO_DETECT_PARAM_CATEGORIES):
            section = CollapsibleSection(category, expanded=(idx == 0))
            content_layout = section.content_layout()
            grid = QGridLayout()
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setHorizontalSpacing(10)
            grid.setVerticalSpacing(6)

            row = 0
            for param_key in get_category_param_keys(category):
                row = self._add_parameter_control(grid, row, param_key)

            content_layout.addLayout(grid)
            reset_section_btn = QPushButton("Reset Section")
            reset_section_btn.clicked.connect(
                lambda _checked=False, cat=category: self._reset_category_to_defaults(cat)
            )
            content_layout.addWidget(reset_section_btn)
            sections_layout.addWidget(section)

        sections_layout.addStretch(1)
        scroll_area.setWidget(sections_container)
        layout.addWidget(scroll_area, 1)

        status_group = QGroupBox("Preview Status")
        status_layout = QGridLayout(status_group)
        status_layout.setHorizontalSpacing(14)
        status_layout.setVerticalSpacing(4)

        self._status_labels["white"] = QLabel("-")
        self._status_labels["black"] = QLabel("-")
        self._status_labels["total"] = QLabel("-")
        self._status_labels["overlays"] = QLabel("-")
        self._status_labels["leftmost"] = QLabel("-")
        self._status_labels["fallback"] = QLabel("Yes" if self._fallback_used else "No")

        status_layout.addWidget(QLabel("Detected White Keys:"), 0, 0)
        status_layout.addWidget(self._status_labels["white"], 0, 1)
        status_layout.addWidget(QLabel("Detected Black Keys:"), 0, 2)
        status_layout.addWidget(self._status_labels["black"], 0, 3)
        status_layout.addWidget(QLabel("Detected Total Keys:"), 1, 0)
        status_layout.addWidget(self._status_labels["total"], 1, 1)
        status_layout.addWidget(QLabel("Overlays Created:"), 1, 2)
        status_layout.addWidget(self._status_labels["overlays"], 1, 3)
        status_layout.addWidget(QLabel("Leftmost Note/Octave:"), 2, 0)
        status_layout.addWidget(self._status_labels["leftmost"], 2, 1)
        status_layout.addWidget(QLabel("Saved Fallback Used:"), 2, 2)
        status_layout.addWidget(self._status_labels["fallback"], 2, 3)
        layout.addWidget(status_group)

        self._warning_label = QLabel("")
        self._warning_label.setWordWrap(True)
        self._warning_label.setStyleSheet("color: #b05400;")
        layout.addWidget(self._warning_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Close | QDialogButtonBox.Cancel)
        close_btn = buttons.button(QDialogButtonBox.Close)
        cancel_btn = buttons.button(QDialogButtonBox.Cancel)
        if close_btn:
            close_btn.clicked.connect(self.accept)
        if cancel_btn:
            cancel_btn.clicked.connect(self.reject)
        layout.addWidget(buttons)

    def _add_parameter_control(self, grid: QGridLayout, row: int, key: str) -> int:
        spec = AUTO_DETECT_PARAM_SPECS[key]
        key_type = spec["type"]
        current_value = self._current_params[key]

        label = QLabel(humanize_auto_detect_param_name(key))
        grid.addWidget(label, row, 0)

        if key_type == "bool":
            checkbox = QCheckBox()
            checkbox.setChecked(bool(current_value))
            checkbox.toggled.connect(lambda checked, k=key: self._on_param_changed(k, checked))
            grid.addWidget(checkbox, row, 1, 1, 2)
            self._control_widgets[key] = {"type": key_type, "spec": spec, "checkbox": checkbox}
            return row + 1

        if key_type == "enum":
            combo = QComboBox()
            for option in spec["options"]:
                combo.addItem(option)
            combo.setCurrentText(str(current_value))
            combo.currentTextChanged.connect(lambda text, k=key: self._on_param_changed(k, text))
            grid.addWidget(combo, row, 1, 1, 2)
            self._control_widgets[key] = {"type": key_type, "spec": spec, "combo": combo}
            return row + 1

        slider = QSlider(Qt.Horizontal)
        slider.setTracking(True)

        if key_type == "int":
            spin = QSpinBox()
            slider.setMinimum(int(spec["min"]))
            slider.setMaximum(int(spec["max"]))
            slider.setSingleStep(int(spec.get("step", 1)))
            spin.setRange(int(spec["min"]), int(spec["max"]))
            spin.setSingleStep(int(spec.get("step", 1)))
            spin.setValue(int(current_value))
            slider.setValue(int(current_value))
            slider.valueChanged.connect(lambda val, k=key: self._on_param_changed(k, val))
            spin.valueChanged.connect(lambda val, k=key: self._on_param_changed(k, val))
            self._control_widgets[key] = {
                "type": key_type,
                "spec": spec,
                "slider": slider,
                "spin": spin,
                "factor": 1,
            }
        else:
            spin = QDoubleSpinBox()
            step = float(spec.get("step", 0.01))
            decimals = self._step_decimals(step)
            factor = int(round(1.0 / step))
            slider.setMinimum(int(round(float(spec["min"]) * factor)))
            slider.setMaximum(int(round(float(spec["max"]) * factor)))
            slider.setSingleStep(1)
            spin.setDecimals(decimals)
            spin.setRange(float(spec["min"]), float(spec["max"]))
            spin.setSingleStep(step)
            spin.setValue(float(current_value))
            slider.setValue(int(round(float(current_value) * factor)))
            slider.valueChanged.connect(
                lambda raw_val, k=key, f=factor: self._on_param_changed(k, raw_val / f)
            )
            spin.valueChanged.connect(lambda val, k=key: self._on_param_changed(k, val))
            self._control_widgets[key] = {
                "type": key_type,
                "spec": spec,
                "slider": slider,
                "spin": spin,
                "factor": factor,
            }

        grid.addWidget(slider, row, 1)
        grid.addWidget(spin, row, 2)
        return row + 1

    def _step_decimals(self, step: float) -> int:
        if step <= 0:
            return 2
        normalized = f"{step:.6f}".rstrip("0").rstrip(".")
        if "." not in normalized:
            return 0
        return len(normalized.split(".")[1])

    def _set_widget_value(self, key: str, value: Any) -> None:
        info = self._control_widgets[key]
        key_type = info["type"]
        self._suppress_events = True
        try:
            if key_type == "bool":
                info["checkbox"].setChecked(bool(value))
                return
            if key_type == "enum":
                info["combo"].setCurrentText(str(value))
                return
            if key_type == "int":
                info["slider"].setValue(int(value))
                info["spin"].setValue(int(value))
                return
            factor = int(info["factor"])
            info["slider"].setValue(int(round(float(value) * factor)))
            info["spin"].setValue(float(value))
        finally:
            self._suppress_events = False

    def _on_param_changed(self, key: str, raw_value: Any) -> None:
        if self._suppress_events:
            return

        current_default = self._defaults.get(key)
        coerced = coerce_auto_detect_param_value(key, raw_value, default_value=current_default)
        self._current_params[key] = coerced
        self._set_widget_value(key, coerced)
        self._persist_params_to_state()
        self._debounce_timer.start()

    def _persist_params_to_state(self) -> None:
        normalized = coerce_auto_detect_params(self._current_params)
        self._current_params = normalized
        self.app_state.calibration.auto_detect_params = dict(normalized)
        self.app_state.unsaved_changes = True

    def _reset_category_to_defaults(self, category: str) -> None:
        for key in get_category_param_keys(category):
            self._current_params[key] = self._defaults[key]
            self._set_widget_value(key, self._defaults[key])
        self._persist_params_to_state()
        self._debounce_timer.start()

    def _reset_all_to_defaults(self) -> None:
        self._current_params = dict(self._defaults)
        for key, value in self._current_params.items():
            self._set_widget_value(key, value)
        self._persist_params_to_state()
        self._debounce_timer.start()

    def _set_warning(self, message: str) -> None:
        if self._warning_label is None:
            return
        self._warning_label.setText(message)

    def _run_preview_detection(self) -> None:
        x, y, width, height = self._keyboard_roi
        cropped = self._source_frame_rgb[y:y + height, x:x + width]
        if cropped.size == 0:
            self._set_warning("Preview detection failed: empty ROI crop.")
            return

        detection_results = self._adapter.detect_from_frame(
            cropped,
            keyboard_region=self._keyboard_roi,
            tuning_params=self._current_params,
            use_profile_fallback=False,
        )
        if detection_results is None:
            self._set_warning("Preview detection failed. Keeping last successful overlays.")
            return

        applied = self._apply_detection_callback(detection_results)
        if not applied:
            self._set_warning("Preview detection was computed but overlays could not be applied.")
            return

        self._set_warning("")
        self._update_status_panel(detection_results)

    def _update_status_panel(self, detection_results: Dict[str, Any]) -> None:
        if not detection_results:
            return

        detected_keys = detection_results.get("detected_keys") or []
        white_count = detection_results.get("detected_white_count")
        black_count = detection_results.get("detected_black_count")
        if white_count is None:
            white_count = sum(1 for item in detected_keys if getattr(item, "key_type", "") == "white")
        if black_count is None:
            black_count = sum(1 for item in detected_keys if getattr(item, "key_type", "") == "black")

        total_count = detection_results.get("total_keys", len(detected_keys))
        overlays_created = len(detected_keys) if detected_keys else total_count
        leftmost_note = detection_results.get("leftmost_note", "?")
        leftmost_octave = detection_results.get("leftmost_octave", "?")

        self._status_labels["white"].setText(str(white_count))
        self._status_labels["black"].setText(str(black_count))
        self._status_labels["total"].setText(str(total_count))
        self._status_labels["overlays"].setText(str(overlays_created))
        self._status_labels["leftmost"].setText(f"{leftmost_note}{leftmost_octave}")
