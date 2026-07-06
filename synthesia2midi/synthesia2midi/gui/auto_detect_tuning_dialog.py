"""
Modal tuning dialog for monolithic auto-detect parameters.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QCoreApplication, Qt
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
    QStyle,
    QTabWidget,
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
    get_advanced_auto_detect_param_keys,
    get_active_auto_detect_defaults,
    get_basic_auto_detect_param_keys,
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
        self.setWindowTitle(QCoreApplication.translate("AutoDetectTuningDialog", "Auto-Detect Tuning"))
        self.setModal(True)
        self.resize(760, 420)

        self.logger = logging.getLogger(f"{__name__}.AutoDetectTuningDialog")
        self.app_state = app_state
        self._source_frame_rgb = np.copy(source_frame_rgb)
        self._keyboard_roi = keyboard_roi
        self._fallback_used = bool(fallback_used)
        self._apply_detection_callback = apply_detection_callback
        self._initial_detection_results = initial_detection_results or {}
        self._defaults = get_active_auto_detect_defaults()
        # Start each tuning session from active defaults for a clean calibration run.
        self._current_params = dict(self._defaults)
        self._control_widgets: Dict[str, Dict[str, Any]] = {}
        self._suppress_events = False

        self._adapter = AutoDetectAdapter()

        self._status_labels: Dict[str, QLabel] = {}
        self._warning_label: Optional[QLabel] = None
        self._setup_ui()
        self._update_status_panel(self._initial_detection_results)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        if self._fallback_used:
            banner = QLabel()
            banner.setWordWrap(True)
            banner.setText(
                QCoreApplication.translate(
                    "AutoDetectTuningDialog",
                    "Initial auto-detect needed a fallback profile (not the default profile). Tune parameters if you want to refine this result.",
                )
            )
            banner.setStyleSheet("color: #8a6d00; background: #fff8db; border: 1px solid #e6d390; padding: 6px;")
            layout.addWidget(banner)

        controls_row = QHBoxLayout()
        reset_all_btn = QPushButton(QCoreApplication.translate("AutoDetectTuningDialog", "Reset All to Active Defaults"))
        reset_all_btn.clicked.connect(self._reset_all_to_defaults)
        controls_row.addWidget(reset_all_btn)
        controls_row.addStretch()
        layout.addLayout(controls_row)

        tabs = QTabWidget()
        tabs.tabBar().setExpanding(False)
        tabs.setStyleSheet("QTabWidget::tab-bar { alignment: right; }")
        tabs.addTab(
            self._build_param_tab(get_basic_auto_detect_param_keys()),
            QCoreApplication.translate("AutoDetectTuningDialog", "Basic"),
        )
        tabs.addTab(
            self._build_param_tab(get_advanced_auto_detect_param_keys()),
            QCoreApplication.translate("AutoDetectTuningDialog", "Advanced"),
        )
        layout.addWidget(tabs, 1)

        status_group = QGroupBox(QCoreApplication.translate("AutoDetectTuningDialog", "Preview Status"))
        status_layout = QGridLayout(status_group)
        status_layout.setHorizontalSpacing(14)
        status_layout.setVerticalSpacing(4)

        self._status_labels["white"] = QLabel("-")
        self._status_labels["black"] = QLabel("-")
        self._status_labels["total"] = QLabel("-")
        self._status_labels["overlays"] = QLabel("-")
        self._status_labels["leftmost"] = QLabel("-")
        self._status_labels["fallback"] = QLabel(
            QCoreApplication.translate("AutoDetectTuningDialog", "Yes")
            if self._fallback_used
            else QCoreApplication.translate("AutoDetectTuningDialog", "No")
        )

        status_layout.addWidget(QLabel(QCoreApplication.translate("AutoDetectTuningDialog", "Detected White Keys:")), 0, 0)
        status_layout.addWidget(self._status_labels["white"], 0, 1)
        status_layout.addWidget(QLabel(QCoreApplication.translate("AutoDetectTuningDialog", "Detected Black Keys:")), 0, 2)
        status_layout.addWidget(self._status_labels["black"], 0, 3)
        status_layout.addWidget(QLabel(QCoreApplication.translate("AutoDetectTuningDialog", "Detected Total Keys:")), 1, 0)
        status_layout.addWidget(self._status_labels["total"], 1, 1)
        status_layout.addWidget(QLabel(QCoreApplication.translate("AutoDetectTuningDialog", "Overlays Created:")), 1, 2)
        status_layout.addWidget(self._status_labels["overlays"], 1, 3)
        status_layout.addWidget(QLabel(QCoreApplication.translate("AutoDetectTuningDialog", "Leftmost Note/Octave:")), 2, 0)
        status_layout.addWidget(self._status_labels["leftmost"], 2, 1)
        status_layout.addWidget(QLabel(QCoreApplication.translate("AutoDetectTuningDialog", "Fallback Profile Used:")), 2, 2)
        status_layout.addWidget(self._status_labels["fallback"], 2, 3)
        layout.addWidget(status_group)

        self._warning_label = QLabel("")
        self._warning_label.setWordWrap(True)
        self._warning_label.setStyleSheet("color: #b05400;")
        layout.addWidget(self._warning_label)

        buttons = QDialogButtonBox()
        save_btn = buttons.addButton(
            QCoreApplication.translate("AutoDetectTuningDialog", "Save"),
            QDialogButtonBox.AcceptRole,
        )
        save_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogApplyButton))
        save_btn.setAutoDefault(True)
        save_btn.setDefault(True)
        save_btn.setStyleSheet(
            "QPushButton {"
            "background-color: #2e7d32;"
            "color: #ffffff;"
            "border: 1px solid #1b5e20;"
            "padding: 6px 14px;"
            "font-weight: 600;"
            "}"
            "QPushButton:hover { background-color: #388e3c; }"
            "QPushButton:pressed { background-color: #2c6e30; }"
        )
        save_btn.clicked.connect(self.accept)
        cancel_btn = buttons.addButton(QDialogButtonBox.Cancel)
        if cancel_btn:
            cancel_btn.clicked.connect(self.reject)
        layout.addWidget(buttons)

    def _build_param_tab(self, param_keys: List[str]) -> QWidget:
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.setSpacing(0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        sections_container = QWidget()
        sections_layout = QVBoxLayout(sections_container)
        sections_layout.setContentsMargins(0, 0, 0, 0)
        sections_layout.setSpacing(8)

        first_section = True
        param_key_set = set(param_keys)
        for category in AUTO_DETECT_PARAM_CATEGORIES:
            category_keys = [
                key
                for key in get_category_param_keys(category)
                if key in param_key_set
            ]
            if not category_keys:
                continue

            section = CollapsibleSection(self._translate_category(category), expanded=first_section)
            first_section = False
            content_layout = section.content_layout()

            if category == "Edge Drift Correction":
                edge_row = QHBoxLayout()
                edge_row.setContentsMargins(0, 0, 0, 0)
                edge_row.setSpacing(14)

                left_column = QWidget()
                left_column_layout = QVBoxLayout(left_column)
                left_column_layout.setContentsMargins(0, 0, 0, 0)
                self._add_directional_edge_control(
                    left_column_layout,
                    key="white_edge_left_shift_ticks",
                    title=QCoreApplication.translate("AutoDetectTuningDialog", "Left Edge Outward"),
                    hint=QCoreApplication.translate("AutoDetectTuningDialog", "outward <-"),
                    title_alignment=Qt.AlignRight,
                    hint_alignment=Qt.AlignRight,
                    slider_inverted=True,
                )
                edge_row.addWidget(left_column, 1)

                right_column = QWidget()
                right_column_layout = QVBoxLayout(right_column)
                right_column_layout.setContentsMargins(0, 0, 0, 0)
                self._add_directional_edge_control(
                    right_column_layout,
                    key="white_edge_right_shift_ticks",
                    title=QCoreApplication.translate("AutoDetectTuningDialog", "Right Edge Outward"),
                    hint=QCoreApplication.translate("AutoDetectTuningDialog", "-> outward"),
                    title_alignment=Qt.AlignLeft,
                    hint_alignment=Qt.AlignLeft,
                    slider_inverted=False,
                )
                edge_row.addWidget(right_column, 1)

                content_layout.addLayout(edge_row)
            else:
                grid = QGridLayout()
                grid.setContentsMargins(0, 0, 0, 0)
                grid.setHorizontalSpacing(10)
                grid.setVerticalSpacing(6)

                row = 0
                for param_key in category_keys:
                    row = self._add_parameter_control(grid, row, param_key)

                content_layout.addLayout(grid)

            reset_section_btn = QPushButton(QCoreApplication.translate("AutoDetectTuningDialog", "Reset Section"))
            reset_section_btn.clicked.connect(
                lambda _checked=False, keys=tuple(category_keys): self._reset_keys_to_defaults(keys)
            )
            content_layout.addWidget(reset_section_btn)
            sections_layout.addWidget(section)

        if first_section:
            empty_label = QLabel(QCoreApplication.translate("AutoDetectTuningDialog", "No parameters available."))
            empty_label.setStyleSheet("color: #666;")
            sections_layout.addWidget(empty_label)

        sections_layout.addStretch(1)
        scroll_area.setWidget(sections_container)
        tab_layout.addWidget(scroll_area)
        return tab

    def _add_directional_edge_control(
        self,
        layout: QVBoxLayout,
        *,
        key: str,
        title: str,
        hint: str,
        title_alignment: Qt.AlignmentFlag,
        hint_alignment: Qt.AlignmentFlag,
        slider_inverted: bool,
    ) -> None:
        spec = AUTO_DETECT_PARAM_SPECS[key]
        current_value = int(self._current_params[key])

        title_label = QLabel(title)
        title_label.setAlignment(title_alignment)
        layout.addWidget(title_label)

        hint_label = QLabel(hint)
        hint_label.setAlignment(hint_alignment)
        hint_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(hint_label)

        slider = QSlider(Qt.Horizontal)
        slider.setTracking(True)
        slider.setInvertedAppearance(slider_inverted)
        slider.setInvertedControls(slider_inverted)
        slider.setMinimum(int(spec["min"]))
        slider.setMaximum(int(spec["max"]))
        slider.setSingleStep(int(spec.get("step", 1)))
        slider.setValue(current_value)
        slider.valueChanged.connect(lambda val, k=key: self._on_param_changed(k, val))
        layout.addWidget(slider)

        self._control_widgets[key] = {
            "type": "int",
            "spec": spec,
            "slider": slider,
            "spin": None,
            "factor": 1,
        }

    def _add_parameter_control(
        self,
        grid: QGridLayout,
        row: int,
        key: str,
        *,
        label_override: Optional[str] = None,
        slider_inverted: bool = False,
    ) -> int:
        spec = AUTO_DETECT_PARAM_SPECS[key]
        key_type = spec["type"]
        current_value = self._current_params[key]

        label = QLabel(label_override or self._translate_param_label(key))
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
        slider.setInvertedAppearance(slider_inverted)
        slider.setInvertedControls(slider_inverted)

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
                slider = info.get("slider")
                spin = info.get("spin")
                if slider is not None:
                    slider.setValue(int(value))
                if spin is not None:
                    spin.setValue(int(value))
                return
            factor = int(info["factor"])
            slider = info.get("slider")
            spin = info.get("spin")
            if slider is not None:
                slider.setValue(int(round(float(value) * factor)))
            if spin is not None:
                spin.setValue(float(value))
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
        self._run_preview_detection()

    def _persist_params_to_state(self) -> None:
        normalized = coerce_auto_detect_params(self._current_params)
        self._current_params = normalized
        self.app_state.calibration.auto_detect_params = dict(normalized)
        self.app_state.unsaved_changes = True

    def _reset_keys_to_defaults(self, keys: Tuple[str, ...]) -> None:
        for key in keys:
            self._current_params[key] = self._defaults[key]
            self._set_widget_value(key, self._defaults[key])
        self._persist_params_to_state()
        self._run_preview_detection()

    def _reset_all_to_defaults(self) -> None:
        self._current_params = dict(self._defaults)
        for key, value in self._current_params.items():
            self._set_widget_value(key, value)
        self._persist_params_to_state()
        self._run_preview_detection()

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

    def _translate_category(self, category: str) -> str:
        translated = {
            "Edge Drift Correction": QCoreApplication.translate("AutoDetectTuningDialog", "Edge Drift Correction"),
            "Black Key Detection": QCoreApplication.translate("AutoDetectTuningDialog", "Black Key Detection"),
            "White Strip Selection": QCoreApplication.translate("AutoDetectTuningDialog", "White Strip Selection"),
            "White Separator Extraction": QCoreApplication.translate(
                "AutoDetectTuningDialog", "White Separator Extraction"
            ),
            "Assignment and Recovery": QCoreApplication.translate("AutoDetectTuningDialog", "Assignment and Recovery"),
            "Geometry and Padding": QCoreApplication.translate("AutoDetectTuningDialog", "Geometry and Padding"),
        }
        return translated.get(category, category)

    def _translate_param_label(self, key: str) -> str:
        translated = {
            "black_upper_ratio": QCoreApplication.translate("AutoDetectTuningDialog", "Black Upper Ratio"),
            "black_bottom_ratio": QCoreApplication.translate("AutoDetectTuningDialog", "Black Bottom Ratio"),
            "black_threshold_method": QCoreApplication.translate("AutoDetectTuningDialog", "Black Threshold Method"),
            "black_threshold": QCoreApplication.translate("AutoDetectTuningDialog", "Black Threshold"),
            "black_adaptive_block_size": QCoreApplication.translate(
                "AutoDetectTuningDialog", "Black Adaptive Block Size"
            ),
            "black_adaptive_c": QCoreApplication.translate("AutoDetectTuningDialog", "Black Adaptive C"),
            "black_column_ratio": QCoreApplication.translate("AutoDetectTuningDialog", "Black Column Ratio"),
            "black_min_width": QCoreApplication.translate("AutoDetectTuningDialog", "Black Min Width"),
            "black_max_width": QCoreApplication.translate("AutoDetectTuningDialog", "Black Max Width"),
            "white_bottom_ratio": QCoreApplication.translate("AutoDetectTuningDialog", "White Bottom Ratio"),
            "white_initial_top_ratio": QCoreApplication.translate("AutoDetectTuningDialog", "White Initial Top Ratio"),
            "white_strip_dark_threshold": QCoreApplication.translate(
                "AutoDetectTuningDialog", "White Strip Dark Threshold"
            ),
            "white_strip_dark_fraction": QCoreApplication.translate(
                "AutoDetectTuningDialog", "White Strip Dark Fraction"
            ),
            "white_strip_min_run": QCoreApplication.translate("AutoDetectTuningDialog", "White Strip Min Run"),
            "white_strip_allow_failures": QCoreApplication.translate(
                "AutoDetectTuningDialog", "White Strip Allow Failures"
            ),
            "white_sep_ratio": QCoreApplication.translate("AutoDetectTuningDialog", "White Sep Ratio"),
            "white_sep_dyn_min": QCoreApplication.translate("AutoDetectTuningDialog", "White Sep Dyn Min"),
            "white_sep_close_kernel": QCoreApplication.translate(
                "AutoDetectTuningDialog", "White Sep Close Kernel"
            ),
            "white_sep_open_kernel": QCoreApplication.translate("AutoDetectTuningDialog", "White Sep Open Kernel"),
            "white_sep_min_width": QCoreApplication.translate("AutoDetectTuningDialog", "White Sep Min Width"),
            "type_aware_assignment": QCoreApplication.translate("AutoDetectTuningDialog", "Type Aware Assignment"),
            "black_recovery_enabled": QCoreApplication.translate("AutoDetectTuningDialog", "Black Recovery Enabled"),
            "black_recovery_ratio": QCoreApplication.translate("AutoDetectTuningDialog", "Black Recovery Ratio"),
            "black_recovery_column_ratio_scale": QCoreApplication.translate(
                "AutoDetectTuningDialog", "Black Recovery Column Ratio Scale"
            ),
            "black_split_max_factor": QCoreApplication.translate("AutoDetectTuningDialog", "Black Split Max Factor"),
            "padding_percent": QCoreApplication.translate("AutoDetectTuningDialog", "Padding Percent"),
            "white_edge_left_shift_ticks": QCoreApplication.translate(
                "AutoDetectTuningDialog", "White Edge Left Shift Ticks"
            ),
            "white_edge_right_shift_ticks": QCoreApplication.translate(
                "AutoDetectTuningDialog", "White Edge Right Shift Ticks"
            ),
        }
        return translated.get(key, humanize_auto_detect_param_name(key))
