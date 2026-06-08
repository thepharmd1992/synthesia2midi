"""Manual keyboard fit tool window."""
from __future__ import annotations

from typing import Dict

from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
)

from synthesia2midi.workflows.manual_keyboard_fit import ManualFitParams


PARAM_SPECS = [
    ("keyboard_width_delta", "Keyboard Width", -1000, 1000),
    ("left_edge_drift", "Left Edge Drift", -500, 500),
    ("right_edge_drift", "Right Edge Drift", -500, 500),
    ("white_y_delta", "White Y", -500, 500),
    ("white_height_delta", "White Height", -500, 500),
    ("white_width_delta", "White Width", -500, 500),
    ("black_y_delta", "Black Y", -500, 500),
    ("black_height_delta", "Black Height", -500, 500),
    ("black_width_delta", "Black Width", -500, 500),
    ("black_x_delta", "Black X Offset", -500, 500),
]


class ManualKeyboardFitDialog(QDialog):
    """Modeless controls for manual overlay keyboard fitting."""

    params_changed = Signal(object)
    mode_changed = Signal(str)
    reset_all_requested = Signal()
    clear_selected_override_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Tool | Qt.WindowCloseButtonHint)
        self.setWindowTitle("Manual Fit")
        self.setModal(False)
        self.setWindowModality(Qt.NonModal)
        self.resize(680, 520)

        self.param_sliders: Dict[str, QSlider] = {}
        self.param_spinboxes: Dict[str, QSpinBox] = {}

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        mode_group = QGroupBox("Edit Mode")
        mode_layout = QHBoxLayout(mode_group)
        self.mode_status_label = QLabel("Editing: Whole Keyboard")
        self.group_fit_radio = QRadioButton("Group Fit")
        self.single_overlay_radio = QRadioButton("Single Overlay")
        self.group_fit_radio.setChecked(True)
        self._mode_button_group = QButtonGroup(self)
        self._mode_button_group.addButton(self.group_fit_radio)
        self._mode_button_group.addButton(self.single_overlay_radio)
        self.group_fit_radio.toggled.connect(self._handle_mode_toggled)
        self.single_overlay_radio.toggled.connect(self._handle_mode_toggled)
        mode_layout.addWidget(self.mode_status_label)
        mode_layout.addStretch()
        mode_layout.addWidget(self.group_fit_radio)
        mode_layout.addWidget(self.single_overlay_radio)
        layout.addWidget(mode_group)

        controls_group = QGroupBox("Keyboard Fit")
        controls_layout = QGridLayout(controls_group)
        controls_layout.setHorizontalSpacing(10)
        controls_layout.setVerticalSpacing(6)

        for row, (name, label, minimum, maximum) in enumerate(PARAM_SPECS):
            label_widget = QLabel(label)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(minimum, maximum)
            slider.setValue(0)
            spinbox = QSpinBox()
            spinbox.setRange(minimum, maximum)
            spinbox.setValue(0)
            spinbox.setFixedWidth(78)

            slider.valueChanged.connect(
                lambda value, param_name=name: self._handle_slider_changed(param_name, value)
            )
            spinbox.valueChanged.connect(
                lambda value, param_name=name: self._handle_spinbox_changed(param_name, value)
            )

            self.param_sliders[name] = slider
            self.param_spinboxes[name] = spinbox
            controls_layout.addWidget(label_widget, row, 0)
            controls_layout.addWidget(slider, row, 1)
            controls_layout.addWidget(spinbox, row, 2)

        layout.addWidget(controls_group, 1)

        action_row = QHBoxLayout()
        self.reset_all_button = QPushButton("Reset All")
        self.clear_selected_override_button = QPushButton("Clear Selected Override")
        self.cancel_button = QPushButton("Cancel")
        self.apply_button = QPushButton("Apply")

        self.reset_all_button.clicked.connect(self.reset_all_requested.emit)
        self.clear_selected_override_button.clicked.connect(self.clear_selected_override_requested.emit)
        self.cancel_button.clicked.connect(self.reject)
        self.apply_button.clicked.connect(self.accept)

        action_row.addWidget(self.reset_all_button)
        action_row.addWidget(self.clear_selected_override_button)
        action_row.addStretch()
        action_row.addWidget(self.cancel_button)
        action_row.addWidget(self.apply_button)
        layout.addLayout(action_row)

    def current_params(self) -> ManualFitParams:
        values = {
            name: float(spinbox.value())
            for name, spinbox in self.param_spinboxes.items()
        }
        return ManualFitParams(**values)

    def reset_controls(self) -> None:
        for name in self.param_spinboxes:
            self._set_control_value(name, 0)

    def _set_control_value(self, name: str, value: int) -> None:
        slider = self.param_sliders[name]
        spinbox = self.param_spinboxes[name]
        with QSignalBlocker(slider), QSignalBlocker(spinbox):
            slider.setValue(value)
            spinbox.setValue(value)

    def _handle_slider_changed(self, name: str, value: int) -> None:
        with QSignalBlocker(self.param_spinboxes[name]):
            self.param_spinboxes[name].setValue(value)
        self.params_changed.emit(self.current_params())

    def _handle_spinbox_changed(self, name: str, value: int) -> None:
        with QSignalBlocker(self.param_sliders[name]):
            self.param_sliders[name].setValue(value)
        self.params_changed.emit(self.current_params())

    def _handle_mode_toggled(self) -> None:
        if self.group_fit_radio.isChecked():
            self.mode_status_label.setText("Editing: Whole Keyboard")
            self.mode_changed.emit("manual_fit_group")
        else:
            self.mode_status_label.setText("Editing: Single Overlay")
            self.mode_changed.emit("manual_fit_single")
