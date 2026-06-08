"""Manual keyboard fit tool window."""
from __future__ import annotations

from typing import Dict, Optional

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
    ("keyboard_top_delta", "Keyboard Top", -500, 500),
    ("left_edge_drift", "Left Edge Drift", -500, 500),
    ("right_edge_drift", "Right Edge Drift", -500, 500),
    ("black_width_delta", "Black Width", -500, 500),
]


class ManualKeyboardFitDialog(QDialog):
    """Modeless controls for manual overlay keyboard fitting."""

    params_changed = Signal(object)
    octave_changed = Signal(int)
    mode_changed = Signal(str)
    reset_all_requested = Signal()
    reset_position_requested = Signal()
    clear_selected_override_requested = Signal()
    setup_confirm_requested = Signal()

    def __init__(self, parent=None, *, initial_octave: int = 0):
        super().__init__(parent, Qt.Tool | Qt.WindowCloseButtonHint)
        self._initial_octave = int(initial_octave)
        self.setWindowTitle("Manual Fit")
        self.setModal(False)
        self.setWindowModality(Qt.NonModal)
        self.resize(760, 560)

        self.param_sliders: Dict[str, QSlider] = {}
        self.param_spinboxes: Dict[str, QSpinBox] = {}
        self.param_reset_buttons: Dict[str, QPushButton] = {}

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self.setup_group = QGroupBox("Setup")
        setup_layout = QGridLayout(self.setup_group)
        self.setup_step_label = QLabel("Fine Tune Overlays")
        self.setup_step_label.setStyleSheet("font-weight: bold;")
        self.setup_instruction_label = QLabel("")
        self.setup_instruction_label.setWordWrap(True)
        self.confirm_step_button = QPushButton("Confirm")
        self.confirm_step_button.setEnabled(False)
        self.confirm_step_button.clicked.connect(self.setup_confirm_requested.emit)
        setup_layout.addWidget(self.setup_step_label, 0, 0)
        setup_layout.addWidget(self.confirm_step_button, 0, 1)
        setup_layout.addWidget(self.setup_instruction_label, 1, 0, 1, 2)
        layout.addWidget(self.setup_group)

        self.mode_group = QGroupBox("Edit Mode")
        mode_layout = QHBoxLayout(self.mode_group)
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
        layout.addWidget(self.mode_group)

        octave_row = QHBoxLayout()
        octave_label = QLabel("Octave")
        octave_label.setStyleSheet("font-weight: bold;")
        self.octave_spinbox = QSpinBox()
        self.octave_spinbox.setRange(-5, 5)
        self.octave_spinbox.setValue(self._initial_octave)
        self.octave_spinbox.setFixedWidth(78)
        self.octave_spinbox.valueChanged.connect(self.octave_changed.emit)
        octave_row.addWidget(octave_label)
        octave_row.addWidget(self.octave_spinbox)
        octave_row.addStretch()
        layout.addLayout(octave_row)

        self.controls_group = QGroupBox("Keyboard Fit")
        controls_layout = QGridLayout(self.controls_group)
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
            reset_button = QPushButton("0")
            reset_button.setFixedWidth(32)
            reset_button.setToolTip(f"Reset {label}")

            slider.valueChanged.connect(
                lambda value, param_name=name: self._handle_slider_changed(param_name, value)
            )
            spinbox.valueChanged.connect(
                lambda value, param_name=name: self._handle_spinbox_changed(param_name, value)
            )
            reset_button.clicked.connect(
                lambda checked=False, param_name=name: self._handle_reset_param(param_name)
            )

            self.param_sliders[name] = slider
            self.param_spinboxes[name] = spinbox
            self.param_reset_buttons[name] = reset_button
            controls_layout.addWidget(label_widget, row, 0)
            controls_layout.addWidget(slider, row, 1)
            controls_layout.addWidget(spinbox, row, 2)
            controls_layout.addWidget(reset_button, row, 3)

        layout.addWidget(self.controls_group, 1)

        action_row = QHBoxLayout()
        self.reset_all_button = QPushButton("Reset All")
        self.reset_position_button = QPushButton("Reset Position")
        self.clear_selected_override_button = QPushButton("Clear Selected Override")
        self.cancel_button = QPushButton("Cancel")
        self.apply_button = QPushButton("Apply")

        self.reset_all_button.clicked.connect(self.reset_all_requested.emit)
        self.reset_position_button.clicked.connect(self.reset_position_requested.emit)
        self.clear_selected_override_button.clicked.connect(self.clear_selected_override_requested.emit)
        self.cancel_button.clicked.connect(self.reject)
        self.apply_button.clicked.connect(self.accept)

        action_row.addWidget(self.reset_all_button)
        action_row.addWidget(self.reset_position_button)
        action_row.addWidget(self.clear_selected_override_button)
        action_row.addStretch()
        action_row.addWidget(self.cancel_button)
        action_row.addWidget(self.apply_button)
        layout.addLayout(action_row)
        self.finish_setup()

    def current_params(self) -> ManualFitParams:
        values = {
            name: float(spinbox.value())
            for name, spinbox in self.param_spinboxes.items()
        }
        return ManualFitParams(**values)

    def reset_controls(self, *, octave_value: Optional[int] = None) -> None:
        for name in self.param_spinboxes:
            self._set_control_value(name, 0)
        if octave_value is None:
            octave_value = self._initial_octave
        with QSignalBlocker(self.octave_spinbox):
            self.octave_spinbox.setValue(int(octave_value))

    def enter_setup_step(self, step_name: str, *, can_confirm: bool = False) -> None:
        labels = {
            "keyboard_box": (
                "Draw Keyboard Area",
                "Draw one rectangle around the visible keyboard area.",
            ),
            "black_bottom": (
                "Set Black Key Bottom",
                "Drag the orange line to the bottom edge of the black keys.",
            ),
            "white_start": (
                "Set White Key Start",
                "Drag the blue line to the start of the white-key detection area.",
            ),
        }
        title, instruction = labels[step_name]
        self.setup_group.show()
        self.setup_step_label.setText(title)
        self.setup_instruction_label.setText(instruction)
        self.confirm_step_button.show()
        self.confirm_step_button.setEnabled(can_confirm)
        self.mode_group.setEnabled(False)
        self.controls_group.setEnabled(False)
        self.reset_all_button.setEnabled(False)
        self.reset_position_button.setEnabled(False)
        self.clear_selected_override_button.setEnabled(False)
        self.apply_button.setEnabled(False)

    def mark_setup_step_ready(self) -> None:
        self.confirm_step_button.setEnabled(True)

    def finish_setup(self) -> None:
        self.setup_group.show()
        self.setup_step_label.setText("Fine Tune Overlays")
        self.setup_instruction_label.setText("")
        self.confirm_step_button.hide()
        self.confirm_step_button.setEnabled(False)
        self.mode_group.setEnabled(True)
        self.controls_group.setEnabled(True)
        self.reset_all_button.setEnabled(True)
        self.reset_position_button.setEnabled(True)
        self.clear_selected_override_button.setEnabled(True)
        self.apply_button.setEnabled(True)

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

    def _handle_reset_param(self, name: str) -> None:
        self._set_control_value(name, 0)
        self.params_changed.emit(self.current_params())

    def _handle_mode_toggled(self) -> None:
        if self.group_fit_radio.isChecked():
            self.mode_status_label.setText("Editing: Whole Keyboard")
            self.mode_changed.emit("manual_fit_group")
        else:
            self.mode_status_label.setText("Editing: Single Overlay")
            self.mode_changed.emit("manual_fit_single")
