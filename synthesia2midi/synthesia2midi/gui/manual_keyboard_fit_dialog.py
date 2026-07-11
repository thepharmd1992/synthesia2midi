"""Manual keyboard fit tool window."""
from __future__ import annotations

from typing import Dict, Optional

from PySide6.QtCore import QCoreApplication, QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
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
    QWidget,
)

from synthesia2midi.workflows.manual_keyboard_fit import LocalFitParams, ManualFitParams


PARAM_SPECS = [
    ("keyboard_width_delta", "Keyboard Width", -1000, 1000),
    ("keyboard_top_delta", "Keyboard Top", -500, 500),
    ("left_edge_drift", "Left Edge Drift", -500, 500),
    ("right_edge_drift", "Right Edge Drift", -500, 500),
    ("white_width_delta", "White Width", -500, 500),
    ("black_width_delta", "Black Width", -500, 500),
    ("left_slant_delta", "Left Slant", -45, 45),
    ("right_slant_delta", "Right Slant", -45, 45),
]

LOCAL_PARAM_SPECS = [
    ("spread_delta", "Spacing", -500, 500),
    ("x_delta", "Move Left / Right", -500, 500),
    ("y_delta", "Move Up / Down", -500, 500),
    ("width_delta", "Overlay Width", -500, 500),
    ("slant_delta", "Tilt", -45, 45),
]


def _translate_param_label(label: str) -> str:
    translated = {
        "Keyboard Width": QCoreApplication.translate("ManualKeyboardFitDialog", "Keyboard Width"),
        "Keyboard Top": QCoreApplication.translate("ManualKeyboardFitDialog", "Keyboard Top"),
        "Left Edge Drift": QCoreApplication.translate("ManualKeyboardFitDialog", "Left Edge Drift"),
        "Right Edge Drift": QCoreApplication.translate("ManualKeyboardFitDialog", "Right Edge Drift"),
        "White Width": QCoreApplication.translate("ManualKeyboardFitDialog", "White Width"),
        "Black Width": QCoreApplication.translate("ManualKeyboardFitDialog", "Black Width"),
        "Left Slant": QCoreApplication.translate("ManualKeyboardFitDialog", "Left Slant"),
        "Right Slant": QCoreApplication.translate("ManualKeyboardFitDialog", "Right Slant"),
        "Spacing": QCoreApplication.translate("ManualKeyboardFitDialog", "Spacing"),
        "Move Left / Right": QCoreApplication.translate("ManualKeyboardFitDialog", "Move Left / Right"),
        "Move Up / Down": QCoreApplication.translate("ManualKeyboardFitDialog", "Move Up / Down"),
        "Overlay Width": QCoreApplication.translate("ManualKeyboardFitDialog", "Overlay Width"),
        "Tilt": QCoreApplication.translate("ManualKeyboardFitDialog", "Tilt"),
    }
    return translated.get(label, label)


class ManualKeyboardFitDialog(QDialog):
    """Modeless controls for manual overlay keyboard fitting."""

    params_changed = Signal(object)
    local_params_changed = Signal(object)
    octave_changed = Signal(int)
    mode_changed = Signal(str)
    reset_all_requested = Signal()
    reset_position_requested = Signal()
    reset_local_requested = Signal()
    clear_selected_override_requested = Signal()
    edit_keyboard_box_requested = Signal()
    setup_back_requested = Signal()
    setup_use_suggested_requested = Signal()

    def __init__(self, parent=None, *, initial_octave: int = 0):
        super().__init__(parent, Qt.Tool | Qt.WindowCloseButtonHint)
        self._initial_octave = int(initial_octave)
        self.setWindowTitle(QCoreApplication.translate("ManualKeyboardFitDialog", "Manual Fit"))
        self.setModal(False)
        self.setWindowModality(Qt.NonModal)
        self.resize(760, 560)

        self.param_sliders: Dict[str, QSlider] = {}
        self.param_spinboxes: Dict[str, QSpinBox] = {}
        self.param_reset_buttons: Dict[str, QPushButton] = {}
        self.param_row_widgets: Dict[str, list[QWidget]] = {}
        self.local_param_sliders: Dict[str, QSlider] = {}
        self.local_param_spinboxes: Dict[str, QSpinBox] = {}
        self.local_param_reset_buttons: Dict[str, QPushButton] = {}

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self.setup_group = QGroupBox(QCoreApplication.translate("ManualKeyboardFitDialog", "Setup"))
        setup_layout = QVBoxLayout(self.setup_group)
        self.setup_step_label = QLabel(QCoreApplication.translate("ManualKeyboardFitDialog", "Fine Tune Overlays"))
        self.setup_step_label.setStyleSheet("font-weight: bold;")
        self.setup_instruction_label = QLabel("")
        self.setup_instruction_label.setWordWrap(True)
        self.setup_back_button = QPushButton(QCoreApplication.translate("ManualKeyboardFitDialog", "Back"))
        self.setup_use_suggested_button = QPushButton(QCoreApplication.translate("ManualKeyboardFitDialog", "Use Suggested"))
        self.setup_cancel_button = QPushButton(QCoreApplication.translate("ManualKeyboardFitDialog", "Cancel"))
        self.setup_back_button.clicked.connect(self.setup_back_requested.emit)
        self.setup_use_suggested_button.clicked.connect(self.setup_use_suggested_requested.emit)
        self.setup_cancel_button.clicked.connect(self.reject)
        setup_button_row = QHBoxLayout()
        setup_button_row.addWidget(self.setup_back_button)
        setup_button_row.addWidget(self.setup_use_suggested_button)
        setup_button_row.addStretch()
        setup_button_row.addWidget(self.setup_cancel_button)
        setup_layout.addWidget(self.setup_step_label)
        setup_layout.addWidget(self.setup_instruction_label)
        setup_layout.addLayout(setup_button_row)
        layout.addWidget(self.setup_group)

        self.fine_tune_widget = QWidget(self)
        fine_tune_layout = QVBoxLayout(self.fine_tune_widget)
        fine_tune_layout.setContentsMargins(0, 0, 0, 0)
        fine_tune_layout.setSpacing(10)

        self.mode_group = QGroupBox(QCoreApplication.translate("ManualKeyboardFitDialog", "Edit Mode"))
        mode_layout = QVBoxLayout(self.mode_group)
        self.mode_choice_layout = QGridLayout()
        self.mode_choice_layout.setContentsMargins(0, 0, 0, 0)
        self.mode_choice_layout.setHorizontalSpacing(12)
        self.mode_choice_layout.setVerticalSpacing(4)
        self.mode_status_label = QLabel(
            QCoreApplication.translate(
                "ManualKeyboardFitDialog", "Move and resize every overlay together."
            )
        )
        self.mode_status_label.setWordWrap(True)
        self.group_fit_radio = QRadioButton(QCoreApplication.translate("ManualKeyboardFitDialog", "All Overlays"))
        self.all_white_radio = QRadioButton(QCoreApplication.translate("ManualKeyboardFitDialog", "All Whites"))
        self.all_black_radio = QRadioButton(QCoreApplication.translate("ManualKeyboardFitDialog", "All Blacks"))
        self.local_fit_radio = QRadioButton(QCoreApplication.translate("ManualKeyboardFitDialog", "Select Overlays"))
        self.single_overlay_radio = QRadioButton(QCoreApplication.translate("ManualKeyboardFitDialog", "Single Overlay"))
        self.group_fit_radio.setChecked(True)
        self._mode_button_group = QButtonGroup(self)
        self._mode_button_group.addButton(self.group_fit_radio)
        self._mode_button_group.addButton(self.all_white_radio)
        self._mode_button_group.addButton(self.all_black_radio)
        self._mode_button_group.addButton(self.local_fit_radio)
        self._mode_button_group.addButton(self.single_overlay_radio)
        self.group_fit_radio.toggled.connect(self._handle_mode_toggled)
        self.all_white_radio.toggled.connect(self._handle_mode_toggled)
        self.all_black_radio.toggled.connect(self._handle_mode_toggled)
        self.local_fit_radio.toggled.connect(self._handle_mode_toggled)
        self.single_overlay_radio.toggled.connect(self._handle_mode_toggled)
        self.mode_buttons = [
            self.group_fit_radio,
            self.all_white_radio,
            self.all_black_radio,
            self.local_fit_radio,
            self.single_overlay_radio,
        ]
        self._mode_layout_columns = 3
        self._reflow_grid(self.mode_choice_layout, self.mode_buttons, 3)
        mode_layout.addLayout(self.mode_choice_layout)
        mode_layout.addWidget(self.mode_status_label)
        fine_tune_layout.addWidget(self.mode_group)

        self.octave_widget = QWidget()
        octave_row = QHBoxLayout(self.octave_widget)
        octave_row.setContentsMargins(0, 0, 0, 0)
        octave_label = QLabel(QCoreApplication.translate("ManualKeyboardFitDialog", "Octave"))
        octave_label.setStyleSheet("font-weight: bold;")
        self.octave_spinbox = QSpinBox()
        self.octave_spinbox.setRange(-5, 5)
        self.octave_spinbox.setValue(self._initial_octave)
        self.octave_spinbox.setFixedWidth(78)
        self.octave_spinbox.valueChanged.connect(self.octave_changed.emit)
        octave_row.addWidget(octave_label)
        octave_row.addWidget(self.octave_spinbox)
        octave_row.addStretch()
        fine_tune_layout.addWidget(self.octave_widget)

        self.controls_group = QGroupBox("")
        controls_layout = QGridLayout(self.controls_group)
        controls_layout.setHorizontalSpacing(10)
        controls_layout.setVerticalSpacing(6)

        for row, (name, label, minimum, maximum) in enumerate(PARAM_SPECS):
            label_widget = QLabel(_translate_param_label(label))
            slider = QSlider(Qt.Horizontal)
            slider.setRange(minimum, maximum)
            slider.setValue(0)
            spinbox = QSpinBox()
            spinbox.setRange(minimum, maximum)
            spinbox.setValue(0)
            spinbox.setFixedWidth(78)
            reset_button = QPushButton("0")
            reset_button.setFixedSize(36, 36)
            reset_text = QCoreApplication.translate(
                "ManualKeyboardFitDialog", "Reset {label}"
            ).format(
                label=_translate_param_label(label)
            )
            reset_button.setToolTip(reset_text)
            reset_button.setAccessibleName(reset_text)
            reset_button.setAccessibleDescription(reset_text)

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
            self.param_row_widgets[name] = [label_widget, slider, spinbox, reset_button]
            controls_layout.addWidget(label_widget, row, 0)
            controls_layout.addWidget(slider, row, 1)
            controls_layout.addWidget(spinbox, row, 2)
            controls_layout.addWidget(reset_button, row, 3)

        fine_tune_layout.addWidget(self.controls_group, 1)

        self.local_controls_group = QGroupBox("")
        local_layout = QGridLayout(self.local_controls_group)
        local_layout.setHorizontalSpacing(10)
        local_layout.setVerticalSpacing(6)

        local_filter_label = QLabel(QCoreApplication.translate("ManualKeyboardFitDialog", "Select"))
        self.local_filter_combo = QComboBox()
        self.local_filter_combo.addItem(QCoreApplication.translate("ManualKeyboardFitDialog", "Black Keys"), "black")
        self.local_filter_combo.addItem(QCoreApplication.translate("ManualKeyboardFitDialog", "White Keys"), "white")
        self.local_filter_combo.addItem(QCoreApplication.translate("ManualKeyboardFitDialog", "All Keys"), "all")
        self.local_selection_label = QLabel(QCoreApplication.translate("ManualKeyboardFitDialog", "Draw a box around problem keys"))
        local_layout.addWidget(local_filter_label, 0, 0)
        local_layout.addWidget(self.local_filter_combo, 0, 1)
        local_layout.addWidget(self.local_selection_label, 0, 2, 1, 2)

        for row, (name, label, minimum, maximum) in enumerate(LOCAL_PARAM_SPECS, start=1):
            label_widget = QLabel(_translate_param_label(label))
            slider = QSlider(Qt.Horizontal)
            slider.setRange(minimum, maximum)
            slider.setValue(0)
            spinbox = QSpinBox()
            spinbox.setRange(minimum, maximum)
            spinbox.setValue(0)
            spinbox.setFixedWidth(78)
            reset_button = QPushButton("0")
            reset_button.setFixedSize(36, 36)
            reset_text = QCoreApplication.translate(
                "ManualKeyboardFitDialog", "Reset {label}"
            ).format(
                label=_translate_param_label(label)
            )
            reset_button.setToolTip(reset_text)
            reset_button.setAccessibleName(reset_text)
            reset_button.setAccessibleDescription(reset_text)

            slider.valueChanged.connect(
                lambda value, param_name=name: self._handle_local_slider_changed(param_name, value)
            )
            spinbox.valueChanged.connect(
                lambda value, param_name=name: self._handle_local_spinbox_changed(param_name, value)
            )
            reset_button.clicked.connect(
                lambda checked=False, param_name=name: self._handle_reset_local_param(param_name)
            )

            self.local_param_sliders[name] = slider
            self.local_param_spinboxes[name] = spinbox
            self.local_param_reset_buttons[name] = reset_button
            local_layout.addWidget(label_widget, row, 0)
            local_layout.addWidget(slider, row, 1)
            local_layout.addWidget(spinbox, row, 2)
            local_layout.addWidget(reset_button, row, 3)

        fine_tune_layout.addWidget(self.local_controls_group, 1)
        self.set_local_selection_count(0)

        self.reset_all_button = QPushButton(QCoreApplication.translate("ManualKeyboardFitDialog", "Reset All"))
        self.reset_position_button = QPushButton(QCoreApplication.translate("ManualKeyboardFitDialog", "Reset Position"))
        self.reset_local_button = QPushButton(QCoreApplication.translate("ManualKeyboardFitDialog", "Reset Local"))
        self.edit_keyboard_box_button = QPushButton(QCoreApplication.translate("ManualKeyboardFitDialog", "Edit Keyboard Box"))
        self.clear_selected_override_button = QPushButton(QCoreApplication.translate("ManualKeyboardFitDialog", "Clear Selected Override"))
        self.cancel_button = QPushButton(QCoreApplication.translate("ManualKeyboardFitDialog", "Cancel"))
        self.apply_button = QPushButton(QCoreApplication.translate("ManualKeyboardFitDialog", "Apply"))

        self.reset_all_button.clicked.connect(self.reset_all_requested.emit)
        self.reset_position_button.clicked.connect(self.reset_position_requested.emit)
        self.reset_local_button.clicked.connect(self.reset_local_requested.emit)
        self.edit_keyboard_box_button.clicked.connect(self.edit_keyboard_box_requested.emit)
        self.clear_selected_override_button.clicked.connect(self.clear_selected_override_requested.emit)
        self.cancel_button.clicked.connect(self.reject)
        self.apply_button.clicked.connect(self.accept)

        self.secondary_action_buttons = [
            self.reset_all_button,
            self.reset_position_button,
            self.reset_local_button,
            self.edit_keyboard_box_button,
            self.clear_selected_override_button,
        ]
        self.secondary_actions_widget = QWidget()
        self.secondary_actions_layout = QGridLayout(self.secondary_actions_widget)
        self.secondary_actions_layout.setContentsMargins(0, 0, 0, 0)
        self.secondary_actions_layout.setHorizontalSpacing(8)
        self.secondary_actions_layout.setVerticalSpacing(6)
        self._secondary_layout_columns = 3
        self._reflow_grid(
            self.secondary_actions_layout, self.secondary_action_buttons, 3
        )
        fine_tune_layout.addWidget(self.secondary_actions_widget)

        self.action_footer = QWidget()
        action_footer_layout = QHBoxLayout(self.action_footer)
        action_footer_layout.setContentsMargins(0, 0, 0, 0)
        action_footer_layout.addStretch()
        action_footer_layout.addWidget(self.cancel_button)
        action_footer_layout.addWidget(self.apply_button)
        fine_tune_layout.addWidget(self.action_footer)
        layout.addWidget(self.fine_tune_widget, 1)
        self.finish_setup()

    def current_params(self) -> ManualFitParams:
        values = {
            name: float(spinbox.value())
            for name, spinbox in self.param_spinboxes.items()
        }
        return ManualFitParams(**values)

    def current_local_params(self) -> LocalFitParams:
        values = {
            name: float(spinbox.value())
            for name, spinbox in self.local_param_spinboxes.items()
        }
        return LocalFitParams(**values)

    def current_local_filter(self) -> str:
        return str(self.local_filter_combo.currentData() or "black")

    def reset_controls(self, *, octave_value: Optional[int] = None) -> None:
        for name in self.param_spinboxes:
            self._set_control_value(name, 0)
        self.reset_local_controls()
        self.set_local_selection_count(0)
        if octave_value is None:
            octave_value = self._initial_octave
        with QSignalBlocker(self.octave_spinbox):
            self.octave_spinbox.setValue(int(octave_value))

    def reset_local_controls(self) -> None:
        for name in self.local_param_spinboxes:
            self._set_local_control_value(name, 0)

    def set_params(self, params: ManualFitParams) -> None:
        for name in self.param_spinboxes:
            self._set_control_value(name, int(getattr(params, name)))

    def set_local_params(self, params: LocalFitParams) -> None:
        for name in self.local_param_spinboxes:
            self._set_local_control_value(name, int(getattr(params, name)))

    def set_local_selection_count(self, count: int) -> None:
        self._set_local_controls_enabled(count > 0)
        if count <= 0:
            self.local_selection_label.setText(
                QCoreApplication.translate("ManualKeyboardFitDialog", "Draw a box around problem keys")
            )
            return
        self.local_selection_label.setText(
            QCoreApplication.translate("ManualKeyboardFitDialog", "{count} selected").format(count=count)
        )

    def enter_setup_step(self, step_name: str) -> None:
        labels = {
            "keyboard_box": (
                QCoreApplication.translate("ManualKeyboardFitDialog", "Step 1 of 3: Draw Keyboard Area"),
                QCoreApplication.translate(
                    "ManualKeyboardFitDialog",
                    "Draw one rectangle around the visible keyboard area.",
                ),
            ),
            "keyboard_box_edit": (
                QCoreApplication.translate("ManualKeyboardFitDialog", "Edit Keyboard Box"),
                QCoreApplication.translate(
                    "ManualKeyboardFitDialog",
                    "Adjust the green boundary bars, or draw a replacement box around the visible keyboard area.",
                ),
            ),
            "black_bottom": (
                QCoreApplication.translate("ManualKeyboardFitDialog", "Step 2 of 3: Set Black Key Bottom"),
                QCoreApplication.translate(
                    "ManualKeyboardFitDialog",
                    "Drag the orange line to slightly above the bottom of black keys.",
                ),
            ),
            "white_start": (
                QCoreApplication.translate("ManualKeyboardFitDialog", "Step 3 of 3: Set White Key Start"),
                QCoreApplication.translate(
                    "ManualKeyboardFitDialog",
                    "Drag the blue line to a bit under the black keys.",
                ),
            ),
        }
        title, instruction = labels[step_name]
        self.setup_group.show()
        self.fine_tune_widget.hide()
        self.setup_step_label.setText(title)
        self.setup_instruction_label.setText(instruction)
        show_step_controls = step_name in {"black_bottom", "white_start"}
        if step_name == "keyboard_box_edit":
            show_step_controls = True
            self.setup_use_suggested_button.setText(QCoreApplication.translate("ManualKeyboardFitDialog", "OK"))
        else:
            self.setup_use_suggested_button.setText(
                QCoreApplication.translate("ManualKeyboardFitDialog", "Use Suggested")
            )
        self.setup_back_button.setVisible(show_step_controls)
        self.setup_use_suggested_button.setVisible(show_step_controls and step_name != "keyboard_box_edit")
        self.adjustSize()

    def finish_setup(self) -> None:
        self.setup_group.hide()
        self.fine_tune_widget.show()
        self.setup_step_label.setText(QCoreApplication.translate("ManualKeyboardFitDialog", "Fine Tune Overlays"))
        self.setup_instruction_label.setText("")
        self.setup_use_suggested_button.setText(
            QCoreApplication.translate("ManualKeyboardFitDialog", "Use Suggested")
        )
        self._sync_mode_control_visibility()
        self.resize(760, 560)

    def set_keyboard_box_edit_confirm_visible(self, visible: bool) -> None:
        self.setup_use_suggested_button.setText(QCoreApplication.translate("ManualKeyboardFitDialog", "OK"))
        self.setup_use_suggested_button.setVisible(visible)
        self.adjustSize()

    def _set_control_value(self, name: str, value: int) -> None:
        slider = self.param_sliders[name]
        spinbox = self.param_spinboxes[name]
        with QSignalBlocker(slider), QSignalBlocker(spinbox):
            slider.setValue(value)
            spinbox.setValue(value)

    def _set_local_control_value(self, name: str, value: int) -> None:
        slider = self.local_param_sliders[name]
        spinbox = self.local_param_spinboxes[name]
        with QSignalBlocker(slider), QSignalBlocker(spinbox):
            slider.setValue(value)
            spinbox.setValue(value)

    def _set_local_controls_enabled(self, enabled: bool) -> None:
        for widget_by_name in (
            self.local_param_sliders,
            self.local_param_spinboxes,
            self.local_param_reset_buttons,
        ):
            for widget in widget_by_name.values():
                widget.setEnabled(enabled)

    def _handle_slider_changed(self, name: str, value: int) -> None:
        with QSignalBlocker(self.param_spinboxes[name]):
            self.param_spinboxes[name].setValue(value)
        self.params_changed.emit(self.current_params())

    def _handle_spinbox_changed(self, name: str, value: int) -> None:
        with QSignalBlocker(self.param_sliders[name]):
            self.param_sliders[name].setValue(value)
        self.params_changed.emit(self.current_params())

    def _handle_local_slider_changed(self, name: str, value: int) -> None:
        with QSignalBlocker(self.local_param_spinboxes[name]):
            self.local_param_spinboxes[name].setValue(value)
        self.local_params_changed.emit(self.current_local_params())

    def _handle_local_spinbox_changed(self, name: str, value: int) -> None:
        with QSignalBlocker(self.local_param_sliders[name]):
            self.local_param_sliders[name].setValue(value)
        self.local_params_changed.emit(self.current_local_params())

    def _handle_reset_param(self, name: str) -> None:
        self._set_control_value(name, 0)
        self.params_changed.emit(self.current_params())

    def _handle_reset_local_param(self, name: str) -> None:
        self._set_local_control_value(name, 0)
        self.local_params_changed.emit(self.current_local_params())

    def _handle_mode_toggled(self) -> None:
        if self.group_fit_radio.isChecked():
            self.mode_status_label.setText(
                QCoreApplication.translate(
                    "ManualKeyboardFitDialog", "Move and resize every overlay together."
                )
            )
            self.mode_changed.emit("manual_fit_group")
        elif self.all_white_radio.isChecked():
            self.mode_status_label.setText(
                QCoreApplication.translate(
                    "ManualKeyboardFitDialog", "Adjust all white-key overlays together."
                )
            )
            self.mode_changed.emit("manual_fit_all_white")
        elif self.all_black_radio.isChecked():
            self.mode_status_label.setText(
                QCoreApplication.translate(
                    "ManualKeyboardFitDialog", "Adjust all black-key overlays together."
                )
            )
            self.mode_changed.emit("manual_fit_all_black")
        elif self.local_fit_radio.isChecked():
            self.mode_status_label.setText(
                QCoreApplication.translate(
                    "ManualKeyboardFitDialog",
                    "Draw around the problem keys, then adjust that selected group.",
                )
            )
            self.mode_changed.emit("manual_fit_local_select")
        elif self.single_overlay_radio.isChecked():
            self.mode_status_label.setText(
                QCoreApplication.translate(
                    "ManualKeyboardFitDialog", "Click one problem key, then adjust only that overlay."
                )
            )
            self.mode_changed.emit("manual_fit_single")
        self._sync_mode_control_visibility()

    def _sync_mode_control_visibility(self) -> None:
        group_controls_visible = (
            self.group_fit_radio.isChecked()
            or self.all_white_radio.isChecked()
            or self.all_black_radio.isChecked()
        )
        self.controls_group.setVisible(group_controls_visible)
        local_controls_visible = self.local_fit_radio.isChecked()
        single_overlay_mode = self.single_overlay_radio.isChecked()
        self.local_controls_group.setVisible(local_controls_visible)
        self.octave_widget.setVisible(not single_overlay_mode)
        self.reset_all_button.setVisible(not single_overlay_mode)
        self.reset_position_button.setVisible(group_controls_visible)
        self.reset_local_button.setVisible(local_controls_visible)
        self.edit_keyboard_box_button.setVisible(group_controls_visible)
        self.clear_selected_override_button.setVisible(single_overlay_mode)
        self.secondary_actions_widget.setVisible(
            any(not button.isHidden() for button in self.secondary_action_buttons)
        )
        self._set_param_row_visible("white_width_delta", not self.all_black_radio.isChecked())
        self._set_param_row_visible("black_width_delta", not self.all_white_radio.isChecked())
        self.layout().activate()
        if single_overlay_mode:
            self.resize(min(self.width(), 680), 300)
        elif self.fine_tune_widget.isVisible():
            self.resize(max(self.width(), 760), 560)

    def _set_param_row_visible(self, name: str, visible: bool) -> None:
        for widget in self.param_row_widgets.get(name, []):
            widget.setVisible(visible)

    @staticmethod
    def _reflow_grid(layout: QGridLayout, widgets: list[QWidget], columns: int) -> None:
        for widget in widgets:
            layout.removeWidget(widget)
        for index, widget in enumerate(widgets):
            layout.addWidget(widget, index // columns, index % columns)
        for column in range(columns):
            layout.setColumnStretch(column, 1)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        columns = 3 if event.size().width() >= 700 else 2
        if columns != getattr(self, "_mode_layout_columns", columns):
            self._mode_layout_columns = columns
            self._reflow_grid(self.mode_choice_layout, self.mode_buttons, columns)
        if columns != getattr(self, "_secondary_layout_columns", columns):
            self._secondary_layout_columns = columns
            self._reflow_grid(
                self.secondary_actions_layout,
                self.secondary_action_buttons,
                columns,
            )
