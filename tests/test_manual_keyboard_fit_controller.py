from types import SimpleNamespace

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtWidgets import QApplication, QLabel, QMessageBox, QWidget

from synthesia2midi.app_config import OverlayConfig
from synthesia2midi.core.app_state import AppState
from synthesia2midi.gui.manual_keyboard_fit_controller import ManualKeyboardFitController


def _flush_qt_deletes():
    QApplication.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    QApplication.processEvents()


def _overlay(key_id=1, note="C", x=0, y=20, width=10, height=40):
    return OverlayConfig(
        key_id=key_id,
        note_octave=4,
        note_name_in_octave=note,
        x=x,
        y=y,
        width=width,
        height=height,
        key_type="LW",
    )


class _SettingsToolWindow:
    def __init__(self, visible=True):
        self.visible = visible
        self.hidden = False
        self.restored = False

    def isVisible(self):
        return self.visible

    def hide(self):
        self.hidden = True
        self.visible = False

    def show_preserving_geometry(self):
        self.restored = True
        self.visible = True


class _KeyboardCanvas:
    def __init__(self):
        self.mode = "off"
        self.callbacks = {}
        self.updates = 0

    def set_manual_fit_mode(self, mode):
        self.mode = mode

    def set_manual_fit_callbacks(self, **callbacks):
        self.callbacks = callbacks

    def clear_manual_fit_callbacks(self):
        self.callbacks = {}
        self.mode = "off"

    def update(self):
        self.updates += 1


class _FakeApp(QWidget):
    def __init__(self):
        super().__init__()
        self.app_state = AppState()
        self.keyboard_canvas = _KeyboardCanvas()
        self.settings_tool_window = _SettingsToolWindow()
        self.control_updates = 0
        self.show_overlays_action = SimpleNamespace(checked=None, setChecked=lambda value: setattr(self.show_overlays_action, "checked", value))
        self.control_panel = SimpleNamespace(
            update_controls_from_state=lambda: setattr(self, "control_updates", self.control_updates + 1),
            update_selected_overlay_display=lambda: None,
        )


def test_manual_fit_controller_warns_when_no_overlays(monkeypatch):
    QApplication.instance() or QApplication([])
    app = _FakeApp()
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args))

    try:
        assert ManualKeyboardFitController(app).open() is False

        assert warnings
        assert "Generate overlays" in warnings[0][2]
    finally:
        app.close()
        app.deleteLater()
        _flush_qt_deletes()


def test_manual_fit_controller_opens_modeless_top_center_and_restores_settings():
    QApplication.instance() or QApplication([])
    app = _FakeApp()
    app.app_state.overlays = [_overlay()]
    controller = ManualKeyboardFitController(app)

    try:
        assert controller.open() is True

        dialog = controller.active_dialog
        assert dialog is not None
        assert dialog.isVisible()
        assert dialog.windowModality() == Qt.NonModal
        assert not dialog.isModal()
        assert dialog.group_fit_radio.isChecked()
        assert app.settings_tool_window.hidden is True
        assert app.keyboard_canvas.mode == "manual_fit_group"

        screen_rect = QApplication.primaryScreen().availableGeometry()
        dialog_rect = dialog.frameGeometry()
        assert abs(dialog_rect.center().x() - screen_rect.center().x()) <= 80
        assert dialog_rect.top() <= screen_rect.top() + 40

        dialog.reject()
        QApplication.processEvents()

        assert app.settings_tool_window.restored is True
        assert app.keyboard_canvas.mode == "off"
    finally:
        if controller.active_dialog is not None:
            controller.active_dialog.reject()
        app.close()
        app.deleteLater()
        _flush_qt_deletes()


def test_manual_fit_dialog_uses_drawn_region_controls():
    QApplication.instance() or QApplication([])
    app = _FakeApp()
    app.app_state.overlays = [_overlay()]
    controller = ManualKeyboardFitController(app)

    try:
        assert controller.open() is True
        dialog = controller.active_dialog

        expected_controls = {
            "keyboard_width_delta",
            "keyboard_top_delta",
            "left_edge_drift",
            "right_edge_drift",
            "black_width_delta",
            "left_slant_delta",
            "right_slant_delta",
        }
        removed_controls = {
            "white_y_delta",
            "white_width_delta",
            "black_x_delta",
            "white_height_delta",
            "black_y_delta",
            "black_height_delta",
            "white_band_top_delta",
            "white_band_bottom_delta",
            "black_band_top_delta",
            "black_band_bottom_delta",
            "white_x_inset",
            "black_x_inset",
        }

        assert expected_controls.issubset(dialog.param_spinboxes)
        assert removed_controls.isdisjoint(dialog.param_spinboxes)
        assert dialog.setup_step_label.text() == "Fine Tune Overlays"
        assert not hasattr(dialog, "set_black_region_button")
        assert not hasattr(dialog, "set_white_region_button")
    finally:
        if controller.active_dialog is not None:
            controller.active_dialog.reject()
        app.close()
        app.deleteLater()
        _flush_qt_deletes()


def test_manual_fit_dialog_modes_show_only_relevant_controls():
    QApplication.instance() or QApplication([])
    app = _FakeApp()
    app.app_state.overlays = [
        _overlay(key_id=1, note="C", x=0, y=20, width=10, height=40),
        _overlay(key_id=2, note="C♯", x=12, y=10, width=6, height=20),
    ]
    controller = ManualKeyboardFitController(app)

    try:
        assert controller.open() is True
        dialog = controller.active_dialog

        assert dialog.group_fit_radio.isChecked()
        assert dialog.group_fit_radio.text() == "All Overlays"
        assert dialog.local_fit_radio.text() == "Select Overlays"
        assert dialog.controls_group.isVisible()
        assert not dialog.local_controls_group.isVisible()
        assert not dialog.mode_status_label.isVisible()
        assert dialog.controls_group.title() == ""
        assert dialog.local_controls_group.title() == ""
        assert dialog.current_local_filter() == "black"
        local_labels = {
            label.text()
            for label in dialog.local_controls_group.findChildren(QLabel)
        }
        assert "Cluster X" not in local_labels
        assert "Cluster Spread" not in local_labels
        assert {"Move Left / Right", "Move Up / Down", "Spacing", "Overlay Width", "Tilt"}.issubset(
            local_labels
        )

        dialog.local_fit_radio.click()

        assert app.keyboard_canvas.mode == "manual_fit_local_select"
        assert not dialog.controls_group.isVisible()
        assert dialog.local_controls_group.isVisible()
        assert not dialog.local_param_spinboxes["x_delta"].isEnabled()

        dialog.single_overlay_radio.click()

        assert app.keyboard_canvas.mode == "manual_fit_single"
        assert not dialog.controls_group.isVisible()
        assert not dialog.local_controls_group.isVisible()
    finally:
        if controller.active_dialog is not None:
            controller.active_dialog.reject()
        app.close()
        app.deleteLater()
        _flush_qt_deletes()


def test_manual_fit_controller_local_fit_selects_black_cluster_and_applies_local_controls():
    QApplication.instance() or QApplication([])
    app = _FakeApp()
    app.app_state.overlays = [
        _overlay(key_id=1, note="C", x=0, y=20, width=10, height=40),
        _overlay(key_id=2, note="C♯", x=12, y=10, width=6, height=20),
        _overlay(key_id=3, note="D", x=24, y=20, width=10, height=40),
        _overlay(key_id=4, note="D♯", x=36, y=10, width=6, height=20),
    ]
    controller = ManualKeyboardFitController(app)

    try:
        assert controller.open() is True
        dialog = controller.active_dialog
        dialog.local_fit_radio.click()

        app.keyboard_canvas.callbacks["local_selection_callback"](0, 0, 50, 50)
        assert controller.session.active_local_key_ids() == {2, 4}
        assert app.keyboard_canvas.callbacks["local_key_ids_callback"]() == {2, 4}
        assert dialog.local_param_spinboxes["x_delta"].isEnabled()

        dialog.local_param_spinboxes["x_delta"].setValue(1)
        white_x = app.app_state.overlays[0].x
        black_x = app.app_state.overlays[1].x
        dialog.local_param_spinboxes["x_delta"].setValue(8)

        assert app.app_state.overlays[0].x == pytest.approx(white_x)
        assert app.app_state.overlays[1].x == pytest.approx(black_x + 7)
    finally:
        if controller.active_dialog is not None:
            controller.active_dialog.reject()
        app.close()
        app.deleteLater()
        _flush_qt_deletes()


def test_manual_fit_controller_local_drag_moves_active_selected_cluster_only():
    QApplication.instance() or QApplication([])
    app = _FakeApp()
    app.app_state.overlays = [
        _overlay(key_id=1, note="C", x=0, y=20, width=10, height=40),
        _overlay(key_id=2, note="C♯", x=12, y=10, width=6, height=20),
        _overlay(key_id=3, note="D", x=24, y=20, width=10, height=40),
        _overlay(key_id=4, note="D♯", x=36, y=10, width=6, height=20),
    ]
    controller = ManualKeyboardFitController(app)

    try:
        assert controller.open() is True
        dialog = controller.active_dialog
        dialog.local_fit_radio.click()
        app.keyboard_canvas.callbacks["local_selection_callback"](0, 0, 50, 50)

        app.keyboard_canvas.callbacks["local_group_move_callback"](1, 0)
        white_x = app.app_state.overlays[0].x
        first_black_x = app.app_state.overlays[1].x
        first_black_y = app.app_state.overlays[1].y
        second_black_x = app.app_state.overlays[3].x
        second_black_y = app.app_state.overlays[3].y
        app.keyboard_canvas.callbacks["local_group_move_callback"](6, -4)

        assert app.app_state.overlays[0].x == pytest.approx(white_x)
        assert app.app_state.overlays[1].x == pytest.approx(first_black_x + 6)
        assert app.app_state.overlays[1].y == pytest.approx(first_black_y - 4)
        assert app.app_state.overlays[3].x == pytest.approx(second_black_x + 6)
        assert app.app_state.overlays[3].y == pytest.approx(second_black_y - 4)
    finally:
        if controller.active_dialog is not None:
            controller.active_dialog.reject()
        app.close()
        app.deleteLater()
        _flush_qt_deletes()


def test_manual_fit_dialog_reset_and_clear_selected_override_update_preview():
    QApplication.instance() or QApplication([])
    app = _FakeApp()
    app.app_state.overlays = [_overlay()]
    app.app_state.ui.selected_overlay_id = 1
    controller = ManualKeyboardFitController(app)
    try:
        assert controller.open() is True
        dialog = controller.active_dialog

        dialog.param_spinboxes["keyboard_top_delta"].setValue(6)
        assert app.app_state.overlays[0].y == 31.1

        app.keyboard_canvas.callbacks["single_move_callback"](0, 100, 50)
        assert controller.session.overridden_key_ids() == {1}

        dialog.clear_selected_override_button.click()
        assert controller.session.overridden_key_ids() == set()

        dialog.reset_all_button.click()
        assert dialog.param_spinboxes["keyboard_top_delta"].value() == 0
        assert app.app_state.overlays[0].height == 28

        dialog.reject()
        QApplication.processEvents()
    finally:
        if controller.active_dialog is not None:
            controller.active_dialog.reject()
        app.close()
        app.deleteLater()
        _flush_qt_deletes()


def test_manual_fit_setup_uses_compact_coach_and_auto_advances_to_fine_tune():
    QApplication.instance() or QApplication([])
    app = _FakeApp()
    app.app_state.overlays = [
        _overlay(key_id=1, note="C", x=0, y=20, width=10, height=40),
        _overlay(key_id=2, note="C♯", x=12, y=10, width=6, height=20),
        _overlay(key_id=3, note="D", x=24, y=20, width=10, height=40),
    ]
    controller = ManualKeyboardFitController(app)

    try:
        assert controller.open(start_setup=True) is True
        dialog = controller.active_dialog

        assert app.keyboard_canvas.mode == "manual_fit_keyboard_box"
        assert dialog.setup_step_label.text() == "Step 1 of 3: Draw Keyboard Area"
        assert dialog.setup_cancel_button.isVisible()
        assert not dialog.setup_back_button.isVisible()
        assert not dialog.setup_use_suggested_button.isVisible()
        assert dialog.setup_group.isVisible()
        assert not dialog.fine_tune_widget.isVisible()
        assert app.keyboard_canvas.callbacks["overlays_visible_callback"]() is False
        assert app.keyboard_canvas.callbacks["setup_instruction_callback"]() == (
            "Draw a box around the visible keyboard"
        )

        app.keyboard_canvas.callbacks["keyboard_box_selected_callback"](10, 100, 70, 200)
        assert app.keyboard_canvas.mode == "manual_fit_black_bottom"
        assert dialog.setup_step_label.text() == "Step 2 of 3: Set Black Key Bottom"
        assert dialog.setup_back_button.isVisible()
        assert dialog.setup_use_suggested_button.isVisible()
        assert set(app.keyboard_canvas.callbacks["region_guides_callback"]()) == {
            "keyboard_box",
            "black",
        }
        assert app.keyboard_canvas.callbacks["setup_instruction_callback"]() == (
            "Drag to slightly above the bottom of black keys"
        )

        app.keyboard_canvas.callbacks["guide_line_changed_callback"]("black_bottom", 140)
        assert app.keyboard_canvas.mode == "manual_fit_black_bottom"
        app.keyboard_canvas.callbacks["guide_line_selected_callback"]("black_bottom", 140)

        assert app.keyboard_canvas.mode == "manual_fit_white_start"
        assert dialog.setup_step_label.text() == "Step 3 of 3: Set White Key Start"
        assert set(app.keyboard_canvas.callbacks["region_guides_callback"]()) == {
            "keyboard_box",
            "black",
            "white",
        }
        assert app.keyboard_canvas.callbacks["setup_instruction_callback"]() == (
            "Drag to a bit under the black keys"
        )

        app.keyboard_canvas.callbacks["guide_line_selected_callback"]("white_start", 152)

        assert app.keyboard_canvas.mode == "manual_fit_group"
        assert not dialog.setup_group.isVisible()
        assert dialog.fine_tune_widget.isVisible()
        assert app.keyboard_canvas.callbacks["overlays_visible_callback"]() is True
        white_left, black, white_right = app.app_state.overlays
        assert (white_left.x, white_left.y, white_left.width, white_left.height) == pytest.approx(
            (13, 159.2, 24, 33.6)
        )
        assert (black.x, black.y, black.width, black.height) == pytest.approx(
            (32.8, 106, 14.4, 28)
        )
        assert (white_right.x, white_right.y, white_right.width, white_right.height) == pytest.approx(
            (43, 159.2, 24, 33.6)
        )
    finally:
        if controller.active_dialog is not None:
            controller.active_dialog.reject()
        app.close()
        app.deleteLater()
        _flush_qt_deletes()


def test_manual_fit_start_setup_restarts_existing_session_without_restoring_stale_overlays():
    QApplication.instance() or QApplication([])
    app = _FakeApp()
    app.app_state.overlays = [
        _overlay(key_id=1, note="C", x=0, y=20, width=10, height=40),
        _overlay(key_id=2, note="C♯", x=12, y=10, width=6, height=20),
        _overlay(key_id=3, note="D", x=24, y=20, width=10, height=40),
    ]
    controller = ManualKeyboardFitController(app)

    try:
        assert controller.open() is True
        stale_dialog = controller.active_dialog
        assert app.keyboard_canvas.mode == "manual_fit_group"

        app.settings_tool_window.visible = True
        app.settings_tool_window.hidden = False
        app.app_state.overlays = [
            _overlay(key_id=1, note="C", x=100, y=120, width=14, height=44),
            _overlay(key_id=2, note="C♯", x=116, y=110, width=8, height=24),
            _overlay(key_id=3, note="D", x=132, y=120, width=14, height=44),
        ]

        assert controller.open(start_setup=True) is True

        assert controller.active_dialog is not stale_dialog
        assert app.keyboard_canvas.mode == "manual_fit_keyboard_box"
        assert app.settings_tool_window.hidden is True
        assert app.keyboard_canvas.callbacks["overlays_visible_callback"]() is False
        assert app.keyboard_canvas.callbacks["setup_instruction_callback"]() == (
            "Draw a box around the visible keyboard"
        )
        assert app.app_state.overlays[0].x == 100
    finally:
        if controller.active_dialog is not None:
            controller.active_dialog.reject()
        app.close()
        app.deleteLater()
        _flush_qt_deletes()


def test_manual_fit_setup_back_and_use_suggested_keep_user_on_canvas_flow():
    QApplication.instance() or QApplication([])
    app = _FakeApp()
    app.app_state.overlays = [
        _overlay(key_id=1, note="C", x=0, y=20, width=10, height=40),
        _overlay(key_id=2, note="C♯", x=12, y=10, width=6, height=20),
        _overlay(key_id=3, note="D", x=24, y=20, width=10, height=40),
    ]
    controller = ManualKeyboardFitController(app)

    try:
        assert controller.open(start_setup=True) is True
        dialog = controller.active_dialog

        app.keyboard_canvas.callbacks["keyboard_box_selected_callback"](10, 100, 70, 200)
        assert app.keyboard_canvas.mode == "manual_fit_black_bottom"

        dialog.setup_back_button.click()
        assert app.keyboard_canvas.mode == "manual_fit_keyboard_box"
        assert dialog.setup_step_label.text() == "Step 1 of 3: Draw Keyboard Area"
        assert set(app.keyboard_canvas.callbacks["region_guides_callback"]()) == {"keyboard_box"}

        app.keyboard_canvas.callbacks["keyboard_box_selected_callback"](10, 100, 70, 200)
        dialog.setup_use_suggested_button.click()
        assert app.keyboard_canvas.mode == "manual_fit_white_start"

        dialog.setup_use_suggested_button.click()
        assert app.keyboard_canvas.mode == "manual_fit_group"
        assert app.keyboard_canvas.callbacks["overlays_visible_callback"]() is True
    finally:
        if controller.active_dialog is not None:
            controller.active_dialog.reject()
        app.close()
        app.deleteLater()
        _flush_qt_deletes()


def test_manual_fit_region_selection_updates_safe_overlay_preview():
    QApplication.instance() or QApplication([])
    app = _FakeApp()
    app.app_state.overlays = [
        _overlay(key_id=1, note="C", x=0, y=20, width=10, height=40),
        _overlay(key_id=2, note="C♯", x=12, y=10, width=6, height=20),
    ]
    controller = ManualKeyboardFitController(app)

    try:
        assert controller.open() is True

        app.keyboard_canvas.callbacks["region_selected_callback"]("white", 50, 100)
        app.keyboard_canvas.callbacks["region_selected_callback"]("black", 10, 40)

        white, black = app.app_state.overlays
        assert (white.x, white.y, white.width, white.height) == pytest.approx((1, 57.5, 8, 35))
        assert (black.x, black.y, black.width, black.height) == pytest.approx((12.6, 14.5, 4.8, 21))
    finally:
        if controller.active_dialog is not None:
            controller.active_dialog.reject()
        app.close()
        app.deleteLater()
        _flush_qt_deletes()


def test_manual_fit_dialog_resets_individual_parameter():
    QApplication.instance() or QApplication([])
    app = _FakeApp()
    app.app_state.overlays = [_overlay()]
    controller = ManualKeyboardFitController(app)

    try:
        assert controller.open() is True
        dialog = controller.active_dialog

        dialog.param_spinboxes["keyboard_top_delta"].setValue(12)
        assert app.app_state.overlays[0].y == pytest.approx(36.2)

        dialog.param_reset_buttons["keyboard_top_delta"].click()

        assert dialog.param_spinboxes["keyboard_top_delta"].value() == 0
        assert app.app_state.overlays[0].y == pytest.approx(26)
    finally:
        if controller.active_dialog is not None:
            controller.active_dialog.reject()
        app.close()
        app.deleteLater()
        _flush_qt_deletes()


def test_manual_fit_reset_position_preserves_other_controls():
    QApplication.instance() or QApplication([])
    app = _FakeApp()
    app.app_state.overlays = [_overlay()]
    controller = ManualKeyboardFitController(app)

    try:
        assert controller.open() is True
        dialog = controller.active_dialog

        dialog.param_spinboxes["keyboard_top_delta"].setValue(6)
        app.keyboard_canvas.callbacks["group_move_callback"](30, 8)
        assert app.app_state.overlays[0].x == pytest.approx(31)
        assert app.app_state.overlays[0].y == pytest.approx(39.1)

        dialog.reset_position_button.click()

        assert controller.session.params.group_dx == pytest.approx(0)
        assert controller.session.params.group_dy == pytest.approx(0)
        assert dialog.param_spinboxes["keyboard_top_delta"].value() == 6
        assert app.app_state.overlays[0].x == pytest.approx(1)
        assert app.app_state.overlays[0].y == pytest.approx(31.1)
    finally:
        if controller.active_dialog is not None:
            controller.active_dialog.reject()
        app.close()
        app.deleteLater()
        _flush_qt_deletes()


def test_manual_fit_octave_cancel_restores_previous_value():
    QApplication.instance() or QApplication([])
    app = _FakeApp()
    app.app_state.overlays = [_overlay()]
    app.app_state.midi.octave_transpose = 1
    app.app_state.unsaved_changes = False
    controller = ManualKeyboardFitController(app)

    try:
        assert controller.open() is True
        dialog = controller.active_dialog
        assert dialog.octave_spinbox.value() == 1

        dialog.octave_spinbox.setValue(2)
        assert app.app_state.midi.octave_transpose == 2

        dialog.reject()
        QApplication.processEvents()

        assert app.app_state.midi.octave_transpose == 1
        assert app.app_state.unsaved_changes is False
    finally:
        if controller.active_dialog is not None:
            controller.active_dialog.reject()
        app.close()
        app.deleteLater()
        _flush_qt_deletes()


def test_manual_fit_octave_apply_commits_value():
    QApplication.instance() or QApplication([])
    app = _FakeApp()
    app.app_state.overlays = [_overlay()]
    app.app_state.midi.octave_transpose = 1
    controller = ManualKeyboardFitController(app)

    try:
        assert controller.open() is True
        dialog = controller.active_dialog

        dialog.octave_spinbox.setValue(2)
        dialog.accept()
        QApplication.processEvents()

        assert app.app_state.midi.octave_transpose == 2
        assert app.app_state.unsaved_changes is True
        assert app.control_updates >= 1
    finally:
        if controller.active_dialog is not None:
            controller.active_dialog.reject()
        app.close()
        app.deleteLater()
        _flush_qt_deletes()


def test_manual_fit_slider_changes_preserve_group_drag_position():
    QApplication.instance() or QApplication([])
    app = _FakeApp()
    app.app_state.overlays = [
        _overlay(key_id=1, note="C", x=0, y=20, width=10, height=40),
        _overlay(key_id=2, note="D", x=24, y=20, width=10, height=40),
    ]
    controller = ManualKeyboardFitController(app)

    try:
        assert controller.open() is True
        dialog = controller.active_dialog

        app.keyboard_canvas.callbacks["group_move_callback"](30, 8)
        assert app.app_state.overlays[0].y == pytest.approx(34)

        dialog.param_spinboxes["keyboard_width_delta"].setValue(20)

        assert controller.session.params.group_dx == pytest.approx(30)
        assert controller.session.params.group_dy == pytest.approx(8)
        assert app.app_state.overlays[0].x > 0
        assert app.app_state.overlays[0].y == pytest.approx(34)
    finally:
        if controller.active_dialog is not None:
            controller.active_dialog.reject()
        app.close()
        app.deleteLater()
        _flush_qt_deletes()
