from types import SimpleNamespace

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtWidgets import QApplication, QMessageBox, QWidget

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
        assert dialog.set_black_region_button.text() == "Set Black Region"
        assert dialog.set_white_region_button.text() == "Set White Region"
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


def test_manual_fit_region_buttons_enter_canvas_region_modes():
    QApplication.instance() or QApplication([])
    app = _FakeApp()
    app.app_state.overlays = [_overlay()]
    controller = ManualKeyboardFitController(app)

    try:
        assert controller.open() is True
        dialog = controller.active_dialog

        dialog.set_black_region_button.click()
        assert app.keyboard_canvas.mode == "manual_fit_black_region"

        dialog.set_white_region_button.click()
        assert app.keyboard_canvas.mode == "manual_fit_white_region"
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
