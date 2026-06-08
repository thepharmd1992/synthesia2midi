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
        self.show_overlays_action = SimpleNamespace(checked=None, setChecked=lambda value: setattr(self.show_overlays_action, "checked", value))
        self.control_panel = SimpleNamespace(
            update_controls_from_state=lambda: None,
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


def test_manual_fit_dialog_reset_and_clear_selected_override_update_preview():
    QApplication.instance() or QApplication([])
    app = _FakeApp()
    app.app_state.overlays = [_overlay()]
    app.app_state.ui.selected_overlay_id = 1
    controller = ManualKeyboardFitController(app)
    try:
        assert controller.open() is True
        dialog = controller.active_dialog

        dialog.param_spinboxes["white_height_delta"].setValue(6)
        assert app.app_state.overlays[0].height == 46

        app.keyboard_canvas.callbacks["single_move_callback"](0, 100, 50)
        assert controller.session.overridden_key_ids() == {1}

        dialog.clear_selected_override_button.click()
        assert controller.session.overridden_key_ids() == set()

        dialog.reset_all_button.click()
        assert dialog.param_spinboxes["white_height_delta"].value() == 0
        assert app.app_state.overlays[0].height == 40

        dialog.reject()
        QApplication.processEvents()
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
        assert app.app_state.overlays[0].y == pytest.approx(28)

        dialog.param_spinboxes["keyboard_width_delta"].setValue(20)

        assert controller.session.params.group_dx == pytest.approx(30)
        assert controller.session.params.group_dy == pytest.approx(8)
        assert app.app_state.overlays[0].x > 0
        assert app.app_state.overlays[0].y == pytest.approx(28)
    finally:
        if controller.active_dialog is not None:
            controller.active_dialog.reject()
        app.close()
        app.deleteLater()
        _flush_qt_deletes()
