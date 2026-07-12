import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QDialog

import synthesia2midi.main as main_module
from synthesia2midi.gui.video_session_ui_controller import VideoSessionUiController
from synthesia2midi.main import Video2MidiApp


def test_video2midi_app_constructs_under_qt_offscreen(monkeypatch):
    monkeypatch.setattr(QTimer, "singleShot", lambda *args, **kwargs: None)
    qt_app = QApplication.instance() or QApplication([])
    window = Video2MidiApp()

    try:
        assert hasattr(window, "control_panel")
        assert hasattr(window, "keyboard_canvas")
        assert hasattr(window, "auto_detect_tuning_controller")
        assert (
            window.calibration_wizard_controller.auto_detect_tuning_controller
            is window.auto_detect_tuning_controller
        )
    finally:
        window.app_state.unsaved_changes = False
        window.close()
        window.deleteLater()
        qt_app.processEvents()


def _trigger_menu_action(window, text):
    for menu_action in window.menuBar().actions():
        menu = menu_action.menu()
        if menu is None:
            continue
        for action in menu.actions():
            if action.text() == text:
                action.trigger()
                return
    raise AssertionError(f"menu action not found: {text}")


class FakeStartupSignal:
    def __init__(self):
        self._slots = []

    def connect(self, slot):
        self._slots.append(slot)

    def emit(self):
        for slot in self._slots:
            slot()


class FakeStartupDialog:
    def __init__(self, parent, *, recent_video_paths=None):
        self.parent = parent
        self.recent_video_paths = list(recent_video_paths or [])
        self.open_local_file = FakeStartupSignal()
        self.open_recent_file = FakeStartupSignal()
        self.download_from_youtube = FakeStartupSignal()

    def exec(self):
        self.open_local_file.emit()
        self.download_from_youtube.emit()
        return QDialog.Accepted


def test_main_window_video_entrypoints_delegate_to_video_session_controller(monkeypatch):
    calls = []
    monkeypatch.setattr(QTimer, "singleShot", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        VideoSessionUiController,
        "open_video_file",
        lambda self: calls.append("open_video_file"),
    )
    monkeypatch.setattr(
        VideoSessionUiController,
        "show_youtube_download_dialog",
        lambda self: calls.append("show_youtube_download_dialog"),
    )
    monkeypatch.setattr(main_module, "StartupDialog", FakeStartupDialog)
    qt_app = QApplication.instance() or QApplication([])
    window = Video2MidiApp()

    try:
        _trigger_menu_action(window, "Open Video File...")
        _trigger_menu_action(window, "Download YouTube Video...")
        window._show_startup_dialog()

        assert calls == [
            "open_video_file",
            "show_youtube_download_dialog",
            "open_video_file",
            "show_youtube_download_dialog",
        ]
    finally:
        window.app_state.unsaved_changes = False
        window.close()
        window.deleteLater()
        qt_app.processEvents()
