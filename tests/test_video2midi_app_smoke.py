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

    def emit(self, *args):
        for slot in self._slots:
            slot(*args)


class FakeStartupDialog:
    action = "reject"
    instances = []

    def __init__(self, parent, *, recent_video_paths=None):
        self.parent = parent
        self.recent_video_paths = list(recent_video_paths or [])
        self.open_local_file = FakeStartupSignal()
        self.open_recent_file = FakeStartupSignal()
        self.download_from_youtube = FakeStartupSignal()
        self.accepted = False
        FakeStartupDialog.instances.append(self)

    def accept(self):
        self.accepted = True

    def reject(self):
        self.accepted = False

    def exec(self):
        if self.action == "local":
            self.open_local_file.emit()
        elif self.action == "youtube":
            self.download_from_youtube.emit()
        elif isinstance(self.action, tuple) and self.action[0] == "recent":
            self.open_recent_file.emit(self.action[1])
        return QDialog.Accepted if self.accepted else QDialog.Rejected


class FakeLoadedSession:
    def release(self):
        pass


def test_main_window_video_entrypoints_delegate_to_video_session_controller(monkeypatch):
    calls = []
    monkeypatch.setattr(QTimer, "singleShot", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        VideoSessionUiController,
        "open_video_file",
        lambda self, parent=None: calls.append(("open_video_file", parent)) or False,
    )
    monkeypatch.setattr(
        VideoSessionUiController,
        "show_youtube_download_dialog",
        lambda self, parent=None: calls.append(
            ("show_youtube_download_dialog", parent)
        )
        or False,
    )
    monkeypatch.setattr(main_module, "StartupDialog", FakeStartupDialog)
    monkeypatch.setattr(main_module.QApplication, "quit", lambda *args: None)
    FakeStartupDialog.instances.clear()
    FakeStartupDialog.action = "local"
    qt_app = QApplication.instance() or QApplication([])
    window = Video2MidiApp()

    try:
        _trigger_menu_action(window, "Open Video File...")
        _trigger_menu_action(window, "Download YouTube Video...")
        window._show_startup_dialog()

        assert calls == [
            ("open_video_file", None),
            ("show_youtube_download_dialog", None),
            ("open_video_file", FakeStartupDialog.instances[-1]),
        ]
    finally:
        window.app_state.unsaved_changes = False
        window.close()
        window.deleteLater()
        qt_app.processEvents()


def test_begin_startup_schedules_selector_without_showing_main(monkeypatch):
    scheduled = []
    monkeypatch.setattr(
        QTimer,
        "singleShot",
        lambda delay, callback: scheduled.append((delay, callback)),
    )
    qt_app = QApplication.instance() or QApplication([])
    window = Video2MidiApp()

    try:
        assert scheduled == []
        assert not window.isVisible()

        window.begin_startup()

        assert len(scheduled) == 1
        assert scheduled[0][0] == 0
        assert scheduled[0][1] == window._show_startup_dialog
    finally:
        window.app_state.unsaved_changes = False
        window.close()
        window.deleteLater()
        qt_app.processEvents()


def test_startup_local_success_accepts_selector_and_shows_main(monkeypatch):
    monkeypatch.setattr(QTimer, "singleShot", lambda *args: None)
    monkeypatch.setattr(main_module, "StartupDialog", FakeStartupDialog)
    FakeStartupDialog.instances.clear()
    FakeStartupDialog.action = "local"
    quit_calls = []
    monkeypatch.setattr(
        main_module.QApplication,
        "quit",
        lambda *args: quit_calls.append("quit"),
    )

    def load_local(self, parent=None):
        assert parent is FakeStartupDialog.instances[-1]
        self.app.video_session = FakeLoadedSession()
        return True

    monkeypatch.setattr(VideoSessionUiController, "open_video_file", load_local)
    qt_app = QApplication.instance() or QApplication([])
    window = Video2MidiApp()

    try:
        window._show_startup_dialog()

        assert FakeStartupDialog.instances[-1].accepted
        assert window.isVisible()
        assert quit_calls == []
    finally:
        window.app_state.unsaved_changes = False
        window.close()
        window.deleteLater()
        qt_app.processEvents()


def test_startup_youtube_success_accepts_selector_and_shows_main(monkeypatch):
    monkeypatch.setattr(QTimer, "singleShot", lambda *args: None)
    monkeypatch.setattr(main_module, "StartupDialog", FakeStartupDialog)
    FakeStartupDialog.instances.clear()
    FakeStartupDialog.action = "youtube"

    def load_youtube(self, parent=None):
        assert parent is FakeStartupDialog.instances[-1]
        self.app.video_session = FakeLoadedSession()
        return True

    monkeypatch.setattr(
        VideoSessionUiController,
        "show_youtube_download_dialog",
        load_youtube,
    )
    qt_app = QApplication.instance() or QApplication([])
    window = Video2MidiApp()

    try:
        window._show_startup_dialog()

        assert FakeStartupDialog.instances[-1].accepted
        assert window.isVisible()
    finally:
        window.app_state.unsaved_changes = False
        window.close()
        window.deleteLater()
        qt_app.processEvents()


def test_startup_recent_success_accepts_selector_and_shows_main(monkeypatch):
    monkeypatch.setattr(QTimer, "singleShot", lambda *args: None)
    monkeypatch.setattr(main_module, "StartupDialog", FakeStartupDialog)
    FakeStartupDialog.instances.clear()
    FakeStartupDialog.action = ("recent", "/tmp/recent.mp4")

    def load_recent(self, filepath):
        assert filepath == "/tmp/recent.mp4"
        self.app.video_session = FakeLoadedSession()
        return True

    monkeypatch.setattr(
        VideoSessionUiController,
        "open_recent_video_file",
        load_recent,
    )
    qt_app = QApplication.instance() or QApplication([])
    window = Video2MidiApp()

    try:
        window._show_startup_dialog()

        assert FakeStartupDialog.instances[-1].accepted
        assert window.isVisible()
    finally:
        window.app_state.unsaved_changes = False
        window.close()
        window.deleteLater()
        qt_app.processEvents()


def test_startup_secondary_cancel_keeps_main_hidden_until_selector_rejects(monkeypatch):
    monkeypatch.setattr(QTimer, "singleShot", lambda *args: None)
    monkeypatch.setattr(main_module, "StartupDialog", FakeStartupDialog)
    FakeStartupDialog.instances.clear()
    FakeStartupDialog.action = "local"
    quit_calls = []
    monkeypatch.setattr(
        main_module.QApplication,
        "quit",
        lambda *args: quit_calls.append("quit"),
    )
    monkeypatch.setattr(
        VideoSessionUiController,
        "open_video_file",
        lambda self, parent=None: False,
    )
    qt_app = QApplication.instance() or QApplication([])
    window = Video2MidiApp()

    try:
        window._show_startup_dialog()

        assert not FakeStartupDialog.instances[-1].accepted
        assert not window.isVisible()
        assert quit_calls == ["quit"]
    finally:
        window.app_state.unsaved_changes = False
        window.close()
        window.deleteLater()
        qt_app.processEvents()
