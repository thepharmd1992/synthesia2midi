import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from synthesia2midi.main import Video2MidiApp


def test_video2midi_app_constructs_under_qt_offscreen(monkeypatch):
    monkeypatch.setattr(QTimer, "singleShot", lambda *args, **kwargs: None)
    qt_app = QApplication.instance() or QApplication([])
    window = Video2MidiApp()

    try:
        assert hasattr(window, "control_panel")
        assert hasattr(window, "keyboard_canvas")
    finally:
        window.app_state.unsaved_changes = False
        window.close()
        window.deleteLater()
        qt_app.processEvents()
