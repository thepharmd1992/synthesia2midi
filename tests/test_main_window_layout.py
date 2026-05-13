from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QSplitter

from synthesia2midi.main import Video2MidiApp


def _make_app(monkeypatch):
    QApplication.instance() or QApplication([])
    monkeypatch.setattr(QTimer, "singleShot", lambda *args, **kwargs: None)
    app = Video2MidiApp()
    return app


def test_main_window_uses_splitter_with_video_priority(monkeypatch):
    app = _make_app(monkeypatch)
    try:
        assert isinstance(app.content_splitter, QSplitter)
        assert app.content_splitter.count() == 2
        assert app.control_panel.minimumWidth() <= 320
        assert app.control_panel.tab_widget.maximumWidth() <= 520
        assert app._settings_splitter_sizes[0] > app._settings_splitter_sizes[1]
    finally:
        app.close()


def test_focus_video_action_hides_and_restores_settings_panel(monkeypatch):
    app = _make_app(monkeypatch)
    try:
        app.focus_video_action.setChecked(True)
        app._toggle_focus_video_mode(True)

        assert app.control_panel.isHidden()
        assert app.focus_video_action.text() == "Show Settings Panel"

        app.focus_video_action.setChecked(False)
        app._toggle_focus_video_mode(False)

        assert not app.control_panel.isHidden()
        assert app.focus_video_action.text() == "Focus Video (Hide Settings)"
    finally:
        app.close()
