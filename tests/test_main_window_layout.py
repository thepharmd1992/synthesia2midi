from PySide6.QtCore import QRect, Qt, QTimer
from PySide6.QtWidgets import QApplication, QSplitter

from synthesia2midi.main import Video2MidiApp


def _make_app(monkeypatch):
    QApplication.instance() or QApplication([])
    monkeypatch.setattr(QTimer, "singleShot", lambda *args, **kwargs: None)
    app = Video2MidiApp()
    return app


def _rect_in_control_panel(control_panel, widget):
    top_left = widget.mapTo(control_panel, widget.rect().topLeft())
    return QRect(top_left, widget.rect().size())


def _assert_no_overlap(control_panel, widgets):
    rects = [_rect_in_control_panel(control_panel, widget) for widget in widgets]
    for index, rect in enumerate(rects):
        for other in rects[index + 1:]:
            assert not rect.intersects(other), f"{rect} overlaps {other}"


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
        assert app.focus_video_action.shortcutContext() == Qt.ApplicationShortcut
        assert app.focus_video_action in app.actions()
    finally:
        app.close()


def test_minimum_width_calibration_controls_do_not_overlap(monkeypatch):
    app = _make_app(monkeypatch)
    try:
        app.resize(1200, 828)
        app.show()
        app.content_splitter.setSizes([900, 300])
        QApplication.processEvents()

        control_panel = app.control_panel
        assert control_panel.width() == control_panel.minimumWidth()

        _assert_no_overlap(
            control_panel,
            [
                control_panel.calibration_wizard_button,
                control_panel.octave_transpose_spin,
                control_panel.calibrate_unlit_button,
                control_panel.unlit_status_label,
            ],
        )

        for key_type in ["LW", "LB", "RW", "RB"]:
            _assert_no_overlap(
                control_panel,
                [
                    control_panel.exemplar_buttons[key_type],
                    control_panel.exemplar_swatches[key_type],
                    control_panel.exemplar_presence_checkboxes[key_type],
                ],
            )
    finally:
        app.close()
