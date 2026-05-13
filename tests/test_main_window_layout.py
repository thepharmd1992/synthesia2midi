from PySide6.QtCore import QSignalBlocker, QRect, Qt, QTimer
from PySide6.QtWidgets import QApplication, QSplitter

from synthesia2midi.main import Video2MidiApp

UNBOUNDED_WIDGET_SIZE = 16777215


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


def test_main_window_uses_splitter_with_settings_priority(monkeypatch):
    app = _make_app(monkeypatch)
    try:
        screen_rect = QApplication.primaryScreen().availableGeometry()
        max_width = screen_rect.width() - 20
        max_height = screen_rect.height() - 40

        assert isinstance(app.content_splitter, QSplitter)
        assert app.content_splitter.count() == 2
        assert app.width() == max_width
        assert app.height() == max_height
        assert app.windowState() & Qt.WindowMaximized
        assert app.control_panel.minimumWidth() <= 320
        assert app.control_panel.maximumWidth() >= 700
        assert app.control_panel.tab_widget.maximumWidth() >= 700
        assert app.control_panel.tab_widget.maximumHeight() == UNBOUNDED_WIDGET_SIZE
        assert app.control_panel.tab_widget.height() >= app.control_panel.height() - 180
        assert app._settings_splitter_sizes[1] >= min(700, max_width - 320)
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


def test_restore_detection_defaults_resets_parameter_sliders_not_toggles(monkeypatch):
    app = _make_app(monkeypatch)
    try:
        app.show()
        control_panel = app.control_panel
        control_panel.tab_widget.setCurrentIndex(2)
        QApplication.processEvents()

        assert control_panel.histogram_detection_cb.isChecked() is False
        assert control_panel.delta_detection_cb.isChecked() is False
        assert control_panel.black_key_filter_cb.isChecked() is False
        assert hasattr(control_panel, "restore_detection_defaults_button")

        toggle_blockers = [
            QSignalBlocker(control_panel.histogram_detection_cb),
            QSignalBlocker(control_panel.delta_detection_cb),
            QSignalBlocker(control_panel.black_key_filter_cb),
        ]
        control_panel.histogram_detection_cb.setChecked(True)
        control_panel.delta_detection_cb.setChecked(False)
        control_panel.black_key_filter_cb.setChecked(True)
        del toggle_blockers

        control_panel.detection_threshold_slider.setValue(12)
        control_panel.histogram_threshold_slider.setValue(44)
        control_panel.rise_delta_slider.setValue(31)
        control_panel.fall_delta_slider.setValue(22)
        control_panel.similarity_ratio_slider.setValue(91)

        control_panel.restore_detection_defaults_button.click()

        assert control_panel.detection_threshold_slider.value() == 50
        assert control_panel.detection_threshold_label.text() == "50%"
        assert control_panel.histogram_threshold_slider.value() == 80
        assert control_panel.histogram_threshold_label.text() == "0.80"
        assert control_panel.rise_delta_slider.value() == 15
        assert control_panel.rise_delta_label.text() == "0.15"
        assert control_panel.fall_delta_slider.value() == 5
        assert control_panel.fall_delta_label.text() == "0.05"
        assert control_panel.similarity_ratio_slider.value() == 60
        assert control_panel.similarity_ratio_label.text() == "0.60"
        assert control_panel.histogram_detection_cb.isChecked() is True
        assert control_panel.delta_detection_cb.isChecked() is False
        assert control_panel.black_key_filter_cb.isChecked() is True
    finally:
        app.app_state.unsaved_changes = False
        app.close()


def test_detection_mode_sliders_share_left_edge(monkeypatch):
    app = _make_app(monkeypatch)
    try:
        app.show()
        app.control_panel.tab_widget.setCurrentIndex(2)
        QApplication.processEvents()

        control_panel = app.control_panel
        slider_rects = [
            _rect_in_control_panel(control_panel, app.control_panel.histogram_threshold_slider),
            _rect_in_control_panel(control_panel, app.control_panel.rise_delta_slider),
            _rect_in_control_panel(control_panel, app.control_panel.fall_delta_slider),
            _rect_in_control_panel(control_panel, app.control_panel.similarity_ratio_slider),
        ]
        left_edges = {rect.left() for rect in slider_rects}
        widths = {rect.width() for rect in slider_rects}

        assert len(left_edges) == 1
        assert widths == {150}
    finally:
        app.close()


def test_spark_auto_calibration_controls_stack_vertically(monkeypatch):
    app = _make_app(monkeypatch)
    try:
        app.show()
        app.control_panel.tab_widget.setCurrentIndex(3)
        QApplication.processEvents()

        control_panel = app.control_panel
        button_rects = {
            key_type: _rect_in_control_panel(control_panel, button)
            for key_type, button in control_panel.auto_calib_buttons.items()
        }

        assert button_rects["LW"].y() < button_rects["LB"].y() < button_rects["RW"].y() < button_rects["RB"].y()
        for key_type in ["LW", "LB", "RW", "RB"]:
            status_rect = _rect_in_control_panel(control_panel, control_panel.auto_calib_status_labels[key_type])
            assert button_rects[key_type].right() < status_rect.left()
            assert status_rect.right() <= control_panel.width()
    finally:
        app.close()
