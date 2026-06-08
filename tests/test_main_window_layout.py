from PySide6.QtCore import QSignalBlocker, QRect, Qt, QTimer
from PySide6.QtWidgets import QApplication, QGroupBox, QScrollArea, QTabWidget

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


def _has_ancestor(widget, ancestor):
    parent = widget.parentWidget()
    while parent is not None:
        if parent is ancestor:
            return True
        parent = parent.parentWidget()
    return False


def test_main_window_prioritizes_video_with_settings_rail_and_tool_window(monkeypatch):
    app = _make_app(monkeypatch)
    try:
        screen_rect = QApplication.primaryScreen().availableGeometry()
        max_width = screen_rect.width() - 20
        max_height = screen_rect.height() - 40

        assert not hasattr(app, "content_splitter")
        assert app.width() == max_width
        assert app.height() == max_height
        assert app.windowState() & Qt.WindowMaximized
        assert not app.settings_rail_button.isHidden()
        assert app.settings_rail_button.width() <= 72
        assert "#0d65ca" in app.settings_rail_button.styleSheet()
        assert app.settings_tool_window.windowFlags() & Qt.Tool
        assert not app.settings_tool_window.isVisible()
        assert isinstance(app.settings_scroll_area, QScrollArea)
        assert app.settings_scroll_area.widget() is app.control_panel
        assert app.settings_scroll_area.widgetResizable()
        assert app.control_panel.minimumWidth() <= 320
        assert app.control_panel.maximumWidth() >= 700
        assert app.control_panel.tab_widget.maximumWidth() >= 700
        assert app.control_panel.tab_widget.maximumHeight() == UNBOUNDED_WIDGET_SIZE
        assert not isinstance(app.control_panel.tab_widget, QTabWidget)
        assert app.control_panel.settings_section_rail.width() <= 125
        assert [
            app.control_panel.settings_section_rail.item(index).text()
            for index in range(app.control_panel.settings_section_rail.count())
        ] == ["Calibration", "Overlays", "Detection", "Spark", "MIDI", "Trim", "Optional"]
    finally:
        app.close()


def test_settings_rail_opens_floating_tool_window(monkeypatch):
    app = _make_app(monkeypatch)
    try:
        app.show()
        QApplication.processEvents()

        assert not app.settings_tool_window.isVisible()

        app.settings_rail_button.click()
        QApplication.processEvents()

        assert app.settings_tool_window.isVisible()
        assert app.settings_tool_window.windowFlags() & Qt.Tool
        assert app.settings_rail_button.isVisible()
    finally:
        app.close()


def test_settings_lower_rail_holds_global_actions_and_status(monkeypatch):
    app = _make_app(monkeypatch)
    try:
        app.show()
        app.settings_rail_button.click()
        QApplication.processEvents()

        control_panel = app.control_panel
        action_widgets = [
            control_panel.convert_button,
            control_panel.conversion_status,
            control_panel.midi_touchup_button,
            control_panel.selected_overlay_label,
        ]

        assert hasattr(control_panel, "settings_rail_actions")
        assert all(_has_ancestor(widget, control_panel.settings_rail_actions) for widget in action_widgets)
        assert all(group.title() != "Main Actions" for group in control_panel.findChildren(QGroupBox))

        rail_rect = _rect_in_control_panel(control_panel, control_panel.settings_section_rail)
        actions_rect = _rect_in_control_panel(control_panel, control_panel.settings_rail_actions)
        stack_rect = _rect_in_control_panel(control_panel, control_panel.tab_widget)

        assert actions_rect.top() > rail_rect.bottom()
        assert actions_rect.left() < stack_rect.left()
        assert actions_rect.width() <= control_panel.settings_section_rail.width()
    finally:
        app.close()


def test_settings_rail_preserves_open_tool_window_position(monkeypatch):
    app = _make_app(monkeypatch)
    try:
        app.show()
        app.settings_rail_button.click()
        QApplication.processEvents()
        custom_pos = app.settings_tool_window.pos()
        custom_pos.setX(custom_pos.x() + 80)
        custom_pos.setY(custom_pos.y() + 40)
        app.settings_tool_window.move(custom_pos)

        app.settings_rail_button.click()
        QApplication.processEvents()

        assert app.settings_tool_window.pos() == custom_pos
    finally:
        app.close()


def test_focus_video_action_hides_and_restores_settings_rail(monkeypatch):
    app = _make_app(monkeypatch)
    try:
        app.show()
        app.settings_rail_button.click()
        QApplication.processEvents()
        assert app.settings_tool_window.isVisible()

        app.focus_video_action.setChecked(True)
        app._toggle_focus_video_mode(True)

        assert app.settings_rail_button.isHidden()
        assert not app.settings_tool_window.isVisible()
        assert app.focus_video_action.text() == "Show Settings Panel"

        app.focus_video_action.setChecked(False)
        app._toggle_focus_video_mode(False)

        assert not app.settings_rail_button.isHidden()
        assert app.settings_tool_window.isVisible()
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
        QApplication.processEvents()

        control_panel = app.control_panel

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
        app.settings_rail_button.click()
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
        assert widths == {110}

        for slider, label in [
            (control_panel.histogram_threshold_slider, control_panel.histogram_threshold_label),
            (control_panel.rise_delta_slider, control_panel.rise_delta_label),
            (control_panel.fall_delta_slider, control_panel.fall_delta_label),
            (control_panel.similarity_ratio_slider, control_panel.similarity_ratio_label),
        ]:
            slider_rect = _rect_in_control_panel(control_panel, slider)
            label_rect = _rect_in_control_panel(control_panel, label)
            assert slider_rect.right() < label_rect.left()
    finally:
        app.close()


def test_overlay_size_controls_stack_for_narrow_settings_window(monkeypatch):
    app = _make_app(monkeypatch)
    try:
        app.show()
        app.settings_rail_button.click()
        app.control_panel.tab_widget.setCurrentIndex(1)
        QApplication.processEvents()

        control_panel = app.control_panel
        white_height_label_rect = _rect_in_control_panel(control_panel, control_panel.white_height_label)
        white_height_dec_rect = _rect_in_control_panel(control_panel, control_panel.white_height_dec_button)
        white_width_label_rect = _rect_in_control_panel(control_panel, control_panel.white_width_label)
        white_width_dec_rect = _rect_in_control_panel(control_panel, control_panel.white_width_dec_button)

        assert white_height_dec_rect.top() > white_height_label_rect.bottom()
        assert white_width_dec_rect.top() > white_width_label_rect.bottom()
    finally:
        app.close()


def test_spark_roi_controls_stack_and_stay_inside_panel(monkeypatch):
    app = _make_app(monkeypatch)
    try:
        app.show()
        app.settings_rail_button.click()
        app.control_panel.tab_widget.setCurrentIndex(3)
        QApplication.processEvents()

        control_panel = app.control_panel
        select_rect = _rect_in_control_panel(control_panel, control_panel.spark_roi_select_button)
        toggle_rect = _rect_in_control_panel(control_panel, control_panel.spark_roi_toggle_button)

        assert toggle_rect.top() > select_rect.bottom()
        assert toggle_rect.right() <= control_panel.width()
    finally:
        app.close()


def test_spark_auto_calibration_controls_stack_vertically(monkeypatch):
    app = _make_app(monkeypatch)
    try:
        app.show()
        app.settings_rail_button.click()
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
            assert status_rect.top() > button_rects[key_type].bottom()
            assert status_rect.right() <= control_panel.width()
    finally:
        app.close()
