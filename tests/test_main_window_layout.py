from types import SimpleNamespace

from PySide6.QtCore import QSignalBlocker, QRect, Qt, QTimer
from PySide6.QtWidgets import QApplication, QGroupBox, QScrollArea, QTabWidget, QToolButton

from synthesia2midi.app_config import OverlayConfig
from synthesia2midi.core.app_state import AppState
from synthesia2midi.gui.controls_qt import ControlPanelQt
from synthesia2midi.gui.main_action_controller import MainActionController
from synthesia2midi.main import Video2MidiApp
from synthesia2midi.workflows.overlay_manager import OverlayAdjustmentResult, OverlayManager

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


def _settings_section_labels(control_panel):
    return [
        control_panel.settings_section_rail.item(index).text()
        for index in range(control_panel.settings_section_rail.count())
    ]


def _show_settings_section(control_panel, label: str) -> None:
    labels = _settings_section_labels(control_panel)
    control_panel.tab_widget.setCurrentIndex(labels.index(label))


def _show_advanced_section(control_panel, key: str) -> None:
    _show_settings_section(control_panel, "Advanced")
    control_panel.advanced_sections[key]._toggle.setChecked(True)
    QApplication.processEvents()


def _make_overlay(*, key_id: int, note_name: str, x: float, width: float, y: float = 2.0, rotation: float = 0.0) -> OverlayConfig:
    return OverlayConfig(
        key_id=key_id,
        note_octave=4,
        note_name_in_octave=note_name,
        x=x,
        y=y,
        width=width,
        height=8,
        key_type="LW",
        rotation_degrees=rotation,
    )


def _seed_quick_adjust_overlays(app_state) -> None:
    app_state.overlays = [
        _make_overlay(key_id=1, note_name="C", x=2, width=6, rotation=0.0),
        _make_overlay(key_id=2, note_name="C♯", x=10, width=4, rotation=0.0),
        _make_overlay(key_id=3, note_name="E", x=18, width=6, rotation=0.0),
        _make_overlay(key_id=4, note_name="F♯", x=26, width=4, rotation=0.0),
    ]


def _make_overlay_adjustment_panel():
    QApplication.instance() or QApplication([])
    app_state = AppState()
    panel = ControlPanelQt(app_state=app_state)
    overlay_manager = OverlayManager(app_state)
    controller = MainActionController(
        SimpleNamespace(overlay_manager=overlay_manager, control_panel=panel)
    )
    panel._test_overlay_manager = overlay_manager
    panel._test_main_action_controller = controller
    panel.overlay_size_adjustment_requested.connect(controller.handle_overlay_size_adjustment)
    return panel, app_state


def _assert_quick_adjust_control(
    control_panel,
    emitted,
    *,
    label_attr: str,
    value_label_attr: str,
    reset_button_attr: str,
    increment_button_attr: str,
    expected_label: str,
    key_color: str,
    dimension: str,
    delta: int,
) -> None:
    assert getattr(control_panel, label_attr).text() == expected_label
    assert getattr(control_panel, value_label_attr).text() == "0"
    assert getattr(control_panel, reset_button_attr).text() == "Reset"
    assert getattr(control_panel, reset_button_attr).maximumWidth() == UNBOUNDED_WIDGET_SIZE

    getattr(control_panel, increment_button_attr).click()

    assert getattr(control_panel, value_label_attr).text() == str(delta)
    assert emitted == [(key_color, dimension, delta)]

    getattr(control_panel, reset_button_attr).click()

    assert getattr(control_panel, value_label_attr).text() == "0"
    assert emitted == [
        (key_color, dimension, delta),
        (key_color, dimension, -delta),
    ]
    emitted.clear()


def test_main_window_prioritizes_video_with_settings_gear_and_tool_window(monkeypatch):
    app = _make_app(monkeypatch)
    try:
        screen_rect = QApplication.primaryScreen().availableGeometry()
        max_width = screen_rect.width() - 20
        max_height = screen_rect.height() - 40

        assert not hasattr(app, "content_splitter")
        assert app.width() == max_width
        assert app.height() == max_height
        assert app.windowState() & Qt.WindowMaximized
        assert not hasattr(app, "settings_rail_button")
        assert isinstance(app.settings_toggle_button, QToolButton)
        assert not app.settings_toggle_button.isHidden()
        assert app.settings_toggle_button.text() == "\u2699"
        assert app.settings_toggle_button.toolTip() == "Show settings"
        assert app.settings_toggle_button.width() == 40
        assert app.settings_toggle_button.height() == 40
        assert app.settings_toggle_button.isCheckable()
        assert not app.settings_toggle_button.isChecked()
        assert not app.video_empty_state.isHidden()
        assert app.video_empty_state.open_video_button.text() == "Open Video"
        assert app.video_empty_state.youtube_button.text() == "Download from YouTube"
        assert app.video_empty_state.settings_button.text() == "Settings"
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
        assert app.control_panel.settings_section_rail.width() >= 98
        assert _settings_section_labels(app.control_panel) == [
            "Guide",
            "Calibration",
            "Overlays",
            "Detection",
            "MIDI",
            "Advanced",
            "Optional",
            "Language",
        ]
    finally:
        app.close()


def test_settings_gear_toggles_floating_tool_window(monkeypatch):
    app = _make_app(monkeypatch)
    try:
        app.show()
        QApplication.processEvents()

        assert not app.settings_tool_window.isVisible()

        app.settings_toggle_button.click()
        QApplication.processEvents()

        assert app.settings_tool_window.isVisible()
        assert app.settings_tool_window.windowFlags() & Qt.Tool
        assert app.settings_toggle_button.isVisible()
        assert app.settings_toggle_button.isChecked()
        assert app.settings_toggle_button.toolTip() == "Hide settings"

        screen_rect = QApplication.primaryScreen().availableGeometry()
        settings_rect = app.settings_tool_window.frameGeometry()
        assert settings_rect.right() >= screen_rect.right() - 80
        assert settings_rect.top() <= screen_rect.top() + 80
        assert app.settings_tool_window.height() <= 620

        app.settings_toggle_button.click()
        QApplication.processEvents()

        assert not app.settings_tool_window.isVisible()
        assert not app.settings_toggle_button.isChecked()
        assert app.settings_toggle_button.toolTip() == "Show settings"
    finally:
        app.close()


def test_settings_lower_rail_holds_global_actions_and_status(monkeypatch):
    app = _make_app(monkeypatch)
    try:
        app.show()
        app.settings_toggle_button.click()
        QApplication.processEvents()

        control_panel = app.control_panel
        action_widgets = [
            control_panel.convert_button,
            control_panel.conversion_status,
            control_panel.midi_touchup_button,
            control_panel.selected_overlay_caption,
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

        caption_rect = _rect_in_control_panel(control_panel, control_panel.selected_overlay_caption)
        value_rect = _rect_in_control_panel(control_panel, control_panel.selected_overlay_label)
        assert abs(caption_rect.center().y() - value_rect.center().y()) <= 2

        control_panel.update_selected_overlay(None)
        assert control_panel.selected_overlay_label.text() == "None"
        control_panel.update_selected_overlay(23)
        assert control_panel.selected_overlay_label.text() == "23"
    finally:
        app.close()


def test_overlays_tab_exposes_manual_fit_entry_point(monkeypatch):
    app = _make_app(monkeypatch)
    try:
        emitted = []
        try:
            app.control_panel.manual_fit_requested.disconnect()
        except (TypeError, RuntimeError):
            pass
        app.control_panel.manual_fit_requested.connect(lambda: emitted.append(True))

        assert app.control_panel.manual_fit_button.text() == "Manual Fit"

        app.control_panel.manual_fit_button.click()

        assert emitted == [True]
    finally:
        app.close()


def test_overlays_tab_exposes_left_and_right_slant_controls(monkeypatch):
    app = _make_app(monkeypatch)
    try:
        emitted = []
        try:
            app.control_panel.overlay_size_adjustment_requested.disconnect()
        except (TypeError, RuntimeError):
            pass
        app.control_panel.overlay_size_adjustment_requested.connect(
            lambda key_color, dimension, delta: emitted.append((key_color, dimension, delta))
        )

        assert app.control_panel.left_slant_label.text() == "Left Slant"
        assert app.control_panel.right_slant_label.text() == "Right Slant"
        assert app.control_panel.left_slant_value_label.text() == "0"
        assert app.control_panel.right_slant_value_label.text() == "0"
        assert app.control_panel.left_slant_reset_button.text() == "Reset"
        assert app.control_panel.right_slant_reset_button.text() == "Reset"

        app.control_panel.left_slant_inc_button.click()
        app.control_panel.right_slant_dec_button.click()

        assert app.control_panel.left_slant_value_label.text() == "1"
        assert app.control_panel.right_slant_value_label.text() == "-1"

        app.control_panel.left_slant_reset_button.click()

        assert app.control_panel.left_slant_value_label.text() == "0"
        assert emitted == [
            ("all", "left_slant", 1),
            ("all", "right_slant", -1),
            ("all", "left_slant", -1),
        ]
    finally:
        app.close()


def test_overlays_tab_exposes_white_and_black_quick_adjust_controls(monkeypatch):
    panel, app_state = _make_overlay_adjustment_panel()
    try:
        emitted = []
        _seed_quick_adjust_overlays(app_state)
        panel.update_controls_from_state()
        panel.overlay_size_adjustment_requested.connect(
            lambda key_color, dimension, delta: emitted.append((key_color, dimension, delta))
        )

        _assert_quick_adjust_control(
            panel,
            emitted,
            label_attr="white_width_label",
            value_label_attr="white_width_value_label",
            reset_button_attr="white_width_reset_button",
            increment_button_attr="white_width_inc_button",
            expected_label="White Key Width",
            key_color="white",
            dimension="width",
            delta=2,
        )
        _assert_quick_adjust_control(
            panel,
            emitted,
            label_attr="black_height_label",
            value_label_attr="black_height_value_label",
            reset_button_attr="black_height_reset_button",
            increment_button_attr="black_height_inc_button",
            expected_label="Black Key Height",
            key_color="black",
            dimension="height",
            delta=2,
        )
    finally:
        panel.close()
        panel.deleteLater()


def test_overlay_quick_adjust_values_reset_when_overlay_baseline_changes():
    panel, app_state = _make_overlay_adjustment_panel()
    try:
        _seed_quick_adjust_overlays(app_state)
        panel.update_controls_from_state()

        panel.white_width_inc_button.click()

        assert panel.white_width_value_label.text() == "2"

        for overlay in app_state.overlays:
            overlay.x += 10

        panel.update_controls_from_state()

        assert panel.white_width_value_label.text() == "0"
    finally:
        panel.close()
        panel.deleteLater()


def test_overlay_quick_adjust_values_reset_when_overlays_are_cleared():
    panel, app_state = _make_overlay_adjustment_panel()
    try:
        _seed_quick_adjust_overlays(app_state)
        panel.update_controls_from_state()

        panel.white_width_inc_button.click()

        assert panel.white_width_value_label.text() == "2"

        app_state.overlays.clear()
        panel.update_controls_from_state()

        assert panel.white_width_value_label.text() == "0"
    finally:
        panel.close()
        panel.deleteLater()


def test_white_width_quick_adjust_becomes_indeterminate_when_backend_partially_applies():
    panel, app_state = _make_overlay_adjustment_panel()
    try:
        emitted = []
        app_state.overlays = [
            _make_overlay(key_id=1, note_name="C", x=10, width=6),
            _make_overlay(key_id=2, note_name="E", x=20, width=2),
            _make_overlay(key_id=3, note_name="F♯", x=30, width=4),
        ]
        panel.update_controls_from_state()
        panel.overlay_size_adjustment_requested.connect(
            lambda key_color, dimension, delta: emitted.append((key_color, dimension, delta))
        )

        first_white = app_state.overlays[0]
        second_white = app_state.overlays[1]
        first_before = (first_white.x, first_white.width)
        second_before = (second_white.x, second_white.width)

        panel.white_width_dec_button.click()

        assert emitted == [("white", "width", -2)]
        assert (first_white.x, first_white.width) == (first_before[0] + 1, first_before[1] - 2)
        assert (second_white.x, second_white.width) == second_before
        assert panel.white_width_value_label.text() == "--"

        emitted.clear()
        panel.white_width_reset_button.click()

        assert emitted == []
        assert panel.white_width_value_label.text() == "--"
    finally:
        panel.close()
        panel.deleteLater()


def test_right_slant_quick_adjust_becomes_indeterminate_when_backend_clamps_mixed_overlays():
    panel, app_state = _make_overlay_adjustment_panel()
    try:
        emitted = []
        app_state.overlays = [
            _make_overlay(key_id=1, note_name="C", x=0, width=4, rotation=0.0),
            _make_overlay(key_id=2, note_name="E", x=12, width=4, rotation=0.0),
            _make_overlay(key_id=3, note_name="G", x=20, width=4, rotation=44.5),
        ]
        panel.update_controls_from_state()
        panel.overlay_size_adjustment_requested.connect(
            lambda key_color, dimension, delta: emitted.append((key_color, dimension, delta))
        )

        middle_overlay = app_state.overlays[1]
        right_overlay = app_state.overlays[2]
        middle_before = middle_overlay.rotation_degrees
        right_before = right_overlay.rotation_degrees

        panel.right_slant_inc_button.click()

        assert emitted == [("all", "right_slant", 1)]
        assert middle_overlay.rotation_degrees > middle_before
        assert right_overlay.rotation_degrees == 45.0
        assert right_overlay.rotation_degrees > right_before
        assert panel.right_slant_value_label.text() == "--"

        emitted.clear()
        panel.right_slant_reset_button.click()

        assert emitted == []
        assert panel.right_slant_value_label.text() == "--"
    finally:
        panel.close()
        panel.deleteLater()


def test_overlay_manager_reports_mixed_results_for_partial_and_clamped_adjustments():
    app_state = AppState()
    app_state.overlays = [
        _make_overlay(key_id=1, note_name="C", x=10, width=6, rotation=0.0),
        _make_overlay(key_id=2, note_name="E", x=20, width=2, rotation=0.0),
        _make_overlay(key_id=3, note_name="G", x=30, width=4, rotation=44.5),
    ]
    overlay_manager = OverlayManager(app_state)

    width_result = overlay_manager.adjust_overlay_sizes("white", "width", -2)
    slant_result = overlay_manager.adjust_overlay_sizes("all", "right_slant", 1)

    assert width_result == OverlayAdjustmentResult(
        key_color="white",
        dimension="width",
        delta=-2,
        status="mixed",
    )
    assert slant_result == OverlayAdjustmentResult(
        key_color="all",
        dimension="right_slant",
        delta=1,
        status="mixed",
    )


def test_main_action_controller_passes_overlay_adjustment_result_back_to_control_panel():
    app_state = AppState()
    app_state.overlays = [_make_overlay(key_id=1, note_name="C", x=10, width=6)]
    overlay_manager = OverlayManager(app_state)
    received = []
    control_panel = SimpleNamespace(
        apply_overlay_adjustment_result=lambda result: received.append(result)
    )
    controller = MainActionController(
        SimpleNamespace(overlay_manager=overlay_manager, control_panel=control_panel)
    )

    controller.handle_overlay_size_adjustment("white", "width", 2)

    assert received == [
        OverlayAdjustmentResult(
            key_color="white",
            dimension="width",
            delta=2,
            status="full",
        )
    ]


def test_settings_gear_preserves_tool_window_position_after_hide_show(monkeypatch):
    app = _make_app(monkeypatch)
    try:
        app.show()
        app.settings_toggle_button.click()
        QApplication.processEvents()
        custom_pos = app.settings_tool_window.pos()
        custom_pos.setX(custom_pos.x() + 80)
        custom_pos.setY(custom_pos.y() + 40)
        app.settings_tool_window.move(custom_pos)

        app.settings_toggle_button.click()
        QApplication.processEvents()
        app.settings_toggle_button.click()
        QApplication.processEvents()

        assert app.settings_tool_window.pos() == custom_pos
    finally:
        app.close()


def test_focus_video_action_hides_and_restores_settings_gear(monkeypatch):
    app = _make_app(monkeypatch)
    try:
        app.show()
        app.settings_toggle_button.click()
        QApplication.processEvents()
        assert app.settings_tool_window.isVisible()

        app.focus_video_action.setChecked(True)
        app._toggle_focus_video_mode(True)

        assert app.settings_toggle_button.isHidden()
        assert not app.settings_tool_window.isVisible()
        assert app.focus_video_action.text() == "Show Settings Panel"
        assert not app.settings_toggle_button.isChecked()
        assert app.settings_toggle_button.toolTip() == "Show settings"

        app.focus_video_action.setChecked(False)
        app._toggle_focus_video_mode(False)

        assert not app.settings_toggle_button.isHidden()
        assert app.settings_tool_window.isVisible()
        assert app.settings_toggle_button.isChecked()
        assert app.settings_toggle_button.toolTip() == "Hide settings"
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
        _show_settings_section(control_panel, "Calibration")
        QApplication.processEvents()

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
        _show_settings_section(control_panel, "Detection")
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
        app.settings_toggle_button.click()
        _show_advanced_section(app.control_panel, "histogram")
        app.control_panel.advanced_sections["delta"]._toggle.setChecked(True)
        app.control_panel.advanced_sections["black_keys"]._toggle.setChecked(True)
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
        app.settings_toggle_button.click()
        _show_settings_section(app.control_panel, "Overlays")
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
        app.settings_toggle_button.click()
        _show_advanced_section(app.control_panel, "repeated_notes")
        QApplication.processEvents()

        control_panel = app.control_panel
        select_rect = _rect_in_control_panel(control_panel, control_panel.spark_roi_select_button)
        toggle_rect = _rect_in_control_panel(control_panel, control_panel.spark_roi_toggle_button)

        assert control_panel.spark_roi_select_button.text() == "Select Flash Area Above Keys"
        assert toggle_rect.top() > select_rect.bottom()
        assert toggle_rect.right() <= control_panel.width()
        assert select_rect.right() <= control_panel.width()
    finally:
        app.close()


def test_spark_auto_calibration_controls_stack_vertically(monkeypatch):
    app = _make_app(monkeypatch)
    try:
        app.show()
        app.settings_toggle_button.click()
        _show_advanced_section(app.control_panel, "repeated_notes")
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
